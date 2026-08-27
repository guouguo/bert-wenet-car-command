import torch
from transformers import BertTokenizer
import pyaudio
import numpy as np
import time
import argparse
from pypinyin import lazy_pinyin, load_phrases_dict
from rknnlite.api import RKNNLite
import torchaudio.compliance.kaldi as kaldi
import os
import queue
import threading
import collections

# ================== 0. 接收外部传入的启动参数 ==================
parser = argparse.ArgumentParser()
parser.add_argument('--mic_device', type=int, default=0, help='麦克风设备号')
parser.add_argument('--wake_enable', type=int, default=1, help='1启用休眠/唤醒门控')
parser.add_argument('--start_asleep', type=int, default=1, help='1=上电先休眠')
parser.add_argument('--wake_thr_dbfs', type=float, default=-40.0, help='能量唤醒阈值(dBFS)')
parser.add_argument('--wake_consecutive', type=int, default=3, help='连续超阈值次数')
parser.add_argument('--wake_preroll', type=float, default=1.0, help='唤醒时回放的 pre-roll 秒数（防丢开头）')
parser.add_argument('--sleep_silence_sec', type=int, default=30, help='静音多少秒回睡')
parser.add_argument('--sleep_dbfs', type=float, default=-65.0, help='静音判定阈值(dBFS)')
args, _ = parser.parse_known_args()

# ================== 1. 配置路径与 GPIO 映射 ==================
BERT_DIR = './bert_door_command_model_pinyin'
BERT_RKNN = 'models/bert_6L6H_nq_gpio.rknn'

WENET_ENC_RKNN = 'models/distill2_encoder_T256_quan_mmsehybird099.rknn'
WENET_CTC_RKNN = 'models/distill2_ctc_T256.rknn'
WENET_UNITS = './units.txt'

GPIO_SYSFS = '/sys/class/gpio'
CAR_PINS = {
    "左前门": 33,   # IO1
    "右前门": 32,   # IO3
    "左后门": 101,  # IO2
    "右后门": 100,  # IO4
    "后备箱": 99    # IO5
}

# ================== 2. 映射与辅助逻辑 ==================
load_phrases_dict({"降": [["xiang"]]})
common_typo_map = {
    "关上": "关闭", "关": "关闭", "开启": "打开", "开一下": "打开", "开": "打开", "停止": "关闭","打打开": "打开","大打开": "打开","关闭闭": "关闭",
    "副价": "副驾", "副驾": "副驾驶", "座驾": "驾驶舱", "有前": "右前", "后背厢": "后备箱"
}
class_to_can = {
    0: 0x11, 1: 0x12, 2: 0x21, 3: 0x22, 4: 0x31, 
    5: 0x32, 6: 0x41, 7: 0x42, 8: 0x51, 9: 0x52
}
class_names = [
    "左前门开", "左前门关", "右前门开", "右前门关",
    "左后门开", "左后门关", "右后门开", "右后门关",
    "后备箱开", "后备箱关", "拒识/闲聊"
]
STRICT_COMMANDS = {
    "打开左前门": 0x11, "关闭左前门": 0x12,
    "打开右前门": 0x21, "关闭右前门": 0x22,
    "打开左后门": 0x31, "关闭左后门": 0x32,
    "打开右后门": 0x41, "关闭右后门": 0x42,
    "打开后备箱": 0x51, "关闭后备箱": 0x52,
}

def normalize_text(text):
    for typo, correct in common_typo_map.items(): text = text.replace(typo, correct)
    return text

def fast_path_match(text):
    return STRICT_COMMANDS.get(text, None)

# ================== 3. GPIO 控制子系统 ==================
def setup_gpio(pin_num):
    export_path = f"{GPIO_SYSFS}/export"
    direction_path = f"{GPIO_SYSFS}/gpio{pin_num}/direction"
    if not os.path.exists(f"{GPIO_SYSFS}/gpio{pin_num}"):
        try:
            with open(export_path, 'w') as f: f.write(str(pin_num))
        except Exception: pass
    try:
        with open(direction_path, 'w') as f: f.write('out')
    except Exception: pass

def set_gpio_value(pin_num, value):
    try:
        with open(f"{GPIO_SYSFS}/gpio{pin_num}/value", 'w') as f: f.write(str(value))
    except Exception: pass

def init_car_gpios():
    print("--> 初始化车门硬件 GPIO...")
    for pin in CAR_PINS.values():
        setup_gpio(pin)
        set_gpio_value(pin, 0)

def execute_gpio_action(can_code):
    if can_code == 0x11: set_gpio_value(CAR_PINS["左前门"], 1)
    elif can_code == 0x12: set_gpio_value(CAR_PINS["左前门"], 0)
    elif can_code == 0x21: set_gpio_value(CAR_PINS["右前门"], 1)
    elif can_code == 0x22: set_gpio_value(CAR_PINS["右前门"], 0)
    elif can_code == 0x31: set_gpio_value(CAR_PINS["左后门"], 1)
    elif can_code == 0x32: set_gpio_value(CAR_PINS["左后门"], 0)
    elif can_code == 0x41: set_gpio_value(CAR_PINS["右后门"], 1)
    elif can_code == 0x42: set_gpio_value(CAR_PINS["右后门"], 0)
    elif can_code == 0x51: set_gpio_value(CAR_PINS["后备箱"], 1)
    elif can_code == 0x52: set_gpio_value(CAR_PINS["后备箱"], 0)

# ================== 4. 音频计算工具 ==================
def get_dbfs(audio_data):
    """计算一段音频的 RMS dBFS 分贝值"""
    if audio_data is None or len(audio_data) == 0: return -100.0
    rms = np.sqrt(np.mean(audio_data**2) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)

# ================== 5. WeNet NPU 引擎 ==================
class WenetRKNNLiteEngine:
    def __init__(self, encoder_path, ctc_path, units_path):
        self.rknn_enc = RKNNLite()
        self.rknn_enc.load_rknn(encoder_path)
        self.rknn_enc.init_runtime()
        self.rknn_ctc = RKNNLite()
        self.rknn_ctc.load_rknn(ctc_path)
        self.rknn_ctc.init_runtime()
        self.id2token = []
        with open(units_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                self.id2token.append(parts[0] if len(parts) == 2 else "")

    def transcribe(self, audio_data):
        waveform = torch.from_numpy(audio_data).unsqueeze(0) * (1 << 15)
        feat = kaldi.fbank(waveform, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0).numpy()
        feats = feat[None, :, :].astype(np.float32)
        if feats.shape[1] > 256: feats_pad = feats[:, :256, :]
        else: feats_pad = np.pad(feats, ((0,0), (0, 256 - feats.shape[1]), (0,0)), mode='constant')
        encoder_out = self.rknn_enc.inference(inputs=[feats_pad])[0]
        probs = self.rknn_ctc.inference(inputs=[encoder_out])[0]
        
        preds = np.argmax(probs, axis=-1)[0]
        tokens = [self.id2token[i] for i in preds if i != 0 and i < len(self.id2token) and self.id2token[i] not in ["<space>", " ", "<blank>"]]
        # 移除连续重复元素
        res = []
        prev = None
        for t in tokens:
            if t != prev: res.append(t); prev = t
        return "".join(res)

    def release(self):
        self.rknn_enc.release()
        self.rknn_ctc.release()

# ================== 6. 主程序初始化 ==================
init_car_gpios()
tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
bert_rknn = RKNNLite()
bert_rknn.load_rknn(BERT_RKNN)
bert_rknn.init_runtime()
asr_engine = WenetRKNNLiteEngine(WENET_ENC_RKNN, WENET_CTC_RKNN, WENET_UNITS)

SAMPLE_RATE = 16000
WINDOW_DURATION = 2.5
STEP_DURATION = 0.5
BYTES_PER_SAMPLE = 2
WINDOW_BYTES = int(WINDOW_DURATION * SAMPLE_RATE * BYTES_PER_SAMPLE)
STEP_BYTES = int(STEP_DURATION * SAMPLE_RATE * BYTES_PER_SAMPLE)
# 预录制缓冲区大小
PREROLL_BYTES = int(args.wake_preroll * SAMPLE_RATE * BYTES_PER_SAMPLE)

audio_queue = queue.Queue(maxsize=100) 

def record_audio_thread():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=2, rate=SAMPLE_RATE, input=True, input_device_index=args.mic_device, frames_per_buffer=1600)
    print(f"🎙️ [后台] 录音线程运行中... (麦克风: {args.mic_device})")
    while True:
        try:
            data = stream.read(1600, exception_on_overflow=False)
            raw_data = np.frombuffer(data, dtype=np.int16)
            mixed = ((raw_data[0::2].astype(np.float32) + raw_data[1::2].astype(np.float32)) / 2).astype(np.int16)
            audio_queue.put(mixed.tobytes())
        except Exception: break

threading.Thread(target=record_audio_thread, daemon=True).start()

# --- 休眠唤醒状态变量 ---
is_awake = not bool(args.start_asleep)
wake_hit_count = 0
last_active_time = time.time()
buffer_window = bytearray()
preroll_buffer = collections.deque(maxlen=PREROLL_BYTES) # 环形缓冲区，防吞字

if not is_awake: print(f"\n💤 系统上电默认【休眠】。大声说话唤醒 (阈值: {args.wake_thr_dbfs}dB)")
else: print("\n🚀 系统就绪！倾听中...")

try:
    while True:
        # 取数据
        new_data_count = 0
        current_chunk = bytearray()
        while new_data_count < STEP_BYTES:
            chunk = audio_queue.get() 
            current_chunk.extend(chunk)
            new_data_count += len(chunk)
            
        # 无论休眠还是唤醒，都持续写入环形预录制缓冲区
        for i in range(0, len(current_chunk), 2):
            preroll_buffer.append(current_chunk[i:i+2])
            
        # ================== A. 休眠态探测 ==================
        if args.wake_enable and not is_awake:
            audio_data = np.frombuffer(current_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            dbfs = get_dbfs(audio_data)
            
            if dbfs >= args.wake_thr_dbfs:
                wake_hit_count += 1
                if wake_hit_count >= args.wake_consecutive:
                    is_awake = True
                    last_active_time = time.time()
                    print(f"\n🔔 [唤醒] 检测到声强 {dbfs:.1f}dB！系统激活！")
                    # 闪灯反馈
                    for pin in CAR_PINS.values(): set_gpio_value(pin, 1)
                    time.sleep(0.2)
                    for pin in CAR_PINS.values(): set_gpio_value(pin, 0)
                    
                    # 核心防吞字：把唤醒前 1 秒的声音（从环形缓冲区）直接塞入主力计算窗口
                    buffer_window = bytearray(b''.join(preroll_buffer))
                    wake_hit_count = 0
            else:
                wake_hit_count = 0
            continue

        # ================== B. 唤醒态推理 ==================

        buffer_window.extend(current_chunk)
        if len(buffer_window) < WINDOW_BYTES:
            continue
            
        current_audio_bytes = buffer_window[:WINDOW_BYTES]
        audio_data = np.frombuffer(current_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        dbfs = get_dbfs(audio_data)
        
        # 1. 声音大于底噪，送去 WeNet 识别
        if dbfs >= args.sleep_dbfs: 
            text_raw = asr_engine.transcribe(audio_data)
            
            if text_raw:
                last_active_time = time.time() # <--- 只有 WeNet 吐出文字，才刷新保活计时器！
                text_norm = normalize_text(text_raw)
                print(f"\n🎤 听到: {text_norm}")
                fast_code = fast_path_match(text_norm)
                
                if fast_code is not None:
                    print(f"[直接匹配] {hex(fast_code)}")
                    execute_gpio_action(fast_code)
                else:
                    pinyin_seq = ''.join(lazy_pinyin(text_norm))
                    enc = tokenizer(pinyin_seq, padding='max_length', truncation=True, max_length=64, return_tensors="np")
                    bert_out = bert_rknn.inference(inputs=[enc['input_ids'].astype(np.int32), enc['attention_mask'].astype(np.int32)])
                    logits = bert_out[0]
                    
                    margin = np.sort(logits, axis=1)[0][::-1][0] - np.sort(logits, axis=1)[0][::-1][1]
                    pred = np.argmax(logits, axis=1)[0]
                    intent = class_names[pred]
                    
                    if pred != 10 and margin >= 3.0:
                        can_code = class_to_can[pred]
                        print(f"✅ [BERT 解析] [{intent}] -> {hex(can_code)}")
                        execute_gpio_action(can_code)
                    else:
                        print(f"🚫 [拦截] 闲聊或置信度低")
        
        # 2. 独立判断休眠（无论外面多吵，只要超过时间没刷新，必定休眠）
        if time.time() - last_active_time > args.sleep_silence_sec:
            is_awake = False
            print(f"\n💤 [休眠] 超过 {args.sleep_silence_sec} 秒未检测到有效人声，回睡省电...")
            buffer_window.clear()

        # 3. 滑动窗口
        buffer_window = buffer_window[STEP_BYTES:]

except KeyboardInterrupt:
    print("停止...")
finally:
    for pin in CAR_PINS.values(): set_gpio_value(pin, 0)
    asr_engine.release()
    bert_rknn.release()
