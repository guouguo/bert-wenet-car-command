import torch
from transformers import BertTokenizer
import pyaudio
import numpy as np
import time
from pypinyin import lazy_pinyin, load_phrases_dict
from rknnlite.api import RKNNLite  #WeNet和BERT都用 RKNN
import torchaudio.compliance.kaldi as kaldi
import os
import queue
import threading

# ================== 1. 配置路径 ==================
BERT_DIR = './bert_door_command_model_pinyin'
#BERT_RKNN = 'bert_onnx/distill6_best/bert_6L6H_nq.rknn'  
BERT_RKNN = 'models/bert_6L6H_nq.rknn'

WENET_ENC_RKNN = 'models/distill2_encoder_T256_quan_mmsehybird099.rknn'
WENET_CTC_RKNN = 'models/distill2_ctc_T256.rknn'
WENET_UNITS = './units.txt'

# ================== 2. 映射与辅助逻辑 ==================
load_phrases_dict({"降": [["xiang"]]})
common_typo_map = {
    "关上": "关闭", "关": "关闭", "开启": "打开", "开一下": "打开", "开": "打开", "停止": "关闭","打打开": "打开","关闭闭": "关闭",
    "副价": "副驾", "副驾": "副驾驶", "座驾": "驾驶舱", "有前": "右前",
    "后背厢": "后备箱", "富家室": "副驾驶", "富驾驶": "副驾驶"
}

#左前:0x10, 右前:0x20, 左后:0x30, 右后:0x40, 后备箱:0x50 | 打开:0x01, 关闭:0x02
class_to_can = {
    0: 0x11,  # 0-左前门开
    1: 0x12,  # 1-左前门关
    2: 0x21,  # 2-右前门开
    3: 0x22,  # 3-右前门关
    4: 0x31,  # 4-左后门开
    5: 0x32,  # 5-左后门关
    6: 0x41,  # 6-右后门开
    7: 0x42,  # 7-右后门关
    8: 0x51,  # 8-后备箱开
    9: 0x52,  # 9-后备箱关
}

class_names = [
    "左前门开", "左前门关", "右前门开", "右前门关",
    "左后门开", "左后门关", "右后门开", "右后门关",
    "后备箱开", "后备箱关", "拒识/闲聊"
]

def normalize_text(text):
    for typo, correct in common_typo_map.items():
        text = text.replace(typo, correct)
    return text

# 只有一字不差地说出这些指令，才允许跳过BERT执行
STRICT_COMMANDS = {
    "打开左前门": 0x11, "左前门打开": 0x11, "开一下左前门": 0x11,
    "关闭左前门": 0x12, "左前门关闭": 0x12, "关一下左前门": 0x12,
    "打开右前门": 0x21, "右前门打开": 0x21, "开一下右前门": 0x21,
    "关闭右前门": 0x22, "右前门关闭": 0x22, "关一下右前门": 0x22,
    "打开左后门": 0x31, "左后门打开": 0x31, "开一下左后门": 0x31,
    "关闭左后门": 0x32, "左后门关闭": 0x32, "关一下左后门": 0x32,
    "打开右后门": 0x41, "右后门打开": 0x41, "开一下右后门": 0x41,
    "关闭右后门": 0x42, "右后门关闭": 0x42, "关一下右后门": 0x42,
    "打开后备箱": 0x51, "后备箱打开": 0x51, "开一下后备箱": 0x51,
    "关闭后备箱": 0x52, "后备箱关闭": 0x52, "关一下后备箱": 0x52,
}

def fast_path_match(text):
    """快速匹配：O(1) 哈希查找"""
    return STRICT_COMMANDS.get(text, None)
   
# ================== 3. WeNet NPU 引擎 ==================
class WenetRKNNLiteEngine:
    def __init__(self, encoder_path, ctc_path, units_path):
        print("--> 初始化 WeNet NPU...")
        self.rknn_enc = RKNNLite()
        if self.rknn_enc.load_rknn(encoder_path) != 0:
            print("加载 Encoder RKNN 失败!")
            exit(1)
        if self.rknn_enc.init_runtime() != 0:
            print("初始化 Encoder Runtime 失败!")
            exit(1)
            
        self.rknn_ctc = RKNNLite()
        if self.rknn_ctc.load_rknn(ctc_path) != 0:
            print("加载 CTC RKNN 失败!")
            exit(1)
        if self.rknn_ctc.init_runtime() != 0:
            print("初始化 CTC Runtime 失败!")
            exit(1)

        self.id2token = []
        if os.path.exists(units_path):
            with open(units_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        self.id2token.append(parts[0])
                    else:
                        self.id2token.append("") 
            print(f"字典加载成功，共 {len(self.id2token)} 个词")
        else:
            print(f"ERROR: 找不到字典文件 {units_path}")

    def compute_fbank(self, waveform):
        waveform = torch.from_numpy(waveform).unsqueeze(0) * (1 << 15)
        feat = kaldi.fbank(waveform, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0)
        return feat.numpy()

    def decode(self, probs):
        preds = np.argmax(probs, axis=-1)[0]
        tokens = []
        prev = -1
        for i in preds:
            if i != prev and i != 0:
                if i < len(self.id2token):
                    token = self.id2token[i]
                    if token not in ["<space>", " ", "<blank>"]:
                        tokens.append(token)
                prev = i
        return "".join(tokens)

    def transcribe(self, audio_data):
        feats = self.compute_fbank(audio_data.astype(np.float32))
        feats = feats[None, :, :].astype(np.float32)
        target_T = 256
        current_T = feats.shape[1]
        
        if current_T > target_T:
            feats_pad = feats[:, :target_T, :]
        else:
            pad_len = target_T - current_T
            feats_pad = np.pad(feats, ((0,0), (0, pad_len), (0,0)), mode='constant', constant_values=0)

        enc_outputs = self.rknn_enc.inference(inputs=[feats_pad])
        encoder_out = enc_outputs[0]
        ctc_outputs = self.rknn_ctc.inference(inputs=[encoder_out])
        probs = ctc_outputs[0]
        return self.decode(probs)

    def release(self):
        self.rknn_enc.release()
        self.rknn_ctc.release()

# ================== 4. 主程序 (多线程滑窗版) ==================

print(">>> 正在初始化 WeNet + BERT RKNN ...")
# 1. 初始化 Tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_DIR)

# 2. 初始化 BERT NPU 引擎
print("--> 初始化 BERT NPU...")
bert_rknn = RKNNLite()
if bert_rknn.load_rknn(BERT_RKNN) != 0:
    print("❌ 加载 BERT RKNN 失败!")
    exit(1)
if bert_rknn.init_runtime() != 0:
    print("❌ 初始化 BERT Runtime 失败!")
    exit(1)

# 3. 初始化 WeNet NPU 引擎
asr_engine = WenetRKNNLiteEngine(WENET_ENC_RKNN, WENET_CTC_RKNN, WENET_UNITS)


# --- 配置参数 ---
SAMPLE_RATE = 16000
WINDOW_DURATION = 2.5
OVERLAP_DURATION = 0.5
BYTES_PER_SAMPLE = 2
WINDOW_BYTES = int(WINDOW_DURATION * SAMPLE_RATE * BYTES_PER_SAMPLE)
OVERLAP_BYTES = int(OVERLAP_DURATION * SAMPLE_RATE * BYTES_PER_SAMPLE)
STEP_DURATION = 0.5
STEP_BYTES = int(STEP_DURATION * SAMPLE_RATE * BYTES_PER_SAMPLE)

audio_queue = queue.Queue(maxsize=100) 

# --- 录音线程函数 ---
def record_audio_thread():
    p = pyaudio.PyAudio()
    chk_size = 1600
    MIC_INDEX = 0
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=2,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=MIC_INDEX,
                        frames_per_buffer=chk_size)
    except Exception as e:
        print(f"❌ 无法打开麦克风: {e}")
        return

    print(f"🎙️ [后台] 录音线程运行中... (每 {STEP_DURATION}s 扫描一次)")
    while True:
        try:
            data = stream.read(chk_size, exception_on_overflow=False)
            raw_data = np.frombuffer(data, dtype=np.int16)
            left = raw_data[0::2].astype(np.float32)
            right = raw_data[1::2].astype(np.float32)
            mixed = ((left + right) / 2).astype(np.int16)
            audio_queue.put(mixed.tobytes())
        except Exception as e:
            print(f"录音出错: {e}")
            break

t = threading.Thread(target=record_audio_thread, daemon=True)
t.start()

# --- 主线程处理循环 ---
buffer_window = bytearray() 
print("\n🚀 系统就绪！请持续说话...")

try:
    while True:
        # 1. 拼数据
        new_data_count = 0
        while new_data_count < STEP_BYTES:
            chunk = audio_queue.get() 
            buffer_window.extend(chunk)
            new_data_count += len(chunk)
        
        if len(buffer_window) < WINDOW_BYTES:
            continue
            
        current_audio_bytes = buffer_window[:WINDOW_BYTES]
        audio_data = np.frombuffer(current_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 2. VAD 过滤噪音
        vol = np.abs(audio_data).mean()
        if vol >= 0.005: 
            start_t = time.time()
            try:
                # ---------------------------------------------
                # 模块 A: 语音识别 (WeNet -> Text)
                # ---------------------------------------------
                text_raw = asr_engine.transcribe(audio_data)
                
                if text_raw:
                    text_norm = normalize_text(text_raw)
                    print(f"\n🎤 听到: {text_norm}")
                    
                    # ---------------------------------------------
                    # 模块 B: 意图理解 (双路架构)
                    # ---------------------------------------------
                    start_intent_t = time.time()
                    
                    # ⚡ 第一步：尝试匹配规则
                    fast_code = fast_path_match(text_norm)
                    
                    if fast_code is not None:
                        # 匹配命中，不调BERT
                        print(f"[直接匹配] 命中标准指令! -> 发送 CAN: {hex(fast_code)}")
                        print(f"   (直接匹配耗时: {time.time()-start_intent_t:.4f}s)")
                        print("-" * 30)
                    else:
                        # 🐢 第二步：匹配未命中，进入BERT
                        print(f"[bert] 触发复杂语义，交由 BERT 分析...")
                        pinyin_seq = ''.join(lazy_pinyin(text_norm))
                        enc = tokenizer(pinyin_seq, padding='max_length', truncation=True, max_length=64, return_tensors="np")
                        
                        input_ids = enc['input_ids'].astype(np.int32)
                        attention_mask = enc['attention_mask'].astype(np.int32)
                        
                        bert_out = bert_rknn.inference(inputs=[input_ids, attention_mask])
                        logits = bert_out[0] 
                        
                        sorted_logits = np.sort(logits, axis=1)[0][::-1]
                        margin = sorted_logits[0] - sorted_logits[1]
                        predicted_class = np.argmax(logits, axis=1)[0]
                        intent_name = class_names[predicted_class]
                        
                        # ---------------------------------------------
                        # 模块 C: 仅bert需要判断
                        # ---------------------------------------------
                        if predicted_class == 10:
                            print(f"🚫 [bert拦截] 检测到无关闲聊 (得分:{sorted_logits[0]:.1f})")
                        elif margin < 3.0: 
                            print(f"🛡️ [低置信度拦截] 疑似误触发! (意图: {intent_name}, Margin: {margin:.2f})")
                        else:
                            can_code = class_to_can[predicted_class]
                            print(f"✅ [bert] 解析成功! [{intent_name}] -> 发送 CAN: {hex(can_code)}")
                            print(f"   (慢路总耗时: {time.time()-start_intent_t:.3f}s | 坚决度: {margin:.1f})")
                        print("-" * 30)

            except Exception as e:
                print(f"推理出错: {e}")

        # 滑动窗口
        buffer_window = buffer_window[STEP_BYTES:] 

except KeyboardInterrupt:
    print("停止...")
finally:
    # 退出前释放资源
    if 'asr_engine' in globals():
        asr_engine.release()
    if 'bert_rknn' in globals():
        bert_rknn.release()
