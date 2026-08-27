import torch
from transformers import BertTokenizer
import pyaudio
import numpy as np
import time
from pypinyin import lazy_pinyin, load_phrases_dict
import onnxruntime as ort  # 用于 BERT (CPU)
from rknnlite.api import RKNNLite  # 用于 WeNet (NPU)
import torchaudio.compliance.kaldi as kaldi
import os
import queue
import threading

# ================== 1. 配置路径 ==================
# BERT (CPU)
BERT_DIR = './bert_door_command_model_pinyin'
BERT_ONNX = 'bert_onnx/bert_door.onnx'

# WeNet (NPU)
WENET_ENC_RKNN = 'exports/30/encoder_T256.rknn'
WENET_CTC_RKNN = 'exports/30/ctc_T256.rknn'
WENET_UNITS = 'units.txt'

# ================== 2. 辅助逻辑 ==================
load_phrases_dict({"降": [["xiang"]]})
door_mapping = {"左前": 0b0000, "右前": 0b0010, "左后": 0b0100, "右后": 0b1000, "后备箱": 0b1100}
action_mapping = {"打开": 0b0001, "关闭": 0b0010}
common_typo_map = {
    "关上": "关闭", "关": "关闭", "开启": "打开", "开一下": "打开", "开": "打开", "停止": "关闭",
    "副价": "副驾", "副驾": "副驾驶", "座驾": "驾驶舱", "有前": "右前",
    "后背厢": "后备箱", "富家室": "副驾驶", "富驾驶": "副驾驶"
}

def normalize_text(text):
    for typo, correct in common_typo_map.items():
        text = text.replace(typo, correct)
    return text

def extract_intent_and_map(text):
    text = normalize_text(text)
    pinyin_text = ''.join(lazy_pinyin(text))
    door_pinyin_map = {"zuoqian": "左前", "youqian": "右前", "zuohou": "左后", "youhou": "右后", "houbeixiang": "后备箱"}
    matched_door = None
    for key, val in door_pinyin_map.items():
        if key in pinyin_text:
            matched_door = val
            break
    matched_action = None
    if "ai" in pinyin_text or "dakai" in pinyin_text or "aiqi" in pinyin_text or "kaiyixia" in pinyin_text:
        matched_action = "打开"
    elif "guan" in pinyin_text or "guanbi" in pinyin_text or "tingzhi" in pinyin_text:
        matched_action = "关闭"
    if matched_door and matched_action:
        return door_mapping.get(matched_door) + action_mapping.get(matched_action)
    return None

# ================== 3. WeNet NPU 引擎 ==================
class WenetRKNNLiteEngine:
    def __init__(self, encoder_path, ctc_path, units_path):
        print("--> 初始化 NPU (RKNNLite)...")
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

# --- 初始化 (一定要有!) ---
print(">>> 正在初始化混合计算引擎 (WeNet NPU + BERT CPU) ...")
# 1. 初始化 Tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
# 2. 初始化 BERT
bert_sess = ort.InferenceSession(BERT_ONNX, providers=['CPUExecutionProvider'])
# 3. 初始化 WeNet 引擎 (你之前丢失的就是这句)
asr_engine = WenetRKNNLiteEngine(WENET_ENC_RKNN, WENET_CTC_RKNN, WENET_UNITS)


# --- 配置参数 ---
SAMPLE_RATE = 16000
WINDOW_DURATION = 2.5  # 窗口总时长 (秒)
OVERLAP_DURATION = 0.5 # 重叠时长 (秒)

# 计算字节数 (16-bit = 2 bytes per sample)
BYTES_PER_SAMPLE = 2
WINDOW_BYTES = int(WINDOW_DURATION * SAMPLE_RATE * BYTES_PER_SAMPLE)   # 80000 bytes
OVERLAP_BYTES = int(OVERLAP_DURATION * SAMPLE_RATE * BYTES_PER_SAMPLE) # 16000 bytes
STEP_DURATION = 0.5
STEP_BYTES = int(STEP_DURATION * SAMPLE_RATE * BYTES_PER_SAMPLE)                              # 64000 bytes

# 线程安全的队列
audio_queue = queue.Queue(maxsize=100) 

# --- 录音线程函数 ---
def record_audio_thread():
    p = pyaudio.PyAudio()
    chk_size = 1600
    MIC_INDEX = 0
    try:
        # RK809 必须使用 channels=2 才能正常启动
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

            # ✅ 修复音频处理：混合左右声道
            # 这样不管你的麦克风接在左边还是右边，都能收到声音
            raw_data = np.frombuffer(data, dtype=np.int16)
            left = raw_data[0::2].astype(np.float32)
            right = raw_data[1::2].astype(np.float32)
            mixed = ((left + right) / 2).astype(np.int16)

            audio_queue.put(mixed.tobytes())

        except Exception as e:
            print(f"录音出错: {e}")
            break

# 启动录音线程
t = threading.Thread(target=record_audio_thread, daemon=True)
t.start()

# --- 主线程处理循环 ---
buffer_window = bytearray() 
print("\n🚀 系统就绪！请持续说话，系统将自动滑窗识别...")

try:
    while True:
        # 1. 从队列取数据，凑够 STEP_BYTES
        new_data_count = 0
        while new_data_count < STEP_BYTES:
            chunk = audio_queue.get() 
            buffer_window.extend(chunk)
            new_data_count += len(chunk)
        
        # 2. 检查缓冲区
        if len(buffer_window) < WINDOW_BYTES:
            continue
            
        # 3. 截取窗口
        current_audio_bytes = buffer_window[:WINDOW_BYTES]
        audio_data = np.frombuffer(current_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # === VAD ===
        vol = np.abs(audio_data).mean()
        if vol < 0.005: 
             pass 
        else:
            # --- 推理 ---
            start_t = time.time()
            try:
                # ASR (NPU)
                text_raw = asr_engine.transcribe(audio_data)
                
                if text_raw:
                    text_norm = normalize_text(text_raw)
                    print(f"🎤 识别 ({time.time()-start_t:.3f}s): {text_norm}")
                    
                    # BERT (CPU)
                    pinyin_seq = ''.join(lazy_pinyin(text_norm))
                    # max_length 已修正为 64
                    enc = tokenizer(pinyin_seq, padding='max_length', truncation=True, max_length=64, return_tensors="np")
                    
                    bert_inputs = {
                        'input_ids': enc['input_ids'].astype(np.int32),
                        'attention_mask': enc['attention_mask'].astype(np.int32),
                        'token_type_ids': enc['token_type_ids'].astype(np.int32)
                    }
                    
                    valid_inputs = {k: v for k, v in bert_inputs.items() if k in [i.name for i in bert_sess.get_inputs()]}
                    
                    pred_idx = None
                    try:
                        bert_out = bert_sess.run(None, valid_inputs)[0]
                        pred_idx = np.argmax(bert_out)
                    except Exception as e:
                        print(f"BERT Error: {e}")

                    # 意图判断
                    code = extract_intent_and_map(text_norm)
                    if code is not None:
                        print(f"✅ 触发指令! CAN: {bin(code)}")
                        print("-" * 30)
                    
                    # 如果需要用 BERT 结果兜底，可以在这里加逻辑
                    # if code is None and pred_idx is not None: ...

            except Exception as e:
                print(f"推理出错: {e}")

        # 4. 滑动窗口
        buffer_window = buffer_window[STEP_BYTES:] 

except KeyboardInterrupt:
    print("停止...")
finally:
    # 退出前释放资源
    if 'asr_engine' in globals():
        asr_engine.release()
