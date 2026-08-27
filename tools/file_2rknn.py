import os
import sys
import time
import glob
import numpy as np
import scipy.io.wavfile as wavfile
from transformers import BertTokenizer
from pypinyin import lazy_pinyin, load_phrases_dict
from rknnlite.api import RKNNLite

# ================== 1. 配置路径 ==================
AUDIO_DIR = './assets/audio'  
BERT_DIR = './bert_door_command_model_pinyin'

# RKNN 模型路径
BERT_RKNN = 'models/bert_6L6H_nq.rknn'
WENET_ENC_RKNN = 'models/distill2_encoder_T256_quan_mmsehybird099.rknn'
WENET_CTC_RKNN = 'models/distill2_ctc_T256.rknn'
WENET_UNITS = 'units.txt'
# ================== 2. 文本处理与意图映射 ==================
load_phrases_dict({"降": [["xiang"]]})

common_typo_map = {
    "打打开": "打开", "请打打开": "请打开", "关闭闭": "关闭", "闭闭": "关闭", "开开": "打开",
    "关上": "关闭", "关": "关闭", "停止": "关闭", "停": "关闭",
    "开启": "打开", "开一下": "打开", "开": "打开", 
    "副价": "副驾", "副驾": "副驾驶", "座驾": "驾驶舱", 
    "有前": "右前", "富家室": "副驾驶", "富驾驶": "副驾驶",
    "后背厢": "后备箱", "后背": "后备箱",
}

def normalize_text(text):
    for _ in range(2): 
        for typo, correct in common_typo_map.items():
            text = text.replace(typo, correct)
    return text

BERT_LABEL_TO_CAN = {
    0: 0x11, 1: 0x12, 2: 0x21, 3: 0x22, 4: 0x31, 
    5: 0x32, 6: 0x41, 7: 0x42, 8: 0x51, 9: 0x52,
}

# ================== 3. NPU 推理引擎 ==================

class BertRKNNLiteEngine:
    def __init__(self, rknn_path, tokenizer_path):
        print("--> 正在初始化 BERT NPU 引擎...")
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(rknn_path) != 0:
            print(f"❌ 加载 BERT RKNN 失败: {rknn_path}")
            sys.exit(1)
        if self.rknn.init_runtime() != 0:
            print("❌ 初始化 BERT Runtime 失败!")
            sys.exit(1)
            
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def extract_intent(self, text):
        # text_clean = normalize_text(text) # 我们已经彻底放弃烂规则，信任模型！
        pinyin_text = ''.join(lazy_pinyin(text))
        
        # 提取特征
        inputs = self.tokenizer(
            pinyin_text, 
            padding="max_length", 
            truncation=True, 
            max_length=64, 
            return_tensors="np"
        )
        
        # 内存连续性保护
        input_ids = np.array(inputs["input_ids"].tolist(), dtype=np.int32)
        attention_mask = np.array(inputs["attention_mask"].tolist(), dtype=np.int32)
        
        if input_ids.shape != (1, 64):
            return None, f"❌ 形状异常: {input_ids.shape}"

        # 执行 NPU 推理
        outputs = self.rknn.inference(inputs=[input_ids, attention_mask])
        logits = outputs[0]
        
        pred_idx = np.argmax(logits, axis=1)[0]
        
        # ✨ 新增：计算 Softmax 概率分布以获取置信度
        exp_logits = np.exp(logits - np.max(logits))
        conf = exp_logits / np.sum(exp_logits)
        max_conf = np.max(conf)
        
        # ✨ 新增：置信度阈值拦截器
        CONF_THRESHOLD = 0.70  # 你可以根据实测情况在这里微调，比如 0.65 或 0.75
        
        # 如果模型分类不是 10，但它非常犹豫（置信度低），强制打回拒识！
        if pred_idx != 10 and max_conf < CONF_THRESHOLD:
            return None, f"🛡️ 低置信拦截 (原预测: {pred_idx}, 置信度: {max_conf:.2f}, 拼音: {pinyin_text})"

        # 正常的处理逻辑 (附带打印置信度)
        if pred_idx == 10:
            return None, f"⚠️ 拒识/闲聊 (预测类别: 10, 置信度: {max_conf:.2f}, 拼音: {pinyin_text})"
        
        can_code = BERT_LABEL_TO_CAN.get(pred_idx)
        return can_code, f"✅ 成功 (预测类别: {pred_idx}, 置信度: {max_conf:.2f})"

    def release(self):
        self.rknn.release()


class WenetRKNNLiteEngine:
    def __init__(self, encoder_path, ctc_path, units_path):
        self.rknn_enc = RKNNLite()
        if self.rknn_enc.load_rknn(encoder_path) != 0: sys.exit(1)
        if self.rknn_enc.init_runtime() != 0: sys.exit(1)

        self.rknn_ctc = RKNNLite()
        if self.rknn_ctc.load_rknn(ctc_path) != 0: sys.exit(1)
        if self.rknn_ctc.init_runtime() != 0: sys.exit(1)

        self.id2token = []
        with open(units_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                self.id2token.append(parts[0] if len(parts) == 2 else "") 

    def compute_fbank(self, waveform):
        import python_speech_features
        waveform = waveform * 32768.0 
        feat = python_speech_features.logfbank(
            waveform, samplerate=16000, winlen=0.025, winstep=0.01,
            nfilt=80, nfft=512, lowfreq=0, highfreq=None, preemph=0.97
        )
        return feat

    def decode(self, probs):
        preds = np.argmax(probs, axis=-1)[0]
        tokens = []
        prev = -1
        for i in preds:
            if i != prev and i != 0 and i < len(self.id2token):
                token = self.id2token[i]
                if token not in ["<space>", " ", "<blank>"]:
                    tokens.append(token)
            prev = i
        return "".join(tokens)

    def transcribe(self, audio_data):
        feats = self.compute_fbank(audio_data.astype(np.float32))
        feats = feats[np.newaxis, :, :].astype(np.float32)
        
        target_T = 256
        current_T = feats.shape[1]
        
        if current_T > target_T:
            feats_pad = feats[:, :target_T, :]
        else:
            pad_len = target_T - current_T
            feats_pad = np.pad(feats, ((0, 0), (0, pad_len), (0, 0)), mode='constant', constant_values=0)

        feats_pad = np.ascontiguousarray(feats_pad)
        enc_outputs = self.rknn_enc.inference(inputs=[feats_pad])
        
        encoder_out = np.ascontiguousarray(enc_outputs[0])
        ctc_outputs = self.rknn_ctc.inference(inputs=[encoder_out])
        
        return self.decode(ctc_outputs[0])

    def release(self):
        self.rknn_enc.release()
        self.rknn_ctc.release()

# ================== 4. 主程序 ==================

def main():
    if not os.path.exists(AUDIO_DIR):
        print(f"❌ 找不到目录 {AUDIO_DIR}")
        sys.exit(1)

    wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    total_files = len(wav_files)
        
    print(f">>> 找到 {total_files} 个音频文件，准备发车...")

    asr_engine = WenetRKNNLiteEngine(WENET_ENC_RKNN, WENET_CTC_RKNN, WENET_UNITS)
    bert_engine = BertRKNNLiteEngine(BERT_RKNN, BERT_DIR)

    print(f"\n{'='*100}")
    print(f"{'文件名':<15} | {'ASR原始识别':<15} | {'意图(HEX)':<10} | {'详情/失败原因'}")
    print(f"{'-'*100}")

    stats = {"success_asr": 0, "triggered": 0, "rejected": 0, "empty": 0, "error": 0}
    start_total_time = time.time()

    for file_path in wav_files:
        filename = os.path.basename(file_path)
        
        print(f"{filename:<15} | ", end="", flush=True) 
        
        try:
            sample_rate, audio_data = wavfile.read(file_path)
            
            if len(audio_data) < 1600:
                print(f"{'(空)':<15} | {'-':<10} | ⚠️ 语音太短")
                continue

            if len(audio_data.shape) > 1: audio_data = np.mean(audio_data, axis=1)
            
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                audio_data = audio_data.astype(np.float32)
                max_val = np.max(np.abs(audio_data))
                if max_val > 0: audio_data /= max_val
            
            # --- 1. ASR ---
            text_raw = asr_engine.transcribe(audio_data)
            if not text_raw:
                print(f"{'(空)':<15} | {'-':<10} | ⚠️ 无法识别")
                stats["empty"] += 1
                continue
            
            stats["success_asr"] += 1
            raw_disp = (text_raw[:13] + '..') if len(text_raw) > 13 else text_raw
            print(f"{raw_disp:<15} | ", end="", flush=True)
            
            # --- 2. BERT ---
            code, debug_msg = bert_engine.extract_intent(text_raw)
            code_str = "-"
            if code is not None:
                code_str = hex(code)
                stats["triggered"] += 1
            else:
                stats["rejected"] += 1
            
            print(f"{code_str:<10} | {debug_msg}", flush=True)

        except Exception as e:
            print(f"❌ 异常: {str(e)[:40]}", flush=True)
            stats["error"] += 1

    total_time = time.time() - start_total_time
    print(f"{'='*100}")
    print(f"📊 测试报告: 共计 {total_files} 条 | 触发 {stats['triggered']} 条 | 拦截 {stats['rejected']} 条 | 耗时 {total_time:.2f}s")
    
    asr_engine.release()
    bert_engine.release()

if __name__ == "__main__":
    main()
