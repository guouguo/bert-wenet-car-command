# Project Audit

## Source

`G:\代码和模型文件\rk3568_bert_wenet`（只读；完成一次递归 inventory 后仅定向读取文本文件）。

## Output

`G:\代码和模型文件\rk3568_bert_wenet_1`

## Platform

RK3568

## Core applications

- `bert_wenet_2rknn.py`：滑动窗口 WeNet RKNN ASR + rule/BERT RKNN NLU + CAN-style command。
- `bert_wenet_2rknn_gpio.py`：wake/sleep、pre-roll、WeNet/BERT RKNN 与 GPIO 硬件演示。

## Kept

- 两个核心程序、`units.txt`。
- BERT tokenizer/config 小文件：`vocab.txt`、`tokenizer_config.json`、`special_tokens_map.json`、`config.json`。
- 四个经主程序路径确认的 RKNN runtime models（仅本地保留）。
- `tools/file_2rknn.py` RKNN 文件测试工具（不包含原始测试音频）。
- `examples/bert_wenet_rknn.py` 早期 WeNet RKNN + BERT ONNX Runtime 混合实现源码。

## Excluded

- 所有 ONNX、训练权重、旧/实验/量化候选 RKNN、`exports/30/`、`exports/r3-onnx/`。
- 依赖已排除模型的 ONNX/旧模型评估工具和性能脚本。
- `useless/`（两个主程序无明确依赖）。
- 原始 `audio/` 真人录音与测试 CSV；未确认可公开授权。

## Generated

`README.md`、`LICENSE`、`requirements.txt`、`.gitignore`、`models/README.md`、`assets/audio/README.md`、`PROJECT_AUDIT.md`。

## Code path changes

- 两个主程序的 WeNet Encoder/CTC 路径统一到 `models/`。
- `bert_wenet_2rknn.py` 的 BERT 路径改为 `models/bert_6L6H_nq.rknn`。
- `bert_wenet_2rknn_gpio.py` 的 BERT 路径改为 `models/bert_6L6H_nq_gpio.rknn`。
- `tools/file_2rknn.py` 的模型路径改到 `models/`，音频目录改为 `assets/audio/`。
- 未修改算法、阈值、GPIO 映射、推理流程或意图映射。

## Runtime models

- `models/bert_6L6H_nq.rknn`
- `models/bert_6L6H_nq_gpio.rknn`
- `models/distill2_encoder_T256_quan_mmsehybird099.rknn`
- `models/distill2_ctc_T256.rknn`

两个原始 BERT 文件同名同大小，但一次限定 SHA256 比对结果不同，故保留为两个运行依赖。

## Git tracked files policy

RKNN、ONNX、训练权重和其他大型二进制模型不进入 Git；仅 `models/README.md` 被跟踪。

## GitHub upload

待本地静态检查、Git 内容检查和 GitHub CLI 前置检查通过后填写。

## Release assets

待源码 push 成功后，仅上传上列四个实际运行依赖。

## Manual follow-up

- 在真实 RK3568 硬件上人工验证 RKNN runtime、麦克风和 GPIO。
- 人工验收 private repository 后再决定是否公开。
