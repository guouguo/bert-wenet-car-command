# Runtime Models

此目录在完整本地整理副本中保存 RK3568 运行所需模型，但二进制模型由 `.gitignore` 排除，不进入 Git history。

需要的文件：

- `bert_6L6H_nq.rknn`：`bert_wenet_2rknn.py` 使用的 BERT 模型。
- `bert_6L6H_nq_gpio.rknn`：`bert_wenet_2rknn_gpio.py` 使用的 BERT 模型。
- `distill2_encoder_T256_quan_mmsehybird099.rknn`：两个主程序共用的 WeNet Encoder。
- `distill2_ctc_T256.rknn`：两个主程序共用的 WeNet CTC。

两个 BERT 源文件原本同名且大小相同，但 SHA256 不同，因此不能视为重复；整理时为 GPIO 版本使用了独立名称。

模型下载地址： https://github.com/guouguo/bert-wenet-car-command/releases/tag/v1.0-models
