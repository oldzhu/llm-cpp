> [English](sample_processing_walkthrough.md)

# MHA 样本处理 walkthrough（形状与索引）

> 说明：本中文版本翻译进行中；完整内容请先阅读英文版。

目标：用一个具体样本追踪从 `[B,T,C]` 到 `[B,H,T,Ch]` 的拆分、注意力计算、再 concat/proj 回 `[B,T,C]` 的全过程，并把每一步映射到代码。
