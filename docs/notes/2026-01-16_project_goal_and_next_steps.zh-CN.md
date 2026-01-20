> [English](2026-01-16_project_goal_and_next_steps.md)

# 项目目标与下一步（2026-01-16）

> 说明：本中文版本翻译进行中；完整内容请先阅读英文版。

摘要：
- 本仓库是一个教学优先、从零开始的 C++ GPT 风格训练/推理项目。
- 重点是可验证（测试）、可解释（文档把公式/形状映射到代码）。

下一步方向（简版）：
- 多头注意力（MHA）
- KV-cache（加速生成）
- 性能优化（CPU：blocked/SIMD）
- 更长期：从 byte tokenizer 迁移到子词 tokenizer（BPE/SentencePiece），让 `vocab_size` 不再固定为 256。
