> [English](README.md)

# build-llm-using-cpp 文档

本目录是项目的“学习层”：用简短、可操作的笔记把数学/算法映射到具体的 C++ 实现。

说明：部分中文文档仍在翻译中；如需完整细节请先参考英文版。

从这里开始：
- `transformer_math_to_code.zh-CN.md` — 公式/张量形状 → 代码位置
- `training_and_inference_walkthrough.zh-CN.md` — 一次完整运行（训练/保存/加载/生成）如何流转
- `autograd_and_memory_layout.zh-CN.md` — 张量内存布局 + 反向传播在本项目如何工作
- `learn_by_hand_walkthrough.zh-CN.md` — 可手算的小例子
- `attention_indexing_trace.zh-CN.md` — 注意力得分到 flat buffer 偏移的追踪
- `tokenizer_choice_gpt2_bpe_vs_sentencepiece.zh-CN.md` — 选择子词 tokenizer（GPT-2 BPE vs SentencePiece）
- `project_lifecycle_guidelines.zh-CN.md` — 项目原则 + 扩展流程

架构：
- `architecture/folder_layout_and_targets.zh-CN.md`
- `architecture/cpp_structures_to_gpt_decoder_math.zh-CN.md`
- `architecture/hidden_states_vs_attention_matrices.zh-CN.md`

变体：
- `variants/mha/README.zh-CN.md`
- `variants/mha/learn_by_hand.zh-CN.md`
- `variants/mha/what_changed_vs_core.zh-CN.md`
- `variants/mha/why_multi_head_attention.zh-CN.md`
- `variants/mha/sample_processing_walkthrough.zh-CN.md`

笔记：
- `notes/2026-01-16_project_goal_and_next_steps.zh-CN.md`
