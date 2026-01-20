> [English](README.md)

# build-llm-using-cpp 文档

本目录是项目的“学习层”：用简短、可操作的笔记把数学/算法映射到具体的 C++ 实现。

从这里开始：
- `transformer_math_to_code.zh-CN.md` — 公式 + 对应的代码位置
- `training_and_inference_walkthrough.zh-CN.md` — 一次完整运行（训练/保存/加载/生成）如何流转
- `autograd_and_memory_layout.zh-CN.md` — 本仓库的张量索引方式 + backward 如何工作
- `learn_by_hand_walkthrough.zh-CN.md` — 可手算的微型数值例子
- `attention_indexing_trace.zh-CN.md` — 把注意力 score 追到 flat-buffer offset
- `tokenizer_choice_gpt2_bpe_vs_sentencepiece.zh-CN.md` — 选择子词 tokenizer（BPE vs SentencePiece）：导入 vs 训练
- `project_lifecycle_guidelines.zh-CN.md` — 项目原则 + 扩展工作流

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

约定：
- 我们更偏好指向**函数与 shape**（例如 `[B,T,C]`），而不是复制大段论文文本。
- 如果引用论文，只引用公开的名称/链接；这里的说明是对本仓库代码的原创解释。

文档契约（很重要）：
- 每当新增一个主要特性（多头注意力、KV-cache、新优化器、RLHF 风格训练等），都应当同步新增/更新文档，确保学习者能把**论文概念 → 公式 → 代码**串起来。
