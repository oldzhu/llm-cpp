> [English](2026-01-16_project_goal_and_next_steps.md)

# 项目目标与下一步（2026-01-16）

背景：这份笔记记录 `build-llm-using-cpp` 当前的意图与推荐路线图。

快照：
- 日期：2026-01-16
- Git：446a032


## 目标（这个项目是为了什么）

本仓库是一个 **从零开始、端到端** 的 C++ GPT 风格 Transformer，主要用于 **学习与教学**：

- 不依赖高层 ML 框架，实现训练 + 推理（采样）。
- 保持一个易于推理与调试的 CPU FP32 基线。
- 通过文档与针对性注释，提供强“数学/论文 → 代码”的可追溯性。
- 通过测试作为正确性护栏，避免重构（例如 attention 改动）时悄悄破坏模型。

实践学习收获：
- 理解 tokenization/数据 batching、前向、loss、反向/autograd、优化器更新如何协同。
- 建立对张量形状、row-major 内存布局与性能瓶颈的直觉。
- 作为将来扩展到 SIMD 与 GPU 后端（CUDA/ROCm 等）的基础。


## 当前基线（现在已经有什么）

- 字节级语言建模（词表大小 256）：任何文本文件都能作为数据。
- 一个很小的 GPT 模型：
	- token + positional embeddings
	- pre-norm decoder blocks
	- 单头 causal self-attention
	- 带 GELU 的 MLP
- 训练循环 +（可选）在同一次运行里从 prompt 生成。
- checkpoint 保存/加载（`.json` 配置 + `.bin` 权重 +（可选）优化器状态）。
- 一个最小的 reverse-mode autograd 引擎。
- 测试可执行程序 + CTest：
	- finite-diff gradchecks（代表性算子）
	- 小型确定性训练回归
- `docs/` 下的教学文档（公式到代码映射、walkthrough、索引追踪、可手算）。


## 下一步（推荐顺序）

### 1) Multi-head attention（MHA）

为什么优先：
- 从单头基线到更真实的 transformer 的最重要升级。
- KV-cache 的设计依赖 attention head 的最终布局。

会改变什么：
- 在模型配置里加入 `n_heads`。
- 把 Q/K/V 拆分为 heads（通常 `[B,T,C] -> [B,H,T,Ch]`，其中 `Ch=C/H`）。
- 每个 head 独立计算 attention，然后 concat 并投影回去。

文档/测试契约：
- 添加独立 MHA 文档并更新 `docs/transformer_math_to_code.md`。
- 添加/扩展测试覆盖新的 attention 路径。


### 2) KV-cache（加速生成）

为什么：
- 让自回归生成的复杂度更接近“随生成长度线性增长”（避免每一步重算全部历史 K/V）。

会改变什么：
- 按 layer 缓存跨 step 的 K/V。
- 为生成增加增量式 forward 路径。
- 更新文档，并可能影响 checkpoint 格式。


### 3) 模型保真度升级（可选，但仍以教学为主）

候选：
- LayerNorm 的 affine 参数（gamma/beta）
- Dropout（可选）
- 更好的初始化/缩放约定


### 4) 性能阶段（CPU）

关注点：
- 用 blocked + SIMD kernels 替换 naive matmul/attention loops。
- profile 热点（matmul、attention、softmax、layernorm）。
- 高频运行正确性测试。


### 5) GPU 后端

做法：
- 先选一个后端（CUDA 或 ROCm），构建最小 kernel layer。
- 先实现核心 kernels（matmul、softmax、layernorm），再集成到 attention。


### 6) 更长期的学习目标

仅在训练/推理栈稳定之后再做：

#### 超越 bytes 的 tokenizer（subword / BPE “word tokens”）

目标：
- 从 byte tokens（`vocab=256`）迁移到子词 tokens（更接近 GPT/Qwen/DeepSeek 风格）。

会改变什么：
- 增加 tokenizer + 词表文件格式（例如 BPE merges + token-to-id 映射）。
- dataset 从“bytes”切换到“token ids”，包含：
	- encode：text → token ids
	- decode：token ids → text（用于生成输出）
- 更新模型配置，使 `vocab_size` 与 tokenizer 词表一致（不再固定 256）。
- 更新 checkpoint：存储 tokenizer identity/version，并拒绝加载 vocab 不匹配的 checkpoint。

文档/测试契约：
- 文档化具体 tokenization 算法与文件格式。
- 添加确定性的 tokenizer round-trip 测试（encode→decode）与一个小的已知例子测试。

说明：
- 这项改动比看起来大，因为它会触及每个边界：数据加载、batching、生成输出、checkpoint 兼容性。

- 混合精度
- Gradient checkpointing
- 分布式训练
- RL 风格微调（需要更多基础设施与更严格文档）


## 文档规则（本仓库不可妥协）

每个主要特性都必须交付：
- 一份把 **论文概念 → 公式 → 张量形状 → 代码位置** 对齐的文档
- 函数边界处的针对性注释
- 当数学/行为变化时，至少一个回归测试
