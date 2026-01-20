> [English](project_lifecycle_guidelines.md)

# 项目生命周期指南（计划 + 设计原则）

本仓库是一个 **学习优先（learning-first）** 的 C/C++ 从零实现 GPT/Transformer 项目。本文档的目的，是在项目变大时明确我们遵守的规则，从而保证：

- 代码始终可读、可教
- 每个新特性都能追溯到数学/论文
- 能轻松对比“核心基线（core baseline）”与“扩展变体（variants）”
- 未来从 CPU 演进到多个 GPU 后端时不丢正确性


## 1) 核心目标（north star）

1) **学习与教学优先于纯性能**（至少在核心概念掌握之前）
- 先追求清晰与正确。
- 性能工作应该在我们能解释数学并验证正确性之后进行。

2) **论文/数学 ↔ 代码可追溯**
- 对任何一个特性，学习者都应能回答：
	- 公式/算法是什么？
	- 张量形状是什么？
	- 哪些函数实现它？
	- 下标与 mask 从哪里来？

3) **增量式演进**
- 一次引入一个概念。
- 每一步要小到足以推理、甚至“在脑中模拟”。

4) **正确性护栏**
- 在重构或优化之前，测试必须先把正确性保护起来。
- 每个会改变数学的主要特性至少要增加一个回归测试。


## 2) 仓库结构原则

### 2.1 保持一个稳定的参考基线

当前基线是我们的 reference implementation：
- CPU FP32
- 小型 autograd
- GPT 风格 decoder-only
- （目前）单头 causal attention

规则：基线必须始终可运行、可读。

### 2.2 重大特性优先作为 variants 添加（compare-by-diff）

添加重大特性（例如 multi-head attention、KV-cache）时，优先创建一个 **variant**，而不是直接在基线中就地重写。

建议模式：

- `src/` 维持基线。
- `variants/<feature_name>/` 包含：
	- 只 fork 必要的最小代码
	- 独立 entry point / target
	- 自己的 docs 与 tests

这样学习者可以对比：
- baseline vs variant 的行为
- baseline vs variant 的公式
- baseline vs variant 的内存布局变化

（当一个 variant 被理解并通过测试后，我们再决定是合并回基线，还是作为教学替代方案长期保留。）

### 2.3 为 CPU→GPU 演进准备 backend/plugin 边界

当开始做 GPU 时，应当拆分：

- **模型逻辑**（shape、计算图结构、高层算法）
- **kernel/backend**（matmul、softmax、layernorm、attention kernels）

这样可以形成：
- CPU reference backend
- CPU SIMD backend
- CUDA backend
- ROCm backend
- Vulkan backend
- MUSA backend

模型层应只依赖一个很小的“kernel API surface”，从而让后端可替换。


## 3) 特性工作流（不可妥协）

对每一个新的主要特性/plugin：

### 3.1 文档交付物

必须交付：

1) **数学到代码映射文档**
- 更新/扩展 `docs/transformer_math_to_code.md`。
- 对新特性，增加独立文档，例如：
	- `docs/multihead_attention.md`
	- `docs/kv_cache.md`
	- `docs/moe.md`

2) **可手算（learn-by-hand）walkthrough**
- 提供一个很小的数值例子 + step-by-step 流程。
- 文件示例：
	- `docs/learn_by_hand_<feature>.md`

3) **索引追踪（当索引/内存发生变化时）**
- 如果特性引入新的张量布局或缓存，添加类似：
	- `docs/attention_indexing_trace.md`

### 3.2 代码交付物

- 在特性边界处添加有针对性的注释：
	- 函数顶部注释写出公式与 shape
	- 说明关键实现选择（mask、scale、数值稳定技巧等）

### 3.3 测试交付物

至少满足以下之一：
- finite-difference gradcheck（对新的可微算子）
- deterministic regression（loss 下降 / 精确数值检查）
- shape 与不变量测试（mask 规则、cache 正确性等）

### 3.4 合并前检查清单

- `ctest` 通过
- docs 已在 `docs/README.md` 中链接
- 特性至少有一个测试
- 代码注释解释清楚 shape 与公式


## 4) CPU→GPU 的演进策略

分阶段推进：

### Stage A — CPU 正确性基线
- 纯 C++ reference kernels（naive loops）
- 清晰的索引
- 测试 + 文档

### Stage B — CPU 性能（仍以教学为主）
- profiling（找热点）
- blocked matmul
- SIMD kernels
- 保留 reference path 用于对照正确性

### Stage C — 第一个 GPU 后端（先选一个）

为了控制复杂度，先选一个：
- CUDA 或 ROCm

最小 kernel 集合：
- matmul
- softmax
- layernorm
- attention（开始时可由 primitives 拼出来）

### Stage D — 逐个添加更多后端

例如：
- ROCm（rocBLAS/MIOpen）
- Vulkan
- MUSA

规则：不要同时引入多个 GPU 后端。


## 5) 依赖/库策略（学习驱动）

默认规则：
- 核心学习路径中，尽量不引入高层 ML 框架。

但当它能促进学习时，也允许引入行业库：
- CUDA：cuBLAS / cuDNN / cuSPARSE / cuSOLVER
- ROCm：rocBLAS / MIOpen

指导原则：
- 当添加库加速路径时，保留 reference 实现用于对照。
- 记录库调用对应的数学含义（例如 GEMM = 矩阵乘）。


## 6) 现代 LLM 技术清单（未来 backlog）

这是一份“现代技巧”的清单，建议未来逐个作为 variants 实现。每一项都应该 **一次只引入一个** 并配齐 docs/tests。

### 架构 / 注意力
- **RoPE**（Rotary Positional Embeddings）
	- 替换/增强位置编码；影响 attention 前的 Q/K。
- **GQA**（Grouped Query Attention）
	- K/V head 少于 Q head；影响 attention shape 与 caching。
- **MLA**
	- 该缩写随论文/作者可能含义不同；在实现前必须明确具体参考定义。

### 归一化 / 激活
- **RMSNorm**
	- 常见于 LLaMA-like 模型的 norm 变体。
- **SwiGLU**
	- 现代 FFN 常用的激活/门控变体。

### Mixture of Experts
- **MoE**（Mixture of Experts）
	- 路由 + expert FFNs；带来负载均衡等问题。

### 训练 / 优化 / 对齐
- **PPO**（Proximal Policy Optimization）
	- RLHF 风格微调常用；需要 policy/value 模型与 reward 信号。
- **GRPO**
	- 常指 group-based relative policy optimization；实现前必须明确参考定义。

### 检索 / 增强
- **RAFT**
	- 缩写含义可能不唯一；实现前必须先明确具体算法定义。

重要说明：
- 对于含义不明确的缩写（MLA/GRPO/RAFT），在写代码前先在特性文档里记录我们指的具体 paper/blog reference。


## 7) 命名 + 带时间戳的笔记

- 使用 `docs/notes/YYYY-MM-DD_<topic>.md` 记录时间戳决策。
- 保持 `docs/README.md` 作为索引入口。


## 8) 从当前基线出发的推荐学习顺序

学习优先的顺序：

1) Multi-head attention（variant）
2) KV-cache（variant）
3) Affine norms（LayerNorm gamma/beta）或 RMSNorm（二选一）
4) RoPE
5) GQA
6) CPU SIMD kernel 实验
7) 第一个 GPU 后端（CUDA 或 ROCm）

每一步都必须遵循第 3 节的工作流。
