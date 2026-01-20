> [English](what_changed_vs_core.md)

# 相比核心基线改变了什么（MHA 变体）

本文解释本仓库的**多头注意力（MHA）扩展**新增了什么、刻意没有改什么，以及实现如何映射到标准 Transformer 方程。

本文从“C++ 结构/模块”的视角写，适合对着源码学习。

## 0) 总览

基线/核心：
- 可运行模型 `model::TinyGPT` 使用的是**单头因果自注意力**。
- 对应实现是 `src/ops.cpp` 里的 `nn::self_attention_1h`。

MHA 扩展（本仓库）：
- 我们把 MHA 作为一个**并排的变体模块**放在 `src/variants/...`。
- 我们**没有**改写基线模型。

这样可以让核心更容易阅读/验证，同时也提供一个“新功能 = 代码 + 文档 + 测试”的工作模板。

## 1) C++ 结构/类改了什么

### 1.1 核心模型类型：不变

以下内容不需要修改：
- `model::Config`
- `model::TinyGPT`
- `TinyGPT::Block`

因此核心模型的成员仍然是：
- token embedding：`wte_` 与 positional embedding：`wpe_`
- 每个 block 的注意力权重：`w_qkv`, `b_qkv`, `w_proj`, `b_proj`
- 每个 block 的 MLP 权重：`w_fc`, `b_fc`, `w_out`, `b_out`
- 最终 LM head：`w_lm_`, `b_lm_`

### 1.2 新的变体 API：新增

我们新增了一个函数（不是新增一个模型类）：
- `nn::variants::mha::self_attention_mha(...)`

文件：
- `src/variants/mha/mha_attention.h`
- `src/variants/mha/mha_attention.cpp`

接口（概念上）：
- 输入与权重与基线注意力相同
- 额外多一个整数参数：`n_heads`

## 2) 参数/成员 shape 改了什么

刻意地：**什么都没改**（先把模板做简单、可对照）。

基线与 MHA 变体使用同样的参数布局：
- `w_qkv: [C, 3C]`, `b_qkv: [3C]`
- `w_proj: [C, C]`, `b_proj: [C]`

这也更贴近实际：很多工程实现确实用这样的 QKV/输出投影存储方式。

区别不在参数形状，而在于对通道维度 $C$ 的**解释方式**。

## 3) 实现层面变了什么（math → code）

### 3.1 基线单头注意力（core）

基线的打分矩阵：

$$
S[i,j] = \frac{Q[i]\cdot K[j]}{\sqrt{C}} + \text{causalMask}(i,j)
$$

$$
P[i,:] = \mathrm{softmax}(S[i,:])
$$

$$
Y[i] = \sum_j P[i,j] V[j]
$$

$$
\mathrm{out} = Y W_{proj} + b_{proj}
$$

实现位置：
- `src/ops.cpp` 的 `nn::self_attention_1h`

### 3.2 多头注意力（variant）

多头注意力把通道维度拆成多个 head。

选择：
- head 数量 $H$（代码里叫 `n_heads`）
- 每个 head 的维度 $D$

要求：

$$
C = H\cdot D
$$

做完 QKV 投影后（每个仍是 `[B,T,C]`），我们 reshape：

- 把 $Q, K, V$ 从 `[B,T,C]` 变成 `[B,T,H,D]`

然后对每个 head 独立计算注意力：

$$
S_h[i,j] = \frac{Q_h[i]\cdot K_h[j]}{\sqrt{D}} + \text{causalMask}(i,j)
$$

$$
P_h[i,:] = \mathrm{softmax}(S_h[i,:])
$$

$$
Y_h[i] = \sum_j P_h[i,j] V_h[j]
$$

最后把各 head 拼回模型宽度并投影：

$$
Y = \mathrm{concat}_h(Y_h) \in \mathbb{R}^{B\times T\times C}
$$

$$
\mathrm{out} = Y W_{proj} + b_{proj}
$$

实现位置：
- `src/variants/mha/mha_attention.cpp` 的 `nn::variants::mha::self_attention_mha`

相对基线的关键差异：
- 缩放因子用的是 $\sqrt{D}$ 而不是 $\sqrt{C}$
- scores/probs 在概念上是 `[B,H,T,T]`（每个 head 一张矩阵）
- 输出先得到 `[B,T,H,D]`，再 reshape 回 `[B,T,C]` 后做投影

## 4) 构建目标（CMake）改了什么

我们把构建拆成可组合 target：

- `llm_core`（库）
	- 基线实现：tensor/ops/model/optim/checkpoint/data/util

- `llm_variant_mha`（库）
	- 编译 `src/variants/mha/mha_attention.cpp`
	- 链接 `llm_core`

- `train_gpt`（可执行文件）
	- 只链接 `llm_core`
	- 因此默认仍运行基线注意力

- `test_build_llm`（可执行文件）
	- 链接 `llm_core` + `llm_variant_mha`

这样增加变体会更容易，同时不会把基线搞得不稳定。

## 5) 测试改了什么

我们新增了一个聚焦测试：当 `n_heads=1` 时，MHA 变体应当与基线一致：

- `tests/test_main.cpp`: `test_mha_matches_1h_when_single_head()`
	- forward 输出与 `nn::self_attention_1h` 一致
	- backward 梯度通过一个标量 cross-entropy loss 来对齐

这是一个很强的“正确性锚点”：当 $H=1$ 时，MHA 应当退化为单头公式。

## 6) 论文映射（仅名称）

- “Attention Is All You Need” —— scaled dot-product attention + multi-head attention
- GPT 风格 decoder-only mask —— 因果 mask 规则是 `j > i` 位置被 mask

（这里引用论文名仅用于定位概念；本文是对本仓库代码的原创解释。）

## 7) 如果你想把 MHA 接到可运行模型里，下一步怎么做

目前 `train_gpt` 刻意保持基线-only。

有两个干净的启用方式：

1) 在 `model::Config` 增加字段 `n_heads`（默认 1），再加 CLI flag `--heads`，然后在 block 内选择：
	 - 当 `n_heads == 1` 时走基线
	 - 当 `n_heads > 1` 时走 MHA 变体

2) 新建一个独立可执行文件（例如 `train_gpt_mha`），链接 `llm_variant_mha` 并无条件使用 MHA。

方案 (1) 可以保留一个二进制文件，并默认保持基线行为不变。
方案 (2) 则对学习来说隔离更彻底。
