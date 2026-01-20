> [English](transformer_math_to_code.md)

# Transformer 数学公式 → 本仓库代码

目标：让你能快速回答“那条公式在代码里的哪里实现？”

本仓库实现一个 *很小的、GPT 风格的 decoder-only Transformer*：
- 字节级词表（$V=256$）
- 学习得到的位置嵌入（positional embeddings）
- pre-norm 残差块（子层之前做 LayerNorm）
- （目前）**单头** causal self-attention

## 记号（形状）

- Batch 大小：`B`
- 序列长度：`T`
- 模型宽度：`C`（`d_model`）
- 词表大小：`V`

张量以 flat array 的形式存储（`std::vector<float>`），布局为 **row-major（行优先）**，也就是 **最后一维是连续的**。

常见形状：
- tokens：`[B,T]`（在代码中是 `std::vector<int32_t>`，长度为 `B*T`）
- hidden states：`X` 为 `[B,T,C]`
- logits：`Z` 为 `[B,T,V]`


## 1) Token + 位置嵌入

数学形式（学习得到的 embeddings）：

- Token embedding 表：$W_{te} \in \mathbb{R}^{V \times C}$
- Position embedding 表：$W_{pe} \in \mathbb{R}^{T_{max} \times C}$

对 batch 中第 $b$ 个样本、位置 $i$ 的 token id（记为 $t_{b,i}$）：

$$E_{b,i,:} = W_{te}[t_{b,i},:]$$
$$X_{b,i,:} = E_{b,i,:} + W_{pe}[i,:]$$

代码位置：
- token embedding lookup：`nn::embedding(...)`（`src/ops.cpp`）
- 加位置嵌入：`model::TinyGPT::add_positional(...)`（`src/model.cpp`）


## 2) Transformer block（pre-norm，decoder-only）

本仓库使用 **pre-norm** 残差块（LayerNorm 在子层之前）：

给定输入 $X$：

### 2.1) Attention 子层

$$H = \mathrm{LN}(X)$$
$$A = \mathrm{CausalSelfAttn}(H)$$
$$X \leftarrow X + A$$

代码位置：
- LN：`nn::layernorm_lastdim(...)`（`src/ops.cpp`）
- Attention：`nn::self_attention_1h(...)`（`src/ops.cpp`）
- 残差加法：`nn::add(...)`（`src/ops.cpp`）

### 2.2) 前馈网络（MLP）子层

两层 MLP + GELU：

$$M = \mathrm{LN}(X)$$
$$F = \mathrm{GELU}(M W_{fc} + b_{fc})$$
$$O = F W_{out} + b_{out}$$
$$X \leftarrow X + O$$

代码位置：
- $XW+b$：`nn::linear_lastdim(...)`（`src/ops.cpp`）
- GELU：`nn::gelu(...)`（`src/ops.cpp`）
- block 串联：`model::TinyGPT::forward_logits(...)`（`src/model.cpp`）


## 3) Causal self-attention（单头）

单头注意力中，向量维度为 $C$：

- 投影权重：
	- $W_{qkv} \in \mathbb{R}^{C \times 3C}$，偏置 $b_{qkv} \in \mathbb{R}^{3C}$
	- $W_{proj} \in \mathbb{R}^{C \times C}$，偏置 $b_{proj} \in \mathbb{R}^{C}$

计算：

$$[Q, K, V] = H W_{qkv} + b_{qkv}$$

带 causal mask 的缩放点积得分：

$$S_{b,i,j} = \frac{Q_{b,i,:} \cdot K_{b,j,:}}{\sqrt{C}} + \begin{cases}0 & j \le i \\ -\infty & j > i\end{cases}$$

对最后一维（keys 维度）做 softmax：

$$P_{b,i,:} = \mathrm{softmax}(S_{b,i,:})$$

加权求和：

$$Y_{b,i,:} = \sum_{j=0}^{T-1} P_{b,i,j} V_{b,j,:}$$

输出投影：

$$A = Y W_{proj} + b_{proj}$$

代码位置：
- `nn::self_attention_1h(...)`（`src/ops.cpp`）
	- Q/K/V 的切片通过 `slice_lastdim_copy(...)`
	- causal mask 通过在 `j > i` 时写入 `-1e9`（近似 $-\infty$）
	- softmax：`nn::softmax_lastdim(...)`


## 4) 最后归一化 + LM head

最后一个 block 之后：

$$X_n = \mathrm{LN}(X)$$
$$Z = X_n W_{lm} + b_{lm}$$

代码位置：
- 最后 LN + logits：`model::TinyGPT::forward_logits(...)`（`src/model.cpp`）


## 5) 交叉熵（next-token prediction）

把 `[B,T,V]` 展平为 `[N,V]`，其中 $N=B\cdot T$。

对每一行 $n$，目标类别为 $y_n$：

$$p_{n,v} = \mathrm{softmax}(z_{n,:})_v$$
$$\ell_n = -\log(p_{n,y_n})$$
$$L = \frac{1}{N} \sum_{n=0}^{N-1} \ell_n$$

代码位置：
- `nn::cross_entropy(...)`（`src/ops.cpp`）


## 6) AdamW 优化器更新

本仓库实现了常见的 AdamW（decoupled weight decay）形式。

给定梯度 $g_t$ 与参数 $\theta_t$：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

偏置修正：

$$\hat m_t = \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

更新：

$$\theta \leftarrow \theta - \alpha \left( \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta \right)$$

代码位置：
- `optim::AdamW::step(...)`（`src/optim.cpp`）


## 参考（论文名）

- “Attention Is All You Need”（Transformer）
- “Language Models are Unsupervised Multitask Learners”（GPT-2）

（这里只引用论文名用于定位与术语统一；本文档是对本仓库实现的原创映射说明。）
