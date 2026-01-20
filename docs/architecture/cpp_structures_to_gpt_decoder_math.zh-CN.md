> [English](cpp_structures_to_gpt_decoder_math.md)

# C++ 结构体/类 → GPT 解码器数学（核心基线）

本文档是一个 **教学优先的映射表**：把本仓库的 C++ 类型（class/struct、成员、方法）对应到标准 decoder-only GPT 的数学。

范围：
- 只覆盖 baseline/core 模型路径（单头 causal attention）。
- 重点解释 **每个成员代表什么**，以及 **方法如何组合成 decoder 的前向**。

记号：
- $B$：batch size
- $T$：一次运行的序列长度
- $T_{max}$：配置的最大序列长度（`cfg.seq_len`）
- $V$：词表大小（byte-level，通常为 256）
- $C$：模型宽度 / embedding 维度（`d_model`）
- $L$：transformer 层数（`n_layers`）

所有张量都以 flat `std::vector<float>` 存储，并显式携带形状。


## 1) 模型对象：`model::TinyGPT`

定义位置：
- `src/model.h`（`class TinyGPT`）
- `src/model.cpp`（实现）

### 1.1 配置：`model::Config`

`model::Config` 是数学意义上的“超参数包”：

- `vocab_size` → $V$
- `seq_len` → $T_{max}$
- `d_model` → $C$
- `n_layers` → $L$

### 1.2 参数作为成员

在 `TinyGPT` 中，可学习参数以 `nn::Tensor` 成员保存。

Embeddings：
- `wte_` 形状 `[V,C]` → token embedding 矩阵 $W_{te}\in\mathbb{R}^{V\times C}$
- `wpe_` 形状 `[T_max,C]` → position embedding 矩阵 $W_{pe}\in\mathbb{R}^{T_{max}\times C}$

最终 LM head：
- `w_lm_` 形状 `[C,V]` → 投影矩阵 $W_{lm}\in\mathbb{R}^{C\times V}$
- `b_lm_` 形状 `[V]` → 偏置 $b_{lm}\in\mathbb{R}^{V}$

Transformer blocks：
- `blocks_` 是 `Block` 的 vector（长度为 $L$）

每个 `Block`（baseline，单头）包含：

Attention（QKV 打包）：
- `w_qkv` 形状 `[C,3C]` → $W_{qkv}=[W_q\;W_k\;W_v]$ 拼接
- `b_qkv` 形状 `[3C]` → $b_{qkv}=[b_q\;b_k\;b_v]$ 拼接
- `w_proj` 形状 `[C,C]` → 输出投影 $W_o$
- `b_proj` 形状 `[C]` → 输出偏置 $b_o$

MLP：
- `w_fc` 形状 `[C,4C]`，`b_fc` 形状 `[4C]` → MLP 第 1 层 linear
- `w_out` 形状 `[4C,C]`，`b_out` 形状 `[C]` → MLP 第 2 层 linear

这对应常见的 GPT 风格 FFN 宽度倍率 4。


## 2) Decoder 前向：`TinyGPT::forward_logits`

实现位置：`src/model.cpp`。

输入 `tokens_bt`（token ids），形状 `[B,T]`（以 flat `std::vector<int32_t>` 存储），前向会计算：

### 2.1 Token + positional embedding

数学：

$$
X = W_{te}[tokens] + W_{pe}[pos]
\quad\in\mathbb{R}^{B\times T\times C}
$$

代码映射：
- `nn::embedding(wte_, tokens_bt, B, T)` → $W_{te}[tokens]$
- `add_positional(x, B, T)` → 对每个位置加上 $W_{pe}$ 对应行

### 2.2 Pre-norm transformer blocks（重复 L 次）

每个 block 是一个 decoder block：causal self-attention + MLP，并采用 **pre-layernorm**：

对层 $\ell=1..L$：

$$
\begin{aligned}
H &= \mathrm{LN}(X) \\
A &= \mathrm{CausalSelfAttn}(H) \\
X &= X + A \\
M &= \mathrm{LN}(X) \\
FF &= \mathrm{MLP}(M) \\
X &= X + FF
\end{aligned}
$$

代码映射：
- `nn::layernorm_lastdim(x, 1e-5f)` → `LN`
- `nn::self_attention_1h(h, blk.w_qkv, blk.b_qkv, blk.w_proj, blk.b_proj)` → 单头 causal attention
- `nn::add(x, a)` → residual
- `nn::linear_lastdim(...), nn::gelu(...), nn::linear_lastdim(...)` → MLP

### 2.3 最后归一化 + LM head

数学：

$$
logits = \mathrm{LN}(X) W_{lm} + b_{lm}
\quad\in\mathbb{R}^{B\times T\times V}
$$

代码映射：
- `nn::layernorm_lastdim(x, 1e-5f)`
- `nn::linear_lastdim(xn, w_lm_, b_lm_)`


## 3) 基线 causal self-attention：`nn::self_attention_1h`

声明在 `src/ops.h`，实现于 `src/ops.cpp`。

输入与权重：
- `x`：`[B,T,C]`
- `w_qkv`：`[C,3C]`，`b_qkv`：`[3C]`
- `w_proj`：`[C,C]`，`b_proj`：`[C]`

### 3.1 QKV 投影

数学：

$$
[Q,K,V] = X W_{qkv} + b_{qkv}
$$

代码映射：
- `qkv = nn::linear_lastdim(x, w_qkv, b_qkv)` → 形状 `[B,T,3C]`
- 通过切片把最后一维分成三份 `[B,T,C]` 的 `q,k,v`

### 3.2 注意力得分 + causal mask

对每个 batch $b$、query 位置 $i$、key 位置 $j$：

$$
S[b,i,j] = \frac{Q[b,i]\cdot K[b,j]}{\sqrt{C}} + \text{mask}(i,j)
$$

causal mask：

$$
	ext{mask}(i,j) = \begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

代码映射：
- 通过显式 loops 计算点积并写入 `scores[b,i,j]`
- mask 实现为 `if (j > i) s = -1e9f;`

### 3.3 Softmax

$$
P[b,i,:] = \mathrm{softmax}(S[b,i,:])
$$

代码映射：
- `probs = nn::softmax_lastdim(scores)`

### 3.4 加权求和

$$
Y[b,i] = \sum_{j=0}^{T-1} P[b,i,j]\,V[b,j]
$$

代码映射：
- 显式 loops 产生 `att`，形状 `[B,T,C]`

### 3.5 输出投影

$$
A = Y W_o + b_o
$$

代码映射：
- `nn::linear_lastdim(att, w_proj, b_proj)`


## 4) Loss：`TinyGPT::loss` + `nn::cross_entropy`

`TinyGPT::loss(...)` 的内部调用：
- `forward_logits(...)` → logits `[B,T,V]`
- reshape 成 `[N,V]`（$N=B\cdot T$）
- `nn::cross_entropy(logits_nv, targets_n)`

数学（均值负对数似然）：

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} -\log\Big(\mathrm{softmax}(logits_n)[y_n]\Big)
$$


## 5) 训练循环：`src/main.cpp`

CLI 可执行程序 `train_gpt` 是最简单的“trainer + sampler”，它会：

1) 解析参数（batch/seq/dmodel/layers/lr/seed）
2) 把数据 bytes 加载到 `data::ByteDataset`
3) 构建 model + optimizer
4) 对每个 step：
	 - 抽样 batch（`ByteDataset::sample_batch`）
	 - 计算 loss（`TinyGPT::loss`）
	 - `loss.backward()`（autograd）
	 - `opt.step(params)`（AdamW 更新）

然后可选：
- 保存 checkpoint（`ckpt::save`）
- 从 prompt 生成（`generate(...)`）


## 6) Autograd 对象：`nn::Tensor` + `nn::Node`

定义在 `src/tensor.h` / `src/tensor.cpp`。

教学心智模型：
- `src/ops.cpp` 中的每个 op 都会创建一个输出 `Tensor out`。
- 如果启用梯度，`out` 会携带一个 `node`，存储：
	- `parents`（输入 tensors）
	- `backward` lambda（把梯度累加到 `parent.grad`）

因此，数学里的“反向传播图”在代码里就是一串 `Node` lambdas 的链。


## 7) 建议阅读顺序（端到端练习）

如果你想通过 tracing 学习端到端流程：

1) `TinyGPT::forward_logits`（decoder 前向的 wiring）
2) `nn::self_attention_1h`（causal attention）
3) `nn::cross_entropy`（loss）
4) `Tensor::backward` / node 遍历（autograd 引擎）
5) `optim::AdamW::step`（参数更新）

然后用 CLI 跑一个很小的训练循环，按上述顺序设置断点。
