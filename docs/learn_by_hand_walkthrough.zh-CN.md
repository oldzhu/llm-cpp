> [English](learn_by_hand_walkthrough.md)

# 可手算（learn-by-hand）walkthrough（论文数学 → 本仓库）

这份文档的设计目标是：你可以 *在纸上* 把关键步骤算一遍。

它结合：
- `docs/training_and_inference_walkthrough.md`（运行时流程）
- `docs/transformer_math_to_code.md`（公式 → 代码）
- `docs/autograd_and_memory_layout.md`（索引 + autograd 心智模型）

目标：你能在脑中模拟一个极小的 GPT 前向与 loss，并理解 C++ 里每个 loop 的含义。


## 0) 用于手算的极小配置

真实仓库常见配置：
- 词表 $V=256$（字节）
- hidden 宽度 $C=64+$
- 序列长度 $T=64+$

为了手算，我们缩小为：

- Batch：$B=1$
- 序列长度：$T=3$
- 模型宽度：$C=2$
- 词表：$V=4$（玩具词表）

你仍然可以照着真实代码的步骤走；只是 shape 更小。


## 1) 一个训练 step（`main.cpp` 做了什么）

一次训练迭代在概念上是：

1) 采样一个 batch `(x,y)`
2) `logits = forward_logits(x)`
3) `loss = cross_entropy(logits, y)`
4) `loss.backward()`
5) `opt.step(params)`

代码位置（函数名）：
- 数据集 batch：`data::ByteDataset::sample_batch`（`src/data.cpp`）
- 前向：`model::TinyGPT::forward_logits`（`src/model.cpp`）
- loss：`model::TinyGPT::loss` → `nn::cross_entropy`（`src/model.cpp`、`src/ops.cpp`）
- backward：`nn::Tensor::backward`（`src/tensor.cpp`）
- 优化器：`optim::AdamW::step`（`src/optim.cpp`）


## 2) 数据集：next-token prediction 的配对

在本仓库（byte-level）里，数据集就是一个 byte 数组。

如果我们抽样一个长度为 $T+1$ 的子串：

- 输入 tokens：$x_t = \text{bytes}[s+t]$，$t=0..T-1$
- targets：$y_t = \text{bytes}[s+t+1]$（右移 1）

所以：

$$y_t = x_{t+1}$$

玩具例子（词表 0..3）：
- 抽到的 bytes（长度 4）：`[2,1,3,0]`
- 则 $x=[2,1,3]$，$y=[1,3,0]$


## 3) Embeddings：tokens → 向量

参数：
- Token embedding 表 $W_{te} \in \mathbb{R}^{V\times C}$
- Position embedding 表 $W_{pe} \in \mathbb{R}^{T_{max}\times C}$

数学：

$$E_{t,:} = W_{te}[x_t,:]$$
$$X_{t,:} = E_{t,:} + W_{pe}[t,:]$$

代码位置：
- `nn::embedding(...)`（`src/ops.cpp`）
- `model::TinyGPT::add_positional(...)`（`src/model.cpp`）

### 手算例子

令 $C=2, V=4$。

Token embeddings $W_{te}$（第 0..3 行对应 token 0..3）：
- token0：$[0.0, 0.0]$
- token1：$[1.0, 0.0]$
- token2：$[0.0, 1.0]$
- token3：$[1.0, 1.0]$

位置 embeddings $W_{pe}$（位置 0..2）：
- pos0：$[0.1, 0.0]$
- pos1：$[0.0, 0.1]$
- pos2：$[0.1, 0.1]$

输入 $x=[2,1,3]$：
- $X_0 = W_{te}[2] + W_{pe}[0] = [0,1] + [0.1,0] = [0.1, 1.0]$
- $X_1 = W_{te}[1] + W_{pe}[1] = [1,0] + [0,0.1] = [1.0, 0.1]$
- $X_2 = W_{te}[3] + W_{pe}[2] = [1,1] + [0.1,0.1] = [1.1, 1.1]$


## 4) Transformer block（pre-norm）

本仓库使用 pre-norm blocks。

### 4.1 LayerNorm（无 affine）

对向量 $x\in\mathbb{R}^{C}$：

$$\mu=\frac{1}{C}\sum_i x_i, \quad \sigma^2=\frac{1}{C}\sum_i (x_i-\mu)^2$$
$$\mathrm{LN}(x)_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}$$

代码位置：
- `nn::layernorm_lastdim(...)`（`src/ops.cpp`）

手算提示：当 $C=2$ 时很快。

例如对 $X_0=[0.1,1.0]$：
- 均值 $\mu=0.55$
- 偏差 $[-0.45,+0.45]$
- 方差 $0.2025$，标准差 $0.45$
- LN 结果约为 $[-1,+1]$


### 4.2 Causal self-attention（单头）

数学（单头）：

$$[Q,K,V] = H W_{qkv} + b_{qkv}$$

scores：

$$S_{i,j} = \frac{Q_i\cdot K_j}{\sqrt{C}}$$

causal mask（decoder-only）：当 $j>i$ 时令 $S_{i,j}=-\infty$。

概率：

$$P_{i,:} = \mathrm{softmax}(S_{i,:})$$

加权求和：

$$Y_i = \sum_j P_{i,j} V_j$$

输出投影：

$$A = Y W_{proj} + b_{proj}$$

代码位置：
- `nn::self_attention_1h(...)`（`src/ops.cpp`）

#### 手算例子（用于直觉，做了简化）

为了避免算太多，我们假设 $Q=K=V=H$（真实模型里并不成立）。

令（LN 之后）：
- $H_0=[-1,+1]$
- $H_1=[+1,-1]$

缩放系数是 $1/\sqrt{2}\approx 0.707$。

在位置 $i=1$ 时，允许的 keys 是 $j=0,1$。

计算 scores：
- $S_{1,0}=0.707*(H_1\cdot H_0)=0.707*((1*-1)+(-1*1))=0.707*(-2)=-1.414$
- $S_{1,1}=0.707*(H_1\cdot H_1)=0.707*(2)=+1.414$

softmax：
- $\exp(-1.414)\approx 0.243$
- $\exp(1.414)\approx 4.113$
- 和约为 $4.356$

所以：
- $P_{1,0}\approx 0.056$
- $P_{1,1}\approx 0.944$

加权求和：

$$Y_1 \approx 0.056 H_0 + 0.944 H_1 \approx [0.888,-0.888]$$

解释：
- token 1 基本上主要注意自己。
- causal mask 确保它不能看未来。


### 4.3 残差连接

注意力之后：

$$X \leftarrow X + A$$

MLP 之后：

$$X \leftarrow X + \mathrm{MLP}(\mathrm{LN}(X))$$

代码位置：
- `nn::add(...)`（`src/ops.cpp`）
- block wiring：`model::TinyGPT::forward_logits(...)`（`src/model.cpp`）


## 5) LM head：hidden → logits

最后一个 block 之后：

$$Z = \mathrm{LN}(X) W_{lm} + b_{lm}$$

`Z` 的形状为 `[B,T,V]`。

代码位置：
- `model::TinyGPT::forward_logits(...)` 内部使用 `nn::linear_lastdim(...)`


## 6) 交叉熵 loss（next-token prediction）

把 `[B,T,V]` 展平为 `[N,V]`，其中 $N=B\cdot T$。

对 logits 行向量 $z$ 与目标类别 $y$：
- $p=\mathrm{softmax}(z)$
- loss $=-\log p_y$

对所有行取平均：

$$L=\frac{1}{N}\sum_n -\log p_{n,y_n}$$

代码位置：
- `nn::cross_entropy(...)`（`src/ops.cpp`）

### 手算例子

令某一行 logits 为 $z=[2,1,0,-1]$，目标 $y=1$。

近似计算 exp：
- $e^2\approx 7.39$, $e^1\approx 2.72$, $e^0=1$, $e^{-1}\approx 0.368$
- 和约为 $11.48$

目标概率：
- $p_1 \approx 2.72/11.48 \approx 0.237$

loss：
- $-\log(0.237) \approx 1.44$

训练时打印的 `loss=...` 就是类似这样的标量。


## 7) 生成（sampling）循环（手算视角）

每一步生成：

1) 取最后最多 `seq_len` 个 token 作为上下文
2) 计算最后位置的 logits
3) 通过 temperature 与可选 top-k 采样下一 token

temperature：
- 在 softmax 前把 logits 除以 `temp`。
- 温度越高，分布越平。

top-k：
- 只保留最大的 K 个 logits，在它们之间采样。

代码位置：
- `src/main.cpp` 中的 `generate(...)` 与相关采样 helper。


## 8) 迷你练习（建议在纸上做）

1) softmax + CE：
- 自己选 4 个 logits 与一个目标类别，计算 softmax 与 CE。

2) causal mask：
- 当位置 $i=0$ 时，验证只有 $j=0$ 允许。

3) 注意力权重：
- 自己选 $Q_1$ 与 $K_0,K_1$，算 scores、softmax、加权和。


## 9) 下一步特性与文档

当我们增加 multi-head attention 与 KV-cache 时，会：
- 为每个特性添加独立文档
- 更新 `docs/transformer_math_to_code.md`
- 在关键位置添加 shape 与公式相关的代码注释
- 扩展测试，防止重构时悄悄破坏正确性
