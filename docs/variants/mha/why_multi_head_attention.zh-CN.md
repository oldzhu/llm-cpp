> [English](why_multi_head_attention.md)

# 为什么多头注意力重要（不只是并行）

这是一份教学笔记，试图回答：
- “如果我有足够计算/内存，单头是不是就够了？”
- “MHA 在实践中到底带来什么？”

我们会把直觉与本仓库的基线实现以及 MHA 变体联系起来。

相关代码：
- 基线（单头）：`src/ops.cpp` 里的 `nn::self_attention_1h`
- 变体（多头）：`src/variants/mha/mha_attention.cpp` 里的 `nn::variants::mha::self_attention_mha`

论文名称（用于定位概念）：
- “Attention Is All You Need”（scaled dot-product attention + multi-head attention）

## 1) 单头注意力在一层里能表达什么

在本仓库的基线里，对每个 query 位置 `i`，我们只会计算一份注意力分布：

- scores：`S[b,i,j]`，shape `[B,T,T]`
- probs： `P[b,i,j]`，shape `[B,T,T]`

单头公式：

$$
S[i,j] = \frac{Q[i]\cdot K[j]}{\sqrt{C}} + \text{causalMask}(i,j)
$$

$$
P[i,:] = \mathrm{softmax}(S[i,:])
$$

$$
Y[i] = \sum_j P[i,j] V[j]
$$

关键限制：
- 对每个 `i`，只有**一行**概率向量 `P[i,:]`。
- 也就是说同一层里，token `i` 只能用**一种方式**“看”整个上下文。

即使你把 `C` 加大（更多通道），每个 token 每层依旧只有一个分布 `P[i,:]`。

## 2) MHA 增加了什么：同时存在的多种注意力模式

多头注意力把通道维度拆成：

- 选择 `H` 个 head
- 每个 head 维度 `D`
- 要求 $C = H\cdot D$

然后对每个 head $h$ 计算各自的注意力分布：

$$
S_h[i,j] = \frac{Q_h[i]\cdot K_h[j]}{\sqrt{D}} + \text{causalMask}(i,j)
$$

$$
P_h[i,:] = \mathrm{softmax}(S_h[i,:])
$$

$$
Y_h[i] = \sum_j P_h[i,j] V_h[j]
$$

把 head 拼起来：

$$
Y[i] = \mathrm{concat}_h(Y_h[i]) \in \mathbb{R}^{C}
$$

再投影：

$$
\mathrm{out} = Y W_{proj} + b_{proj}
$$

直观理解：
- 有 $H$ 个 head，就能同时拥有 **$H$ 份不同的注意力分布**（在同一个上下文上）。
- 各个 head 可以专门化（例如：一个 head 关注最近 token，一个关注标点，一个关注主谓宾等）。

这是表达能力的提升，而不仅仅是“更好并行”。

## 3) 为什么“单头 + 更大 C”不等价

一个常见直觉是：“资源足够就把 `C` 做大”。

但关键瓶颈是：每层每个 token 的“注意力图”数量：
- 单头：1 张图 `P[i,:]`
- 多头：$H$ 张图 `P_h[i,:]`

如果模型需要在同一层里为了不同目的同时强烈关注不同位置，单头必须在同一份概率分布里做折衷，把概率质量混在一起。

MHA 允许这些不同的“关注理由”以不同 head 的形式并存。

## 4) 在我们的代码里，这体现在 shape 上

基线（`self_attention_1h`）：
- Q,K,V 是 `[B,T,C]`
- scores/probs 是 `[B,T,T]`

MHA 变体（`self_attention_mha`）：
- Q,K,V reshape 为 `[B,T,H,D]`
- scores/probs 在概念上是 `[B,H,T,T]`

因此，多出来的 `H` 维度就对应“多张注意力图”。

另外注意缩放：
- 基线用 $1/\sqrt{C}$
- MHA 用 $1/\sqrt{D}$

这样即使每个 head 的通道更少，score 的数值规模也保持在相似范围。

## 5) 训练动态（实际好处）

MHA 往往训练更稳定，因为各个 head 可以形成不同的梯度通路：
- 每个 head 有自己的 score/softmax 输出
- 每个 head 在拼接之前会先产生自己的加权和

在本仓库的变体里，这体现在显式的 `[B,H,T,T]` scores/probs buffer 和按 head 的循环。

## 6) 什么时候单头就够了

- 教学基线、小任务、或非常小的模型，单头完全没问题。
- 本仓库保留单头基线路径，是因为它更容易读、更容易验证。

当你希望在更现实的规模/上下文下获得更好的质量时，MHA 是标准积木。
