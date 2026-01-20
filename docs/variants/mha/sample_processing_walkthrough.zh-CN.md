> [English](sample_processing_walkthrough.md)

# MHA 样本处理 walkthrough（形状与索引）

本文用一个具体、逐步的例子，演示一次 forward pass 在本仓库里是如何计算注意力的。

我们会对同一组“概念输入”做两次流程：
1) 基线 `nn::self_attention_1h`（单头）
2) 变体 `nn::variants::mha::self_attention_mha`（多头）

重点是 shape、索引、以及每个中间张量的含义。

相关代码：
- 基线：`src/ops.cpp` 的 `nn::self_attention_1h`
- 变体：`src/variants/mha/mha_attention.cpp` 的 `nn::variants::mha::self_attention_mha`

## 0) 选择很小的具体尺寸

令：
- `B = 1`（一个 batch 元素）
- `T = 3`（三个 token）
- `C = 4`（模型宽度）

对 MHA，选择：
- `H = 2` 个 head
- `D = C/H = 2`（每个 head 的维度）

假设我们已经有 hidden states `x`：
- `x` shape `[B,T,C] = [1,3,4]`

内存布局提醒（row-major）：
- `[B,T,C]`：`off(b,t,c) = (b*T + t)*C + c`
- `[B,T,T]`：`off(b,i,j) = (b*T + i)*T + j`
- 对 MHA 变体的 `[B,H,T,T]`：
	- `off(b,h,i,j) = ((b*H + h)*T + i)*T + j`

## 1) 两者共享步骤：QKV 投影

基线与 MHA 都会做：

- `qkv = linear_lastdim(x, w_qkv, b_qkv)`

shape：
- `x` 是 `[1,3,4]`
- `w_qkv` 是 `[4, 12]`（因为 `3C = 12`）
- `qkv` 是 `[1,3,12]`

然后在最后一维切片得到：
- `q = qkv[..., 0:C]` → `[1,3,4]`
- `k = qkv[..., C:2C]` → `[1,3,4]`
- `v = qkv[..., 2C:3C]` → `[1,3,4]`

到这里，基线与 MHA 是完全相同的。

## 2) 基线：计算 scores `[B,T,T]`

基线的 scores 张量：
- `scores` shape `[1,3,3]`

对每个 `i`（query 位置）与 `j`（key 位置）：

$$
S[i,j] = \frac{Q[i]\cdot K[j]}{\sqrt{C}} + \text{causalMask}(i,j)
$$

在代码（`src/ops.cpp`）中，它对应这样一段循环：
- `qi = (bb*T + i)*C` 是 `q[b,i,:]` 的 base offset
- `kj = (bb*T + j)*C` 是 `k[b,j,:]` 的 base offset
- 再对 `c=0..C-1` 做 inner sum

### 例子：计算 `scores[0,2,1]`

计算 base offset：
- `qi = (0*3 + 2)*4 = 8` → `q[0,2,:]` 存在 `q.data[8..11]`
- `kj = (0*3 + 1)*4 = 4` → `k[0,1,:]` 存在 `k.data[4..7]`

点积：

$$
q[0,2,:]\cdot k[0,1,:] = \sum_{c=0}^{3} q.data[8+c]\;k.data[4+c]
$$

缩放：
- 乘 `1/sqrt(C) = 1/sqrt(4) = 1/2`

mask：
- 因为 `j=1 <= i=2`，允许（不 mask）

写入 `[B,T,T]` buffer 的位置：
- `off = (0*3 + 2)*3 + 1 = 7`
- 因此 `scores.data[7] = scaledDot`

这就是基线循环该如何被阅读。

## 3) MHA 变体：reshape 并按 head 计算 scores `[B,H,T,T]`

MHA 变体会 reshape：

- `q4 = reshape(q, {B,T,H,D})` → `[1,3,2,2]`
- `k4 = reshape(k, {B,T,H,D})` → `[1,3,2,2]`
- `v4 = reshape(v, {B,T,H,D})` → `[1,3,2,2]`

重要：本仓库的 `reshape` 是一个 **view**（共享同一块 flat buffer）。
也就是说 `q4.data` 与 `q.data` 指向同一份存储。

接着按 head 计算：

$$
S_h[i,j] = \frac{Q_h[i]\cdot K_h[j]}{\sqrt{D}} + \text{causalMask}(i,j)
$$

相对基线的关键差异：
- 点积只在 `d=0..D-1` 上求和（每个 head 自己的维度）
- 缩放用 `1/sqrt(D)`
- 每个 head `h` 分别计算 scores

### 例子：计算 head 1 的 `scores[0,h=1,i=2,j=1]`

这里取 `B=1,T=3,H=2,D=2`。

变体代码（`mha_attention.cpp`）使用 base offset：

- `q_base = ((b*T + i)*H + h)*D`
- `k_base = ((b*T + j)*H + h)*D`

代入 `b=0,i=2,j=1,h=1`：

- `q_base = ((0*3 + 2)*2 + 1)*2 = (5)*2 = 10`
- `k_base = ((0*3 + 1)*2 + 1)*2 = (3)*2 = 6`

因此 head-1 的向量位于：
- `Q_{h=1}[i=2]` 在 `q4.data[10..11]`
- `K_{h=1}[j=1]` 在 `k4.data[6..7]`

对 `D=2` 做点积：

$$
	ext{dot} = q4.data[10]\cdot k4.data[6] + q4.data[11]\cdot k4.data[7]
$$

缩放：
- 乘 `1/sqrt(D) = 1/sqrt(2)`

mask：
- `j=1 <= i=2`，允许

写入 `[B,H,T,T]` buffer 的位置：

- `off = ((b*H + h)*T + i)*T + j`
- `off = ((0*2 + 1)*3 + 2)*3 + 1 = (5)*3 + 1 = 16`

因此：
- `scores.data[16] = scaledDot`

对比基线：
- 基线的 `scores[0,2,1]` 写在单一矩阵里：`scores.data[7]`
- MHA 会为每个 head 存一张矩阵；head-1 的 `(i=2,j=1)` 写在不同 offset。

## 4) Softmax 与加权求和（基线 vs MHA）

基线：
- `probs = softmax_lastdim(scores)`，shape `[B,T,T]`
- `att[b,i,c] = sum_j probs[b,i,j] * v[b,j,c]` 得到 `[B,T,C]`

MHA 变体：
- `probs = softmax_lastdim(scores)`，shape `[B,H,T,T]`
- `att[b,i,h,d] = sum_j probs[b,h,i,j] * v4[b,j,h,d]` 得到 `[B,T,H,D]`
- 然后 `reshape(att, {B,T,C})` 把 head 拼回去

含义：
- 基线每个 token 只有一行 `probs[b,i,:]`
- MHA 每个 token 有 $H$ 行 `probs[b,h,i,:]`（每个 head 一种注意力模式）

## 4.1) 数值例子（基线：一行 query）

仍使用上面的尺寸：`B=1, T=3, C=4`。

我们计算 `b=0, i=2`（第 3 个 token 关注 `j=0,1,2`）的整行注意力。

选择具体向量：

- `i=2` 的 Query：
	- $Q[2] = [1,0,1,0]$
- keys：
	- $K[0] = [1,0,0,0]$
	- $K[1] = [0,1,0,0]$
	- $K[2] = [1,1,0,0]$

计算缩放点积（基线缩放 $1/\sqrt{C} = 1/2$）：

$$
\begin{aligned}
S[2,0] &= \frac{Q[2]\cdot K[0]}{\sqrt{4}} = \tfrac{1}{2}(1\cdot 1 + 0\cdot 0 + 1\cdot 0 + 0\cdot 0) = 0.5\\
S[2,1] &= \frac{Q[2]\cdot K[1]}{\sqrt{4}} = \tfrac{1}{2}(1\cdot 0 + 0\cdot 1 + 1\cdot 0 + 0\cdot 0) = 0\\
S[2,2] &= \frac{Q[2]\cdot K[2]}{\sqrt{4}} = \tfrac{1}{2}(1\cdot 1 + 0\cdot 1 + 1\cdot 0 + 0\cdot 0) = 0.5
\end{aligned}
$$

对 `i=2`，因果 mask 允许所有 `j<=2`，所以：

$$
S[2,:] = [0.5,\;0,\;0.5]
$$

对 $j$ 维做 softmax：

$$
\mathrm{softmax}(S[2,:])_j = \frac{e^{S[2,j]}}{\sum_{m=0}^{2} e^{S[2,m]}}
$$

数值上（$e^{0.5}\approx 1.6487$）：

$$
\sum e^{S} = 1.6487 + 1 + 1.6487 = 4.2974
$$

因此概率大约是：

$$
P[2,:] \approx [0.383,\;0.233,\;0.383]
$$

选择 $V[j]$：

- $V[0] = [1,0,0,0]$
- $V[1] = [0,2,0,0]$
- $V[2] = [0,0,3,0]$

加权求和：

$$
\mathrm{att}[2] = \sum_{j=0}^{2} P[2,j]\;V[j]
$$

近似得到：

$$
\mathrm{att}[2] \approx
0.383[1,0,0,0] +
0.233[0,2,0,0] +
0.383[0,0,3,0]
= [0.383,\;0.466,\;1.149,\;0]
$$

这正是基线循环在做的事情，只是我们把中间的 `scores`/`probs` 写了出来。

## 4.2) 数值例子（MHA：一个 head）

对 MHA，取 `H=2` 且 `D=C/H=2`。流程对每个 head 重复，但：

- 使用每个 head 的向量 $Q_h[i], K_h[j], V_h[j] \in \mathbb{R}^{D}$
- 缩放使用 $1/\sqrt{D} = 1/\sqrt{2}$

下面给出 `i=2`、`h=1` 的一行例子：

- $Q_{1}[2] = [1,\,-1]$
- $K_{1}[0]=[1,\,0]$, $K_{1}[1]=[0,\,1]$, $K_{1}[2]=[1,\,1]$

缩放得分（$1/\sqrt{2}\approx 0.7071$）：

$$
\begin{aligned}
S_{1}[2,0] &= (1\cdot 1 + (-1)\cdot 0)\cdot 0.7071 \approx 0.7071\\
S_{1}[2,1] &= (1\cdot 0 + (-1)\cdot 1)\cdot 0.7071 \approx -0.7071\\
S_{1}[2,2] &= (1\cdot 1 + (-1)\cdot 1)\cdot 0.7071 = 0
\end{aligned}
$$

softmax（近似）：

$$
e^{0.7071}\approx 2.028,\; e^{-0.7071}\approx 0.493,\; e^{0}=1,\;\;\sum\approx 3.521
$$

$$
P_{1}[2,:] \approx [0.576,\;0.140,\;0.284]
$$

选择每个 head 的 $V$：

- $V_{1}[0]=[10,\,0]$, $V_{1}[1]=[0,\,10]$, $V_{1}[2]=[5,\,5]$

那么该 head 的输出是：

$$
\mathrm{att}_{1}[2] = \sum_{j=0}^{2} P_{1}[2,j]\;V_{1}[j]
\approx 0.576[10,0] + 0.140[0,10] + 0.284[5,5]
\approx [7.18,\;2.82]
$$

head `h=0` 同样会计算自己的 $\mathrm{att}_{0}[2]\in\mathbb{R}^{2}$（来自不同的切片）。拼接两个 head 后，就又回到了 4 维向量。

## 5) 这些步骤在代码里的位置

下面是把上面的 math/numeric 例子映射到具体代码路径的“搜索地图”。

基线（`src/ops.cpp` 的 `nn::self_attention_1h`）：

- QKV 投影：`qkv = linear_lastdim(x, w_qkv, b_qkv)`
- 切出 Q/K/V：`slice_lastdim_copy(qkv, 0, C)`, `slice_lastdim_copy(qkv, C, C)`, `slice_lastdim_copy(qkv, 2*C, C)`
- scores（点积 + 缩放 + mask）：`bb, i, j` 的嵌套循环计算
	- `qi = (bb*T + i)*C`，`kj = (bb*T + j)*C`
	- `s += q[qi+c] * k[kj+c]`，再 `s *= 1/sqrt(C)`
	- 因果 mask：`if (j > i) s = -1e9f`
	- 写入：`scores[(bb*T + i)*T + j] = s`
- softmax：`probs = softmax_lastdim(scores)`（同一个 `softmax_lastdim` 也会被其他地方复用）
- 加权求和：`bb, i, c` 的循环计算
	- `sum_j probs[(bb*T + i)*T + j] * v[(bb*T + j)*C + c]`

MHA 变体（`src/variants/mha/mha_attention.cpp` 的 `nn::variants::mha::self_attention_mha`）：

- QKV 投影：`qkv = nn::linear_lastdim(x, w_qkv, b_qkv)`
- 切出 Q/K/V：本文件内的 `slice_lastdim_copy` 小 helper（语义与 core 相同）
- 拆 head（view reshape）：`q4 = nn::reshape(q, {B,T,H,D})`（`k4`、`v4` 同理）
- 每 head 的 scores：`bb, hh, i, j` 的嵌套循环计算
	- `q_base = ((bb*T + i)*H + hh)*D`，`k_base = ((bb*T + j)*H + hh)*D`
	- `s += q4[q_base+d] * k4[k_base+d]`，再 `s *= 1/sqrt(D)`
	- 因果 mask：`if (j > i) s = -1e9f`
	- 写入：`scores[(((bb*H + hh)*T + i)*T + j)] = s`
- softmax：`probs = nn::softmax_lastdim(scores)`
	- 注意：`softmax_lastdim` 总是对**最后一维**归一化，因此对 `[B,H,T,T]` 会在每个 `(b,h,i)` 行上独立对 `j` 做 softmax。
- 每 head 加权求和：循环计算
	- `att[((bb*T + i)*H + hh)*D + d] = sum_j probs[p_off] * v4[v_off]`
- 拼 head：`att_cat = nn::reshape(att, {B,T,C})`

如果你想看单个 score 元素更深的索引追踪，可参考 `docs/attention_indexing_trace.md`。

## 6) 读代码时可以尝试的小练习

一个不错的练习是：在 score 循环里打断点，然后手动验证一个元素：

- 选择 `b=0, i=2, j=1`
- 计算 `qi/kj`（基线）或 `q_base/k_base`（MHA）
- 验证点积里到底乘了哪些 index
- 验证最终写入 `scores.data[...]` 的 offset

然后再验证一个被 mask 的元素（`j > i`）。

---

本文刻意聚焦于 shape 与 offset；其中的数值段落也可以用来检查 score/softmax/weighted-sum 管线的行为是否符合直觉。
