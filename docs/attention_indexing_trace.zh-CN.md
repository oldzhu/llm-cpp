> [English](attention_indexing_trace.md)

# 注意力索引追踪（公式 → loop → flat offset）

这篇文档是注意力方程与 C++ for-loop 的 **逐行桥梁（line-for-line bridge）**。

它回答类似问题：
- `scores[b,i,j]` 在内存里到底放在什么位置？
- $Q_i\cdot K_j$ 里究竟是哪几个下标在相乘？
- causal mask 在哪里发生？

相关文档与代码：
- 内存/索引公式：`docs/autograd_and_memory_layout.md`
- 注意力公式：`docs/transformer_math_to_code.md`
- 实现：`src/ops.cpp` 中的 `nn::self_attention_1h(...)`


## 0) 准备：使用很小的形状

用一个很小的配置，这样你可以在纸上算 offset：

- Batch：$B=1$
- 序列长度：$T=3$
- 模型宽度：$C=2$

我们追踪一个具体标量：

- `scores[b,i,j]`，其中 `b=0, i=1, j=0`


## 1) 我们要实现的论文公式

单头注意力的缩放点积得分：

$$S_{b,i,j}=\frac{Q_{b,i,:}\cdot K_{b,j,:}}{\sqrt{C}}$$

decoder-only 的 causal mask：
- 当 $j>i$ 时，令 $S_{b,i,j}=-\infty$（实现里用一个很大的负数近似）。


## 2) Row-major 索引公式（精确形式）

来自 `docs/autograd_and_memory_layout.md`。

### 2.1 形状 `[B,T,C]`

对 `(b,t,c)` 的 flat offset：

$$\mathrm{off}_{BTC}(b,t,c)=(b\cdot T + t)\cdot C + c$$

### 2.2 形状 `[B,T,T]`

对 `(b,i,j)` 的 flat offset：

$$\mathrm{off}_{BTT}(b,i,j)=(b\cdot T + i)\cdot T + j$$


## 3) 本仓库里 Q 和 K 从哪里来

在 `nn::self_attention_1h(x, w_qkv, b_qkv, w_proj, b_proj)` 内部：

1) `qkv = linear_lastdim(x, w_qkv, b_qkv)`
- `qkv` 形状：`[B,T,3C]`

2) 对最后一维切片：
- `q = slice_lastdim_copy(qkv, 0, C)` → 形状 `[B,T,C]`
- `k = slice_lastdim_copy(qkv, C, C)` → 形状 `[B,T,C]`
- `v = slice_lastdim_copy(qkv, 2*C, C)` → 形状 `[B,T,C]`

因此，计算 score 的点积一定是用 `[B,T,C]` 的索引规则从 `q`、`k` 的 flat buffer 里读数据。


## 4) 追踪一个元素：`scores[0,1,0]`

### 4.1 `scores[0,1,0]` 的 flat 位置

`scores` 形状是 `[B,T,T] = [1,3,3]`。

计算 flat offset：

$$\mathrm{off}_{BTT}(0,1,0)=(0\cdot 3 + 1)\cdot 3 + 0 = 3$$

所以：
- `scores[0,1,0]` 存在 `scores.data[3]`。

这与实现中的写法一致：
- 写入位置为 `scores.data[((bb*T + i)*T + j)]`。

### 4.2 `q[0,1,:]` 的 flat 起点

`q` 形状为 `[B,T,C]=[1,3,2]`。

`q[0,1,:]` 这段连续向量的起始 offset：

$$qBase = \mathrm{off}_{BTC}(0,1,0)=(0\cdot 3 + 1)\cdot 2 + 0 = 2$$

因此：
- `q[0,1,0]` 是 `q.data[2]`
- `q[0,1,1]` 是 `q.data[3]`

### 4.3 `k[0,0,:]` 的 flat 起点

`k[0,0,:]` 的起始 offset：

$$kBase = \mathrm{off}_{BTC}(0,0,0)=(0\cdot 3 + 0)\cdot 2 + 0 = 0$$

因此：
- `k[0,0,0]` 是 `k.data[0]`
- `k[0,0,1]` 是 `k.data[1]`

### 4.4 点积逐项展开（对应 inner loop）

论文形式：

$$Q_{0,1,:}\cdot K_{0,0,:} = Q_{0,1,0}K_{0,0,0} + Q_{0,1,1}K_{0,0,1}$$

flat-buffer 形式：

$$= q.data[qBase+0]\cdot k.data[kBase+0] + q.data[qBase+1]\cdot k.data[kBase+1]$$

代入 `qBase=2`, `kBase=0`：

$$= q.data[2]\cdot k.data[0] + q.data[3]\cdot k.data[1]$$

这与代码里的 inner loop 完全一致：
- 代码会计算 `qi = (bb*T + i)*C` 与 `kj = (bb*T + j)*C`
- 然后对 `c=0..C-1` 累加 `q.data[qi+c] * k.data[kj+c]`

对 `(bb=0, i=1, j=0)`：
- `qi = (0*3 + 1)*2 = 2`（与 `qBase` 相同）
- `kj = (0*3 + 0)*2 = 0`（与 `kBase` 相同）

### 4.5 应用 scale 与 mask

scale：

$$S_{0,1,0} = (\text{dot})\cdot \frac{1}{\sqrt{C}} = (\text{dot})\cdot \frac{1}{\sqrt{2}}$$

causal mask 检查：
- 此处 `j=0`、`i=1`，所以 `j>i` 为 false。
- 该值允许写入。

最终：
- `scores.data[3] = scaledDot`。


## 4.6 与 C++ loop 变量逐个对上（精确对应）

本节把上面的数学/索引推导与 `src/ops.cpp` 中 `nn::self_attention_1h` 使用的 **具体变量** 对齐。

仍然使用第 0 节的小配置：
- `B=1, T=3, C=2`
- 追踪 `bb=0, i=1, j=0`

### 4.6.1 loop 中 `qi` 与 `kj` 的计算

代码：
- `qi = (bb*T + i) * C`
- `kj = (bb*T + j) * C`

代入：
- `qi = (0*3 + 1) * 2 = 2`
- `kj = (0*3 + 0) * 2 = 0`

它们就是两个连续向量的 base offset：
- `q[0,1,:]` 起于 `q.data[2]`
- `k[0,0,:]` 起于 `k.data[0]`

### 4.6.2 inner loop 就是对 `c` 求和

代码计算：

$$s = \sum_{c=0}^{C-1} q.data[qi + c]\cdot k.data[kj + c]$$

当 `C=2, qi=2, kj=0`：

$$s = q.data[2]\cdot k.data[0] + q.data[3]\cdot k.data[1]$$

随后乘上缩放系数：

$$s \leftarrow s\cdot \frac{1}{\sqrt{C}} = s\cdot \frac{1}{\sqrt{2}}$$

并应用 causal mask：
- 此处 `j=0` 且 `i=1`，`j>i` 为 false → 不覆盖。

### 4.6.3 写入 `scores.data[...]` 就是 `[B,T,T]` 的 offset 公式

代码写入：
- `scores.data[(bb*T + i)*T + j] = s`

代入：
- `(bb*T + i)*T + j = (0*3 + 1)*3 + 0 = 3`

也就是：
- `scores.data[3] = s`

这正是第 4.1 节中的 $\mathrm{off}_{BTT}(0,1,0)$。


## 5) 追踪一个被 mask 的元素：`scores[0,1,2]`

flat offset：

$$\mathrm{off}_{BTT}(0,1,2)=(0\cdot 3 + 1)\cdot 3 + 2 = 5$$

因为 `j=2 > i=1`，实现会把该得分覆盖为一个很大的负数：
- `scores.data[5] = -1e9`（近似“$-\infty$”）。

这就是本仓库实现 decoder-only causal masking 的方式。


## 6)（可选）继续追踪：probs 与加权求和

当 `scores` 计算好之后：

1) `probs = softmax_lastdim(scores)`
- `probs` 形状同样是 `[B,T,T]`
- 固定 `(b,i)` 时，`probs[b,i,:]` 这一行在内存中是连续的。

2) `att[b,i,c] = sum_j probs[b,i,j] * v[b,j,c]`
- `att` 形状是 `[B,T,C]`

它们沿用同样的 offset 规则：
- `probs` 使用 `off_BTT(b,i,j)`
- `v` 与 `att` 使用 `off_BTC(b,t,c)`

如果你想继续做同样的练习，一个很好的下一目标是追踪：
- `att[0,1,0]`（某个输出 token 的某个通道）


## 参考（论文名）

- “Attention Is All You Need”（scaled dot-product attention）
- GPT 风格 decoder-only 的 causal masking（causal attention）

（这里只引用论文名用于定位；本文档是对本仓库实现的原创讲解。）
