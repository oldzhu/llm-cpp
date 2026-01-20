> [English](hidden_states_vs_attention_matrices.md)

# Hidden states `[B,T,C]` vs attention 矩阵 `[B,T,T]`（含 flat-buffer offsets）

这是一份针对 decoder 中两种常见张量形状的速查笔记：

- **Hidden states**：`X[b,t,c]`，形状 `[B,T,C]`
- **Attention scores/probs**：`S[b,i,j]` / `P[b,i,j]`，形状 `[B,T,T]`

目标：在你去读 `nn::self_attention_1h` 的嵌套 loops 之前，让索引看起来“显然”。

记号：
- `b`：batch 索引
- `t`：token 位置（time）
- `c`：channel（特征维）
- `i`：query 位置
- `j`：key 位置

所有张量以 row-major 存储（最后一维连续）。


## 1) Hidden states：`[B,T,C]`

含义：
- 对每个 batch 元素 `b`
- 对每个 token 位置 `t`
- 存一个长度为 `C` 的**向量**（token embedding / hidden state）

索引：
- `X[b,t,c]`

flat offset：

$$
	ext{offset}(b,t,c) = (b\cdot T + t)\cdot C + c
$$

### 例子 A（简单数字）

令 `B=1`, `T=3`, `C=4`。

`X` 一共有 `1*3*4 = 12` 个 float。

- token 0 向量 `X[0,0,:]` 占 offsets `0..3`
- token 1 向量 `X[0,1,:]` 占 offsets `4..7`
- token 2 向量 `X[0,2,:]` 占 offsets `8..11`

具体：
- `X[0,1,2]` → `(0*3 + 1)*4 + 2 = 6`
- `X[0,2,0]` → `(0*3 + 2)*4 + 0 = 8`

因此 `[B,T,C]` 的直觉是“按 token、按 channel 存向量”。

### 例子 B（两个 batch）

令 `B=2`, `T=3`, `C=4`。

- batch 0 占 offsets `0..11`
- batch 1 占 offsets `12..23`

例如：
- `X[1,0,0]` → `(1*3 + 0)*4 + 0 = 12`
- `X[1,2,3]` → `(1*3 + 2)*4 + 3 = 23`


## 2) 注意力 scores/probs：`[B,T,T]`

含义：
- 对每个 batch 元素 `b`
- 对每个 query 位置 `i`
- 对每个 key 位置 `j` 存一个标量得分/概率

所以每一行 `S[b,i,:]` 是一个长度为 `T` 的向量：token `i` 对所有 token `j` 的关注程度。

索引：
- `S[b,i,j]`

flat offset：

$$
	ext{offset}(b,i,j) = (b\cdot T + i)\cdot T + j
$$

### 例子 C（矩阵视角）

令 `B=1`, `T=3`。

`S` 一共有 `1*3*3 = 9` 个 float。

对 `b=0`，可以把它看作 3×3 矩阵：

- 行 `i=0` 是 offsets `0..2`（列 `j=0..2`）
- 行 `i=1` 是 offsets `3..5`
- 行 `i=2` 是 offsets `6..8`

例如：
- `S[0,2,1]` → `(0*3 + 2)*3 + 1 = 7`
- `S[0,0,2]` → `(0*3 + 0)*3 + 2 = 2`

它的直觉是“token-to-token”：对每个 `(i,j)` 一标量。


## 3) 为什么注意力矩阵没有 `c` 维

在注意力中，得分来自两个长度为 `C`（或每 head 长度为 `D`）向量的**点积**：

$$
S[b,i,j] = \frac{Q[b,i,:] \cdot K[b,j,:]}{\sqrt{C}} + \text{mask}(i,j)
$$

展开点积：

$$
Q[b,i,:] \cdot K[b,j,:] = \sum_{c=0}^{C-1} Q[b,i,c]\;K[b,j,c]
$$

这一步求和把 channel 维 `c` **约掉**，因此对每个 `(b,i,j)` 只产生 **一个标量**。

接下来 softmax：

$$
P[b,i,:] = \mathrm{softmax}(S[b,i,:])
$$

仍然是一标量对应一个 `(b,i,j)`。


## 4) 一个端到端小例子（只看 shapes）

设 `B=1, T=3, C=4`：

- hidden states：`X` 是 `[1,3,4]`
- 投影后的 Q/K/V 都是 `[1,3,4]`
- scores：`S` 是 `[1,3,3]`
- probs：`P` 是 `[1,3,3]`
- attention 输出：`Y` 是 `[1,3,4]`

关键心智动作：
- `C` 是“每个 token 的向量宽度”。
- `T` 是 token 个数。
- scores/probs 以 token 对 `(i,j)` 为坐标，因此是 `T×T`。


## 5) 这在代码里体现在哪里

基线注意力实现：
- `src/ops.cpp` 中的 `nn::self_attention_1h`

评分 loops 直接遵循上面的索引方式：
- 先为 `[B,T,C]` buffers 计算 `qi` 与 `kj` base offsets
- 再用 `(b,i,j)` 公式写入 `[B,T,T]` buffer

下一步（如果你准备好了）：选定具体的 `B,T,C,b,i,j`，从 `Q`/`K` 的内存 offsets 一路追踪到最终的 `S[b,i,j]` 位置。
