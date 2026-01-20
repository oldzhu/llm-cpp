> [English](learn_by_hand.md)

# MHA 可手算示例

目标：在不被大矩阵淹没的情况下，检查 *shape 逻辑* 与“每个 head 独立计算”的关键点。

## 设置

使用一个 batch、两个 token：
- `B=1`, `T=2`

使用两个 head：
- `H=2`

每个 head 的维度取 1：
- `D=1`，因此总模型维度为 `C = H*D = 2`

于是每个 token 的 embedding 是一个长度为 2 的向量，我们把它看成两个 head：

- head0 使用通道 0
- head1 使用通道 1

假设做完 QKV 线性层之后，我们已经得到了：

- `Q,K,V` 的 shape 都是 `[T,C] = [2,2]`（省略 batch 维）

记：

- token 0：`q0=[q00,q01]`, `k0=[k00,k01]`, `v0=[v00,v01]`
- token 1：`q1=[q10,q11]`, `k1=[k10,k11]`, `v1=[v10,v11]`

把通道 reshape 成 head（概念上）：
- head0 看到的是标量对 `(q00,q10)` 等
- head1 看到的是标量对 `(q01,q11)` 等

## 分数（因果 mask）

对 head0（维度是标量）：

- 对 token `i=0`：
	- 到 `j=0` 的 score：`s00 = q00*k00 / sqrt(1)`
	- 到 `j=1`：未来位置被 mask → `-inf`
- 对 token `i=1`：
	- 到 `j=0`：`s10 = q10*k00`
	- 到 `j=1`：`s11 = q10*k10`

head1 完全相同，只是用通道 1：
- `s00' = q01*k01`，等等。

## Softmax 与输出

因为 `T=2`，每行 softmax 很容易：
- 对 token 0，每个 head 的 softmax 都是 `[1, 0]`（因为 `j=1` 被 mask）。
- 对 token 1，每个 head 的 softmax 是对两个数做 softmax。

每个 head 的输出：
- `y0_head0 = v00`
- `y0_head1 = v01`

因此对 token 0，拼接后的结果是 `[v00, v01]`。

这是一个很好的检查：在因果 mask 下，第一个 token 只能看自己，所以输出必须恰好等于它自己的 $V$（对每个 head 都成立）。

## 映射到代码

这些步骤对应的循环在：
- `src/variants/mha/mha_attention.cpp`（`scores`、`softmax_lastdim`、以及 `att` 的循环）

因果 mask 的规则与基线相同：`j > i` 被 mask。
