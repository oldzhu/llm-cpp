> [English](2026-01-25_backend_seam_bmm_vs_cuda.md)

# 2026-01-25 — 记录：后端缝、`bmm` 与注意力里的 “batched GEMM”

背景：我们已经为 `nn::matmul2d` 引入了 CPU 后端缝（这样未来接 CUDA/HIP 时无需改模型代码）。在决定下一步（先把缝扩展到 `bmm` 还是直接做 CUDA）之前，这里把讨论内容记录下来，并补充一段教学说明：什么是 “batched GEMM”，以及它为什么与 attention 强相关。

## 1) “把后端缝扩展到 `bmm`”是什么意思？

当前有两个相关算子：

- `nn::matmul2d(a, b)`：**二维矩阵**相乘
  - `a`: `[m,k]`
  - `b`: `[k,n]`
  - 输出: `[m,n]`

- `nn::bmm(a, b)`：**批量矩阵**相乘
  - `a`: `[B,M,K]`
  - `b`: `[B,K,N]`
  - 输出: `[B,M,N]`
  - 语义：对每个 batch 下标 `bb`，执行 `out[bb] = a[bb] @ b[bb]`

今天：`matmul2d` 已经走后端接口；但 `bmm` 仍在 `src/ops.cpp` 里用 CPU loop 直接实现。

所以“把缝扩展到 `bmm`”就是让 `nn::bmm` 也通过后端执行（像 `nn::matmul2d` 一样），这样未来 CUDA/HIP 不会只对 2D matmul 有“单点加速路径”。

### 两个增量实现选项

方案 A（最简单的缝）：把 `bmm` 实现为对 backend `matmul2d` 的 `B` 次调用。

- 优点：接口增长最小；非常直观。
- 缺点：在 GPU 上通常不够快（每个 batch 一次 kernel/library 调用，会有 launch 开销）。

方案 B（更完整的缝）：在后端接口里加入 `bmm_fwd/bmm_bwd` 虚函数。

- 优点：后端可以用真正的 batched GEMM（厂商库或自研 kernel）。
- 缺点：接口更大；维护成本更高。

## 2) 为什么在 attention 里很重要：优化版 attention 经常大量使用 batched GEMM

基线实现里，attention 主要用显式 loop 和中间张量。更优化的实现（以及大多数生产栈）会把重计算表达成 **GEMM** 或 **batched GEMM**，因为厂商库可以把它跑得非常快。

### 什么是 GEMM？

GEMM 是通用矩阵乘法（General Matrix Multiply）的常用叫法：

- $C = A B$，其中 $A$ 是 `[m,k]`，$B$ 是 `[k,n]`。

### 什么是 batched GEMM？

batched GEMM 指：你有很多个 shape 相同、彼此独立的矩阵乘法，希望作为一个“批量操作”执行：

- 对 `bb = 0..B-1`：$C_{bb} = A_{bb} B_{bb}$

这与 `bmm` 的语义一一对应：

- `A`: `[B,M,K]`
- `B`: `[B,K,N]`
- `C`: `[B,M,N]`

优秀的 GPU 后端通常会针对 batched GEMM 提供专门实现（或进行融合），而不是简单地 launch `B` 次单独 GEMM。

### Attention 例子：QK^T 与 PV 本质上是 batched GEMM

多头注意力常见 shape：

- `Q`: `[B, H, T, D]`
- `K`: `[B, H, T, D]`
- `V`: `[B, H, T, D]`

对每个 batch `b` 和 head `h`：

1) scores：$S[b,h] = Q[b,h] @ K[b,h]^T$ 得到 `[T,T]`
2) probs：`P = softmax(S)` 得到 `[T,T]`
3) 输出：$Y[b,h] = P[b,h] @ V[b,h]$ 得到 `[T,D]`

这两次矩阵乘法都可以看成在 `B*H` 个独立问题上做同样 shape 的 GEMM，因此天然适合 batched GEMM。

#### 具体数字示例

取：

- `B = 2`（batch size）
- `H = 4`（heads）
- `T = 8`（序列长度）
- `D = 16`（head_dim）

则每个 `(b,h)`：

- `Q[b,h]` 是 `[T,D] = [8,16]`
- `K[b,h]^T` 是 `[D,T] = [16,8]`
- scores 是 `[8,16] @ [16,8] -> [8,8]`

一共有 `B*H = 8` 个这样的 GEMM。

可以把它表达为 batched GEMM：

- `A` = Q reshape/pack 为 `[B*H, T, D]` = `[8, 8, 16]`
- `B` = K reshape/pack 为 `[B*H, D, T]` = `[8, 16, 8]`
- `C` = scores 为 `[B*H, T, T]` = `[8, 8, 8]`

第二次乘法 `P @ V` 也是 batched GEMM：

- `A` = probs `[B*H, T, T]` = `[8, 8, 8]`
- `B` = V `[B*H, T, D]` = `[8, 8, 16]`
- `C` = Y `[B*H, T, D]` = `[8, 8, 16]`

### 为什么这会影响我们下一步的选择

- 如果后端缝只覆盖 `matmul2d`，我们仍然可以很快做出 CUDA POC。
- 如果尽早让 `bmm` 也走后端缝，未来做 attention/MHA 优化时会更统一（尤其是面向 GEMM 的实现风格）。

两条路线都合理；取舍是：现在更简单 vs 未来在 attention 上的路径更干净。
