> [简体中文](2026-01-25_backend_seam_bmm_vs_cuda.zh-CN.md)

# 2026-01-25 — Notes: backend seam, `bmm`, and “batched GEMM” in attention

Context: We introduced a CPU backend seam for `nn::matmul2d` (so later CUDA/HIP can plug in without changing model code). Before choosing the next step (extend seam to `bmm` vs go directly to CUDA), we captured the discussion and added a teaching note on “batched GEMM”.

## 1) What it means to “extend the backend seam to `bmm`”

We currently have two related ops:

- `nn::matmul2d(a, b)` multiplies **2D matrices**
  - `a`: `[m,k]`
  - `b`: `[k,n]`
  - output: `[m,n]`

- `nn::bmm(a, b)` multiplies a **batch of matrices**
  - `a`: `[B,M,K]`
  - `b`: `[B,K,N]`
  - output: `[B,M,N]`
  - semantics: for each batch index `bb`, do `out[bb] = a[bb] @ b[bb]`

Today, `matmul2d` routes through the backend interface. `bmm` still uses CPU loops directly inside `src/ops.cpp`.

So “extend the seam to `bmm`” means making `nn::bmm` also call into the backend (similar to `nn::matmul2d`) so that future CUDA/HIP work doesn’t have a one-off fast path only for 2D.

### Two incremental implementation options

Option A (simplest seam): implement `bmm` as `B` calls to backend `matmul2d`.

- Pros: minimal interface growth; easy to reason about.
- Cons: not the fastest on GPU (kernel launch overhead per batch).

Option B (more complete seam): add explicit `bmm_fwd/bmm_bwd` virtuals in the backend.

- Pros: allows a backend to use a true batched GEMM implementation (library or fused kernel).
- Cons: larger interface; more code to maintain.

## 2) Why we care: attention uses batched GEMMs in optimized variants

In the baseline, attention is implemented with explicit loops and intermediate tensors. In more optimized implementations (and in most production stacks), the heavy work is expressed as **GEMM** (matrix multiply) or **batched GEMM** calls, because vendor libraries can run them extremely efficiently.

### What is GEMM?

GEMM is the standard name for general matrix multiplication:

- $C = A B$ where $A$ is `[m,k]` and $B$ is `[k,n]`.

### What is batched GEMM?

Batched GEMM means you have **many independent matrix multiplications** of the *same* shapes and want to run them as one logical operation:

- for `bb = 0..B-1`: $C_{bb} = A_{bb} B_{bb}$

This corresponds exactly to `bmm` with shapes:

- `A`: `[B,M,K]`
- `B`: `[B,K,N]`
- `C`: `[B,M,N]`

A good GPU backend often has a specialized path for this (or can fuse it), instead of launching `B` separate GEMMs.

### Attention example: QK^T and PV are batched GEMMs

For multi-head attention, a common shape convention is:

- `Q`: `[B, H, T, D]`
- `K`: `[B, H, T, D]`
- `V`: `[B, H, T, D]`

Compute (per batch `b` and head `h`):

1) scores: $S[b,h] = Q[b,h] @ K[b,h]^T$ producing `[T,T]`
2) probs: `P = softmax(S)` producing `[T,T]`
3) output: $Y[b,h] = P[b,h] @ V[b,h]$ producing `[T,D]`

Those two matmuls are **batched across** `B * H` independent problems.

#### Concrete numeric sample

Let:

- `B = 2` (batch size)
- `H = 4` (heads)
- `T = 8` (sequence length)
- `D = 16` (head_dim)

Then:

- `Q` and `K` blocks: each `(b,h)` has a matrix `[T,D] = [8,16]`
- `K^T` is `[D,T] = [16,8]`
- scores per (b,h): `[8,16] @ [16,8] -> [8,8]`

You have `B*H = 8` such GEMMs.

This is naturally expressed as a batched GEMM:

- `A` = Q reshaped/packed as `[B*H, T, D]` = `[8, 8, 16]`
- `B` = K reshaped/packed as `[B*H, D, T]` = `[8, 16, 8]`
- `C` = scores `[B*H, T, T]` = `[8, 8, 8]`

Then the second matmul `P @ V` is another batched GEMM:

- `A` = probs `[B*H, T, T]` = `[8, 8, 8]`
- `B` = V `[B*H, T, D]` = `[8, 8, 16]`
- `C` = Y `[B*H, T, D]` = `[8, 8, 16]`

### Why this affects our next step choice

- If we keep only `matmul2d` behind the seam, we can still build a CUDA POC quickly.
- If we also route `bmm` through the seam early, it makes later attention optimizations more uniform (especially for MHA variants), and encourages writing attention in a “GEMM-first” style.

Both are valid; the tradeoff is simplicity now vs a cleaner path for attention later.
