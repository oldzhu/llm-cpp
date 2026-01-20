> [简体中文](autograd_and_memory_layout.zh-CN.md)

# Autograd + memory layout (how to read the C++ like a paper)

This doc covers the two pieces that are usually *implicit* in papers but *crucial* when learning from a from-scratch C/C++ implementation:

1) **How tensors are laid out in memory** (flat arrays + indexing)
2) **How reverse-mode autograd works here** (graph construction + backward execution)

It complements:
- `docs/transformer_math_to_code.md` (equations → functions/files)
- `docs/training_and_inference_walkthrough.md` (runtime flow)


## Part A — Tensor storage and indexing

### A.1 Storage

`nn::Tensor` stores:
- `data`: `shared_ptr<vector<float>>` (flat contiguous buffer)
- `shape`: `vector<int>` (e.g. `{B,T,C}`)

The buffer is **row-major** (C/C++ default). That means the **last dimension is contiguous**.

### A.2 Generic row-major indexing

For shape `[d0,d1,...,d_{k-1}]` and index `[i0,i1,...,i_{k-1}]`, the flat offset is:

$$\text{offset} = (((i_0\cdot d_1 + i_1)\cdot d_2 + i_2)\cdots)\cdot d_{k-1} + i_{k-1}$$

Common special cases used throughout this repo:

- `[B,T,C]` (hidden states)
  - `offset = (b*T + t)*C + c`
- `[B,T,T]` (attention score/prob matrices)
  - `offset = (b*T + i)*T + j`
- `[M,N]` (matmul)
  - `offset = i*N + j`

### A.3 Worked example: `[B=2,T=3,C=4]`

Let `X` be `[2,3,4]`.

- `X[0,0,:]` starts at offset `(0*3+0)*4 = 0`
- `X[0,1,:]` starts at offset `(0*3+1)*4 = 4`
- `X[1,0,:]` starts at offset `(1*3+0)*4 = 12`

So each time step is a contiguous run of `C` floats.

This is why many ops loop over `b`, `t` and then do a tight inner loop over `c`.

### A.4 Reshape semantics in this repo

`nn::reshape(x, new_shape)`:
- **shares** the same `data` buffer (view)
- creates a new `Tensor` object with a new `shape`
- if grads are enabled, it routes gradients back to the parent by summing the flat grad buffer 1:1

Important learning note:
- There is no stride metadata; reshape is only valid when the element count matches exactly.


## Part B — Autograd engine: what happens when you call backward?

### B.1 What a `Tensor` carries

A tensor may carry:
- `requires_grad` (bool)
- `grad` buffer (same size as `data`) when `requires_grad=true`
- optional `node` pointer that encodes *how it was produced*

`node` contains:
- `parents`: vector of parent tensors
- `backward`: a function that consumes `out.grad` and accumulates into each parent’s grad

### B.2 When does an op build a graph?

Almost every op calls an internal helper like `want_grad(t)` which checks:
- global grad mode is enabled (`nn::is_grad_enabled()`)
- AND the tensor has `requires_grad=true`

So graph-building is controlled by:
- **global mode**: `nn::GradMode no_grad(false);` disables graph creation in a scope
- **per-tensor flag**: `requires_grad`

In generation, the code uses `nn::GradMode no_grad(false);` so inference doesn’t build graphs.

### B.3 Backward execution model

Calling `loss.backward()` does:

1) Ensure `loss` is scalar; seed `loss.grad = 1`
2) Build a **topological order** (parents before children)
3) Iterate in reverse topo order and run each node’s `backward` lambda

Implementation:
- `nn::Tensor::backward()` in `src/tensor.cpp`

Key point:
- Gradients are **accumulated** (`+=`) into parent buffers.
- This matches how papers write sums over paths; in code you literally see the `+=`.

### B.4 Example: `add`

Forward math:
$$y = a + b$$

Backward:
$$\frac{\partial L}{\partial a} += \frac{\partial L}{\partial y}$$
$$\frac{\partial L}{\partial b} += \frac{\partial L}{\partial y}$$

In code:
- `nn::add(...)` in `src/ops.cpp` creates a node with parents `{a,b}` and in backward does
  - `(*a.grad)[i] += (*out.grad)[i]`
  - `(*b.grad)[i] += (*out.grad)[i]`

This is a good first “paper equation → C++ loop” to study.

### B.5 Example: `matmul2d`

Forward math:
$$O = A B$$

Backward (matrix calculus):
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} B^T$$
$$\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial O}$$

In code:
- `nn::matmul2d(...)` in `src/ops.cpp` computes these two formulas with explicit loops.

Learning tip:
- Compare the indices in the loops to the shapes:
  - `dA[i,kk]` sums over `j`
  - `dB[kk,j]` sums over `i`

### B.6 Example: softmax backward as a Jacobian-vector product

Softmax:
$$y = \mathrm{softmax}(x)$$

Full Jacobian is expensive. We use the standard efficient identity:
$$\frac{\partial L}{\partial x} = y \odot \left(\frac{\partial L}{\partial y} - \langle \frac{\partial L}{\partial y}, y \rangle\right)$$

In code:
- `nn::softmax_lastdim(...)` in `src/ops.cpp`

### B.7 Cross entropy backward

For logits $z$ and softmax probs $p$ with target class $y$:
$$\frac{\partial L}{\partial z_v} = \frac{1}{N}(p_v - \mathbb{1}[v=y])$$

In code:
- `nn::cross_entropy(...)` in `src/ops.cpp` stores the probs from forward and uses the identity above in backward.


## Part C — Attention memory layout + loops (what to watch)

In 1-head attention we build:
- `scores`: `[B,T,T]`
- `probs`: `[B,T,T]`
- `v`: `[B,T,C]`
- output `att`: `[B,T,C]`

The key loops mirror the math:

1) Scores:
$$S[b,i,j] = (Q[b,i,:]\cdot K[b,j,:]) / \sqrt{C}$$

2) Softmax over `j` (keys)

3) Weighted sum:
$$Y[b,i,c] = \sum_j P[b,i,j] V[b,j,c]$$

In code:
- `nn::self_attention_1h(...)` in `src/ops.cpp`.

Learning tip:
- Track which index is contiguous in memory:
  - `c` is contiguous in `[B,T,C]`
  - `j` is contiguous in `[B,T,T]` when you fix `(b,i)`

This will later matter for performance work (SIMD / blocking / GPU kernels).


## Part D — “Docs contract” for future features

This repo is a learning project. When we add a new feature, we should also add/extend docs so users can map:

- **paper concept / formula** → **exact function(s)** → **exact tensor shapes** → **any important implementation choices**

Practical checklist for every new feature:

1) Update `docs/transformer_math_to_code.md`
- Add equations and list the new/changed functions.

2) Add a dedicated doc if the feature is substantial
- Examples:
  - `docs/multihead_attention.md`
  - `docs/kv_cache.md`
  - `docs/rlhf_overview.md` (if we ever go there)

3) Add targeted code comments
- Prefer short “math block” comments at the top of core functions.
- Prefer shape annotations like `[B,T,C]`.

4) Update `docs/training_and_inference_walkthrough.md` if runtime flow changes
- e.g. new inference path, caching, new checkpoint fields.

5) Add/extend tests when math changes
- At least one regression test to prevent silent correctness breaks.


## References (paper names)

- “Attention Is All You Need” (scaled dot-product attention)
- “Language Models are Unsupervised Multitask Learners” (GPT-2 style decoder-only LM training objective)

(We reference paper names for orientation; this document is an original explanation of this repo’s implementation.)
