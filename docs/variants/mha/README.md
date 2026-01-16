# Multi-head attention variant (MHA)

This variant adds a **naive, explicit** multi-head causal self-attention implementation.

Code:
- `src/variants/mha/mha_attention.h`
- `src/variants/mha/mha_attention.cpp`

## What changes vs the baseline?

Baseline (`nn::self_attention_1h`) treats the channel dimension as one head of width $C$.

This variant (`nn::variants::mha::self_attention_mha`) keeps the *same parameter layout*:
- `w_qkv: [C,3C]`, `b_qkv: [3C]`
- `w_proj: [C,C]`, `b_proj: [C]`

…but changes the *interpretation* of $C$:

- Choose `n_heads = H`
- Require $C = H \cdot D$ where $D$ is the per-head dimension
- After computing `q,k,v` as `[B,T,C]`, reshape each to `[B,T,H,D]`

Then for each head $h$:

$$
S_{h}[i,j] = \frac{Q_{h}[i] \cdot K_{h}[j]}{\sqrt{D}} + \text{causalMask}(i,j)
$$

$$
P_{h}[i,:] = \mathrm{softmax}(S_{h}[i,:])
$$

$$
Y_{h}[i] = \sum_{j} P_{h}[i,j] V_{h}[j]
$$

Concatenate heads back to `[B,T,C]` (implemented as a reshape from `[B,T,H,D]`), then apply the output projection.

## Why this variant exists

- It’s a **teaching-first template** for adding variants:
  - A new module under `src/variants/...`
  - A small set of docs under `docs/variants/...`
  - A focused test proving something important
- It keeps indexing explicit so we can later:
  - validate correctness
  - add KV-cache/GQA/RoPE side-by-side
  - swap kernels behind a backend boundary without changing semantics
