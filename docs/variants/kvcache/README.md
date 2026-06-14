> [简体中文](README.zh-CN.md)

# KV-cache variant

This variant adds an **incremental forward path** using a Key-Value cache, so that autoregressive generation avoids recomputing K and V for all past tokens at every step.

Code:
- `src/variants/kvcache/kvcache_attention.h`
- `src/variants/kvcache/kvcache_attention.cpp`

## What changes vs the baseline?

Baseline (`nn::self_attention_1h`) recomputes Q, K, V for the entire sequence `[B,T,C]` every time a new token is generated.

This variant splits attention into two API surfaces:

1. **Prefill** — `self_attention_prefill(x, w_qkv, b_qkv, w_proj, b_proj, cache)`:
   - Computes Q, K, V for the full context of length T.
   - Stores K and V into the cache.
   - Computes attention normally (with causal masking).
   - Returns output `[B,T,C]`.

2. **Step** — `self_attention_step(x_step, w_qkv, b_qkv, w_proj, b_proj, cache)`:
   - Takes a single new token `[B,1,C]`.
   - Computes `Q_new, K_new, V_new`.
   - Appends `K_new, V_new` to the cache.
   - Computes attention of `Q_new` against **all cached K** (no causal mask needed — the new token is at the end of the sequence).
   - Returns output `[B,1,C]`.

Time complexity changes:
- Baseline generation: O(T²) per step (dot product with all past K)
- KV-cache generation: O(T) per step (dot product with cached K, no recomputation)

## Structure

```
KVCache {
    k_cache_ : [B, T_max, C]  // cached keys
    v_cache_ : [B, T_max, C]  // cached values
    cur_len  : int            // current cache length
}
```

The cache stores plain `nn::Tensor` without autograd (generation-only path).

## Why this variant exists

- Enables efficient autoregressive generation (no redundant computation).
- Prerequisite for GPU-based inference (cached K/V can stay on device).
- Follows the same variant pattern as MHA: docs + code + test.
- Keeps indexing explicit so learners can trace every memory access.

## Intended use in generation

```
// Prefill: process the full prompt
auto out = kvcache::self_attention_prefill(x_prompt, ..., cache);

// Step: generate one token at a time
for each new token:
    auto out_step = kvcache::self_attention_step(x_new, ..., cache);
```
