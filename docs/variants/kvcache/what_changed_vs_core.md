> [简体中文](what_changed_vs_core.zh-CN.md)

# KV-cache extension: what changed vs the core baseline

This doc explains what the **KV-cache extension** added, what it intentionally did not change, and how it maps to standard transformer inference patterns.

## 0) Big picture

Baseline/core:
- `nn::self_attention_1h` computes full `[B,T,C]` attention every call, recomputing K and V dot products for all `i,j` pairs.

KV-cache extension:
- We added a cache-based incremental path as a **side-by-side variant** under `src/variants/kvcache/`.
- We did **not** modify the baseline attention function.

## 1) What we changed in C++ structures/classes

### 1.1 New data structure: `KVCache`

```cpp
struct KVCache {
    int B, T_max, C;
    int cur_len = 0;

    nn::Tensor k_cache_; // [B, T_max, C]
    nn::Tensor v_cache_; // [B, T_max, C]

    KVCache(int B, int T_max, int C);
    void reset();
};
```

The cache tensors are plain data storage (no autograd). They are only used in generation mode where gradients are disabled.

### 1.2 New functions (not new model class)

Two new functions:
- `nn::variants::kvcache::self_attention_prefill` — fills cache and returns output
- `nn::variants::kvcache::self_attention_step` — uses cache, returns single-position output

Files:
- `src/variants/kvcache/kvcache_attention.h`
- `src/variants/kvcache/kvcache_attention.cpp`

### 1.3 Core model types: unchanged

No changes to:
- `model::Config`, `model::TinyGPT`, `TinyGPT::Block`
- `nn::ops` layer primitives (`linear_lastdim`, `softmax_lastdim`, etc.)

## 2) What changed in parameters / shapes

**Nothing.** Both functions accept the same parameter shapes as `nn::self_attention_1h`:
- `w_qkv: [C, 3C]`, `b_qkv: [3C]`
- `w_proj: [C, C]`, `b_proj: [C]`

The only difference is **how K and V are reused** across calls.

## 3) Math differences

### Prefill: same as baseline, plus cache write

For input `x` of shape `[B,T,C]`:

1. `qkv = linear(x, w_qkv, b_qkv)` → `[B,T,3C]`
2. `q = qkv[:,:,:C]`, `k = qkv[:,:,C:2C]`, `v = qkv[:,:,2C:3C]`
3. **Copy `k` into `cache.k_cache_[:,:T,:]`, `v` into `cache.v_cache_[:,:T,:]`**
4. `cache.cur_len = T`
5. Compute attention exactly as baseline (causal mask, `scale = 1/√C`)

### Step: no causal mask, scalar query (T=1)

For input `x_step` of shape `[B,1,C]`:

1. `qkv = linear(x_step, w_qkv, b_qkv)` → `[B,1,3C]`
2. `q_new = qkv[:,:,:C]`, `k_new = qkv[:,:,C:2C]`, `v_new = qkv[:,:,2C:3C]`
3. **Append `k_new` at position `cache.cur_len`, `v_new` at `cache.cur_len`**
4. `cache.cur_len += 1`
5. Compute scores for query position 0 against all cached K:

$$
S[0,j] = \frac{Q_{new}[0] \cdot K_{cache}[j]}{\sqrt{C}},\quad j \in [0, \text{cur\_len}-1]
$$

No causal mask — the new token is the last position so it can attend to everything.

6. `probs = softmax(scores)`
7. `att[0] = sum_j probs[j] * V_cache[j]`
8. `out = linear(att, w_proj, b_proj)`

### Key simplification

In the step path, T_query = 1 and T_key = cur_len. The attention matrix is `[B,1,cur_len]` instead of `[B,T,T]`. This is the source of the O(T) vs O(T²) speedup.

## 4) Build target changes (CMake)

Added `llm_variant_kvcache` library:
- Compiles `src/variants/kvcache/kvcache_attention.cpp`
- Links against `llm_core`

`test_build_llm` links `llm_variant_kvcache` for verification tests.

## 5) Tests

One focused test proving cache equivalence:
- `test_kvcache_matches_full_attention()`:
  - Run prefill on T tokens (fills cache).
  - Run step on T+1-st token (uses cache).
  - Re-run attention from scratch on all T+1 tokens.
  - Verify the T+1-st output matches.

## 6) When to use

The KV-cache is a generation-only optimization. Training still uses the full-attention path. The cache is useful in:
- CLI generation mode (`--prompt ... --gen N`)
- Any autoregressive text generation scenario

## 7) What's next

- Combine with MHA variant for multi-head KV-cache.
- Extend backend seam to route per-step attention through accelerated kernels.
- Enable GPU-resident cache for zero-copy inference.
