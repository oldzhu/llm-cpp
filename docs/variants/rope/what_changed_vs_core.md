> [简体中文](what_changed_vs_core.zh-CN.md)

# RoPE extension: what changed vs the core baseline

## 0) Big picture

Baseline: absolute positional embeddings `Wpe[pos]` added to token embeddings before attention.

RoPE variant: position encoded via **rotation** of Q and K vectors, producing relative position awareness.

## 1) What we changed

### Removed
- Absolute positional embedding addition (no `add_positional` step)

### Added
- `rope_rotate()`: rotates Q and K in-place by position-dependent angles
- `self_attention_rope()`: full attention function with RoPE

### Unchanged
- V (values) are not rotated
- Attention computation (scores, softmax, weighted sum, projection) is identical to baseline
- Parameter layout: `w_qkv: [C,3C]`, `w_proj: [C,C]`, `bias` shapes all same

## 2) Math differences

Baseline forward:
```
X = embed(tokens) + Wpe[pos]
[Q,K,V] = X @ W_qkv + b_qkv
attention(Q,K,V) → output
```

RoPE forward:
```
X = embed(tokens)         // no wpe
[Q,K,V] = X @ W_qkv + b_qkv
Q' = RoPE_rotate(Q)       // position-aware rotation
K' = RoPE_rotate(K)
attention(Q',K',V) → output
```

## 3) Key property: relative position

`dot(RoPE(Q_i), RoPE(K_j))` depends on `cos((i-j) * theta_k)` — the relative position, not the absolute position.

## 4) When to use

- RoPE replaces the need for learned positional embeddings
- Works with any sequence length (no wpe size limitation)
- Compatible with KV-cache (RoPE can be applied per-step)
