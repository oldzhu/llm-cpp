> [简体中文](what_changed_vs_core.zh-CN.md)

# GQA extension: what changed vs MHA

## 0) Big picture

MHA: Q, K, V all have `n_heads`. Each Q head independently attends to its own K/V head.

GQA: K and V have `n_kv_heads` (fewer). Each K/V head serves `n_heads / n_kv_heads` Q heads.

## 1) What changed

### API
- Takes 4D Q, K, V tensors (pre-split into heads)
- Additional parameter: `n_kv_heads`

### Head mapping
```
kv_idx(q_head) = q_head * n_kv_heads / n_heads
```

### Constraints
- `n_heads % n_kv_heads == 0`
- All heads have same head dimension D

## 2) Memory impact

KV-cache size proportional to `n_kv_heads` instead of `n_heads`. With n_kv_heads=2, n_heads=8: 4× reduction in KV-cache memory.

## 3) When to use

- Reduce KV-cache memory in inference
- Special case n_kv_heads=1 is MQA (Multi-Query Attention)
- Special case n_kv_heads=n_heads is standard MHA
