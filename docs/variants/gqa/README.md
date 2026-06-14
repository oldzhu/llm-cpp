> [简体中文](README.zh-CN.md)

# Grouped-Query Attention (GQA) variant

This variant adds **Grouped-Query Attention**, where K and V have fewer heads than Q. Multiple Q heads share the same KV head.

Code:
- `src/variants/gqa/gqa_attention.h`
- `src/variants/gqa/gqa_attention.cpp`

## What changes vs MHA?

MHA: same number of heads for Q, K, V. Each Q head attends to its own K/V head.

GQA: fewer K/V heads. Each K/V head is shared by a group of Q heads.

## Math

```
Q: [B, T, n_heads, D]
K: [B, T, n_kv_heads, D]
V: [B, T, n_kv_heads, D]

heads_per_kv = n_heads / n_kv_heads
kv_idx(q_head) = q_head / heads_per_kv

For each (b, i), each Q head hq:
  hv = kv_idx(hq)
  S_hq[i,j] = dot(Q[b,i,hq,:], K[b,j,hv,:]) / sqrt(D) + causal_mask
  P_hq[i,:] = softmax(S_hq[i,:])
  Y_hq[i,:] = sum_j P_hq[i,j] * V[b,j,hv,:]

Output: concat(Y_hq) → [B, T, n_heads * D]
```

## Special cases

- `n_kv_heads == n_heads` → standard Multi-Head Attention
- `n_kv_heads == 1` → Multi-Query Attention (MQA, all Q share one KV)

## Why this variant exists

- GQA is used in LLaMA 2/3, reducing KV-cache memory by `n_heads / n_kv_heads`
- Demonstrates sharing pattern: multiple Q heads map to the same K/V
- Follows the variant pattern: docs + code + test

## Paper reference

"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023)
