#pragma once

#include "tensor.h"

namespace nn::variants::gqa {

// Grouped-Query Attention (GQA).
//
// Q: [B, T, n_heads, D]     — n_heads query heads
// K: [B, T, n_kv_heads, D]  — n_kv_heads key heads (n_kv_heads <= n_heads)
// V: [B, T, n_kv_heads, D]  — n_kv_heads value heads
//
// Each Q head maps to one KV head:
//   kv_idx(q_head) = q_head * n_kv_heads / n_heads
//
// Scale: 1 / sqrt(D)
// Causal mask: j > i → -inf
//
// Returns: [B, T, n_heads * D]  (concatenated heads)
// Requires: n_heads % n_kv_heads == 0
Tensor self_attention_gqa(const Tensor& q_4d,
                          const Tensor& k_4d,
                          const Tensor& v_4d,
                          int n_heads,
                          int n_kv_heads);

} // namespace nn::variants::gqa
