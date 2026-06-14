#pragma once

#include "tensor.h"

namespace nn::variants::rope {

// Apply Rotary Positional Embedding (RoPE) to Q and K.
//
// For each position i and dimension pair (2k, 2k+1):
//   theta_k = 1.0 / pow(10000, 2*k / C)
//   cos = cos(i * theta_k), sin = sin(i * theta_k)
//   q'[2k]   = q[2k] * cos - q[2k+1] * sin
//   q'[2k+1] = q[2k+1] * cos + q[2k] * sin
//   same rotation for k
//
// q: [B,T,C], k: [B,T,C] — modified in-place.
// C must be even (Rotary works in pairs).
void rope_rotate(Tensor& q, Tensor& k, int B, int T, int C);

// Self-attention with RoPE (single head).
// Same parameter layout as self_attention_1h:
//   x: [B,T,C], w_qkv: [C,3C], b_qkv: [3C], w_proj: [C,C], b_proj: [C]
//
// Differs from baseline by:
// 1. No positional embedding addition (RoPE replaces absolute positions)
// 2. Q and K are rotated via RoPE before computing scores
Tensor self_attention_rope(const Tensor& x,
                           const Tensor& w_qkv,
                           const Tensor& b_qkv,
                           const Tensor& w_proj,
                           const Tensor& b_proj);

} // namespace nn::variants::rope
