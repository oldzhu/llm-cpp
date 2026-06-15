#pragma once

#include "tensor.h"

namespace nn::variants::mla {

// Multi-Head Latent Attention (MLA) — DeepSeek V2/V3-style.
//
// Unlike standard attention where K and V are projected directly from input,
// MLA first compresses input into a low-dimensional latent representation,
// then decompresses K and V from the latent.
//
// This reduces KV-cache size from 2*C to latent_dim (typically ~1/4 to ~1/16).
//
// Parameters:
//   x:      [B,T,C]  — input hidden states
//   w_q:    [C,C]    — query projection
//   b_q:    [C]      — query bias
//   w_dkv:  [C, L]   — KV compression (down-projection), L = latent_dim
//   b_dkv:  [L]      — compression bias
//   w_uk:   [L, C]   — key decompression (up-projection)
//   w_uv:   [L, C]   — value decompression (up-projection)
//   w_o:    [C, C]   — output projection
//   b_o:    [C]      — output bias
//
// Math:
//   Q     = x·w_q + b_q           // [B,T,C]
//   c_KV  = x·w_dkv + b_dkv       // [B,T,L]  (compressed, L << C)
//   K     = c_KV·w_uk             // [B,T,C]  (decompressed)
//   V     = c_KV·w_uv             // [B,T,C]  (decompressed)
//   attn  = softmax(Q·K^T/√C + mask)·V  → [B,T,C]
//   out   = attn·w_o + b_o        // [B,T,C]
//
// Requires: latent_dim (L) < C for compression benefit.

Tensor self_attention_mla(const Tensor& x,
                           const Tensor& w_q,
                           const Tensor& b_q,
                           const Tensor& w_dkv,
                           const Tensor& b_dkv,
                           const Tensor& w_uk,
                           const Tensor& w_uv,
                           const Tensor& w_o,
                           const Tensor& b_o);

} // namespace nn::variants::mla
