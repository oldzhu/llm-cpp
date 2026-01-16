#pragma once

#include "ops.h"

namespace nn::variants::mha {

// Naive multi-head causal self-attention.
//
// Parameters follow the baseline layout:
// - x:      [B,T,C]
// - w_qkv:  [C,3C], b_qkv: [3C]
// - w_proj: [C,C],  b_proj: [C]
//
// The only difference from `nn::self_attention_1h` is that we *split the channel*
// dimension into heads:
// - C = n_heads * head_dim
// - Q,K,V are reshaped to [B,T,n_heads,head_dim]
// - attention is computed independently per head
// - heads are concatenated back to [B,T,C] then projected.
//
// This variant is intentionally slow/explicit to keep indexing and autograd easy
// to verify.
Tensor self_attention_mha(const Tensor& x,
                          const Tensor& w_qkv,
                          const Tensor& b_qkv,
                          const Tensor& w_proj,
                          const Tensor& b_proj,
                          int n_heads);

} // namespace nn::variants::mha
