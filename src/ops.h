#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tensor.h"

namespace nn {

Tensor add(const Tensor& a, const Tensor& b);
Tensor add_inplace(const Tensor& a, const Tensor& b); // returns new tensor, but computes like inplace add
Tensor mul(const Tensor& a, const Tensor& b);
Tensor mul_scalar(const Tensor& a, float s);
Tensor add_scalar(const Tensor& a, float s);

Tensor matmul2d(const Tensor& a, const Tensor& b); // (m,k) @ (k,n) -> (m,n)
Tensor bmm(const Tensor& a, const Tensor& b);      // (B,M,K) @ (B,K,N) -> (B,M,N)

Tensor gelu(const Tensor& x);
Tensor layernorm_lastdim(const Tensor& x, float eps);
Tensor softmax_lastdim(const Tensor& x);

Tensor reshape(const Tensor& x, const std::vector<int>& new_shape);

// Embedding lookup: idx is [B,T] int32 tokens.
Tensor embedding(const Tensor& weight_vocab_by_dim, const std::vector<std::int32_t>& idx_bt, int B, int T);

// Causal self-attention (1 head) for simplicity.
// x: [B,T,C], w_qkv: [C, 3C], b_qkv: [3C], w_proj: [C,C], b_proj: [C]
Tensor self_attention_1h(const Tensor& x,
                         const Tensor& w_qkv,
                         const Tensor& b_qkv,
                         const Tensor& w_proj,
                         const Tensor& b_proj);

// Linear on last dim for [B,T,Cin] with weight [Cin,Cout], bias [Cout].
Tensor linear_lastdim(const Tensor& x, const Tensor& w, const Tensor& b);

// Cross entropy for logits [N,V] and targets [N]. Returns scalar loss.
Tensor cross_entropy(const Tensor& logits_nv, const std::vector<std::int32_t>& targets_n);

} // namespace nn
