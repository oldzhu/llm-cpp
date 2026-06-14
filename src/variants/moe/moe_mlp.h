#pragma once

#include "tensor.h"

namespace nn::variants::moe {

// MoE MLP: mixture-of-experts feed-forward network.
//
// For each token in x [B*T, C]:
//   1. Router: gates = softmax(x @ w_router + b_router) → [n_experts]
//   2. Top-K: keep top_k experts, zero rest, renormalize
//   3. For each selected expert e: y_e = GELU(x @ w_fc_e + b_fc_e) @ w_out_e + b_out_e
//   4. Output: y = sum_e gate[e] * y_e
//
// Returns {output [B*T, C], balance_loss (scalar)}.
// Input x is already flattened to [B*T, C].
// expert_weights: array of 4 Tensors per expert {w_fc, b_fc, w_out, b_out}
//   all stored as raw pointers in a flat list: [e0_wfc, e0_bfc, e0_wout, e0_bout, e1_wfc, ...]

struct MoEOutput {
  Tensor y;           // [B*T, C]
  Tensor balance_loss; // [1] scalar
};

MoEOutput moe_mlp_forward(const Tensor& x_2d,           // [N, C]
                           const Tensor& w_router,      // [C, n_experts]
                           const Tensor& b_router,      // [n_experts]
                           const std::vector<const Tensor*>& expert_params, // n_experts * 4 pointers
                           int n_experts,
                           int top_k,
                           int interm_dim);

} // namespace nn::variants::moe
