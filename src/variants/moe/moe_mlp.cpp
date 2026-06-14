#include "variants/moe/moe_mlp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "ops.h"

namespace nn::variants::moe {

MoEOutput moe_mlp_forward(const Tensor& x_2d,
                           const Tensor& w_router,
                           const Tensor& b_router,
                           const std::vector<const Tensor*>& expert_params,
                           int n_experts,
                           int top_k,
                           int interm_dim) {
  if (x_2d.shape.size() != 2) throw std::runtime_error("MoE: x must be [N, C]");
  const int N = x_2d.shape[0];
  const int C = x_2d.shape[1];
  if (n_experts <= 0 || top_k <= 0 || top_k > n_experts) throw std::runtime_error("MoE: invalid n_experts/top_k");
  if (static_cast<int>(expert_params.size()) != n_experts * 4)
    throw std::runtime_error("MoE: expert_params size mismatch");

  // 1) Router: gates = softmax(x @ w_router + b_router)
  // Use raw matmul2d to avoid reshape → linear_lastdim chain
  Tensor router_raw = nn::matmul2d(x_2d, w_router); // [N, n_experts]
  Tensor logits = Tensor::zeros({N, n_experts}, false);
  for (int n = 0; n < N; ++n) {
    for (int e = 0; e < n_experts; ++e) {
      (*logits.data)[static_cast<std::size_t>(n) * n_experts + e] =
          (*router_raw.data)[static_cast<std::size_t>(n) * n_experts + e] + (*b_router.data)[static_cast<std::size_t>(e)];
    }
  }
  Tensor gates = nn::softmax_lastdim(logits); // [N, n_experts]

  // 2) Top-K selection
  std::vector<float> expert_fraction(static_cast<std::size_t>(n_experts), 0.0f);
  std::vector<float> expert_prob(static_cast<std::size_t>(n_experts), 0.0f);

  std::vector<float> topk_gates(static_cast<std::size_t>(N) * n_experts, 0.0f);
  for (int n = 0; n < N; ++n) {
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(static_cast<std::size_t>(n_experts));
    for (int e = 0; e < n_experts; ++e) {
      ranked.emplace_back((*gates.data)[static_cast<std::size_t>(n) * n_experts + e], e);
    }
    std::nth_element(ranked.begin(), ranked.begin() + top_k, ranked.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    float sum = 0.0f;
    for (int k = 0; k < top_k; ++k) sum += ranked[static_cast<std::size_t>(k)].first;
    if (sum > 0.0f) {
      for (int k = 0; k < top_k; ++k) {
        const int e = ranked[static_cast<std::size_t>(k)].second;
        topk_gates[static_cast<std::size_t>(n) * n_experts + e] = ranked[static_cast<std::size_t>(k)].first / sum;
        expert_fraction[static_cast<std::size_t>(e)] += 1.0f;
      }
    }

    for (int e = 0; e < n_experts; ++e) {
      expert_prob[static_cast<std::size_t>(e)] += (*gates.data)[static_cast<std::size_t>(n) * n_experts + e];
    }
  }

  for (int e = 0; e < n_experts; ++e) {
    expert_fraction[static_cast<std::size_t>(e)] /= static_cast<float>(N);
    expert_prob[static_cast<std::size_t>(e)] /= static_cast<float>(N);
  }

  // 3) Expert computation
  Tensor y = Tensor::zeros({N, C}, false);
  for (int e = 0; e < n_experts; ++e) {
    std::vector<int> routed_tokens;
    std::vector<float> routed_gates;
    for (int n = 0; n < N; ++n) {
      const float g = topk_gates[static_cast<std::size_t>(n) * n_experts + e];
      if (g > 0.0f) {
        routed_tokens.push_back(n);
        routed_gates.push_back(g);
      }
    }

    if (routed_tokens.empty()) continue;

    const Tensor* w_fc  = expert_params[static_cast<std::size_t>(e) * 4 + 0];
    const Tensor* b_fc  = expert_params[static_cast<std::size_t>(e) * 4 + 1];
    const Tensor* w_out = expert_params[static_cast<std::size_t>(e) * 4 + 2];
    const Tensor* b_out = expert_params[static_cast<std::size_t>(e) * 4 + 3];

    const int R = static_cast<int>(routed_tokens.size());

    // Gather tokens for this expert
    std::vector<float> xe_data(static_cast<std::size_t>(R) * C);
    for (int r = 0; r < R; ++r) {
      const int n = routed_tokens[static_cast<std::size_t>(r)];
      for (int c = 0; c < C; ++c) {
        xe_data[static_cast<std::size_t>(r) * C + c] =
            (*x_2d.data)[static_cast<std::size_t>(n) * C + c];
      }
    }

    Tensor xe = Tensor::zeros({R, C}, false);
    *xe.data = xe_data;

    // Expert FFN: GELU(xe @ w_fc + b_fc) @ w_out + b_out
    Tensor h_raw = nn::matmul2d(xe, *w_fc); // [R, interm_dim]
    Tensor gelu_out = Tensor::zeros({R, interm_dim}, false);
    for (int i = 0; i < R * interm_dim; ++i) {
      const float v = (*h_raw.data)[static_cast<std::size_t>(i)] + (*b_fc->data)[static_cast<std::size_t>(i % interm_dim)];
      const float c = 0.044715f;
      const float s = 0.7978845608f;
      const float u = s * (v + c * v * v * v);
      const float t = std::tanh(u);
      (*gelu_out.data)[static_cast<std::size_t>(i)] = 0.5f * v * (1.0f + t);
    }

    Tensor proj = nn::matmul2d(gelu_out, *w_out); // [R, C]

    // Add bias and accumulate into output
    for (int r = 0; r < R; ++r) {
      const int n = routed_tokens[static_cast<std::size_t>(r)];
      const float gate = routed_gates[static_cast<std::size_t>(r)];
      for (int c = 0; c < C; ++c) {
        const float val = (*proj.data)[static_cast<std::size_t>(r) * C + c] + (*b_out->data)[static_cast<std::size_t>(c)];
        (*y.data)[static_cast<std::size_t>(n) * C + c] += gate * val;
      }
    }
  }

  // 4) Load balancing loss
  float balance = 0.0f;
  for (int e = 0; e < n_experts; ++e) {
    balance += expert_fraction[static_cast<std::size_t>(e)] * expert_prob[static_cast<std::size_t>(e)];
  }
  balance *= static_cast<float>(n_experts);

  Tensor balance_loss = Tensor::zeros({1}, false);
  (*balance_loss.data)[0] = balance;

  return {y, balance_loss};
}

} // namespace nn::variants::moe
