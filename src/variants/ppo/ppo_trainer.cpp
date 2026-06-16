#include "variants/ppo/ppo_trainer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ops.h"

namespace nn::variants::ppo {

Tensor value_forward(const Tensor& hidden_2d, const Tensor& w_value, const Tensor& b_value) {
  // V(s) = hidden @ w_value + b_value
  // hidden: [N, C], w_value: [C, 1], b_value: [1]
  if (hidden_2d.shape.size() != 2) throw std::runtime_error("value_forward: hidden must be [N,C]");
  const int N = hidden_2d.shape[0];
  const int C = hidden_2d.shape[1];
  if (w_value.shape != std::vector<int>({C, 1})) throw std::runtime_error("value_forward: w_value must be [C,1]");
  if (b_value.shape != std::vector<int>({1})) throw std::runtime_error("value_forward: b_value must be [1]");

  Tensor raw = nn::matmul2d(hidden_2d, w_value); // [N, 1]
  Tensor values = Tensor::zeros({N}, false);
  for (int n = 0; n < N; ++n) {
    (*values.data)[static_cast<std::size_t>(n)] = (*raw.data)[static_cast<std::size_t>(n)] + (*b_value.data)[0];
  }
  return values;
}

std::vector<float> compute_gae(const std::vector<float>& rewards,
                                const std::vector<float>& values,
                                float gamma, float lambda) {
  // GAE: A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
  // where δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
  const int T = static_cast<int>(rewards.size());
  std::vector<float> advantages(T, 0.0f);
  float gae = 0.0f;
  for (int t = T - 1; t >= 0; --t) {
    float delta = rewards[static_cast<std::size_t>(t)] - values[static_cast<std::size_t>(t)];
    if (t < T - 1) {
      delta += gamma * values[static_cast<std::size_t>(t + 1)];
    }
    gae = delta + gamma * lambda * gae;
    advantages[static_cast<std::size_t>(t)] = gae;
  }
  return advantages;
}

float clip_surrogate_loss(float log_prob_new, float log_prob_old,
                          float advantage, float clip_eps) {
  float ratio = std::exp(log_prob_new - log_prob_old);
  float clipped = std::max(std::min(ratio, 1.0f + clip_eps), 1.0f - clip_eps);
  return -std::min(ratio * advantage, clipped * advantage);
}

} // namespace nn::variants::ppo
