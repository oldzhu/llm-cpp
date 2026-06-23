#include "variants/ppo/ppo_trainer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ops.h"
#include "optim.h"

namespace nn::variants::ppo {

Tensor value_forward(const Tensor& hidden_2d, const Tensor& w_value, const Tensor& b_value) {
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

std::vector<float> mock_reward_per_token(float loss_value, int num_tokens) {
  // Simple reward: lower loss = higher reward
  // reward = -loss (so that decreasing loss → increasing reward)
  std::vector<float> rewards(static_cast<std::size_t>(num_tokens), -loss_value);
  return rewards;
}

float ppo_update_value_head(const Tensor& hidden_2d,
                             const std::vector<float>& returns,
                             Tensor& vw, Tensor& vb,
                             float vf_coef,
                             optim::AdamW& optimizer,
                             const std::vector<nn::Tensor*>& params) {
  // Compute value predictions
  Tensor values = value_forward(hidden_2d, vw, vb);
  int N = static_cast<int>(returns.size());

  // MSE loss: L = mean((V(s) - R)^2)
  float mse_sum = 0.0f;
  for (int n = 0; n < N; ++n) {
    float diff = (*values.data)[static_cast<std::size_t>(n)] - returns[static_cast<std::size_t>(n)];
    mse_sum += diff * diff;
  }
  float mse = mse_sum / static_cast<float>(N);

  // Manual backward for MSE: dL/dV = 2*(V-R)/N
  std::vector<float> dvalues(N, 0.0f);
  for (int n = 0; n < N; ++n) {
    dvalues[static_cast<std::size_t>(n)] = 2.0f * ((*values.data)[static_cast<std::size_t>(n)] - returns[static_cast<std::size_t>(n)]) / static_cast<float>(N);
  }

  // Backprop through value head: dL/dw = hidden^T @ dvalues
  // dL/dw: [C, 1], dL/db: [1]
  int C = hidden_2d.shape[1];
  std::vector<float> dw(C, 0.0f);
  float db = 0.0f;
  for (int n = 0; n < N; ++n) {
    float dv = dvalues[static_cast<std::size_t>(n)];
    for (int c = 0; c < C; ++c) {
      dw[static_cast<std::size_t>(c)] += (*hidden_2d.data)[static_cast<std::size_t>(n) * C + c] * dv;
    }
    db += dv;
  }

  // Accumulate gradients (zero first)
  for (std::size_t i = 0; i < static_cast<std::size_t>(C); ++i) (*vw.grad)[i] = dw[i];
  (*vb.grad)[0] = db;

  // Apply optimizer step
  optimizer.step(params);

  return mse * vf_coef;
}

} // namespace nn::variants::ppo
