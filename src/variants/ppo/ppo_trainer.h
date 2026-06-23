#pragma once

#include <vector>
#include "tensor.h"
#include "optim.h"

namespace nn::variants::ppo {

// PPO training — RL stage for LLM alignment.
//
// ValueHead: predicts expected future return V(s) for each token.
// GAE: Generalized Advantage Estimation for stable advantage computation.
// Clipped surrogate: PPO's core objective — limits policy update size.
//
// This is a teaching-first, explicitly-looped implementation.

struct PPOConfig {
  float clip_eps = 0.2f;   // PPO clipping epsilon
  float gamma = 0.99f;     // discount factor for returns
  float lambda = 0.95f;    // GAE lambda for advantage smoothing
  float vf_coef = 0.5f;    // value function loss coefficient
  int ppo_steps = 4;       // number of PPO update steps per batch
};

// Value head: linear projection from hidden state to scalar value
//   V(s) = hidden @ w_value + b_value
Tensor value_forward(const Tensor& hidden_2d, const Tensor& w_value, const Tensor& b_value);

// Compute GAE advantages from rewards and values
//   A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}  where δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
std::vector<float> compute_gae(const std::vector<float>& rewards,
                                const std::vector<float>& values,
                                float gamma, float lambda);

// Clipped surrogate PPO loss for policy
//   L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
float clip_surrogate_loss(float log_prob_new, float log_prob_old,
                          float advantage, float clip_eps);

// Mock reward: uses model's cross-entropy loss as reward signal.
// Lower LM loss → better generation → higher reward.
//   reward[t] = -loss_per_position (simplified: scalar reward)
std::vector<float> mock_reward_per_token(float loss_value, int num_tokens);

// PPO training loop: runs one PPO iteration (rollout + advantage + update).
// Returns total PPO loss for monitoring.
//   hidden: [N,C] hidden states from model forward
//   vw, vb: value head weights
//   advantages: pre-computed GAE advantages
//   returns: pre-computed returns (values + advantages)
//   optimizer: AdamW optimizer for value head
float ppo_update_value_head(const Tensor& hidden_2d,
                             const std::vector<float>& returns,
                             Tensor& vw, Tensor& vb,
                             float vf_coef,
                             optim::AdamW& optimizer,
                             const std::vector<nn::Tensor*>& params);

} // namespace nn::variants::ppo
