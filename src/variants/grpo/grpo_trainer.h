#pragma once

#include <vector>

namespace nn::variants::grpo {

// GRPO — Group Relative Policy Optimization (DeepSeek-R1 style).
//
// Unlike PPO which uses a learned value function for advantage estimation,
// GRPO generates multiple outputs per prompt, scores them, and computes
// advantages as normalized relative scores within each group.
//
// This eliminates the need for a separate value model and reward model,
// making the RL training pipeline simpler.

struct GRPOConfig {
  int group_size = 4;      // K: number of outputs per prompt
  float clip_eps = 0.2f;   // clipping epsilon for surrogate loss
  float kl_beta = 0.01f;   // KL penalty coefficient (optional)
  int grpo_steps = 4;      // update steps per iteration
};

// Compute relative advantages within a group.
// For each group of K scores, normalize: A_i = (R_i - mean(R)) / (std(R) + 1e-8)
// Returns advantages of same length as input.
std::vector<float> compute_grpo_advantages(const std::vector<float>& scores,
                                            int group_size);

// GRPO clipped surrogate loss (same formula as PPO).
// L = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
// where ratio = exp(log_prob_new - log_prob_old)
float grpo_surrogate_loss(float log_prob_new, float log_prob_old,
                          float advantage, float clip_eps);

} // namespace nn::variants::grpo
