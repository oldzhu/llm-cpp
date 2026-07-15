#include "variants/grpo/grpo_trainer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nn::variants::grpo {

std::vector<float> compute_grpo_advantages(const std::vector<float>& scores,
                                            int group_size) {
  // For each group of K consecutive scores, normalize to get advantages.
  // A_i = (R_i - mean(R_group)) / (std(R_group) + 1e-8)
  const int total = static_cast<int>(scores.size());
  if (total % group_size != 0) {
    throw std::runtime_error("GRPO: scores size must be divisible by group_size");
  }

  std::vector<float> advantages(total, 0.0f);
  const int num_groups = total / group_size;

  for (int g = 0; g < num_groups; ++g) {
    const int start = g * group_size;
    const int end = start + group_size;

    // Compute group mean
    float sum = 0.0f;
    for (int i = start; i < end; ++i) sum += scores[static_cast<std::size_t>(i)];
    float mean = sum / static_cast<float>(group_size);

    // Compute group std
    float var = 0.0f;
    for (int i = start; i < end; ++i) {
      float d = scores[static_cast<std::size_t>(i)] - mean;
      var += d * d;
    }
    float stddev = std::sqrt(var / static_cast<float>(group_size) + 1e-8f);

    // Normalize
    for (int i = start; i < end; ++i) {
      advantages[static_cast<std::size_t>(i)] =
          (scores[static_cast<std::size_t>(i)] - mean) / stddev;
    }
  }

  return advantages;
}

float grpo_surrogate_loss(float log_prob_new, float log_prob_old,
                          float advantage, float clip_eps) {
  // Same clipped surrogate formula as PPO
  float ratio = std::exp(log_prob_new - log_prob_old);
  float clipped = std::max(std::min(ratio, 1.0f + clip_eps), 1.0f - clip_eps);
  return -std::min(ratio * advantage, clipped * advantage);
}

} // namespace nn::variants::grpo
