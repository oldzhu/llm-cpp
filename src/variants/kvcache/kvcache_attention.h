#pragma once

#include <cstdint>
#include <vector>

#include "tensor.h"

namespace model { class TinyGPT; }

namespace nn::variants::kvcache {

struct KVCache {
  int B;
  int T_max;
  int C;
  int cur_len = 0;

  nn::Tensor k_cache_; // [B, T_max, C]
  nn::Tensor v_cache_; // [B, T_max, C]

  KVCache(int B, int T_max, int C);
  void reset();
};

// Prefill: full forward pass on [B,T,C] context, fills cache.
// Returns attention output [B,T,C].
// Requires T <= cache.T_max.
nn::Tensor self_attention_prefill(const nn::Tensor& x,      // [B,T,C]
                                  const nn::Tensor& w_qkv,  // [C,3C]
                                  const nn::Tensor& b_qkv,  // [3C]
                                  const nn::Tensor& w_proj, // [C,C]
                                  const nn::Tensor& b_proj, // [C]
                                  KVCache& cache);

// Incremental step: processes one new token [B,1,C] using the cache.
// Returns attention output [B,1,C].
// No causal mask needed (query is at the end of the sequence).
// Requires cache.cur_len < cache.T_max before calling.
nn::Tensor self_attention_step(const nn::Tensor& x_step,   // [B,1,C]
                               const nn::Tensor& w_qkv,    // [C,3C]
                               const nn::Tensor& b_qkv,    // [3C]
                               const nn::Tensor& w_proj,   // [C,C]
                               const nn::Tensor& b_proj,   // [C]
                               KVCache& cache);

// Full model forward with KV-cache (generation-only, no autograd).
// Uses the prefill attention path for the initial context.
// Returns logits [1, T, V].
nn::Tensor model_prefill(const class model::TinyGPT& gpt,
                          const std::vector<std::int32_t>& tokens,
                          int B,
                          int T,
                          std::vector<KVCache>& layer_caches);

// Single-step model forward with KV-cache (generation-only, no autograd).
// Returns logits [1, 1, V] for next-token distribution.
nn::Tensor model_step(const class model::TinyGPT& gpt,
                       std::int32_t next_token,
                       int B,
                       int position,
                       std::vector<KVCache>& layer_caches);

} // namespace nn::variants::kvcache
