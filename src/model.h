#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tensor.h"

namespace model {

struct Config {
  int vocab_size = 256;
  int seq_len = 64;
  int d_model = 64;
  int n_layers = 1;
};

struct Params {
  std::vector<nn::Tensor*> tensors;
};

struct ParamsConst {
  std::vector<const nn::Tensor*> tensors;
};

class TinyGPT {
 public:
  explicit TinyGPT(const Config& cfg, std::uint64_t seed = 1);

  const Config& cfg() const { return cfg_; }

  nn::Tensor forward_logits(const std::vector<std::int32_t>& tokens_bt, int B, int T);

  // Convenience: compute loss for next-token prediction.
  nn::Tensor loss(const std::vector<std::int32_t>& tokens_bt, const std::vector<std::int32_t>& targets_bt, int B, int T);

  void zero_grad();

  Params parameters();
  ParamsConst parameters_const() const;

 private:
  Config cfg_;

  // Embeddings
  nn::Tensor wte_; // [V,C]
  nn::Tensor wpe_; // [T,C]

  // One or more transformer blocks (1-head attention, MLP)
  struct Block {
    nn::Tensor w_qkv;  // [C,3C]
    nn::Tensor b_qkv;  // [3C]
    nn::Tensor w_proj; // [C,C]
    nn::Tensor b_proj; // [C]

    nn::Tensor w_fc;   // [C,4C]
    nn::Tensor b_fc;   // [4C]
    nn::Tensor w_out;  // [4C,C]
    nn::Tensor b_out;  // [C]
  };

  std::vector<Block> blocks_;

  // Final LM head
  nn::Tensor w_lm_; // [C,V]
  nn::Tensor b_lm_; // [V]

  nn::Tensor add_positional(const nn::Tensor& x, int B, int T);
};

} // namespace model
