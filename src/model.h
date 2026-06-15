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
  int norm_type = 0; // 0 = LayerNorm (default), 1 = RMSNorm
  int mlp_type = 0;  // 0 = GELU (default), 1 = SwiGLU, 2 = MoE
  int swiglu_interm = 0; // intermediate dim for SwiGLU (0 = 3*d_model)
  int n_experts = 4; // number of experts for MoE (mlp_type==2)
  int top_k = 2;     // top-K experts per token for MoE
  int attn_type = 0; // 0 = 1-head (default), 1 = MHA, 2 = GQA, 3 = MLA
  int pos_type = 0;  // 0 = wpe (default), 1 = RoPE
  int attn_n_heads = 1;  // number of attention heads (for attn_type>=1)
  int attn_n_kv = 1;     // number of KV heads (for attn_type==2, GQA)
  int mla_latent_dim = 0; // latent dim for MLA (attn_type==3, 0 = C/4)
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

    nn::Tensor ln_attn_gamma; // [C] — LayerNorm before attention
    nn::Tensor ln_attn_beta;  // [C]
    nn::Tensor ln_mlp_gamma;  // [C] — LayerNorm before MLP
    nn::Tensor ln_mlp_beta;   // [C]

    // SwiGLU parameters (used when cfg.mlp_type == 1)
    nn::Tensor swiglu_gate;   // [C, interm]
    nn::Tensor swiglu_gate_b; // [interm]
    nn::Tensor swiglu_up;     // [C, interm]
    nn::Tensor swiglu_up_b;   // [interm]
    nn::Tensor swiglu_down;   // [interm, C]
    nn::Tensor swiglu_down_b; // [C]

    // MoE parameters (used when cfg.mlp_type == 2)
    nn::Tensor moe_router_w; // [C, n_experts]
    nn::Tensor moe_router_b; // [n_experts]
    // Expert weights stored as flat vectors of Tensors (n_experts * 4)
    std::vector<nn::Tensor> moe_expert_wfc;
    std::vector<nn::Tensor> moe_expert_bfc;
    std::vector<nn::Tensor> moe_expert_wout;
    std::vector<nn::Tensor> moe_expert_bout;
    // MLA parameters (used when cfg.attn_type == 3)
    nn::Tensor mla_w_q;   // [C, C]
    nn::Tensor mla_b_q;   // [C]
    nn::Tensor mla_w_dkv; // [C, L]
    nn::Tensor mla_b_dkv; // [L]
    nn::Tensor mla_w_uk;  // [L, C]
    nn::Tensor mla_w_uv;  // [L, C]
    nn::Tensor mla_w_o;   // [C, C]
    nn::Tensor mla_b_o;   // [C]
  };

  std::vector<Block> blocks_;

  // Final LN before LM head
  nn::Tensor ln_final_gamma_; // [C]
  nn::Tensor ln_final_beta_;  // [C]

  // MoE auxiliary loss (populated by forward_logits when mlp_type==2)
  float moe_balance_loss_ = 0.0f;

  // Final LM head
  nn::Tensor w_lm_; // [C,V]
  nn::Tensor b_lm_; // [V]

  nn::Tensor add_positional(const nn::Tensor& x, int B, int T);
};

} // namespace model
