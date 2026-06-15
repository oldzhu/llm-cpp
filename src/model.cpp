#include "model.h"

#include <cmath>
#include <stdexcept>

#include "ops.h"
#include "variants/moe/moe_mlp.h"
#include "variants/mha/mha_attention.h"
#include "variants/gqa/gqa_attention.h"
#include "variants/rope/rope_attention.h"
#include "variants/mla/mla_attention.h"

namespace model {

using nn::Tensor;

static float init_std(int fan_in) {
  return 1.0f / std::sqrt(static_cast<float>(fan_in));
}

TinyGPT::TinyGPT(const Config& cfg, std::uint64_t seed) : cfg_(cfg) {
  if (cfg_.vocab_size <= 0) throw std::runtime_error("vocab_size must be > 0");
  if (cfg_.seq_len <= 0) throw std::runtime_error("seq_len must be > 0");
  if (cfg_.d_model <= 0) throw std::runtime_error("d_model must be > 0");
  if (cfg_.n_layers <= 0) throw std::runtime_error("n_layers must be > 0");

  const int V = cfg_.vocab_size;
  const int T = cfg_.seq_len;
  const int C = cfg_.d_model;

  wte_ = Tensor::randn({V, C}, init_std(C), seed ^ 0xA11CEULL, true);
  wpe_ = Tensor::randn({T, C}, init_std(C), seed ^ 0xBEEFULL, true);

  blocks_.resize(static_cast<std::size_t>(cfg_.n_layers));
  for (int i = 0; i < cfg_.n_layers; ++i) {
    Block blk;
    const std::uint64_t s = seed ^ (0x1000ULL + static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL);
    blk.w_qkv = Tensor::randn({C, 3 * C}, init_std(C), s ^ 1, true);
    blk.b_qkv = Tensor::zeros({3 * C}, true);
    blk.w_proj = Tensor::randn({C, C}, init_std(C), s ^ 2, true);
    blk.b_proj = Tensor::zeros({C}, true);

    blk.w_fc = Tensor::randn({C, 4 * C}, init_std(C), s ^ 3, true);
    blk.b_fc = Tensor::zeros({4 * C}, true);
    blk.w_out = Tensor::randn({4 * C, C}, init_std(4 * C), s ^ 4, true);
    blk.b_out = Tensor::zeros({C}, true);

    blk.ln_attn_gamma = Tensor::zeros({C}, true);
    blk.ln_attn_beta  = Tensor::zeros({C}, true);
    blk.ln_mlp_gamma  = Tensor::zeros({C}, true);
    blk.ln_mlp_beta   = Tensor::zeros({C}, true);
    for (int ci = 0; ci < C; ++ci) {
      (*blk.ln_attn_gamma.data)[ci] = 1.0f;
      (*blk.ln_mlp_gamma.data)[ci] = 1.0f;
    }

    const int interm = (cfg_.swiglu_interm > 0) ? cfg_.swiglu_interm : (3 * C);
    blk.swiglu_gate   = Tensor::randn({C, interm}, init_std(C), s ^ 5, true);
    blk.swiglu_gate_b = Tensor::zeros({interm}, true);
    blk.swiglu_up     = Tensor::randn({C, interm}, init_std(C), s ^ 6, true);
    blk.swiglu_up_b   = Tensor::zeros({interm}, true);
    blk.swiglu_down   = Tensor::randn({interm, C}, init_std(interm), s ^ 7, true);
    blk.swiglu_down_b = Tensor::zeros({C}, true);

    // MoE router + expert params
    const int n_exp = cfg_.n_experts;
    blk.moe_router_w = Tensor::randn({C, n_exp}, init_std(C), s ^ 8, true);
    blk.moe_router_b = Tensor::zeros({n_exp}, true);
    blk.moe_expert_wfc.resize(static_cast<std::size_t>(n_exp));
    blk.moe_expert_bfc.resize(static_cast<std::size_t>(n_exp));
    blk.moe_expert_wout.resize(static_cast<std::size_t>(n_exp));
    blk.moe_expert_bout.resize(static_cast<std::size_t>(n_exp));
    for (int e = 0; e < n_exp; ++e) {
      blk.moe_expert_wfc[static_cast<std::size_t>(e)]  = Tensor::randn({C, 4 * C}, init_std(C), s ^ (9 + e * 4), true);
      blk.moe_expert_bfc[static_cast<std::size_t>(e)]  = Tensor::zeros({4 * C}, true);
      blk.moe_expert_wout[static_cast<std::size_t>(e)] = Tensor::randn({4 * C, C}, init_std(4 * C), s ^ (9 + e * 4 + 1), true);
      blk.moe_expert_bout[static_cast<std::size_t>(e)] = Tensor::zeros({C}, true);
    }
    // Shared experts
    for (int e = 0; e < cfg_.n_shared; ++e) {
      blk.moe_shared_wfc.emplace_back(Tensor::randn({C, 4*C}, init_std(C), s ^ (100 + e*4), true));
      blk.moe_shared_bfc.emplace_back(Tensor::zeros({4*C}, true));
      blk.moe_shared_wout.emplace_back(Tensor::randn({4*C, C}, init_std(4*C), s ^ (100 + e*4 + 1), true));
      blk.moe_shared_bout.emplace_back(Tensor::zeros({C}, true));
    }

    // MLA params
    const int mla_L = (cfg_.mla_latent_dim > 0) ? cfg_.mla_latent_dim : (C / 4);
    blk.mla_w_q   = Tensor::randn({C, C}, init_std(C), s ^ 10, true);
    blk.mla_b_q   = Tensor::zeros({C}, true);
    blk.mla_w_dkv = Tensor::randn({C, mla_L}, init_std(C), s ^ 11, true);
    blk.mla_b_dkv = Tensor::zeros({mla_L}, true);
    blk.mla_w_uk  = Tensor::randn({mla_L, C}, init_std(mla_L), s ^ 12, true);
    blk.mla_w_uv  = Tensor::randn({mla_L, C}, init_std(mla_L), s ^ 13, true);
    blk.mla_w_o   = Tensor::randn({C, C}, init_std(C), s ^ 14, true);
    blk.mla_b_o   = Tensor::zeros({C}, true);

    blocks_[static_cast<std::size_t>(i)] = std::move(blk);
  }

  w_lm_ = Tensor::randn({C, V}, init_std(C), seed ^ 0xC0FFEEULL, true);
  b_lm_ = Tensor::zeros({V}, true);

  ln_final_gamma_ = Tensor::zeros({C}, true);
  ln_final_beta_  = Tensor::zeros({C}, true);
  for (int ci = 0; ci < C; ++ci) {
    (*ln_final_gamma_.data)[ci] = 1.0f;
  }
}

Tensor TinyGPT::add_positional(const Tensor& x, int B, int T) {
  const int C = cfg_.d_model;
  if (wpe_.shape != std::vector<int>({cfg_.seq_len, C})) {
    throw std::runtime_error("wpe shape mismatch");
  }
  if (x.shape != std::vector<int>({B, T, C})) throw std::runtime_error("add_positional: x shape mismatch");

  Tensor out = Tensor::zeros({B, T, C}, x.requires_grad || wpe_.requires_grad);
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      const std::size_t o_off = (static_cast<std::size_t>(b) * T + t) * C;
      const std::size_t p_off = static_cast<std::size_t>(t) * C;
      for (int c = 0; c < C; ++c) {
        (*out.data)[o_off + c] = (*x.data)[o_off + c] + (*wpe_.data)[p_off + c];
      }
    }
  }

  if (out.requires_grad) {
    out.node = std::make_shared<nn::Node>();
    out.node->parents = {x, wpe_};
    out.node->backward = [B, T, C](Tensor& o) {
      Tensor& px = o.node->parents[0];
      Tensor& pwpe = o.node->parents[1];
      const std::size_t n = o.numel();
      if (px.requires_grad) {
        for (std::size_t i = 0; i < n; ++i) (*px.grad)[i] += (*o.grad)[i];
      }
      if (pwpe.requires_grad) {
        for (int t = 0; t < T; ++t) {
          const std::size_t p_off = static_cast<std::size_t>(t) * C;
          for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int b = 0; b < B; ++b) {
              const std::size_t o_off = (static_cast<std::size_t>(b) * T + t) * C;
              sum += (*o.grad)[o_off + c];
            }
            (*pwpe.grad)[p_off + c] += sum;
          }
        }
      }
    };
  }

  return out;
}

Tensor TinyGPT::forward_logits(const std::vector<std::int32_t>& tokens_bt, int B, int T) {
  const int V = cfg_.vocab_size;
  const int C = cfg_.d_model;
  if (T > cfg_.seq_len) throw std::runtime_error("forward_logits: T exceeds configured seq_len");
  if (static_cast<int>(tokens_bt.size()) != B * T) throw std::runtime_error("forward_logits: tokens size mismatch");
  if (wte_.shape != std::vector<int>({V, C})) throw std::runtime_error("wte shape mismatch");

  moe_balance_loss_ = 0.0f;

  // === Embedding stage ===
  // Token ids -> vectors: X = Wte[tokens] + Wpe[pos] (if pos_type == 0)
  Tensor x = nn::embedding(wte_, tokens_bt, B, T); // [B,T,C]
  if (cfg_.pos_type == 0) x = add_positional(x, B, T);

  for (int li = 0; li < cfg_.n_layers; ++li) {
    Block& blk = blocks_[static_cast<std::size_t>(li)];

    // === Transformer block (pre-norm) ===
    // Attention sublayer:
    Tensor h;
    if (cfg_.norm_type == 0)
      h = nn::layernorm_affine(x, blk.ln_attn_gamma, blk.ln_attn_beta, 1e-5f);
    else
      h = nn::rmsnorm_affine(x, blk.ln_attn_gamma, 1e-5f);
    Tensor a;
    if (cfg_.attn_type == 0 && cfg_.pos_type == 1) {
      // 1-head + RoPE: use RoPE variant (applies rotation, no wpe needed)
      a = nn::variants::rope::self_attention_rope(h, blk.w_qkv, blk.b_qkv, blk.w_proj, blk.b_proj);
    } else if (cfg_.attn_type == 0) {
      // 1-head attention (default)
      a = nn::self_attention_1h(h, blk.w_qkv, blk.b_qkv, blk.w_proj, blk.b_proj);
    } else if (cfg_.attn_type == 1) {
      // Multi-head attention
      if (cfg_.d_model % cfg_.attn_n_heads != 0) throw std::runtime_error("MHA: d_model must be divisible by n_heads");
      a = nn::variants::mha::self_attention_mha(h, blk.w_qkv, blk.b_qkv, blk.w_proj, blk.b_proj, cfg_.attn_n_heads);
    } else if (cfg_.attn_type == 2) {
      // GQA — slice K/V to n_kv*hd columns, reshape to 4D, call variant
      int n_heads = cfg_.attn_n_heads;
      int n_kv = (cfg_.attn_n_kv > 0 ? cfg_.attn_n_kv : 1);
      int hd = cfg_.d_model / n_heads;
      int kv_dim = n_kv * hd;
      if (cfg_.d_model % n_heads != 0) throw std::runtime_error("GQA: d_model must be divisible by n_heads");
      if (n_heads % n_kv != 0) throw std::runtime_error("GQA: n_heads must be divisible by n_kv");

      Tensor qkv = nn::linear_lastdim(h, blk.w_qkv, blk.b_qkv); // [B,T,3C]
      int B = h.shape[0], T = h.shape[1], C3 = qkv.shape[2];
      int C = cfg_.d_model;

      // Extract Q (full C cols), K/V (first kv_dim cols each)
      auto slice = [&](int offset, int len) {
        nn::Tensor out = nn::Tensor::zeros({B, T, len}, false);
        for (int i = 0; i < B*T; ++i)
          for (int j = 0; j < len; ++j)
            (*out.data)[i*len + j] = (*qkv.data)[i*C3 + offset + j];
        return out;
      };
      Tensor q = slice(0, C);
      Tensor k = slice(C, kv_dim);
      Tensor v = slice(2*C, kv_dim);

      nn::Tensor q4 = nn::reshape(q, {B, T, n_heads, hd});
      nn::Tensor k4 = nn::reshape(k, {B, T, n_kv, hd});
      nn::Tensor v4 = nn::reshape(v, {B, T, n_kv, hd});

      nn::Tensor a_gqa = nn::variants::gqa::self_attention_gqa(q4, k4, v4, n_heads, n_kv);
      a = nn::linear_lastdim(a_gqa, blk.w_proj, blk.b_proj);
    } else if (cfg_.attn_type == 3) {
      a = nn::variants::mla::self_attention_mla(h,
           blk.mla_w_q, blk.mla_b_q, blk.mla_w_dkv, blk.mla_b_dkv,
           blk.mla_w_uk, blk.mla_w_uv, blk.mla_w_o, blk.mla_b_o);
    }
    x = nn::add(x, a);

    // MLP sublayer:
    Tensor m;
    if (cfg_.norm_type == 0)
      m = nn::layernorm_affine(x, blk.ln_mlp_gamma, blk.ln_mlp_beta, 1e-5f);
    else
      m = nn::rmsnorm_affine(x, blk.ln_mlp_gamma, 1e-5f);

    Tensor ff;
    if (cfg_.mlp_type == 0) {
      ff = nn::linear_lastdim(m, blk.w_fc, blk.b_fc);
      ff = nn::gelu(ff);
      ff = nn::linear_lastdim(ff, blk.w_out, blk.b_out);
    } else if (cfg_.mlp_type == 1) {
      Tensor gate = nn::linear_lastdim(m, blk.swiglu_gate, blk.swiglu_gate_b);
      Tensor up   = nn::linear_lastdim(m, blk.swiglu_up, blk.swiglu_up_b);
      gate = nn::silu(gate);
      ff = nn::mul(gate, up);
      ff = nn::linear_lastdim(ff, blk.swiglu_down, blk.swiglu_down_b);
    } else {
      // MoE MLP
      const int N = B * T;
      Tensor x2 = nn::reshape(m, {N, C});
      std::vector<const Tensor*> expert_ptrs;
      expert_ptrs.reserve(static_cast<std::size_t>(cfg_.n_experts) * 4);
      for (int e = 0; e < cfg_.n_experts; ++e) {
        expert_ptrs.push_back(&blk.moe_expert_wfc[static_cast<std::size_t>(e)]);
        expert_ptrs.push_back(&blk.moe_expert_bfc[static_cast<std::size_t>(e)]);
        expert_ptrs.push_back(&blk.moe_expert_wout[static_cast<std::size_t>(e)]);
        expert_ptrs.push_back(&blk.moe_expert_bout[static_cast<std::size_t>(e)]);
      }
      std::vector<const Tensor*> shared_ptrs;
      for (int e = 0; e < cfg_.n_shared; ++e) {
        shared_ptrs.push_back(&blk.moe_shared_wfc[static_cast<std::size_t>(e)]);
        shared_ptrs.push_back(&blk.moe_shared_bfc[static_cast<std::size_t>(e)]);
        shared_ptrs.push_back(&blk.moe_shared_wout[static_cast<std::size_t>(e)]);
        shared_ptrs.push_back(&blk.moe_shared_bout[static_cast<std::size_t>(e)]);
      }
      auto moe_out = nn::variants::moe::moe_mlp_forward(x2, blk.moe_router_w, blk.moe_router_b,
                                                           expert_ptrs, cfg_.n_experts, cfg_.top_k, 4 * C, shared_ptrs);
      moe_balance_loss_ += (*moe_out.balance_loss.data)[0];
      ff = nn::reshape(moe_out.y, {B, T, C});
    }
    x = nn::add(x, ff);
  }

  // Final norm + LM head:
  Tensor xn;
  if (cfg_.norm_type == 0)
    xn = nn::layernorm_affine(x, ln_final_gamma_, ln_final_beta_, 1e-5f);
  else
    xn = nn::rmsnorm_affine(x, ln_final_gamma_, 1e-5f);
  Tensor logits = nn::linear_lastdim(xn, w_lm_, b_lm_); // [B,T,V]
  return logits;
}

Tensor TinyGPT::loss(const std::vector<std::int32_t>& tokens_bt,
                     const std::vector<std::int32_t>& targets_bt,
                     int B,
                     int T) {
  Tensor logits = forward_logits(tokens_bt, B, T);
  Tensor logits2 = nn::reshape(logits, {B * T, cfg_.vocab_size});
  Tensor ce_loss = nn::cross_entropy(logits2, targets_bt);
  if (cfg_.mlp_type == 2 && moe_balance_loss_ > 0.0f) {
    // Add auxiliary MoE load balancing loss (scaled)
    nn::Tensor total = nn::add_scalar(ce_loss, moe_balance_loss_ * 0.01f);
    return total;
  }
  return ce_loss;
}

void TinyGPT::zero_grad() {
  wte_.zero_grad();
  wpe_.zero_grad();
  for (auto& blk : blocks_) {
    blk.w_qkv.zero_grad();
    blk.b_qkv.zero_grad();
    blk.w_proj.zero_grad();
    blk.b_proj.zero_grad();
    blk.w_fc.zero_grad();
    blk.b_fc.zero_grad();
    blk.w_out.zero_grad();
    blk.b_out.zero_grad();
    blk.ln_attn_gamma.zero_grad();
    blk.ln_attn_beta.zero_grad();
    blk.ln_mlp_gamma.zero_grad();
    blk.ln_mlp_beta.zero_grad();
    blk.swiglu_gate.zero_grad();
    blk.swiglu_gate_b.zero_grad();
    blk.swiglu_up.zero_grad();
    blk.swiglu_up_b.zero_grad();
    blk.swiglu_down.zero_grad();
    blk.swiglu_down_b.zero_grad();
    blk.mla_w_q.zero_grad(); blk.mla_b_q.zero_grad();
    blk.mla_w_dkv.zero_grad(); blk.mla_b_dkv.zero_grad();
    blk.mla_w_uk.zero_grad(); blk.mla_w_uv.zero_grad();
    blk.mla_w_o.zero_grad(); blk.mla_b_o.zero_grad();
    blk.moe_router_w.zero_grad();
    blk.moe_router_b.zero_grad();
    for (auto& t : blk.moe_expert_wfc)  t.zero_grad();
    for (auto& t : blk.moe_expert_bfc)  t.zero_grad();
    for (auto& t : blk.moe_expert_wout) t.zero_grad();
    for (auto& t : blk.moe_expert_bout) t.zero_grad();
  }
  w_lm_.zero_grad();
  b_lm_.zero_grad();
  ln_final_gamma_.zero_grad();
  ln_final_beta_.zero_grad();
}

Params TinyGPT::parameters() {
  Params p;
  p.tensors.push_back(&wte_);
  p.tensors.push_back(&wpe_);
  for (auto& blk : blocks_) {
    p.tensors.push_back(&blk.w_qkv);
    p.tensors.push_back(&blk.b_qkv);
    p.tensors.push_back(&blk.w_proj);
    p.tensors.push_back(&blk.b_proj);
    p.tensors.push_back(&blk.w_fc);
    p.tensors.push_back(&blk.b_fc);
    p.tensors.push_back(&blk.w_out);
    p.tensors.push_back(&blk.b_out);
    p.tensors.push_back(&blk.ln_attn_gamma);
    p.tensors.push_back(&blk.ln_attn_beta);
    p.tensors.push_back(&blk.ln_mlp_gamma);
    p.tensors.push_back(&blk.ln_mlp_beta);
  }
  p.tensors.push_back(&w_lm_);
  p.tensors.push_back(&b_lm_);
  p.tensors.push_back(&ln_final_gamma_);
  p.tensors.push_back(&ln_final_beta_);
  for (auto& blk : blocks_) {
    p.tensors.push_back(&blk.swiglu_gate);
    p.tensors.push_back(&blk.swiglu_gate_b);
    p.tensors.push_back(&blk.swiglu_up);
    p.tensors.push_back(&blk.swiglu_up_b);
    p.tensors.push_back(&blk.swiglu_down);
    p.tensors.push_back(&blk.swiglu_down_b);
  }
  for (auto& blk : blocks_) {
    p.tensors.push_back(&blk.mla_w_q); p.tensors.push_back(&blk.mla_b_q);
    p.tensors.push_back(&blk.mla_w_dkv); p.tensors.push_back(&blk.mla_b_dkv);
    p.tensors.push_back(&blk.mla_w_uk); p.tensors.push_back(&blk.mla_w_uv);
    p.tensors.push_back(&blk.mla_w_o); p.tensors.push_back(&blk.mla_b_o);
  }
  for (auto& blk : blocks_) {
    p.tensors.push_back(&blk.moe_router_w);
    p.tensors.push_back(&blk.moe_router_b);
    for (auto& t : blk.moe_expert_wfc)  p.tensors.push_back(&t);
    for (auto& t : blk.moe_expert_bfc)  p.tensors.push_back(&t);
    for (auto& t : blk.moe_expert_wout) p.tensors.push_back(&t);
    for (auto& t : blk.moe_expert_bout) p.tensors.push_back(&t);
  }
  return p;
}

ParamsConst TinyGPT::parameters_const() const {
  ParamsConst p;
  p.tensors.push_back(&wte_);
  p.tensors.push_back(&wpe_);
  for (const auto& blk : blocks_) {
    p.tensors.push_back(&blk.w_qkv);
    p.tensors.push_back(&blk.b_qkv);
    p.tensors.push_back(&blk.w_proj);
    p.tensors.push_back(&blk.b_proj);
    p.tensors.push_back(&blk.w_fc);
    p.tensors.push_back(&blk.b_fc);
    p.tensors.push_back(&blk.w_out);
    p.tensors.push_back(&blk.b_out);
    p.tensors.push_back(&blk.ln_attn_gamma);
    p.tensors.push_back(&blk.ln_attn_beta);
    p.tensors.push_back(&blk.ln_mlp_gamma);
    p.tensors.push_back(&blk.ln_mlp_beta);
  }
  p.tensors.push_back(&w_lm_);
  p.tensors.push_back(&b_lm_);
  p.tensors.push_back(&ln_final_gamma_);
  p.tensors.push_back(&ln_final_beta_);
  for (const auto& blk : blocks_) {
    p.tensors.push_back(&blk.swiglu_gate);
    p.tensors.push_back(&blk.swiglu_gate_b);
    p.tensors.push_back(&blk.swiglu_up);
    p.tensors.push_back(&blk.swiglu_up_b);
    p.tensors.push_back(&blk.swiglu_down);
    p.tensors.push_back(&blk.swiglu_down_b);
  }
  for (const auto& blk : blocks_) {
    p.tensors.push_back(&blk.mla_w_q); p.tensors.push_back(&blk.mla_b_q);
    p.tensors.push_back(&blk.mla_w_dkv); p.tensors.push_back(&blk.mla_b_dkv);
    p.tensors.push_back(&blk.mla_w_uk); p.tensors.push_back(&blk.mla_w_uv);
    p.tensors.push_back(&blk.mla_w_o); p.tensors.push_back(&blk.mla_b_o);
  }
  for (const auto& blk : blocks_) {
    p.tensors.push_back(&blk.moe_router_w);
    p.tensors.push_back(&blk.moe_router_b);
    for (const auto& t : blk.moe_expert_wfc)  p.tensors.push_back(&t);
    for (const auto& t : blk.moe_expert_bfc)  p.tensors.push_back(&t);
    for (const auto& t : blk.moe_expert_wout) p.tensors.push_back(&t);
    for (const auto& t : blk.moe_expert_bout) p.tensors.push_back(&t);
  }
  return p;
}

} // namespace model
