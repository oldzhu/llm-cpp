#include "model.h"

#include <cmath>
#include <stdexcept>

#include "ops.h"

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

    blocks_[static_cast<std::size_t>(i)] = std::move(blk);
  }

  w_lm_ = Tensor::randn({C, V}, init_std(C), seed ^ 0xC0FFEEULL, true);
  b_lm_ = Tensor::zeros({V}, true);
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

  // === Embedding stage ===
  // Token ids -> vectors: X = Wte[tokens] + Wpe[pos]
  Tensor x = nn::embedding(wte_, tokens_bt, B, T); // [B,T,C]
  x = add_positional(x, B, T);

  for (int li = 0; li < cfg_.n_layers; ++li) {
    Block& blk = blocks_[static_cast<std::size_t>(li)];

    // === Transformer block (pre-norm) ===
    // Attention sublayer:
    //   H = LN(X)
    //   A = CausalSelfAttn(H)
    //   X = X + A
    Tensor h = nn::layernorm_lastdim(x, 1e-5f);
    Tensor a = nn::self_attention_1h(h, blk.w_qkv, blk.b_qkv, blk.w_proj, blk.b_proj);
    x = nn::add(x, a);

    // MLP sublayer:
    //   M  = LN(X)
    //   FF = GELU(M W_fc + b_fc) W_out + b_out
    //   X  = X + FF
    Tensor m = nn::layernorm_lastdim(x, 1e-5f);
    Tensor ff = nn::linear_lastdim(m, blk.w_fc, blk.b_fc);
    ff = nn::gelu(ff);
    ff = nn::linear_lastdim(ff, blk.w_out, blk.b_out);
    x = nn::add(x, ff);
  }

  // Final norm + LM head:
  //   Xn = LN(X)
  //   logits = Xn W_lm + b_lm
  Tensor xn = nn::layernorm_lastdim(x, 1e-5f);
  Tensor logits = nn::linear_lastdim(xn, w_lm_, b_lm_); // [B,T,V]
  return logits;
}

Tensor TinyGPT::loss(const std::vector<std::int32_t>& tokens_bt,
                     const std::vector<std::int32_t>& targets_bt,
                     int B,
                     int T) {
  Tensor logits = forward_logits(tokens_bt, B, T);
  Tensor logits2 = nn::reshape(logits, {B * T, cfg_.vocab_size});
  return nn::cross_entropy(logits2, targets_bt);
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
  }
  w_lm_.zero_grad();
  b_lm_.zero_grad();
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
  }
  p.tensors.push_back(&w_lm_);
  p.tensors.push_back(&b_lm_);
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
  }
  p.tensors.push_back(&w_lm_);
  p.tensors.push_back(&b_lm_);
  return p;
}

} // namespace model
