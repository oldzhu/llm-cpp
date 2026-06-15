#include "variants/mla/mla_attention.h"

#include <cmath>
#include <stdexcept>

#include "ops.h"

namespace nn::variants::mla {

static bool want_grad(const Tensor& t) {
  return is_grad_enabled() && t.requires_grad;
}

Tensor self_attention_mla(const Tensor& x,
                           const Tensor& w_q,
                           const Tensor& b_q,
                           const Tensor& w_dkv,
                           const Tensor& b_dkv,
                           const Tensor& w_uk,
                           const Tensor& w_uv,
                           const Tensor& w_o,
                           const Tensor& b_o) {
  // Multi-Head Latent Attention (explicit, teaching-first)
  //
  // Shapes:
  //   x: [B,T,C]   w_q: [C,C]   b_q: [C]
  //   w_dkv: [C,L]  b_dkv: [L]
  //   w_uk: [L,C]   w_uv: [L,C]
  //   w_o: [C,C]    b_o: [C]
  //   L = latent_dim  (L < C for compression benefit)

  if (x.shape.size() != 3) throw std::runtime_error("MLA: x must be [B,T,C]");
  const int B = x.shape[0];
  const int T = x.shape[1];
  const int C = x.shape[2];

  if (w_q.shape != std::vector<int>({C, C})) throw std::runtime_error("MLA: w_q must be [C,C]");
  if (b_q.shape != std::vector<int>({C})) throw std::runtime_error("MLA: b_q must be [C]");
  if (w_dkv.shape.size() != 2) throw std::runtime_error("MLA: w_dkv must be [C,L]");
  const int L = w_dkv.shape[1]; // latent dimension
  if (w_dkv.shape[0] != C) throw std::runtime_error("MLA: w_dkv dim0 must be C");
  if (b_dkv.shape != std::vector<int>({L})) throw std::runtime_error("MLA: b_dkv must be [L]");
  if (w_uk.shape != std::vector<int>({L, C})) throw std::runtime_error("MLA: w_uk must be [L,C]");
  if (w_uv.shape != std::vector<int>({L, C})) throw std::runtime_error("MLA: w_uv must be [L,C]");
  if (w_o.shape != std::vector<int>({C, C})) throw std::runtime_error("MLA: w_o must be [C,C]");
  if (b_o.shape != std::vector<int>({C})) throw std::runtime_error("MLA: b_o must be [C]");

  // 1) Q = x @ w_q + b_q
  Tensor q3 = nn::linear_lastdim(x, w_q, b_q); // [B,T,C]

  // 2) Compress: c_KV = x @ w_dkv + b_dkv
  Tensor c_kv3 = nn::linear_lastdim(x, w_dkv, b_dkv); // [B,T,L]

  // 3) Decompress K and V from latent
  // K = c_KV @ w_uk: [B,T,L] @ [L,C] → [B,T,C]
  Tensor k_raw = nn::matmul2d(nn::reshape(c_kv3, {B * T, L}), w_uk); // [B*T, C]
  k_raw = nn::reshape(k_raw, {B, T, C});
  // Add zero bias (MLA typically has no per-head bias for K)
  Tensor k3 = nn::reshape(k_raw, {B, T, C});

  // V = c_KV @ w_uv: [B,T,L] @ [L,C] → [B,T,C]
  Tensor v_raw = nn::matmul2d(nn::reshape(c_kv3, {B * T, L}), w_uv); // [B*T, C]
  v_raw = nn::reshape(v_raw, {B, T, C});
  Tensor v3 = nn::reshape(v_raw, {B, T, C});

  // 4) Compute attention scores: Q @ K^T / sqrt(C) + causal mask
  Tensor scores = Tensor::zeros({B, T, T}, want_grad(q3) || want_grad(k3));
  const float scale = 1.0f / std::sqrt(static_cast<float>(C));
  for (int bb = 0; bb < B; ++bb) {
    for (int i = 0; i < T; ++i) {
      for (int j = 0; j < T; ++j) {
        float s = 0.0f;
        const std::size_t qi = (static_cast<std::size_t>(bb) * T + i) * C;
        const std::size_t kj = (static_cast<std::size_t>(bb) * T + j) * C;
        for (int c = 0; c < C; ++c) s += (*q3.data)[qi + c] * (*k3.data)[kj + c];
        s *= scale;
        if (j > i) s = -1e9f;
        (*scores.data)[(static_cast<std::size_t>(bb) * T + i) * T + j] = s;
      }
    }
  }

  // Backward for scores
  if (scores.requires_grad) {
    scores.node = std::make_shared<Node>();
    scores.node->parents = {q3, k3};
    scores.node->backward = [B, T, C, scale](Tensor& o) {
      Tensor& qq = o.node->parents[0];
      Tensor& kk = o.node->parents[1];
      if (qq.requires_grad) {
        for (int bb = 0; bb < B; ++bb) {
          for (int i = 0; i < T; ++i) {
            const std::size_t qi = (static_cast<std::size_t>(bb) * T + i) * C;
            for (int c = 0; c < C; ++c) {
              float sum = 0.0f;
              for (int j = 0; j < T; ++j) {
                if (j > i) continue;
                const std::size_t kj = (static_cast<std::size_t>(bb) * T + j) * C;
                sum += (*o.grad)[(static_cast<std::size_t>(bb) * T + i) * T + j] * (*kk.data)[kj + c];
              }
              (*qq.grad)[qi + c] += sum * scale;
            }
          }
        }
      }
      if (kk.requires_grad) {
        for (int bb = 0; bb < B; ++bb) {
          for (int j = 0; j < T; ++j) {
            const std::size_t kj = (static_cast<std::size_t>(bb) * T + j) * C;
            for (int c = 0; c < C; ++c) {
              float sum = 0.0f;
              for (int i = j; i < T; ++i) {
                const std::size_t qi = (static_cast<std::size_t>(bb) * T + i) * C;
                sum += (*o.grad)[(static_cast<std::size_t>(bb) * T + i) * T + j] * (*qq.data)[qi + c];
              }
              (*kk.grad)[kj + c] += sum * scale;
            }
          }
        }
      }
    };
  }

  // 5) Softmax + weighted sum
  Tensor probs = nn::softmax_lastdim(scores); // [B,T,T]
  Tensor att = Tensor::zeros({B, T, C}, want_grad(probs) || want_grad(v3));
  for (int bb = 0; bb < B; ++bb) {
    for (int i = 0; i < T; ++i) {
      for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (int j = 0; j < T; ++j) {
          const float p = (*probs.data)[(static_cast<std::size_t>(bb) * T + i) * T + j];
          sum += p * (*v3.data)[(static_cast<std::size_t>(bb) * T + j) * C + c];
        }
        (*att.data)[(static_cast<std::size_t>(bb) * T + i) * C + c] = sum;
      }
    }
  }

  if (att.requires_grad) {
    att.node = std::make_shared<Node>();
    att.node->parents = {probs, v3};
    att.node->backward = [B, T, C](Tensor& o) {
      Tensor& p = o.node->parents[0];
      Tensor& vv = o.node->parents[1];
      if (p.requires_grad) {
        for (int bb = 0; bb < B; ++bb) {
          for (int i = 0; i < T; ++i) {
            for (int j = 0; j < T; ++j) {
              float sum = 0.0f;
              for (int c = 0; c < C; ++c) {
                sum += (*o.grad)[(static_cast<std::size_t>(bb) * T + i) * C + c] *
                       (*vv.data)[(static_cast<std::size_t>(bb) * T + j) * C + c];
              }
              (*p.grad)[(static_cast<std::size_t>(bb) * T + i) * T + j] += sum;
            }
          }
        }
      }
      if (vv.requires_grad) {
        for (int bb = 0; bb < B; ++bb) {
          for (int j = 0; j < T; ++j) {
            for (int c = 0; c < C; ++c) {
              float sum = 0.0f;
              for (int i = 0; i < T; ++i) {
                sum += (*p.data)[(static_cast<std::size_t>(bb) * T + i) * T + j] *
                       (*o.grad)[(static_cast<std::size_t>(bb) * T + i) * C + c];
              }
              (*vv.grad)[(static_cast<std::size_t>(bb) * T + j) * C + c] += sum;
            }
          }
        }
      }
    };
  }

  // 6) Output projection
  Tensor out = nn::linear_lastdim(att, w_o, b_o); // [B,T,C]
  return out;
}

} // namespace nn::variants::mla
