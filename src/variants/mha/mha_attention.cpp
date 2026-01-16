#include "variants/mha/mha_attention.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nn::variants::mha {

static bool want_grad(const Tensor& t) {
  return is_grad_enabled() && t.requires_grad;
}

static Tensor slice_lastdim_copy(const Tensor& x, int offset, int length) {
  if (x.shape.empty()) throw std::runtime_error("slice_lastdim: empty shape");
  const int D = x.shape.back();
  if (offset < 0 || length <= 0 || offset + length > D) throw std::runtime_error("slice_lastdim: invalid slice");
  const std::size_t outer = x.numel() / static_cast<std::size_t>(D);

  std::vector<int> out_shape = x.shape;
  out_shape.back() = length;
  Tensor out = Tensor::zeros(out_shape, want_grad(x));

  for (std::size_t o = 0; o < outer; ++o) {
    const std::size_t base_in = o * static_cast<std::size_t>(D) + static_cast<std::size_t>(offset);
    const std::size_t base_out = o * static_cast<std::size_t>(length);
    for (int i = 0; i < length; ++i) {
      (*out.data)[base_out + static_cast<std::size_t>(i)] = (*x.data)[base_in + static_cast<std::size_t>(i)];
    }
  }

  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {x};
    out.node->backward = [offset, length, D](Tensor& o) {
      Tensor& px = o.node->parents[0];
      if (!px.requires_grad) return;
      const std::size_t outer2 = o.numel() / static_cast<std::size_t>(length);
      for (std::size_t outi = 0; outi < outer2; ++outi) {
        const std::size_t base_in = outi * static_cast<std::size_t>(D) + static_cast<std::size_t>(offset);
        const std::size_t base_out = outi * static_cast<std::size_t>(length);
        for (int i = 0; i < length; ++i) {
          (*px.grad)[base_in + static_cast<std::size_t>(i)] += (*o.grad)[base_out + static_cast<std::size_t>(i)];
        }
      }
    };
  }

  return out;
}

Tensor self_attention_mha(const Tensor& x,
                          const Tensor& w_qkv,
                          const Tensor& b_qkv,
                          const Tensor& w_proj,
                          const Tensor& b_proj,
                          int n_heads) {
  // Naive causal multi-head attention.
  // Shapes:
  //   x: [B,T,C]
  //   w_qkv: [C,3C], b_qkv: [3C]
  //   w_proj: [C,C], b_proj: [C]
  //   C = n_heads * D
  //
  // Math (per head h):
  //   [Q,K,V] = x W_qkv + b_qkv
  //   S_h[i,j] = (Q_h[i]·K_h[j]) / sqrt(D) + mask
  //   P_h[i,:] = softmax(S_h[i,:])
  //   Y_h[i]   = sum_j P_h[i,j] V_h[j]
  //   Y = concat_h(Y_h)   // [B,T,C]
  //   out = Y W_proj + b_proj

  if (x.shape.size() != 3) throw std::runtime_error("mha: x must be [B,T,C]");
  if (n_heads <= 0) throw std::runtime_error("mha: n_heads must be > 0");

  const int B = x.shape[0];
  const int T = x.shape[1];
  const int C = x.shape[2];
  if (C % n_heads != 0) throw std::runtime_error("mha: C must be divisible by n_heads");
  const int D = C / n_heads;

  if (w_qkv.shape != std::vector<int>({C, 3 * C})) throw std::runtime_error("mha: w_qkv shape mismatch");
  if (b_qkv.shape != std::vector<int>({3 * C})) throw std::runtime_error("mha: b_qkv shape mismatch");
  if (w_proj.shape != std::vector<int>({C, C})) throw std::runtime_error("mha: w_proj shape mismatch");
  if (b_proj.shape != std::vector<int>({C})) throw std::runtime_error("mha: b_proj shape mismatch");

  Tensor qkv = nn::linear_lastdim(x, w_qkv, b_qkv); // [B,T,3C]
  Tensor q = slice_lastdim_copy(qkv, 0, C);         // [B,T,C]
  Tensor k = slice_lastdim_copy(qkv, C, C);         // [B,T,C]
  Tensor v = slice_lastdim_copy(qkv, 2 * C, C);     // [B,T,C]

  Tensor q4 = nn::reshape(q, {B, T, n_heads, D}); // [B,T,H,D]
  Tensor k4 = nn::reshape(k, {B, T, n_heads, D}); // [B,T,H,D]
  Tensor v4 = nn::reshape(v, {B, T, n_heads, D}); // [B,T,H,D]

  // scores: [B,H,T,T]
  Tensor scores = Tensor::zeros({B, n_heads, T, T}, want_grad(q4) || want_grad(k4));
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  for (int bb = 0; bb < B; ++bb) {
    for (int hh = 0; hh < n_heads; ++hh) {
      for (int i = 0; i < T; ++i) {
        for (int j = 0; j < T; ++j) {
          float s = 0.0f;
          const std::size_t q_base = ((static_cast<std::size_t>(bb) * T + i) * n_heads + hh) * D;
          const std::size_t k_base = ((static_cast<std::size_t>(bb) * T + j) * n_heads + hh) * D;
          for (int d = 0; d < D; ++d) {
            s += (*q4.data)[q_base + static_cast<std::size_t>(d)] * (*k4.data)[k_base + static_cast<std::size_t>(d)];
          }
          s *= scale;
          if (j > i) s = -1e9f;
          const std::size_t s_off = (((static_cast<std::size_t>(bb) * n_heads + hh) * T + i) * T + j);
          (*scores.data)[s_off] = s;
        }
      }
    }
  }

  if (scores.requires_grad) {
    scores.node = std::make_shared<Node>();
    scores.node->parents = {q4, k4};
    scores.node->backward = [B, T, n_heads, D, scale](Tensor& o) {
      Tensor& qq = o.node->parents[0];
      Tensor& kk = o.node->parents[1];

      if (qq.requires_grad) {
        for (int bb = 0; bb < B; ++bb) {
          for (int hh = 0; hh < n_heads; ++hh) {
            for (int i = 0; i < T; ++i) {
              const std::size_t q_base = ((static_cast<std::size_t>(bb) * T + i) * n_heads + hh) * D;
              for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < T; ++j) {
                  if (j > i) continue;
                  const std::size_t s_off = (((static_cast<std::size_t>(bb) * n_heads + hh) * T + i) * T + j);
                  const std::size_t k_base = ((static_cast<std::size_t>(bb) * T + j) * n_heads + hh) * D;
                  sum += (*o.grad)[s_off] * (*kk.data)[k_base + static_cast<std::size_t>(d)];
                }
                (*qq.grad)[q_base + static_cast<std::size_t>(d)] += sum * scale;
              }
            }
          }
        }
      }

      if (kk.requires_grad) {
        for (int bb = 0; bb < B; ++bb) {
          for (int hh = 0; hh < n_heads; ++hh) {
            for (int j = 0; j < T; ++j) {
              const std::size_t k_base = ((static_cast<std::size_t>(bb) * T + j) * n_heads + hh) * D;
              for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int i = j; i < T; ++i) {
                  const std::size_t s_off = (((static_cast<std::size_t>(bb) * n_heads + hh) * T + i) * T + j);
                  const std::size_t q_base = ((static_cast<std::size_t>(bb) * T + i) * n_heads + hh) * D;
                  sum += (*o.grad)[s_off] * (*qq.data)[q_base + static_cast<std::size_t>(d)];
                }
                (*kk.grad)[k_base + static_cast<std::size_t>(d)] += sum * scale;
              }
            }
          }
        }
      }
    };
  }

  Tensor probs = nn::softmax_lastdim(scores); // [B,H,T,T]

  // att: [B,T,H,D]
  Tensor att = Tensor::zeros({B, T, n_heads, D}, want_grad(probs) || want_grad(v4));
  for (int bb = 0; bb < B; ++bb) {
    for (int i = 0; i < T; ++i) {
      for (int hh = 0; hh < n_heads; ++hh) {
        for (int d = 0; d < D; ++d) {
          float sum = 0.0f;
          for (int j = 0; j < T; ++j) {
            const std::size_t p_off = (((static_cast<std::size_t>(bb) * n_heads + hh) * T + i) * T + j);
            const float p = (*probs.data)[p_off];
            const std::size_t v_off = ((static_cast<std::size_t>(bb) * T + j) * n_heads + hh) * D + static_cast<std::size_t>(d);
            sum += p * (*v4.data)[v_off];
          }
          const std::size_t a_off = ((static_cast<std::size_t>(bb) * T + i) * n_heads + hh) * D + static_cast<std::size_t>(d);
          (*att.data)[a_off] = sum;
        }
      }
    }
  }

  if (att.requires_grad) {
    att.node = std::make_shared<Node>();
    att.node->parents = {probs, v4};
    att.node->backward = [B, T, n_heads, D](Tensor& o) {
      Tensor& p = o.node->parents[0];
      Tensor& vv = o.node->parents[1];

      if (p.requires_grad) {
        for (int bb = 0; bb < B; ++bb) {
          for (int i = 0; i < T; ++i) {
            for (int hh = 0; hh < n_heads; ++hh) {
              for (int j = 0; j < T; ++j) {
                float sum = 0.0f;
                for (int d = 0; d < D; ++d) {
                  const std::size_t a_off = ((static_cast<std::size_t>(bb) * T + i) * n_heads + hh) * D + static_cast<std::size_t>(d);
                  const std::size_t v_off = ((static_cast<std::size_t>(bb) * T + j) * n_heads + hh) * D + static_cast<std::size_t>(d);
                  sum += (*o.grad)[a_off] * (*vv.data)[v_off];
                }
                const std::size_t p_off = (((static_cast<std::size_t>(bb) * n_heads + hh) * T + i) * T + j);
                (*p.grad)[p_off] += sum;
              }
            }
          }
        }
      }

      if (vv.requires_grad) {
        for (int bb = 0; bb < B; ++bb) {
          for (int j = 0; j < T; ++j) {
            for (int hh = 0; hh < n_heads; ++hh) {
              for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int i = 0; i < T; ++i) {
                  const std::size_t p_off = (((static_cast<std::size_t>(bb) * n_heads + hh) * T + i) * T + j);
                  const std::size_t a_off = ((static_cast<std::size_t>(bb) * T + i) * n_heads + hh) * D + static_cast<std::size_t>(d);
                  sum += (*p.data)[p_off] * (*o.grad)[a_off];
                }
                const std::size_t v_off = ((static_cast<std::size_t>(bb) * T + j) * n_heads + hh) * D + static_cast<std::size_t>(d);
                (*vv.grad)[v_off] += sum;
              }
            }
          }
        }
      }
    };
  }

  Tensor att_cat = nn::reshape(att, {B, T, C});
  Tensor proj = nn::linear_lastdim(att_cat, w_proj, b_proj);
  return proj;
}

} // namespace nn::variants::mha
