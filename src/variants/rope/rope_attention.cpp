#include "variants/rope/rope_attention.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "ops.h"

namespace nn::variants::rope {

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

void rope_rotate(Tensor& q, Tensor& k, int B, int T, int C) {
  // Compute rotation angles once per position-pair
  if (C % 2 != 0) throw std::runtime_error("rope_rotate: C must be even");
  const int n_pairs = C / 2;
  const float base = 10000.0f;

  // theta[k] for each pair k
  std::vector<float> theta(static_cast<std::size_t>(n_pairs));
  for (int pk = 0; pk < n_pairs; ++pk) {
    theta[static_cast<std::size_t>(pk)] = 1.0f / std::pow(base, 2.0f * static_cast<float>(pk) / static_cast<float>(C));
  }

  // For each batch, position, and pair: apply rotation
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      const float pos_f = static_cast<float>(t);
      const std::size_t base_off = (static_cast<std::size_t>(b) * T + t) * C;

      for (int pk = 0; pk < n_pairs; ++pk) {
        const float angle = pos_f * theta[static_cast<std::size_t>(pk)];
        const float cos_a = std::cos(angle);
        const float sin_a = std::sin(angle);

        const std::size_t i0 = base_off + static_cast<std::size_t>(2 * pk);
        const std::size_t i1 = base_off + static_cast<std::size_t>(2 * pk + 1);

        // Rotate q
        float q0 = (*q.data)[i0];
        float q1 = (*q.data)[i1];
        (*q.data)[i0] = q0 * cos_a - q1 * sin_a;
        (*q.data)[i1] = q1 * cos_a + q0 * sin_a;

        // Rotate k
        float k0 = (*k.data)[i0];
        float k1 = (*k.data)[i1];
        (*k.data)[i0] = k0 * cos_a - k1 * sin_a;
        (*k.data)[i1] = k1 * cos_a + k0 * sin_a;
      }
    }
  }
}

Tensor self_attention_rope(const Tensor& x,
                           const Tensor& w_qkv,
                           const Tensor& b_qkv,
                           const Tensor& w_proj,
                           const Tensor& b_proj) {
  // Causal self-attention with RoPE (single head).
  // Shapes:
  //   x: [B,T,C]
  //   w_qkv: [C,3C], b_qkv: [3C]
  //   w_proj: [C,C], b_proj: [C]
  //
  // Math:
  //   [Q,K,V] = x W_qkv + b_qkv
  //   RoPE-rotate Q and K
  //   S[i,j] = (Q[i]·K[j]) / sqrt(C) + mask(j>i → -inf)
  //   P[i,:] = softmax(S[i,:])
  //   Y[i]   = sum_j P[i,j] V[j]
  //   out    = Y W_proj + b_proj

  if (x.shape.size() != 3) throw std::runtime_error("rope attn: x must be [B,T,C]");
  const int B = x.shape[0];
  const int T = x.shape[1];
  const int C = x.shape[2];
  if (C % 2 != 0) throw std::runtime_error("rope attn: C must be even for RoPE");

  if (w_qkv.shape != std::vector<int>({C, 3 * C})) throw std::runtime_error("rope attn: w_qkv shape mismatch");
  if (b_qkv.shape != std::vector<int>({3 * C})) throw std::runtime_error("rope attn: b_qkv shape mismatch");
  if (w_proj.shape != std::vector<int>({C, C})) throw std::runtime_error("rope attn: w_proj shape mismatch");
  if (b_proj.shape != std::vector<int>({C})) throw std::runtime_error("rope attn: b_proj shape mismatch");

  Tensor qkv = nn::linear_lastdim(x, w_qkv, b_qkv); // [B,T,3C]

  Tensor q = slice_lastdim_copy(qkv, 0, C);     // [B,T,C]
  Tensor k = slice_lastdim_copy(qkv, C, C);     // [B,T,C]
  Tensor v = slice_lastdim_copy(qkv, 2 * C, C); // [B,T,C]

  // Apply RoPE to Q and K
  rope_rotate(q, k, B, T, C);

  // scores: [B,T,T]
  Tensor scores = Tensor::zeros({B, T, T}, want_grad(q) || want_grad(k));
  const float scale = 1.0f / std::sqrt(static_cast<float>(C));
  for (int bb = 0; bb < B; ++bb) {
    for (int i = 0; i < T; ++i) {
      for (int j = 0; j < T; ++j) {
        float s = 0.0f;
        const std::size_t qi = (static_cast<std::size_t>(bb) * T + i) * C;
        const std::size_t kj = (static_cast<std::size_t>(bb) * T + j) * C;
        for (int c = 0; c < C; ++c) s += (*q.data)[qi + c] * (*k.data)[kj + c];
        s *= scale;
        if (j > i) s = -1e9f;
        (*scores.data)[(static_cast<std::size_t>(bb) * T + i) * T + j] = s;
      }
    }
  }

  if (scores.requires_grad) {
    scores.node = std::make_shared<Node>();
    scores.node->parents = {q, k};
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

  Tensor probs = nn::softmax_lastdim(scores); // [B,T,T]

  // att = probs @ v => [B,T,C]
  Tensor att = Tensor::zeros({B, T, C}, want_grad(probs) || want_grad(v));
  for (int bb = 0; bb < B; ++bb) {
    for (int i = 0; i < T; ++i) {
      for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (int j = 0; j < T; ++j) {
          const float p = (*probs.data)[(static_cast<std::size_t>(bb) * T + i) * T + j];
          sum += p * (*v.data)[(static_cast<std::size_t>(bb) * T + j) * C + c];
        }
        (*att.data)[(static_cast<std::size_t>(bb) * T + i) * C + c] = sum;
      }
    }
  }

  if (att.requires_grad) {
    att.node = std::make_shared<Node>();
    att.node->parents = {probs, v};
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

  Tensor proj = nn::linear_lastdim(att, w_proj, b_proj);
  return proj;
}

} // namespace nn::variants::rope
