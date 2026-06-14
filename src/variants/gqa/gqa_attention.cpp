#include "variants/gqa/gqa_attention.h"

#include <cmath>
#include <stdexcept>

#include "ops.h"

namespace nn::variants::gqa {

static bool want_grad(const Tensor& t) {
  return is_grad_enabled() && t.requires_grad;
}

Tensor self_attention_gqa(const Tensor& q_4d,
                          const Tensor& k_4d,
                          const Tensor& v_4d,
                          int n_heads,
                          int n_kv_heads) {
  // Grouped-Query Attention:
  //   Q: [B,T,H_q,D],  K: [B,T,H_kv,D],  V: [B,T,H_kv,D]
  //   H_q = n_heads, H_kv = n_kv_heads, H_q % H_kv == 0
  //   Group ratio: heads_per_kv = n_heads / n_kv_heads
  //   kv_idx(qh) = qh / heads_per_kv
  //
  // Math per (b, t_q) for each Q head hq:
  //   hv = hq * n_kv_heads / n_heads
  //   S_hq[t_q, t_k] = dot(Q[b,t_q,hq,:], K[b,t_k,hv,:]) / sqrt(D) + mask
  //   P_hq[t_q,:] = softmax(S_hq[t_q,:])
  //   Y_hq[t_q,:] = sum_tk P_hq[t_q,t_k] * V[b,t_k,hv,:]
  // Concat: Y → [B,T,H_q*D]

  if (q_4d.shape.size() != 4 || k_4d.shape.size() != 4 || v_4d.shape.size() != 4) {
    throw std::runtime_error("GQA: inputs must be 4D [B,T,heads,D]");
  }
  if (n_heads <= 0 || n_kv_heads <= 0) throw std::runtime_error("GQA: heads must be > 0");
  if (n_heads % n_kv_heads != 0) throw std::runtime_error("GQA: n_heads must be divisible by n_kv_heads");

  const int B = q_4d.shape[0];
  const int T = q_4d.shape[1];
  const int Hq = q_4d.shape[2];
  const int D = q_4d.shape[3];

  if (Hq != n_heads) throw std::runtime_error("GQA: Q heads mismatch");
  if (k_4d.shape[0] != B || k_4d.shape[1] != T || k_4d.shape[2] != n_kv_heads || k_4d.shape[3] != D) {
    throw std::runtime_error("GQA: K shape mismatch");
  }
  if (v_4d.shape[0] != B || v_4d.shape[1] != T || v_4d.shape[2] != n_kv_heads || v_4d.shape[3] != D) {
    throw std::runtime_error("GQA: V shape mismatch");
  }

  const int heads_per_kv = n_heads / n_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  // scores: [B,Hq,T,T] — one T×T matrix per query head
  Tensor scores = Tensor::zeros({B, Hq, T, T}, want_grad(q_4d) || want_grad(k_4d));

  for (int bb = 0; bb < B; ++bb) {
    for (int hq = 0; hq < Hq; ++hq) {
      const int hv = hq / heads_per_kv;  // corresponding KV head
      for (int i = 0; i < T; ++i) {
        for (int j = 0; j < T; ++j) {
          float s = 0.0f;
          const std::size_t q_base = (((static_cast<std::size_t>(bb) * T + i) * Hq + hq) * D);
          const std::size_t k_base = (((static_cast<std::size_t>(bb) * T + j) * n_kv_heads + hv) * D);
          for (int d = 0; d < D; ++d) {
            s += (*q_4d.data)[q_base + static_cast<std::size_t>(d)] *
                 (*k_4d.data)[k_base + static_cast<std::size_t>(d)];
          }
          s *= scale;
          if (j > i) s = -1e9f; // causal mask
          const std::size_t s_off = (((static_cast<std::size_t>(bb) * Hq + hq) * T + i) * T + j);
          (*scores.data)[s_off] = s;
        }
      }
    }
  }

  if (scores.requires_grad) {
    scores.node = std::make_shared<Node>();
    scores.node->parents = {q_4d, k_4d};
    scores.node->backward = [B, T, Hq, n_kv_heads, D, heads_per_kv, scale](Tensor& o) {
      Tensor& qq = o.node->parents[0];
      Tensor& kk = o.node->parents[1];

      if (qq.requires_grad) {
        for (int bb = 0; bb < B; ++bb) {
          for (int hq = 0; hq < Hq; ++hq) {
            const int hv = hq / heads_per_kv;
            for (int i = 0; i < T; ++i) {
              const std::size_t q_base = (((static_cast<std::size_t>(bb) * T + i) * Hq + hq) * D);
              for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < T; ++j) {
                  if (j > i) continue;
                  const std::size_t s_off = (((static_cast<std::size_t>(bb) * Hq + hq) * T + i) * T + j);
                  const std::size_t k_base = (((static_cast<std::size_t>(bb) * T + j) * n_kv_heads + hv) * D);
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
          for (int hv = 0; hv < n_kv_heads; ++hv) {
            for (int j = 0; j < T; ++j) {
              const std::size_t k_base = (((static_cast<std::size_t>(bb) * T + j) * n_kv_heads + hv) * D);
              for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int i = j; i < T; ++i) {
                  const int hq_start = hv * heads_per_kv;
                  const int hq_end = hq_start + heads_per_kv;
                  for (int hq = hq_start; hq < hq_end; ++hq) {
                    const std::size_t s_off = (((static_cast<std::size_t>(bb) * Hq + hq) * T + i) * T + j);
                    const std::size_t q_base = (((static_cast<std::size_t>(bb) * T + i) * Hq + hq) * D);
                    sum += (*o.grad)[s_off] * (*qq.data)[q_base + static_cast<std::size_t>(d)];
                  }
                }
                (*kk.grad)[k_base + static_cast<std::size_t>(d)] += sum * scale;
              }
            }
          }
        }
      }
    };
  }

  Tensor probs = nn::softmax_lastdim(scores); // [B,Hq,T,T]

  // att: [B,T,Hq,D]
  Tensor att = Tensor::zeros({B, T, Hq, D}, want_grad(probs) || want_grad(v_4d));
  for (int bb = 0; bb < B; ++bb) {
    for (int i = 0; i < T; ++i) {
      for (int hq = 0; hq < Hq; ++hq) {
        const int hv = hq / heads_per_kv;
        for (int d = 0; d < D; ++d) {
          float sum = 0.0f;
          for (int j = 0; j < T; ++j) {
            const std::size_t p_off = (((static_cast<std::size_t>(bb) * Hq + hq) * T + i) * T + j);
            const float p = (*probs.data)[p_off];
            const std::size_t v_off = (((static_cast<std::size_t>(bb) * T + j) * n_kv_heads + hv) * D) + static_cast<std::size_t>(d);
            sum += p * (*v_4d.data)[v_off];
          }
          const std::size_t a_off = (((static_cast<std::size_t>(bb) * T + i) * Hq + hq) * D) + static_cast<std::size_t>(d);
          (*att.data)[a_off] = sum;
        }
      }
    }
  }

  if (att.requires_grad) {
    att.node = std::make_shared<Node>();
    att.node->parents = {probs, v_4d};
    att.node->backward = [B, T, Hq, n_kv_heads, D, heads_per_kv](Tensor& o) {
      Tensor& p = o.node->parents[0];
      Tensor& vv = o.node->parents[1];

      if (p.requires_grad) {
        for (int bb = 0; bb < B; ++bb) {
          for (int i = 0; i < T; ++i) {
            for (int hq = 0; hq < Hq; ++hq) {
              const int hv = hq / heads_per_kv;
              for (int j = 0; j < T; ++j) {
                float sum = 0.0f;
                for (int d = 0; d < D; ++d) {
                  const std::size_t a_off = (((static_cast<std::size_t>(bb) * T + i) * Hq + hq) * D) + static_cast<std::size_t>(d);
                  const std::size_t v_off = (((static_cast<std::size_t>(bb) * T + j) * n_kv_heads + hv) * D) + static_cast<std::size_t>(d);
                  sum += (*o.grad)[a_off] * (*vv.data)[v_off];
                }
                const std::size_t p_off = (((static_cast<std::size_t>(bb) * Hq + hq) * T + i) * T + j);
                (*p.grad)[p_off] += sum;
              }
            }
          }
        }
      }

      if (vv.requires_grad) {
        for (int bb = 0; bb < B; ++bb) {
          for (int j = 0; j < T; ++j) {
            for (int hv = 0; hv < n_kv_heads; ++hv) {
              for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                const int hq_start = hv * heads_per_kv;
                const int hq_end = hq_start + heads_per_kv;
                for (int i = 0; i < T; ++i) {
                  for (int hq = hq_start; hq < hq_end; ++hq) {
                    const std::size_t p_off = (((static_cast<std::size_t>(bb) * Hq + hq) * T + i) * T + j);
                    const std::size_t a_off = (((static_cast<std::size_t>(bb) * T + i) * Hq + hq) * D) + static_cast<std::size_t>(d);
                    sum += (*p.data)[p_off] * (*o.grad)[a_off];
                  }
                }
                const std::size_t v_off = (((static_cast<std::size_t>(bb) * T + j) * n_kv_heads + hv) * D) + static_cast<std::size_t>(d);
                (*vv.grad)[v_off] += sum;
              }
            }
          }
        }
      }
    };
  }

  Tensor att_cat = nn::reshape(att, {B, T, Hq * D});
  return att_cat;
}

} // namespace nn::variants::gqa
