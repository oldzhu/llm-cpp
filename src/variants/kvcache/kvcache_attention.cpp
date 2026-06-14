#include "variants/kvcache/kvcache_attention.h"

#include <cmath>
#include <stdexcept>

#include "model.h"
#include "ops.h"

namespace nn::variants::kvcache {

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

KVCache::KVCache(int B, int T_max, int C) : B(B), T_max(T_max), C(C) {
  k_cache_ = Tensor::zeros({B, T_max, C}, false);
  v_cache_ = Tensor::zeros({B, T_max, C}, false);
  cur_len = 0;
}

void KVCache::reset() {
  cur_len = 0;
}

Tensor self_attention_prefill(const Tensor& x,
                               const Tensor& w_qkv,
                               const Tensor& b_qkv,
                               const Tensor& w_proj,
                               const Tensor& b_proj,
                               KVCache& cache) {
  // Full causal self-attention, with cache population.
  //
  // Shapes:
  //   x: [B,T,C]
  //   w_qkv: [C,3C], b_qkv: [3C]
  //   w_proj: [C,C], b_proj: [C]
  //
  // Math:
  //   [Q,K,V] = x W_qkv + b_qkv
  //   Save K, V → cache
  //   S[i,j] = (Q[i]·K[j]) / sqrt(C) + mask(j>i → -inf)
  //   P[i,:] = softmax(S[i,:])
  //   Y[i]   = sum_j P[i,j] V[j]
  //   out    = Y W_proj + b_proj

  if (x.shape.size() != 3) throw std::runtime_error("kvcache prefill: x must be [B,T,C]");
  const int B = x.shape[0];
  const int T = x.shape[1];
  const int C = x.shape[2];
  if (T < 1) throw std::runtime_error("kvcache prefill: T must be >= 1");
  if (T > cache.T_max) throw std::runtime_error("kvcache prefill: T exceeds cache.T_max");

  if (w_qkv.shape != std::vector<int>({C, 3 * C})) throw std::runtime_error("kvcache prefill: w_qkv shape mismatch");
  if (b_qkv.shape != std::vector<int>({3 * C})) throw std::runtime_error("kvcache prefill: b_qkv shape mismatch");
  if (w_proj.shape != std::vector<int>({C, C})) throw std::runtime_error("kvcache prefill: w_proj shape mismatch");
  if (b_proj.shape != std::vector<int>({C})) throw std::runtime_error("kvcache prefill: b_proj shape mismatch");

  if (cache.B != B || cache.C != C) throw std::runtime_error("kvcache prefill: cache dimensions mismatch");

  Tensor qkv = nn::linear_lastdim(x, w_qkv, b_qkv); // [B,T,3C]

  Tensor q = slice_lastdim_copy(qkv, 0, C);     // [B,T,C]
  Tensor k = slice_lastdim_copy(qkv, C, C);     // [B,T,C]
  Tensor v = slice_lastdim_copy(qkv, 2 * C, C); // [B,T,C]

  // Populate cache
  for (int bb = 0; bb < B; ++bb) {
    for (int t = 0; t < T; ++t) {
      const std::size_t src_base = (static_cast<std::size_t>(bb) * T + t) * C;
      const std::size_t dst_base = (static_cast<std::size_t>(bb) * cache.T_max + t) * C;
      for (int c = 0; c < C; ++c) {
        (*cache.k_cache_.data)[dst_base + c] = (*k.data)[src_base + c];
        (*cache.v_cache_.data)[dst_base + c] = (*v.data)[src_base + c];
      }
    }
  }
  cache.cur_len = T;

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
        if (j > i) s = -1e9f; // causal mask
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

Tensor self_attention_step(const Tensor& x_step,
                            const Tensor& w_qkv,
                            const Tensor& b_qkv,
                            const Tensor& w_proj,
                            const Tensor& b_proj,
                            KVCache& cache) {
  // Incremental attention: one new token, using cached K/V.
  //
  // Shapes:
  //   x_step: [B,1,C]
  //   w_qkv: [C,3C], b_qkv: [3C]
  //   w_proj: [C,C], b_proj: [C]
  //
  // Math:
  //   [Q_new,K_new,V_new] = x_step W_qkv + b_qkv
  //   Append K_new, V_new → cache
  //   S[0,j] = (Q_new·K_cache[j]) / sqrt(C)  (no causal mask — last position)
  //   P[0,:] = softmax(S[0,:])
  //   Y[0]   = sum_j P[0,j] * V_cache[j]
  //   out    = Y W_proj + b_proj
  //
  // This function is generation-only (no autograd).

  if (x_step.shape.size() != 3) throw std::runtime_error("kvcache step: x_step must be [B,1,C]");
  const int B = x_step.shape[0];
  const int T_q = x_step.shape[1];
  const int C = x_step.shape[2];
  if (T_q != 1) throw std::runtime_error("kvcache step: expected T=1 in x_step");
  if (cache.cur_len >= cache.T_max) throw std::runtime_error("kvcache step: cache full");

  if (w_qkv.shape != std::vector<int>({C, 3 * C})) throw std::runtime_error("kvcache step: w_qkv shape mismatch");
  if (b_qkv.shape != std::vector<int>({3 * C})) throw std::runtime_error("kvcache step: b_qkv shape mismatch");
  if (w_proj.shape != std::vector<int>({C, C})) throw std::runtime_error("kvcache step: w_proj shape mismatch");
  if (b_proj.shape != std::vector<int>({C})) throw std::runtime_error("kvcache step: b_proj shape mismatch");

  if (cache.B != B || cache.C != C) throw std::runtime_error("kvcache step: cache dimensions mismatch");

  Tensor qkv_step = nn::linear_lastdim(x_step, w_qkv, b_qkv); // [B,1,3C]

  Tensor q_new = slice_lastdim_copy(qkv_step, 0, C);     // [B,1,C]
  Tensor k_new = slice_lastdim_copy(qkv_step, C, C);     // [B,1,C]
  Tensor v_new = slice_lastdim_copy(qkv_step, 2 * C, C); // [B,1,C]

  // Append K_new, V_new to cache
  const int pos = cache.cur_len;
  for (int bb = 0; bb < B; ++bb) {
    const std::size_t src_base = static_cast<std::size_t>(bb) * 1 * C;
    const std::size_t dst_base = (static_cast<std::size_t>(bb) * cache.T_max + pos) * C;
    for (int c = 0; c < C; ++c) {
      (*cache.k_cache_.data)[dst_base + c] = (*k_new.data)[src_base + c];
      (*cache.v_cache_.data)[dst_base + c] = (*v_new.data)[src_base + c];
    }
  }
  cache.cur_len = pos + 1;

  const int T_key = cache.cur_len;
  const float scale = 1.0f / std::sqrt(static_cast<float>(C));

  // scores: [B,1,T_key] — no causal mask needed (query is at end)
  Tensor scores = Tensor::zeros({B, 1, T_key}, false);
  for (int bb = 0; bb < B; ++bb) {
    for (int j = 0; j < T_key; ++j) {
      float s = 0.0f;
      const std::size_t q_off = static_cast<std::size_t>(bb) * 1 * C;
      const std::size_t k_off = (static_cast<std::size_t>(bb) * cache.T_max + j) * C;
      for (int c = 0; c < C; ++c) {
        s += (*q_new.data)[q_off + c] * (*cache.k_cache_.data)[k_off + c];
      }
      s *= scale;
      (*scores.data)[static_cast<std::size_t>(bb) * 1 * T_key + j] = s;
    }
  }

  Tensor probs = nn::softmax_lastdim(scores); // [B,1,T_key]

  // att: [B,1,C]
  Tensor att = Tensor::zeros({B, 1, C}, false);
  for (int bb = 0; bb < B; ++bb) {
    for (int c = 0; c < C; ++c) {
      float sum = 0.0f;
      for (int j = 0; j < T_key; ++j) {
        const float p = (*probs.data)[static_cast<std::size_t>(bb) * 1 * T_key + j];
        const std::size_t v_off = (static_cast<std::size_t>(bb) * cache.T_max + j) * C + c;
        sum += p * (*cache.v_cache_.data)[v_off];
      }
      (*att.data)[static_cast<std::size_t>(bb) * 1 * C + c] = sum;
    }
  }

  Tensor out = nn::linear_lastdim(att, w_proj, b_proj); // [B,1,C]
  return out;
}

Tensor model_prefill(const model::TinyGPT& gpt,
                      const std::vector<std::int32_t>& tokens,
                      int B,
                      int T,
                      std::vector<KVCache>& layer_caches) {
  const int C = gpt.cfg().d_model;
  const int n_layers = gpt.cfg().n_layers;
  const int norm_type = gpt.cfg().norm_type;
  const int mlp_type = gpt.cfg().mlp_type;
  const int swiglu_param_base = 2 + n_layers * 12 + 4;

  if (T > gpt.cfg().seq_len) throw std::runtime_error("model_prefill: T exceeds seq_len");
  if (static_cast<int>(layer_caches.size()) != n_layers) throw std::runtime_error("model_prefill: layer_caches size mismatch");

  auto params = gpt.parameters_const();

  // tokens → embeddings
  Tensor x = nn::embedding(*params.tensors[0], tokens, B, T); // wte

  // Add positional embeddings
  {
    const nn::Tensor& wpe = *params.tensors[1];
    Tensor xp = Tensor::zeros({B, T, C}, false);
    for (int b = 0; b < B; ++b) {
      for (int t = 0; t < T; ++t) {
        const std::size_t o_off = (static_cast<std::size_t>(b) * T + t) * C;
        const std::size_t p_off = static_cast<std::size_t>(t) * C;
        for (int c = 0; c < C; ++c) {
          (*xp.data)[o_off + c] = (*x.data)[o_off + c] + (*wpe.data)[p_off + c];
        }
      }
    }
    x = xp;
  }

  // Process each layer
  for (int li = 0; li < n_layers; ++li) {
    const int base = 2 + li * 12; // w_qkv, b_qkv, w_proj, b_proj, w_fc, b_fc, w_out, b_out, ln_attn_gamma, ln_attn_beta, ln_mlp_gamma, ln_mlp_beta
    const nn::Tensor& w_qkv = *params.tensors[base + 0];
    const nn::Tensor& b_qkv = *params.tensors[base + 1];
    const nn::Tensor& w_proj = *params.tensors[base + 2];
    const nn::Tensor& b_proj = *params.tensors[base + 3];
    const nn::Tensor& w_fc = *params.tensors[base + 4];
    const nn::Tensor& b_fc = *params.tensors[base + 5];
    const nn::Tensor& w_out = *params.tensors[base + 6];
    const nn::Tensor& b_out = *params.tensors[base + 7];
    const nn::Tensor& ln_attn_gamma = *params.tensors[base + 8];
    const nn::Tensor& ln_attn_beta  = *params.tensors[base + 9];
    const nn::Tensor& ln_mlp_gamma  = *params.tensors[base + 10];
    const nn::Tensor& ln_mlp_beta   = *params.tensors[base + 11];

    // Attention sublayer (prefill — fills cache)
    Tensor h;
    if (norm_type == 0)
      h = nn::layernorm_affine(x, ln_attn_gamma, ln_attn_beta, 1e-5f);
    else
      h = nn::rmsnorm_affine(x, ln_attn_gamma, 1e-5f);
    Tensor a = self_attention_prefill(h, w_qkv, b_qkv, w_proj, b_proj, layer_caches[li]);
    x = nn::add(x, a);

    // MLP sublayer
    Tensor m;
    if (norm_type == 0)
      m = nn::layernorm_affine(x, ln_mlp_gamma, ln_mlp_beta, 1e-5f);
    else
      m = nn::rmsnorm_affine(x, ln_mlp_gamma, 1e-5f);
    Tensor ff;
    if (mlp_type == 0) {
      ff = nn::linear_lastdim(m, w_fc, b_fc);
      ff = nn::gelu(ff);
      ff = nn::linear_lastdim(ff, w_out, b_out);
    } else {
      const nn::Tensor& sg  = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 0];
      const nn::Tensor& sgb = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 1];
      const nn::Tensor& su  = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 2];
      const nn::Tensor& sub = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 3];
      const nn::Tensor& sd  = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 4];
      const nn::Tensor& sdb = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 5];
      Tensor gate = nn::linear_lastdim(m, sg, sgb);
      Tensor up   = nn::linear_lastdim(m, su, sub);
      gate = nn::silu(gate);
      ff = nn::mul(gate, up);
      ff = nn::linear_lastdim(ff, sd, sdb);
    }
    x = nn::add(x, ff);
  }

  // Final norm + LM head
  Tensor xn;
  if (norm_type == 0)
    xn = nn::layernorm_affine(x, *params.tensors[2 + n_layers * 12 + 2], *params.tensors[2 + n_layers * 12 + 3], 1e-5f);
  else
    xn = nn::rmsnorm_affine(x, *params.tensors[2 + n_layers * 12 + 2], 1e-5f);
  const nn::Tensor& w_lm = *params.tensors[2 + n_layers * 12 + 0];
  const nn::Tensor& b_lm = *params.tensors[2 + n_layers * 12 + 1];
  Tensor logits = nn::linear_lastdim(xn, w_lm, b_lm); // [B,T,V]
  return logits;
}

Tensor model_step(const model::TinyGPT& gpt,
                   std::int32_t next_token,
                   int B,
                   int position,
                   std::vector<KVCache>& layer_caches) {
  const int C = gpt.cfg().d_model;
  const int n_layers = gpt.cfg().n_layers;
  const int norm_type = gpt.cfg().norm_type;
  const int mlp_type = gpt.cfg().mlp_type;
  const int swiglu_param_base = 2 + n_layers * 12 + 4;
  const int maxT = gpt.cfg().seq_len;

  if (position >= maxT) throw std::runtime_error("model_step: position exceeds seq_len");
  if (static_cast<int>(layer_caches.size()) != n_layers) throw std::runtime_error("model_step: layer_caches size mismatch");

  auto params = gpt.parameters_const();

  // Token embedding for a single token
  const std::vector<std::int32_t> tokens = {next_token};
  Tensor x = nn::embedding(*params.tensors[0], tokens, B, 1); // [B,1,C]
  const nn::Tensor& wpe = *params.tensors[1];

  // Add positional embedding at the given global position
  Tensor xp = Tensor::zeros({B, 1, C}, false);
  for (int b = 0; b < B; ++b) {
    const std::size_t o_off = static_cast<std::size_t>(b) * C;
    const std::size_t p_off = static_cast<std::size_t>(position) * C;
    for (int c = 0; c < C; ++c) {
      (*xp.data)[o_off + c] = (*x.data)[o_off + c] + (*wpe.data)[p_off + c];
    }
  }
  x = xp; // [B,1,C]

  // Process each layer
  for (int li = 0; li < n_layers; ++li) {
    const int base = 2 + li * 12;
    const nn::Tensor& w_qkv = *params.tensors[base + 0];
    const nn::Tensor& b_qkv = *params.tensors[base + 1];
    const nn::Tensor& w_proj = *params.tensors[base + 2];
    const nn::Tensor& b_proj = *params.tensors[base + 3];
    const nn::Tensor& w_fc = *params.tensors[base + 4];
    const nn::Tensor& b_fc = *params.tensors[base + 5];
    const nn::Tensor& w_out = *params.tensors[base + 6];
    const nn::Tensor& b_out = *params.tensors[base + 7];
    const nn::Tensor& ln_attn_gamma = *params.tensors[base + 8];
    const nn::Tensor& ln_attn_beta  = *params.tensors[base + 9];
    const nn::Tensor& ln_mlp_gamma  = *params.tensors[base + 10];
    const nn::Tensor& ln_mlp_beta   = *params.tensors[base + 11];

    // Attention sublayer (step — uses cache)
    Tensor h = nn::layernorm_affine(x, ln_attn_gamma, ln_attn_beta, 1e-5f);
    Tensor a = self_attention_step(h, w_qkv, b_qkv, w_proj, b_proj, layer_caches[li]);
    x = nn::add(x, a);

    // MLP sublayer
    Tensor m;
    if (norm_type == 0)
      m = nn::layernorm_affine(x, ln_mlp_gamma, ln_mlp_beta, 1e-5f);
    else
      m = nn::rmsnorm_affine(x, ln_mlp_gamma, 1e-5f);
    Tensor ff;
    if (mlp_type == 0) {
      ff = nn::linear_lastdim(m, w_fc, b_fc);
      ff = nn::gelu(ff);
      ff = nn::linear_lastdim(ff, w_out, b_out);
    } else {
      const nn::Tensor& sg  = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 0];
      const nn::Tensor& sgb = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 1];
      const nn::Tensor& su  = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 2];
      const nn::Tensor& sub = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 3];
      const nn::Tensor& sd  = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 4];
      const nn::Tensor& sdb = *params.tensors[swiglu_param_base + static_cast<std::size_t>(li) * 6 + 5];
      Tensor gate = nn::linear_lastdim(m, sg, sgb);
      Tensor up   = nn::linear_lastdim(m, su, sub);
      gate = nn::silu(gate);
      ff = nn::mul(gate, up);
      ff = nn::linear_lastdim(ff, sd, sdb);
    }
    x = nn::add(x, ff);
  }

  // Final norm + LM head
  Tensor xn;
  if (norm_type == 0)
    xn = nn::layernorm_affine(x, *params.tensors[2 + n_layers * 12 + 2], *params.tensors[2 + n_layers * 12 + 3], 1e-5f);
  else
    xn = nn::rmsnorm_affine(x, *params.tensors[2 + n_layers * 12 + 2], 1e-5f);
  const nn::Tensor& w_lm = *params.tensors[2 + n_layers * 12 + 0];
  const nn::Tensor& b_lm = *params.tensors[2 + n_layers * 12 + 1];
  Tensor logits = nn::linear_lastdim(xn, w_lm, b_lm); // [B,1,V]
  return logits;
}

} // namespace nn::variants::kvcache
