#include "ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "backend/registry.h"

namespace nn {

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

static void ensure_same_shape(const Tensor& a, const Tensor& b, const char* op) {
  if (a.shape != b.shape) {
    throw std::runtime_error(std::string(op) + ": shape mismatch");
  }
}

Tensor add(const Tensor& a, const Tensor& b) {
  ensure_same_shape(a, b, "add");
  Tensor out = Tensor::zeros(a.shape, want_grad(a) || want_grad(b));
  const std::size_t n = out.numel();
  for (std::size_t i = 0; i < n; ++i) {
    (*out.data)[i] = (*a.data)[i] + (*b.data)[i];
  }
  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {a, b};
    out.node->backward = [](Tensor& o) {
      const Tensor& pa = o.node->parents[0];
      const Tensor& pb = o.node->parents[1];
      const std::size_t n2 = o.numel();
      if (pa.requires_grad) {
        if (!pa.grad) throw std::runtime_error("add backward: missing grad for parent a");
        for (std::size_t i = 0; i < n2; ++i) (*pa.grad)[i] += (*o.grad)[i];
      }
      if (pb.requires_grad) {
        if (!pb.grad) throw std::runtime_error("add backward: missing grad for parent b");
        for (std::size_t i = 0; i < n2; ++i) (*pb.grad)[i] += (*o.grad)[i];
      }
    };
  }
  return out;
}

Tensor add_inplace(const Tensor& a, const Tensor& b) {
  // Same as add, but kept as an explicit op name so you can later swap with real inplace.
  return add(a, b);
}

Tensor mul(const Tensor& a, const Tensor& b) {
  ensure_same_shape(a, b, "mul");
  Tensor out = Tensor::zeros(a.shape, want_grad(a) || want_grad(b));
  const std::size_t n = out.numel();
  for (std::size_t i = 0; i < n; ++i) {
    (*out.data)[i] = (*a.data)[i] * (*b.data)[i];
  }
  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {a, b};
    out.node->backward = [](Tensor& o) {
      const Tensor& pa = o.node->parents[0];
      const Tensor& pb = o.node->parents[1];
      const std::size_t n2 = o.numel();
      if (pa.requires_grad) {
        for (std::size_t i = 0; i < n2; ++i) (*pa.grad)[i] += (*o.grad)[i] * (*pb.data)[i];
      }
      if (pb.requires_grad) {
        for (std::size_t i = 0; i < n2; ++i) (*pb.grad)[i] += (*o.grad)[i] * (*pa.data)[i];
      }
    };
  }
  return out;
}

Tensor mul_scalar(const Tensor& a, float s) {
  Tensor out = Tensor::zeros(a.shape, want_grad(a));
  const std::size_t n = out.numel();
  for (std::size_t i = 0; i < n; ++i) (*out.data)[i] = (*a.data)[i] * s;
  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {a};
    out.node->backward = [s](Tensor& o) {
      const Tensor& pa = o.node->parents[0];
      if (pa.requires_grad) {
        const std::size_t n2 = o.numel();
        for (std::size_t i = 0; i < n2; ++i) (*pa.grad)[i] += (*o.grad)[i] * s;
      }
    };
  }
  return out;
}

Tensor add_scalar(const Tensor& a, float s) {
  Tensor out = Tensor::zeros(a.shape, want_grad(a));
  const std::size_t n = out.numel();
  for (std::size_t i = 0; i < n; ++i) (*out.data)[i] = (*a.data)[i] + s;
  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {a};
    out.node->backward = [](Tensor& o) {
      const Tensor& pa = o.node->parents[0];
      if (pa.requires_grad) {
        const std::size_t n2 = o.numel();
        for (std::size_t i = 0; i < n2; ++i) (*pa.grad)[i] += (*o.grad)[i];
      }
    };
  }
  return out;
}

Tensor reshape(const Tensor& x, const std::vector<int>& new_shape) {
  const std::size_t n0 = x.numel();
  const std::size_t n1 = numel_of(new_shape);
  if (n0 != n1) throw std::runtime_error("reshape: numel mismatch");

  Tensor out;
  out.data = x.data; // view: share storage
  out.shape = new_shape;
  out.requires_grad = want_grad(x);
  if (out.requires_grad) {
    out.grad = std::make_shared<std::vector<float>>(n1, 0.0f);
    out.node = std::make_shared<Node>();
    out.node->parents = {x};
    out.node->backward = [](Tensor& o) {
      const Tensor& px = o.node->parents[0];
      if (!px.requires_grad) return;
      const std::size_t n = o.numel();
      for (std::size_t i = 0; i < n; ++i) {
        (*px.grad)[i] += (*o.grad)[i];
      }
    };
  }
  return out;
}

Tensor matmul2d(const Tensor& a, const Tensor& b) {
  if (a.shape.size() != 2 || b.shape.size() != 2) {
    throw std::runtime_error("matmul2d expects 2D tensors");
  }
  const int m = a.shape[0];
  const int k = a.shape[1];
  const int k2 = b.shape[0];
  const int n = b.shape[1];
  if (k != k2) throw std::runtime_error("matmul2d: inner dim mismatch");

  Tensor out = Tensor::zeros({m, n}, want_grad(a) || want_grad(b));

  backend::get().matmul2d_fwd(m, k, n, a.data->data(), b.data->data(), out.data->data());

  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {a, b};
    out.node->backward = [m, k, n](Tensor& o) {
      const Tensor& pa = o.node->parents[0];
      const Tensor& pb = o.node->parents[1];

      backend::get().matmul2d_bwd(m,
                                 k,
                                 n,
                                 pa.data->data(),
                                 pb.data->data(),
                                 o.grad->data(),
                                 pa.requires_grad ? pa.grad->data() : nullptr,
                                 pb.requires_grad ? pb.grad->data() : nullptr);
    };
  }

  return out;
}

Tensor bmm(const Tensor& a, const Tensor& b) {
  if (a.shape.size() != 3 || b.shape.size() != 3) {
    throw std::runtime_error("bmm expects 3D tensors");
  }
  const int B = a.shape[0];
  const int M = a.shape[1];
  const int K = a.shape[2];
  if (b.shape[0] != B || b.shape[1] != K) throw std::runtime_error("bmm: shape mismatch");
  const int N = b.shape[2];

  Tensor out = Tensor::zeros({B, M, N}, want_grad(a) || want_grad(b));
  for (int bb = 0; bb < B; ++bb) {
    const std::size_t a_off = static_cast<std::size_t>(bb) * M * K;
    const std::size_t b_off = static_cast<std::size_t>(bb) * K * N;
    const std::size_t o_off = static_cast<std::size_t>(bb) * M * N;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          sum += (*a.data)[a_off + static_cast<std::size_t>(i) * K + kk] *
                 (*b.data)[b_off + static_cast<std::size_t>(kk) * N + j];
        }
        (*out.data)[o_off + static_cast<std::size_t>(i) * N + j] = sum;
      }
    }
  }

  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {a, b};
    out.node->backward = [B, M, K, N](Tensor& o) {
      const Tensor& pa = o.node->parents[0];
      const Tensor& pb = o.node->parents[1];

      for (int bb = 0; bb < B; ++bb) {
        const std::size_t a_off = static_cast<std::size_t>(bb) * M * K;
        const std::size_t b_off = static_cast<std::size_t>(bb) * K * N;
        const std::size_t o_off = static_cast<std::size_t>(bb) * M * N;

        // dA = dO @ B^T
        if (pa.requires_grad) {
          for (int i = 0; i < M; ++i) {
            for (int kk = 0; kk < K; ++kk) {
              float sum = 0.0f;
              for (int j = 0; j < N; ++j) {
                sum += (*o.grad)[o_off + static_cast<std::size_t>(i) * N + j] *
                       (*pb.data)[b_off + static_cast<std::size_t>(kk) * N + j];
              }
              (*pa.grad)[a_off + static_cast<std::size_t>(i) * K + kk] += sum;
            }
          }
        }

        // dB = A^T @ dO
        if (pb.requires_grad) {
          for (int kk = 0; kk < K; ++kk) {
            for (int j = 0; j < N; ++j) {
              float sum = 0.0f;
              for (int i = 0; i < M; ++i) {
                sum += (*pa.data)[a_off + static_cast<std::size_t>(i) * K + kk] *
                       (*o.grad)[o_off + static_cast<std::size_t>(i) * N + j];
              }
              (*pb.grad)[b_off + static_cast<std::size_t>(kk) * N + j] += sum;
            }
          }
        }
      }
    };
  }

  return out;
}

Tensor gelu(const Tensor& x) {
  Tensor out = Tensor::zeros(x.shape, want_grad(x));
  const std::size_t n = out.numel();
  for (std::size_t i = 0; i < n; ++i) {
    const float v = (*x.data)[i];
    // tanh approximation
    const float c = 0.044715f;
    const float s = 0.7978845608f; // sqrt(2/pi)
    const float u = s * (v + c * v * v * v);
    const float t = std::tanh(u);
    (*out.data)[i] = 0.5f * v * (1.0f + t);
  }

  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {x};
    out.node->backward = [](Tensor& o) {
      const Tensor& px = o.node->parents[0];
      const std::size_t n2 = o.numel();
      for (std::size_t i = 0; i < n2; ++i) {
        const float v = (*px.data)[i];
        // derivative of gelu(tanh approx)
        const float c = 0.044715f;
        const float s = 0.7978845608f;
        const float u = s * (v + c * v * v * v);
        const float t = std::tanh(u);
        const float sech2 = 1.0f - t * t;
        const float du = s * (1.0f + 3.0f * c * v * v);
        const float d = 0.5f * (1.0f + t) + 0.5f * v * sech2 * du;
        (*px.grad)[i] += (*o.grad)[i] * d;
      }
    };
  }

  return out;
}

Tensor layernorm_lastdim(const Tensor& x, float eps) {
  // LayerNorm over the last dimension.
  // For each "row" (all dims except the last):
  //   mean = (1/D) * sum_i x_i
  //   var  = (1/D) * sum_i (x_i - mean)^2
  //   y_i  = (x_i - mean) / sqrt(var + eps)
  // This implementation is the *no affine* variant (no gamma/beta).
  if (x.shape.empty()) throw std::runtime_error("layernorm: empty shape");
  const int D = x.shape.back();
  const std::size_t outer = x.numel() / static_cast<std::size_t>(D);

  Tensor out = Tensor::zeros(x.shape, want_grad(x));
  std::vector<float> mean(outer, 0.0f);
  std::vector<float> invstd(outer, 0.0f);

  for (std::size_t o = 0; o < outer; ++o) {
    float m = 0.0f;
    for (int i = 0; i < D; ++i) m += (*x.data)[o * D + static_cast<std::size_t>(i)];
    m /= static_cast<float>(D);
    mean[o] = m;

    float v = 0.0f;
    for (int i = 0; i < D; ++i) {
      const float d = (*x.data)[o * D + static_cast<std::size_t>(i)] - m;
      v += d * d;
    }
    v /= static_cast<float>(D);
    invstd[o] = 1.0f / std::sqrt(v + eps);

    for (int i = 0; i < D; ++i) {
      const float xn = ((*x.data)[o * D + static_cast<std::size_t>(i)] - m) * invstd[o];
      (*out.data)[o * D + static_cast<std::size_t>(i)] = xn;
    }
  }

  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {x};
    out.node->backward = [D, mean = std::move(mean), invstd = std::move(invstd)](Tensor& o) mutable {
      const Tensor& px = o.node->parents[0];
      const std::size_t outer2 = o.numel() / static_cast<std::size_t>(D);
      for (std::size_t outi = 0; outi < outer2; ++outi) {
        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        for (int i = 0; i < D; ++i) {
          const float dy = (*o.grad)[outi * D + static_cast<std::size_t>(i)];
          sum_dy += dy;
          const float xhat = (*o.data)[outi * D + static_cast<std::size_t>(i)];
          sum_dy_xhat += dy * xhat;
        }
        for (int i = 0; i < D; ++i) {
          const float dy = (*o.grad)[outi * D + static_cast<std::size_t>(i)];
          const float xhat = (*o.data)[outi * D + static_cast<std::size_t>(i)];
          const float dx = (dy - sum_dy / static_cast<float>(D) - xhat * (sum_dy_xhat / static_cast<float>(D))) * invstd[outi];
          (*px.grad)[outi * D + static_cast<std::size_t>(i)] += dx;
        }
      }
    };
  }

  return out;
}

Tensor softmax_lastdim(const Tensor& x) {
  // Softmax over last dimension (stable):
  //   y_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
  // Backward uses the classic Jacobian-vector product for softmax:
  //   dL/dx = y * (dL/dy - dot(dL/dy, y))
  if (x.shape.empty()) throw std::runtime_error("softmax: empty shape");
  const int D = x.shape.back();
  const std::size_t outer = x.numel() / static_cast<std::size_t>(D);

  Tensor out = Tensor::zeros(x.shape, want_grad(x));
  for (std::size_t o = 0; o < outer; ++o) {
    float mx = -1e30f;
    for (int i = 0; i < D; ++i) mx = std::max(mx, (*x.data)[o * D + static_cast<std::size_t>(i)]);
    float denom = 0.0f;
    for (int i = 0; i < D; ++i) {
      const float e = std::exp((*x.data)[o * D + static_cast<std::size_t>(i)] - mx);
      (*out.data)[o * D + static_cast<std::size_t>(i)] = e;
      denom += e;
    }
    const float inv = 1.0f / denom;
    for (int i = 0; i < D; ++i) {
      (*out.data)[o * D + static_cast<std::size_t>(i)] *= inv;
    }
  }

  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {x};
    out.node->backward = [D](Tensor& o) {
      const Tensor& px = o.node->parents[0];
      const std::size_t outer2 = o.numel() / static_cast<std::size_t>(D);
      for (std::size_t outi = 0; outi < outer2; ++outi) {
        float dot = 0.0f;
        for (int i = 0; i < D; ++i) {
          dot += (*o.grad)[outi * D + static_cast<std::size_t>(i)] * (*o.data)[outi * D + static_cast<std::size_t>(i)];
        }
        for (int i = 0; i < D; ++i) {
          const float y = (*o.data)[outi * D + static_cast<std::size_t>(i)];
          (*px.grad)[outi * D + static_cast<std::size_t>(i)] += y * ((*o.grad)[outi * D + static_cast<std::size_t>(i)] - dot);
        }
      }
    };
  }

  return out;
}

Tensor linear_lastdim(const Tensor& x, const Tensor& w, const Tensor& b) {
  // Linear layer on the last dimension.
  // x: [B,T,Cin], w:[Cin,Cout], b:[Cout]
  // For each (b,t): y[b,t,:] = x[b,t,:] @ w + b
  // x: [B,T,Cin], w:[Cin,Cout], b:[Cout]
  if (x.shape.size() != 3 || w.shape.size() != 2 || b.shape.size() != 1) {
    throw std::runtime_error("linear_lastdim: invalid shapes");
  }
  const int B = x.shape[0];
  const int T = x.shape[1];
  const int Cin = x.shape[2];
  if (w.shape[0] != Cin) throw std::runtime_error("linear_lastdim: Cin mismatch");
  const int Cout = w.shape[1];
  if (b.shape[0] != Cout) throw std::runtime_error("linear_lastdim: bias mismatch");

  // Flatten to 2D: [B*T, Cin]
  Tensor x2 = reshape(x, {B * T, Cin});
  Tensor y2 = matmul2d(x2, w); // [B*T, Cout]

  // Add bias
  Tensor out = Tensor::zeros({B, T, Cout}, want_grad(y2) || want_grad(b));
  for (int n = 0; n < B * T; ++n) {
    for (int j = 0; j < Cout; ++j) {
      (*out.data)[static_cast<std::size_t>(n) * Cout + j] = (*y2.data)[static_cast<std::size_t>(n) * Cout + j] + (*b.data)[j];
    }
  }

  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {y2, b};
    out.node->backward = [B, T, Cout](Tensor& o) {
      Tensor& py2 = o.node->parents[0];
      Tensor& pb = o.node->parents[1];
      const int N = B * T;

      if (py2.requires_grad) {
        for (int n = 0; n < N; ++n) {
          for (int j = 0; j < Cout; ++j) {
            (*py2.grad)[static_cast<std::size_t>(n) * Cout + j] += (*o.grad)[static_cast<std::size_t>(n) * Cout + j];
          }
        }
      }
      if (pb.requires_grad) {
        for (int j = 0; j < Cout; ++j) {
          float sum = 0.0f;
          for (int n = 0; n < N; ++n) sum += (*o.grad)[static_cast<std::size_t>(n) * Cout + j];
          (*pb.grad)[j] += sum;
        }
      }
    };
  }

  return out;
}

Tensor embedding(const Tensor& weight_vocab_by_dim, const std::vector<std::int32_t>& idx_bt, int B, int T) {
  // Embedding lookup.
  // weight: [V,C], idx: [B,T] token ids -> out: [B,T,C]
  // out[b,t,:] = weight[idx[b,t],:]
  if (weight_vocab_by_dim.shape.size() != 2) throw std::runtime_error("embedding: weight must be 2D");
  const int V = weight_vocab_by_dim.shape[0];
  const int C = weight_vocab_by_dim.shape[1];
  if (static_cast<int>(idx_bt.size()) != B * T) throw std::runtime_error("embedding: idx size mismatch");

  Tensor out = Tensor::zeros({B, T, C}, want_grad(weight_vocab_by_dim));
  for (int n = 0; n < B * T; ++n) {
    const int token = idx_bt[static_cast<std::size_t>(n)];
    if (token < 0 || token >= V) throw std::runtime_error("embedding: token out of range");
    const std::size_t w_off = static_cast<std::size_t>(token) * C;
    const std::size_t o_off = static_cast<std::size_t>(n) * C;
    for (int c = 0; c < C; ++c) {
      (*out.data)[o_off + c] = (*weight_vocab_by_dim.data)[w_off + c];
    }
  }

  if (out.requires_grad) {
    out.node = std::make_shared<Node>();
    out.node->parents = {weight_vocab_by_dim};
    out.node->backward = [idx_bt, B, T, V, C](Tensor& o) {
      Tensor& w = o.node->parents[0];
      if (!w.requires_grad) return;
      for (int n = 0; n < B * T; ++n) {
        const int token = idx_bt[static_cast<std::size_t>(n)];
        const std::size_t w_off = static_cast<std::size_t>(token) * C;
        const std::size_t o_off = static_cast<std::size_t>(n) * C;
        for (int c = 0; c < C; ++c) {
          (*w.grad)[w_off + c] += (*o.grad)[o_off + c];
        }
      }
    };
  }

  return out;
}

Tensor self_attention_1h(const Tensor& x,
                         const Tensor& w_qkv,
                         const Tensor& b_qkv,
                         const Tensor& w_proj,
                         const Tensor& b_proj) {
  // Causal self-attention, 1 head.
  // Shapes:
  //   x: [B,T,C]
  //   w_qkv: [C,3C], b_qkv: [3C]
  //   w_proj: [C,C], b_proj: [C]
  // Math:
  //   [Q,K,V] = x W_qkv + b_qkv
  //   S[i,j] = (Q[i]·K[j]) / sqrt(C) + mask(j>i -> -inf)
  //   P[i,:] = softmax(S[i,:])
  //   Y[i]   = sum_j P[i,j] V[j]
  //   out    = Y W_proj + b_proj
  // x: [B,T,C]
  if (x.shape.size() != 3) throw std::runtime_error("attn: x must be [B,T,C]");
  const int B = x.shape[0];
  const int T = x.shape[1];
  const int C = x.shape[2];
  if (w_qkv.shape != std::vector<int>({C, 3 * C})) throw std::runtime_error("attn: w_qkv shape mismatch");
  if (b_qkv.shape != std::vector<int>({3 * C})) throw std::runtime_error("attn: b_qkv shape mismatch");
  if (w_proj.shape != std::vector<int>({C, C})) throw std::runtime_error("attn: w_proj shape mismatch");
  if (b_proj.shape != std::vector<int>({C})) throw std::runtime_error("attn: b_proj shape mismatch");

  // qkv = linear(x)
  Tensor qkv = linear_lastdim(x, w_qkv, b_qkv); // [B,T,3C]

  Tensor q = slice_lastdim_copy(qkv, 0, C);
  Tensor k = slice_lastdim_copy(qkv, C, C);
  Tensor v = slice_lastdim_copy(qkv, 2 * C, C);

  // For now, implement attention without split autograd linkage; we will route gradients via an explicit node below.
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
                if (j > i) continue; // masked
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
              for (int i = j; i < T; ++i) { // only where j<=i contributes
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

  Tensor probs = softmax_lastdim(scores); // [B,T,T]

  // out = probs @ v => [B,T,C]
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

  // Attach backward for att with respect to probs and v
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

  Tensor proj = linear_lastdim(att, w_proj, b_proj);
  return proj;
}

Tensor cross_entropy(const Tensor& logits_nv, const std::vector<std::int32_t>& targets_n) {
  // Mean cross-entropy loss for a batch of logits.
  // logits: [N,V], targets: [N]
  // For each row n:
  //   p = softmax(logits[n,:])
  //   loss_n = -log(p[y_n])
  // Return: mean(loss_n)
  if (logits_nv.shape.size() != 2) throw std::runtime_error("cross_entropy: logits must be [N,V]");
  const int N = logits_nv.shape[0];
  const int V = logits_nv.shape[1];
  if (static_cast<int>(targets_n.size()) != N) throw std::runtime_error("cross_entropy: targets size mismatch");

  Tensor loss = Tensor::zeros({1}, want_grad(logits_nv));

  // Forward: compute mean negative log likelihood.
  std::vector<float> probs(N * V);
  float total = 0.0f;
  for (int n = 0; n < N; ++n) {
    float mx = -1e30f;
    for (int v = 0; v < V; ++v) mx = std::max(mx, (*logits_nv.data)[static_cast<std::size_t>(n) * V + v]);
    float denom = 0.0f;
    for (int v = 0; v < V; ++v) {
      const float e = std::exp((*logits_nv.data)[static_cast<std::size_t>(n) * V + v] - mx);
      probs[static_cast<std::size_t>(n) * V + v] = e;
      denom += e;
    }
    const float inv = 1.0f / denom;
    for (int v = 0; v < V; ++v) probs[static_cast<std::size_t>(n) * V + v] *= inv;

    const int y = targets_n[static_cast<std::size_t>(n)];
    if (y < 0 || y >= V) throw std::runtime_error("cross_entropy: target out of range");
    const float p = std::max(probs[static_cast<std::size_t>(n) * V + y], 1e-12f);
    total += -std::log(p);
  }
  total /= static_cast<float>(N);
  (*loss.data)[0] = total;

  if (loss.requires_grad) {
    loss.node = std::make_shared<Node>();
    loss.node->parents = {logits_nv};
    loss.node->backward = [N, V, probs = std::move(probs), targets_n](Tensor& o) mutable {
      Tensor& logits = o.node->parents[0];
      const float scale = (*o.grad)[0] / static_cast<float>(N);
      for (int n = 0; n < N; ++n) {
        for (int v = 0; v < V; ++v) {
          float g = probs[static_cast<std::size_t>(n) * V + v];
          const int y = targets_n[static_cast<std::size_t>(n)];
          if (v == y) g -= 1.0f;
          (*logits.grad)[static_cast<std::size_t>(n) * V + v] += g * scale;
        }
      }
    };
  }

  return loss;
}

} // namespace nn
