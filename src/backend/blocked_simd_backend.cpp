#include "backend/blocked_simd_backend.h"

#include <algorithm>
#include <cstring>
#include <vector>

#if defined(_MSC_VER) || defined(__SSE__)
#include <immintrin.h>
#define HAVE_AVX 1
#else
#define HAVE_AVX 0
#endif

namespace backend {

static constexpr int BLOCK_M = 64;
static constexpr int BLOCK_N = 64;
static constexpr int BLOCK_K = 64;

// Naive matmul (for small dimensions or fallback)
static void naive_matmul_fwd(int m, int k, int n, const float* a, const float* b, float* c) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        sum += a[i * k + kk] * b[kk * n + j];
      }
      c[i * n + j] = sum;
    }
  }
}

// Pack a panel of A [bm, bk] into a local buffer, row-major
static void pack_a(int bm, int bk, int K, const float* a, int a_row, int a_col, float* packed) {
  for (int i = 0; i < bm; ++i) {
    const float* src = a + (a_row + i) * K + a_col;
    float* dst = packed + i * bk;
    for (int kk = 0; kk < bk; ++kk) {
      dst[kk] = src[kk];
    }
  }
}

// Pack a panel of B [bk, bn] into a local buffer, row-major
static void pack_b(int bk, int bn, int N, const float* b, int b_row, int b_col, float* packed) {
  for (int kk = 0; kk < bk; ++kk) {
    const float* src = b + (b_row + kk) * N + b_col;
    float* dst = packed + kk * bn;
    for (int j = 0; j < bn; ++j) {
      dst[j] = src[j];
    }
  }
}

// Inner kernel: C[bm, bn] += A_packed[bm, bk] @ B_packed[bk, bn]
// AVX: 8-wide FMA
static void micro_kernel(int bm, int bn, int bk, const float* a_pack, const float* b_pack, float* c, int ldc) {
#if HAVE_AVX
  for (int i = 0; i < bm; ++i) {
    for (int j = 0; j < bn; j += 8) {
      // Handle full 8-wide or partial remainder
      const int rem = std::min(8, bn - j);
      if (rem == 8) {
        __m256 cval = _mm256_loadu_ps(c + i * ldc + j);
        for (int kk = 0; kk < bk; ++kk) {
          __m256 aval = _mm256_set1_ps(a_pack[i * bk + kk]);
          __m256 bval = _mm256_loadu_ps(b_pack + kk * bn + j);
          cval = _mm256_fmadd_ps(aval, bval, cval);
        }
        _mm256_storeu_ps(c + i * ldc + j, cval);
      } else {
        // Remainder: scalar loop
        for (int kk = 0; kk < bk; ++kk) {
          const float aval = a_pack[i * bk + kk];
          for (int r = 0; r < rem; ++r) {
            c[i * ldc + j + r] += aval * b_pack[kk * bn + j + r];
          }
        }
      }
    }
  }
#else
  // Fallback: scalar loop
  for (int i = 0; i < bm; ++i) {
    for (int j = 0; j < bn; ++j) {
      float sum = 0.0f;
      for (int kk = 0; kk < bk; ++kk) {
        sum += a_pack[i * bk + kk] * b_pack[kk * bn + j];
      }
      c[i * ldc + j] += sum;
    }
  }
#endif
}

void BlockedSimdCpuBackend::matmul2d_fwd(int m, int k, int n, const float* a, const float* b, float* c) {
  // Zero output
  std::memset(c, 0, static_cast<std::size_t>(m) * n * sizeof(float));

  // Allocate pack buffers once
  std::vector<float> a_pack(static_cast<std::size_t>(BLOCK_M) * BLOCK_K);
  std::vector<float> b_pack(static_cast<std::size_t>(BLOCK_K) * BLOCK_N);

  for (int mb = 0; mb < m; mb += BLOCK_M) {
    const int bm = std::min(BLOCK_M, m - mb);

    for (int nb = 0; nb < n; nb += BLOCK_N) {
      const int bn = std::min(BLOCK_N, n - nb);

      for (int kb = 0; kb < k; kb += BLOCK_K) {
        const int bk = std::min(BLOCK_K, k - kb);

        pack_a(bm, bk, k, a, mb, kb, a_pack.data());
        pack_b(bk, bn, n, b, kb, nb, b_pack.data());

        micro_kernel(bm, bn, bk, a_pack.data(), b_pack.data(), c + mb * n + nb, n);
      }
    }
  }
}

void BlockedSimdCpuBackend::matmul2d_bwd(int m,
                                         int k,
                                         int n,
                                         const float* a_mk,
                                         const float* b_kn,
                                         const float* d_out_mn,
                                         float* d_a_mk,
                                         float* d_b_kn) {
  // dA += dC @ B^T    (shape: [m,k] += [m,n] @ [n,k])
  if (d_a_mk != nullptr) {
    // Allocate pack buffers
    std::vector<float> dc_pack(static_cast<std::size_t>(BLOCK_M) * BLOCK_N);
    std::vector<float> bt_pack(static_cast<std::size_t>(BLOCK_N) * BLOCK_K);

    for (int mb = 0; mb < m; mb += BLOCK_M) {
      const int bm = std::min(BLOCK_M, m - mb);
      for (int kb = 0; kb < k; kb += BLOCK_K) {
        const int bk = std::min(BLOCK_K, k - kb);
        for (int nb = 0; nb < n; nb += BLOCK_N) {
          const int bn = std::min(BLOCK_N, n - nb);

          // Pack dC panel [bm, bn]
          for (int i = 0; i < bm; ++i) {
            std::memcpy(dc_pack.data() + i * bn, d_out_mn + (mb + i) * n + nb,
                        static_cast<std::size_t>(bn) * sizeof(float));
          }
          // Pack B^T panel [bn, bk] = B[kb:kb+bk, nb:nb+bn]^T → row-major [bn, bk]
          for (int j = 0; j < bn; ++j) {
            for (int kk = 0; kk < bk; ++kk) {
              bt_pack[j * bk + kk] = b_kn[(kb + kk) * n + (nb + j)];
            }
          }

          micro_kernel(bm, bk, bn, dc_pack.data(), bt_pack.data(), d_a_mk + mb * k + kb, k);
        }
      }
    }
  }

  // dB += A^T @ dC    (shape: [k,n] += [k,m] @ [m,n])
  if (d_b_kn != nullptr) {
    std::vector<float> at_pack(static_cast<std::size_t>(BLOCK_K) * BLOCK_M);
    std::vector<float> dc_pack2(static_cast<std::size_t>(BLOCK_M) * BLOCK_N);

    for (int kb = 0; kb < k; kb += BLOCK_K) {
      const int bk = std::min(BLOCK_K, k - kb);
      for (int nb = 0; nb < n; nb += BLOCK_N) {
        const int bn = std::min(BLOCK_N, n - nb);
        for (int mb = 0; mb < m; mb += BLOCK_M) {
          const int bm = std::min(BLOCK_M, m - mb);

          // Pack A^T panel [bk, bm] = A[mb:mb+bm, kb:kb+bk]^T
          for (int kk = 0; kk < bk; ++kk) {
            for (int i = 0; i < bm; ++i) {
              at_pack[kk * bm + i] = a_mk[(mb + i) * k + (kb + kk)];
            }
          }
          // Pack dC panel [bm, bn]
          for (int i = 0; i < bm; ++i) {
            std::memcpy(dc_pack2.data() + i * bn, d_out_mn + (mb + i) * n + nb,
                        static_cast<std::size_t>(bn) * sizeof(float));
          }

          micro_kernel(bk, bn, bm, at_pack.data(), dc_pack2.data(), d_b_kn + kb * n + nb, n);
        }
      }
    }
  }
}

} // namespace backend
