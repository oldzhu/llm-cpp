#include "backend/cpu_backend.h"

namespace backend {

void CpuBackend::matmul2d_fwd(int m, int k, int n, const float* a_mk, const float* b_kn, float* out_mn) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        sum += a_mk[i * k + kk] * b_kn[kk * n + j];
      }
      out_mn[i * n + j] = sum;
    }
  }
}

void CpuBackend::matmul2d_bwd(int m,
                             int k,
                             int n,
                             const float* a_mk,
                             const float* b_kn,
                             const float* d_out_mn,
                             float* d_a_mk,
                             float* d_b_kn) {
  // dA += dO @ B^T
  if (d_a_mk) {
    for (int i = 0; i < m; ++i) {
      for (int kk = 0; kk < k; ++kk) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
          sum += d_out_mn[i * n + j] * b_kn[kk * n + j];
        }
        d_a_mk[i * k + kk] += sum;
      }
    }
  }

  // dB += A^T @ dO
  if (d_b_kn) {
    for (int kk = 0; kk < k; ++kk) {
      for (int j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < m; ++i) {
          sum += a_mk[i * k + kk] * d_out_mn[i * n + j];
        }
        d_b_kn[kk * n + j] += sum;
      }
    }
  }
}

} // namespace backend
