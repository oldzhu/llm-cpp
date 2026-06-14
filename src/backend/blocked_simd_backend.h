#pragma once

#include "backend/backend.h"

namespace backend {

// Blocked matmul with AVX SIMD inner kernel.
// Teaching-first: explicit blocking + vectorization.
//
// Matmul strategy:
//   C[M,N] = A[M,K] @ B[K,N]
//   1. Block outer dimensions: tile M and N into BLOCK_M × BLOCK_N panels
//   2. For each (m_block, n_block), accumulate K dimension in BLOCK_K steps
//   3. Inner kernel: 8-wide AVX FMA over a micro-tile of A and B
//
// Backward:
//   dA[M,K] += dC[M,N] @ B^T[N,K]   → same blocked pattern, transposed
//   dB[K,N] += A^T[K,M] @ dC[M,N]   → same blocked pattern, transposed

class BlockedSimdCpuBackend final : public KernelBackend {
 public:
  void matmul2d_fwd(int m, int k, int n, const float* a_mk, const float* b_kn, float* out_mn) override;
  void matmul2d_bwd(int m,
                    int k,
                    int n,
                    const float* a_mk,
                    const float* b_kn,
                    const float* d_out_mn,
                    float* d_a_mk,
                    float* d_b_kn) override;
};

} // namespace backend
