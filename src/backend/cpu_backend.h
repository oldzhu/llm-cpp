#pragma once

#include "backend/backend.h"

namespace backend {

class CpuBackend final : public KernelBackend {
 public:
  void matmul2d_fwd(int m, int k, int n, const float* a_mk, const float* b_kn, float* out_mn) override;

  void matmul2d_bwd(int m,
                    int k,
                    int n,
                    const float* a_mk,
                    const float* b_kn,
                    const float* d_out_mn,
                    float* d_a_mk, // may be nullptr
                    float* d_b_kn  // may be nullptr
                    ) override;
};

} // namespace backend
