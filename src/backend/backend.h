#pragma once

// Backend boundary.
//
// Intent:
// - Keep model math/op semantics stable in `nn::*`.
// - Route hot kernels (matmul, softmax, attention, etc.) through a backend
//   without rewriting model code.
//
// Contract notes:
// - All tensors are row-major contiguous float buffers (today).
// - Backward kernels must *accumulate* gradients into dA/dB (i.e., `+=`).

namespace backend {

enum class DeviceType {
  Cpu,
  Cuda,
  Hip,
};

struct Device {
  DeviceType type = DeviceType::Cpu;
  int device_id = 0;
};

struct KernelBackend {
  virtual ~KernelBackend() = default;

  // C = A @ B
  // A: [m,k], B: [k,n], C: [m,n]
  virtual void matmul2d_fwd(int m, int k, int n, const float* a_mk, const float* b_kn, float* out_mn) = 0;

  // Backward accumulation:
  //   if (d_a_mk) dA += dC @ B^T
  //   if (d_b_kn) dB += A^T @ dC
  virtual void matmul2d_bwd(int m,
                            int k,
                            int n,
                            const float* a_mk,
                            const float* b_kn,
                            const float* d_out_mn,
                            float* d_a_mk,
                            float* d_b_kn) = 0;
};

} // namespace backend
