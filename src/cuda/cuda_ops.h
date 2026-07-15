#pragma once

// High-level CUDA ops — thin wrappers around kernels with launch configs
// Each op takes DeviceTensor references and dispatches the right kernel

#include "cuda/cuda_tensor.h"

namespace nn::cuda::ops {

// out = ReLU(in)
void relu(const DeviceTensor& in, DeviceTensor& out);

// out = softmax(in) — row-wise
void softmax(const DeviceTensor& in, DeviceTensor& out);

// out = a + b
void add(const DeviceTensor& a, const DeviceTensor& b, DeviceTensor& out);

// out = alpha * in
void scale(const DeviceTensor& in, DeviceTensor& out, float alpha);

// C = A × B   (R×K = R×C @ C×K), naive kernel
void matmul(const DeviceTensor& A, const DeviceTensor& B, DeviceTensor& C);

// C = A × B   using tiled shared-memory kernel
void matmul_tiled(const DeviceTensor& A, const DeviceTensor& B, DeviceTensor& C);

} // namespace nn::cuda::ops
