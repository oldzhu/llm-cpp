#pragma once

// CUDA kernels — explicit per-element loops for learning
// Matmul  (for tiny models where the naive O(N^3) is educational)
// ReLU, Softmax (row-wise), Add, Scale, LayerNorm

#include <cuda_runtime.h>
#include <cstddef>

namespace nn::cuda::kernels {

// ---- Activation ----

// out[i] = max(0, in[i])
__global__ void relu_fwd(const float* __restrict__ in, float* __restrict__ out,
                         size_t n);

// ---- Softmax (row-wise) ----
// For input (R×C), softmax each row in-place or out-of-place
__global__ void softmax_fwd(const float* __restrict__ in, float* __restrict__ out,
                             size_t rows, size_t cols);

// ---- Element-wise ----

// out[i] = a[i] + b[i]
__global__ void add(const float* __restrict__ a, const float* __restrict__ b,
                    float* __restrict__ out, size_t n);

// out[i] = alpha * in[i]
__global__ void scale(const float* __restrict__ in, float* __restrict__ out,
                      float alpha, size_t n);

// ---- Matmul ----
// C(R×K) = A(R×C) × B(C×K)
// One CUDA thread per output element (naive, for learning)
__global__ void matmul_naive(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              size_t R, size_t C, size_t K);

// Shared-memory tiled matmul (more efficient, still educational)
__global__ void matmul_tiled(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              size_t R, size_t C, size_t K);
} // namespace nn::cuda::kernels
