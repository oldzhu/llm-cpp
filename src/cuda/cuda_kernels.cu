#include "cuda/cuda_kernels.h"
#include <cmath>
#include <cfloat>

namespace nn::cuda::kernels {

__global__ void relu_fwd(const float* __restrict__ in, float* __restrict__ out,
                         size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = fmaxf(0.0f, in[i]);
  }
}

__global__ void softmax_fwd(const float* __restrict__ in, float* __restrict__ out,
                             size_t rows, size_t cols) {
  // One thread per row
  size_t r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;

  const float* row_in = in + r * cols;
  float* row_out = out + r * cols;

  // Find max for numerical stability
  float max_val = -FLT_MAX;
  for (size_t c = 0; c < cols; ++c) {
    max_val = fmaxf(max_val, row_in[c]);
  }

  // Compute exp sum
  float sum = 0.0f;
  for (size_t c = 0; c < cols; ++c) {
    float v = expf(row_in[c] - max_val);
    row_out[c] = v;
    sum += v;
  }

  // Normalize
  for (size_t c = 0; c < cols; ++c) {
    row_out[c] /= sum;
  }
}

__global__ void add(const float* __restrict__ a, const float* __restrict__ b,
                    float* __restrict__ out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

__global__ void scale(const float* __restrict__ in, float* __restrict__ out,
                      float alpha, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = alpha * in[i];
  }
}

__global__ void matmul_naive(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              size_t R, size_t C, size_t K) {
  // Thread (row, col) computes C[row][col]
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < R && col < K) {
    float sum = 0.0f;
    for (size_t k = 0; k < C; ++k) {
      sum += A[row * C + k] * B[k * K + col];
    }
    C[row * K + col] = sum;
  }
}

#define TILE_SIZE 16

__global__ void matmul_tiled(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              size_t R, size_t C, size_t K) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
  size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;
  for (size_t t = 0; t < (C + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load tile from A
    size_t a_col = t * TILE_SIZE + threadIdx.x;
    if (row < R && a_col < C)
      tileA[threadIdx.y][threadIdx.x] = A[row * C + a_col];
    else
      tileA[threadIdx.y][threadIdx.x] = 0.0f;

    // Load tile from B
    size_t b_row = t * TILE_SIZE + threadIdx.y;
    if (b_row < C && col < K)
      tileB[threadIdx.y][threadIdx.x] = B[b_row * K + col];
    else
      tileB[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Compute partial dot product
    for (size_t k = 0; k < TILE_SIZE; ++k) {
      sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < R && col < K) {
    C[row * K + col] = sum;
  }
}

} // namespace nn::cuda::kernels
