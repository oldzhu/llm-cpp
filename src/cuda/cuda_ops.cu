#include "cuda/cuda_ops.h"
#include "cuda/cuda_kernels.h"
#include <algorithm>

namespace nn::cuda::ops {

void relu(const DeviceTensor& in, DeviceTensor& out) {
  size_t n = in.size();
  int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  kernels::relu_fwd<<<blocks, threads>>>(in.data(), out.data(), n);
  cudaDeviceSynchronize();
}

void softmax(const DeviceTensor& in, DeviceTensor& out) {
  size_t rows = in.rows(), cols = in.cols();
  int threads = 256;
  int blocks = static_cast<int>((rows + threads - 1) / threads);
  kernels::softmax_fwd<<<blocks, threads>>>(in.data(), out.data(), rows, cols);
  cudaDeviceSynchronize();
}

void add(const DeviceTensor& a, const DeviceTensor& b, DeviceTensor& out) {
  size_t n = a.size();
  int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  kernels::add<<<blocks, threads>>>(a.data(), b.data(), out.data(), n);
  cudaDeviceSynchronize();
}

void scale(const DeviceTensor& in, DeviceTensor& out, float alpha) {
  size_t n = in.size();
  int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  kernels::scale<<<blocks, threads>>>(in.data(), out.data(), alpha, n);
  cudaDeviceSynchronize();
}

void matmul(const DeviceTensor& A, const DeviceTensor& B, DeviceTensor& C) {
  size_t R = A.rows(), K_common = A.cols(), K = B.cols();
  dim3 threads(16, 16);
  dim3 blocks(
    static_cast<unsigned int>((K + 15) / 16),
    static_cast<unsigned int>((R + 15) / 16)
  );
  kernels::matmul_naive<<<blocks, threads>>>(A.data(), B.data(), C.data(),
                                              R, K_common, K);
  cudaDeviceSynchronize();
}

void matmul_tiled(const DeviceTensor& A, const DeviceTensor& B, DeviceTensor& C) {
  size_t R = A.rows(), K_common = A.cols(), K = B.cols();
  dim3 threads(16, 16);
  dim3 blocks(
    static_cast<unsigned int>((K + 15) / 16),
    static_cast<unsigned int>((R + 15) / 16)
  );
  kernels::matmul_tiled<<<blocks, threads>>>(A.data(), B.data(), C.data(),
                                              R, K_common, K);
  cudaDeviceSynchronize();
}

} // namespace nn::cuda::ops
