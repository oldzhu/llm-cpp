// CUDA backend tests — device management, tensor ops, kernels

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

#include "cuda/cuda_device.h"
#include "cuda/cuda_tensor.h"
#include "cuda/cuda_ops.h"

#define EXPECT_TRUE(cond, msg) \
  do { if (!(cond)) { printf("[FAIL] %s\n", msg); return 1; } else { printf("[PASS] %s\n", msg); } } while(0)

#define EXPECT_NEAR(a, b, eps, msg) \
  do { if (std::fabs((a)-(b)) > (eps)) { printf("[FAIL] %s: %.4f vs %.4f\n", msg, (float)(a), (float)(b)); return 1; } else { printf("[PASS] %s\n", msg); } } while(0)

static int test_device_enumeration() {
  printf("[RUN ] CUDA device enumeration\n");
  auto devices = nn::cuda::list_devices();
  EXPECT_TRUE(!devices.empty(), "CUDA: at least one GPU found");
  EXPECT_TRUE(!devices[0].name.empty(), "CUDA: device has name");
  return 0;
}

static int test_device_alloc_free() {
  printf("[RUN ] CUDA device alloc/free\n");
  void* ptr = nn::cuda::device_alloc(1024);
  EXPECT_TRUE(ptr != nullptr, "CUDA: device_alloc returns non-null");
  nn::cuda::device_free(ptr);
  EXPECT_TRUE(true, "CUDA: device_free does not crash");
  return 0;
}

static int test_h2d_d2h() {
  printf("[RUN ] CUDA H→D→H round-trip\n");
  constexpr size_t N = 256;
  std::vector<float> host_in(N);
  std::vector<float> host_out(N, 0.0f);

  for (size_t i = 0; i < N; ++i) host_in[i] = static_cast<float>(i) * 0.1f;

  void* dev = nn::cuda::device_alloc(N * sizeof(float));
  nn::cuda::copy_to_device(dev, host_in.data(), N * sizeof(float));
  nn::cuda::copy_to_host(host_out.data(), dev, N * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    EXPECT_NEAR(host_out[i], host_in[i], 1e-5f, "CUDA: H→D→H correct");
  }

  nn::cuda::device_free(dev);
  return 0;
}

static int test_device_tensor_upload_download() {
  printf("[RUN ] CUDA DeviceTensor upload/download\n");
  constexpr size_t R = 4, C = 8;
  nn::cuda::DeviceTensor dt(R, C);
  EXPECT_TRUE(!dt.empty(), "CUDA: DeviceTensor not empty");
  EXPECT_TRUE(dt.rows() == R, "CUDA: DeviceTensor rows correct");
  EXPECT_TRUE(dt.cols() == C, "CUDA: DeviceTensor cols correct");

  std::vector<float> host_in(R * C);
  std::vector<float> host_out(R * C, 0.0f);
  for (size_t i = 0; i < R * C; ++i) host_in[i] = static_cast<float>(i);

  dt.upload(host_in.data(), R * C);
  dt.download(host_out.data(), R * C);

  for (size_t i = 0; i < R * C; ++i) {
    EXPECT_NEAR(host_out[i], host_in[i], 1e-5f, "CUDA: DeviceTensor R/T correct");
  }
  return 0;
}

static int test_relu() {
  printf("[RUN ] CUDA ReLU\n");
  constexpr size_t N = 1024;
  nn::cuda::DeviceTensor in(1, N), out(1, N);
  std::vector<float> host_in(N);
  std::vector<float> host_out(N, 0.0f);

  for (size_t i = 0; i < N; ++i)
    host_in[i] = (static_cast<float>(i) - 512.0f) * 0.1f; // mixes neg/pos

  in.upload(host_in.data(), N);
  nn::cuda::ops::relu(in, out);
  out.download(host_out.data(), N);

  for (size_t i = 0; i < N; ++i) {
    float expected = host_in[i] > 0.0f ? host_in[i] : 0.0f;
    EXPECT_NEAR(host_out[i], expected, 1e-5f, "CUDA: ReLU correct");
  }
  return 0;
}

static int test_softmax() {
  printf("[RUN ] CUDA Softmax\n");
  constexpr size_t R = 2, C = 4;
  nn::cuda::DeviceTensor in(R, C), out(R, C);
  std::vector<float> host_in = {
    1.0f, 2.0f, 3.0f, 4.0f,
    0.0f, 0.0f, 1.0f, 0.0f
  };
  std::vector<float> host_out(R * C, 0.0f);

  in.upload(host_in.data(), R * C);
  nn::cuda::ops::softmax(in, out);
  out.download(host_out.data(), R * C);

  // Row 0: softmax of (1,2,3,4) — should sum to 1
  float sum0 = host_out[0] + host_out[1] + host_out[2] + host_out[3];
  EXPECT_NEAR(sum0, 1.0f, 1e-5f, "CUDA: Softmax row 0 sums to 1");
  EXPECT_TRUE(host_out[3] > host_out[0], "CUDA: Softmax preserves order (4 > 1)");

  // Row 1: softmax of (0,0,1,0) — all equal except element 2 higher
  float sum1 = host_out[4] + host_out[5] + host_out[6] + host_out[7];
  EXPECT_NEAR(sum1, 1.0f, 1e-5f, "CUDA: Softmax row 1 sums to 1");
  return 0;
}

static int test_matmul_naive() {
  printf("[RUN ] CUDA Matmul naive\n");
  constexpr size_t R = 4, C = 3, K = 2;
  nn::cuda::DeviceTensor A(R, C), B(C, K), D(R, K);
  std::vector<float> host_A(R * C), host_B(C * K), host_D(R * K, 0.0f);

  // A = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
  for (size_t i = 0; i < R * C; ++i) host_A[i] = static_cast<float>(i + 1);
  // B = [[1,0],[0,1],[1,1]]
  host_B[0]=1; host_B[1]=0; host_B[2]=0; host_B[3]=1; host_B[4]=1; host_B[5]=1;

  A.upload(host_A.data(), R * C);
  B.upload(host_B.data(), C * K);
  nn::cuda::ops::matmul(A, B, D);
  D.download(host_D.data(), R * K);

  // C = A @ B = [[4,5],[10,11],[16,17],[22,23]]
  EXPECT_NEAR(host_D[0], 4.0f, 0.01f, "CUDA: matmul C[0,0]=4");
  EXPECT_NEAR(host_D[1], 5.0f, 0.01f, "CUDA: matmul C[0,1]=5");
  EXPECT_NEAR(host_D[6], 22.0f, 0.01f, "CUDA: matmul C[3,0]=22");
  EXPECT_NEAR(host_D[7], 23.0f, 0.01f, "CUDA: matmul C[3,1]=23");
  return 0;
}

static int test_matmul_tiled() {
  printf("[RUN ] CUDA Matmul tiled\n");
  constexpr size_t R = 4, C = 3, K = 2;
  nn::cuda::DeviceTensor A(R, C), B(C, K), D(R, K);
  std::vector<float> host_A(R * C), host_B(C * K), host_D(R * K, 0.0f);

  for (size_t i = 0; i < R * C; ++i) host_A[i] = static_cast<float>(i + 1);
  host_B[0]=1; host_B[1]=0; host_B[2]=0; host_B[3]=1; host_B[4]=1; host_B[5]=1;

  A.upload(host_A.data(), R * C);
  B.upload(host_B.data(), C * K);
  nn::cuda::ops::matmul_tiled(A, B, D);
  D.download(host_D.data(), R * K);

  EXPECT_NEAR(host_D[0], 4.0f, 0.01f, "CUDA: tiled matmul C[0,0]=4");
  EXPECT_NEAR(host_D[1], 5.0f, 0.01f, "CUDA: tiled matmul C[0,1]=5");
  return 0;
}

static int test_add_scale() {
  printf("[RUN ] CUDA Add + Scale\n");
  constexpr size_t N = 512;
  nn::cuda::DeviceTensor a(1, N), b(1, N), sum(1, N), scaled(1, N);
  std::vector<float> host_a(N), host_b(N), host_sum(N, 0.0f), host_scaled(N, 0.0f);

  for (size_t i = 0; i < N; ++i) {
    host_a[i] = static_cast<float>(i);
    host_b[i] = static_cast<float>(N - i);
  }

  a.upload(host_a.data(), N);
  b.upload(host_b.data(), N);
  nn::cuda::ops::add(a, b, sum);
  nn::cuda::ops::scale(sum, scaled, 0.5f);
  sum.download(host_sum.data(), N);
  scaled.download(host_scaled.data(), N);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_NEAR(host_sum[i], host_a[i] + host_b[i], 1e-4f, "CUDA: add correct");
    EXPECT_NEAR(host_scaled[i], (host_a[i] + host_b[i]) * 0.5f, 1e-4f, "CUDA: scale correct");
  }
  return 0;
}

int main() {
  printf("\n===== CUDA Backend Tests =====\n\n");

  int fails = 0;
  fails += test_device_enumeration();
  fails += test_device_alloc_free();
  fails += test_h2d_d2h();
  fails += test_device_tensor_upload_download();
  fails += test_relu();
  fails += test_softmax();
  fails += test_matmul_naive();
  fails += test_matmul_tiled();
  fails += test_add_scale();

  printf("\n===== %s =====\n", fails == 0 ? "ALL CUDA TESTS PASSED" : "SOME CUDA TESTS FAILED");
  return fails;
}
