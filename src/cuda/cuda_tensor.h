#pragma once

// Device tensor — a 2D float matrix on GPU with RAII memory management
// Mirrors the host-side nn::Matrix for seamless host↔device interchange

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

namespace nn::cuda {

class DeviceTensor {
 public:
  DeviceTensor() = default;

  DeviceTensor(size_t rows, size_t cols)
      : rows_(rows), cols_(cols), size_(rows * cols) {
    cudaMalloc(&data_, size_ * sizeof(float));
  }

  ~DeviceTensor() {
    if (data_) cudaFree(data_);
  }

  // Move-only
  DeviceTensor(DeviceTensor&& other) noexcept
      : data_(other.data_), rows_(other.rows_),
        cols_(other.cols_), size_(other.size_) {
    other.data_ = nullptr;
    other.rows_ = 0; other.cols_ = 0; other.size_ = 0;
  }

  DeviceTensor& operator=(DeviceTensor&& other) noexcept {
    if (this != &other) {
      if (data_) cudaFree(data_);
      data_ = other.data_;
      rows_ = other.rows_;
      cols_ = other.cols_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.rows_ = 0; other.cols_ = 0; other.size_ = 0;
    }
    return *this;
  }

  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;

  // Upload from host float array
  void upload(const float* host_data, size_t n) {
    if (n != size_) throw std::runtime_error("DeviceTensor::upload size mismatch");
    cudaMemcpy(data_, host_data, n * sizeof(float), cudaMemcpyHostToDevice);
  }

  // Download to host float array
  void download(float* host_data, size_t n) const {
    if (n != size_) throw std::runtime_error("DeviceTensor::download size mismatch");
    cudaMemcpy(host_data, data_, n * sizeof(float), cudaMemcpyDeviceToHost);
  }

  // Zero-fill on device
  void zero() {
    cudaMemset(data_, 0, size_ * sizeof(float));
  }

  float* data()   { return data_; }
  const float* data() const { return data_; }
  size_t rows()   const { return rows_; }
  size_t cols()   const { return cols_; }
  size_t size()   const { return size_; }
  bool   empty()  const { return size_ == 0; }

 private:
  float* data_ = nullptr;
  size_t rows_ = 0;
  size_t cols_ = 0;
  size_t size_ = 0;
};

} // namespace nn::cuda
