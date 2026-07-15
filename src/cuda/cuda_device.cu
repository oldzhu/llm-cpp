#include "cuda/cuda_device.h"

namespace nn::cuda {

std::vector<DeviceInfo> list_devices() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) return {};

  std::vector<DeviceInfo> devices;
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, i);
    if (err != cudaSuccess) continue;

    DeviceInfo info;
    info.id = i;
    info.name = std::string(prop.name);
    info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    devices.push_back(info);
  }
  return devices;
}

bool select_device(int device_id) {
  return cudaSetDevice(device_id) == cudaSuccess;
}

DeviceInfo current_device() {
  int id = 0;
  cudaGetDevice(&id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, id);

  DeviceInfo info;
  info.id = id;
  info.name = std::string(prop.name);
  info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
  info.compute_capability_major = prop.major;
  info.compute_capability_minor = prop.minor;
  info.multiprocessor_count = prop.multiProcessorCount;
  return info;
}

void* device_alloc(size_t bytes) {
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA malloc failed: " +
                             std::string(cudaGetErrorString(err)));
  }
  return ptr;
}

void device_free(void* ptr) {
  if (ptr) cudaFree(ptr);
}

void copy_to_device(void* dst_device, const void* src_host, size_t bytes) {
  cudaError_t err = cudaMemcpy(dst_device, src_host, bytes,
                                cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA H→D copy failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void copy_to_host(void* dst_host, const void* src_device, size_t bytes) {
  cudaError_t err = cudaMemcpy(dst_host, src_device, bytes,
                                cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA D→H copy failed: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void sync() {
  cudaDeviceSynchronize();
}

DeviceGuard::DeviceGuard(int device_id) {
  cudaGetDevice(&previous_device_);
  if (device_id != previous_device_) {
    cudaSetDevice(device_id);
  }
}

DeviceGuard::~DeviceGuard() {
  cudaSetDevice(previous_device_);
}

} // namespace nn::cuda
