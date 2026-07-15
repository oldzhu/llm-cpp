#pragma once

// CUDA device management — detection, selection, memory allocation/free
// Handles multi-GPU enumeration and context management

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <stdexcept>

namespace nn::cuda {

struct DeviceInfo {
  int id;
  std::string name;
  size_t total_memory_mb;
  int compute_capability_major;
  int compute_capability_minor;
  int multiprocessor_count;
};

// Enumerate all available CUDA GPUs
std::vector<DeviceInfo> list_devices();

// Select primary device by index, returns true on success
bool select_device(int device_id);

// Get current device info
DeviceInfo current_device();

// Allocate device memory
void* device_alloc(size_t bytes);

// Free device memory
void device_free(void* ptr);

// Copy host→device
void copy_to_device(void* dst_device, const void* src_host, size_t bytes);

// Copy device→host
void copy_to_host(void* dst_host, const void* src_device, size_t bytes);

// Synchronize current device
void sync();

// RAII device guard — select on enter, restore on exit
class DeviceGuard {
 public:
  explicit DeviceGuard(int device_id);
  ~DeviceGuard();
 private:
  int previous_device_;
};

} // namespace nn::cuda
