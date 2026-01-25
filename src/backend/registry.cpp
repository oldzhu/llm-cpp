#include "backend/registry.h"

#include <stdexcept>

#include "backend/cpu_backend.h"

namespace backend {

static std::unique_ptr<KernelBackend>& global_backend() {
  static std::unique_ptr<KernelBackend> g_backend;
  return g_backend;
}

KernelBackend& get() {
  auto& gb = global_backend();
  if (!gb) {
    gb = std::make_unique<CpuBackend>();
  }
  return *gb;
}

void set(std::unique_ptr<KernelBackend> backend) {
  if (!backend) throw std::runtime_error("backend::set: backend must not be null");
  global_backend() = std::move(backend);
}

Device current_device() {
  // CPU-only today. When CUDA/HIP is added, this can be expanded.
  return Device{DeviceType::Cpu, 0};
}

} // namespace backend
