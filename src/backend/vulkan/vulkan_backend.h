#pragma once

#include "backend/backend.h"

namespace backend {

// Vulkan compute backend — POC (proof-of-concept) mode.
//
// Teaching-first intent:
// - Demonstrates full Vulkan compute pipeline: instance → device → shader → dispatch → readback
// - Single-precision matmul only (matmul2d_fwd)
// - Host↔device copies per call (not device-resident — see Stage D docs)
//
// Requires: Vulkan SDK (LunarG), `vulkan-1.lib`, a GPU with Vulkan 1.1+ support
//
// Shader: `shaders/matmul_fwd.comp` → compiled to SPIR-V at build time

class VulkanBackend final : public KernelBackend {
 public:
  VulkanBackend();
  ~VulkanBackend() override;

  // Returns true if Vulkan device was successfully initialised.
  bool is_ready() const { return device_ != nullptr; }

  void matmul2d_fwd(int m, int k, int n, const float* a_mk, const float* b_kn, float* out_mn) override;

  // Backward: currently delegates to CPU reference (Vulkan POC covers fwd only).
  void matmul2d_bwd(int m,
                    int k,
                    int n,
                    const float* a_mk,
                    const float* b_kn,
                    const float* d_out_mn,
                    float* d_a_mk,
                    float* d_b_kn) override;

 private:
  struct Impl;
  Impl* impl_ = nullptr;

  // Opaque Vulkan objects accessed via pImpl
  void* instance_ = nullptr;
  void* device_ = nullptr;
};

} // namespace backend
