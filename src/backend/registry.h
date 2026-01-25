#pragma once

#include <memory>

#include "backend/backend.h"

namespace backend {

// Global backend registry.
//
// Teaching-first intent:
// - Keep model/op semantics stable in `nn::*`.
// - Allow swapping kernel implementations without changing model code.
//
// Default backend is CPU.
KernelBackend& get();

// Replaces the global backend. The caller owns the passed backend.
void set(std::unique_ptr<KernelBackend> backend);

// Current device target (initially CPU-only).
Device current_device();

} // namespace backend
