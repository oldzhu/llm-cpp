#pragma once

// Backend boundary (placeholder).
//
// Intent:
// - Keep model math/op semantics stable in `nn::*`.
// - Route hot kernels (matmul, softmax, attention, etc.) through a backend later
//   without rewriting model code.
//
// This file intentionally starts as a minimal interface sketch so we can evolve it
// alongside real variants/backends.

namespace backend {

struct KernelBackend {
  virtual ~KernelBackend() = default;
};

} // namespace backend
