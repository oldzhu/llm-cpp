#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nn {

struct Tensor;

struct Node {
  std::vector<Tensor> parents;
  std::function<void(Tensor& out)> backward;
};

struct Tensor {
  std::shared_ptr<std::vector<float>> data;
  std::shared_ptr<std::vector<float>> grad;
  std::vector<int> shape;
  bool requires_grad = false;
  std::shared_ptr<Node> node;
  std::string debug_name;

  Tensor() = default;

  static Tensor zeros(const std::vector<int>& shape, bool requires_grad = false);
  static Tensor randn(const std::vector<int>& shape, float stddev, std::uint64_t seed, bool requires_grad = false);

  std::size_t numel() const;
  void zero_grad();

  // Backprop from this scalar tensor (e.g., loss).
  void backward();
};

// Global grad mode (thread-local). When disabled, ops will not build backward graphs.
bool is_grad_enabled();
void set_grad_enabled(bool enabled);

struct GradMode {
  bool prev;
  explicit GradMode(bool enabled);
  ~GradMode();
  GradMode(const GradMode&) = delete;
  GradMode& operator=(const GradMode&) = delete;
};

// Utility
std::size_t numel_of(const std::vector<int>& shape);

} // namespace nn
