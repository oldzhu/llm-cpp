#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "util.h"

namespace nn {

static thread_local bool g_grad_enabled = true;

bool is_grad_enabled() { return g_grad_enabled; }

void set_grad_enabled(bool enabled) { g_grad_enabled = enabled; }

GradMode::GradMode(bool enabled) : prev(is_grad_enabled()) {
  set_grad_enabled(enabled);
}

GradMode::~GradMode() {
  set_grad_enabled(prev);
}

std::size_t numel_of(const std::vector<int>& shape) {
  if (shape.empty()) return 0;
  std::size_t n = 1;
  for (int d : shape) {
    if (d <= 0) throw std::runtime_error("invalid shape dimension");
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

std::size_t Tensor::numel() const {
  return numel_of(shape);
}

Tensor Tensor::zeros(const std::vector<int>& shape_, bool requires_grad_) {
  Tensor t;
  t.shape = shape_;
  t.requires_grad = requires_grad_;
  const std::size_t n = numel_of(shape_);
  t.data = std::make_shared<std::vector<float>>(n, 0.0f);
  if (requires_grad_) {
    t.grad = std::make_shared<std::vector<float>>(n, 0.0f);
  }
  return t;
}

Tensor Tensor::randn(const std::vector<int>& shape_, float stddev, std::uint64_t seed, bool requires_grad_) {
  Tensor t;
  t.shape = shape_;
  t.requires_grad = requires_grad_;
  const std::size_t n = numel_of(shape_);
  t.data = std::make_shared<std::vector<float>>(n);
  if (requires_grad_) {
    t.grad = std::make_shared<std::vector<float>>(n, 0.0f);
  }

  util::Rng rng(seed);
  // Box-Muller
  for (std::size_t i = 0; i < n; i += 2) {
    const float u1 = std::max(rng.next_f01(), 1e-7f);
    const float u2 = rng.next_f01();
    const float r = std::sqrt(-2.0f * std::log(u1));
    const float theta = 2.0f * 3.1415926535f * u2;
    const float z0 = r * std::cos(theta);
    const float z1 = r * std::sin(theta);
    (*t.data)[i] = z0 * stddev;
    if (i + 1 < n) (*t.data)[i + 1] = z1 * stddev;
  }
  return t;
}

void Tensor::zero_grad() {
  if (requires_grad && grad) {
    std::fill(grad->begin(), grad->end(), 0.0f);
  }
}

static void build_topo(const Tensor& t, std::unordered_set<const Node*>& visited, std::vector<Tensor>& topo) {
  if (!t.node) {
    topo.push_back(t);
    return;
  }
  const Node* key = t.node.get();
  if (visited.find(key) != visited.end()) return;
  visited.insert(key);
  for (const auto& p : t.node->parents) {
    build_topo(p, visited, topo);
  }
  topo.push_back(t);
}

void Tensor::backward() {
  if (numel() != 1) {
    throw std::runtime_error("backward() expects a scalar tensor");
  }
  if (!requires_grad) {
    // Allow calling backward on a tensor that doesn't require grad; nothing to do.
    return;
  }
  if (!grad) {
    grad = std::make_shared<std::vector<float>>(1, 0.0f);
  }

  (*grad)[0] = 1.0f;

  std::unordered_set<const Node*> visited;
  std::vector<Tensor> topo;
  topo.reserve(1024);
  build_topo(*this, visited, topo);

  // Traverse reverse topo; call backward for nodes.
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    Tensor& cur = *it;
    if (cur.node && cur.node->backward) {
      cur.node->backward(cur);
    }
  }
}

} // namespace nn
