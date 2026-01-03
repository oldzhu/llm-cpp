#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "tensor.h"

namespace optim {

struct AdamWConfig {
  float lr = 3e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.01f;
};

class AdamW {
 public:
  explicit AdamW(const AdamWConfig& cfg);

  const AdamWConfig& cfg() const { return cfg_; }

  void step(const std::vector<nn::Tensor*>& params);

  struct ExportedState {
    std::uint64_t t = 0;
    std::vector<std::vector<float>> m; // per-param
    std::vector<std::vector<float>> v; // per-param
  };

  ExportedState export_state(const std::vector<nn::Tensor*>& params) const;
  void import_state(const std::vector<nn::Tensor*>& params, const ExportedState& st);

 private:
  AdamWConfig cfg_;
  std::int64_t t_ = 0;

  struct State {
    std::vector<float> m;
    std::vector<float> v;
  };

  std::unordered_map<const nn::Tensor*, State> state_;
};

} // namespace optim
