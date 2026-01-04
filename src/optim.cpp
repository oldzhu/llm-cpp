#include "optim.h"

#include <cmath>
#include <stdexcept>

namespace optim {

AdamW::AdamW(const AdamWConfig& cfg) : cfg_(cfg) {
  if (cfg_.lr <= 0.0f) throw std::runtime_error("AdamW: lr must be > 0");
  if (cfg_.beta1 <= 0.0f || cfg_.beta1 >= 1.0f) throw std::runtime_error("AdamW: beta1 out of range");
  if (cfg_.beta2 <= 0.0f || cfg_.beta2 >= 1.0f) throw std::runtime_error("AdamW: beta2 out of range");
  if (cfg_.eps <= 0.0f) throw std::runtime_error("AdamW: eps must be > 0");
}

void AdamW::step(const std::vector<nn::Tensor*>& params) {
  // AdamW update (decoupled weight decay):
  //   m = b1*m + (1-b1)*g
  //   v = b2*v + (1-b2)*g^2
  //   mhat = m/(1-b1^t), vhat = v/(1-b2^t)
  //   theta -= lr * ( mhat/(sqrt(vhat)+eps) + wd*theta )
  ++t_;
  const float b1 = cfg_.beta1;
  const float b2 = cfg_.beta2;
  const float lr = cfg_.lr;
  const float eps = cfg_.eps;
  const float wd = cfg_.weight_decay;

  const float b1t = 1.0f - std::pow(b1, static_cast<float>(t_));
  const float b2t = 1.0f - std::pow(b2, static_cast<float>(t_));

  for (nn::Tensor* p : params) {
    if (!p) continue;
    if (!p->requires_grad) continue;
    if (!p->grad) throw std::runtime_error("AdamW: parameter missing grad buffer");

    auto& st = state_[p];
    const std::size_t n = p->numel();
    if (st.m.size() != n) {
      st.m.assign(n, 0.0f);
      st.v.assign(n, 0.0f);
    }

    for (std::size_t i = 0; i < n; ++i) {
      const float g = (*p->grad)[i];
      st.m[i] = b1 * st.m[i] + (1.0f - b1) * g;
      st.v[i] = b2 * st.v[i] + (1.0f - b2) * (g * g);

      const float mhat = st.m[i] / b1t;
      const float vhat = st.v[i] / b2t;
      const float update = mhat / (std::sqrt(vhat) + eps) + wd * (*p->data)[i];
      (*p->data)[i] -= lr * update;
    }
  }
}

AdamW::ExportedState AdamW::export_state(const std::vector<nn::Tensor*>& params) const {
  ExportedState out;
  out.t = static_cast<std::uint64_t>(t_);
  out.m.reserve(params.size());
  out.v.reserve(params.size());

  for (const nn::Tensor* p : params) {
    if (!p) {
      out.m.emplace_back();
      out.v.emplace_back();
      continue;
    }
    const auto it = state_.find(p);
    if (it == state_.end()) {
      out.m.emplace_back(p->numel(), 0.0f);
      out.v.emplace_back(p->numel(), 0.0f);
    } else {
      out.m.push_back(it->second.m);
      out.v.push_back(it->second.v);
    }
  }
  return out;
}

void AdamW::import_state(const std::vector<nn::Tensor*>& params, const ExportedState& st) {
  t_ = static_cast<std::int64_t>(st.t);
  if (st.m.size() != params.size() || st.v.size() != params.size()) {
    throw std::runtime_error("AdamW::import_state: param count mismatch");
  }
  for (std::size_t i = 0; i < params.size(); ++i) {
    nn::Tensor* p = params[i];
    if (!p) continue;
    State s;
    const std::size_t n = p->numel();
    if (st.m[i].size() != n || st.v[i].size() != n) {
      throw std::runtime_error("AdamW::import_state: tensor numel mismatch");
    }
    s.m = st.m[i];
    s.v = st.v[i];
    state_[p] = std::move(s);
  }
}

} // namespace optim
