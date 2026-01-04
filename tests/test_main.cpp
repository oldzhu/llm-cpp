#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "data.h"
#include "model.h"
#include "ops.h"
#include "optim.h"
#include "tensor.h"
#include "util.h"

namespace {

int g_failures = 0;

void expect_true(bool cond, const std::string& msg) {
  if (!cond) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
  }
}

void expect_near(float a, float b, float tol, const std::string& msg) {
  const float diff = std::fabs(a - b);
  if (!(diff <= tol)) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "  a=" << a << " b=" << b << " diff=" << diff << " tol=" << tol << "\n";
  }
}

float compute_loss_matmul_ce(const std::vector<float>& a_data,
                            const std::vector<float>& b_data,
                            int m,
                            int k,
                            int n,
                            const std::vector<std::int32_t>& targets) {
  nn::GradMode no_grad(false);

  nn::Tensor a = nn::Tensor::zeros({m, k}, false);
  nn::Tensor b = nn::Tensor::zeros({k, n}, false);
  *a.data = a_data;
  *b.data = b_data;

  nn::Tensor logits = nn::matmul2d(a, b); // [m,n]
  nn::Tensor loss = nn::cross_entropy(logits, targets);
  return (*loss.data)[0];
}

void test_gradcheck_matmul2d_via_cross_entropy() {
  std::cout << "[RUN ] gradcheck matmul2d (via cross_entropy)\n";

  const int m = 3;
  const int k = 4;
  const int n = 5;
  const float eps = 1e-3f;

  // Deterministic init
  util::Rng rng(123);
  std::vector<float> a_data(static_cast<std::size_t>(m) * k);
  std::vector<float> b_data(static_cast<std::size_t>(k) * n);
  for (float& v : a_data) v = (rng.next_f01() - 0.5f) * 0.2f;
  for (float& v : b_data) v = (rng.next_f01() - 0.5f) * 0.2f;

  std::vector<std::int32_t> targets(static_cast<std::size_t>(m));
  for (int i = 0; i < m; ++i) targets[static_cast<std::size_t>(i)] = i % n;

  // Analytic gradients
  nn::Tensor a = nn::Tensor::zeros({m, k}, true);
  nn::Tensor b = nn::Tensor::zeros({k, n}, true);
  *a.data = a_data;
  *b.data = b_data;

  nn::Tensor logits = nn::matmul2d(a, b);
  nn::Tensor loss = nn::cross_entropy(logits, targets);
  loss.backward();

  // Numeric gradients for a
  for (std::size_t i = 0; i < a_data.size(); ++i) {
    std::vector<float> ap = a_data;
    std::vector<float> am = a_data;
    ap[i] += eps;
    am[i] -= eps;
    const float lp = compute_loss_matmul_ce(ap, b_data, m, k, n, targets);
    const float lm = compute_loss_matmul_ce(am, b_data, m, k, n, targets);
    const float gn = (lp - lm) / (2.0f * eps);
    const float ga = (*a.grad)[i];

    // Loose-ish tolerance because we’re doing finite diff and softmax/log.
    const float tol = 3e-2f;
    expect_near(ga, gn, tol, "matmul2d gradcheck: dL/dA[" + std::to_string(i) + "]");
  }

  // Numeric gradients for b
  for (std::size_t i = 0; i < b_data.size(); ++i) {
    std::vector<float> bp = b_data;
    std::vector<float> bm = b_data;
    bp[i] += eps;
    bm[i] -= eps;
    const float lp = compute_loss_matmul_ce(a_data, bp, m, k, n, targets);
    const float lm = compute_loss_matmul_ce(a_data, bm, m, k, n, targets);
    const float gn = (lp - lm) / (2.0f * eps);
    const float ga = (*b.grad)[i];

    const float tol = 3e-2f;
    expect_near(ga, gn, tol, "matmul2d gradcheck: dL/dB[" + std::to_string(i) + "]");
  }
}

float compute_loss_layernorm_ce(const std::vector<float>& x_data, int N, int V, const std::vector<std::int32_t>& targets) {
  nn::GradMode no_grad(false);

  nn::Tensor x = nn::Tensor::zeros({N, V}, false);
  *x.data = x_data;

  nn::Tensor y = nn::layernorm_lastdim(x, 1e-5f);
  nn::Tensor loss = nn::cross_entropy(y, targets);
  return (*loss.data)[0];
}

void test_gradcheck_layernorm_lastdim_via_cross_entropy() {
  std::cout << "[RUN ] gradcheck layernorm_lastdim (via cross_entropy)\n";

  const int N = 4;
  const int V = 6;
  const float eps = 1e-3f;

  util::Rng rng(456);
  std::vector<float> x_data(static_cast<std::size_t>(N) * V);
  for (float& v : x_data) v = (rng.next_f01() - 0.5f) * 0.5f;

  std::vector<std::int32_t> targets(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) targets[static_cast<std::size_t>(i)] = (i + 1) % V;

  nn::Tensor x = nn::Tensor::zeros({N, V}, true);
  *x.data = x_data;

  nn::Tensor y = nn::layernorm_lastdim(x, 1e-5f);
  nn::Tensor loss = nn::cross_entropy(y, targets);
  loss.backward();

  for (std::size_t i = 0; i < x_data.size(); ++i) {
    std::vector<float> xp = x_data;
    std::vector<float> xm = x_data;
    xp[i] += eps;
    xm[i] -= eps;
    const float lp = compute_loss_layernorm_ce(xp, N, V, targets);
    const float lm = compute_loss_layernorm_ce(xm, N, V, targets);
    const float gn = (lp - lm) / (2.0f * eps);
    const float ga = (*x.grad)[i];

    const float tol = 5e-2f;
    expect_near(ga, gn, tol, "layernorm gradcheck: dL/dX[" + std::to_string(i) + "]");
  }
}

void test_tiny_training_regression_loss_decreases() {
  std::cout << "[RUN ] tiny training regression (loss decreases)\n";

  // Synthetic dataset >= 1024 bytes.
  std::vector<std::uint8_t> bytes(2048);
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i & 0xFF);
  data::ByteDataset ds(std::move(bytes));

  model::Config cfg;
  cfg.vocab_size = 256;
  cfg.seq_len = 32;
  cfg.d_model = 32;
  cfg.n_layers = 1;

  const std::uint64_t seed = 7;
  model::TinyGPT gpt(cfg, seed);

  optim::AdamWConfig ocfg;
  ocfg.lr = 1e-3f;
  ocfg.weight_decay = 0.01f;
  optim::AdamW opt(ocfg);

  util::Rng rng(seed ^ 0xDEADBEEF);

  auto one_step_loss = [&]() -> float {
    data::Batch batch = ds.sample_batch(2, 32, rng);
    gpt.zero_grad();
    nn::Tensor loss = gpt.loss(batch.x, batch.y, batch.B, batch.T);
    loss.backward();
    opt.step(gpt.parameters().tensors);
    return (*loss.data)[0];
  };

  const float l0 = one_step_loss();
  float last = l0;
  for (int i = 0; i < 19; ++i) last = one_step_loss();

  // We just want a stable regression guardrail, not a strict metric.
  expect_true(last < l0, "expected training loss to decrease (l0=" + std::to_string(l0) + ", lN=" + std::to_string(last) + ")");
}

} // namespace

int main() {
  try {
    test_gradcheck_matmul2d_via_cross_entropy();
    test_gradcheck_layernorm_lastdim_via_cross_entropy();
    test_tiny_training_regression_loss_decreases();

    if (g_failures == 0) {
      std::cout << "[OK  ] all tests passed\n";
      return 0;
    }
    std::cerr << "[DONE] failures: " << g_failures << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] unhandled exception: " << e.what() << "\n";
    return 2;
  }
}
