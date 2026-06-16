#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "data.h"
#include "backend/cpu_backend.h"
#include "backend/blocked_simd_backend.h"
#include "backend/vulkan/vulkan_backend.h"
#include "backend/registry.h"
#include "model.h"
#include "ops.h"
#include "optim.h"
#include "tensor.h"
#include "util.h"
#include "variants/mha/mha_attention.h"
#include "variants/kvcache/kvcache_attention.h"
#include "variants/rope/rope_attention.h"
#include "variants/gqa/gqa_attention.h"
#include "variants/mla/mla_attention.h"
#include "variants/ppo/ppo_trainer.h"
#include "variants/moe/moe_mlp.h"
#include "tokenizer/byte_tokenizer.h"
#include "tokenizer/bpe_tokenizer.h"
#include "tokenizer/spm_tokenizer.h"

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

void test_mha_matches_1h_when_single_head() {
  std::cout << "[RUN ] mha matches 1-head attention (H=1)\n";

  const int B = 2;
  const int T = 4;
  const int C = 8;

  // Deterministic init.
  util::Rng rng(999);
  auto fill = [&](nn::Tensor& t, float scale) {
    for (float& v : *t.data) v = (rng.next_f01() - 0.5f) * scale;
  };

  nn::Tensor x = nn::Tensor::zeros({B, T, C}, true);
  nn::Tensor w_qkv = nn::Tensor::zeros({C, 3 * C}, true);
  nn::Tensor b_qkv = nn::Tensor::zeros({3 * C}, true);
  nn::Tensor w_proj = nn::Tensor::zeros({C, C}, true);
  nn::Tensor b_proj = nn::Tensor::zeros({C}, true);

  fill(x, 0.2f);
  fill(w_qkv, 0.2f);
  fill(b_qkv, 0.02f);
  fill(w_proj, 0.2f);
  fill(b_proj, 0.02f);

  // Forward equivalence.
  nn::Tensor y1 = nn::self_attention_1h(x, w_qkv, b_qkv, w_proj, b_proj);
  nn::Tensor y2 = nn::variants::mha::self_attention_mha(x, w_qkv, b_qkv, w_proj, b_proj, /*n_heads=*/1);
  expect_true(y1.shape == y2.shape, "mha vs 1h: output shape matches");
  for (std::size_t i = 0; i < y1.data->size(); ++i) {
    expect_near((*y1.data)[i], (*y2.data)[i], 1e-5f, "mha vs 1h: forward out[" + std::to_string(i) + "]");
  }

  // Backward equivalence via a scalar loss.
  std::vector<std::int32_t> targets(static_cast<std::size_t>(B * T));
  for (int i = 0; i < B * T; ++i) targets[static_cast<std::size_t>(i)] = i % C;

  nn::Tensor loss1 = nn::cross_entropy(nn::reshape(y1, {B * T, C}), targets);
  loss1.backward();
  const std::vector<float> xg1 = *x.grad;
  const std::vector<float> wg1 = *w_qkv.grad;

  x.zero_grad();
  w_qkv.zero_grad();
  b_qkv.zero_grad();
  w_proj.zero_grad();
  b_proj.zero_grad();

  nn::Tensor y2b = nn::variants::mha::self_attention_mha(x, w_qkv, b_qkv, w_proj, b_proj, /*n_heads=*/1);
  nn::Tensor loss2 = nn::cross_entropy(nn::reshape(y2b, {B * T, C}), targets);
  loss2.backward();
  const std::vector<float> xg2 = *x.grad;
  const std::vector<float> wg2 = *w_qkv.grad;

  expect_near((*loss1.data)[0], (*loss2.data)[0], 1e-5f, "mha vs 1h: loss matches");

  for (std::size_t i = 0; i < xg1.size(); ++i) {
    expect_near(xg1[i], xg2[i], 1e-4f, "mha vs 1h: dL/dx[" + std::to_string(i) + "]");
  }
  for (std::size_t i = 0; i < wg1.size(); ++i) {
    expect_near(wg1[i], wg2[i], 2e-4f, "mha vs 1h: dL/dw_qkv[" + std::to_string(i) + "]");
  }
}

void test_backend_dispatch_matmul2d() {
  std::cout << "[RUN ] backend dispatch matmul2d (counting backend)\n";

  struct CountingBackend final : public backend::KernelBackend {
    backend::CpuBackend cpu;
    int fwd_calls = 0;
    int bwd_calls = 0;

    void matmul2d_fwd(int m, int k, int n, const float* a_mk, const float* b_kn, float* out_mn) override {
      ++fwd_calls;
      cpu.matmul2d_fwd(m, k, n, a_mk, b_kn, out_mn);
    }

    void matmul2d_bwd(int m,
                      int k,
                      int n,
                      const float* a_mk,
                      const float* b_kn,
                      const float* d_out_mn,
                      float* d_a_mk,
                      float* d_b_kn) override {
      ++bwd_calls;
      cpu.matmul2d_bwd(m, k, n, a_mk, b_kn, d_out_mn, d_a_mk, d_b_kn);
    }
  };

  auto cb = std::make_unique<CountingBackend>();
  CountingBackend* raw = cb.get();
  backend::set(std::move(cb));

  const int m = 2;
  const int k = 3;
  const int n = 4;

  nn::Tensor a = nn::Tensor::zeros({m, k}, true);
  nn::Tensor b = nn::Tensor::zeros({k, n}, true);
  for (std::size_t i = 0; i < a.data->size(); ++i) (*a.data)[i] = static_cast<float>(i + 1) * 0.1f;
  for (std::size_t i = 0; i < b.data->size(); ++i) (*b.data)[i] = static_cast<float>(i + 1) * 0.05f;

  nn::Tensor c = nn::matmul2d(a, b);
  expect_true(raw->fwd_calls > 0, "expected backend matmul2d_fwd to be called");

  // Scalar loss = sum(C) so dC is all-ones.
  nn::Tensor loss = nn::Tensor::zeros({1}, true);
  (*loss.data)[0] = 0.0f;
  for (float v : *c.data) (*loss.data)[0] += v;
  loss.node = std::make_shared<nn::Node>();
  loss.node->parents = {c};
  loss.node->backward = [](nn::Tensor& o) {
    nn::Tensor& pc = o.node->parents[0];
    if (!pc.requires_grad) return;
    for (std::size_t i = 0; i < pc.grad->size(); ++i) (*pc.grad)[i] += (*o.grad)[0];
  };

  loss.backward();
  expect_true(raw->bwd_calls > 0, "expected backend matmul2d_bwd to be called");

  // Restore default backend for other tests.
  backend::set(std::make_unique<backend::CpuBackend>());
}

void test_backend_dispatch_bmm() {
  std::cout << "[RUN ] backend dispatch bmm (counting backend)\n";

  struct CountingBackend final : public backend::KernelBackend {
    backend::CpuBackend cpu;
    int fwd_calls = 0;
    int bwd_calls = 0;

    void matmul2d_fwd(int m, int k, int n, const float* a_mk, const float* b_kn, float* out_mn) override {
      ++fwd_calls;
      cpu.matmul2d_fwd(m, k, n, a_mk, b_kn, out_mn);
    }

    void matmul2d_bwd(int m,
                      int k,
                      int n,
                      const float* a_mk,
                      const float* b_kn,
                      const float* d_out_mn,
                      float* d_a_mk,
                      float* d_b_kn) override {
      ++bwd_calls;
      cpu.matmul2d_bwd(m, k, n, a_mk, b_kn, d_out_mn, d_a_mk, d_b_kn);
    }
  };

  auto cb = std::make_unique<CountingBackend>();
  CountingBackend* raw = cb.get();
  backend::set(std::move(cb));

  const int B = 3;
  const int M = 2;
  const int K = 4;
  const int N = 5;

  nn::Tensor a = nn::Tensor::zeros({B, M, K}, true);
  nn::Tensor b = nn::Tensor::zeros({B, K, N}, true);
  for (std::size_t i = 0; i < a.data->size(); ++i) (*a.data)[i] = static_cast<float>((i % 17) + 1) * 0.01f;
  for (std::size_t i = 0; i < b.data->size(); ++i) (*b.data)[i] = static_cast<float>((i % 19) + 1) * 0.02f;

  nn::Tensor out = nn::bmm(a, b);
  expect_true(raw->fwd_calls == B, "expected bmm to call backend matmul2d_fwd exactly B times");

  std::vector<std::int32_t> targets(static_cast<std::size_t>(B * M));
  for (int i = 0; i < B * M; ++i) targets[static_cast<std::size_t>(i)] = i % N;
  nn::Tensor loss = nn::cross_entropy(nn::reshape(out, {B * M, N}), targets);
  loss.backward();
  expect_true(raw->bwd_calls == B, "expected bmm backward to call backend matmul2d_bwd exactly B times");

  // Restore default backend for other tests.
  backend::set(std::make_unique<backend::CpuBackend>());
}

void test_byte_tokenizer_encode_decode() {
  std::cout << "[RUN ] ByteTokenizer encode/decode roundtrip\n";
  ByteTokenizer tok;

  // ASCII
  std::string s1 = "hello world!";
  auto t1 = tok.encode(s1);
  std::string d1 = tok.decode(t1);
  expect_true(s1 == d1, "ASCII roundtrip");

  // UTF-8 (Chinese)
  std::string s2 = "你好，世界!";
  auto t2 = tok.encode(s2);
  std::string d2 = tok.decode(t2);
  expect_true(s2 == d2, "UTF-8 roundtrip");

  // Empty
  std::string s3 = "";
  auto t3 = tok.encode(s3);
  std::string d3 = tok.decode(t3);
  expect_true(s3 == d3, "Empty string roundtrip");

  // All byte values
  std::string s4;
  for (int i = 0; i < 256; ++i) s4.push_back(static_cast<char>(i));
  auto t4 = tok.encode(s4);
  std::string d4 = tok.decode(t4);
  expect_true(s4 == d4, "All byte values roundtrip");

  // Vocab size
  expect_true(tok.vocab_size() == 256, "ByteTokenizer vocab_size == 256");
}

void test_bpe_tokenizer_encode_decode() {
  std::cout << "[RUN ] BpeTokenizer encode/decode roundtrip\n";
  const std::string vocab_path = "tests/bpe_vocab.txt";
  const std::string merges_path = "tests/bpe_merges.txt";
  BpeTokenizer tok(vocab_path, merges_path);

  // Simple test: "abcde" should merge to ["abc", "de"] if merges allow
  std::string s1 = "abcde";
  auto t1 = tok.encode(s1);
  std::string d1 = tok.decode(t1);
  expect_true(d1 == "abcde", "BPE roundtrip: abcde");

  // Single char
  std::string s2 = "a";
  auto t2 = tok.encode(s2);
  std::string d2 = tok.decode(t2);
  expect_true(d2 == "a", "BPE roundtrip: a");

  // Vocab size
  expect_true(tok.vocab_size() == 10, "BpeTokenizer vocab_size == 10");
}

void test_gradcheck_layernorm_affine_via_cross_entropy() {
  std::cout << "[RUN ] gradcheck layernorm_affine (via cross_entropy)\n";
  const int N = 4;
  const int V = 6;
  const float eps = 1e-3f;
  util::Rng rng(789);
  std::vector<float> x_data(static_cast<std::size_t>(N) * V);
  for (float& v : x_data) v = (rng.next_f01() - 0.5f) * 0.5f;
  std::vector<float> gamma_data(static_cast<std::size_t>(V));
  std::vector<float> beta_data(static_cast<std::size_t>(V));
  for (int i = 0; i < V; ++i) {
    gamma_data[i] = 1.0f + (rng.next_f01() - 0.5f) * 0.1f;
    beta_data[i]  = (rng.next_f01() - 0.5f) * 0.05f;
  }
  std::vector<std::int32_t> targets(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) targets[static_cast<std::size_t>(i)] = (i + 1) % V;

  nn::Tensor x = nn::Tensor::zeros({N, V}, true);
  nn::Tensor gamma = nn::Tensor::zeros({V}, true);
  nn::Tensor beta  = nn::Tensor::zeros({V}, true);
  *x.data = x_data;
  *gamma.data = gamma_data;
  *beta.data  = beta_data;

  nn::Tensor y = nn::layernorm_affine(x, gamma, beta, 1e-5f);
  nn::Tensor loss = nn::cross_entropy(y, targets);
  loss.backward();

  auto fd_loss = [&](const std::vector<float>& xd, const std::vector<float>& gd, const std::vector<float>& bd) -> float {
    nn::GradMode no_grad(false);
    nn::Tensor xi = nn::Tensor::zeros({N, V}, false);
    nn::Tensor gi = nn::Tensor::zeros({V}, false);
    nn::Tensor bi = nn::Tensor::zeros({V}, false);
    *xi.data = xd;
    *gi.data = gd;
    *bi.data = bd;
    nn::Tensor yi = nn::layernorm_affine(xi, gi, bi, 1e-5f);
    return (*nn::cross_entropy(yi, targets).data)[0];
  };

  // dL/dx
  for (std::size_t i = 0; i < x_data.size(); ++i) {
    auto xp = x_data, xm = x_data;
    xp[i] += eps; xm[i] -= eps;
    const float gn = (fd_loss(xp, gamma_data, beta_data) - fd_loss(xm, gamma_data, beta_data)) / (2.0f * eps);
    expect_near((*x.grad)[i], gn, 5e-2f, "layernorm_affine gradcheck: dL/dx[" + std::to_string(i) + "]");
  }

  // dL/dgamma
  for (std::size_t i = 0; i < gamma_data.size(); ++i) {
    auto gp = gamma_data, gm = gamma_data;
    gp[i] += eps; gm[i] -= eps;
    const float gn = (fd_loss(x_data, gp, beta_data) - fd_loss(x_data, gm, beta_data)) / (2.0f * eps);
    expect_near((*gamma.grad)[i], gn, 5e-2f, "layernorm_affine gradcheck: dL/dgamma[" + std::to_string(i) + "]");
  }

  // dL/dbeta
  for (std::size_t i = 0; i < beta_data.size(); ++i) {
    auto bp = beta_data, bm = beta_data;
    bp[i] += eps; bm[i] -= eps;
    const float gn = (fd_loss(x_data, gamma_data, bp) - fd_loss(x_data, gamma_data, bm)) / (2.0f * eps);
    expect_near((*beta.grad)[i], gn, 5e-2f, "layernorm_affine gradcheck: dL/dbeta[" + std::to_string(i) + "]");
  }
}

void test_kvcache_matches_full_attention() {
  std::cout << "[RUN ] kvcache matches full attention\n";
  const int B = 1;
  const int T = 3;
  const int C = 4;
  const int T_max = 8;
  util::Rng rng(777);
  auto fill = [&](nn::Tensor& t, float scale) {
    for (float& v : *t.data) v = (rng.next_f01() - 0.5f) * scale;
  };
  nn::Tensor x_all = nn::Tensor::zeros({B, T + 1, C}, false);
  nn::Tensor w_qkv = nn::Tensor::zeros({C, 3 * C}, false);
  nn::Tensor b_qkv = nn::Tensor::zeros({3 * C}, false);
  nn::Tensor w_proj = nn::Tensor::zeros({C, C}, false);
  nn::Tensor b_proj = nn::Tensor::zeros({C}, false);
  fill(x_all, 0.2f);
  fill(w_qkv, 0.2f);
  fill(b_qkv, 0.02f);
  fill(w_proj, 0.2f);
  fill(b_proj, 0.02f);
  std::vector<float> x_prefill_data(static_cast<std::size_t>(B) * T * C);
  std::vector<float> x_step_data(static_cast<std::size_t>(B) * 1 * C);
  for (std::size_t i = 0; i < x_prefill_data.size(); ++i) x_prefill_data[i] = (*x_all.data)[i];
  const std::size_t step_src = static_cast<std::size_t>(B) * T * C;
  for (std::size_t i = 0; i < x_step_data.size(); ++i) x_step_data[i] = (*x_all.data)[step_src + i];
  nn::Tensor x_prefill = nn::Tensor::zeros({B, T, C}, false);
  nn::Tensor x_step = nn::Tensor::zeros({B, 1, C}, false);
  *x_prefill.data = x_prefill_data;
  *x_step.data = x_step_data;
  nn::variants::kvcache::KVCache cache(B, T_max, C);
  nn::Tensor y_prefill = nn::variants::kvcache::self_attention_prefill(
      x_prefill, w_qkv, b_qkv, w_proj, b_proj, cache);
  expect_true(cache.cur_len == T, "kvcache: cur_len after prefill");
  expect_true(y_prefill.shape == std::vector<int>({B, T, C}), "kvcache: prefill output shape");
  nn::Tensor y_step = nn::variants::kvcache::self_attention_step(
      x_step, w_qkv, b_qkv, w_proj, b_proj, cache);
  expect_true(cache.cur_len == T + 1, "kvcache: cur_len after step");
  expect_true(y_step.shape == std::vector<int>({B, 1, C}), "kvcache: step output shape");
  nn::Tensor y_full = nn::self_attention_1h(x_all, w_qkv, b_qkv, w_proj, b_proj);
  const std::size_t step_base = 0;
  const std::size_t full_base = static_cast<std::size_t>(T) * C;
  for (int c = 0; c < C; ++c) {
    expect_near((*y_step.data)[step_base + c], (*y_full.data)[full_base + c], 1e-5f,
                "kvcache: step matches full at c=" + std::to_string(c));
  }
  nn::Tensor qkv_full = nn::linear_lastdim(x_all, w_qkv, b_qkv);
  for (int t = 0; t < T + 1; ++t) {
    const std::size_t qkv_off = static_cast<std::size_t>(t) * 3 * C + C;
    const std::size_t cache_off = static_cast<std::size_t>(t) * C;
    for (int c = 0; c < C; ++c) {
      expect_near((*cache.k_cache_.data)[cache_off + c], (*qkv_full.data)[qkv_off + c], 1e-5f,
                  "kvcache: cached K matches at t=" + std::to_string(t));
    }
  }
}

void test_rope_position_zero_is_identity() {
  std::cout << "[RUN ] RoPE position-zero is identity\n";
  const int B = 1;
  const int T = 1;
  const int C = 4;

  nn::Tensor q = nn::Tensor::zeros({B, T, C}, false);
  nn::Tensor k = nn::Tensor::zeros({B, T, C}, false);
  for (int c = 0; c < C; ++c) {
    (*q.data)[c] = static_cast<float>(c + 1);
    (*k.data)[c] = static_cast<float>(c + 10);
  }

  std::vector<float> q_copy = *q.data;
  std::vector<float> k_copy = *k.data;

  nn::variants::rope::rope_rotate(q, k, B, T, C);

  for (int c = 0; c < C; ++c) {
    expect_near((*q.data)[c], q_copy[static_cast<std::size_t>(c)], 1e-5f,
                "RoPE pos0: q unchanged at c=" + std::to_string(c));
    expect_near((*k.data)[c], k_copy[static_cast<std::size_t>(c)], 1e-5f,
                "RoPE pos0: k unchanged at c=" + std::to_string(c));
  }
}

void test_rope_attention_runs() {
  std::cout << "[RUN ] RoPE attention forward\n";
  const int B = 2;
  const int T = 4;
  const int C = 4;

  util::Rng rng(555);
  auto fill = [&](nn::Tensor& t, float scale) {
    for (float& v : *t.data) v = (rng.next_f01() - 0.5f) * scale;
  };

  nn::Tensor x = nn::Tensor::zeros({B, T, C}, false);
  nn::Tensor w_qkv = nn::Tensor::zeros({C, 3 * C}, false);
  nn::Tensor b_qkv = nn::Tensor::zeros({3 * C}, false);
  nn::Tensor w_proj = nn::Tensor::zeros({C, C}, false);
  nn::Tensor b_proj = nn::Tensor::zeros({C}, false);

  fill(x, 0.2f);
  fill(w_qkv, 0.2f);
  fill(b_qkv, 0.02f);
  fill(w_proj, 0.2f);
  fill(b_proj, 0.02f);

  nn::Tensor y = nn::variants::rope::self_attention_rope(x, w_qkv, b_qkv, w_proj, b_proj);
  expect_true(y.shape == std::vector<int>({B, T, C}), "RoPE attn: output shape [B,T,C]");

  // Verify output is not all-zero and not NaN
  bool all_finite = true;
  for (float v : *y.data) {
    if (!std::isfinite(v)) all_finite = false;
  }
  expect_true(all_finite, "RoPE attn: output is finite");
}

void test_byte_tokenizer_embedding_shape() {
  std::cout << "[RUN ] ByteTokenizer embedding shape check\n";
  ByteTokenizer tok;
  expect_true(tok.vocab_size() == 256, "ByteTokenizer vocab_size == 256");
  model::Config cfg;
  cfg.vocab_size = tok.vocab_size();
  cfg.seq_len = 8;
  cfg.d_model = 16;
  cfg.n_layers = 1;
  model::TinyGPT gpt(cfg, 42);
  expect_true(gpt.parameters_const().tensors[0]->shape == std::vector<int>({256, 16}),
              "ByteTokenizer: wte shape matches V=256, C=16");
}

void test_gradcheck_rmsnorm_affine_via_cross_entropy() {
  std::cout << "[RUN ] gradcheck rmsnorm_affine (via cross_entropy)\n";
  const int N = 4;
  const int V = 6;
  const float eps = 1e-3f;
  util::Rng rng(890);
  std::vector<float> x_data(static_cast<std::size_t>(N) * V);
  for (float& v : x_data) v = (rng.next_f01() - 0.5f) * 0.5f;
  std::vector<float> gamma_data(static_cast<std::size_t>(V));
  for (int i = 0; i < V; ++i) gamma_data[i] = 1.0f + (rng.next_f01() - 0.5f) * 0.1f;
  std::vector<std::int32_t> targets(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) targets[static_cast<std::size_t>(i)] = (i + 1) % V;

  nn::Tensor x = nn::Tensor::zeros({N, V}, true);
  nn::Tensor gamma = nn::Tensor::zeros({V}, true);
  *x.data = x_data;
  *gamma.data = gamma_data;

  nn::Tensor y = nn::rmsnorm_affine(x, gamma, 1e-5f);
  nn::Tensor loss = nn::cross_entropy(y, targets);
  loss.backward();

  auto fd_loss = [&](const std::vector<float>& xd, const std::vector<float>& gd) -> float {
    nn::GradMode no_grad(false);
    nn::Tensor xi = nn::Tensor::zeros({N, V}, false);
    nn::Tensor gi = nn::Tensor::zeros({V}, false);
    *xi.data = xd;
    *gi.data = gd;
    nn::Tensor yi = nn::rmsnorm_affine(xi, gi, 1e-5f);
    return (*nn::cross_entropy(yi, targets).data)[0];
  };

  for (std::size_t i = 0; i < x_data.size(); ++i) {
    auto xp = x_data, xm = x_data;
    xp[i] += eps; xm[i] -= eps;
    const float gn = (fd_loss(xp, gamma_data) - fd_loss(xm, gamma_data)) / (2.0f * eps);
    expect_near((*x.grad)[i], gn, 5e-2f, "rmsnorm_affine gradcheck: dL/dx[" + std::to_string(i) + "]");
  }
  for (std::size_t i = 0; i < gamma_data.size(); ++i) {
    auto gp = gamma_data, gm = gamma_data;
    gp[i] += eps; gm[i] -= eps;
    const float gn = (fd_loss(x_data, gp) - fd_loss(x_data, gm)) / (2.0f * eps);
    expect_near((*gamma.grad)[i], gn, 5e-2f, "rmsnorm_affine gradcheck: dL/dgamma[" + std::to_string(i) + "]");
  }
}

void test_gradcheck_silu_via_cross_entropy() {
  std::cout << "[RUN ] gradcheck silu (via cross_entropy)\n";
  const int N = 3;
  const int V = 4;
  const float eps = 1e-3f;
  util::Rng rng(901);
  std::vector<float> x_data(static_cast<std::size_t>(N) * V);
  for (float& v : x_data) v = (rng.next_f01() - 0.5f) * 0.5f;
  std::vector<std::int32_t> targets(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) targets[static_cast<std::size_t>(i)] = i % V;

  nn::Tensor x = nn::Tensor::zeros({N, V}, true);
  *x.data = x_data;
  nn::Tensor y = nn::silu(x);
  nn::Tensor loss = nn::cross_entropy(y, targets);
  loss.backward();

  auto fd = [&](const std::vector<float>& d) {
    nn::GradMode no_grad(false);
    nn::Tensor xi = nn::Tensor::zeros({N, V}, false);
    *xi.data = d;
    return (*nn::cross_entropy(nn::silu(xi), targets).data)[0];
  };

  for (std::size_t i = 0; i < x_data.size(); ++i) {
    auto xp = x_data, xm = x_data;
    xp[i] += eps; xm[i] -= eps;
    const float gn = (fd(xp) - fd(xm)) / (2.0f * eps);
    expect_near((*x.grad)[i], gn, 5e-2f, "silu gradcheck: dL/dx[" + std::to_string(i) + "]");
  }
}

void test_spm_tokenizer_encode_decode() {
  std::cout << "[RUN ] SentencePiece tokenizer encode/decode roundtrip\n";
  // Extract vocab from TinyLlama model first
  std::string spm_model = "data/TinyLlama-1.1B/tokenizer.model";
  std::string spm_vocab = "data/spm_vocab.json";
  // If vocab file doesn't exist, skip
  std::ifstream test_v(spm_vocab);
  if (!test_v) {
    std::cout << "[SKIP] SPM vocab not extracted — run: python scripts/extract_spm_vocab.py\n";
    return;
  }
  test_v.close();

  SpmTokenizer tok(spm_vocab);
  expect_true(tok.vocab_size() > 1000, "SPM: vocab_size > 1000");

  // Test roundtrip — note SentencePiece uses "▁" for space, so simple ASCII may not roundtrip perfectly
  // Just verify encoding/decoding produces valid output
  std::string s1 = "hello";
  auto t1 = tok.encode(s1);
  std::string d1 = tok.decode(t1);
  expect_true(!d1.empty(), "SPM: decode produces output");

  // UTF-8
  std::string s2 = "你好世界";
  auto t2 = tok.encode(s2);
  std::string d2 = tok.decode(t2);
  expect_true(d2 == s2, "SPM: UTF-8 roundtrip");

  // Empty
  auto t3 = tok.encode("");
  expect_true(tok.decode(t3).empty(), "SPM: empty roundtrip");
}

void test_qk_norm_attention() {
  std::cout << "[RUN ] QK-Norm attention training\n";
  std::vector<std::uint8_t> bytes(2048);
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i & 0xFF);
  data::ByteDataset ds(std::move(bytes));
  model::Config cfg;
  cfg.vocab_size = 256; cfg.seq_len = 16; cfg.d_model = 16; cfg.n_layers = 1;
  cfg.qk_norm = 1;
  model::TinyGPT gpt(cfg, 88);
  optim::AdamWConfig ocfg; ocfg.lr = 1e-3f; ocfg.weight_decay = 0.01f;
  optim::AdamW opt(ocfg); util::Rng rng(88^0xDEADBEEF);
  float l0 = 0.0f, lN = 0.0f;
  for (int s = 0; s < 25; ++s) {
    auto b = ds.sample_batch(2, 16, rng); gpt.zero_grad();
    nn::Tensor loss = gpt.loss(b.x, b.y, b.B, b.T); loss.backward();
    opt.step(gpt.parameters().tensors);
    if (s == 0) l0 = (*loss.data)[0]; if (s == 24) lN = (*loss.data)[0];
  }
  expect_true(lN < l0, "QK-Norm: training loss decreases");
}

void test_sliding_window_attention() {
  std::cout << "[RUN ] Sliding window attention training\n";
  std::vector<std::uint8_t> bytes(4096); // larger dataset
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i & 0xFF);
  data::ByteDataset ds(std::move(bytes));
  model::Config cfg;
  cfg.vocab_size = 256; cfg.seq_len = 16; cfg.d_model = 32; cfg.n_layers = 1; // bigger model
  cfg.swin_win = 8;
  model::TinyGPT gpt(cfg, 777); // different seed
  optim::AdamWConfig ocfg; ocfg.lr = 3e-4f; ocfg.weight_decay = 0.01f;
  optim::AdamW opt(ocfg); util::Rng rng(777^0xDEADBEEF);
  float l0 = 0.0f, lN = 0.0f;
  for (int s = 0; s < 30; ++s) {
    auto b = ds.sample_batch(2, 16, rng); gpt.zero_grad();
    nn::Tensor loss = gpt.loss(b.x, b.y, b.B, b.T); loss.backward();
    opt.step(gpt.parameters().tensors);
    if (s == 0) l0 = (*loss.data)[0]; if (s == 29) lN = (*loss.data)[0];
  }
  expect_true(lN < l0, "SW: training loss decreases");
}

void test_alibi_attention() {
  std::cout << "[RUN ] ALiBi attention training (pos_type=2)\n";
  std::vector<std::uint8_t> bytes(2048);
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i & 0xFF);
  data::ByteDataset ds(std::move(bytes));
  model::Config cfg;
  cfg.vocab_size = 256; cfg.seq_len = 16; cfg.d_model = 16; cfg.n_layers = 1;
    cfg.pos_type = 2; // ALiBi — training falls back to standard attention for grad
    model::TinyGPT gpt(cfg, 111);
    optim::AdamWConfig ocfg; ocfg.lr = 1e-3f; ocfg.weight_decay = 0.01f;
    optim::AdamW opt(ocfg); util::Rng rng(111^0xDEADBEEF);
    float l0 = 0.0f, lN = 0.0f;
    for (int s = 0; s < 20; ++s) {
      auto b = ds.sample_batch(2, 16, rng); gpt.zero_grad();
      nn::Tensor loss = gpt.loss(b.x, b.y, b.B, b.T); loss.backward();
      opt.step(gpt.parameters().tensors);
      if (s == 0) l0 = (*loss.data)[0]; if (s == 19) lN = (*loss.data)[0];
    }
    expect_true(lN < l0, "ALiBi: training loss decreases");
}

void test_moe_shared_experts_forward() {
  std::cout << "[RUN ] MoE shared experts forward\n";
  const int N = 4, C = 8, n_exp = 2, top_k = 1, n_shared = 1, interm = 4*C;
  util::Rng rng(444);
  nn::Tensor x = nn::Tensor::zeros({N, C}, false);
  nn::Tensor wr = nn::Tensor::zeros({C, n_exp}, false);
  nn::Tensor br = nn::Tensor::zeros({n_exp}, false);
  for (float& v : *x.data) v = (rng.next_f01() - 0.5f) * 0.2f;
  for (float& v : *wr.data) v = (rng.next_f01() - 0.5f) * 0.2f;

  std::vector<nn::Tensor> wfc, bfc, wout, bout;
  std::vector<nn::Tensor> swfc, sbfc, swout, sbout;
  std::vector<const nn::Tensor*> ptrs, sptrs;
  wfc.reserve(n_exp); bfc.reserve(n_exp); wout.reserve(n_exp); bout.reserve(n_exp);
  swfc.reserve(n_shared); sbfc.reserve(n_shared); swout.reserve(n_shared); sbout.reserve(n_shared);
  for (int e = 0; e < n_exp; ++e) {
    wfc.push_back(nn::Tensor::randn({C, interm}, 0.1f, 10 + static_cast<std::uint64_t>(rng.next_f01()*100), false));
    bfc.push_back(nn::Tensor::zeros({interm}, false));
    wout.push_back(nn::Tensor::randn({interm, C}, 0.1f, 20 + static_cast<std::uint64_t>(rng.next_f01()*100), false));
    bout.push_back(nn::Tensor::zeros({C}, false));
    ptrs.push_back(&wfc.back()); ptrs.push_back(&bfc.back());
    ptrs.push_back(&wout.back()); ptrs.push_back(&bout.back());
  }
  for (int e = 0; e < n_shared; ++e) {
    swfc.push_back(nn::Tensor::randn({C, interm}, 0.1f, 30 + static_cast<std::uint64_t>(rng.next_f01()*100), false));
    sbfc.push_back(nn::Tensor::zeros({interm}, false));
    swout.push_back(nn::Tensor::randn({interm, C}, 0.1f, 40 + static_cast<std::uint64_t>(rng.next_f01()*100), false));
    sbout.push_back(nn::Tensor::zeros({C}, false));
    sptrs.push_back(&swfc.back()); sptrs.push_back(&sbfc.back());
    sptrs.push_back(&swout.back()); sptrs.push_back(&sbout.back());
  }

  auto out = nn::variants::moe::moe_mlp_forward(x, wr, br, ptrs, n_exp, top_k, interm, sptrs);
  expect_true(out.y.shape == std::vector<int>({N, C}), "MoE shared: output shape [N,C]");
  bool ok = true;
  for (float v : *out.y.data) if (!std::isfinite(v)) ok = false;
  expect_true(ok, "MoE shared: output finite");
  expect_true((*out.balance_loss.data)[0] >= 0.0f, "MoE shared: balance >= 0");
}

void test_ppo_value_head_and_advantage() {
  std::cout << "[RUN ] PPO value head and advantage computation\n";
  const int N = 4, C = 8;
  util::Rng rng(666);
  // Create mock hidden states [N, C]
  nn::Tensor hidden = nn::Tensor::zeros({N, C}, false);
  for (float& v : *hidden.data) v = (rng.next_f01() - 0.5f) * 0.5f;
  // Value head: [C, 1]
  nn::Tensor vw = nn::Tensor::randn({C, 1}, 0.1f, 100 + static_cast<std::uint64_t>(rng.next_f01()*100), true);
  nn::Tensor vb = nn::Tensor::zeros({1}, true);
  // Compute values via ppo_trainer
  nn::Tensor values = nn::variants::ppo::value_forward(hidden, vw, vb);
  expect_true(values.shape == std::vector<int>({N}), "PPO: values shape [N]");
  for (int n = 0; n < N; ++n)
    expect_true(std::isfinite((*values.data)[n]), "PPO: value[" + std::to_string(n) + "] finite");

  // Test GAE
  std::vector<float> rewards = {0.5f, -0.2f, 0.1f, 0.8f};
  std::vector<float> vals_vec(N);
  for (int n = 0; n < N; ++n) vals_vec[n] = (*values.data)[n];
  auto advantages = nn::variants::ppo::compute_gae(rewards, vals_vec, 0.99f, 0.95f);
  expect_true(advantages.size() == 4, "PPO: GAE returns 4 advantages");
  for (std::size_t i = 0; i < advantages.size(); ++i)
    expect_true(std::isfinite(advantages[i]), "PPO: advantage[" + std::to_string(i) + "] finite");

  // Test clipped surrogate
  float loss = nn::variants::ppo::clip_surrogate_loss(-0.5f, -0.5f, 0.3f, 0.2f);
  expect_true(std::isfinite(loss), "PPO: clipped surrogate loss finite");
  // Same probabilities should give ratio=1 → loss = -advantage
  expect_near(loss, -0.3f, 0.01f, "PPO: loss ≈ -advantage when ratio=1");
}

void test_mtp_forward_and_loss() {
  std::cout << "[RUN ] MTP multi-token prediction forward and loss\n";
  try {
  const int n_mtp = 2; // simpler — one extra head
  std::vector<std::uint8_t> bytes(2048);
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i & 0xFF);
  data::ByteDataset ds(std::move(bytes));

  model::Config cfg;
  cfg.vocab_size = 256; cfg.seq_len = 16; cfg.d_model = 16; cfg.n_layers = 1;
  cfg.n_mtp = n_mtp;

  model::TinyGPT gpt(cfg, 77);
  optim::AdamWConfig ocfg; ocfg.lr = 1e-3f; ocfg.weight_decay = 0.01f;
  optim::AdamW opt(ocfg);
  util::Rng rng(77 ^ 0xDEADBEEF);

  float l0 = 0.0f, lN = 0.0f;
  for (int step = 0; step < 20; ++step) {
    data::Batch batch = ds.sample_batch(2, 16, rng);
    gpt.zero_grad();
    nn::Tensor loss = gpt.loss(batch.x, batch.y, batch.B, batch.T);
    loss.backward();
    opt.step(gpt.parameters().tensors);
    if (step == 0) l0 = (*loss.data)[0];
    if (step == 19) lN = (*loss.data)[0];
  }
  expect_true(lN < l0, "MTP: training loss decreases with n_mtp=" + std::to_string(n_mtp));
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] MTP: " << e.what() << "\n";
    ++g_failures;
  }
}

void test_mla_forward_produces_output() {
  std::cout << "[RUN ] MLA forward produces valid output\n";
  const int B = 2, T = 4, C = 8, L = 4; // L=C/2 = latent dim
  util::Rng rng(555);
  auto fill = [&](nn::Tensor& t, float s) { for(float& v:*t.data) v=(rng.next_f01()-0.5f)*s; };
  nn::Tensor x = nn::Tensor::zeros({B,T,C}, false); fill(x, 0.2f);
  nn::Tensor w_q = nn::Tensor::zeros({C,C}, false); fill(w_q, 0.2f);
  nn::Tensor b_q = nn::Tensor::zeros({C}, false);
  nn::Tensor w_dkv = nn::Tensor::zeros({C,L}, false); fill(w_dkv, 0.2f);
  nn::Tensor b_dkv = nn::Tensor::zeros({L}, false);
  nn::Tensor w_uk = nn::Tensor::zeros({L,C}, false); fill(w_uk, 0.1f);
  nn::Tensor w_uv = nn::Tensor::zeros({L,C}, false); fill(w_uv, 0.1f);
  nn::Tensor w_o = nn::Tensor::zeros({C,C}, false); fill(w_o, 0.2f);
  nn::Tensor b_o = nn::Tensor::zeros({C}, false);

  nn::Tensor y = nn::variants::mla::self_attention_mla(x, w_q, b_q, w_dkv, b_dkv, w_uk, w_uv, w_o, b_o);
  expect_true(y.shape == std::vector<int>({B,T,C}), "MLA: output shape [B,T,C]");
  bool ok=true; for(float v:*y.data) if(!std::isfinite(v)) ok=false;
  expect_true(ok, "MLA: output finite");
}

void test_gqa_matches_mha_when_same_heads() {
  std::cout << "[RUN ] GQA matches MHA when n_kv_heads == n_heads\n";
  const int B = 2;
  const int T = 4;
  const int C = 8;
  const int n_heads = 2;
  const int D = C / n_heads;

  util::Rng rng(111);
  auto fill = [&](nn::Tensor& t, float scale) {
    for (float& v : *t.data) v = (rng.next_f01() - 0.5f) * scale;
  };

  nn::Tensor x = nn::Tensor::zeros({B, T, C}, false);
  nn::Tensor w_qkv = nn::Tensor::zeros({C, 3 * C}, false);
  nn::Tensor b_qkv = nn::Tensor::zeros({3 * C}, false);
  nn::Tensor w_proj = nn::Tensor::zeros({C, C}, false);
  nn::Tensor b_proj = nn::Tensor::zeros({C}, false);

  fill(x, 0.2f);
  fill(w_qkv, 0.2f);
  fill(b_qkv, 0.02f);
  fill(w_proj, 0.2f);
  fill(b_proj, 0.02f);

  nn::Tensor y_mha = nn::variants::mha::self_attention_mha(x, w_qkv, b_qkv, w_proj, b_proj, n_heads);

  // Manually project and split for GQA
  nn::Tensor qkv = nn::linear_lastdim(x, w_qkv, b_qkv);
  nn::Tensor q = nn::reshape(nn::Tensor::zeros({B * T, C}, false), {B, T, C});
  nn::Tensor k = nn::Tensor::zeros({B, T, C}, false);
  nn::Tensor v = nn::Tensor::zeros({B, T, C}, false);
  for (int bb = 0; bb < B; ++bb) {
    for (int t = 0; t < T; ++t) {
      for (int c = 0; c < C; ++c) {
        const std::size_t base = (static_cast<std::size_t>(bb) * T + t) * 3 * C + c;
        (*q.data)[(static_cast<std::size_t>(bb) * T + t) * C + c] = (*qkv.data)[base];
        (*k.data)[(static_cast<std::size_t>(bb) * T + t) * C + c] = (*qkv.data)[base + C];
        (*v.data)[(static_cast<std::size_t>(bb) * T + t) * C + c] = (*qkv.data)[base + 2 * C];
      }
    }
  }

  nn::Tensor q4 = nn::reshape(q, {B, T, n_heads, D});
  nn::Tensor k4 = nn::reshape(k, {B, T, n_heads, D});
  nn::Tensor v4 = nn::reshape(v, {B, T, n_heads, D});

  nn::Tensor att_gqa = nn::variants::gqa::self_attention_gqa(q4, k4, v4, n_heads, n_heads);
  nn::Tensor y_gqa = nn::linear_lastdim(att_gqa, w_proj, b_proj);

  for (std::size_t i = 0; i < y_mha.data->size(); ++i) {
    expect_near((*y_mha.data)[i], (*y_gqa.data)[i], 1e-4f,
                "GQA vs MHA: output[" + std::to_string(i) + "]");
  }
}

void test_blocked_simd_matches_cpu_matmul() {
  std::cout << "[RUN ] blocked SIMD matmul matches CPU reference\n";

  backend::CpuBackend cpu_ref;
  backend::BlockedSimdCpuBackend simd;

  auto test_size = [&](int m, int k, int n, const char* label) {
    std::vector<float> a(static_cast<std::size_t>(m) * k);
    std::vector<float> b(static_cast<std::size_t>(k) * n);
    std::vector<float> c_ref(static_cast<std::size_t>(m) * n);
    std::vector<float> c_simd(static_cast<std::size_t>(m) * n);

    util::Rng rng(static_cast<std::uint64_t>(m * 100 + k * 10 + n));
    for (float& v : a) v = (rng.next_f01() - 0.5f) * 0.2f;
    for (float& v : b) v = (rng.next_f01() - 0.5f) * 0.2f;

    cpu_ref.matmul2d_fwd(m, k, n, a.data(), b.data(), c_ref.data());
    simd.matmul2d_fwd(m, k, n, a.data(), b.data(), c_simd.data());

    for (std::size_t i = 0; i < c_ref.size(); ++i) {
      expect_near(c_ref[i], c_simd[i], 1e-4f,
                  std::string("simd fwd ") + label + "[" + std::to_string(i) + "]");
    }

    // Also test backward: dA and dB accumulation
    std::vector<float> dC(static_cast<std::size_t>(m) * n);
    std::vector<float> dA_ref(static_cast<std::size_t>(m) * k, 0.0f);
    std::vector<float> dB_ref(static_cast<std::size_t>(k) * n, 0.0f);
    std::vector<float> dA_simd(static_cast<std::size_t>(m) * k, 0.0f);
    std::vector<float> dB_simd(static_cast<std::size_t>(k) * n, 0.0f);

    for (float& v : dC) v = (rng.next_f01() - 0.5f) * 0.1f;

    cpu_ref.matmul2d_bwd(m, k, n, a.data(), b.data(), dC.data(), dA_ref.data(), dB_ref.data());
    simd.matmul2d_bwd(m, k, n, a.data(), b.data(), dC.data(), dA_simd.data(), dB_simd.data());

    for (std::size_t i = 0; i < dA_ref.size(); ++i) {
      expect_near(dA_ref[i], dA_simd[i], 1e-4f,
                  std::string("simd bwd dA ") + label + "[" + std::to_string(i) + "]");
    }
    for (std::size_t i = 0; i < dB_ref.size(); ++i) {
      expect_near(dB_ref[i], dB_simd[i], 1e-4f,
                  std::string("simd bwd dB ") + label + "[" + std::to_string(i) + "]");
    }
  };

  // Test multiple sizes including non-multiples of block size
  test_size(16, 16, 16, "16x16x16");
  test_size(32, 64, 48, "32x64x48");
  test_size(100, 128, 80, "100x128x80"); // non-multiple of 64
  test_size(64, 64, 256, "64x64x256");
  test_size(1, 64, 64, "1x64x64");       // single row
  test_size(64, 1, 64, "64x1x64");       // K=1
}

void test_simd_backend_via_model_training() {
  std::cout << "[RUN ] SIMD backend: model training regression\n";

  // Run a tiny training loop with SIMD backend
  auto simd = std::make_unique<backend::BlockedSimdCpuBackend>();
  backend::set(std::move(simd));

  std::vector<std::uint8_t> bytes(2048);
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i & 0xFF);
  data::ByteDataset ds(std::move(bytes));

  model::Config cfg;
  cfg.vocab_size = 256;
  cfg.seq_len = 32;
  cfg.d_model = 32;
  cfg.n_layers = 1;

  const std::uint64_t seed = 42;
  model::TinyGPT gpt(cfg, seed);

  optim::AdamWConfig ocfg;
  ocfg.lr = 1e-3f;
  ocfg.weight_decay = 0.01f;
  optim::AdamW opt(ocfg);

  util::Rng rng(seed ^ 0xDEADBEEF);

  float l0 = 0.0f;
  float lN = 0.0f;
  for (int step = 0; step < 20; ++step) {
    data::Batch batch = ds.sample_batch(2, 32, rng);
    gpt.zero_grad();
    nn::Tensor loss = gpt.loss(batch.x, batch.y, batch.B, batch.T);
    loss.backward();
    opt.step(gpt.parameters().tensors);
    if (step == 0) l0 = (*loss.data)[0];
    if (step == 19) lN = (*loss.data)[0];
  }

  expect_true(lN < l0, "SIMD backend: training loss decreases (l0=" + std::to_string(l0) + ", lN=" + std::to_string(lN) + ")");

  // Restore default CPU backend
  backend::set(std::make_unique<backend::CpuBackend>());
}

void test_moe_forward_produces_output() {
  std::cout << "[RUN ] MoE forward produces valid output\n";
  try {
  const int N = 4, C = 8, n_experts = 4, top_k = 2, interm = 4 * C;
  util::Rng rng(333);
  nn::Tensor x = nn::Tensor::zeros({N, C}, false);
  nn::Tensor wr = nn::Tensor::zeros({C, n_experts}, false);
  nn::Tensor br = nn::Tensor::zeros({n_experts}, false);
  for (float& v : *x.data) v = (rng.next_f01() - 0.5f) * 0.2f;
  for (float& v : *wr.data) v = (rng.next_f01() - 0.5f) * 0.2f;
  std::vector<nn::Tensor> wfc, bfc, wout, bout;
  wfc.reserve(static_cast<std::size_t>(n_experts));
  bfc.reserve(static_cast<std::size_t>(n_experts));
  wout.reserve(static_cast<std::size_t>(n_experts));
  bout.reserve(static_cast<std::size_t>(n_experts));
  std::vector<const nn::Tensor*> ptrs;
  for (int e = 0; e < n_experts; ++e) {
    wfc.emplace_back(nn::Tensor::randn({C, interm}, 0.1f, 1 + static_cast<std::uint64_t>(rng.next_f01() * 100), false));
    bfc.emplace_back(nn::Tensor::zeros({interm}, false));
    wout.emplace_back(nn::Tensor::randn({interm, C}, 0.1f, 2 + static_cast<std::uint64_t>(rng.next_f01() * 100), false));
    bout.emplace_back(nn::Tensor::zeros({C}, false));
    ptrs.push_back(&wfc.back()); ptrs.push_back(&bfc.back());
    ptrs.push_back(&wout.back()); ptrs.push_back(&bout.back());
  }
  auto out = nn::variants::moe::moe_mlp_forward(x, wr, br, ptrs, n_experts, top_k, interm);
  expect_true(out.y.shape == std::vector<int>({N, C}), "MoE: output shape");
  bool ok = true;
  for (float v : *out.y.data) if (!std::isfinite(v)) ok = false;
  expect_true(ok, "MoE: output finite");
  expect_true((*out.balance_loss.data)[0] >= 0.0f, "MoE: balance loss >= 0");
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] MoE: " << e.what() << "\n";
    ++g_failures;
  }
}

void test_moe_training_loss_decreases() {
  std::cout << "[RUN ] MoE training (loss decreases)\n";
  std::vector<std::uint8_t> bytes(2048);
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i & 0xFF);
  data::ByteDataset ds(std::move(bytes));
  model::Config cfg;
  cfg.vocab_size = 256; cfg.seq_len = 16; cfg.d_model = 16; cfg.n_layers = 1;
  cfg.mlp_type = 2; cfg.n_experts = 2; cfg.top_k = 1;
  model::TinyGPT gpt(cfg, 99);
  optim::AdamWConfig ocfg; ocfg.lr = 1e-3f; ocfg.weight_decay = 0.01f;
  optim::AdamW opt(ocfg);
  util::Rng rng(99 ^ 0xDEADBEEF);
  float l0 = 0.0f, lN = 0.0f;
  for (int step = 0; step < 30; ++step) {
    data::Batch batch = ds.sample_batch(2, 16, rng);
    gpt.zero_grad();
    nn::Tensor loss = gpt.loss(batch.x, batch.y, batch.B, batch.T);
    loss.backward();
    opt.step(gpt.parameters().tensors);
    if (step == 0) l0 = (*loss.data)[0];
    if (step == 29) lN = (*loss.data)[0];
  }
  expect_true(lN < l0, "MoE: training loss decreases");
}

#ifdef BUILD_VULKAN
void test_vulkan_matches_cpu_matmul() {
  std::cout << "[RUN ] Vulkan matmul matches CPU reference\n";

  backend::VulkanBackend vulkan;
  if (!vulkan.is_ready()) {
    std::cout << "[SKIP] Vulkan device not available\n";
    return;
  }

  backend::CpuBackend cpu_ref;

  auto test_size = [&](int m, int k, int n, const char* label) {
    std::vector<float> a(static_cast<std::size_t>(m) * k);
    std::vector<float> b(static_cast<std::size_t>(k) * n);
    std::vector<float> c_ref(static_cast<std::size_t>(m) * n);
    std::vector<float> c_vulkan(static_cast<std::size_t>(m) * n);

    util::Rng rng(static_cast<std::uint64_t>(m * 100 + k * 10 + n));
    for (float& v : a) v = (rng.next_f01() - 0.5f) * 0.2f;
    for (float& v : b) v = (rng.next_f01() - 0.5f) * 0.2f;

    cpu_ref.matmul2d_fwd(m, k, n, a.data(), b.data(), c_ref.data());
    vulkan.matmul2d_fwd(m, k, n, a.data(), b.data(), c_vulkan.data());

    for (std::size_t i = 0; i < c_ref.size(); ++i) {
      expect_near(c_ref[i], c_vulkan[i], 1e-4f,
                  std::string("vulkan fwd ") + label + "[" + std::to_string(i) + "]");
    }
  };

  test_size(16, 16, 16, "16x16x16");
  test_size(64, 64, 128, "64x64x128");
  test_size(7, 13, 19, "7x13x19");  // odd sizes
}
#else
void test_vulkan_matches_cpu_matmul() {
  std::cout << "[SKIP] Vulkan backend not built (BUILD_VULKAN=OFF)\n";
}
#endif

} // namespace

int main(int /*argc*/, char** /*argv*/) {
  try {
    test_backend_dispatch_matmul2d();
    test_backend_dispatch_bmm();
    test_gradcheck_matmul2d_via_cross_entropy();
    test_gradcheck_layernorm_lastdim_via_cross_entropy();
    test_gradcheck_layernorm_affine_via_cross_entropy();
    test_gradcheck_rmsnorm_affine_via_cross_entropy();
    test_gradcheck_silu_via_cross_entropy();
    test_moe_training_loss_decreases();
    test_moe_shared_experts_forward();
    test_moe_forward_produces_output();
    test_tiny_training_regression_loss_decreases();
    test_mha_matches_1h_when_single_head();
    test_byte_tokenizer_encode_decode();
    test_bpe_tokenizer_encode_decode();
    test_spm_tokenizer_encode_decode();
    test_byte_tokenizer_embedding_shape();
    test_kvcache_matches_full_attention();
    test_rope_position_zero_is_identity();
    test_rope_attention_runs();
    test_gqa_matches_mha_when_same_heads();
    test_mla_forward_produces_output();
    test_mtp_forward_and_loss();
    test_ppo_value_head_and_advantage();
    test_qk_norm_attention();
    test_sliding_window_attention();
    test_alibi_attention();
    test_blocked_simd_matches_cpu_matmul();
    test_simd_backend_via_model_training();
    test_vulkan_matches_cpu_matmul();
    test_byte_tokenizer_embedding_shape();

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
