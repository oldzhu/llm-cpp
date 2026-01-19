#include <chrono>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "data.h"
#include "checkpoint.h"
#include "model.h"
#include "optim.h"
#include "util.h"

struct Args {
  std::string data_path;
  int steps = 200;
  int batch = 4;
  int seq = 64;
  int dmodel = 64;
  int layers = 1;
  float lr = 3e-4f;
  std::uint64_t seed = 1;

  // Checkpoints
  std::string load_prefix;
  std::string save_prefix;
  bool save_optim = true;

  // Sampling
  std::string prompt;
  int gen_tokens = 128;
  float temperature = 1.0f;
  int topk = 0; // 0 = disabled

  // Debug / sanity checks
  int print_next_top = 0; // 0 = disabled; prints once before the first generated token
  bool print_next_each_step = false;

  // Dataset-based sanity check: sample contexts from the data file and evaluate next-byte prediction.
  int sanity_next_from_data = 0; // 0 = disabled
  int sanity_ctx = 0;            // 0 = use cfg.seq_len
  int sanity_top = 10;

  bool ascii_only = false;
  bool escape_bytes = false;
};

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return std::string(argv[++i]);
    };

    if (k == "--data") a.data_path = need("--data");
    else if (k == "--steps") a.steps = std::stoi(need("--steps"));
    else if (k == "--batch") a.batch = std::stoi(need("--batch"));
    else if (k == "--seq") a.seq = std::stoi(need("--seq"));
    else if (k == "--dmodel") a.dmodel = std::stoi(need("--dmodel"));
    else if (k == "--layers") a.layers = std::stoi(need("--layers"));
    else if (k == "--lr") a.lr = std::stof(need("--lr"));
    else if (k == "--seed") a.seed = static_cast<std::uint64_t>(std::stoull(need("--seed")));
    else if (k == "--load") a.load_prefix = need("--load");
    else if (k == "--save") a.save_prefix = need("--save");
    else if (k == "--save-opt") a.save_optim = (std::stoi(need("--save-opt")) != 0);
    else if (k == "--prompt") a.prompt = need("--prompt");
    else if (k == "--gen") a.gen_tokens = std::stoi(need("--gen"));
    else if (k == "--temp") a.temperature = std::stof(need("--temp"));
    else if (k == "--topk") a.topk = std::stoi(need("--topk"));
    else if (k == "--print-next-top") a.print_next_top = std::stoi(need("--print-next-top"));
    else if (k == "--print-next-top-each-step") a.print_next_each_step = (std::stoi(need("--print-next-top-each-step")) != 0);
    else if (k == "--sanity-next-from-data") a.sanity_next_from_data = std::stoi(need("--sanity-next-from-data"));
    else if (k == "--sanity-ctx") a.sanity_ctx = std::stoi(need("--sanity-ctx"));
    else if (k == "--sanity-top") a.sanity_top = std::stoi(need("--sanity-top"));
    else if (k == "--ascii-only") a.ascii_only = (std::stoi(need("--ascii-only")) != 0);
    else if (k == "--escape-bytes") a.escape_bytes = (std::stoi(need("--escape-bytes")) != 0);
    else if (k == "--help" || k == "-h") {
      std::cout
          << "Usage:\n"
          << "  train_gpt --data <path> [--steps N] [--batch B] [--seq T] [--dmodel C] [--layers L] [--lr LR] [--seed S] [--save PREFIX]\n"
          << "  train_gpt --load PREFIX [--steps N] [--data <path>] [--save PREFIX] [--save-opt 0|1]\n"
            << "  train_gpt [--data <path> --steps N ...] --prompt <text> [--gen N] [--temp X] [--topk K] [--print-next-top N] [--print-next-top-each-step 0|1] [--ascii-only 0|1] [--escape-bytes 0|1] [--load PREFIX]\n"
            << "  train_gpt --data <path> --steps 0 --sanity-next-from-data N [--sanity-ctx T] [--sanity-top K] [--load PREFIX]\n\n"
          << "Notes:\n"
          << "- If --prompt is set, the program will (optionally) train first (if --steps > 0) and then generate text.\n"
          << "- Tokenization is byte-level (vocab=256).\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + k);
    }
  }
  if (a.steps > 0 && a.data_path.empty()) {
    throw std::runtime_error("--data is required when --steps > 0");
  }
  if (a.sanity_next_from_data > 0 && a.data_path.empty()) {
    throw std::runtime_error("--data is required when --sanity-next-from-data > 0");
  }
  if (a.gen_tokens < 0) throw std::runtime_error("--gen must be >= 0");
  if (a.temperature <= 0.0f) throw std::runtime_error("--temp must be > 0");
  if (a.topk < 0) throw std::runtime_error("--topk must be >= 0");
  if (a.print_next_top < 0) throw std::runtime_error("--print-next-top must be >= 0");
  if (a.sanity_next_from_data < 0) throw std::runtime_error("--sanity-next-from-data must be >= 0");
  if (a.sanity_ctx < 0) throw std::runtime_error("--sanity-ctx must be >= 0");
  if (a.sanity_top < 0) throw std::runtime_error("--sanity-top must be >= 0");
  return a;
}

static std::vector<std::int32_t> encode_bytes(const std::string& s) {
  std::vector<std::int32_t> out;
  out.reserve(s.size());
  for (unsigned char ch : s) {
    out.push_back(static_cast<std::int32_t>(ch));
  }
  return out;
}

static int sample_from_logits(const float* logits, int V, util::Rng& rng, float temperature, int topk) {
  if (V <= 0) throw std::runtime_error("sample_from_logits: invalid V");
  if (temperature <= 0.0f) throw std::runtime_error("sample_from_logits: invalid temperature");

  // Optional top-k truncation
  std::vector<int> idx(V);
  for (int i = 0; i < V; ++i) idx[i] = i;

  if (topk > 0 && topk < V) {
    std::nth_element(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int b) {
      return logits[a] > logits[b];
    });
    idx.resize(static_cast<std::size_t>(topk));
  }

  float mx = -1e30f;
  for (int i : idx) mx = std::max(mx, logits[i] / temperature);
  float denom = 0.0f;
  std::vector<float> probs(idx.size());
  for (std::size_t j = 0; j < idx.size(); ++j) {
    const float v = std::exp((logits[idx[j]] / temperature) - mx);
    probs[j] = v;
    denom += v;
  }
  const float inv = 1.0f / denom;
  for (float& p : probs) p *= inv;

  const float r = rng.next_f01();
  float cdf = 0.0f;
  for (std::size_t j = 0; j < probs.size(); ++j) {
    cdf += probs[j];
    if (r < cdf) return idx[j];
  }
  return idx.back();
}

static bool is_ascii_allowed_byte(unsigned char b) {
  // Allow common whitespace plus printable ASCII range.
  if (b == '\n' || b == '\r' || b == '\t') return true;
  return (b >= 32 && b <= 126);
}

static std::vector<int> allowed_indices_ascii(int V) {
  std::vector<int> idx;
  idx.reserve(static_cast<std::size_t>(V));
  for (int i = 0; i < V; ++i) {
    const unsigned char b = static_cast<unsigned char>(i & 0xFF);
    if (is_ascii_allowed_byte(b)) idx.push_back(i);
  }
  // Fallback: if V is weird/small and whitelist is empty, allow all.
  if (idx.empty()) {
    idx.resize(static_cast<std::size_t>(V));
    for (int i = 0; i < V; ++i) idx[i] = i;
  }
  return idx;
}

static int sample_from_logits_filtered(const float* logits,
                                       int V,
                                       util::Rng& rng,
                                       float temperature,
                                       int topk,
                                       const std::vector<int>& allowed) {
  if (V <= 0) throw std::runtime_error("sample_from_logits_filtered: invalid V");
  if (temperature <= 0.0f) throw std::runtime_error("sample_from_logits_filtered: invalid temperature");
  if (allowed.empty()) throw std::runtime_error("sample_from_logits_filtered: empty allowed set");

  std::vector<int> idx = allowed;

  if (topk > 0 && topk < static_cast<int>(idx.size())) {
    std::nth_element(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int b) {
      return logits[a] > logits[b];
    });
    idx.resize(static_cast<std::size_t>(topk));
  }

  float mx = -1e30f;
  for (int i : idx) mx = std::max(mx, logits[i] / temperature);
  float denom = 0.0f;
  std::vector<float> probs(idx.size());
  for (std::size_t j = 0; j < idx.size(); ++j) {
    const float v = std::exp((logits[idx[j]] / temperature) - mx);
    probs[j] = v;
    denom += v;
  }
  const float inv = 1.0f / denom;
  for (float& p : probs) p *= inv;

  const float r = rng.next_f01();
  float cdf = 0.0f;
  for (std::size_t j = 0; j < probs.size(); ++j) {
    cdf += probs[j];
    if (r < cdf) return idx[j];
  }
  return idx.back();
}

static void print_generated_byte(int token, bool escape_bytes) {
  const unsigned char ch = static_cast<unsigned char>(token & 0xFF);
  if (!escape_bytes) {
    std::cout << static_cast<char>(ch);
    return;
  }

  // Preserve common whitespace in a readable way.
  if (ch == '\n') {
    std::cout << "\n";
    return;
  }
  if (ch == '\r') {
    std::cout << "\r";
    return;
  }
  if (ch == '\t') {
    std::cout << "\t";
    return;
  }

  if (ch >= 32 && ch <= 126) {
    std::cout << static_cast<char>(ch);
    return;
  }

  static const char* hex = "0123456789ABCDEF";
  std::cout << "\\x" << hex[(ch >> 4) & 0xF] << hex[ch & 0xF];
}

static std::string token_to_display(int token, bool escape_bytes) {
  const unsigned char ch = static_cast<unsigned char>(token & 0xFF);
  if (!escape_bytes) {
    if (ch >= 32 && ch <= 126) return std::string(1, static_cast<char>(ch));
    if (ch == '\n') return "\\n";
    if (ch == '\r') return "\\r";
    if (ch == '\t') return "\\t";
    return ".";
  }

  if (ch == '\n') return "\\n";
  if (ch == '\r') return "\\r";
  if (ch == '\t') return "\\t";
  if (ch >= 32 && ch <= 126) return std::string(1, static_cast<char>(ch));

  static const char* hex = "0123456789ABCDEF";
  std::string s;
  s.push_back('\\');
  s.push_back('x');
  s.push_back(hex[(ch >> 4) & 0xF]);
  s.push_back(hex[ch & 0xF]);
  return s;
}

static void print_next_token_distribution(const float* logits,
                                          int V,
                                          float temperature,
                                          int topn,
                                          bool ascii_only,
                                          bool escape_bytes) {
  if (topn <= 0) return;
  if (V <= 0) throw std::runtime_error("print_next_token_distribution: invalid V");
  if (temperature <= 0.0f) throw std::runtime_error("print_next_token_distribution: invalid temperature");

  const std::vector<int> allowed = ascii_only ? allowed_indices_ascii(V) : std::vector<int>();

  // Build candidate indices.
  std::vector<int> cand;
  if (ascii_only) {
    cand = allowed;
  } else {
    cand.resize(static_cast<std::size_t>(V));
    for (int i = 0; i < V; ++i) cand[static_cast<std::size_t>(i)] = i;
  }

  float mx = -1e30f;
  for (int i : cand) mx = std::max(mx, logits[i] / temperature);

  std::vector<float> probs(cand.size());
  float denom = 0.0f;
  for (std::size_t j = 0; j < cand.size(); ++j) {
    const float v = std::exp((logits[cand[j]] / temperature) - mx);
    probs[j] = v;
    denom += v;
  }
  const float inv = 1.0f / denom;
  for (float& p : probs) p *= inv;

  // Get top-N.
  std::vector<std::size_t> order(cand.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  const int N = std::min<int>(topn, static_cast<int>(order.size()));
  std::partial_sort(order.begin(), order.begin() + N, order.end(), [&](std::size_t a, std::size_t b) {
    return probs[a] > probs[b];
  });

  std::cout << "\nTop-" << N << " next tokens (after temperature, before sampling):\n";
  for (int r = 0; r < N; ++r) {
    const std::size_t j = order[static_cast<std::size_t>(r)];
    const int tok = cand[j];
    std::cout << "  " << r + 1 << ") id=" << tok << "  p=" << probs[j] << "  tok='" << token_to_display(tok, escape_bytes) << "'\n";
  }
  std::cout.flush();
}

static int argmax_index(const float* x, int n) {
  int best_i = 0;
  float best_v = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] > best_v) {
      best_v = x[i];
      best_i = i;
    }
  }
  return best_i;
}

static void sanity_check_next_from_data(model::TinyGPT& gpt,
                                        const std::vector<std::uint8_t>& bytes,
                                        int n_trials,
                                        int ctx_len,
                                        int topn,
                                        util::Rng& rng,
                                        bool escape_bytes) {
  if (n_trials <= 0) return;
  if (ctx_len <= 0) throw std::runtime_error("sanity_check_next_from_data: ctx_len must be > 0");
  if (bytes.size() < static_cast<std::size_t>(ctx_len + 1)) throw std::runtime_error("sanity_check_next_from_data: dataset too small for ctx_len");
  if (topn < 0) throw std::runtime_error("sanity_check_next_from_data: topn must be >= 0");

  const int V = gpt.cfg().vocab_size;
  if (V != 256) throw std::runtime_error("sanity_check_next_from_data: expected V=256 byte vocab");

  const std::int32_t max_start = static_cast<std::int32_t>(bytes.size() - static_cast<std::size_t>(ctx_len + 1));
  int correct_top1 = 0;

  nn::GradMode no_grad(false);
  for (int t = 0; t < n_trials; ++t) {
    const std::int32_t start = rng.uniform_int(0, max_start);
    std::vector<std::int32_t> ctx;
    ctx.resize(static_cast<std::size_t>(ctx_len));
    for (int i = 0; i < ctx_len; ++i) {
      ctx[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(bytes[static_cast<std::size_t>(start + i)]);
    }
    const int expected = static_cast<int>(bytes[static_cast<std::size_t>(start + ctx_len)]);

    nn::Tensor logits = gpt.forward_logits(ctx, 1, ctx_len); // [1,ctx_len,V]
    const std::size_t base = static_cast<std::size_t>(ctx_len - 1) * static_cast<std::size_t>(V);
    const float* row = logits.data->data() + base;

    const int pred = argmax_index(row, V);
    if (pred == expected) ++correct_top1;

    // Print a small readable context preview.
    std::cout << "\n[sanity " << (t + 1) << "/" << n_trials << "] start=" << start << " ctx_len=" << ctx_len << "\n";
    std::cout << "  expected next: id=" << expected << " tok='" << token_to_display(expected, escape_bytes) << "'\n";
    std::cout << "  pred top1:    id=" << pred << " tok='" << token_to_display(pred, escape_bytes) << "'\n";

    if (topn > 0) {
      // Use temperature=1 and no ASCII filtering for a pure next-byte check.
      print_next_token_distribution(row, V, 1.0f, topn, false, escape_bytes);
    }
  }

  std::cout << "\nSanity top1 accuracy: " << correct_top1 << "/" << n_trials
            << " = " << (static_cast<double>(correct_top1) / static_cast<double>(n_trials)) << "\n";
}

static void generate(model::TinyGPT& gpt,
                     const std::string& prompt,
                     int gen_tokens,
                     float temperature,
                     int topk,
                     int print_next_top,
                     bool print_next_each_step,
                     bool ascii_only,
                     bool escape_bytes,
                     util::Rng& rng) {
  std::vector<std::int32_t> tokens = encode_bytes(prompt);
  if (tokens.empty()) {
    // Start from a newline if no prompt is provided.
    tokens.push_back(static_cast<std::int32_t>('\n'));
  }

  std::cout << prompt;
  std::cout.flush();

  nn::GradMode no_grad(false);
  const int V = gpt.cfg().vocab_size;
  const std::vector<int> allowed = ascii_only ? allowed_indices_ascii(V) : std::vector<int>();

  for (int step = 0; step < gen_tokens; ++step) {
    const int maxT = gpt.cfg().seq_len;
    const int T = static_cast<int>(std::min<std::size_t>(tokens.size(), static_cast<std::size_t>(maxT)));
    std::vector<std::int32_t> ctx(tokens.end() - T, tokens.end());

    nn::Tensor logits = gpt.forward_logits(ctx, 1, T); // [1,T,V]
    const std::size_t base = static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(V);

    if (print_next_top > 0 && (print_next_each_step || step == 0)) {
      if (print_next_each_step) {
        std::cout << "\n[gen step " << step << "]";
      }
      print_next_token_distribution(logits.data->data() + base, V, temperature, print_next_top, ascii_only, escape_bytes);
    }

    const int next = ascii_only
                         ? sample_from_logits_filtered(logits.data->data() + base, V, rng, temperature, topk, allowed)
                         : sample_from_logits(logits.data->data() + base, V, rng, temperature, topk);
    tokens.push_back(next);

    print_generated_byte(next, escape_bytes);
    std::cout.flush();
  }
  std::cout << "\n";
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);

    std::uint64_t start_step = 0;

    model::Config cfg;
    cfg.vocab_size = 256;
    cfg.seq_len = args.seq;
    cfg.d_model = args.dmodel;
    cfg.n_layers = args.layers;

    optim::AdamWConfig ocfg;
    ocfg.lr = args.lr;
    ocfg.weight_decay = 0.01f;

    // If loading, override config from checkpoint JSON
    if (!args.load_prefix.empty()) {
      const ckpt::LoadedConfig lc = ckpt::read_config(args.load_prefix);
      cfg = lc.model;
      ocfg = lc.optim;
      start_step = lc.step;
    }

    std::vector<std::uint8_t> data_bytes;
    std::unique_ptr<data::ByteDataset> ds;
    if (!args.data_path.empty() && (args.steps > 0 || args.sanity_next_from_data > 0)) {
      data_bytes = util::read_file_bytes(args.data_path);
      if (args.steps > 0) {
        ds = std::make_unique<data::ByteDataset>(data_bytes);
      }
    }

    model::TinyGPT gpt(cfg, args.seed);
    optim::AdamW opt(ocfg);

    if (!args.load_prefix.empty()) {
      std::uint64_t loaded_step = 0;
      ckpt::load(args.load_prefix, gpt, opt, loaded_step);
      start_step = loaded_step;
      std::cout << "loaded checkpoint '" << args.load_prefix << "' at step " << start_step << "\n";
    }

    util::Rng rng(args.seed ^ 0xDEADBEEF);

    if (args.steps > 0) {
      if (!ds) throw std::runtime_error("internal: dataset not initialized");
      const int train_seq = std::min(args.seq, gpt.cfg().seq_len);
      const auto t0 = std::chrono::high_resolution_clock::now();
      for (int local = 1; local <= args.steps; ++local) {
        const std::uint64_t step = start_step + static_cast<std::uint64_t>(local);
        data::Batch batch = ds->sample_batch(args.batch, train_seq, rng);

        gpt.zero_grad();
        nn::Tensor loss = gpt.loss(batch.x, batch.y, batch.B, batch.T);
        loss.backward();

        opt.step(gpt.parameters().tensors);

        if (local == 1 || local % 10 == 0) {
          const auto tn = std::chrono::high_resolution_clock::now();
          const double sec = std::chrono::duration<double>(tn - t0).count();
          std::cout << "step " << step << "  (" << local << "/" << args.steps << ")  loss=" << (*loss.data)[0] << "  time=" << sec << "s\n";
        }
      }

      start_step += static_cast<std::uint64_t>(args.steps);
    }

    if (args.sanity_next_from_data > 0) {
      const int ctx_len = (args.sanity_ctx > 0) ? args.sanity_ctx : gpt.cfg().seq_len;
      if (ctx_len > gpt.cfg().seq_len) throw std::runtime_error("--sanity-ctx exceeds model seq_len");
      util::Rng srng(args.seed ^ 0x51514E455854ULL);
      sanity_check_next_from_data(gpt, data_bytes, args.sanity_next_from_data, ctx_len, args.sanity_top, srng, args.escape_bytes);
    }

    if (!args.save_prefix.empty()) {
      const bool save_opt = args.save_optim && (start_step > 0);
      ckpt::save(args.save_prefix, gpt, opt, start_step, save_opt);
      std::cout << "saved checkpoint '" << args.save_prefix << "' at step " << start_step
                << (save_opt ? " (with optimizer state)" : "") << "\n";
    }

    if (!args.prompt.empty()) {
      util::Rng grng(args.seed ^ 0xABCDEF123456ULL);
      generate(gpt,
               args.prompt,
               args.gen_tokens,
               args.temperature,
               args.topk,
               args.print_next_top,
               args.print_next_each_step,
               args.ascii_only,
               args.escape_bytes,
               grng);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
