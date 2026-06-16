#include <chrono>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>


#include "data.h"
#include "checkpoint.h"
#include "model.h"
#include "optim.h"
#include "util.h"

#include "tokenizer/byte_tokenizer.h"
#include "tokenizer/bpe_tokenizer.h"
#include "tokenizer/spm_tokenizer.h"
#include "variants/kvcache/kvcache_attention.h"

struct Args {
    // Tokenizer
    std::string tokenizer_type = "byte"; // "byte" or "bpe" or "sp" (SentencePiece)
    std::string bpe_vocab_path;
    std::string bpe_merges_path;
  std::string data_path;
  int steps = 200;
  int batch = 4;
  int seq = 64;
  int dmodel = 64;
  int layers = 1;
  float lr = 3e-4f;
  std::uint64_t seed = 1;
  std::string norm_type = "layernorm"; // "layernorm" or "rmsnorm"
  std::string mlp_type = "gelu";       // "gelu" or "swiglu" or "moe"
  std::string attn_type = "1head";     // "1head", "mha", "gqa", "mla"
  std::string pos_type = "wpe";        // "wpe", "rope"
  int attn_n_heads = 1;
  int attn_n_kv = 1;

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

  // Dataset-based sanity check at a specific byte offset.
  // The context is bytes[offset .. offset+ctx_len-1], and the expected next byte is bytes[offset+ctx_len].
  std::int64_t sanity_offset = -1; // -1 = disabled
  int sanity_preview = 120;        // how many context bytes to print (from the end)

  bool ascii_only = false;
  bool escape_bytes = false;
  bool use_kvcache = false;
  bool debug_layer = false;
  std::string log_loss_path;
  std::string tokenize_output; // tokenize data and write token IDs here
  std::string token_data;      // load pre-tokenized token IDs from here
  bool serve_mode = false;     // interactive stdin→stdout JSON generation loop

  // Web UI integration
  bool progress_json = false;
  int  progress_grads = 0;
  bool pipe_stdin = false;
  std::string dump_weights_path;
  std::string dump_weights_format = "binary";
  int save_interval = 0;
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
    else if (k == "--sanity-offset") a.sanity_offset = static_cast<std::int64_t>(std::stoll(need("--sanity-offset")));
    else if (k == "--sanity-preview") a.sanity_preview = std::stoi(need("--sanity-preview"));
    else if (k == "--ascii-only") a.ascii_only = (std::stoi(need("--ascii-only")) != 0);
    else if (k == "--escape-bytes") a.escape_bytes = (std::stoi(need("--escape-bytes")) != 0);
    else if (k == "--kvcache") a.use_kvcache = true;
    else if (k == "--debug-layer") a.debug_layer = true;
    else if (k == "--log-loss") a.log_loss_path = need("--log-loss");
    else if (k == "--tokenize-output") a.tokenize_output = need("--tokenize-output");
    else if (k == "--token-data") a.token_data = need("--token-data");
    else if (k == "--progress-json") a.progress_json = true;
    else if (k == "--progress-grads") a.progress_grads = std::stoi(need("--progress-grads"));
    else if (k == "--pipe-stdin") a.pipe_stdin = true;
    else if (k == "--dump-weights") a.dump_weights_path = need("--dump-weights");
    else if (k == "--dump-weights-format") a.dump_weights_format = need("--dump-weights-format");
    else if (k == "--save-interval") a.save_interval = std::stoi(need("--save-interval"));
    else if (k == "--serve") a.serve_mode = true;
    else if (k == "--norm") a.norm_type = need("--norm");
    else if (k == "--mlp") a.mlp_type = need("--mlp");
    else if (k == "--attn") a.attn_type = need("--attn");
    else if (k == "--pos") a.pos_type = need("--pos");
    else if (k == "--n-heads") a.attn_n_heads = std::stoi(need("--n-heads"));
    else if (k == "--n-kv") a.attn_n_kv = std::stoi(need("--n-kv"));
    else if (k == "--tokenizer") a.tokenizer_type = need("--tokenizer");
    else if (k == "--bpe-vocab") a.bpe_vocab_path = need("--bpe-vocab");
    else if (k == "--bpe-merges") a.bpe_merges_path = need("--bpe-merges");
    else if (k == "--help" || k == "-h") {
      std::cout
          << "Usage:\n"
          << "  train_gpt --data <path> [--steps N] [--batch B] [--seq T] [--dmodel C] [--layers L] [--lr LR] [--seed S] [--save PREFIX] [--tokenizer byte|bpe] [--bpe-vocab <file>] [--bpe-merges <file>]\n"
          << "  train_gpt --load PREFIX [--steps N] [--data <path>] [--save PREFIX] [--save-opt 0|1]\n"
            << "  train_gpt [--data <path> --steps N ...] --prompt <text> [--gen N] [--temp X] [--topk K] [--print-next-top N] [--print-next-top-each-step 0|1] [--ascii-only 0|1] [--escape-bytes 0|1] [--load PREFIX] [--tokenizer byte|bpe] [--bpe-vocab <file>] [--bpe-merges <file>] [--kvcache]\n"
            << "  train_gpt --data <path> --steps 0 --sanity-next-from-data N [--sanity-ctx T] [--sanity-top K] [--load PREFIX]\n"
            << "  train_gpt --data <path> --steps 0 --sanity-offset OFF [--sanity-ctx T] [--sanity-top K] [--sanity-preview N] [--load PREFIX]\n\n"
          << "Notes:\n"
          << "- If --prompt is set, the program will (optionally) train first (if --steps > 0) and then generate text.\n"
          << "- Tokenization is pluggable: --tokenizer byte (default) or bpe. For bpe, --bpe-vocab and --bpe-merges are required.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + k);
    }
  }
  if (a.steps > 0 && a.data_path.empty() && a.token_data.empty()) {
    throw std::runtime_error("--data or --token-data is required when --steps > 0");
  }
  if (a.sanity_next_from_data > 0 && a.data_path.empty()) {
    throw std::runtime_error("--data is required when --sanity-next-from-data > 0");
  }
  if (a.sanity_offset >= 0 && a.data_path.empty()) {
    throw std::runtime_error("--data is required when --sanity-offset is set");
  }
  if (a.gen_tokens < 0) throw std::runtime_error("--gen must be >= 0");
  if (a.temperature <= 0.0f) throw std::runtime_error("--temp must be > 0");
  if (a.topk < 0) throw std::runtime_error("--topk must be >= 0");
  if (a.print_next_top < 0) throw std::runtime_error("--print-next-top must be >= 0");
  if (a.sanity_next_from_data < 0) throw std::runtime_error("--sanity-next-from-data must be >= 0");
  if (a.sanity_ctx < 0) throw std::runtime_error("--sanity-ctx must be >= 0");
  if (a.sanity_top < 0) throw std::runtime_error("--sanity-top must be >= 0");
  if (a.sanity_offset < -1) throw std::runtime_error("--sanity-offset must be >= 0 (or -1 to disable)");
  if (a.sanity_preview < 0) throw std::runtime_error("--sanity-preview must be >= 0");
  return a;
}

// Global stop flag for graceful shutdown via stdin
static std::atomic<bool> g_stop_requested{false};

static void stdin_reader_thread() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "EXIT") {
      g_stop_requested.store(true);
      break;
    }
  }
}

// Helper: write named weight tensor stats as JSON fragment
static void write_weight_stats_json(std::ostream& os, const model::TinyGPT& gpt) {
  auto pc = gpt.parameters_const();
  const int n_layers = gpt.cfg().n_layers;
  os << "\"w_stats\":{";
  // Build name list matching parameter order
  struct NamedIdx { std::string name; int idx; };
  std::vector<NamedIdx> names = {{"wte",0}, {"wpe",1}};
  for (int li = 0; li < n_layers; ++li) {
    const int b = 2 + li * 12;
    std::string p = "L"+std::to_string(li)+"_";
    names.push_back({p+"w_qkv", b+0});
    names.push_back({p+"b_qkv", b+1});
    names.push_back({p+"w_proj", b+2});
    names.push_back({p+"b_proj", b+3});
    names.push_back({p+"w_fc", b+4});
    names.push_back({p+"b_fc", b+5});
    names.push_back({p+"w_out", b+6});
    names.push_back({p+"b_out", b+7});
    names.push_back({p+"ln_attn_gamma", b+8});
    names.push_back({p+"ln_attn_beta", b+9});
    names.push_back({p+"ln_mlp_gamma", b+10});
    names.push_back({p+"ln_mlp_beta", b+11});
  }
  names.push_back({"w_lm", 2 + n_layers*12 + 0});
  names.push_back({"b_lm", 2 + n_layers*12 + 1});
  names.push_back({"ln_final_gamma", 2 + n_layers*12 + 2});
  names.push_back({"ln_final_beta", 2 + n_layers*12 + 3});

  bool first = true;
  for (auto& ni : names) {
    if (ni.idx >= static_cast<int>(pc.tensors.size())) continue;
    const float* d = pc.tensors[ni.idx]->data->data();
    const std::size_t N = pc.tensors[ni.idx]->numel();
    if (N == 0) continue;
    float mn = d[0], mx = d[0], sum = 0.0f, sum2 = 0.0f;
    for (std::size_t i = 0; i < N; ++i) {
      float v = d[i]; if (v<mn) mn=v; if (v>mx) mx=v; sum+=v; sum2+=v*v;
    }
    float mean = sum / static_cast<float>(N);
    float rms = std::sqrt(sum2 / static_cast<float>(N));
    if (!first) os << ",";
    first = false;
    os << "\"" << ni.name << "\":{" << "\"mean\":" << mean << ",\"rms\":" << rms << ",\"min\":" << mn << ",\"max\":" << mx << "}";
  }
  os << "}";
}


// Use pluggable tokenizer
static std::vector<std::int32_t> encode_prompt(const std::string& s, const Tokenizer& tokenizer) {
  std::vector<int> tokens = tokenizer.encode(s);
  // Convert to int32_t for model
  std::vector<std::int32_t> out(tokens.begin(), tokens.end());
  return out;
}

static std::string decode_one_token(int token_id, const Tokenizer& tokenizer) {
  return tokenizer.decode(std::vector<int>{token_id});
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

static std::string bytes_to_preview(const std::vector<std::uint8_t>& bytes,
                                    std::int64_t start,
                                    int ctx_len,
                                    int preview,
                                    bool escape_bytes) {
  if (ctx_len <= 0 || preview <= 0) return std::string();
  const std::int64_t avail = static_cast<std::int64_t>(ctx_len);
  const std::int64_t shown = std::min<std::int64_t>(avail, static_cast<std::int64_t>(preview));
  const std::int64_t from = start + (avail - shown);
  std::string out;
  out.reserve(static_cast<std::size_t>(shown));
  for (std::int64_t i = 0; i < shown; ++i) {
    out += token_to_display(static_cast<int>(bytes[static_cast<std::size_t>(from + i)]), escape_bytes);
  }
  if (shown < avail) return std::string("...") + out;
  return out;
}

static void print_tensor_stats(const char* label, const float* data, int N) {
  if (N <= 0) return;
  float mn = data[0], mx = data[0], sum = 0.0f, sum2 = 0.0f;
  for (int i = 0; i < N; ++i) {
    const float v = data[i];
    if (v < mn) mn = v; if (v > mx) mx = v;
    sum += v; sum2 += v * v;
  }
  const float mean = sum / static_cast<float>(N);
  const float rms = std::sqrt(sum2 / static_cast<float>(N));
  std::cout << "  " << label << " [" << N << "]  mean=" << mean << "  rms=" << rms << "  min=" << mn << "  max=" << mx << "\n";
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

static void sanity_check_next_at_offset(model::TinyGPT& gpt,
                                        const std::vector<std::uint8_t>& bytes,
                                        std::int64_t offset,
                                        int ctx_len,
                                        int topn,
                                        int preview,
                                        bool escape_bytes) {
  if (offset < 0) throw std::runtime_error("sanity_check_next_at_offset: offset must be >= 0");
  if (ctx_len <= 0) throw std::runtime_error("sanity_check_next_at_offset: ctx_len must be > 0");
  if (topn < 0) throw std::runtime_error("sanity_check_next_at_offset: topn must be >= 0");

  const std::int64_t needed = offset + static_cast<std::int64_t>(ctx_len) + 1;
  if (needed > static_cast<std::int64_t>(bytes.size())) {
    std::ostringstream oss;
    oss << "sanity_check_next_at_offset: need bytes[" << offset << ".." << (needed - 1)
        << "] but dataset has " << bytes.size() << " bytes";
    throw std::runtime_error(oss.str());
  }

  const int V = gpt.cfg().vocab_size;
  if (V != 256) throw std::runtime_error("sanity_check_next_at_offset: expected V=256 byte vocab");

  std::vector<std::int32_t> ctx;
  ctx.resize(static_cast<std::size_t>(ctx_len));
  for (int i = 0; i < ctx_len; ++i) {
    ctx[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(bytes[static_cast<std::size_t>(offset + i)]);
  }
  const int expected = static_cast<int>(bytes[static_cast<std::size_t>(offset + ctx_len)]);

  nn::GradMode no_grad(false);
  nn::Tensor logits = gpt.forward_logits(ctx, 1, ctx_len); // [1,ctx_len,V]
  const std::size_t base = static_cast<std::size_t>(ctx_len - 1) * static_cast<std::size_t>(V);
  const float* row = logits.data->data() + base;

  const int pred = argmax_index(row, V);

  std::cout << "\n[sanity offset] offset=" << offset << " ctx_len=" << ctx_len << " (predicts byte @ "
            << (offset + ctx_len) << ")\n";
  const std::string ctx_preview = bytes_to_preview(bytes, offset, ctx_len, preview, escape_bytes);
  if (!ctx_preview.empty()) {
    std::cout << "  context preview: \"" << ctx_preview << "\"\n";
  }
  std::cout << "  expected next: id=" << expected << " tok='" << token_to_display(expected, escape_bytes) << "'\n";
  std::cout << "  pred top1:    id=" << pred << " tok='" << token_to_display(pred, escape_bytes) << "'";
  std::cout << (pred == expected ? "  [OK]\n" : "  [MISMATCH]\n");

  if (topn > 0) {
    // Use temperature=1 and no ASCII filtering for a pure next-byte check.
    print_next_token_distribution(row, V, 1.0f, topn, false, escape_bytes);
  }
}

static void generate_kvcache(model::TinyGPT& gpt,
                              const std::string& prompt,
                              int gen_tokens,
                              float temperature,
                              int topk,
                              int print_next_top,
                              bool print_next_each_step,
                              bool ascii_only,
                              bool escape_bytes,
                              bool debug_layer,
                              util::Rng& rng,
                              const Tokenizer& tokenizer) {
  std::vector<std::int32_t> tokens = encode_prompt(prompt, tokenizer);
  if (tokens.empty()) {
    tokens.push_back(static_cast<std::int32_t>('\n'));
  }

  const bool is_byte_tokenizer = (dynamic_cast<const ByteTokenizer*>(&tokenizer) != nullptr);
  if (!is_byte_tokenizer && (ascii_only || escape_bytes)) {
    throw std::runtime_error("--ascii-only/--escape-bytes are only supported with the byte tokenizer (vocab=256)");
  }

  std::cout << prompt;
  std::cout.flush();

  nn::GradMode no_grad(false);
  const int V = gpt.cfg().vocab_size;
  const int C = gpt.cfg().d_model;
  const int n_layers = gpt.cfg().n_layers;
  const int maxT = gpt.cfg().seq_len;
  const int B = 1;

  if (static_cast<int>(tokens.size()) > maxT) {
    tokens.erase(tokens.begin(), tokens.end() - maxT);
  }

  std::vector<nn::variants::kvcache::KVCache> layer_caches;
  layer_caches.reserve(static_cast<std::size_t>(n_layers));
  for (int li = 0; li < n_layers; ++li) {
    layer_caches.emplace_back(B, maxT, C);
  }

  const int T_init = static_cast<int>(tokens.size());

  nn::Tensor logits = nn::variants::kvcache::model_prefill(gpt, tokens, B, T_init, layer_caches);

  const std::vector<int> allowed = ascii_only ? allowed_indices_ascii(V) : std::vector<int>();

  for (int step = 0; step < gen_tokens; ++step) {
    const std::size_t base = static_cast<std::size_t>(T_init + step - 1) * static_cast<std::size_t>(V);

    if (step == 0) {
      if (print_next_top > 0) {
        print_next_token_distribution(logits.data->data() + base, V, temperature, print_next_top, ascii_only, escape_bytes);
      }
      const int next = ascii_only
                           ? sample_from_logits_filtered(logits.data->data() + base, V, rng, temperature, topk, allowed)
                           : sample_from_logits(logits.data->data() + base, V, rng, temperature, topk);
      tokens.push_back(next);
      if (is_byte_tokenizer) {
        print_generated_byte(next, escape_bytes);
      } else {
        std::cout << decode_one_token(next, tokenizer);
      }
      std::cout.flush();
    } else {
      const int position = T_init + step - 1;
      logits = nn::variants::kvcache::model_step(gpt, tokens.back(), B, position, layer_caches);

      if (print_next_top > 0 && print_next_each_step) {
        std::cout << "\n[gen step " << step << "]";
        print_next_token_distribution(logits.data->data(), V, temperature, print_next_top, ascii_only, escape_bytes);
      }

      const int next = ascii_only
                           ? sample_from_logits_filtered(logits.data->data(), V, rng, temperature, topk, allowed)
                           : sample_from_logits(logits.data->data(), V, rng, temperature, topk);
      tokens.push_back(next);
      if (is_byte_tokenizer) {
        print_generated_byte(next, escape_bytes);
      } else {
        std::cout << decode_one_token(next, tokenizer);
      }
      std::cout.flush();
    }
  }
  std::cout << "\n";
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
                     bool debug_layer,
                     util::Rng& rng,
                     const Tokenizer& tokenizer) {
  std::vector<std::int32_t> tokens = encode_prompt(prompt, tokenizer);
  if (tokens.empty()) {
    // Start from a newline if no prompt is provided.
    tokens.push_back(static_cast<std::int32_t>('\n'));
  }

  const bool is_byte_tokenizer = (dynamic_cast<const ByteTokenizer*>(&tokenizer) != nullptr);
  if (!is_byte_tokenizer && (ascii_only || escape_bytes)) {
    throw std::runtime_error("--ascii-only/--escape-bytes are only supported with the byte tokenizer (vocab=256)");
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

    if (debug_layer && (print_next_each_step || step == 0)) {
      // Print per-layer weight and logit stats
      const model::Config& cfg = gpt.cfg();
      // Re-run forward with debug using a helper — for simplicity, print logits stats
      print_tensor_stats("logits[last]", logits.data->data() + base, V);
      // Print embedding + positional stats via the model's params
      auto pc = gpt.parameters_const();
      print_tensor_stats("wte", pc.tensors[0]->data->data(), static_cast<int>(pc.tensors[0]->numel()));
      print_tensor_stats("wpe", pc.tensors[1]->data->data(), static_cast<int>(pc.tensors[1]->numel()));
      for (int li = 0; li < cfg.n_layers; ++li) {
        const int bi = 2 + li * 12;
        std::string prefix = "L" + std::to_string(li) + " w_qkv";
        print_tensor_stats(prefix.c_str(), pc.tensors[bi]->data->data(), static_cast<int>(pc.tensors[bi]->numel()));
      }
    }

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

    if (is_byte_tokenizer) {
      print_generated_byte(next, escape_bytes);
    } else {
      std::cout << decode_one_token(next, tokenizer);
    }
    std::cout.flush();
  }
  std::cout << "\n";
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);

    std::uint64_t start_step = 0;


    // Instantiate tokenizer
    std::unique_ptr<Tokenizer> tokenizer;
    if (args.tokenizer_type == "byte") {
      tokenizer = std::make_unique<ByteTokenizer>();
    } else if (args.tokenizer_type == "bpe") {
      if (args.bpe_vocab_path.empty() || args.bpe_merges_path.empty()) {
        throw std::runtime_error("BPE tokenizer requires --bpe-vocab and --bpe-merges");
      }
      tokenizer = std::make_unique<BpeTokenizer>(args.bpe_vocab_path, args.bpe_merges_path);
    } else if (args.tokenizer_type == "sp") {
      if (args.bpe_vocab_path.empty()) throw std::runtime_error("SPM tokenizer requires --bpe-vocab (JSON vocab)");
      tokenizer = std::make_unique<SpmTokenizer>(args.bpe_vocab_path);
    } else {
      throw std::runtime_error("Unknown --tokenizer type: " + args.tokenizer_type);
    }

    // Tokenize mode: read data file, tokenize, write token IDs
    if (!args.tokenize_output.empty()) {
      if (args.tokenizer_type != "bpe") throw std::runtime_error("--tokenize-output requires --tokenizer bpe");
      if (args.data_path.empty()) throw std::runtime_error("--tokenize-output requires --data");
      auto bytes = util::read_file_bytes(args.data_path);
      std::string text(bytes.begin(), bytes.end());
      std::vector<int> ids = tokenizer->encode(text);
      std::cout << "Tokenized " << text.size() << " chars → " << ids.size() << " tokens (ratio " << static_cast<float>(ids.size())/static_cast<float>(text.size()) << ")\n";
      // Write binary int32
      std::ofstream out(args.tokenize_output, std::ios::binary);
      if (!out) throw std::runtime_error("Failed to open tokenize output: " + args.tokenize_output);
      for (int id : ids) {
        std::int32_t v = static_cast<std::int32_t>(id);
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
      }
      std::cout << "Saved " << ids.size() << " token IDs to " << args.tokenize_output << "\n";
      return 0;
    }

    // Data pipeline: byte (for --tokenizer byte) or token-ID (for --tokenizer bpe)
    std::vector<std::uint8_t> data_bytes;
    std::vector<std::int32_t> token_ids;
    std::unique_ptr<data::ByteDataset> ds;
    std::unique_ptr<data::TokenDataset> tds;

    if (args.tokenizer_type == "bpe" && (args.steps > 0)) {
      if (args.token_data.empty()) throw std::runtime_error("BPE training requires --token-data <file> (use --tokenize-output first)");
      std::ifstream tfin(args.token_data, std::ios::binary);
      if (!tfin) throw std::runtime_error("Failed to open token data: " + args.token_data);
      tfin.seekg(0, std::ios::end);
      std::size_t fsize = static_cast<std::size_t>(tfin.tellg());
      tfin.seekg(0);
      std::size_t count = fsize / sizeof(std::int32_t);
      token_ids.resize(count);
      tfin.read(reinterpret_cast<char*>(token_ids.data()), static_cast<std::streamsize>(fsize));
      std::cout << "Loaded " << count << " BPE token IDs from " << args.token_data << "\n";
      tds = std::make_unique<data::TokenDataset>(token_ids);
    } else if (!args.data_path.empty() && args.steps > 0) {
      data_bytes = util::read_file_bytes(args.data_path);
      ds = std::make_unique<data::ByteDataset>(data_bytes);
    }

    // Sanity checks: always byte-based (reads from text file)
    if (!args.data_path.empty() && (args.sanity_next_from_data > 0 || args.sanity_offset >= 0)) {
      data_bytes = util::read_file_bytes(args.data_path);
    }

    model::Config cfg;
    cfg.vocab_size = tokenizer->vocab_size();
    cfg.seq_len = args.seq;
    cfg.d_model = args.dmodel;
    cfg.n_layers = args.layers;
    cfg.norm_type = (args.norm_type == "rmsnorm") ? 1 : 0;
    cfg.mlp_type  = (args.mlp_type == "swiglu") ? 1 : ((args.mlp_type == "moe") ? 2 : 0);
    cfg.attn_type = (args.attn_type == "mha") ? 1 : ((args.attn_type == "gqa") ? 2 : ((args.attn_type == "mla") ? 3 : 0));
    cfg.pos_type  = (args.pos_type == "rope") ? 1 : 0;
    cfg.attn_n_heads = args.attn_n_heads > 0 ? args.attn_n_heads : 1;
    cfg.attn_n_kv = args.attn_n_kv > 0 ? args.attn_n_kv : 1;

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

    model::TinyGPT gpt(cfg, args.seed);
    optim::AdamW opt(ocfg);

    if (!args.load_prefix.empty()) {
      std::uint64_t loaded_step = 0;
      ckpt::load(args.load_prefix, gpt, opt, loaded_step);
      start_step = loaded_step;
      std::cout << "loaded checkpoint '" << args.load_prefix << "' at step " << start_step << "\n";
    }

    util::Rng rng(args.seed ^ 0xDEADBEEF);

    // Start stdin reader for graceful shutdown
    std::thread stdin_thread;
    if (args.pipe_stdin) {
      stdin_thread = std::thread(stdin_reader_thread);
    }

    if (args.steps > 0) {
      if (!ds && !tds) throw std::runtime_error("internal: dataset not initialized");
      const int train_seq = std::min(args.seq, gpt.cfg().seq_len);
      const auto t0 = std::chrono::high_resolution_clock::now();
      for (int local = 1; local <= args.steps; ++local) {
        const std::uint64_t step = start_step + static_cast<std::uint64_t>(local);
        data::Batch batch = ds ? ds->sample_batch(args.batch, train_seq, rng)
                               : tds->sample_batch(args.batch, train_seq, rng);

        gpt.zero_grad();
        nn::Tensor loss = gpt.loss(batch.x, batch.y, batch.B, batch.T);
        loss.backward();

        opt.step(gpt.parameters().tensors);
        const auto tn = std::chrono::high_resolution_clock::now();
        const double sec = std::chrono::duration<double>(tn - t0).count();

        if (args.progress_json) {
          std::cout << "{\"type\":\"step\"" << ",\"step\":" << step << ",\"loss\":" << (*loss.data)[0] << ",\"time\":" << sec;
          if (args.progress_grads > 0 && local % args.progress_grads == 0) {
            std::cout << ",";
            write_weight_stats_json(std::cout, gpt);
          }
          std::cout << "}\n" << std::flush;
        } else if (local == 1 || local % 10 == 0) {
          std::cout << "step " << step << "  (" << local << "/" << args.steps << ")  loss=" << (*loss.data)[0] << "  time=" << sec << "s\n";
        }

        if (!args.log_loss_path.empty() && (local == 1 || local % 10 == 0)) {
          std::ofstream lf(args.log_loss_path, std::ios::app);
          if (lf) {
            if (local == 1) lf << "step,loss,time\n";
            lf << step << "," << (*loss.data)[0] << "," << sec << "\n";
          }
        }

        // Periodic checkpoint save
        if (args.save_interval > 0 && !args.save_prefix.empty() && local % args.save_interval == 0) {
          ckpt::save(args.save_prefix, gpt, opt, step, args.save_optim);
        }

        // Graceful shutdown via stdin
        if (args.pipe_stdin && g_stop_requested.load()) {
          if (!args.save_prefix.empty()) {
            ckpt::save(args.save_prefix, gpt, opt, step, args.save_optim);
            if (args.progress_json) std::cout << "{\"type\":\"saved\",\"step\":" << step << ",\"prefix\":\"" << args.save_prefix << "\"}\n" << std::flush;
          }
          if (args.progress_json) std::cout << "{\"type\":\"stopped\",\"step\":" << step << "}\n" << std::flush;
          start_step = step;
          goto training_done;
        }
      }

      start_step += static_cast<std::uint64_t>(args.steps);
    }
    training_done:

    // === Serve mode: interactive generation via stdin/stdout JSON ===
    if (args.serve_mode) {
      nn::GradMode no_grad(false);
      std::string line;
      while (std::getline(std::cin, line)) {
        if (line.empty() || line == "EXIT") break;
        // Parse JSON: {"prompt":"...","temp":0.8,"topk":40,"gen":50}
        std::string prompt;
        float temp = 0.8f; int topk = 40, gen = 50;
        // Simple key-value parse (avoids JSON dependency)
        auto extract = [&](const std::string& key) -> std::string {
          std::size_t p = line.find("\"" + key + "\"");
          if (p == std::string::npos) return "";
          p = line.find(":", p + key.size() + 2);
          if (p == std::string::npos) return "";
          p++;
          while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) p++;
          if (p >= line.size()) return "";
          if (line[p] == '"') {
            p++; std::size_t e = line.find("\"", p);
            if (e != std::string::npos) return line.substr(p, e - p);
          } else {
            std::size_t e = p;
            while (e < line.size() && (std::isdigit(line[e]) || line[e] == '.' || line[e] == '-')) e++;
            return line.substr(p, e - p);
          }
          return "";
        };
        prompt = extract("prompt");
        std::string ts = extract("temp"); if (!ts.empty()) temp = std::stof(ts);
        std::string tks = extract("topk"); if (!tks.empty()) topk = std::stoi(tks);
        std::string gs = extract("gen"); if (!gs.empty()) gen = std::stoi(gs);

        if (prompt.empty()) {
          std::cout << "{\"error\":\"missing prompt\"}\n" << std::flush;
          continue;
        }

        std::vector<std::int32_t> tokens = encode_prompt(prompt, *tokenizer);
        if (tokens.empty()) tokens.push_back(static_cast<std::int32_t>('\n'));

        const int V = gpt.cfg().vocab_size;
        util::Rng srng(args.seed ^ 0xABCDEF123456ULL);

        for (int step = 0; step < gen; ++step) {
          const int maxT = gpt.cfg().seq_len;
          const int T = static_cast<int>(std::min<std::size_t>(tokens.size(), static_cast<std::size_t>(maxT)));
          std::vector<std::int32_t> ctx(tokens.end() - T, tokens.end());
          nn::Tensor logits = gpt.forward_logits(ctx, 1, T);
          const std::size_t base = static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(V);
          int next = sample_from_logits(logits.data->data() + base, V, srng, temp, topk);
          tokens.push_back(next);
          std::string text = decode_one_token(next, *tokenizer);
          // Output JSON per token
          std::cout << "{\"token\":" << next << ",\"text\":\"" << text << "\"}\n" << std::flush;
        }
        std::cout << "{\"done\":true}\n" << std::flush;
      }
      return 0;
    }

    // Join stdin thread
    if (args.pipe_stdin && stdin_thread.joinable()) {
      stdin_thread.detach(); // detach since we're done
    }

    // Dump weights on exit if requested
    if (!args.dump_weights_path.empty()) {
      ckpt::export_weights(args.dump_weights_path, args.dump_weights_format, gpt);
    }

    if (args.sanity_next_from_data > 0) {
      const int ctx_len = (args.sanity_ctx > 0) ? args.sanity_ctx : gpt.cfg().seq_len;
      if (ctx_len > gpt.cfg().seq_len) throw std::runtime_error("--sanity-ctx exceeds model seq_len");
      util::Rng srng(args.seed ^ 0x51514E455854ULL);
      sanity_check_next_from_data(gpt, data_bytes, args.sanity_next_from_data, ctx_len, args.sanity_top, srng, args.escape_bytes);
    }

    if (args.sanity_offset >= 0) {
      const int ctx_len = (args.sanity_ctx > 0) ? args.sanity_ctx : gpt.cfg().seq_len;
      if (ctx_len > gpt.cfg().seq_len) throw std::runtime_error("--sanity-ctx exceeds model seq_len");
      sanity_check_next_at_offset(gpt, data_bytes, args.sanity_offset, ctx_len, args.sanity_top, args.sanity_preview, args.escape_bytes);
    }

    if (!args.save_prefix.empty()) {
      const bool save_opt = args.save_optim && (start_step > 0);
      ckpt::save(args.save_prefix, gpt, opt, start_step, save_opt);
      std::cout << "saved checkpoint '" << args.save_prefix << "' at step " << start_step
                << (save_opt ? " (with optimizer state)" : "") << "\n";
    }

    if (!args.prompt.empty()) {
      util::Rng grng(args.seed ^ 0xABCDEF123456ULL);
      if (args.use_kvcache) {
        generate_kvcache(gpt,
             args.prompt,
             args.gen_tokens,
             args.temperature,
             args.topk,
             args.print_next_top,
             args.print_next_each_step,
             args.ascii_only,
             args.escape_bytes,
             args.debug_layer,
             grng,
             *tokenizer);
      } else {
        generate(gpt,
             args.prompt,
             args.gen_tokens,
             args.temperature,
             args.topk,
             args.print_next_top,
             args.print_next_each_step,
             args.ascii_only,
             args.escape_bytes,
             args.debug_layer,
             grng,
              *tokenizer);
      }
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
