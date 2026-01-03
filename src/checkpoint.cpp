#include "checkpoint.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "tensor.h"
#include "util.h"

namespace ckpt {

static std::string read_text_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open file: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static void write_text_file(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("failed to write file: " + path);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!out) throw std::runtime_error("failed to write file: " + path);
}

static std::string json_key(const char* k) {
  return std::string("\"") + k + "\"";
}

static std::string fmt_f64(double v) {
  std::ostringstream ss;
  ss.setf(std::ios::scientific);
  ss << std::setprecision(12) << v;
  return ss.str();
}

static std::int64_t extract_i64(const std::string& json, const char* key) {
  const std::string k = json_key(key);
  const std::size_t pos = json.find(k);
  if (pos == std::string::npos) throw std::runtime_error(std::string("missing key in json: ") + key);
  std::size_t p = json.find(':', pos + k.size());
  if (p == std::string::npos) throw std::runtime_error("invalid json (missing ':')");
  ++p;
  while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;

  bool neg = false;
  if (p < json.size() && json[p] == '-') {
    neg = true;
    ++p;
  }

  std::int64_t v = 0;
  bool any = false;
  while (p < json.size() && std::isdigit(static_cast<unsigned char>(json[p]))) {
    any = true;
    v = v * 10 + (json[p] - '0');
    ++p;
  }
  if (!any) throw std::runtime_error(std::string("invalid number for key: ") + key);
  return neg ? -v : v;
}

static double extract_f64(const std::string& json, const char* key) {
  const std::string k = json_key(key);
  const std::size_t pos = json.find(k);
  if (pos == std::string::npos) throw std::runtime_error(std::string("missing key in json: ") + key);
  std::size_t p = json.find(':', pos + k.size());
  if (p == std::string::npos) throw std::runtime_error("invalid json (missing ':')");
  ++p;
  while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;

  // scan a floating literal, very forgiving
  const std::size_t start = p;
  while (p < json.size()) {
    const char c = json[p];
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
      ++p;
      continue;
    }
    break;
  }
  if (p == start) throw std::runtime_error(std::string("invalid float for key: ") + key);
  return std::stod(json.substr(start, p - start));
}

static bool extract_bool(const std::string& json, const char* key) {
  const std::string k = json_key(key);
  const std::size_t pos = json.find(k);
  if (pos == std::string::npos) throw std::runtime_error(std::string("missing key in json: ") + key);
  std::size_t p = json.find(':', pos + k.size());
  if (p == std::string::npos) throw std::runtime_error("invalid json (missing ':')");
  ++p;
  while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
  if (json.compare(p, 4, "true") == 0) return true;
  if (json.compare(p, 5, "false") == 0) return false;
  throw std::runtime_error(std::string("invalid bool for key: ") + key);
}

LoadedConfig read_config(const std::string& prefix) {
  const std::string path = prefix + ".json";
  const std::string j = read_text_file(path);

  LoadedConfig cfg;
  cfg.model.vocab_size = static_cast<int>(extract_i64(j, "vocab_size"));
  cfg.model.seq_len = static_cast<int>(extract_i64(j, "seq_len"));
  cfg.model.d_model = static_cast<int>(extract_i64(j, "d_model"));
  cfg.model.n_layers = static_cast<int>(extract_i64(j, "n_layers"));

  cfg.optim.lr = static_cast<float>(extract_f64(j, "lr"));
  cfg.optim.beta1 = static_cast<float>(extract_f64(j, "beta1"));
  cfg.optim.beta2 = static_cast<float>(extract_f64(j, "beta2"));
  cfg.optim.eps = static_cast<float>(extract_f64(j, "eps"));
  cfg.optim.weight_decay = static_cast<float>(extract_f64(j, "weight_decay"));

  cfg.step = static_cast<std::uint64_t>(extract_i64(j, "step"));
  cfg.has_optim_state = extract_bool(j, "has_optim_state");

  return cfg;
}

static void write_u32(std::ofstream& out, std::uint32_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
static void write_u64(std::ofstream& out, std::uint64_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
static std::uint32_t read_u32(std::ifstream& in) {
  std::uint32_t v;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}
static std::uint64_t read_u64(std::ifstream& in) {
  std::uint64_t v;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

void save(const std::string& prefix, const model::TinyGPT& gpt, const optim::AdamW& opt, std::uint64_t step, bool save_optim_state) {
  // JSON config
  const model::Config& mc = gpt.cfg();

  const optim::AdamWConfig& oc = opt.cfg();

  const std::string json =
      std::string("{\n") +
      "  \"format\": \"build-llm-using-cpp-checkpoint\",\n" +
      "  \"version\": 1,\n" +
      "  \"vocab_size\": " + std::to_string(mc.vocab_size) + ",\n" +
      "  \"seq_len\": " + std::to_string(mc.seq_len) + ",\n" +
      "  \"d_model\": " + std::to_string(mc.d_model) + ",\n" +
      "  \"n_layers\": " + std::to_string(mc.n_layers) + ",\n" +
      "  \"lr\": " + fmt_f64(oc.lr) + ",\n" +
      "  \"beta1\": " + fmt_f64(oc.beta1) + ",\n" +
      "  \"beta2\": " + fmt_f64(oc.beta2) + ",\n" +
      "  \"eps\": " + fmt_f64(oc.eps) + ",\n" +
      "  \"weight_decay\": " + fmt_f64(oc.weight_decay) + ",\n" +
      "  \"step\": " + std::to_string(step) + ",\n" +
      "  \"has_optim_state\": " + std::string(save_optim_state ? "true" : "false") + "\n" +
      "}\n";

  write_text_file(prefix + ".json", json);

  // Binary weights (+ optional optim)
  const auto params_const = gpt.parameters_const().tensors;
  std::ofstream out(prefix + ".bin", std::ios::binary);
  if (!out) throw std::runtime_error("failed to write: " + prefix + ".bin");

  const char magic[8] = {'B', 'G', 'P', 'T', 'C', 'K', 'P', 'T'};
  out.write(magic, sizeof(magic));
  write_u32(out, 1); // version
  write_u32(out, save_optim_state ? 1u : 0u);
  write_u64(out, step);
  write_u32(out, static_cast<std::uint32_t>(params_const.size()));

  for (const nn::Tensor* p : params_const) {
    const std::uint64_t n = static_cast<std::uint64_t>(p->numel());
    write_u64(out, n);
    out.write(reinterpret_cast<const char*>(p->data->data()), static_cast<std::streamsize>(n * sizeof(float)));
  }

  if (save_optim_state) {
    const auto params_mut = const_cast<model::TinyGPT&>(gpt).parameters().tensors;
    const auto st = opt.export_state(params_mut);
    write_u64(out, st.t);
    for (std::size_t i = 0; i < params_mut.size(); ++i) {
      const std::uint64_t n = static_cast<std::uint64_t>(params_mut[i]->numel());
      out.write(reinterpret_cast<const char*>(st.m[i].data()), static_cast<std::streamsize>(n * sizeof(float)));
      out.write(reinterpret_cast<const char*>(st.v[i].data()), static_cast<std::streamsize>(n * sizeof(float)));
    }
  }

  if (!out) throw std::runtime_error("failed while writing: " + prefix + ".bin");
}

void load(const std::string& prefix, model::TinyGPT& gpt, optim::AdamW& opt, std::uint64_t& step_out) {
  std::ifstream in(prefix + ".bin", std::ios::binary);
  if (!in) throw std::runtime_error("failed to open: " + prefix + ".bin");

  char magic[8];
  in.read(magic, sizeof(magic));
  const char expected[8] = {'B', 'G', 'P', 'T', 'C', 'K', 'P', 'T'};
  if (std::memcmp(magic, expected, sizeof(expected)) != 0) {
    throw std::runtime_error("invalid checkpoint magic");
  }

  const std::uint32_t version = read_u32(in);
  if (version != 1) throw std::runtime_error("unsupported checkpoint version");

  const std::uint32_t has_opt = read_u32(in);
  step_out = read_u64(in);
  const std::uint32_t nparams = read_u32(in);

  auto params = gpt.parameters().tensors;
  if (nparams != static_cast<std::uint32_t>(params.size())) {
    throw std::runtime_error("checkpoint param-count mismatch (different model config?)");
  }

  for (std::size_t i = 0; i < params.size(); ++i) {
    nn::Tensor* p = params[i];
    const std::uint64_t n = read_u64(in);
    if (n != static_cast<std::uint64_t>(p->numel())) {
      throw std::runtime_error("checkpoint tensor size mismatch (different model config?)");
    }
    in.read(reinterpret_cast<char*>(p->data->data()), static_cast<std::streamsize>(n * sizeof(float)));
    if (!in) throw std::runtime_error("failed reading tensor data");
  }

  if (has_opt != 0) {
    optim::AdamW::ExportedState st;
    st.t = read_u64(in);
    st.m.resize(params.size());
    st.v.resize(params.size());
    for (std::size_t i = 0; i < params.size(); ++i) {
      const std::size_t n = params[i]->numel();
      st.m[i].resize(n);
      st.v[i].resize(n);
      in.read(reinterpret_cast<char*>(st.m[i].data()), static_cast<std::streamsize>(n * sizeof(float)));
      in.read(reinterpret_cast<char*>(st.v[i].data()), static_cast<std::streamsize>(n * sizeof(float)));
      if (!in) throw std::runtime_error("failed reading optimizer state");
    }
    opt.import_state(params, st);
  }
}

} // namespace ckpt
