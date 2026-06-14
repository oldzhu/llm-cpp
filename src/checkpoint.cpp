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
      "  \"version\": 5,\n" +
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
  write_u32(out, 5); // version (pipeline composition)
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
  if (version != 1 && version != 2 && version != 3 && version != 4 && version != 5) throw std::runtime_error("unsupported checkpoint version");

  const std::uint32_t has_opt = read_u32(in);
  step_out = read_u64(in);
  const std::uint32_t nparams = read_u32(in);

  auto params = gpt.parameters().tensors;
  std::vector<std::size_t> v1_to_new; // only populated for version == 1
  std::uint32_t v1_nparams = 0;

  if (version == 1) {
    v1_nparams = nparams;
    const int n_layers = gpt.cfg().n_layers;

    v1_to_new.reserve(static_cast<std::size_t>(v1_nparams));
    v1_to_new.push_back(0);
    v1_to_new.push_back(1);
    for (int li = 0; li < n_layers; ++li) {
      const std::size_t base = 2 + static_cast<std::size_t>(li) * 12;
      v1_to_new.push_back(base + 0); v1_to_new.push_back(base + 1);
      v1_to_new.push_back(base + 2); v1_to_new.push_back(base + 3);
      v1_to_new.push_back(base + 4); v1_to_new.push_back(base + 5);
      v1_to_new.push_back(base + 6); v1_to_new.push_back(base + 7);
    }
    v1_to_new.push_back(2 + static_cast<std::size_t>(n_layers) * 12 + 0);
    v1_to_new.push_back(2 + static_cast<std::size_t>(n_layers) * 12 + 1);

    if (v1_to_new.size() != static_cast<std::size_t>(v1_nparams)) {
      throw std::runtime_error("v1 checkpoint: internal index mapping mismatch");
    }

    for (std::size_t iold = 0; iold < static_cast<std::size_t>(v1_nparams); ++iold) {
      nn::Tensor* p = params[v1_to_new[iold]];
      const std::uint64_t n = read_u64(in);
      if (n != static_cast<std::uint64_t>(p->numel())) {
        throw std::runtime_error("v1 checkpoint tensor size mismatch (different model config?)");
      }
      in.read(reinterpret_cast<char*>(p->data->data()), static_cast<std::streamsize>(n * sizeof(float)));
      if (!in) throw std::runtime_error("failed reading v1 tensor data");
    }

    for (int li = 0; li < n_layers; ++li) {
      const std::size_t base = 2 + static_cast<std::size_t>(li) * 12;
      nn::Tensor* ag = params[base + 8];
      nn::Tensor* ab = params[base + 9];
      nn::Tensor* mg = params[base + 10];
      nn::Tensor* mb = params[base + 11];
      for (std::size_t j = 0; j < ag->numel(); ++j) (*ag->data)[j] = 1.0f;
      for (std::size_t j = 0; j < ab->numel(); ++j) (*ab->data)[j] = 0.0f;
      for (std::size_t j = 0; j < mg->numel(); ++j) (*mg->data)[j] = 1.0f;
      for (std::size_t j = 0; j < mb->numel(); ++j) (*mb->data)[j] = 0.0f;
    }
    nn::Tensor* fg = params[2 + static_cast<std::size_t>(n_layers) * 12 + 2];
    nn::Tensor* fb = params[2 + static_cast<std::size_t>(n_layers) * 12 + 3];
    for (std::size_t j = 0; j < fg->numel(); ++j) (*fg->data)[j] = 1.0f;
    for (std::size_t j = 0; j < fb->numel(); ++j) (*fb->data)[j] = 0.0f;
  } else {
    // v2 or v3 — load all params (v3 adds SwiGLU params)
    if (nparams > static_cast<std::uint32_t>(params.size())) {
      throw std::runtime_error("checkpoint param-count mismatch (different model config?)");
    }
    for (std::size_t i = 0; i < params.size(); ++i) {
      nn::Tensor* p = params[i];
      if (i >= static_cast<std::size_t>(nparams)) {
        // SwiGLU params not present in v2 checkpoint — init from existing or zeros
        for (std::size_t j = 0; j < p->numel(); ++j) (*p->data)[j] = 0.0f;
        continue;
      }
      const std::uint64_t n = read_u64(in);
      if (n != static_cast<std::uint64_t>(p->numel())) {
        throw std::runtime_error("checkpoint tensor size mismatch (different model config?)");
      }
      in.read(reinterpret_cast<char*>(p->data->data()), static_cast<std::streamsize>(n * sizeof(float)));
      if (!in) throw std::runtime_error("failed reading tensor data");
    }
    // For v2: if we loaded fewer params than the model has, remaining are SwiGLU — init zeros
    for (std::size_t i = nparams; i < params.size(); ++i) {
      nn::Tensor* p = params[i];
      for (std::size_t j = 0; j < p->numel(); ++j) (*p->data)[j] = 0.0f;
    }
  }

  if (has_opt != 0) {
    optim::AdamW::ExportedState st;
    st.t = read_u64(in);
    st.m.resize(params.size());
    st.v.resize(params.size());
    if (version == 1) {
      // v1 optimizer state only covers the old params.
      // Load state into mapped positions, zero-init new LN param state.
      for (std::size_t i = 0; i < params.size(); ++i) {
        const std::size_t n = params[i]->numel();
        st.m[i].resize(n, 0.0f);
        st.v[i].resize(n, 0.0f);
      }
      for (std::size_t iold = 0; iold < static_cast<std::size_t>(v1_nparams); ++iold) {
        const std::size_t inew = v1_to_new[iold];
        const std::size_t n = params[inew]->numel();
        in.read(reinterpret_cast<char*>(st.m[inew].data()), static_cast<std::streamsize>(n * sizeof(float)));
        in.read(reinterpret_cast<char*>(st.v[inew].data()), static_cast<std::streamsize>(n * sizeof(float)));
        if (!in) throw std::runtime_error("failed reading v1 optimizer state");
      }
    } else {
      for (std::size_t i = 0; i < params.size(); ++i) {
        const std::size_t n = params[i]->numel();
        st.m[i].resize(n);
        st.v[i].resize(n);
        in.read(reinterpret_cast<char*>(st.m[i].data()), static_cast<std::streamsize>(n * sizeof(float)));
        in.read(reinterpret_cast<char*>(st.v[i].data()), static_cast<std::streamsize>(n * sizeof(float)));
        if (!in) throw std::runtime_error("failed reading optimizer state");
      }
    }
    opt.import_state(params, st);
  }
}

static std::vector<std::string> param_names(const model::TinyGPT& gpt) {
  const int n = gpt.cfg().n_layers;
  std::vector<std::string> names;
  names = {"wte","wpe"};
  for (int li = 0; li < n; ++li) {
    std::string p = "L" + std::to_string(li) + "_";
    names.push_back(p+"w_qkv");
    names.push_back(p+"b_qkv");
    names.push_back(p+"w_proj");
    names.push_back(p+"b_proj");
    names.push_back(p+"w_fc");
    names.push_back(p+"b_fc");
    names.push_back(p+"w_out");
    names.push_back(p+"b_out");
    names.push_back(p+"ln_attn_gamma");
    names.push_back(p+"ln_attn_beta");
    names.push_back(p+"ln_mlp_gamma");
    names.push_back(p+"ln_mlp_beta");
  }
  names.push_back("w_lm");
  names.push_back("b_lm");
  names.push_back("ln_final_gamma");
  names.push_back("ln_final_beta");
  for (int li = 0; li < n; ++li) {
    std::string p = "L" + std::to_string(li) + "_";
    names.push_back(p+"swiglu_gate"); names.push_back(p+"swiglu_gate_b");
    names.push_back(p+"swiglu_up");   names.push_back(p+"swiglu_up_b");
    names.push_back(p+"swiglu_down"); names.push_back(p+"swiglu_down_b");
  }
  for (int li = 0; li < n; ++li) {
    std::string p = "L" + std::to_string(li) + "_";
    names.push_back(p+"moe_router_w"); names.push_back(p+"moe_router_b");
    for (int e = 0; e < gpt.cfg().n_experts; ++e) {
      std::string ep = p + "e" + std::to_string(e) + "_";
      names.push_back(ep+"wfc"); names.push_back(ep+"bfc");
      names.push_back(ep+"wout"); names.push_back(ep+"bout");
    }
  }
  return names;
}

void export_weights(const std::string& path, const std::string& format, const model::TinyGPT& gpt) {
  auto pc = gpt.parameters_const().tensors;
  auto names = param_names(gpt);

  if (format == "json") {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("failed to write: " + path);
    out << "{";
    for (std::size_t i = 0; i < pc.size() && i < names.size(); ++i) {
      if (i > 0) out << ",";
      out << "\"" << names[i] << "\":{\"shape\":[";
      const auto& s = pc[i]->shape;
      for (std::size_t j = 0; j < s.size(); ++j) { if(j>0) out<<","; out<<s[j]; }
      out << "],\"data\":[";
      const std::size_t N = pc[i]->numel();
      const float* d = pc[i]->data->data();
      for (std::size_t j = 0; j < N; ++j) { if(j>0) out<<","; out << d[j]; }
      out << "]}";
    }
    out << "}\n";
  } else if (format == "safetensors") {
    // Safetensors: 8-byte header (u64 LE) + JSON metadata + concatenated float data
    std::ostringstream meta;
    meta << "{";
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < pc.size() && i < names.size(); ++i) {
      if (i > 0) meta << ",";
      const std::uint64_t nbytes = pc[i]->numel() * sizeof(float);
      meta << "\"" << names[i] << "\":{\"dtype\":\"F32\",\"shape\":[";
      const auto& s = pc[i]->shape;
      for (std::size_t j = 0; j < s.size(); ++j) { if(j>0) meta<<","; meta<<s[j]; }
      meta << "],\"data_offsets\":[" << offset << "," << (offset+nbytes) << "]}";
      offset += nbytes;
    }
    meta << "}";
    std::string meta_str = meta.str();
    std::uint64_t header_size = static_cast<std::uint64_t>(meta_str.size());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to write: " + path);
    out.write(reinterpret_cast<const char*>(&header_size), 8);
    out.write(meta_str.data(), static_cast<std::streamsize>(meta_str.size()));
    for (std::size_t i = 0; i < pc.size(); ++i) {
      const std::size_t nbytes = pc[i]->numel() * sizeof(float);
      out.write(reinterpret_cast<const char*>(pc[i]->data->data()), static_cast<std::streamsize>(nbytes));
    }
  } else {
    // "binary" (default): reuse checkpoint binary format without optimizer state
    optim::AdamWConfig oc; oc.lr = 0.0f;
    optim::AdamW opt(oc);
    save(path, gpt, opt, 0, false);
  }
}

} // namespace ckpt
