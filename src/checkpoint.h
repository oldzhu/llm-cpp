#pragma once

#include <cstdint>
#include <string>

#include "model.h"
#include "optim.h"

namespace ckpt {

struct LoadedConfig {
  model::Config model;
  optim::AdamWConfig optim;
  std::uint64_t step = 0;
  bool has_optim_state = false;
};

// Reads <prefix>.json and returns config. Does not touch model weights.
LoadedConfig read_config(const std::string& prefix);

// Loads weights (+ optional AdamW state) from <prefix>.bin into an already-constructed model.
// If has optimizer state in the file, it will be loaded into `opt`.
void load(const std::string& prefix, model::TinyGPT& gpt, optim::AdamW& opt, std::uint64_t& step_out);

// Saves config to <prefix>.json and weights (+ optional AdamW state) to <prefix>.bin.
void save(const std::string& prefix, const model::TinyGPT& gpt, const optim::AdamW& opt, std::uint64_t step, bool save_optim_state);

} // namespace ckpt
