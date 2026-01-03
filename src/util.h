#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace util {

std::vector<std::uint8_t> read_file_bytes(const std::string& path);

// Tiny, deterministic RNG (xorshift64*) for reproducible experiments.
struct Rng {
  std::uint64_t state;
  explicit Rng(std::uint64_t seed = 0x12345678ULL) : state(seed ? seed : 0x12345678ULL) {}

  std::uint64_t next_u64();
  std::uint32_t next_u32();
  float next_f01(); // [0,1)
  std::int32_t uniform_int(std::int32_t lo_inclusive, std::int32_t hi_inclusive);
};

float fast_inv_sqrt(float x);

} // namespace util
