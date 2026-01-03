#include "util.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace util {

std::vector<std::uint8_t> read_file_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open file: " + path);
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) {
    throw std::runtime_error("failed to stat file: " + path);
  }
  in.seekg(0, std::ios::beg);

  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (!bytes.empty()) {
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) {
      throw std::runtime_error("failed to read file: " + path);
    }
  }
  return bytes;
}

std::uint64_t Rng::next_u64() {
  // xorshift64*
  std::uint64_t x = state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  state = x;
  return x * 2685821657736338717ULL;
}

std::uint32_t Rng::next_u32() {
  return static_cast<std::uint32_t>(next_u64() >> 32);
}

float Rng::next_f01() {
  // 24-bit mantissa -> float in [0,1)
  const std::uint32_t u = next_u32() >> 8;
  return static_cast<float>(u) * (1.0f / 16777216.0f);
}

std::int32_t Rng::uniform_int(std::int32_t lo_inclusive, std::int32_t hi_inclusive) {
  if (hi_inclusive < lo_inclusive) {
    throw std::runtime_error("uniform_int: invalid range");
  }
  const std::uint32_t span = static_cast<std::uint32_t>(hi_inclusive - lo_inclusive + 1);
  const std::uint32_t r = next_u32();
  return lo_inclusive + static_cast<std::int32_t>(r % span);
}

float fast_inv_sqrt(float x) {
  return 1.0f / std::sqrt(x);
}

} // namespace util
