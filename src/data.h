#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "util.h"

namespace data {

struct Batch {
  int B = 0;
  int T = 0;
  std::vector<std::int32_t> x; // [B*T]
  std::vector<std::int32_t> y; // [B*T]
};

class ByteDataset {
 public:
  explicit ByteDataset(std::vector<std::uint8_t> bytes);

  std::size_t size() const;

  Batch sample_batch(int B, int T, util::Rng& rng) const;

 private:
  std::vector<std::uint8_t> bytes_;
};

} // namespace data
