#include "data.h"

#include <stdexcept>

namespace data {

ByteDataset::ByteDataset(std::vector<std::uint8_t> bytes) : bytes_(std::move(bytes)) {
  if (bytes_.size() < 1024) {
    throw std::runtime_error("dataset too small (need at least 1024 bytes)");
  }
}

std::size_t ByteDataset::size() const {
  return bytes_.size();
}

Batch ByteDataset::sample_batch(int B, int T, util::Rng& rng) const {
  if (B <= 0 || T <= 0) throw std::runtime_error("sample_batch: invalid B/T");
  if (bytes_.size() < static_cast<std::size_t>(T + 1)) throw std::runtime_error("sample_batch: dataset too small");

  const std::int32_t max_start = static_cast<std::int32_t>(bytes_.size() - static_cast<std::size_t>(T + 1));

  Batch batch;
  batch.B = B;
  batch.T = T;
  batch.x.resize(static_cast<std::size_t>(B) * T);
  batch.y.resize(static_cast<std::size_t>(B) * T);

  for (int b = 0; b < B; ++b) {
    const std::int32_t start = rng.uniform_int(0, max_start);
    for (int t = 0; t < T; ++t) {
      const std::uint8_t xb = bytes_[static_cast<std::size_t>(start + t)];
      const std::uint8_t yb = bytes_[static_cast<std::size_t>(start + t + 1)];
      batch.x[static_cast<std::size_t>(b) * T + t] = static_cast<std::int32_t>(xb);
      batch.y[static_cast<std::size_t>(b) * T + t] = static_cast<std::int32_t>(yb);
    }
  }

  return batch;
}

} // namespace data
