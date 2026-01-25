#pragma once
#include "tokenizer.h"

// Byte-level tokenizer: 1 byte = 1 token, vocab size 256
struct ByteTokenizer : public Tokenizer {
    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& tokens) const override;
    int vocab_size() const override { return 256; }
};
