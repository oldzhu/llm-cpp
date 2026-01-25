#pragma once
#include <string>
#include <vector>

// Abstract base class for all tokenizers
struct Tokenizer {
    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& tokens) const = 0;
    virtual int vocab_size() const = 0;
    virtual ~Tokenizer() {}
};
