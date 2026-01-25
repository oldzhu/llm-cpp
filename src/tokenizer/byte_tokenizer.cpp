#include "byte_tokenizer.h"
#include <stdexcept>

std::vector<int> ByteTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (unsigned char ch : text) {
        tokens.push_back(static_cast<int>(ch));
    }
    return tokens;
}

std::string ByteTokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    text.reserve(tokens.size());
    for (int t : tokens) {
        if (t < 0 || t > 255) throw std::runtime_error("ByteTokenizer: token out of range");
        text.push_back(static_cast<unsigned char>(t));
    }
    return text;
}
