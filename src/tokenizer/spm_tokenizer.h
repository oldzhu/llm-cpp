#pragma once
#include "tokenizer.h"
#include <string>
#include <vector>
#include <unordered_map>

// SentencePiece-compatible tokenizer.
// Reads a JSON vocab file extracted by scripts/extract_spm_vocab.py.
// Uses simple longest-prefix-match encoding (similar to SentencePiece Unigram).
struct SpmTokenizer : public Tokenizer {
    explicit SpmTokenizer(const std::string& vocab_path);
    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& tokens) const override;
    int vocab_size() const override { return static_cast<int>(id_to_piece.size()); }

private:
    std::vector<std::string> id_to_piece;
    std::unordered_map<std::string, int> piece_to_id;
    std::vector<float> scores; // optional, for future use
};
