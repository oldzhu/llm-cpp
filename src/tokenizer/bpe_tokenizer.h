#pragma once
#include "tokenizer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// Hash for pair<string, string>
struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const {
        return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
    }
};

// Minimal GPT-2 style BPE tokenizer (load-only, no training)
struct BpeTokenizer : public Tokenizer {
    // Construct from vocab and merges file paths
    BpeTokenizer(const std::string& vocab_path, const std::string& merges_path);
    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& tokens) const override;
    int vocab_size() const override { return static_cast<int>(id_to_token.size()); }

    // --- Internal ---
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks;
};
