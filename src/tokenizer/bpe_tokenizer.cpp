#include "bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

// Custom hash for std::pair<std::string, std::string> (needed for MSVC)
namespace std {
    template<>
    struct hash<std::pair<std::string, std::string>> {
        std::size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
        }
    };
}

// Remove custom hash for std::pair (use standard hash)

// Load vocab and merges (GPT-2 format)
static void load_vocab_and_merges(const std::string& vocab_path, const std::string& merges_path,
                                  std::unordered_map<std::string, int>& token_to_id,
                                  std::vector<std::string>& id_to_token,
                                  std::unordered_map<std::pair<std::string, std::string>, int, PairHash>& bpe_ranks) {
    // Vocab: one token per line
    std::ifstream vf(vocab_path);
    if (!vf) throw std::runtime_error("BpeTokenizer: failed to open vocab file");
    std::string line;
    while (std::getline(vf, line)) {
        if (!line.empty()) {
            int id = static_cast<int>(id_to_token.size());
            token_to_id[line] = id;
            id_to_token.push_back(line);
        }
    }
    vf.close();
    // Merges: skip first line, then each line is 'A B'
    std::ifstream mf(merges_path);
    if (!mf) throw std::runtime_error("BpeTokenizer: failed to open merges file");
    std::getline(mf, line); // skip header
    int rank = 0;
    while (std::getline(mf, line)) {
        std::istringstream iss(line);
        std::string a, b;
        if (iss >> a >> b) {
            bpe_ranks[{a, b}] = rank++;
        }
    }
    mf.close();
}

BpeTokenizer::BpeTokenizer(const std::string& vocab_path, const std::string& merges_path) {
    load_vocab_and_merges(vocab_path, merges_path, token_to_id, id_to_token, bpe_ranks);
}

// BPE encode: split into bytes, then iteratively merge pairs by rank
std::vector<int> BpeTokenizer::encode(const std::string& text) const {
    // Start: split text into bytes as unicode chars (for minimal test, just bytes)
    std::vector<std::string> symbols;
    for (unsigned char ch : text) {
        symbols.push_back(std::string(1, ch));
    }
    // Map to vocab if possible
    std::vector<std::string> tokens = symbols;
    // Merge loop
    while (tokens.size() > 1) {
        int best_rank = INT_MAX;
        int best_pos = -1;
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            auto it = bpe_ranks.find({tokens[i], tokens[i+1]});
            if (it != bpe_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = static_cast<int>(i);
            }
        }
        if (best_pos == -1) break;
        // Merge best pair
        tokens[best_pos] = tokens[best_pos] + tokens[best_pos+1];
        tokens.erase(tokens.begin() + best_pos + 1);
    }
    // Map to ids
    std::vector<int> ids;
    for (const auto& t : tokens) {
        auto it = token_to_id.find(t);
        if (it == token_to_id.end()) throw std::runtime_error("BpeTokenizer: unknown token: " + t);
        ids.push_back(it->second);
    }
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    for (int id : tokens) {
        if (id < 0 || id >= static_cast<int>(id_to_token.size())) throw std::runtime_error("BpeTokenizer: token id out of range");
        text += id_to_token[id];
    }
    return text;
}
