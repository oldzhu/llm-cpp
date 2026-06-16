#include "spm_tokenizer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <climits>

// Minimal JSON parser for SPM vocab: {"piece": {"id": N, "score": S}, ...}
SpmTokenizer::SpmTokenizer(const std::string& vocab_path) {
    std::ifstream f(vocab_path);
    if (!f) throw std::runtime_error("SpmTokenizer: failed to open " + vocab_path);
    std::string j(std::istreambuf_iterator<char>(f), {});

    // Parse {"key1": {"id": N1, "score": S1}, "key2": ...}
    std::size_t pos = 0;
    while (pos < j.size() && j[pos] != '{') ++pos;
    if (pos >= j.size()) throw std::runtime_error("SpmTokenizer: invalid JSON");
    ++pos;

    int max_id = -1;
    std::vector<std::pair<int, std::pair<std::string, float>>> entries;

    while (pos < j.size()) {
        while (pos < j.size() && (j[pos]==' '||j[pos]=='\n'||j[pos]=='\r'||j[pos]=='\t'||j[pos]==',')) ++pos;
        if (pos >= j.size() || j[pos] == '}') break;
        if (j[pos] != '"') break;
        // Read key
        std::size_t ks = pos + 1, ke = ks;
        while (ke < j.size()) { if (j[ke]=='\\'&&ke+1<j.size()) { ke+=2; continue; } if (j[ke]=='"') break; ++ke; }
        std::string piece = j.substr(ks, ke - ks);
        // Simple unescape
        for (std::size_t i = 0; i < piece.size(); ++i) {
            if (piece[i] == '\\' && i+1 < piece.size()) {
                char c = piece[i+1];
                if (c == 'n') { piece.replace(i, 2, "\n"); }
                else if (c == 't') { piece.replace(i, 2, "\t"); }
                else if (c == 'r') { piece.replace(i, 2, "\r"); }
                else if (c == '"') { piece.replace(i, 2, "\""); }
                else if (c == '\\') { piece.replace(i, 2, "\\"); }
            }
        }
        pos = ke + 1;
        while (pos < j.size() && j[pos] != '{') ++pos;
        if (pos >= j.size()) break;
        ++pos; // skip '{'
        // Read id and score
        int id = -1; float score = 0.0f;
        while (pos < j.size() && j[pos] != '}') {
            while (pos < j.size() && (j[pos]==' '||j[pos]=='\n'||j[pos]=='\t'||j[pos]==',')) ++pos;
            if (j[pos] == '}') break;
            if (j[pos] != '"') { ++pos; continue; }
            std::size_t fks = pos + 1, fke = fks;
            while (fke < j.size() && j[fke] != '"') ++fke;
            std::string fkey = j.substr(fks, fke - fks);
            pos = fke + 1;
            while (pos < j.size() && j[pos] != ':') ++pos;
            ++pos;
            while (pos < j.size() && (j[pos]==' '||j[pos]=='\t')) ++pos;
            if (fkey == "id") {
                id = 0;
                while (pos < j.size() && j[pos] >= '0' && j[pos] <= '9') { id = id*10 + (j[pos]-'0'); ++pos; }
            } else if (fkey == "score") {
                std::size_t ns = pos;
                while (pos < j.size() && (j[pos]=='-'||j[pos]=='.'||j[pos]=='e'||j[pos]=='E'||j[pos]=='+'||(j[pos]>='0'&&j[pos]<='9'))) ++pos;
                score = std::stof(j.substr(ns, pos - ns));
            }
        }
        ++pos; // skip '}'
        if (id >= 0) {
            entries.push_back({id, {piece, score}});
            if (id > max_id) max_id = id;
        }
    }

    if (max_id < 0) throw std::runtime_error("SpmTokenizer: no tokens found");

    id_to_piece.resize(static_cast<std::size_t>(max_id) + 1);
    scores.resize(static_cast<std::size_t>(max_id) + 1, 0.0f);
    for (auto& e : entries) {
        id_to_piece[static_cast<std::size_t>(e.first)] = e.second.first;
        piece_to_id[e.second.first] = e.first;
        scores[static_cast<std::size_t>(e.first)] = e.second.second;
    }
}

std::vector<int> SpmTokenizer::encode(const std::string& text) const {
    // Simple longest-prefix-match segmentation (similar to SentencePiece Unigram)
    // Walk through text, at each position find the longest matching token
    std::vector<int> ids;
    std::size_t pos = 0;
    while (pos < text.size()) {
        int best_id = -1;
        std::size_t best_len = 0;
        // Try increasing prefixes
        std::string prefix;
        for (std::size_t end = pos; end < text.size(); ++end) {
            prefix += text[end];
            auto it = piece_to_id.find(prefix);
            if (it != piece_to_id.end()) {
                best_id = it->second;
                best_len = end - pos + 1;
            }
        }
        if (best_id >= 0) {
            ids.push_back(best_id);
            pos += best_len;
        } else {
            // Fallback: encode byte as raw (like SentencePiece's <0xNN>)
            ids.push_back(static_cast<int>(static_cast<unsigned char>(text[pos])) + 3); // simple fallback
            ++pos;
        }
    }
    return ids;
}

std::string SpmTokenizer::decode(const std::vector<int>& tokens) const {
    std::string out;
    for (int id : tokens) {
        if (id >= 0 && id < static_cast<int>(id_to_piece.size())) {
            out += id_to_piece[static_cast<std::size_t>(id)];
        }
    }
    return out;
}
