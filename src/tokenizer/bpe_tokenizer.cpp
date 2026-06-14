#include "bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

namespace std {
    template<>
    struct hash<std::pair<std::string, std::string>> {
        std::size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
        }
    };
}

// GPT-2 bytes_to_unicode: maps raw bytes 0-255 to unicode characters
static std::unordered_map<int, std::string> byte_unicode_map() {
  std::unordered_map<int, std::string> m;
  for (int i = 33; i <= 126; ++i) m[i] = std::string(1, static_cast<char>(i));
  for (int i = 161; i <= 172; ++i) m[i] = std::string(1, static_cast<char>(i));
  for (int i = 174; i <= 255; ++i) m[i] = std::string(1, static_cast<char>(i));
  int n = 0;
  for (int i = 0; i <= 255; ++i) {
    if (m.find(i) == m.end()) {
      int cp = 256 + n++;
      m[i] = std::string(1, static_cast<char>(0xC0 | (cp >> 6))) +
             std::string(1, static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }
  return m;
}

static std::unordered_map<std::string, int> unicode_byte_map() {
  auto b2u = byte_unicode_map();
  std::unordered_map<std::string, int> u2b;
  for (auto& p : b2u) u2b[p.second] = p.first;
  return u2b;
}

// Parse JSON string literal
static std::string unescape_json(const std::string& s) {
  std::string out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 1 < s.size()) {
      switch (s[++i]) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u':
          if (i + 4 < s.size()) {
            std::string hex = s.substr(i + 1, 4);
            char* e = nullptr;
            long val = std::strtol(hex.c_str(), &e, 16);
            if (e == hex.c_str() + 4) {
              if (val < 0x80) out += static_cast<char>(val);
              else if (val < 0x800) {
                out += static_cast<char>(0xC0 | (val >> 6));
                out += static_cast<char>(0x80 | (val & 0x3F));
              } else {
                out += static_cast<char>(0xE0 | (val >> 12));
                out += static_cast<char>(0x80 | ((val >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (val & 0x3F));
              }
              i += 4;
            }
          }
          break;
        default: out += s[i]; break;
      }
    } else {
      out += s[i];
    }
  }
  return out;
}

static bool load_json_vocab(const std::string& path,
                            std::unordered_map<std::string, int>& t2i,
                            std::vector<std::string>& i2t) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  std::size_t sz = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::string j(sz, '\0');
  f.read(&j[0], static_cast<std::streamsize>(sz));

  std::size_t pos = 0;
  while (pos < j.size() && j[pos] != '{') ++pos;
  if (pos >= j.size()) return false;
  ++pos;
  while (pos < j.size()) {
    while (pos < j.size() && (j[pos]==' '||j[pos]=='\n'||j[pos]=='\r'||j[pos]=='\t'||j[pos]==',')) ++pos;
    if (pos >= j.size() || j[pos] == '}') break;
    if (j[pos] != '"') return false;
    std::size_t ks = pos + 1, ke = ks;
    while (ke < j.size()) { if (j[ke]=='\\') { ke+=2; continue; } if (j[ke]=='"') break; ++ke; }
    std::string key = unescape_json(j.substr(ks, ke - ks));
    pos = ke + 1;
    while (pos < j.size() && j[pos] != ':') ++pos;
    ++pos;
    while (pos < j.size() && (j[pos]==' '||j[pos]=='\t')) ++pos;
    int val = 0;
    while (pos < j.size() && j[pos]>='0' && j[pos]<='9') { val = val*10 + (j[pos]-'0'); ++pos; }
    if (static_cast<int>(i2t.size()) <= val) i2t.resize(static_cast<std::size_t>(val) + 1);
    i2t[static_cast<std::size_t>(val)] = key;
    t2i[key] = val;
  }
  return !i2t.empty();
}

static void load_vocab_merges(const std::string& vp, const std::string& mp,
                              std::unordered_map<std::string, int>& t2i,
                              std::vector<std::string>& i2t,
                              std::unordered_map<std::pair<std::string,std::string>,int,PairHash>& ranks) {
  if (!load_json_vocab(vp, t2i, i2t)) {
    std::ifstream vf(vp);
    if (!vf) throw std::runtime_error("BpeTokenizer: failed to open vocab file");
    std::string line;
    while (std::getline(vf, line)) {
      if (!line.empty()) { int id = static_cast<int>(i2t.size()); t2i[line] = id; i2t.push_back(line); }
    }
  }
  std::ifstream mf(mp);
  if (!mf) throw std::runtime_error("BpeTokenizer: failed to open merges file");
  std::string line; std::getline(mf, line);
  int rank = 0;
  while (std::getline(mf, line)) {
    std::istringstream iss(line);
    std::string a, b;
    if (iss >> a >> b) ranks[{a, b}] = rank++;
  }
}

BpeTokenizer::BpeTokenizer(const std::string& vp, const std::string& mp) {
  load_vocab_merges(vp, mp, token_to_id, id_to_token, bpe_ranks);
}

std::vector<int> BpeTokenizer::encode(const std::string& text) const {
  static auto b2u = byte_unicode_map();
  std::vector<std::string> tokens;
  for (unsigned char byte : text) {
    auto it = b2u.find(static_cast<int>(byte));
    if (it != b2u.end()) tokens.push_back(it->second);
  }
  if (tokens.empty()) return {};

  while (tokens.size() > 1) {
    int best = INT_MAX, best_i = -1;
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
      auto it = bpe_ranks.find({tokens[i], tokens[i+1]});
      if (it != bpe_ranks.end() && it->second < best) { best = it->second; best_i = static_cast<int>(i); }
    }
    if (best_i == -1) break;
    tokens[best_i] = tokens[best_i] + tokens[best_i+1];
    tokens.erase(tokens.begin() + best_i + 1);
  }

  std::vector<int> ids;
  for (auto& t : tokens) {
    auto it = token_to_id.find(t);
    if (it == token_to_id.end()) throw std::runtime_error("BPE: unknown token: " + t);
    ids.push_back(it->second);
  }
  return ids;
}

std::string BpeTokenizer::decode(const std::vector<int>& tokens) const {
  static auto u2b = unicode_byte_map();
  std::string text;
  for (int id : tokens) {
    if (id < 0 || id >= static_cast<int>(id_to_token.size()))
      throw std::runtime_error("BPE: token id out of range");
    const std::string& tok = id_to_token[id];
    for (std::size_t i = 0; i < tok.size(); ) {
      unsigned char c = static_cast<unsigned char>(tok[i]);
      std::size_t clen = 1;
      if ((c & 0x80) == 0) clen = 1;
      else if ((c & 0xE0) == 0xC0) clen = 2;
      else if ((c & 0xF0) == 0xE0) clen = 3;
      else clen = 4;
      if (i + clen > tok.size()) clen = 1;
      std::string ch = tok.substr(i, clen);
      auto it = u2b.find(ch);
      if (it != u2b.end()) text += static_cast<char>(it->second);
      else if (clen == 1) text += ch;
      i += clen;
    }
  }
  return text;
}
