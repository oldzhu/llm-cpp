# 2026-01-25: Tokenizer/Embedding plugin design discussion

## Pluggable tokenizer/embedding design (summary)
- Use a similar pattern as backend dispatch (CPU/NVIDIA/AMD): define a tokenizer interface (abstract base class) with methods like `encode(text) -> tokens` and `decode(tokens) -> text`.
- Implement each tokenizer (Byte, GPT-2 BPE, SentencePiece) as a separate class, all inheriting from the interface.
- Add a registry/factory to select the tokenizer at runtime (by config or CLI flag).
- Embedding layer should depend only on vocab size and token type, not on tokenizer internals.
- Document the shape/contract for each tokenizer and embedding, and add tests for encode/decode roundtrip and embedding lookup.

## Benefits
- Swap tokenizers and embeddings dynamically for experiments.
- Debug/test each variant in isolation.
- Cleanly extend to new tokenization methods in the future.

## Next: design sketch (class/struct layout and registry pattern)

---

## Design sketch: C++-style pseudocode

```cpp
// Tokenizer interface
struct Tokenizer {
  virtual std::vector<int> encode(const std::string& text) const = 0;
  virtual std::string decode(const std::vector<int>& tokens) const = 0;
  virtual int vocab_size() const = 0;
  virtual ~Tokenizer() {}
};

// Byte-level tokenizer
struct ByteTokenizer : public Tokenizer {
  // ...implement encode/decode for bytes...
};

// GPT-2 BPE tokenizer
struct BpeTokenizer : public Tokenizer {
  // ...implement encode/decode for BPE...
};

// SentencePiece Unigram tokenizer
struct SpUnigramTokenizer : public Tokenizer {
  // ...implement encode/decode for SentencePiece...
};

// Tokenizer registry/factory
struct TokenizerRegistry {
  static Tokenizer* create(const std::string& name, const std::string& vocab_path);
};

// Usage example
std::unique_ptr<Tokenizer> tok(TokenizerRegistry::create("bpe", "bpe_vocab.txt"));
auto tokens = tok->encode("你好，世界！");
std::string text = tok->decode(tokens);
```

---

## Design sketch: abstract diagram

```
+-------------------+
|   Tokenizer (ABC) |
+-------------------+
        ^
        |
+-------------------+    +-------------------+    +--------------------------+
|   ByteTokenizer   |    |   BpeTokenizer    |    |   SpUnigramTokenizer     |
+-------------------+    +-------------------+    +--------------------------+

All implement:
  encode(text) -> tokens
  decode(tokens) -> text
  vocab_size()

+-------------------+
| TokenizerRegistry |
+-------------------+
      |
      v
+-------------------+
|   Tokenizer*      |
+-------------------+

Embedding layer:
  - Only depends on vocab_size() and token type
  - Not on tokenizer internals
```
