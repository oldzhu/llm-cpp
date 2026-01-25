# 2026-01-25: BPE Tokenizer – Test and Feature Ideas

## Recommended BPE Tests
- Encode/decode roundtrip for various strings (ASCII, Unicode, edge cases)
- Test with multi-token merges (e.g., "abcde" → ["abc", "de"])
- Test unknown/invalid input handling (should throw or fallback)
- Test vocab size and token-to-id/id-to-token consistency
- Test with real GPT-2 vocab/merges files (if available)
- Test CLI integration: run with --tokenizer bpe and check output

## Possible BPE Features to Add
- Byte fallback: allow encoding bytes not in vocab as special tokens
- Unicode-aware splitting: handle multi-byte UTF-8 characters more robustly
- Caching of BPE merges for speed
- Support for special tokens (e.g., <unk>, <pad>, <bos>, <eos>)
- Export/load vocab/merges in JSON or other formats
- Add BPE training (learn merges from data, not just load)
- Add SentencePiece Unigram or WordPiece as alternative tokenizers
- Add CLI flag to print tokenization results for a given string
