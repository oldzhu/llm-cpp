# 2026-01-25: CPU backend validation and tokenizer/Chinese embedding discussion

## CPU backend: training/inference re-validation
- Rebuilt and ran `ctest` in Debug: all tests passed (1/1).
- Ran a quick smoke run: trained 60 steps on `the-verdict.txt`, saved a checkpoint, then generated from it. Loss dropped smoothly; generation works (still noisy at only 60 steps, as expected).
- Ran dataset “next-byte sanity” against that checkpoint: got 2/5 top‑1 correct; also expected at low step count (improves with more steps/larger model).
- How to run these modes is documented in `README.md`; flags are implemented in `src/main.cpp`.

## Can byte-level embedding represent Chinese?
- Yes: if your input is UTF‑8, byte-level vocab=256 can represent Chinese, because Chinese characters are just sequences of UTF‑8 bytes (typically 3 bytes/char).
- However, byte-level is inefficient for Chinese (and any non-ASCII):
  - Sequence length is longer (more tokens per character), so the model needs more context and compute.
  - The model may generate invalid UTF‑8 byte sequences more often early in training, so output can look garbled unless you use `--escape-bytes 1` (or post-process bytes as UTF‑8).
- So: “cannot embed Chinese” is not quite right; it can, but it’s a rough baseline and tends to be lower quality/slower to learn on Chinese-heavy corpora.

## Next step: switching from bytes → BPE (or similar)
- GPT‑2 style byte-fallback BPE is a great next upgrade: it still supports any Unicode text (so Chinese is fine), but reduces token counts by merging frequent byte sequences into multi-byte tokens.
- If desired, implement a minimal “load-only” tokenizer (no training) where:
  - keep the current byte tokenizer as fallback,
  - load a BPE vocab/merges file,
  - bump `vocab_size` accordingly and update checkpoint format.
- SentencePiece Unigram is also a good option for Chinese.
