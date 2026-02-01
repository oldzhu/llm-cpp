> [简体中文](2026-02-01_doc_sync_audit.zh-CN.md)

# 2026-02-01 — Doc sync audit (backends + tokenizers)

## Goal

Bring documentation back in sync with the current architecture after adding:
- a real backend seam (CPU backend today; CUDA/HIP planned)
- pluggable tokenizers (ByteTokenizer + a minimal BPE tokenizer)

## What changed

### Docs

- Updated backend docs to reflect the existing seam and current folder naming:
  - backend seam is real (KernelBackend + registry; matmul/bmm route through it)
  - corrected `src/backends/*` → `src/backend/*`
- Updated training/inference walkthrough to clarify:
  - tokenization is pluggable
  - the current training/sanity data pipeline is still byte-based (ByteDataset, V=256)
- Added a status section to the tokenizer choice note:
  - Tokenizer interface exists
  - ByteTokenizer and a minimal/import-style BpeTokenizer exist
  - end-to-end subword training still needs a token-id dataset path
- Updated autograd/memory layout note:
  - `matmul2d` builds the autograd node in `src/ops.cpp` but executes math via the backend (CPU backend uses explicit loops)

### Code

- Fixed generation output for non-byte tokenizers:
  - generated token ids are decoded via `Tokenizer::decode` and printed
  - `--ascii-only` / `--escape-bytes` are restricted to the byte tokenizer
- Prevented confusing runs:
  - training and dataset-based sanity checks now require `--tokenizer byte` (until a token-id dataset path exists)

## Verification

- Ran `ctest -C Debug` (passed).
