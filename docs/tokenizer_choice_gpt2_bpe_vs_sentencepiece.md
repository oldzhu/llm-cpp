# Tokenizer choice: GPT-2 BPE vs SentencePiece (BPE/Unigram)

This note is a decision aid for the next-step feature: moving from the current **byte vocab (256)** to **subword tokens** (word-like tokens) by adding a tokenizer + vocabulary artifacts and switching the dataset/model to that vocab size.

Scope of this doc:
- Compare **classic GPT-2-style BPE** vs **SentencePiece** approaches.
- Clarify what it means to **import an existing vocab/merges** vs **learn/train a tokenizer** from data.
- Provide a recommendation aligned with this repo’s constraints (teaching-first, minimal deps, deterministic).

## Current baseline constraints (what we should preserve)

- Teaching-first, from-scratch C++.
- Determinism / reproducibility.
- Minimal dependencies.
- Clear “math/data → code” mapping.
- Checkpoint compatibility matters (tokenization must be tied to checkpoints).

## Option A: GPT-2-style “classic BPE” (byte-level)

### What it is

A BPE tokenizer that starts from a **byte-level alphabet** and repeatedly merges the most frequent adjacent pairs during training (or uses a precomputed merge table at runtime).

In practice, GPT-2 tokenization often includes:
- A byte-to-symbol mapping to ensure every input byte is representable.
- A pre-tokenization step (commonly regex-ish) that splits text into pieces before applying merges.

### Pros

- Very common and well understood.
- Robust on arbitrary text: byte-level start means no “unknown character” edge cases.
- Runtime is conceptually simple: encode bytes → apply merge ranks → ids.
- Good “teaching surface area”: merges are tangible and easy to illustrate.
- Easier to implement from scratch (vs SentencePiece Unigram) while staying dependency-free.

### Cons

- Exact GPT-2 compatibility has extra details beyond “BPE merges” (byte mapping + pretokenization conventions).
- If we want full interoperability with GPT-2 tokenizers, we must match these details precisely.
- BPE training is another piece of engineering (though manageable).

## Option B: SentencePiece

SentencePiece is a tokenizer framework that commonly produces either:
- **BPE** model, or
- **Unigram LM** model (a different subword algorithm)

### What it is

- Works directly on the input text stream.
- Typically uses a whitespace marker token (often `▁`) to preserve word boundary information.

### Pros

- Good, predictable whitespace handling (helpful for decoding and for “word-ish token” behavior).
- Widely used for many open models.
- Unigram can be strong in practice and conceptually nice (token inventory + probabilities).

### Cons

- Implementing SentencePiece-compatible behavior from scratch is non-trivial.
  - Unigram in particular is a bigger algorithmic leap than BPE merges.
- Using the official SentencePiece library conflicts with the repo’s “minimal dependencies” goal.
- Exact compatibility may require matching normalization/special token conventions.

## Interoperability vs simplicity: the key trade

- If we prioritize **teaching and minimal deps**, start with a self-contained BPE implementation (GPT-2-ish).
- If we prioritize **compatibility with existing open models/tokenizer assets**, SentencePiece file formats are common, but implementing full compatibility is more work.

## “Import an existing vocab/merges” vs “learn/train one”

This choice is independent of GPT-2 vs SentencePiece: it’s about where the tokenizer artifacts come from.

### 1) Import existing artifacts (use a pretrained tokenizer)

You take vocab/merge/model files produced elsewhere and use them directly.

Examples of artifacts:
- GPT-2 BPE: `vocab.json` + `merges.txt`
- SentencePiece: `tokenizer.model` (plus optional `.vocab`)

Pros:
- Immediate interoperability with that ecosystem.
- No need to implement tokenizer training first.

Cons:
- File formats and edge-case behavior can be complicated.
- Vocab sizes are often large (e.g., 30k–100k), which increases embedding + softmax cost in this CPU FP32 baseline.
- Checkpoints must record the tokenizer identity/version to avoid mismatched vocabularies.

### 2) Learn/train tokenizer artifacts from our corpus

You train a tokenizer (BPE merges or Unigram model) on our dataset (e.g., `the-verdict.txt` or a larger corpus) and produce a small vocab.

Pros:
- We can pick a smaller vocab (e.g., 2k–10k) to keep training fast.
- Strong teaching value: learners see how tokenization is constructed.
- Artifacts can be made intentionally simple and fully documented.

Cons:
- Not directly compatible with other tokenizers/checkpoints.
- Requires writing and validating a training pipeline.

## Recommendation for this repo (pragmatic v1)

For a first milestone that extends the core without “magic dependencies”:

1) Implement a **simple BPE tokenizer** as a new module (self-contained C++).
2) Start with a small vocab target (e.g., 2k–8k) trained on our dataset.
3) Keep the artifact format deliberately simple (e.g., a flat text file for merges + token list).
4) Add strict checkpoint compatibility rules:
   - checkpoint stores tokenizer name/version + vocab size
   - loading fails if tokenizer mismatch

Then, if we later want compatibility:
- Add a second tokenizer backend that can import GPT-2 merges/vocab, or a SentencePiece-compatible reader.

## What needs to change in the codebase (high-level)

- Data pipeline:
  - dataset becomes `std::vector<int32_t> token_ids` instead of bytes
  - batching samples token substrings
- Model config:
  - `vocab_size` becomes dynamic based on tokenizer vocab
- Sampling/printing:
  - generation prints decoded tokens (token ids → string)
- Checkpoints:
  - store tokenizer identity + vocab hash/version and refuse mismatches

## Tests we should add (minimum)

- Tokenizer round-trip:
  - decode(encode(text)) is “reasonable” (exact equality depends on whitespace conventions)
- Golden tokenization example:
  - a tiny known string has a known token-id sequence
- Checkpoint compatibility:
  - loading fails when vocab/tokenizer metadata mismatches

## Decision checklist

Choose GPT-2-style BPE first if:
- minimal dependencies is non-negotiable
- we want to teach the core algorithm clearly
- we are OK being “GPT-2-ish” initially, then iterating

Choose SentencePiece compatibility first if:
- interoperability with existing tokenizers is the top priority
- we accept more implementation complexity (or an external dependency)

