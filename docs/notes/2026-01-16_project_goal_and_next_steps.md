> [简体中文](2026-01-16_project_goal_and_next_steps.zh-CN.md)

# Project goal and next steps (2026-01-16)

Context: this note captures the current intent and recommended roadmap for `build-llm-using-cpp`.

Snapshot:
- Date: 2026-01-16
- Git: 446a032


## Goal (what this project is for)

This repository is a **from-scratch, end-to-end GPT-style Transformer** in C++ designed primarily for **learning and teaching**:

- Implement training + inference (sampling) without high-level ML libraries.
- Keep a CPU FP32 baseline that is easy to reason about and debug.
- Provide strong “math/paper → code” traceability via docs and targeted comments.
- Keep correctness guardrails (tests) so refactors (e.g., attention changes) don’t silently break the model.

Practical learning outcomes:
- Understand how tokenization/data batching, forward pass, loss, backward/autograd, and optimizer updates work together.
- Build intuition for tensor shapes, row-major memory layout, and performance bottlenecks.
- Create a foundation that can later grow into SIMD and GPU backends (CUDA/ROCm/etc.).


## Current baseline (what exists today)

- Byte-level language modeling (vocab size 256): any text file works as data.
- Tiny GPT model:
  - learned token + positional embeddings
  - pre-norm decoder blocks
  - single-head causal self-attention
  - MLP with GELU
- Training loop + optional prompt generation in one run.
- Checkpoint save/load (`.json` config + `.bin` weights + optional optimizer state).
- Minimal autograd engine for reverse-mode differentiation.
- Test executable + CTest:
  - finite-diff gradchecks (representative ops)
  - tiny deterministic training regression
- Teaching docs under `docs/` (math-to-code mapping, walkthroughs, indexing trace, learn-by-hand).


## Next steps (recommended order)

### 1) Multi-head attention (MHA)
Why next:
- It’s the most important fidelity upgrade from the single-head baseline.
- KV-cache design depends on the final attention head layout.

What changes:
- Add `n_heads` to model config.
- Reshape/split Q/K/V into heads (typically `[B,T,C] -> [B,H,T,Ch]` where `Ch=C/H`).
- Compute attention per head, then concat and project back.

Docs/tests contract:
- Add a dedicated MHA doc and update `docs/transformer_math_to_code.md`.
- Add/extend tests to cover the new attention path.


### 2) KV-cache for faster generation
Why:
- Makes autoregressive generation scale roughly with generated length (avoid recomputing all past K/V each step).

What changes:
- Cache per-layer K/V across steps.
- Add an incremental forward path for generation.
- Update docs and likely checkpoint format.


### 3) Model fidelity upgrades (optional, still educational)
Candidates:
- LayerNorm affine parameters (gamma/beta)
- Dropout (optional)
- Better init/scaling conventions


### 4) Performance phase (CPU)
Focus:
- Replace naive matmul/attention loops with blocked + SIMD kernels.
- Profile hotspots (matmul, attention, softmax, layernorm).
- Keep correctness tests running frequently.


### 5) GPU backend(s)
Approach:
- Start with one backend (CUDA or ROCm), build a minimal kernel layer.
- Implement core kernels first (matmul, softmax, layernorm) then integrate into attention.


### 6) Longer-term learning goals
Only after the core training/inference stack is solid:

#### Tokenizer beyond bytes (subword / BPE “word tokens”)
Goal:
- Move from byte tokens (vocab=256) to subword tokens like GPT/Qwen/DeepSeek style tokenization.

What changes:
- Add a tokenizer + vocab file format (e.g., BPE merges + token-to-id map).
- Switch the dataset from “bytes” to “token ids”, including:
  - encode: text → token ids
  - decode: token ids → text (for generation output)
- Update model config so `vocab_size` matches the tokenizer vocabulary (instead of fixed 256).
- Update checkpoints to store tokenizer identity/version and refuse to load mismatched vocabularies.

Docs/tests contract:
- Document the exact tokenization algorithm + file formats used.
- Add a deterministic tokenizer round-trip test (encode→decode) and a small known-example tokenization test.

Notes:
- This is a bigger change than it looks because it touches every boundary: data loading, batching, sampling output, and checkpoint compatibility.

- Mixed precision
- Gradient checkpointing
- Distributed training
- RL-style fine-tuning (requires much more infrastructure and careful documentation)


## Documentation rule (non-negotiable for this repo)

Every major feature must ship with:
- a doc that maps **paper concept → equations → tensor shapes → code locations**
- targeted code comments at the function boundaries
- at least one regression test if math/behavior changes
