# Project lifecycle guidelines (plan + design principles)

This repo is a **learning-first** from-scratch GPT/Transformer implementation in C/C++. The purpose of this document is to define the rules we follow as the project grows so that:

- the code stays understandable and teachable
- every new feature is traceable to math/papers
- we can compare “core baseline” vs “extended variants” easily
- we can evolve from CPU → multiple GPU backends without losing correctness


## 1) Core goals (north star)

1) **Learning and teaching over raw performance** (at least until the core concepts are mastered)
- Prefer clarity and correctness first.
- Performance work comes after we can explain the math and verify correctness.

2) **Paper/math ↔ code traceability**
- For any feature, a learner should be able to answer:
  - What is the formula/algorithm?
  - What are the tensor shapes?
  - Which functions implement it?
  - Where do the indices and masks come from?

3) **Incremental evolution**
- Add one concept at a time.
- Keep each step small enough to reason about and “simulate in brain”.

4) **Correctness guardrails**
- Tests must protect correctness before we refactor or optimize.
- Every major feature that changes math should add at least one regression test.


## 2) Repo structure principles

### 2.1 Keep a stable reference baseline

The current baseline is the reference implementation:
- CPU FP32
- tiny autograd
- GPT-style decoder-only
- (currently) single-head causal attention

Rule: keep the baseline working and readable at all times.

### 2.2 Add new features as variants first (compare-by-diff)

When adding a major feature (e.g., multi-head attention, KV-cache), prefer creating a **variant** rather than rewriting the baseline in-place.

Proposed pattern:

- `src/` remains the baseline.
- `variants/<feature_name>/` contains:
  - a minimal fork of only the necessary code
  - a separate entry point / target
  - its own docs and tests

This lets learners compare:
- baseline vs variant behavior
- baseline vs variant formulas
- baseline vs variant memory layout changes

(Once a variant is understood and tested, we can decide to merge it into baseline or keep it as an educational alternative.)

### 2.3 Backend/plugin split for CPU→GPU evolution

When we start GPU work, we should separate:

- **model logic** (shapes, graph structure, high-level algorithm)
- **kernel backend** (matmul, softmax, layernorm, attention kernels)

That enables:
- CPU reference backend
- CPU SIMD backend
- CUDA backend
- ROCm backend
- Vulkan backend
- MUSA backend

The model should call a small “kernel API surface” so backends are swappable.


## 3) Feature workflow (non-negotiable)

For every new major feature/plugin:

### 3.1 Documentation deliverables

You must ship:

1) **Math-to-code mapping doc**
- Add/extend `docs/transformer_math_to_code.md`.
- For new feature, add a dedicated doc, e.g.:
  - `docs/multihead_attention.md`
  - `docs/kv_cache.md`
  - `docs/moe.md`

2) **Learn-by-hand walkthrough doc**
- Provide a tiny numeric example + step-by-step flow.
- Example file:
  - `docs/learn_by_hand_<feature>.md`

3) **Indexing trace (if indexing/memory changes)**
- If the feature introduces new tensor layouts or caching, add a trace doc like:
  - `docs/attention_indexing_trace.md`

### 3.2 Code deliverables

- Targeted comments at feature boundaries:
  - top-of-function comments with formulas and shapes
  - mention any important implementation choices (masking, scaling, stability tricks)

### 3.3 Testing deliverables

At least one of:
- finite-difference gradcheck (for new differentiable ops)
- deterministic regression (loss decreases / exact numeric checks)
- shape and invariants tests (masking rules, cache correctness)

### 3.4 Review checklist

Before merging:
- `ctest` passes
- docs are linked from `docs/README.md`
- feature has at least one test
- code comments explain shapes and formulas


## 4) CPU→GPU progression strategy

We evolve in stages:

### Stage A — CPU correctness baseline
- Pure C++ reference kernels (naive loops)
- Clear indexing
- Tests + docs

### Stage B — CPU performance (still educational)
- profiling (find hotspots)
- blocked matmul
- SIMD kernels
- still keep a reference path for correctness

### Stage C — First GPU backend (choose one)
Start with one backend to reduce complexity:
- CUDA *or* ROCm first

Minimal kernel set to bring up first:
- matmul
- softmax
- layernorm
- attention (may be built from primitives initially)

### Stage D — Add more frameworks/backends
One by one:
- ROCm (rocBLAS/MIOpen)
- Vulkan
- MUSA

Rule: do not add multiple GPU backends simultaneously.


## 5) Libraries policy (learning-driven)

Default rule:
- Prefer “no high-level ML frameworks” in the core learning path.

However, we *do* allow adding industry libraries when it helps learning:
- CUDA: cuBLAS / cuDNN / cuSPARSE / cuSOLVER
- ROCm: rocBLAS / MIOpen

Guideline:
- When adding a library-backed path, keep a reference implementation for comparison.
- Document what the library call corresponds to mathematically (e.g., GEMM = matrix multiply).


## 6) Catalog of modern LLM techniques to learn (future backlog)

This is a curated list of “modern” techniques you may want to implement as variants over time. Each item should be introduced **one at a time** with docs/tests.

### Architecture / attention
- **RoPE** (Rotary Positional Embeddings)
  - Replace/add positional encoding; affects Q/K before attention.
- **GQA** (Grouped Query Attention)
  - Fewer K/V heads than Q heads; impacts attention shapes and caching.
- **MLA**
  - Term can vary by paper/author; treat as a feature proposal that requires a specific reference/definition before implementation.

### Normalization / activations
- **RMSNorm**
  - Norm variant commonly used in LLaMA-like models.
- **SwiGLU**
  - Activation/gating variant in modern FFNs.

### Mixture of Experts
- **MoE** (Mixture of Experts)
  - Routing + expert FFNs; adds load balancing concerns.

### Training / optimization / alignment
- **PPO** (Proximal Policy Optimization)
  - Common in RLHF-style fine-tuning; requires policy/value modeling and reward signals.
- **GRPO**
  - Often refers to group-based relative policy optimization methods; requires a clear reference definition before implementing.

### Retrieval / augmentation
- **RAFT**
  - Acronym can vary; treat as a retrieval/augmentation-related backlog item and define the exact algorithm before implementing.

Important note:
- For ambiguous acronyms (MLA/GRPO/RAFT), we should record the exact paper/blog reference we mean in the feature doc before coding.


## 7) Naming + timestamped notes

- Use `docs/notes/YYYY-MM-DD_<topic>.md` for time-stamped decisions.
- Keep `docs/README.md` as the index.


## 8) Recommended next steps from the current baseline

Learning-first order:

1) Multi-head attention (variant)
2) KV-cache (variant)
3) Affine norms (LayerNorm gamma/beta) or RMSNorm (choose one)
4) RoPE
5) GQA
6) CPU SIMD kernel experiments
7) First GPU backend (CUDA or ROCm)

Each step must follow the workflow in Section 3.
