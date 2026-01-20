> [简体中文](copilot-instructions.zh-CN.md)

# Copilot instructions (build-llm-using-cpp)

This repository is a **teaching-first, from-scratch** C++ implementation of a tiny GPT-style model.

**Primary goal:** keep the codebase easy to reason about and verify. Prefer clarity and correctness over cleverness.

## Non‑negotiables

- **Keep changes minimal and explicit.** Avoid refactors unless they directly support the requested feature.
- **Preserve determinism.** If touching RNG, sampling, data ordering, or initialization, keep runs reproducible.
- **No “magic” dependencies.** Prefer standard C++ and tiny self-contained utilities.
- **Teach through the code.** When adding a feature, also add an explanation that maps math → code.

## Docs & tests contract

When implementing a new feature that changes behavior or adds a new algorithm:

1) Update docs (high-level + learn-by-hand) and keep them consistent.
2) Add or update tests that fail before the change and pass after.
3) Run the test suite.

The canonical guidance lives here:
- `docs/project_lifecycle_guidelines.md`
- `docs/README.md`

## Bilingual docs (English + 简体中文)

From now on, every Markdown document must have **two versions**:
- English: `X.md`
- Simplified Chinese: `X.zh-CN.md`

Every English doc must include a top-of-file switch link:
- `> [简体中文](X.zh-CN.md)`

Every Chinese doc must include a top-of-file switch link:
- `> [English](X.md)`

When adding a new doc or changing behavior:
- Update both language versions in the same change.
- If the Chinese translation is not complete yet, include a clear “translation in progress” note, and do not remove the English version.

## Implementation style

- Prefer **shape-first** code: state tensor shapes and index conventions in code/doc.
- Keep tensor layout assumptions explicit (row-major, flat offsets, etc.).
- Avoid adding inline comments that restate obvious code; prefer comments that connect to formulas, shapes, and invariants.

## Extending the project

- If exploring alternative implementations (e.g., attention variants, RoPE, KV-cache, GQA), prefer adding them as **side-by-side variants** rather than rewriting the baseline.
- If introducing a new backend path (SIMD/GPU), keep a clean boundary between:
  - model math / op semantics
  - kernel/backend implementation details

## When uncertain

- Ask for clarification on the exact paper/variant and the expected behavior (shapes, masking, caching semantics).
- Default to adding a small, verifiable stepping-stone (plus docs/tests) rather than a large one-shot implementation.
