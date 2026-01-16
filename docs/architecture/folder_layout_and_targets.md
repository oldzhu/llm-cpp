# Folder layout + CMake target scheme (proposal)

This repo is intentionally small, but we want a structure that scales to:
- multiple algorithm variants (MHA, KV-cache, RoPE, GQA, etc.)
- multiple backends (naive CPU, SIMD, CUDA/HIP/Vulkan…)
- teaching-first docs + tests per feature

The guiding rule: **the baseline stays readable**. New features land as additive modules.

## Proposed folder layout

```
src/
  core/                 # “baseline” model + ops (CPU reference)
    tensor.*
    ops.*
    model.*
    optim.*
    checkpoint.*
    data.*
    util.*

  backend/              # kernel/backend boundary (initially CPU-only)
    backend.h           # interface + backend selection (later)
    cpu/                # CPU reference backend implementation (later)

  variants/             # side-by-side algorithm variants
    mha/                # multi-head attention variant module
      mha_attention.h
      mha_attention.cpp

apps/
  train_gpt_main.cpp    # CLI executable(s) (optional future split)

tests/
  test_main.cpp         # test harness; can include variant tests

docs/
  README.md
  architecture/
    folder_layout_and_targets.md
  variants/
    mha/
      README.md
      learn_by_hand.md
```

Notes:
- This doc describes the *intended* structure. We can migrate gradually.
- Variants should be **additive**: no rewrites of baseline unless necessary.

## Proposed CMake targets

Keep targets small and composable:

- **`llm_core`** (static library)
  - Baseline math + autograd + model + training utilities
  - No CLI parsing

- **`llm_backend`** (INTERFACE library initially)
  - Provides include path for backend abstractions
  - Future: `llm_backend_cpu`, `llm_backend_cuda`, etc.

- **`llm_variant_mha`** (static library)
  - Multi-head attention implementation as a side-by-side variant
  - Depends on `llm_core`

- **`train_gpt`** (executable)
  - The CLI app, links `llm_core` (and optionally variants)

- **`test_build_llm`** (executable)
  - Test runner, links `llm_core` and selected variants

### Why this scheme?

- **Faster iteration**: variants compile independently.
- **Teaching clarity**: you can diff baseline vs variant at file/module level.
- **Backend evolution**: later we can route hot ops through a backend interface without rewriting the model code.

## Adding a new variant (template)

For a new feature (example: KV-cache):

1) Code: `src/variants/kv_cache/...`
2) Docs (2 files minimum): `docs/variants/kv_cache/README.md` + `learn_by_hand.md`
3) Test: add one focused test to `tests/test_main.cpp`
4) Wire up: add a `llm_variant_kv_cache` library in CMake and link it into tests

That way, each feature is **code + docs + test**, all discoverable.
