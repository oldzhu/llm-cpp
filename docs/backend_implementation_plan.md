> [简体中文](backend_implementation_plan.zh-CN.md)

# Backend implementation plan (CPU → CUDA → HIP)

This plan turns the high-level design in `docs/backend_design_cpu_cuda_hip.md` into concrete, incremental steps with explicit file-level changes.

## Current state (what we have today)

- `nn::Tensor` stores data/grad as `shared_ptr<vector<float>>` (CPU-only, contiguous row-major).
- `nn::*` ops in `src/ops.cpp` implement forward math and backward math directly (CPU loops) and build autograd `Node`s inline.
- `src/backend/backend.h` exists but is a placeholder (no dispatch, no concrete backend).

This is great as a teaching baseline, but it means *there is no seam* to swap in CUDA/HIP kernels.

## Guiding principles

- Keep **model code unchanged** (`src/model.cpp` continues calling `nn::matmul2d`, `nn::self_attention_1h`, etc.).
- Keep **autograd semantics unchanged** (same graph structure, same gradient accumulation semantics).
- Introduce a **backend seam** with minimal refactor first (CPU backend behind interface), then extend to CUDA/HIP.
- Avoid “silent layout changes”: any packed layout must be explicit.

## Milestone 1 — Create a real backend seam (CPU backend)

**Goal:** refactor *implementation*, not *behavior*.

### 1.1 Expand the backend interface

Modify `src/backend/backend.h` from placeholder to a minimal kernel interface focused on the first hot path: GEMM.

Proposed additions:

- `enum class DeviceType { Cpu, Cuda, Hip };`
- `struct Device { DeviceType type; int device_id = 0; };`
- `class KernelBackend` with virtual methods for:
  - `matmul2d_fwd` (compute `C = A @ B`)
  - `matmul2d_bwd` (compute `dA += dC @ B^T` and `dB += A^T @ dC`)

Keep the interface “shape-first”: pass `m,k,n` and raw pointers.

### 1.2 Implement `CpuBackend`

Add new files:

- `src/backend/cpu_backend.h`
- `src/backend/cpu_backend.cpp`

Implementation should reuse the exact loops currently in `nn::matmul2d` to preserve determinism and numeric behavior.

### 1.3 Add backend selection (global, explicit)

Add new files:

- `src/backend/registry.h`
- `src/backend/registry.cpp`

Expose:

- `backend::KernelBackend& backend::get();`
- `void backend::set(std::unique_ptr<KernelBackend>);`
- `Device backend::current_device();` (initially always CPU)

Default is `CpuBackend`.

### 1.4 Refactor `nn::matmul2d` to call the backend

Modify `src/ops.cpp`:

- In `nn::matmul2d` forward: allocate output tensor, then call `backend::get().matmul2d_fwd(...)`.
- In `nn::matmul2d` backward lambda: call `backend::get().matmul2d_bwd(...)`.

Important: keep autograd graph structure identical (parents `{a,b}`, same accumulation into `pa.grad`/`pb.grad`). Only replace the inner loops.

### 1.5 Tests: dispatch preserves correctness

Add/extend tests (prefer small, deterministic shapes):

- Add `tests/test_backend_matmul.cpp`:
  - Compare `nn::matmul2d` results against a tiny hand-computed reference for a few shapes.
  - Check backward gradients for small matrices by finite differences (optional) or by comparing against the old implementation (if temporarily kept as `matmul2d_reference`).

Acceptance criteria:
- No behavior changes in existing tests.
- New tests pass on CPU-only build.

## Milestone 2 — CUDA proof-of-concept (matmul only)

**Goal:** get a CUDA backend compiling and running with a minimal surface area.

### 2.1 CMake options

Modify `CMakeLists.txt`:

- Add `option(BUILD_CUDA "Enable CUDA backend" OFF)`
- When enabled:
  - enable CUDA language
  - build CUDA backend translation unit(s)
  - link cuBLAS (and CUDA runtime)

### 2.2 CUDA backend skeleton

Add new files:

- `src/backend/cuda/cuda_backend.h`
- `src/backend/cuda/cuda_backend.cu`

Implement `matmul2d_fwd` via cuBLAS `sgemm`.

### 2.3 Memory strategy (POC vs real)

There are two options; pick explicitly:

**Option A (POC, simplest):**
- Inputs/outputs remain CPU `vector<float>`.
- `CudaBackend::matmul2d_fwd` copies A and B to device, runs cuBLAS, copies C back.
- Slow, but demonstrates layering and keeps changes minimal.

**Option B (real path, more work):**
- Introduce device-resident buffers and make `Tensor` device-aware.
- Avoid per-op host↔device copies.

For teaching-first incremental progress, start with Option A, but document clearly that it is not “performance mode”.

### 2.4 Tests

- Add `tests/test_backend_cuda_matmul.cpp` behind `BUILD_CUDA`.
- Compare CUDA matmul output against CPU within tolerances.

## Milestone 3 — HIP/ROCm proof-of-concept (matmul only)

Mirror Milestone 2:

- Add `option(BUILD_HIP "Enable HIP backend" OFF)`.
- Add `src/backend/hip/hip_backend.h/.cpp` (or `.cu` depending on HIP build style).
- Use rocBLAS for `sgemm`.
- Add a HIP matmul test behind `BUILD_HIP`.

## Milestone 4 — Make GPU actually fast (device tensors + expanded op coverage)

**Goal:** move from “matmul offload” to “GPU execution”.

### 4.1 Device-aware storage

Evolve `nn::Tensor` to support a device buffer, while keeping shape-first semantics.

One minimal approach:

- Add `nn::Device` + `nn::Storage` concept (CPU pointer vs device pointer).
- Make it explicit when tensors live on CPU vs GPU.
- Add copy ops: `to(Device)`.

### 4.2 Expand accelerated ops based on profiling

Likely next ops:

- `softmax_lastdim` (scores/probs are big)
- `layernorm_lastdim`
- attention (as a later variant; possibly fused)

Keep CPU fallback for everything.

### 4.3 Determinism policy

- Define expected tolerances per backend.
- For GPU, document non-determinism sources (parallel reductions) and how to enable deterministic modes (if desired), with any performance trade-offs.

## Suggested task ordering (practical)

1) Milestone 1 (backend seam, CPU only) — low risk, highest leverage.
2) CUDA matmul POC (BUILD_CUDA) — validates plumbing.
3) HIP matmul POC (BUILD_HIP) — validates portability boundary.
4) Decide between “POC offload” vs “real device tensors” path.

