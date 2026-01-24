> [简体中文](backend_design_cpu_cuda_hip.zh-CN.md)

# Short design: CPU/CUDA/HIP backends without duplicating model code

This is a short design note for extending this repo to run on different targets (CPU, NVIDIA GPU, AMD GPU) while keeping the project teaching-first and correctness-first.

## Goals

- Keep **one** set of model/training/inference code.
- Keep semantics explicit and verifiable (shapes, layouts, determinism expectations).
- Allow swapping implementations for performance: CPU reference vs GPU backends.
- Avoid “magic dependencies”; prefer small, explicit boundaries and (optionally) vendor libraries for GEMM.

## Non-goals (initially)

- No attempt at state-of-the-art fused attention on day 1.
- No mixed-precision training pipeline in the first milestone (FP16/BF16) unless we explicitly accept non-trivial numeric/determinism changes.
- No requirement that CPU and GPU produce bit-identical floats; we target *close* numeric agreement with well-defined tolerances.

## Layer boundary: model → ops → backend

### Principle

- **Model layer** should only express math and shapes.
- **Ops layer** defines stable operator semantics (what to compute).
- **Backend layer** provides implementations (how to compute), per device target.

### Proposed minimal API

Introduce a backend selector and route ops through it.

- `enum class DeviceType { Cpu, Cuda, Hip };`
- `struct Device { DeviceType type; int device_id; };`

At minimum, route these ops:
- `matmul2d`
- `softmax_lastdim`
- `layernorm` (if present as a distinct op)
- `self_attention_*` (or keep attention as composition initially)

The CPU path stays the reference implementation.

## Data + layout contracts

Keep layout assumptions explicit (shape-first):

- `Tensor` is row-major contiguous.
- For GPU backends (initially), also require contiguous buffers.
- If we introduce packed/blocked layouts later, they must be represented explicitly (e.g., separate tensor type or a layout tag), not silently.

## Backend strategy by target

### CPU backend (reference)

- Keep existing implementations as the correctness oracle.
- Optional later: add SIMD/blocking, or a BLAS-backed matmul, behind the same op.

### NVIDIA backend (CUDA)

Milestone order:

1) **GEMM first**: map `matmul2d` to cuBLAS.
2) Add minimal tensor/device memory management (`cudaMalloc`, `cudaMemcpy`, streams).
3) Consider small custom kernels for elementwise ops (add/mul/gelu) if they dominate.

Notes:
- cuBLAS gives large wins quickly, with low code volume.
- Fused attention can be a later variant, not a rewrite.

### AMD backend (HIP/ROCm)

Milestone order mirrors CUDA:

1) Map `matmul2d` to rocBLAS.
2) Add HIP memory management (`hipMalloc`, `hipMemcpy`).
3) Add simple HIP kernels for elementwise ops if needed.

Notes:
- Keep the operator semantics identical to CPU.
- Expect tuning differences (wavefront=64, LDS behavior), but hide them inside the backend.

## Determinism + testing policy

- CPU backend remains the source of truth.
- Add backend tests that compare CPU vs GPU outputs with tolerances:
  - Forward pass: max abs/relative error checks.
  - Backward pass: gradient checks (small sizes) with tolerances.
- Make reduction order differences explicit in docs (sum is not associative).

## Build system (CMake) sketch

Add optional targets:

- `BUILD_CUDA` (OFF by default)
- `BUILD_HIP` (OFF by default)

When enabled:
- build backend translation units (e.g., `src/backends/cuda/*.cu` or `src/backends/hip/*.cpp` with HIP)
- link vendor libs (cuBLAS / rocBLAS)

Keep the default build path dependency-free (CPU only).

## Suggested file layout (incremental)

- `src/ops.*` (semantics + dispatch stubs)
- `src/backends/cpu/*` (existing or reorganized minimally)
- `src/backends/cuda/*` (new)
- `src/backends/hip/*` (new)
- `docs/backend_design_cpu_cuda_hip.md` (this doc)

## Milestones

1) **Refactor boundary only**: introduce `Device` + dispatch without changing results.
2) **CUDA matmul**: cuBLAS-backed `matmul2d` behind the same API + tests.
3) **HIP matmul**: rocBLAS-backed `matmul2d` + tests.
4) Expand op coverage based on profiler evidence (attention, layernorm, elementwise fusion).
