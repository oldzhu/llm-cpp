> [简体中文](hardware_stack_and_portability.zh-CN.md)

# Application-layer code vs hardware (CPU / NVIDIA / AMD)

This note preserves two Q&A-style explanations from a planning discussion, for later reference.

## Q: What is the relationship between application-layer coding and the hardware layer?

At a high level, “application layer code” and “hardware layer execution” influence each other through a stack of abstractions. You can often keep *model semantics* identical (same math, same shapes), but to *fully utilize* different targets (CPU vs NVIDIA GPU vs AMD GPU) you usually need either:

- different **backend implementations** (different kernels / libraries), or
- to write against a **portable abstraction** (and accept some constraints).

Below is a practical “full-stack” view, then what changes when you target CPU/NVIDIA/AMD, and what it implies for our project design.

### 1) The full stack (from your code to electrons)

Think in layers (top → bottom):

1) **Model / algorithm layer**
- Defines the math: attention, MLP, layernorm, optimizer, sampling.
- What matters: tensor shapes, numerical stability, determinism requirements.

2) **Operator semantics layer (framework-ish layer)**
- A stable API like `matmul`, `softmax`, `layernorm`, `attention`, `cross_entropy`.
- Should be backend-agnostic: “what it computes”, not “how”.

3) **Kernel / backend layer**
- CPU backend: loops + SIMD + threads, or calls to oneDNN/MKL/OpenBLAS.
- GPU backend: launches kernels and/or calls vendor libraries (cuBLAS/cuDNN on NVIDIA; rocBLAS/MIOpen on AMD).
- This is where hardware utilization is decided: tiling, shared memory, warp/wavefront behavior, fusion, etc.

4) **Compiler / codegen layer**
- CPU: MSVC/Clang/GCC + auto-vectorization + OpenMP/TBB.
- NVIDIA: NVCC (CUDA), or Clang CUDA; compiles to PTX/SASS.
- AMD: HIP-Clang; compiles to GCN/ROCm code objects.

5) **Runtime + driver layer**
- CPU: OS scheduler, thread runtime, NUMA policies.
- NVIDIA: CUDA runtime + NVIDIA driver.
- AMD: ROCm runtime + AMDGPU driver.

6) **Hardware microarchitecture**
- CPU: cache hierarchy, SIMD width (AVX2/AVX-512), core count, memory bandwidth.
- NVIDIA GPU: SMs, warps (32 threads), tensor cores, HBM bandwidth.
- AMD GPU: CUs, wavefronts (usually 64), matrix cores (MFMA), HBM bandwidth.

**Key idea:** The upper layers choose *what* to compute; the lower layers decide *how fast* and *how parallel* it runs. But the “how” can constrain the “what” (precision choices, layout choices, fusion choices).

### 2) Do you need different application-layer coding for CPU vs GPU?

You have three common approaches:

**A) Separate backends behind one operator API (recommended for this project)**
- You write model code once (calls `ops::matmul`, `ops::attention`…).
- You implement CPU backend first (reference-correct, easy to read).
- Then add GPU backends that implement the same ops but call CUDA/HIP kernels or libraries.
- Result: same training loop / inference loop; only backend selection changes.

This is how most serious systems are structured because it preserves correctness and teaching clarity.

**B) Write directly in target-specific APIs**
- NVIDIA: write CUDA kernels everywhere; call cuBLAS, use CUDA graphs, etc.
- AMD: write HIP kernels everywhere; call rocBLAS, etc.
- Fast to get performance on one target, but you “fork” the codebase and pay portability costs.

**C) Use a portable GPU programming model**
- Options: SYCL, OpenCL, (sometimes) Kokkos/RAJA, or “HIP as portability layer” (HIP can target NVIDIA too, with caveats).
- You still typically need target-tuning, but you reduce forked code.

**Bottom line:** You do *not* want two separate model implementations. You want one model implementation + multiple backends.

### 3) How they impact each other (concrete examples)

**Example 1: `matmul` / GEMM dominates Transformers**
- Model layer says: multiply `[M,K] x [K,N] -> [M,N]`.
- Hardware layer reality:
  - On CPU, best performance comes from cache blocking + SIMD + threading.
  - On NVIDIA/AMD GPUs, best performance comes from tiling into shared memory/LDS and using tensor/matrix cores.
- Impact on application code:
  - You don’t change the math, but you *do* care about shapes being friendly: e.g. `K`/`N` multiples of 8/16 for tensor cores, larger batch sizes, etc.
  - You may choose to store weights in layouts preferred by the backend (packed formats).

**Example 2: Attention and memory bandwidth**
- Naive attention materializes `[B,T,T]` scores/probs, which is huge.
- GPUs are extremely fast at compute, but attention often becomes **memory-bandwidth bound** and **kernel-launch overhead bound**.
- Impact:
  - Backends push you toward *fused attention kernels* (compute `softmax(QK^T)V` without writing big intermediates).
  - That can change what intermediate tensors “exist” in code (but semantics stay the same).
  - CPU might still use a simpler path because kernel fusion complexity isn’t worth it at small sizes.

**Example 3: Precision and determinism**
- GPUs often run fastest in FP16/BF16 (and sometimes TF32 on NVIDIA).
- CPUs typically run FP32/FP16 depending on ISA support.
- Impact:
  - If you change dtype, you may need loss scaling, different epsilon choices, and more careful numerics.
  - Determinism: parallel reductions (`sum`) can be non-associative; CPU threads/GPU blocks can change accumulation order → tiny numeric diffs.

### 4) Additional layers you should be aware of (beyond “app” and “hardware”)

Yes—at minimum you should explicitly account for these middle layers in design:

- **Operator API layer**: keeps model code stable.
- **Backend/kernel layer**: where performance lives.
- **Compiler/codegen**: can make or break performance (vectorization, fast-math, kernel unrolling).
- **Runtime/driver**: governs scheduling, memory allocation, kernel launch, stream execution.
- **Libraries**: cuBLAS/rocBLAS, cuDNN/MIOpen, oneDNN/MKL—often the easiest way to get most of peak performance.

### 5) CPU vs NVIDIA vs AMD: what “good utilization” tends to require

**CPU (x86/ARM)**
- Use: SIMD + threading + cache blocking.
- Typical steps:
  - Multi-thread outer loops (batch, heads, sequence blocks).
  - Ensure contiguous memory access; avoid strided loads.
  - Prefer calling a tuned BLAS for GEMM before writing your own.

**NVIDIA GPU**
- Use: CUDA + cuBLAS/cuDNN + fused kernels; exploit tensor cores.
- Typical steps:
  - Keep data on device; minimize host↔device transfers.
  - Fuse ops (bias+gelu, layernorm, attention).
  - Use FP16/BF16/TF32 paths where acceptable.

**AMD GPU**
- Use: HIP + rocBLAS/MIOpen; tune for wavefront/occupancy; use MFMA matrix ops.
- Typical steps:
  - Similar to NVIDIA at a high level (keep data resident, fuse, use library GEMM).
  - Kernel tuning differs (wavefront 64, LDS behavior, different sweet spots).

### 6) How this becomes part of our project plan/design

A clean teaching-first design that still scales to GPU is:

- Define a small **backend-agnostic tensor + ops interface** (already mostly present conceptually).
- Keep a **CPU reference backend** as the correctness oracle.
- Add a **GPU backend** that initially calls vendor GEMM (cuBLAS/rocBLAS) for `matmul`, and progressively replaces hotspots with custom fused kernels (attention, layernorm) only when needed.
- Maintain a small set of **shape/layout contracts** so both backends agree.

## Follow-up: summary of the same idea, more concretely

Yes—CPU vs GPU differences absolutely “reach upward” into how you should structure (and sometimes slightly change) your application-layer implementation, but the goal is: keep the *model math* the same and swap the *backend*.

### Relationship in one sentence

- Hardware is fundamentally about parallelism + memory hierarchy; your application layer chooses shapes/layouts/precision and whether to materialize intermediates, and those choices can make an implementation either “friendly” or “hostile” to a target.

### Do you need different application-layer coding?

You generally want **one model/training loop**, but **multiple backends**.

A practical design is:
- **Model code (shared):** calls high-level ops like `matmul`, `softmax`, `layernorm`, `attention`, `cross_entropy`.
- **Backend implementations (different):**
  - CPU backend: plain C++ loops + SIMD/threading (or call BLAS).
  - NVIDIA backend: CUDA kernels and/or cuBLAS/cuDNN.
  - AMD backend: HIP kernels and/or rocBLAS/MIOpen.

So: you usually *don’t* rewrite “the whole application,” but you *do* write different low-level kernels or different “op implementations” per target to fully utilize it.

### Extra layers (again)

- Operator API layer, backend/kernel layer, compiler/codegen, runtime/driver, and tuned libraries are the key “middle layers” between your code and the silicon.

### CPU vs NVIDIA vs AMD (again)

- CPU: SIMD + threads + cache locality.
- NVIDIA: keep data on device, rely on cuBLAS, use tensor cores where possible, fuse kernels.
- AMD: keep data on device, rely on rocBLAS, tune for wavefront 64/LDS, use MFMA paths.

