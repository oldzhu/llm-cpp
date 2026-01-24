> [English](hardware_stack_and_portability.md)

# 应用层代码 vs 硬件层（CPU / NVIDIA / AMD）

本文档把一次规划讨论中的两段“问答式解释”原样整理出来，便于后续参考与学习。

## 问：应用层编码与硬件层之间是什么关系？

从高层看，“应用层代码”和“硬件层执行”之间通过一整套抽象层相互影响。你通常可以保持**模型语义**不变（同样的数学、同样的 shape），但如果想**充分利用**不同目标（CPU vs NVIDIA GPU vs AMD GPU），一般需要做到二选一（或两者结合）：

- 提供不同的**后端实现**（不同 kernel / 不同库调用），或
- 编码时选择某种**可移植抽象**（同时接受一定约束）。

下面先给出一个实用的“全栈视角”，再说明针对 CPU/NVIDIA/AMD 时会有什么变化，以及这对项目设计意味着什么。

### 1) 全栈视角（从你的代码到电子学）

可以按层级（上 → 下）理解：

1) **模型 / 算法层**
- 定义数学：attention、MLP、layernorm、optimizer、sampling。
- 关键关注：张量 shape、数值稳定性、确定性（reproducibility）要求。

2) **算子语义层（类似“框架层”）**
- 稳定 API：`matmul`、`softmax`、`layernorm`、`attention`、`cross_entropy`。
- 目标是与后端无关：描述“算的是什么”，而不是“怎么算”。

3) **kernel / 后端层**
- CPU 后端：for-loop + SIMD + 线程，或调用 oneDNN/MKL/OpenBLAS。
- GPU 后端：launch kernels，和/或调用厂商库（NVIDIA 的 cuBLAS/cuDNN；AMD 的 rocBLAS/MIOpen）。
- 这里决定了硬件利用率：tiling、shared memory / LDS、warp/wavefront 行为、算子融合等。

4) **编译器 / 代码生成层**
- CPU：MSVC/Clang/GCC + 自动向量化 + OpenMP/TBB。
- NVIDIA：NVCC（CUDA）或 Clang CUDA；编译到 PTX/SASS。
- AMD：HIP-Clang；编译到 GCN/ROCm code objects。

5) **运行时 + 驱动层**
- CPU：OS scheduler、线程运行时、NUMA 策略。
- NVIDIA：CUDA runtime + NVIDIA driver。
- AMD：ROCm runtime + AMDGPU driver。

6) **硬件微架构**
- CPU：cache 层级、SIMD 宽度（AVX2/AVX-512）、核心数、内存带宽。
- NVIDIA GPU：SM、warp（32 threads）、tensor cores、HBM 带宽。
- AMD GPU：CU、wavefront（通常 64）、矩阵核心（MFMA）、HBM 带宽。

**关键点：** 上层决定“算什么”；下层决定“算得多快、并行度如何”。但“怎么快”也会反过来约束上层的选择（precision、layout、fusion 等）。

### 2) 为 CPU vs GPU 需要写不同的应用层代码吗？

常见有三种路线：

**A) 在同一套算子 API 之下提供多个后端（推荐给本项目）**
- 模型代码写一份（调用 `ops::matmul`、`ops::attention` 等）。
- 先实现 CPU 后端（参考正确、易读）。
- 再增加 GPU 后端，实现相同算子语义，但内部调用 CUDA/HIP kernel 或库。
- 结果：训练/推理主流程基本不变，只需要切换后端选择。

这也是严肃系统最常见的结构：既保正确性又保可教学性。

**B) 直接写目标特定 API（更“硬核”，但容易分叉）**
- NVIDIA：到处写 CUDA kernel、调 cuBLAS、用 CUDA graphs 等。
- AMD：到处写 HIP kernel、调 rocBLAS 等。
- 优点是对单一目标更快落地性能；缺点是代码库容易分叉、可移植性成本高。

**C) 使用可移植的 GPU 编程模型**
- 选择：SYCL、OpenCL、（有时）Kokkos/RAJA，或“HIP 作为可移植层”（HIP 也可以跑 NVIDIA，但存在差异/限制）。
- 仍然需要做目标调优，但能减少代码分叉。

**结论：** 不建议写两套“模型实现”。更好的做法是：同一套模型语义 + 多个后端。

### 3) 它们如何相互影响（具体例子）

**例子 1：`matmul` / GEMM 是 Transformer 的大头**
- 模型层：`[M,K] x [K,N] -> [M,N]`。
- 硬件现实：
  - CPU：高性能来自 cache blocking + SIMD + 多线程。
  - NVIDIA/AMD GPU：高性能来自 shared memory/LDS tiling + tensor/matrix cores。
- 对应用层的影响：
  - 数学不变，但需要关注 shape 是否“友好”：例如 `K`/`N` 是否是 8/16 的倍数（有利于 tensor cores），batch 是否足够大等。
  - 权重可能需要以某些后端偏好的 packed layout 存储。

**例子 2：Attention 与内存带宽**
- 朴素 attention 会显式 materialize `[B,T,T]` scores/probs，体积巨大。
- GPU 计算很快，但 attention 常常变成**带宽瓶颈**或**kernel-launch 开销瓶颈**。
- 影响：
  - 后端会推动你使用 *fused attention*（计算 `softmax(QK^T)V` 时尽量不写出巨大中间量）。
  - 这会改变代码里“中间张量是否存在”的形式（但语义不变）。
  - 对小规模场景，CPU 可能仍然用更直观的实现，因为融合的复杂度不划算。

**例子 3：精度与确定性**
- GPU 通常在 FP16/BF16（以及 NVIDIA 的 TF32）下最快。
- CPU 常见是 FP32（也可能用 FP16，取决于 ISA）。
- 影响：
  - 若换 dtype，可能需要 loss scaling、不同 epsilon、更谨慎的数值处理。
  - 确定性：并行 reduce（`sum`）不满足结合律；线程/blocks 的累加顺序变化会导致数值出现微小差异。

### 4) 除了“应用层”和“硬件层”，还有哪些关键层？

至少应显式纳入设计考虑的中间层包括：

- **算子 API 层**：让模型代码稳定。
- **后端/kernel 层**：性能主要在这里。
- **编译器/代码生成**：向量化、fast-math、unroll 等可能决定性能上限。
- **运行时/驱动**：调度、内存分配、kernel launch、stream 执行。
- **高性能库**：cuBLAS/rocBLAS、cuDNN/MIOpen、oneDNN/MKL 等通常是“最容易拿到大头性能”的方式。

### 5) CPU vs NVIDIA vs AMD：如何才算“利用好硬件”？

**CPU（x86/ARM）**
- 关键：SIMD + 多线程 + cache blocking。
- 常见动作：
  - 对外层循环并行（batch/head/sequence block）。
  - 保证连续内存访问，避免 strided load。
  - 优先调用成熟 BLAS 做 GEMM，再考虑自研。

**NVIDIA GPU**
- 关键：CUDA + cuBLAS/cuDNN + 融合 kernel；用 tensor cores。
- 常见动作：
  - 数据尽量常驻显存，减少 host↔device 传输。
  - 融合算子（bias+gelu、layernorm、attention）。
  - 在可接受的数值前提下使用 FP16/BF16/TF32。

**AMD GPU**
- 关键：HIP + rocBLAS/MIOpen；面向 wavefront/occupancy 调优；用 MFMA。
- 常见动作：
  - 与 NVIDIA 类似：数据常驻、使用库 GEMM、必要时做融合。
  - 调优点不同：wavefront=64、LDS 行为、甜点维度不同。

### 6) 如何融入项目的计划/设计

一个教学优先、同时为 GPU 扩展留好路的设计是：

- 定义一个**后端无关**的 Tensor + ops 接口（概念上已有雏形）。
- 保持一个**CPU 参考后端**作为 correctness oracle。
- 增加 GPU 后端时：先用厂商 GEMM（cuBLAS/rocBLAS）接入 `matmul`，再在需要时逐步替换热点为自研融合 kernel（attention、layernorm）。
- 维护一份小而明确的 **shape/layout 契约**，保证后端一致。

## 追问：把同样的观点再更具体地说一遍

是的——CPU vs GPU 的差异会“向上渗透”，影响你如何组织（有时也会轻微影响）应用层实现；但核心目标是：保持**模型数学**不变，通过替换**后端**来适配不同硬件。

### 一句话关系

- 硬件的本质是并行与内存层级；应用层选择 shape/layout/precision 以及是否 materialize 中间量，这些选择决定实现对某个目标是“友好”还是“反友好”。

### 应用层是否要写两份？

通常希望：**训练/推理主流程写一份**，但**后端实现写多份**。

可行设计：
- **共享模型代码：**只调用 `matmul/softmax/layernorm/attention/cross_entropy` 等高层算子。
- **不同后端实现：**
  - CPU：C++ loops + SIMD/线程（或 BLAS）。
  - NVIDIA：CUDA kernels 和/或 cuBLAS/cuDNN。
  - AMD：HIP kernels 和/或 rocBLAS/MIOpen。

所以：一般不需要重写“整个应用”，但为了充分利用硬件，需要为不同目标提供不同的 kernel/算子实现。

### 其他层（再次强调）

- 算子 API、后端/kernel、编译器/代码生成、运行时/驱动、以及高性能库是连接应用与硅片的关键中间层。

### CPU vs NVIDIA vs AMD（再次强调）

- CPU：SIMD + 多线程 + cache 局部性。
- NVIDIA：数据常驻显存、优先 cuBLAS、能用 tensor cores 就用、做融合。
- AMD：数据常驻显存、优先 rocBLAS、面向 wavefront/LDS 调优、用 MFMA。
