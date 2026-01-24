> [English](backend_design_cpu_cuda_hip.md)

# 简短设计：在不复制模型代码的前提下支持 CPU/CUDA/HIP 后端

这是一份“短设计笔记”，目标是让本仓库可以在不同目标上运行（CPU、NVIDIA GPU、AMD GPU），同时保持教学优先与正确性优先。

## 目标

- 训练/推理/模型主体代码只写**一份**。
- 语义显式且可验证（shapes、layouts、确定性期望）。
- 允许通过替换实现来获得性能：CPU 参考实现 vs GPU 后端。
- 避免“魔法依赖”；边界要小且明确，并且（可选）用厂商库来实现 GEMM。

## 非目标（第一阶段暂不做）

- 不在第一天就做 SOTA 的 fused attention。
- 第一阶段不引入完整的混合精度训练管线（FP16/BF16），除非明确接受数值/确定性上的复杂变化。
- 不要求 CPU 与 GPU bit-identical；目标是“数值接近”，并用明确的 tolerance 检查。

## 分层边界：model → ops → backend

### 原则

- **Model 层**只表达数学与 shape。
- **Ops 层**定义稳定的算子语义（算什么）。
- **Backend 层**提供具体实现（怎么算），按目标设备区分。

### 建议的最小 API

引入后端选择，并通过它路由算子实现。

- `enum class DeviceType { Cpu, Cuda, Hip };`
- `struct Device { DeviceType type; int device_id; };`

至少需要路由这些算子：
- `matmul2d`
- `softmax_lastdim`
- `layernorm`（如果它是独立算子）
- `self_attention_*`（或先把 attention 作为组合实现，后续再做融合变体）

CPU 路径保持为参考正确实现。

## 数据与 layout 契约

保持 layout 假设显式（shape-first）：

- `Tensor` 为 row-major contiguous。
- GPU 后端（第一阶段）同样只支持 contiguous buffer。
- 如果未来引入 packed/blocked layout，必须显式表示（例如独立 tensor 类型或 layout tag），不能“偷偷换布局”。

## 各目标后端策略

### CPU 后端（参考实现）

- 保留现有实现作为 correctness oracle。
- 后续可选：加入 SIMD/blocking，或增加 BLAS-backed matmul，但仍需在同一算子语义之下。

### NVIDIA 后端（CUDA）

里程碑顺序：

1) **先做 GEMM**：把 `matmul2d` 映射到 cuBLAS。
2) 增加最小化的 device memory 管理（`cudaMalloc`、`cudaMemcpy`、streams）。
3) 若 elementwise 成为热点，再增加简单自定义 kernel（add/mul/gelu）。

备注：
- cuBLAS 通常能用很小的代码量带来很大的性能提升。
- fused attention 可以作为后续“变体”，而不是改写基线。

### AMD 后端（HIP/ROCm）

里程碑与 CUDA 对齐：

1) 把 `matmul2d` 映射到 rocBLAS。
2) 增加 HIP 内存管理（`hipMalloc`、`hipMemcpy`）。
3) 需要时再加简单 HIP elementwise kernels。

备注：
- 算子语义必须与 CPU 一致。
- 调优点会不同（wavefront=64、LDS 行为等），但这些差异应封装在后端内部。

## 确定性与测试策略

- CPU 后端是“真值参考”。
- 增加后端测试：用 tolerance 对比 CPU vs GPU 输出：
  - forward：max abs/relative error。
  - backward：小尺寸的 gradient checks（带 tolerance）。
- 在文档中明确 reduce 顺序差异（sum 不满足结合律）。

## 构建系统（CMake）草案

增加可选开关：

- `BUILD_CUDA`（默认 OFF）
- `BUILD_HIP`（默认 OFF）

开启后：
- 编译后端源文件（例如 `src/backends/cuda/*.cu` 或 `src/backends/hip/*`）
- 链接厂商库（cuBLAS / rocBLAS）

默认构建路径保持“仅 CPU、无额外依赖”。

## 建议的文件布局（增量式）

- `src/ops.*`（语义 + dispatch stub）
- `src/backends/cpu/*`（现有代码，尽量小改动）
- `src/backends/cuda/*`（新增）
- `src/backends/hip/*`（新增）
- `docs/backend_design_cpu_cuda_hip.md`（本文档）

## 里程碑

1) **只做边界重构**：引入 `Device` + dispatch，但不改变结果。
2) **CUDA matmul**：在同一 API 下接入 cuBLAS-backed `matmul2d` + 测试。
3) **HIP matmul**：接入 rocBLAS-backed `matmul2d` + 测试。
4) 根据 profiler 证据扩展覆盖（attention、layernorm、elementwise 融合）。
