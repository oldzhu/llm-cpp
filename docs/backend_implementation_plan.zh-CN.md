> [English](backend_implementation_plan.md)

# 后端实现计划（CPU → CUDA → HIP）

本文档把 `docs/backend_design_cpu_cuda_hip.md` 里的高层设计，落到可执行的增量步骤，并明确列出“要改哪些文件、加哪些接口”。

## 当前状态（今天代码是什么样）

- `nn::Tensor` 用 `shared_ptr<vector<float>>` 保存 data/grad（仅 CPU、row-major contiguous）。
- `src/ops.cpp` 里的 `nn::*` ops 直接用 CPU for-loop 做 forward/backward，并在函数内构建 autograd `Node`。
- `src/backend/backend.h` 存在，但只是占位（没有 dispatch、没有具体后端）。

这非常适合作为教学基线，但也意味着：**目前没有“可替换的缝”去接 CUDA/HIP kernels**。

## 指导原则

- **模型代码不动**（`src/model.cpp` 继续调用 `nn::matmul2d`、`nn::self_attention_1h` 等）。
- **autograd 语义不变**（计算图结构不变、梯度累加语义不变）。
- 先做“最小重构”引入后端边界（先把 CPU 实现放到后端里），再逐步扩展到 CUDA/HIP。
- 不允许“偷偷换 layout”：任何 packed layout 必须显式呈现。

## 里程碑 1 — 先做一个真实的后端缝（CPU 后端）

**目标：**只改实现方式，不改行为。

### 1.1 扩展后端接口

把 `src/backend/backend.h` 从占位扩展为最小 kernel 接口，先聚焦最热的 GEMM。

建议新增：

- `enum class DeviceType { Cpu, Cuda, Hip };`
- `struct Device { DeviceType type; int device_id = 0; };`
- `class KernelBackend`：至少提供以下虚函数：
  - `matmul2d_fwd`（计算 `C = A @ B`）
  - `matmul2d_bwd`（计算 `dA += dC @ B^T` 与 `dB += A^T @ dC`）

保持“shape-first”：传 `m,k,n` 和原始指针。

### 1.2 实现 `CpuBackend`

新增文件：

- `src/backend/cpu_backend.h`
- `src/backend/cpu_backend.cpp`

实现应复用当前 `nn::matmul2d` 里的循环结构，以尽可能保持确定性与数值行为一致。

### 1.3 增加后端选择（全局、显式）

新增文件：

- `src/backend/registry.h`
- `src/backend/registry.cpp`

对外提供：

- `backend::KernelBackend& backend::get();`
- `void backend::set(std::unique_ptr<KernelBackend>);`
- `Device backend::current_device();`（第一阶段先固定 CPU）

默认使用 `CpuBackend`。

### 1.4 重构 `nn::matmul2d` 通过后端执行

修改 `src/ops.cpp`：

- `nn::matmul2d` forward：分配输出张量后，调用 `backend::get().matmul2d_fwd(...)`。
- `nn::matmul2d` backward lambda：调用 `backend::get().matmul2d_bwd(...)`。

关键要求：计算图结构保持不变（parents 仍是 `{a,b}`；梯度仍然按 `+=` 写入 `pa.grad` / `pb.grad`）。只是把内层循环挪到后端。

### 1.5 测试：dispatch 不改变正确性

新增/扩展测试（优先小尺寸、可重复）：

- 新增 `tests/test_backend_matmul.cpp`：
  - 对几个 shape，用手算/小参考实现对比 `nn::matmul2d` forward。
  - backward 可做：
    - 有限差分（小尺寸，较慢但直观），或
    - 若临时保留一个 `matmul2d_reference`，则对比旧实现。

验收标准：
- 现有测试不变且全部通过。
- 新增测试在 CPU-only 构建下通过。

## 里程碑 2 — CUDA 可跑的最小 POC（只做 matmul）

**目标：**让 CUDA 后端能编译并跑通，先把“管道”打通。

### 2.1 CMake 开关

修改 `CMakeLists.txt`：

- 新增 `option(BUILD_CUDA "Enable CUDA backend" OFF)`
- 开启后：
  - 启用 CUDA language
  - 编译 CUDA 后端源文件
  - 链接 cuBLAS（以及 CUDA runtime）

### 2.2 CUDA 后端骨架

新增文件：

- `src/backend/cuda/cuda_backend.h`
- `src/backend/cuda/cuda_backend.cu`

用 cuBLAS `sgemm` 实现 `matmul2d_fwd`。

### 2.3 内存策略：POC vs 真正性能路径

这里必须显式选择，两条路线：

**方案 A（POC，最简单）：**
- 输入/输出仍然是 CPU 的 `vector<float>`。
- `CudaBackend::matmul2d_fwd` 内部：把 A/B 拷到 device，调用 cuBLAS，再把 C 拷回 host。
- 很慢，但能清晰展示分层与接口。

**方案 B（真正路径，工作量更大）：**
- 引入 device-resident buffer，让 `Tensor` 具备“在 GPU 上”的存储。
- 避免每个算子都 host↔device 拷贝。

教学优先的增量策略：先做方案 A 打通流程，并在文档里明确“这不是性能模式”。

### 2.4 测试

- 新增 `tests/test_backend_cuda_matmul.cpp`（用 `BUILD_CUDA` 包起来）。
- CUDA 输出与 CPU 对比（带 tolerance）。

## 里程碑 3 — HIP/ROCm 可跑的最小 POC（只做 matmul）

与里程碑 2 对齐：

- 新增 `option(BUILD_HIP "Enable HIP backend" OFF)`。
- 新增 `src/backend/hip/hip_backend.h/.cpp`（或 HIP 的编译方式对应的后缀）。
- 用 rocBLAS 实现 `sgemm`。
- 新增 HIP matmul 测试（`BUILD_HIP` 条件）。

## 里程碑 4 — 让 GPU 真正快起来（device tensors + 扩展算子覆盖）

**目标：**从“matmul offload”走向“GPU execution”。

### 4.1 device-aware 存储

演进 `nn::Tensor`，让它能持有 device buffer，同时保持 shape-first 语义。

一个最小方案：

- 引入 `nn::Device` + `nn::Storage` 概念（CPU 指针 vs device 指针）。
- 明确 tensor 在 CPU 还是 GPU 上。
- 提供拷贝算子：`to(Device)`。

### 4.2 基于 profiling 扩展加速算子

下一批高价值算子通常是：

- `softmax_lastdim`（scores/probs 很大）
- `layernorm_lastdim`
- attention（后续再做“变体/融合”，不强行改写基线）

并保持 CPU fallback。

### 4.3 确定性策略

- 定义各后端的误差容忍标准。
- 对 GPU：在文档中说明非确定性来源（并行 reduce），以及如何开启 deterministic mode（如果需要）与性能代价。

## 推荐的实际执行顺序

1) 里程碑 1（后端缝，CPU-only）——风险最低、收益最大。
2) CUDA matmul POC（BUILD_CUDA）——验证接口与构建链路。
3) HIP matmul POC（BUILD_HIP）——验证可移植边界。
4) 决策是否进入“device tensors 真正路径”。
