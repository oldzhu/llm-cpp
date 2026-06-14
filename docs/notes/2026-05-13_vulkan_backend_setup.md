# Vulkan Compute Backend Setup (POC)

> [简体中文](2026-05-13_vulkan_backend_setup.zh-CN.md)

## Quick start

### 1. Install Vulkan SDK
Download LunarG Vulkan SDK from https://vulkan.lunarg.com/
Or: `winget install KhronosGroup.VulkanSDK`

### 2. Build
```powershell
cmake -S . -B build -DBUILD_VULKAN=ON
cmake --build build --config Debug
```

### 3. Verify
```powershell
ctest --test-dir build -C Debug
```
Test `test_vulkan_matches_cpu_matmul` compares Vulkan matmul against CPU reference.

## Architecture

```
nn::matmul2d → backend::get().matmul2d_fwd()
    → VulkanBackend::matmul2d_fwd()
        → upload A,B to device buffers
        → dispatch compute shader (16×16 workgroups)
        → readback C to host
```

## Files
- `src/backend/vulkan/vulkan_backend.h` — Backend class
- `src/backend/vulkan/vulkan_backend.cpp` — Vulkan compute pipeline implementation
- `shaders/matmul_fwd.comp` — GLSL matmul shader (compiled to SPIR-V at build)

## Limitations (POC mode)
- Forward matmul only (backward → CPU fallback)
- Host↔device copies per call (not device-resident)
- Requires Vulkan 1.1+ GPU

# Vulkan 计算后端配置

1. 安装 Vulkan SDK：https://vulkan.lunarg.com/
2. 构建：`cmake -S . -B build -DBUILD_VULKAN=ON`
3. 测试：`ctest --test-dir build -C Debug`
