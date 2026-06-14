> [English](2026-05-13_vulkan_backend_setup.md)

# Vulkan 计算后端配置（POC）

## 快速开始

1. 安装 Vulkan SDK：https://vulkan.lunarg.com/ 或 `winget install KhronosGroup.VulkanSDK`
2. 构建：`cmake -S . -B build -DBUILD_VULKAN=ON; cmake --build build --config Debug`
3. 验证：`ctest --test-dir build -C Debug`

## 架构
`nn::matmul2d` → `backend::get().matmul2d_fwd()` → Vulkan compute dispatch

## 文件
- `src/backend/vulkan/vulkan_backend.h/.cpp` — Vulkan 后端实现
- `shaders/matmul_fwd.comp` — GLSL 着色器源码

## 限制
- 仅前向 matmul（反向 → CPU 回退）
- 每次调用主机↔设备拷贝（非设备常驻）
