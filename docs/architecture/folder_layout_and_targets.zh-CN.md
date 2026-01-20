> [English](folder_layout_and_targets.md)

# 目录结构 + CMake targets 方案（提案）

本仓库刻意保持很小，但我们希望结构能扩展到：
- 多种算法变体（MHA、KV-cache、RoPE、GQA 等）
- 多种后端（naive CPU、SIMD、CUDA/HIP/Vulkan…）
- 每个特性都有教学文档与测试

指导原则：**基线必须保持可读**。新特性以“增量模块”的方式加入。


## 建议的目录结构

```
src/
	core/                 # “baseline” 模型与算子（CPU reference）
		tensor.*
		ops.*
		model.*
		optim.*
		checkpoint.*
		data.*
		util.*

	backend/              # kernel/backend 边界（初期仍是 CPU-only）
		backend.h           # interface + backend 选择（未来）
		cpu/                # CPU reference backend 实现（未来）

	variants/             # side-by-side 算法变体
		mha/                # multi-head attention 变体模块
			mha_attention.h
			mha_attention.cpp

apps/
	train_gpt_main.cpp    # CLI 可执行程序（未来可选拆分）

tests/
	test_main.cpp         # 测试入口；可包含 variant tests

docs/
	README.md
	architecture/
		folder_layout_and_targets.md
	variants/
		mha/
			README.md
			learn_by_hand.md
```

说明：
- 本文描述的是“目标结构”。可以逐步迁移。
- variants 应该是 **增量式** 的：除非必要，不重写 baseline。


## 建议的 CMake targets

保持 targets 小而可组合：

- **`llm_core`**（static library）
	- 基线数学 + autograd + model + 训练辅助
	- 不包含 CLI 参数解析

- **`llm_backend`**（初期是 INTERFACE library）
	- 提供 backend 抽象的 include path
	- 未来：`llm_backend_cpu`、`llm_backend_cuda` 等

- **`llm_variant_mha`**（static library）
	- 多头注意力变体（side-by-side）
	- 依赖 `llm_core`

- **`train_gpt`**（executable）
	- CLI 应用，链接 `llm_core`（以及可选 variants）

- **`test_build_llm`**（executable）
	- 测试运行器，链接 `llm_core` 与选定 variants

### 为什么这样划分？

- **更快迭代**：variants 可独立编译。
- **教学清晰**：可以在文件/模块层面对比 baseline 与 variant。
- **后端演进**：将来可通过 backend interface 把热点算子路由出去，而不用重写模型代码。


## 添加新 variant 的模板

以 KV-cache 为例：

1) 代码：`src/variants/kv_cache/...`
2) 文档（至少 2 份）：`docs/variants/kv_cache/README.md` + `learn_by_hand.md`
3) 测试：在 `tests/test_main.cpp` 中添加一个聚焦测试
4) CMake：新增 `llm_variant_kv_cache` library，并在 tests 中链接

这样每个特性都会以“**代码 + 文档 + 测试**”的方式被发现与学习。
