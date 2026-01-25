> [English](README.md)

# build-llm-using-cpp

一个从零开始（CPU、FP32）的 GPT 风格 Transformer 训练项目（C++）。

目标：
- 以训练为中心（前向 + 反向 + 优化器）
- 保留一个易于推理/调试的 CPU FP32 基线
- 尽量少的依赖（仅标准库）

## 构建（Windows）

前置：
- CMake（建议 >= 3.20）
- MSVC Build Tools（建议 Visual Studio 2022）

PowerShell：

```powershell
cmake -S . -B build
cmake --build build --config Release -j
```

## 构建（Linux）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

说明：
- 本仓库目标是在 Windows 与 Linux 上使用同一份源码构建。
- 在 Windows 上交叉编译 Linux（二进制），或在 Linux 上交叉编译 Windows（二进制），可以通过 toolchain 实现，但不属于基线范围。

## 运行（训练）

默认会在一个文本文件上训练一个很小的“字节级”语言模型：

```powershell
.\\build\\Release\\train_gpt.exe --data .\\data\\the-verdict.txt --steps 200 --batch 4 --seq 64 --dmodel 64 --layers 1
```

说明：
- 分词已可插拔：可用 --tokenizer byte（默认）或 --tokenizer bpe（需词表/合并文件）。嵌入层和模型配置自动用 tokenizer.vocab_size()。
- 本项目为极小基线模型，训练步数/模型越大效果越好。

## 生成（采样）

你可以先训练，然后在同一次运行里从 prompt 生成：

```powershell
.\\build\\Debug\\train_gpt.exe --data .\\data\\the-verdict.txt --steps 400 --batch 4 --seq 64 --dmodel 64 --layers 1 --lr 0.0003 --prompt "Hello, I'm a tiny GPT. " --gen 200 --temp 1.0 --topk 40
```

如果希望在训练早期输出更“可读”，可用 ASCII-only 和/或字节转义：

```powershell
.\\build\\Debug\\train_gpt.exe --load .\\data\\ckpt_tiny --steps 0 --prompt "Hello: " --gen 200 --temp 0.8 --topk 40 --ascii-only 1 --escape-bytes 1
```

## Sanity 检查（数据集的下一字节）

这些模式回答一个简单问题：
“给定数据集中的上下文，模型的 top-1 预测是否等于数据集的**真实下一字节**？”

重要说明：
- 如果只在 `the-verdict.txt` 上训练了大约 ~60 step，结果出现对/错混合是正常的。
- 如果希望这个检查更稳定地正确，请增加训练步数（例如 `1000–5000+`）和/或增大模型规模。

随机抽样 N 次：

```powershell
.\\build\\Debug\\train_gpt.exe --data .\\data\\the-verdict.txt --steps 0 --load .\\data\\ckpt_tiny --sanity-next-from-data 5 --sanity-ctx 64 --sanity-top 10 --escape-bytes 0
```

指定一个固定 offset（精确定位某个片段）：

```powershell
& .\\build\\Debug\\train_gpt.exe --% --data .\\data\\the-verdict.txt --steps 0 --load .\\data\\ckpt_tiny --sanity-offset 0 --sanity-ctx 64 --sanity-preview 120 --sanity-top 10 --escape-bytes 0
```

提示：
- 在 PowerShell 中，如果通过 `& <exe>` 调用并带 `--flag`，建议加 `--%` 来避免 PowerShell 解释 `--...`。

## Checkpoint（保存/加载）

保存 checkpoint（写出 `PREFIX.json` + `PREFIX.bin`）：

```powershell
.\\build\\Debug\\train_gpt.exe --data .\\data\\the-verdict.txt --steps 1000 --batch 4 --seq 64 --dmodel 64 --layers 1 --save .\\data\\ckpt_tiny
```

从 checkpoint 继续训练：

```powershell
.\\build\\Debug\\train_gpt.exe --load .\\data\\ckpt_tiny --data .\\data\\the-verdict.txt --steps 500 --save .\\data\\ckpt_tiny
```

只加载并生成（不训练）：

```powershell
.\\build\\Debug\\train_gpt.exe --load .\\data\\ckpt_tiny --steps 0 --prompt "Hello: " --gen 200 --temp 1.0 --topk 40
```

## 备注

- 词表是 256 个字节（0–255）。这是一种刻意的简化，用来先把训练端到端打通。
- 这个基线之后的常见下一步：SIMD、混合精度、CUDA 后端、ROCm/HIP 后端。

## 学习文档

建议从以下开始：
- `docs/README.md` / `docs/README.zh-CN.md`
- `docs/transformer_math_to_code.zh-CN.md`
- `docs/training_and_inference_walkthrough.zh-CN.md`

项目原则与工作流：
- `docs/project_lifecycle_guidelines.zh-CN.md`
- `.github/copilot-instructions.zh-CN.md`（本仓库的 Copilot 持久指导)
