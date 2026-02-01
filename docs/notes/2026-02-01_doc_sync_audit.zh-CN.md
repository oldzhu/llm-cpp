> [English](2026-02-01_doc_sync_audit.md)

# 2026-02-01 — 文档同步审计（后端 + tokenizer）

## 目标

在引入以下改动后，把文档重新同步到当前实现：
- 真实的 backend seam（当前 CPU 后端；规划 CUDA/HIP）
- 可插拔 tokenizer（ByteTokenizer + 最小 BPE tokenizer）

## 本次修改

### 文档

- 更新后端相关文档以反映现有 seam，并修正目录命名：
  - seam 已经存在（KernelBackend + registry；matmul/bmm 通过后端执行）
  - 将 `src/backends/*` 更正为 `src/backend/*`
- 更新训练/推理 walkthrough：
  - 说明 tokenization 是可插拔的
  - 说明当前训练/数据集 sanity 流水线仍然是 byte-based（ByteDataset，V=256）
- 在 tokenizer 选择笔记中加入“状态更新”：
  - Tokenizer 接口已存在
  - 已实现 ByteTokenizer 与最小/导入式 BpeTokenizer
  - 端到端 subword 训练仍需要新增“token id 数据集”路径
- 更新 autograd/内存布局笔记：
  - `matmul2d` 在 `src/ops.cpp` 中构建 autograd node，但实际计算通过 backend 执行（CPU 后端使用显式 for-loop）

### 代码

- 修复非 byte tokenizer 的生成输出：
  - 生成的 token id 会通过 `Tokenizer::decode` 解码并打印
  - `--ascii-only` / `--escape-bytes` 仅允许用于 byte tokenizer
- 避免误导性的运行方式：
  - 训练与 dataset-based sanity 检查现在要求 `--tokenizer byte`（直到实现 token-id dataset 路径）

## 验证

- 运行 `ctest -C Debug`（通过）。
