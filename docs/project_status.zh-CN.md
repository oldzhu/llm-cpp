# 项目当前状态 — 2026-06-14

> [EN](project_status.md)

## 项目概况

从零开始的 C++ GPT 风格 Transformer，支持训练/推理/导入/Web UI/测试。

## 已完成组件

### C++ 核心
- 张量 + 自动求导引擎 (`tensor.cpp/h`)
- 15+ 算子 (`ops.cpp/h`: matmul, softmax, LN, GELU, SiLU, RMSNorm, attention)
- TinyGPT 模型 (`model.cpp/h`: 嵌入, 位置, 块, LM头)
- AdamW 优化器 (`optim.cpp/h`)
- 检查点 v4 带 v1-v3 向后兼容 (`checkpoint.cpp/h`)
- 字节数据集 + Token数据集 (`data.cpp/h`)

### C++ 变体 (src/variants/)
- MHA — 多头注意力 (n_heads)
- KV-cache — 增量生成
- RoPE — 旋转位置编码
- GQA — 分组查询注意力 (n_kv_heads)
- MoE — 混合专家 FFN

### C++ 后端
- CPU 参考后端 (naive matmul)
- BlockedSimd 后端 (AVX FMA, 64×64 tiles)
- Vulkan 后端 (可选, POC模式, 需要Vulkan SDK)
- 权重导出 (binary + JSON + safetensors)

### C++ 分词器
- ByteTokenizer (V=256)
- BpeTokenizer (GPT-2: JSON vocab + bytes_to_unicode 映射)

### C++ CLI
- 30+ 标志 (--data, --steps, --dmodel, --norm, --mlp, --tokenizer...)
- --progress-json 输出
- --pipe-stdin 优雅关机 (EXIT → 保存 → 停止)
- --serve 模式 (stdin→stdout JSON 生成循环)
- --dump-weights + 格式 (binary/json/safetensors)

### Python Web 服务器
- FastAPI + WebSocket (15 路由)
- 训练会话管理 (子进程)
- 代码浏览器 (文件树 + AI 解释)
- AI 聊天代理 (DeepSeek/OpenAI/Anthropic 兼容)
- 对话会话 (加载检查点 → 生成)
- Playwright E2E 测试框架 (待 Node.js/Playwright 安装)

### Web UI 标签页 (6)
- ⚙ 配置 — 训练配置表单 + 预设 + 工具提示
- 📊 监控 — 实时损失曲线 (Chart.js) + 权重统计
- 📁 代码 — 文件树 + 交互式 SVG 架构图 + AI 面板
- 🤖 AI 对话 — 项目感知 AI 聊天
- 📖 学习 — 7 章渐进式教程
- 💬 对话 — 与训练模型聊天 (C++ --serve)

### i18n
- EN/中文 切换 (~150 字符串)
- 工具提示 (50+ 配置字段)

### 脚本
- `import_safetensors.py` — GPT-2 + LLaMA 映射
- 2 个映射文件 (gpt2.json, llama.json)

### 测试
- C++: 20 个测试 (ctest)
- Python: 44 个测试 (pytest)
- 测试基础指南: `docs/testing_guide.md`

### 文档
- 双语 EN+中文 所有文档
- Chat 跟踪 (`docs/chat/chat-*.md`)
- 设计说明: `docs/notes/`
- 变体说明: `docs/variants/`

## 关键指标

| 指标 | 值 |
|------|-----|
| C++ 源文件 | ~18 |
| Python 文件 | ~12 |
| 测试通过 | 64 (20 C++ + 44 Python) |
| Web 路由 | 15 |
| Web 标签页 | 6 |
| i18n 字符串 | ~150 |
| 变体 | 5 |
| 后端 | 3 |
| 检查点版本 | v4 |
