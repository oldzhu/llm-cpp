> [English](tokenizer_choice_gpt2_bpe_vs_sentencepiece.md)

# Tokenizer 选择：GPT-2 BPE vs SentencePiece（BPE/Unigram）

这份笔记用于辅助一个“下一步特性”的决策：从当前的 **byte vocab（256）** 迁移到 **子词（subword）token**（更“像词”的 tokens），这需要：

- 增加 tokenizer + 词表工件（artifacts）
- 数据集与模型的 `vocab_size` 跟随该词表（不再固定为 256）

本文范围：
- 对比 **经典 GPT-2 风格 BPE** 与 **SentencePiece**。
- 解释“**导入已有 vocab/merges**”与“**从语料训练 tokenizer**”的区别。
- 给出与本仓库约束一致的推荐（教学优先、最少依赖、确定性）。


## 当前基线约束（我们希望保持的东西）

- 教学优先，从零 C++。
- 可复现 / 确定性。
- 尽量少的依赖。
- 清晰的“数学/数据 → 代码”映射。
- checkpoint 兼容性很重要（tokenization 必须与 checkpoint 绑定）。


## 选项 A：GPT-2 风格的“经典 BPE”（byte-level 起步）

### 它是什么

一种以 **字节级字母表（byte-level alphabet）** 为起点的 BPE tokenizer：训练时反复合并频率最高的相邻 token 对；推理时使用预先计算好的 merge ranks/表进行合并。

在实践中，GPT-2 tokenization 通常还包含：
- byte→symbol 映射，保证任何输入 byte 都可表示。
- 一步 pre-tokenization（常见是偏 regex 的规则）在 merges 之前先把文本拆成片段。

### 优点

- 非常常见，资料丰富。
- 对任意文本鲁棒：byte-level 起步意味着几乎不存在“未知字符”问题。
- 运行时概念上简单：encode bytes → 依据 merge ranks 合并 → token ids。
- 教学面很友好：merges 很直观，容易讲清楚。
- 在保持零依赖的前提下，更容易从零实现（相比 SentencePiece 的 Unigram）。

### 缺点

- 想做到与 GPT-2 **完全兼容**，除了“BPE merges”还需要处理额外细节（byte 映射 + pretokenization 的约定）。
- 如果目标是与 GPT-2 tokenizer 工件互操作，就必须精确匹配这些细节。
- BPE 训练本身也是一段工程（但规模可控）。


## 选项 B：SentencePiece

SentencePiece 是一个 tokenizer 框架，常见产物是：
- **BPE** 模型，或
- **Unigram LM** 模型（另一种子词算法）

### 它是什么

- 直接作用在输入文本流上。
- 通常使用空白符标记（常见是 `▁`）来保留词边界信息。

### 优点

- 空白符处理更统一、更可预测（对 decode 与“更像词”的 token 行为有帮助）。
- 许多开源模型生态广泛使用。
- Unigram 在实践中表现常常很好，概念上也很优雅（token inventory + 概率）。

### 缺点

- 从零实现 SentencePiece 兼容行为并不简单。
  - 尤其是 Unigram，相对 BPE merges 是更大的一步算法跨越。
- 直接使用官方 SentencePiece 库会与本仓库“最少依赖”的目标冲突。
- 想做到精确兼容，可能还要匹配 normalization / special tokens 等约定。


## 互操作性 vs 简洁性：关键权衡

- 若优先 **教学与最少依赖**：先做一个自包含 BPE（GPT-2-ish）。
- 若优先 **兼容既有开源模型/tokenizer 工件**：SentencePiece 工件格式很常见，但完整兼容成本更高。


## “导入已有 vocab/merges” vs “从语料训练”

这个选择独立于 GPT-2 vs SentencePiece：它讨论的是 tokenizer 工件来自哪里。

### 1) 导入已有工件（使用预训练 tokenizer）

直接使用外部产出的 vocab/merge/model 文件。

工件例子：
- GPT-2 BPE：`vocab.json` + `merges.txt`
- SentencePiece：`tokenizer.model`（以及可选的 `.vocab`）

优点：
- 立刻与该生态互操作。
- 不必先实现 tokenizer training。

缺点：
- 文件格式与边界行为可能复杂。
- 词表通常更大（例如 30k–100k），会显著增加 CPU FP32 下 embedding + softmax 成本。
- checkpoint 必须记录 tokenizer 的 identity/version，避免 vocab 不匹配。

### 2) 在我们的语料上训练 tokenizer 工件

在我们的数据集上训练 tokenizer（BPE merges 或 Unigram 模型），生成一个较小词表。

优点：
- 可以选择较小 vocab（例如 2k–10k）来保持训练速度。
- 教学价值高：学习者能看到 tokenization 是如何构造出来的。
- 工件格式可以设计得更简单、并完整文档化。

缺点：
- 与其他 tokenizer/checkpoint 不直接兼容。
- 需要编写并验证 training pipeline。


## 本仓库的推荐方案（务实 v1）

作为第一个里程碑，且不引入“魔法依赖”：

1) 新增一个 **简单 BPE tokenizer** 模块（自包含 C++）。
2) 先用较小 vocab 目标（例如 2k–8k），在我们的数据集上训练。
3) 工件格式刻意保持简单（例如 merges 的纯文本文件 + token 列表）。
4) 加入严格的 checkpoint 兼容规则：
   - checkpoint 记录 tokenizer 名称/版本 + vocab size
   - load 时若 tokenizer 不匹配则直接失败

若未来要提高兼容性：
- 再增加一个 tokenizer backend，用于导入 GPT-2 merges/vocab 或提供 SentencePiece 兼容 reader。


## 代码库需要改变什么（高层）

- 数据流水线：
  - dataset 从 bytes 变为 `std::vector<int32_t> token_ids`
  - batching 采样 token 子串
- 模型配置：
  - `vocab_size` 变为由 tokenizer vocab 决定的动态值
- 采样/打印：
  - 生成时打印解码后的 token（token ids → string）
- checkpoint：
  - 存储 tokenizer identity + vocab hash/version，并拒绝不匹配


## 最小应添加的测试

- tokenizer round-trip：
  - decode(encode(text)) 的结果“合理”（是否严格相等取决于空白符约定）
- golden tokenization example：
  - 一个很小的已知字符串对应一个已知 token id 序列
- checkpoint 兼容性：
  - 当 vocab/tokenizer 元数据不匹配时，load 必须失败


## 决策清单

优先做 GPT-2 风格 BPE（GPT-2-ish）如果：
- 最少依赖不可妥协
- 希望把核心算法讲清楚
- 可以接受先做“GPT-2-ish”，再迭代到更高兼容性

优先做 SentencePiece 兼容如果：
- 与现有 tokenizer 工件互操作是最高优先级
- 接受更高实现复杂度（或外部依赖）
