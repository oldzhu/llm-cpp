> [English](tokenizer_choice_gpt2_bpe_vs_sentencepiece.md)

# Tokenizer 选择：GPT-2 BPE vs SentencePiece（BPE/Unigram）

这份文档用于支持一个下一步特性：从当前的 **byte vocab=256** 切换到 **子词（subword）token**（更“像词”的 token），需要增加 tokenizer + vocab 工件，并让数据集/模型的 `vocab_size` 跟随该词表。

## 对比要点（简版）

### GPT-2 风格 BPE（byte-level）
优点：
- 更容易在本仓库“零依赖”地自实现（更适合教学）。
- byte-level 起步，对任意文本鲁棒，不容易遇到未知字符。

缺点：
- 想做到与 GPT-2 完全兼容，需要处理一些额外细节（byte 映射、pretokenize 规则等）。

### SentencePiece（BPE 或 Unigram）
优点：
- 对空格/词边界处理更统一（常见 `▁` 标记），很多模型生态在用。

缺点：
- 从零实现完整兼容更复杂；引入官方库则不符合“最少依赖”的目标。

## “导入 vocab/merges” vs “从语料训练”是什么意思？

- 导入（import）：直接使用已有 tokenizer 的词表/merges（例如 GPT-2 的 `vocab.json` + `merges.txt`，或 SentencePiece 的 `.model`）。
  - 优点：快速兼容某个生态。
  - 缺点：格式/细节复杂，词表常很大（CPU FP32 下 softmax 成本更高）。

- 训练（learn/train）：在我们的语料上训练一个较小的 BPE/Unigram tokenizer，生成自己的 vocab/merges。
  - 优点：可以选较小 vocab（例如 2k–10k），更快、更适合教学。
  - 缺点：与外部 tokenizer 不直接兼容。

> 说明：本中文版本为摘要；更多细节见英文版。
