> [English](hidden_states_vs_attention_matrices.md)

# Hidden states 与 attention 矩阵的区别

> 说明：本中文版本翻译进行中；完整内容请先阅读英文版。

这篇文档解释 `[B,T,C]`（隐状态）与 `[B,T,T]`（注意力权重/得分）分别代表什么，并强调它们在内存中的 flat offset 如何计算。
