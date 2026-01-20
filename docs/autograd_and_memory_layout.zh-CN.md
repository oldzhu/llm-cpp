> [English](autograd_and_memory_layout.md)

# Autograd 与内存布局（如何像读论文一样读 C++）

> 说明：本中文版本翻译进行中；完整内容请先阅读英文版。

要点：
- 论文里常被省略，但在从零实现中非常关键：
  - 张量的 flat buffer 如何对应到 `[B,T,C]` 等形状
  - 反向传播/计算图在本仓库如何组织

建议配合：
- `docs/attention_indexing_trace.md`
- `docs/transformer_math_to_code.md`
