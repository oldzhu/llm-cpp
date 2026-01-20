> [English](attention_indexing_trace.md)

# 注意力索引追踪（Attention indexing trace）

> 说明：本中文版本翻译进行中；完整内容请先阅读英文版。

这篇文档用一个具体的例子把注意力相关张量（例如 scores/weights/output）的下标与 flat buffer 的偏移对应起来，帮助你把“数学符号”落到“数组地址”。
