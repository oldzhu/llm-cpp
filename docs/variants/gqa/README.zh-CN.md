> [English](README.md)

# GQA（分组查询注意力）变体

此变体添加 **分组查询注意力**，其中 K 和 V 的头数少于 Q。多个 Q 头共享同一个 KV 头。

## 与 MHA 相比有何变化？

MHA：Q、K、V 头数相同。每个 Q 头关注自己的 K/V 头。

GQA：K/V 头数更少。每个 K/V 头被一组 Q 头共享。

## 为什么需要此变体

- GQA 用于 LLaMA 2/3，将 KV 缓存内存减少 `n_heads / n_kv_heads`
- 展示共享模式：多个 Q 头映射到相同 K/V
- 遵循变体模式：文档 + 代码 + 测试
