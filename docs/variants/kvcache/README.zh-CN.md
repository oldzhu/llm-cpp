> [English](README.md)

# KV-cache 变体

此变体使用 Key-Value 缓存添加 **增量前向路径**，使得自回归生成在每一步都无需重新计算所有历史 token 的 K 和 V。

代码：
- `src/variants/kvcache/kvcache_attention.h`
- `src/variants/kvcache/kvcache_attention.cpp`

## 与基线相比有何变化？

基线 (`nn::self_attention_1h`) 在每次生成新 token 时都会重新计算完整序列 `[B,T,C]` 的 Q、K、V。

此变体将注意力分为两个 API 接口：

1. **预填充** — `self_attention_prefill(x, w_qkv, b_qkv, w_proj, b_proj, cache)`：
   - 计算长度为 T 的完整上下文的 Q、K、V。
   - 将 K 和 V 存储到缓存中。
   - 正常计算注意力（带因果掩码）。
   - 返回输出 `[B,T,C]`。

2. **逐步** — `self_attention_step(x_step, w_qkv, b_qkv, w_proj, b_proj, cache)`：
   - 接受单个新 token `[B,1,C]`。
   - 计算 `Q_new, K_new, V_new`。
   - 将 `K_new, V_new` 追加到缓存中。
   - 计算 `Q_new` 对 **所有缓存 K** 的注意力（无需因果掩码——新 token 位于序列末尾）。
   - 返回输出 `[B,1,C]`。

时间复杂度变化：
- 基线生成：每步 O(T²)（与所有历史 K 做点积）
- KV-cache 生成：每步 O(T)（与缓存 K 做点积，无需重新计算）

## 结构

```
KVCache {
    k_cache_ : [B, T_max, C]  // 缓存的 keys
    v_cache_ : [B, T_max, C]  // 缓存的 values
    cur_len  : int            // 当前缓存长度
}
```

缓存存储的是普通 `nn::Tensor`，不附加 autograd（仅用于生成路径）。

## 为什么需要此变体

- 实现高效的自回归生成（无冗余计算）。
- 是基于 GPU 推理的前提（缓存的 K/V 可以驻留在设备上）。
- 遵循与 MHA 相同的变体模式：文档 + 代码 + 测试。
- 索引保持显式化，学习者可以追踪每次内存访问。

## 生成流程示意

```
// 预填充：处理完整的 prompt
auto out = kvcache::self_attention_prefill(x_prompt, ..., cache);

// 逐步：逐个 token 生成
for each new token:
    auto out_step = kvcache::self_attention_step(x_new, ..., cache);
```
