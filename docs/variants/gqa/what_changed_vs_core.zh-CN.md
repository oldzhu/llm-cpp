> [English](what_changed_vs_core.md)

# GQA 扩展：与 MHA 相比的变化

## 0) 整体视图

MHA：Q、K、V 都有 `n_heads`。每个 Q 头独立关注自己的 K/V 头。

GQA：K 和 V 只有 `n_kv_heads`（更少）。每个 K/V 头服务 `n_heads / n_kv_heads` 个 Q 头。

## 1) 变更内容

### API
- 接受 4D Q、K、V 张量（已预分为头）
- 额外参数：`n_kv_heads`

### 头映射
```
kv_idx(q_head) = q_head * n_kv_heads / n_heads
```

## 2) 内存影响

KV-cache 大小与 `n_kv_heads` 而非 `n_heads` 成正比。
