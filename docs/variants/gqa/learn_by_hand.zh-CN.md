> [English](learn_by_hand.md)

# GQA 手算示例

## 配置
- `n_heads=4`, `n_kv_heads=2`, `D=1`
- Q 头 0,1 → KV 头 0；Q 头 2,3 → KV 头 1

（详细计算见英文版）

## 关键洞察

Q 头 h0 和 h1 映射到同一个 KV 头 (hv0)，因此从相同的 K/V 产生不同的输出。每个 KV 头"服务" 2 个 Q 头。
