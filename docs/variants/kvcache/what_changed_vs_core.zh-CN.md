> [English](what_changed_vs_core.md)

# KV-cache 扩展：与核心基线相比的变化

本文档解释了 **KV-cache 扩展** 添加了什么、刻意没有改变什么，以及如何映射到标准 Transformer 推理模式。

## 0) 整体视图

基线/核心：
- `nn::self_attention_1h` 每次调用都计算完整的 `[B,T,C]` 注意力，为所有 `i,j` 对重新计算 K 和 V 点积。

KV-cache 扩展：
- 我们在 `src/variants/kvcache/` 下添加了基于缓存的增量路径，作为 **并列变体**。
- 我们 **没有** 修改基线注意力函数。

## 1) C++ 结构/类的变化

### 1.1 新增数据结构：`KVCache`

```cpp
struct KVCache {
    int B, T_max, C;
    int cur_len = 0;

    nn::Tensor k_cache_; // [B, T_max, C]
    nn::Tensor v_cache_; // [B, T_max, C]

    KVCache(int B, int T_max, int C);
    void reset();
};
```

缓存张量是纯数据存储（无 autograd）。它们仅在禁用梯度的生成模式下使用。

### 1.2 新函数（非新模型类）

两个新函数：
- `nn::variants::kvcache::self_attention_prefill` — 填充缓存并返回输出
- `nn::variants::kvcache::self_attention_step` — 使用缓存，返回单位置输出

文件：
- `src/variants/kvcache/kvcache_attention.h`
- `src/variants/kvcache/kvcache_attention.cpp`

### 1.3 核心模型类型：未变

未更改：
- `model::Config`、`model::TinyGPT`、`TinyGPT::Block`
- `nn::ops` 层原语（`linear_lastdim`、`softmax_lastdim` 等）

## 2) 参数/形状的变化

**无。** 两个函数接受与 `nn::self_attention_1h` 相同的参数形状：
- `w_qkv: [C, 3C]`, `b_qkv: [3C]`
- `w_proj: [C, C]`, `b_proj: [C]`

唯一区别是 **K 和 V 如何在调用之间被复用**。

## 3) 数学差异

### 预填充：与基线相同，外加缓存写入

对输入 `x` 形状 `[B,T,C]`：

1. `qkv = linear(x, w_qkv, b_qkv)` → `[B,T,3C]`
2. `q = qkv[:,:,:C]`, `k = qkv[:,:,C:2C]`, `v = qkv[:,:,2C:3C]`
3. **将 `k` 拷贝到 `cache.k_cache_[:,:T,:]`，`v` 拷贝到 `cache.v_cache_[:,:T,:]`**
4. `cache.cur_len = T`
5. 完全按照基线计算注意力（因果掩码，`scale = 1/√C`）

### 逐步：无因果掩码，标量查询（T=1）

对输入 `x_step` 形状 `[B,1,C]`：

1. `qkv = linear(x_step, w_qkv, b_qkv)` → `[B,1,3C]`
2. `q_new = qkv[:,:,:C]`, `k_new = qkv[:,:,C:2C]`, `v_new = qkv[:,:,2C:3C]`
3. **将 `k_new` 追加到位置 `cache.cur_len`，`v_new` 追加到 `cache.cur_len`**
4. `cache.cur_len += 1`
5. 计算查询位置 0 对所有缓存 K 的得分：

$$
S[0,j] = \frac{Q_{new}[0] \cdot K_{cache}[j]}{\sqrt{C}},\quad j \in [0, \text{cur\_len}-1]
$$

无需因果掩码——新 token 是最后一个位置，可以关注所有内容。

6. `probs = softmax(scores)`
7. `att[0] = sum_j probs[j] * V_cache[j]`
8. `out = linear(att, w_proj, b_proj)`

### 关键简化

在 step 路径中，T_query = 1，T_key = cur_len。注意力矩阵是 `[B,1,cur_len]` 而非 `[B,T,T]`。这就是 O(T) vs O(T²) 加速的来源。

## 4) 构建目标变化（CMake）

添加了 `llm_variant_kvcache` 库：
- 编译 `src/variants/kvcache/kvcache_attention.cpp`
- 链接 `llm_core`

`test_build_llm` 链接 `llm_variant_kvcache` 用于验证测试。

## 5) 测试

一个聚焦的测试证明缓存等价性：
- `test_kvcache_matches_full_attention()`：
  - 对 T 个 token 运行 prefill（填充缓存）。
  - 对第 T+1 个 token 运行 step（使用缓存）。
  - 对所有 T+1 个 token 从头重新运行注意力。
  - 验证第 T+1 个输出匹配。

## 6) 何时使用

KV-cache 是仅用于生成的优化。训练仍使用完整注意力路径。缓存适用于：
- CLI 生成模式（`--prompt ... --gen N`）
- 任何自回归文本生成场景

## 7) 下一步

- 与 MHA 变体结合实现多头 KV-cache。
- 扩展 backend seam 以通过加速 kernel 路由逐步注意力。
- 启用 GPU 驻留缓存以实现零拷贝推理。
