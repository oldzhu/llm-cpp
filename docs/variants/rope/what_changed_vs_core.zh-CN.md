> [English](what_changed_vs_core.md)

# RoPE 扩展：与核心基线相比的变化

## 0) 整体视图

基线：绝对位置嵌入 `Wpe[pos]` 在注意力之前加到 token 嵌入上。

RoPE 变体：通过 **旋转** Q 和 K 向量来编码位置，产生相对位置感知。

## 1) 变更内容

### 移除
- 绝对位置嵌入加法（无 `add_positional` 步骤）

### 新增
- `rope_rotate()`：按位置相关角度就地旋转 Q 和 K
- `self_attention_rope()`：带 RoPE 的完整注意力函数

### 未变更
- V（values）不旋转
- 注意力计算（scores、softmax、加权和、投影）与基线相同
- 参数布局：`w_qkv: [C,3C]`、`w_proj: [C,C]`，bias 形状全部相同

## 2) 数学差异

（见英文版）

## 3) 关键性质：相对位置

`dot(RoPE(Q_i), RoPE(K_j))` 依赖于 `cos((i-j) * theta_k)` —— 是相对位置而非绝对位置。

## 4) 何时使用

- RoPE 取代学习型位置嵌入的需求
- 适用于任意序列长度（无 wpe 大小限制）
- 与 KV-cache 兼容（RoPE 可以每步应用）
