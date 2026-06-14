> [English](README.md)

# RoPE（旋转位置编码）变体

此变体为自注意力添加 **旋转位置编码**，用基于旋转的相对位置编码替代学习型绝对位置嵌入。

代码：
- `src/variants/rope/rope_attention.h`
- `src/variants/rope/rope_attention.cpp`

## 与基线相比有何变化？

基线 (`nn::self_attention_1h`)：在注意力之前将学习型位置嵌入 `Wpe[pos]` 加到 token 嵌入上。

RoPE 变体 (`nn::variants::rope::self_attention_rope`)：
- 移除绝对位置嵌入加法。
- 通过线性投影计算 Q 和 K 后，将每对维度按与位置成正比的角度旋转。

## 数学

对维度对 `(2k, 2k+1)` 和位置 `i`：

```
theta_k  = 1.0 / (10000 ^ (2k / C))
angle    = i * theta_k
cos, sin = cos(angle), sin(angle)

q'[2k]   = q[2k] * cos - q[2k+1] * sin
q'[2k+1] = q[2k+1] * cos + q[2k] * sin
```

对 K 应用相同旋转。

关键性质：`dot(Q_i_rotated, K_j_rotated)` 仅依赖于 `(i - j)`，为模型提供 **相对位置感知** 而无需学习型位置嵌入。

## 为什么需要此变体

- RoPE 用于 LLaMA、Qwen 和大多数现代 LLM。
- 展示相对位置编码（位置取决于 Q·K 点积，而非加法）。
- 遵循变体模式：文档 + 代码 + 测试。
- 显式循环教学。

## 论文参考

"RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
