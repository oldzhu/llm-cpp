> [简体中文](README.zh-CN.md)

# Rotary Positional Embedding (RoPE) variant

This variant adds **Rotary Positional Embedding** to self-attention, replacing the learned absolute positional embeddings with rotation-based relative position encoding.

Code:
- `src/variants/rope/rope_attention.h`
- `src/variants/rope/rope_attention.cpp`

## What changes vs the baseline?

Baseline (`nn::self_attention_1h`): adds learned positional embedding `Wpe[pos]` to the token embedding before attention.

RoPE variant (`nn::variants::rope::self_attention_rope`):
- Removes absolute positional embedding addition.
- After computing Q and K (via linear projection), rotates each pair of dimensions by an angle proportional to the position.

## Math

For dimension pair `(2k, 2k+1)` and position `i`:

```
theta_k  = 1.0 / (10000 ^ (2k / C))
angle    = i * theta_k
cos, sin = cos(angle), sin(angle)

q'[2k]   = q[2k] * cos - q[2k+1] * sin
q'[2k+1] = q[2k+1] * cos + q[2k] * sin
```

Same rotation applied to K.

The key property: `dot(Q_i_rotated, K_j_rotated)` depends only on `(i - j)`, giving the model **relative position awareness** without learned position embeddings.

## Structure

```cpp
void rope_rotate(Tensor& q, Tensor& k, int B, int T, int C);

Tensor self_attention_rope(const Tensor& x,
                           const Tensor& w_qkv,
                           const Tensor& b_qkv,
                           const Tensor& w_proj,
                           const Tensor& b_proj);
```

## Why this variant exists

- RoPE is used in LLaMA, Qwen, and most modern LLMs.
- Demonstrates relative position encoding (position depends on Q·K dot products, not additions).
- Follows the variant pattern: docs + code + test.
- Explicit loops for teaching.

## Paper reference

"RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
