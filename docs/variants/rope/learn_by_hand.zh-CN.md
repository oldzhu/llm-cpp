> [English](learn_by_hand.md)

# RoPE 手算示例

目标：用手算数字验证旋转数学。

## 配置

- `C=2`（一对维度）
- `T=2` 个位置

唯一 theta：`theta_0 = 1.0 / 10000^(0/2) = 1.0`

位置 0：`angle=0`，`cos=1.0`，`sin=0.0` → 无旋转
位置 1：`angle=1`，`cos≈0.5403`，`sin≈0.8415`

## 旋转前 Q 和 K

（见英文版）

## 旋转后

位置 0 无变化。位置 1 的旋转见英文版数值对比。

## 注意力得分

（见英文版详细计算）

缩放后得分：
```
Row 0: [12.0207, -inf]
Row 1: [13.7102, 37.4692]
```

## 代码映射

旋转循环：`rope_rotate` in `src/variants/rope/rope_attention.cpp`
注意力循环：`self_attention_rope` in `src/variants/rope/rope_attention.cpp`
