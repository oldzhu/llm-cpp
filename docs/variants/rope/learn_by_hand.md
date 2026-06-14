> [简体中文](learn_by_hand.zh-CN.md)

# RoPE learn-by-hand (tiny example)

Goal: verify the rotation math with hand-computable numbers.

## Setup

- `C=2` (one dimension pair)
- `T=2` positions
- `B=1`

The only theta: `theta_0 = 1.0 / 10000^(0/2) = 1.0 / 1.0 = 1.0`

Position 0: `angle = 0 * 1.0 = 0`, `cos=1.0`, `sin=0.0`
Position 1: `angle = 1 * 1.0 = 1`, `cos≈0.5403`, `sin≈0.8415`

## Input Q and K (before RoPE)

```
Q:  q[0] = [1.0, 2.0]    q[1] = [3.0, 4.0]
K:  k[0] = [5.0, 6.0]    k[1] = [7.0, 8.0]
```

## After RoPE rotation

### Position 0 (cos=1, sin=0)
```
q'[0,0] = 1.0*1.0 - 2.0*0.0 = 1.0
q'[0,1] = 2.0*1.0 + 1.0*0.0 = 2.0
k'[0,0] = 5.0*1.0 - 6.0*0.0 = 5.0
k'[0,1] = 6.0*1.0 + 5.0*0.0 = 6.0
```
→ No change (position 0 is identity rotation)

### Position 1 (cos≈0.5403, sin≈0.8415)
```
q'[1,0] = 3.0*0.5403 - 4.0*0.8415 = 1.6209 - 3.3660 = -1.7451
q'[1,1] = 4.0*0.5403 + 3.0*0.8415 = 2.1612 + 2.5245 = 4.6857
k'[1,0] = 7.0*0.5403 - 8.0*0.8415 = 3.7821 - 6.7320 = -2.9499
k'[1,1] = 8.0*0.5403 + 7.0*0.8415 = 4.3224 + 5.8905 = 10.2129
```

## Attention scores (scale=1/sqrt(2) ≈ 0.7071, causal mask)

### Scores after RoPE
```
q'[0]·k'[0] = 1.0*5.0 + 2.0*6.0 = 17.0
q'[0]·k'[1] = masked (j>i)

q'[1]·k'[0] = (-1.7451)*5.0 + 4.6857*6.0 = -8.7255 + 28.1142 = 19.3887
q'[1]·k'[1] = (-1.7451)*(-2.9499) + 4.6857*10.2129 = 5.1479 + 47.8499 = 52.9978
```

Multiply by scale 0.7071:
```
Row 0: [12.0207, -inf]
Row 1: [13.7102, 37.4692]
```

## Key property: relative position

Without RoPE: `q[1]·k[0] = 3*5 + 4*6 = 39`, `q[1]·k[1] = 3*7 + 4*8 = 53`
With RoPE: the dot products encode the relative distance through rotation.

## Mapping to code

Rotation loop: `rope_rotate` in `src/variants/rope/rope_attention.cpp`
Attention loop: `self_attention_rope` in `src/variants/rope/rope_attention.cpp`
