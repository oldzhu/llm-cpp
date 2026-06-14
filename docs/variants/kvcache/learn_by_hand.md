> [简体中文](learn_by_hand.zh-CN.md)

# KV-cache learn-by-hand (tiny example)

Goal: verify the shape logic and "cache integrity" — the KV-cache step must produce the same output as a full recomputation.

## Setup

Use a single batch and three tokens:
- `B=1`, `T=3`

Model width `C=2`. Use tiny, hand-picked values (scale omitted for clarity).

### Weights (kept tiny)

We ignore QKV projection weights for this example and work directly with projected Q, K, V values. This focuses on the cache mechanism itself.

Given (after QKV projection):

Token 0: `q0=[1,1]`, `k0=[1,0]`, `v0=[2,0]`
Token 1: `q1=[0,1]`, `k1=[0,1]`, `v1=[0,3]`
Token 2: `q2=[1,2]`, `k2=[2,0]`, `v2=[1,1]`

## Prefill (full 3-token sequence)

### Scores (causal, scale=1)

For token 0 (i=0):
- j=0: `q0·k0 = 1*1 + 1*0 = 1`
- j=1: masked (future)
- j=2: masked (future)
- Row: `[1, -inf, -inf]`

For token 1 (i=1):
- j=0: `q1·k0 = 0*1 + 1*0 = 0`
- j=1: `q1·k1 = 0*0 + 1*1 = 1`
- j=2: masked (future)
- Row: `[0, 1, -inf]`

For token 2 (i=2):
- j=0: `q2·k0 = 1*1 + 2*0 = 1`
- j=1: `q2·k1 = 1*0 + 2*1 = 2`
- j=2: `q2·k2 = 1*2 + 2*0 = 2`
- Row: `[1, 2, 2]`

### After softmax (approximate)

Token 0: `[1.0, 0.0, 0.0]`
Token 1: `[0.269, 0.731, 0.0]`
Token 2: `[0.211, 0.576, 0.212]`

### Output Y (weighted V)

Y[0] = 1.0 * v0 = `[2, 0]`
Y[1] = 0.269 * v0 + 0.731 * v1 = `[0.538, 2.193]`
Y[2] = 0.211 * v0 + 0.576 * v1 + 0.212 * v2 = `[0.634, 1.940]`

### Cache state after prefill

```
k_cache = [k0, k1, k2]  = [[1,0], [0,1], [2,0]]
v_cache = [v0, v1, v2]  = [[2,0], [0,3], [1,1]]
cur_len = 3
```

## Step (generate token 3)

New token after QKV projection: `q3=[0,2]`, `k3=[1,1]`, `v3=[3,0]`

### Append to cache

```
k_cache = [k0, k1, k2, k3]  = [[1,0], [0,1], [2,0], [1,1]]
v_cache = [v0, v1, v2, v3]  = [[2,0], [0,3], [1,1], [3,0]]
cur_len = 4
```

### Scores (no causal mask needed for last position)

q3 · k0 = 0*1 + 2*0 = 0
q3 · k1 = 0*0 + 2*1 = 2
q3 · k2 = 0*2 + 2*0 = 0
q3 · k3 = 0*1 + 2*1 = 2

Row: `[0, 2, 0, 2]`

### After softmax

`[0.061, 0.450, 0.061, 0.450]` (approximately)

### Output Y[3]

Y[3] = 0.061*v0 + 0.450*v1 + 0.061*v2 + 0.450*v3
     = 0.061*[2,0] + 0.450*[0,3] + 0.061*[1,1] + 0.450*[3,0]
     = [0.122+0+0.061+1.350, 0+1.350+0.061+0]
     = `[1.533, 1.411]` (approximately)

## Verification: full recomputation gives the same Y[3]

With all 4 tokens and fresh attention, the scores for token 3 are:
- i=3: same dot products as above → same softmax → same Y[3].

So the cache produces **identical output** to a full recomputation.

## Mapping to code

The prefill logic mirrors `nn::self_attention_1h` but also writes K and V into cache tensors.

The step logic is in `self_attention_step` in:
- `src/variants/kvcache/kvcache_attention.cpp`

No causal mask is needed in the step because the query is always at the end.
