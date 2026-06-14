> [简体中文](learn_by_hand.zh-CN.md)

# GQA learn-by-hand (tiny example)

Goal: verify the grouping logic with hand-computable numbers.

## Setup

- `B=1`, `T=2`, `n_heads=4`, `n_kv_heads=2`, `D=1`
- `heads_per_kv = 4/2 = 2`
- Q heads: 0,1,2,3 → KV heads: 0,0,1,1

## Q, K, V data

**Q** (per head, per token):
```
h0: q[0]=[1.0],  q[1]=[2.0]
h1: q[0]=[3.0],  q[1]=[4.0]
h2: q[0]=[5.0],  q[1]=[6.0]
h3: q[0]=[7.0],  q[1]=[8.0]
```

**K** (per kv-head, per token):
```
hv0: k[0]=[10.0], k[1]=[11.0]
hv1: k[0]=[20.0], k[1]=[21.0]
```

**V** (per kv-head, per token):
```
hv0: v[0]=[0.1],  v[1]=[0.2]
hv1: v[0]=[0.3],  v[1]=[0.4]
```

## Scores (scale=1, causal mask)

Q head h0 → KV head hv0:
- i=0: s[0,0]=1.0*10.0=10.0, s[0,1]=masked
- i=1: s[1,0]=2.0*10.0=20.0, s[1,1]=2.0*11.0=22.0

Q head h2 → KV head hv1:
- i=0: s[0,0]=5.0*20.0=100.0, s[0,1]=masked
- i=1: s[1,0]=6.0*20.0=120.0, s[1,1]=6.0*21.0=126.0

## After softmax

h0: row0=[1,0], row1=[0.119, 0.881]
h2: row0=[1,0], row1=[0.00247, 0.998]

## Output

h0, t=0: Y = 1.0*v0[0] = 0.1
h0, t=1: Y = 0.119*v0[0] + 0.881*v0[1] = 0.1881
h2, t=0: Y = 1.0*v1[0] = 0.3
h2, t=1: Y = 0.00247*v1[0] + 0.998*v1[1] = 0.3999

## Key insight

h1 and h0 map to the same KV head (hv0), so they produce different outputs from the same K/V. h3 maps to hv1 with outputs similar to those described above. Each KV head "serves" 2 Q heads.
