> [简体中文](learn_by_hand.zh-CN.md)

# MHA learn-by-hand (tiny example)

Goal: sanity-check the *shape logic* and “per-head independence” without getting lost in large matrices.

## Setup

Use a single batch and two tokens:
- `B=1`, `T=2`

Use two heads:
- `H=2`

Use per-head dim 1:
- `D=1` so total model dim is `C = H*D = 2`

So each token embedding is a length-2 vector that we view as two heads:

- head0 takes channel 0
- head1 takes channel 1

Assume after the QKV linear we already have:

- `Q,K,V` each shaped `[T,C] = [2,2]` (batch omitted)

Let:

- Token 0: `q0=[q00,q01]`, `k0=[k00,k01]`, `v0=[v00,v01]`
- Token 1: `q1=[q10,q11]`, `k1=[k10,k11]`, `v1=[v10,v11]`

Reshape to heads (conceptually):
- head0 sees scalar pairs `(q00,q10)` etc.
- head1 sees scalar pairs `(q01,q11)` etc.

## Scores (causal)

For head0 (dim is scalar):

- For token i=0:
  - score to j=0: `s00 = q00*k00 / sqrt(1)`
  - score to j=1: masked (future) → `-inf`
- For token i=1:
  - score to j=0: `s10 = q10*k00`
  - score to j=1: `s11 = q10*k10`

Head1 is identical but uses channel 1:
- `s00' = q01*k01`, etc.

## Softmax and output

Because `T=2`, softmax per row is easy:
- For token 0, each head softmax is `[1, 0]` due to masking.
- For token 1, each head softmax is softmax over two numbers.

Outputs per head:
- `y0_head0 = v00`
- `y0_head1 = v01`

So for token 0, concatenation gives `[v00, v01]`.

This is a good check: with causal masking, the first token can only attend to itself, so the output must be exactly its own V for each head.

## Mapping to code

The loops implementing these steps are in:
- `src/variants/mha/mha_attention.cpp` (`scores`, `softmax_lastdim`, and `att` loops)

The causal mask is the same rule as the baseline: `j > i` is masked.
