# Sample processing walkthrough: baseline 1-head vs MHA variant

This is a concrete, step-by-step walkthrough of how one forward pass computes attention in this repo.

We do **two passes** over the same conceptual inputs:
1) baseline `nn::self_attention_1h` (single head)
2) variant `nn::variants::mha::self_attention_mha` (multi head)

We focus on shapes, indexing, and what each intermediate tensor *means*.

Relevant code:
- Baseline: `nn::self_attention_1h` in `src/ops.cpp`
- Variant: `nn::variants::mha::self_attention_mha` in `src/variants/mha/mha_attention.cpp`

## 0) Choose tiny concrete sizes

Let:
- `B = 1` (one batch element)
- `T = 3` (three tokens)
- `C = 4` (model width)

For MHA, choose:
- `H = 2` heads
- `D = C/H = 2` per-head dim

We assume we already have the hidden states `x`:
- `x` shape `[B,T,C] = [1,3,4]`

Memory layout reminder (row-major):
- `[B,T,C]`: `off(b,t,c) = (b*T + t)*C + c`
- `[B,T,T]`: `off(b,i,j) = (b*T + i)*T + j`
- For MHA variant’s `[B,H,T,T]` we use:
  - `off(b,h,i,j) = ((b*H + h)*T + i)*T + j`

## 1) Step shared by both: QKV projection

Both baseline and MHA do:

- `qkv = linear_lastdim(x, w_qkv, b_qkv)`

Shapes:
- `x` is `[1,3,4]`
- `w_qkv` is `[4, 12]` (because `3C = 12`)
- `qkv` is `[1,3,12]`

Then slice into:
- `q = qkv[..., 0:C]` → `[1,3,4]`
- `k = qkv[..., C:2C]` → `[1,3,4]`
- `v = qkv[..., 2C:3C]` → `[1,3,4]`

So far, baseline and MHA are identical.

## 2) Baseline: compute scores `[B,T,T]`

Baseline scores tensor:
- `scores` shape `[1,3,3]`

For each `i` (query position) and `j` (key position):

$$
S[i,j] = \frac{Q[i]\cdot K[j]}{\sqrt{C}} + \text{causalMask}(i,j)
$$

In code (`src/ops.cpp`), this is the loop:
- `qi = (bb*T + i)*C` is the base offset of `q[b,i,:]`
- `kj = (bb*T + j)*C` is the base offset of `k[b,j,:]`
- inner sum over `c=0..C-1`

### Example: compute `scores[0,2,1]`

Compute base offsets:
- `qi = (0*3 + 2)*4 = 8` → `q[0,2,:]` lives at `q.data[8..11]`
- `kj = (0*3 + 1)*4 = 4` → `k[0,1,:]` lives at `k.data[4..7]`

Dot product:

$$
q[0,2,:]\cdot k[0,1,:] = \sum_{c=0}^{3} q.data[8+c]\;k.data[4+c]
$$

Scale:
- multiply by `1/sqrt(C) = 1/sqrt(4) = 1/2`

Mask:
- since `j=1 <= i=2`, this is allowed (not masked)

Write location in `[B,T,T]` buffer:
- `off = (0*3 + 2)*3 + 1 = 7`
- so `scores.data[7] = scaledDot`

This is exactly how to read the baseline loop.

## 3) MHA variant: reshape and compute scores per head `[B,H,T,T]`

The MHA variant reshapes:

- `q4 = reshape(q, {B,T,H,D})` → `[1,3,2,2]`
- `k4 = reshape(k, {B,T,H,D})` → `[1,3,2,2]`
- `v4 = reshape(v, {B,T,H,D})` → `[1,3,2,2]`

Important: `reshape` in this repo is a **view** (shares the same underlying flat buffer).
So `q4.data` and `q.data` point to the same storage.

Then the variant computes:

$$
S_h[i,j] = \frac{Q_h[i]\cdot K_h[j]}{\sqrt{D}} + \text{causalMask}(i,j)
$$

The key differences vs baseline:
- dot product sums only over `d=0..D-1` (per-head dim)
- scaling uses `1/sqrt(D)`
- scores are computed separately for each head `h`

### Example: compute head 1 score `scores[0,h=1,i=2,j=1]`

For the chosen sizes `B=1,T=3,H=2,D=2`.

The variant uses base offsets (see code in `mha_attention.cpp`):

- `q_base = ((b*T + i)*H + h)*D`
- `k_base = ((b*T + j)*H + h)*D`

Plug in `b=0,i=2,j=1,h=1`:

- `q_base = ((0*3 + 2)*2 + 1)*2 = (5)*2 = 10`
- `k_base = ((0*3 + 1)*2 + 1)*2 = (3)*2 = 6`

So head-1 vectors are:
- `Q_{h=1}[i=2]` lives at `q4.data[10..11]`
- `K_{h=1}[j=1]` lives at `k4.data[6..7]`

Dot product over `D=2`:

$$
\text{dot} = q4.data[10]\cdot k4.data[6] + q4.data[11]\cdot k4.data[7]
$$

Scale:
- multiply by `1/sqrt(D) = 1/sqrt(2)`

Mask:
- `j=1 <= i=2` allowed

Write location in `[B,H,T,T]` buffer:

- `off = ((b*H + h)*T + i)*T + j`
- `off = ((0*2 + 1)*3 + 2)*3 + 1 = (5)*3 + 1 = 16`

So:
- `scores.data[16] = scaledDot`

Compare this to baseline:
- baseline `scores[0,2,1]` was stored in a *single* matrix at `scores.data[7]`
- MHA stores head-specific matrices; head-1’s `(i=2,j=1)` lives at a different offset.

## 4) Softmax and weighted sum (baseline vs MHA)

Baseline:
- `probs = softmax_lastdim(scores)` with shape `[B,T,T]`
- `att[b,i,c] = sum_j probs[b,i,j] * v[b,j,c]` produces `[B,T,C]`

MHA variant:
- `probs = softmax_lastdim(scores)` with shape `[B,H,T,T]`
- `att[b,i,h,d] = sum_j probs[b,h,i,j] * v4[b,j,h,d]` produces `[B,T,H,D]`
- then `reshape(att, {B,T,C})` concatenates heads

Interpretation:
- baseline gives **one** probability row `probs[b,i,:]`
- MHA gives **H** probability rows `probs[b,h,i,:]` (different attention patterns per head)

## 4.1) Numeric worked example (baseline, one query row)

Keep the same sizes as above: `B=1, T=3, C=4`.

We’ll compute the entire attention row for `b=0, i=2` (the 3rd token attending to tokens `j=0,1,2`).

Pick explicit vectors:

- Query at position `i=2`:
  - $Q[2] = [1,0,1,0]$
- Keys at positions $j\in\{0,1,2\}$:
  - $K[0] = [1,0,0,0]$
  - $K[1] = [0,1,0,0]$
  - $K[2] = [1,1,0,0]$

Compute scaled dot-products (baseline uses $1/\sqrt{C} = 1/2$):

$$
\begin{aligned}
S[2,0] &= \frac{Q[2]\cdot K[0]}{\sqrt{4}} = \tfrac{1}{2}(1\cdot 1 + 0\cdot 0 + 1\cdot 0 + 0\cdot 0) = 0.5\\
S[2,1] &= \frac{Q[2]\cdot K[1]}{\sqrt{4}} = \tfrac{1}{2}(1\cdot 0 + 0\cdot 1 + 1\cdot 0 + 0\cdot 0) = 0\\
S[2,2] &= \frac{Q[2]\cdot K[2]}{\sqrt{4}} = \tfrac{1}{2}(1\cdot 1 + 0\cdot 1 + 1\cdot 0 + 0\cdot 0) = 0.5
\end{aligned}
$$

Causal mask for `i=2` allows all `j<=2`, so the masked scores are:

$$
S[2,:] = [0.5,\;0,\;0.5]
$$

Softmax over $j$:

$$
\mathrm{softmax}(S[2,:])_j = \frac{e^{S[2,j]}}{\sum_{m=0}^{2} e^{S[2,m]}}
$$

Numerically (using $e^{0.5}\approx 1.6487$):

$$
\sum e^{S} = 1.6487 + 1 + 1.6487 = 4.2974
$$

So the attention probabilities are approximately:

$$
P[2,:] \approx [0.383,\;0.233,\;0.383]
$$

Now pick explicit values for $V[j]$ (still 4-wide vectors):

- $V[0] = [1,0,0,0]$
- $V[1] = [0,2,0,0]$
- $V[2] = [0,0,3,0]$

Compute the attended output at `i=2`:

$$
\mathrm{att}[2] = \sum_{j=0}^{2} P[2,j]\;V[j]
$$

Approximately:

$$
\mathrm{att}[2] \approx
0.383[1,0,0,0] +
0.233[0,2,0,0] +
0.383[0,0,3,0]
= [0.383,\;0.466,\;1.149,\;0]
$$

This is exactly what the baseline loop computes, just without writing down the intermediate `scores`/`probs` buffers explicitly.

## 4.2) Numeric worked example (MHA, one head)

For MHA we use `H=2` and `D=C/H=2`. MHA repeats the same process per head, but:

- it uses per-head vectors $Q_h[i], K_h[j], V_h[j] \in \mathbb{R}^{D}$
- it scales by $1/\sqrt{D} = 1/\sqrt{2}$

Here is a single-head row example for `h=1` at the same query position `i=2`:

- $Q_{1}[2] = [1,\,-1]$
- $K_{1}[0]=[1,\,0]$, $K_{1}[1]=[0,\,1]$, $K_{1}[2]=[1,\,1]$

Scaled scores ($1/\sqrt{2}\approx 0.7071$):

$$
\begin{aligned}
S_{1}[2,0] &= (1\cdot 1 + (-1)\cdot 0)\cdot 0.7071 \approx 0.7071\\
S_{1}[2,1] &= (1\cdot 0 + (-1)\cdot 1)\cdot 0.7071 \approx -0.7071\\
S_{1}[2,2] &= (1\cdot 1 + (-1)\cdot 1)\cdot 0.7071 = 0
\end{aligned}
$$

Softmax (approx):

$$
e^{0.7071}\approx 2.028,\; e^{-0.7071}\approx 0.493,\; e^{0}=1,\;\;\sum\approx 3.521
$$

$$
P_{1}[2,:] \approx [0.576,\;0.140,\;0.284]
$$

Pick per-head values:

- $V_{1}[0]=[10,\,0]$, $V_{1}[1]=[0,\,10]$, $V_{1}[2]=[5,\,5]$

Then the per-head attended result is:

$$
\mathrm{att}_{1}[2] = \sum_{j=0}^{2} P_{1}[2,j]\;V_{1}[j]
\approx 0.576[10,0] + 0.140[0,10] + 0.284[5,5]
\approx [7.18,\;2.82]
$$

Head `h=0` computes its own $\mathrm{att}_{0}[2]\in\mathbb{R}^{2}$ the same way (with different $Q_0,K_0,V_0$ slices). Concatenating the two heads gives a 4-wide vector again.

## 5) Where these steps live in code

This is the “search map” from the math/numeric examples above to the exact code paths.

Baseline (`nn::self_attention_1h` in `src/ops.cpp`):

- QKV projection: `qkv = linear_lastdim(x, w_qkv, b_qkv)`
- Q/K/V slices: `slice_lastdim_copy(qkv, 0, C)`, `slice_lastdim_copy(qkv, C, C)`, `slice_lastdim_copy(qkv, 2*C, C)`
- Scores (dot-product + scale + mask): the nested loops over `bb, i, j` compute
  - `qi = (bb*T + i)*C` and `kj = (bb*T + j)*C`
  - `s += q[qi+c] * k[kj+c]`, then `s *= 1/sqrt(C)`
  - causal mask: `if (j > i) s = -1e9f`
  - write: `scores[(bb*T + i)*T + j] = s`
- Softmax: `probs = softmax_lastdim(scores)` (same `softmax_lastdim` used everywhere)
- Weighted sum: the loops over `bb, i, c` compute
  - `sum_j probs[(bb*T + i)*T + j] * v[(bb*T + j)*C + c]`

MHA variant (`nn::variants::mha::self_attention_mha` in `src/variants/mha/mha_attention.cpp`):

- QKV projection: `qkv = nn::linear_lastdim(x, w_qkv, b_qkv)`
- Q/K/V slices: local `slice_lastdim_copy` helper (same semantics as core)
- Head split (view reshape): `q4 = nn::reshape(q, {B,T,H,D})` (and same for `k4`, `v4`)
- Scores per head: the nested loops over `bb, hh, i, j` compute
  - `q_base = ((bb*T + i)*H + hh)*D` and `k_base = ((bb*T + j)*H + hh)*D`
  - `s += q4[q_base+d] * k4[k_base+d]`, then `s *= 1/sqrt(D)`
  - causal mask: `if (j > i) s = -1e9f`
  - write: `scores[(((bb*H + hh)*T + i)*T + j)] = s`
- Softmax: `probs = nn::softmax_lastdim(scores)`
  - Note: `softmax_lastdim` always normalizes the **last** dimension, so for `[B,H,T,T]` it normalizes across `j` independently for each `(b,h,i)` row.
- Weighted sum per head: loops compute
  - `att[((bb*T + i)*H + hh)*D + d] = sum_j probs[p_off] * v4[v_off]`
- Concatenate heads: `att_cat = nn::reshape(att, {B,T,C})`

If you want a deeper indexing trace for a single score element, see `docs/attention_indexing_trace.md`.

## 6) What to try next while reading code

A good exercise is to set a breakpoint in the score loops and manually verify one element:

- pick `b=0, i=2, j=1`
- compute `qi/kj` (baseline) or `q_base/k_base` (MHA)
- verify the exact indices multiplied in the dot product
- verify the final write offset in `scores.data[...]`

Then repeat for a masked element where `j > i`.

---

This walkthrough is intentionally about shapes and offsets, but the numeric sections above are a good sanity check for how the score/softmax/weighted-sum pipeline behaves.
