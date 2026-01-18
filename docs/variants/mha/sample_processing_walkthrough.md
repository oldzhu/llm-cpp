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

## 5) What to try next while reading code

A good exercise is to set a breakpoint in the score loops and manually verify one element:

- pick `b=0, i=2, j=1`
- compute `qi/kj` (baseline) or `q_base/k_base` (MHA)
- verify the exact indices multiplied in the dot product
- verify the final write offset in `scores.data[...]`

Then repeat for a masked element where `j > i`.

---

This walkthrough is intentionally about shapes and offsets. If you want, we can add a follow-up note that assigns explicit numeric values to `q` and `k` and computes a real score number end-to-end (including softmax) for the same tiny sizes.
