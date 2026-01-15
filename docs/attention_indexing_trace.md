# Attention indexing trace (equation → loop → flat offset)

This doc is a **line-for-line bridge** between the attention equations and the C++ loops.

It answers questions like:
- “Where does `scores[b,i,j]` live in memory?”
- “Which exact indices are multiplied for $Q_i\cdot K_j$?”
- “Where does the causal mask happen?”

Relevant docs and code:
- Memory/indexing formulas: `docs/autograd_and_memory_layout.md`
- Attention equations: `docs/transformer_math_to_code.md`
- Implementation: `nn::self_attention_1h(...)` in `src/ops.cpp`


## 0) Setup: tiny shapes

Use a tiny configuration so you can compute offsets on paper:

- Batch: $B=1$
- Sequence length: $T=3$
- Model width: $C=2$

We will trace a single scalar:

- `scores[b,i,j]` with `b=0, i=1, j=0`


## 1) The paper equation we’re implementing

Single-head scaled dot-product attention score:

$$S_{b,i,j}=\frac{Q_{b,i,:}\cdot K_{b,j,:}}{\sqrt{C}}$$

Causal mask (decoder-only):
- If $j>i$, set $S_{b,i,j}=-\infty$ (implemented as a large negative number).


## 2) Row-major indexing formulas (exact)

From `docs/autograd_and_memory_layout.md`.

### 2.1 For shape `[B,T,C]`

Flat offset for `(b,t,c)`:

$$\mathrm{off}_{BTC}(b,t,c)=(b\cdot T + t)\cdot C + c$$

### 2.2 For shape `[B,T,T]`

Flat offset for `(b,i,j)`:

$$\mathrm{off}_{BTT}(b,i,j)=(b\cdot T + i)\cdot T + j$$


## 3) Where Q and K come from in this repo

Inside `nn::self_attention_1h(x, w_qkv, b_qkv, w_proj, b_proj)`:

1) `qkv = linear_lastdim(x, w_qkv, b_qkv)`
- `qkv` shape: `[B,T,3C]`

2) slice last dimension:
- `q = slice_lastdim_copy(qkv, 0, C)` → shape `[B,T,C]`
- `k = slice_lastdim_copy(qkv, C, C)` → shape `[B,T,C]`
- `v = slice_lastdim_copy(qkv, 2*C, C)` → shape `[B,T,C]`

So the dot product for the score always reads from the flat buffers of `q` and `k` using `[B,T,C]` indexing.


## 4) Trace one element: `scores[0,1,0]`

### 4.1 Flat location of `scores[0,1,0]`

`scores` has shape `[B,T,T] = [1,3,3]`.

Compute its flat offset:

$$\mathrm{off}_{BTT}(0,1,0)=(0\cdot 3 + 1)\cdot 3 + 0 = 3$$

So:
- `scores[0,1,0]` is stored at `scores.data[3]`.

This matches the implementation pattern in the code:
- it writes into `scores.data[((bb*T + i)*T + j)]`.

### 4.2 Flat base of `q[0,1,:]`

`q` has shape `[B,T,C]=[1,3,2]`.

Base offset of the contiguous vector `q[0,1,:]`:

$$qBase = \mathrm{off}_{BTC}(0,1,0)=(0\cdot 3 + 1)\cdot 2 + 0 = 2$$

So:
- `q[0,1,0]` is `q.data[2]`
- `q[0,1,1]` is `q.data[3]`

### 4.3 Flat base of `k[0,0,:]`

Base offset of `k[0,0,:]`:

$$kBase = \mathrm{off}_{BTC}(0,0,0)=(0\cdot 3 + 0)\cdot 2 + 0 = 0$$

So:
- `k[0,0,0]` is `k.data[0]`
- `k[0,0,1]` is `k.data[1]`

### 4.4 Dot product term-by-term (this is the inner loop)

Paper:

$$Q_{0,1,:}\cdot K_{0,0,:} = Q_{0,1,0}K_{0,0,0} + Q_{0,1,1}K_{0,0,1}$$

Flat-buffer form:

$$= q.data[qBase+0]\cdot k.data[kBase+0] + q.data[qBase+1]\cdot k.data[kBase+1]$$

With `qBase=2`, `kBase=0`:

$$= q.data[2]\cdot k.data[0] + q.data[3]\cdot k.data[1]$$

This corresponds directly to the code’s inner loop:
- it computes `qi = (bb*T + i)*C` and `kj = (bb*T + j)*C`
- then sums `q.data[qi + c] * k.data[kj + c]` for `c=0..C-1`.

For `(bb=0, i=1, j=0)`:
- `qi = (0*3 + 1)*2 = 2` (same as `qBase`)
- `kj = (0*3 + 0)*2 = 0` (same as `kBase`)

### 4.5 Apply the scale and mask

Scale:

$$S_{0,1,0} = (\text{dot})\cdot \frac{1}{\sqrt{C}} = (\text{dot})\cdot \frac{1}{\sqrt{2}}$$

Causal mask check:
- here `j=0` and `i=1`, so `j>i` is false.
- the value is allowed and is written normally.

Finally:
- `scores.data[3] = scaledDot`.


## 5) Trace a masked element: `scores[0,1,2]`

Compute flat offset:

$$\mathrm{off}_{BTT}(0,1,2)=(0\cdot 3 + 1)\cdot 3 + 2 = 5$$

Because `j=2 > i=1`, the implementation overwrites the score with a large negative number:
- `scores.data[5] = -1e9` (approx “$-\infty$”).

This is how decoder-only causal masking is implemented here.


## 6) (Optional next traces) probs and weighted sum

Once `scores` is computed:

1) `probs = softmax_lastdim(scores)`
- `probs` shape is also `[B,T,T]`
- for a fixed `(b,i)`, the row `probs[b,i,:]` is contiguous in memory.

2) `att[b,i,c] = sum_j probs[b,i,j] * v[b,j,c]`
- `att` shape is `[B,T,C]`

These follow the same offset rules:
- `probs` uses `off_BTT(b,i,j)`
- `v` and `att` use `off_BTC(b,t,c)`

If you want to continue the same exercise, a good next target is tracing:
- `att[0,1,0]` (one channel of one output token)


## References (paper names)

- “Attention Is All You Need” (scaled dot-product attention)
- GPT-style decoder-only masking (causal attention)

(We reference paper names for orientation; this document is an original explanation of this repo’s implementation.)
