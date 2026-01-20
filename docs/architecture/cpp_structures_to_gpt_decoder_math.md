> [简体中文](cpp_structures_to_gpt_decoder_math.zh-CN.md)

# C++ structures → GPT decoder math (base core)

This document is a **teaching-first map** from this repo’s C++ types (classes/structs, members, methods) to the standard decoder-only GPT math.

Scope:
- Baseline/core model path only (single-head causal attention).
- Focus on **what each member means** and **how methods compose** into the decoder forward pass.

Notation:
- $B$ = batch size
- $T$ = sequence length for a given run
- $T_{max}$ = configured maximum sequence length (`cfg.seq_len`)
- $V$ = vocab size (byte-level, typically 256)
- $C$ = model width / embedding dim (`d_model`)
- $L$ = number of transformer layers (`n_layers`)

All tensors are stored as flat `std::vector<float>` with explicit shapes.

## 1) The model object: `model::TinyGPT`

Defined in:
- `src/model.h` (`class TinyGPT`)
- `src/model.cpp` (implementation)

### 1.1 Configuration: `model::Config`

`model::Config` is the math “hyperparameter bundle”:

- `vocab_size` → $V$
- `seq_len` → $T_{max}$
- `d_model` → $C$
- `n_layers` → $L$

### 1.2 Parameters as members

In `TinyGPT`, learnable parameters are stored as `nn::Tensor` members.

Embeddings:
- `wte_` shape `[V,C]` → token embedding matrix $W_{te}\in\mathbb{R}^{V\times C}$
- `wpe_` shape `[T_max,C]` → positional embedding matrix $W_{pe}\in\mathbb{R}^{T_{max}\times C}$

Final LM head:
- `w_lm_` shape `[C,V]` → projection $W_{lm}\in\mathbb{R}^{C\times V}$
- `b_lm_` shape `[V]` → bias $b_{lm}\in\mathbb{R}^{V}$

Transformer blocks:
- `blocks_` is a vector of `Block` (length $L$)

Each `Block` contains (baseline, 1-head):

Attention (packed QKV):
- `w_qkv` shape `[C,3C]` → $W_{qkv}=[W_q\;W_k\;W_v]$ concatenated
- `b_qkv` shape `[3C]` → $b_{qkv}=[b_q\;b_k\;b_v]$ concatenated
- `w_proj` shape `[C,C]` → output projection $W_o$
- `b_proj` shape `[C]` → output bias $b_o$

MLP:
- `w_fc` shape `[C,4C]`, `b_fc` shape `[4C]` → first MLP linear
- `w_out` shape `[4C,C]`, `b_out` shape `[C]` → second MLP linear

This corresponds to the common GPT-style FFN width multiplier 4.

## 2) Decoder forward pass: `TinyGPT::forward_logits`

Implemented in `src/model.cpp`.

Given `tokens_bt` (token ids) with shape `[B,T]` (stored as a flat `std::vector<int32_t>`), the forward pass computes:

### 2.1 Token + positional embedding

Math:

$$
X = W_{te}[tokens] + W_{pe}[pos]
\quad\in\mathbb{R}^{B\times T\times C}
$$

Code mapping:
- `nn::embedding(wte_, tokens_bt, B, T)` → $W_{te}[tokens]$
- `add_positional(x, B, T)` → adds $W_{pe}$ row for each position

### 2.2 Pre-norm transformer blocks (repeat L times)

Each block is a decoder block with causal self-attention and an MLP, using **pre-layernorm**:

For layer $\ell=1..L$:

$$
\begin{aligned}
H &= \mathrm{LN}(X) \\
A &= \mathrm{CausalSelfAttn}(H) \\
X &= X + A \\
M &= \mathrm{LN}(X) \\
FF &= \mathrm{MLP}(M) \\
X &= X + FF
\end{aligned}
$$

Code mapping:
- `nn::layernorm_lastdim(x, 1e-5f)` → `LN`
- `nn::self_attention_1h(h, blk.w_qkv, blk.b_qkv, blk.w_proj, blk.b_proj)` → causal attention (1 head)
- `nn::add(x, a)` → residual
- `nn::linear_lastdim(...), nn::gelu(...), nn::linear_lastdim(...)` → MLP

### 2.3 Final norm + LM head

Math:

$$
logits = \mathrm{LN}(X) W_{lm} + b_{lm}
\quad\in\mathbb{R}^{B\times T\times V}
$$

Code mapping:
- `nn::layernorm_lastdim(x, 1e-5f)`
- `nn::linear_lastdim(xn, w_lm_, b_lm_)`

## 3) Causal self-attention (baseline): `nn::self_attention_1h`

Declared in `src/ops.h`, implemented in `src/ops.cpp`.

Input/weights:
- `x`: `[B,T,C]`
- `w_qkv`: `[C,3C]`, `b_qkv`: `[3C]`
- `w_proj`: `[C,C]`, `b_proj`: `[C]`

### 3.1 QKV projection

Math:

$$
[Q,K,V] = X W_{qkv} + b_{qkv}
$$

Code mapping:
- `qkv = nn::linear_lastdim(x, w_qkv, b_qkv)` → shape `[B,T,3C]`
- `q,k,v` extracted by slicing last dimension into three `[B,T,C]` tensors

### 3.2 Attention scores + causal mask

For each batch $b$, query position $i$, key position $j$:

$$
S[b,i,j] = \frac{Q[b,i]\cdot K[b,j]}{\sqrt{C}} + \text{mask}(i,j)
$$

where causal mask is:

$$
\text{mask}(i,j) = \begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

Code mapping:
- explicit loops compute dot products and write `scores[b,i,j]`
- masking is `if (j > i) s = -1e9f;`

### 3.3 Softmax

$$
P[b,i,:] = \mathrm{softmax}(S[b,i,:])
$$

Code mapping:
- `probs = nn::softmax_lastdim(scores)`

### 3.4 Weighted sum

$$
Y[b,i] = \sum_{j=0}^{T-1} P[b,i,j]\,V[b,j]
$$

Code mapping:
- explicit loops produce `att` with shape `[B,T,C]`

### 3.5 Output projection

$$
A = Y W_o + b_o
$$

Code mapping:
- `nn::linear_lastdim(att, w_proj, b_proj)`

## 4) Loss: `TinyGPT::loss` + `nn::cross_entropy`

`TinyGPT::loss(...)` calls:
- `forward_logits(...)` → logits `[B,T,V]`
- reshape to `[N,V]` where $N=B\cdot T$
- `nn::cross_entropy(logits_nv, targets_n)`

Math (mean negative log likelihood):

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} -\log\Big(\mathrm{softmax}(logits_n)[y_n]\Big)
$$

## 5) Training loop: `src/main.cpp`

The CLI executable `train_gpt` is the simplest “trainer + sampler”. It does:

1) Parse args (batch/seq/dmodel/layers/lr/seed)
2) Load data bytes into `data::ByteDataset`
3) Construct model + optimizer
4) For each step:
   - sample a batch (`ByteDataset::sample_batch`)
   - compute loss (`TinyGPT::loss`)
   - `loss.backward()` (autograd)
   - `opt.step(params)` (AdamW)

Then optionally:
- save checkpoint (`ckpt::save`)
- generate from a prompt (`generate(...)`)

## 6) Autograd objects: `nn::Tensor` + `nn::Node`

Defined in `src/tensor.h` / `src/tensor.cpp`.

Teaching mental model:
- Each op in `src/ops.cpp` creates an output `Tensor out`.
- If gradients are enabled, `out` gets a `node` that stores:
  - `parents` (the input tensors)
  - `backward` lambda that accumulates into `parent.grad`

So the math “backprop graph” is literally built as a chain of `Node` lambdas.

## 7) Suggested reading order (practice)

If you want to learn by tracing end-to-end:

1) `TinyGPT::forward_logits` (decoder forward wiring)
2) `nn::self_attention_1h` (causal attention)
3) `nn::cross_entropy` (loss)
4) `Tensor::backward` / node traversal (autograd engine)
5) `optim::AdamW::step` (parameter update)

Then run a tiny training loop from the CLI and set breakpoints in that order.
