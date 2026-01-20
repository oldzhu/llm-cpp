> [简体中文](learn_by_hand_walkthrough.zh-CN.md)

# Learn-by-hand walkthrough (paper math → this repo)

This document is meant to be *worked through on paper*.

It combines:
- `docs/training_and_inference_walkthrough.md` (runtime flow)
- `docs/transformer_math_to_code.md` (equations → code)
- `docs/autograd_and_memory_layout.md` (indexing + autograd mental model)

The goal: you can simulate a tiny GPT forward + loss in your head and understand what each loop in C++ is doing.


## 0) Tiny configuration for hand simulation

The real repo typically uses:
- vocab $V=256$ (bytes)
- hidden width $C=64+$
- sequence length $T=64+$

For learning-by-hand, we shrink it:

- Batch $B=1$
- Sequence length $T=3$
- Model width $C=2$
- Vocabulary size $V=4$ (toy)

You can still follow the exact same steps as the real code; only shapes differ.


## 1) One training step (what main.cpp does)

One iteration of training conceptually is:

1) Sample a batch `(x,y)`
2) `logits = forward_logits(x)`
3) `loss = cross_entropy(logits, y)`
4) `loss.backward()`
5) `opt.step(params)`

Where in code (names):
- Dataset batch: `data::ByteDataset::sample_batch` (src/data.cpp)
- Forward: `model::TinyGPT::forward_logits` (src/model.cpp)
- Loss: `model::TinyGPT::loss` → `nn::cross_entropy` (src/model.cpp, src/ops.cpp)
- Backward: `nn::Tensor::backward` (src/tensor.cpp)
- Optimizer: `optim::AdamW::step` (src/optim.cpp)


## 2) Dataset: next-token prediction pairs

In this repo (byte-level), a dataset is an array of bytes.

If we sample a substring of length $T+1$:

- Input tokens: $x_t = \text{bytes}[s+t]$ for $t=0..T-1$
- Targets: $y_t = \text{bytes}[s+t+1]$ (shifted by one)

So:

$$y_t = x_{t+1}$$

Hand example with toy vocab $0..3$:
- sampled bytes (length 4): `[2,1,3,0]`
- then $x=[2,1,3]$ and $y=[1,3,0]$


## 3) Embeddings: tokens → vectors

Parameters:
- Token embedding table $W_{te} \in \mathbb{R}^{V\times C}$
- Positional embedding table $W_{pe} \in \mathbb{R}^{T_{max}\times C}$

Math:

$$E_{t,:} = W_{te}[x_t,:]$$
$$X_{t,:} = E_{t,:} + W_{pe}[t,:]$$

Where in code:
- `nn::embedding(...)` in src/ops.cpp
- `model::TinyGPT::add_positional(...)` in src/model.cpp

### Hand example

Let $C=2, V=4$.

Token embeddings $W_{te}$ (rows are tokens 0..3):
- token0: $[0.0, 0.0]$
- token1: $[1.0, 0.0]$
- token2: $[0.0, 1.0]$
- token3: $[1.0, 1.0]$

Positional embeddings $W_{pe}$ for positions 0..2:
- pos0: $[0.1, 0.0]$
- pos1: $[0.0, 0.1]$
- pos2: $[0.1, 0.1]$

Input $x=[2,1,3]$ gives:
- $X_0 = W_{te}[2] + W_{pe}[0] = [0,1] + [0.1,0] = [0.1, 1.0]$
- $X_1 = W_{te}[1] + W_{pe}[1] = [1,0] + [0,0.1] = [1.0, 0.1]$
- $X_2 = W_{te}[3] + W_{pe}[2] = [1,1] + [0.1,0.1] = [1.1, 1.1]$


## 4) Transformer block (pre-norm)

This repo uses pre-norm blocks.

### 4.1 LayerNorm (no affine)

For a vector $x\in\mathbb{R}^{C}$:

$$\mu=\frac{1}{C}\sum_i x_i, \quad \sigma^2=\frac{1}{C}\sum_i (x_i-\mu)^2$$
$$\mathrm{LN}(x)_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}$$

Where in code:
- `nn::layernorm_lastdim(...)` in src/ops.cpp

Hand tip: when $C=2$, this is quick.
Example for $X_0=[0.1,1.0]$:
- mean $\mu=0.55$
- deviations $[-0.45,+0.45]$
- variance $0.2025$, std $0.45$
- LN $\approx [-1,+1]$


### 4.2 Causal self-attention (1 head)

Math (single head):

$$[Q,K,V] = H W_{qkv} + b_{qkv}$$

Scores:

$$S_{i,j} = \frac{Q_i\cdot K_j}{\sqrt{C}}$$

Causal mask (decoder-only): if $j>i$ set $S_{i,j}=-\infty$.

Probabilities:

$$P_{i,:} = \mathrm{softmax}(S_{i,:})$$

Weighted sum:

$$Y_i = \sum_j P_{i,j} V_j$$

Output projection:

$$A = Y W_{proj} + b_{proj}$$

Where in code:
- `nn::self_attention_1h(...)` in src/ops.cpp

#### Hand example (simplified for intuition)

To keep the arithmetic small, pretend $Q=K=V=H$ (not true in the actual model).

Let (after LN) we have:
- $H_0=[-1,+1]$
- $H_1=[+1,-1]$

Scale is $1/\sqrt{2}\approx 0.707$.

At position $i=1$, allowed keys are $j=0,1$.

Compute scores:
- $S_{1,0}=0.707*(H_1\cdot H_0)=0.707*((1*-1)+(-1*1))=0.707*(-2)=-1.414$
- $S_{1,1}=0.707*(H_1\cdot H_1)=0.707*(2)=+1.414$

Softmax:
- $\exp(-1.414)\approx 0.243$
- $\exp(1.414)\approx 4.113$
- sum $\approx 4.356$

So:
- $P_{1,0}\approx 0.056$
- $P_{1,1}\approx 0.944$

Weighted sum:

$$Y_1 \approx 0.056 H_0 + 0.944 H_1 \approx [0.888,-0.888]$$

Interpretation:
- Token 1 mostly attends to itself.
- The causal mask guarantees it cannot attend to the future.


### 4.3 Residual connections

After attention:

$$X \leftarrow X + A$$

After MLP:

$$X \leftarrow X + \mathrm{MLP}(\mathrm{LN}(X))$$

Where in code:
- `nn::add(...)` in src/ops.cpp
- Block wiring in `model::TinyGPT::forward_logits(...)` in src/model.cpp


## 5) LM head: hidden → logits

After the last block:

$$Z = \mathrm{LN}(X) W_{lm} + b_{lm}$$

`Z` has shape `[B,T,V]`.

Where in code:
- `model::TinyGPT::forward_logits(...)` uses `nn::linear_lastdim(...)`


## 6) Cross-entropy loss (next-token prediction)

Flatten `[B,T,V]` into `[N,V]` where $N=B\cdot T$.

For logits row $z$ and target class $y$:
- $p=\mathrm{softmax}(z)$
- loss $=-\log p_y$

Mean over all rows:

$$L=\frac{1}{N}\sum_n -\log p_{n,y_n}$$

Where in code:
- `nn::cross_entropy(...)` in src/ops.cpp

### Hand example

Let one row logits be $z=[2,1,0,-1]$ and target $y=1$.

Compute exp (approximately):
- $e^2\approx 7.39$, $e^1\approx 2.72$, $e^0=1$, $e^{-1}\approx 0.368$
- sum $\approx 11.48$

Probability of target:
- $p_1 \approx 2.72/11.48 \approx 0.237$

Loss:
- $-\log(0.237) \approx 1.44$

This is the scalar printed as `loss=...` during training.


## 7) Generation (sampling) loop by hand

At each generation step:

1) Take the last up-to-`seq_len` tokens as context
2) Compute logits at the last position
3) Sample next token using temperature and optional top-k

Temperature:
- Divide logits by `temp` before softmax.
- Higher temperature → flatter distribution.

Top-k:
- Keep only the top K logits, sample among them.

Where in code:
- `generate(...)` and sampling helpers in src/main.cpp


## 8) Mini-exercises (do these on paper)

1) Softmax + CE:
- Choose 4 logits and a target class; compute softmax and CE.

2) Causal masking:
- For position $i=0$, confirm only $j=0$ is allowed.

3) Attention weights:
- Pick 2 vectors for $Q_1$ and $K_0,K_1$; compute scores, softmax, and the weighted sum.


## 9) Next features and docs

When we add multi-head attention and KV-cache, we will:
- add a dedicated doc for each feature
- update `docs/transformer_math_to_code.md`
- add targeted code comments showing shape changes and formulas
- extend tests so refactors don’t silently break correctness
