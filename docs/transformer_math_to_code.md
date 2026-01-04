# Transformer math → this code

Goal: make it easy to answer “where is *that* equation implemented?”

This repo implements a *tiny GPT-style decoder-only Transformer* with:
- byte-level vocab (V=256)
- learned positional embeddings
- pre-norm blocks
- **single-head** causal self-attention (for now)

## Notation (shapes)

- Batch size: `B`
- Sequence length: `T`
- Model width: `C` (`d_model`)
- Vocabulary size: `V`

Tensors are stored **row-major** in flat arrays.

Common shapes:
- tokens: `[B,T]` (stored as `std::vector<int32_t>` length `B*T`)
- hidden states: `X` is `[B,T,C]`
- logits: `Z` is `[B,T,V]`


## 1) Token + position embeddings

Math (learned embeddings):

- Token embedding table: $W_{te} \in \mathbb{R}^{V \times C}$
- Position embedding table: $W_{pe} \in \mathbb{R}^{T_{max} \times C}$

For token ids $t_{b,i}$ at batch $b$ and position $i$:

$$E_{b,i,:} = W_{te}[t_{b,i},:]$$
$$X_{b,i,:} = E_{b,i,:} + W_{pe}[i,:]$$

Where in code:
- Token embedding lookup: `nn::embedding(...)` in `src/ops.cpp`
- Add position embeddings: `model::TinyGPT::add_positional(...)` in `src/model.cpp`


## 2) Transformer block (pre-norm, decoder-only)

This repo uses **pre-norm** residual blocks (LayerNorm before the sublayer):

For input $X$:

### 2.1) Attention sublayer

$$H = \mathrm{LN}(X)$$
$$A = \mathrm{CausalSelfAttn}(H)$$
$$X \leftarrow X + A$$

Where in code:
- LN: `nn::layernorm_lastdim(...)` in `src/ops.cpp`
- Attention: `nn::self_attention_1h(...)` in `src/ops.cpp`
- Residual add: `nn::add(...)` in `src/ops.cpp`

### 2.2) Feed-forward (MLP) sublayer

This is a 2-layer MLP with GELU:

$$M = \mathrm{LN}(X)$$
$$F = \mathrm{GELU}(M W_{fc} + b_{fc})$$
$$O = F W_{out} + b_{out}$$
$$X \leftarrow X + O$$

Where in code:
- `nn::linear_lastdim(...)` implements $XW + b$ in `src/ops.cpp`
- `nn::gelu(...)` in `src/ops.cpp`
- Block wiring: `model::TinyGPT::forward_logits(...)` in `src/model.cpp`


## 3) Causal self-attention (single head)

For a single head, with $C$-dim vectors:

- Projection weights:
  - $W_{qkv} \in \mathbb{R}^{C \times 3C}$ and bias $b_{qkv} \in \mathbb{R}^{3C}$
  - $W_{proj} \in \mathbb{R}^{C \times C}$ and bias $b_{proj} \in \mathbb{R}^{C}$

Compute:

$$[Q, K, V] = H W_{qkv} + b_{qkv}$$

Scaled dot-product scores with causal mask:

$$S_{b,i,j} = \frac{Q_{b,i,:} \cdot K_{b,j,:}}{\sqrt{C}} + \begin{cases}0 & j \le i \\ -\infty & j > i\end{cases}$$

Softmax over last dimension (keys):

$$P_{b,i,:} = \mathrm{softmax}(S_{b,i,:})$$

Weighted sum:

$$Y_{b,i,:} = \sum_{j=0}^{T-1} P_{b,i,j} V_{b,j,:}$$

Output projection:

$$A = Y W_{proj} + b_{proj}$$

Where in code:
- `nn::self_attention_1h(...)` in `src/ops.cpp`.
  - Q/K/V slicing is done via `slice_lastdim_copy(...)`.
  - Causal mask is implemented by setting `-1e9` for `j > i`.
  - Softmax: `nn::softmax_lastdim(...)`.


## 4) Final normalization + LM head

After the last block:

$$X_n = \mathrm{LN}(X)$$
$$Z = X_n W_{lm} + b_{lm}$$

Where in code:
- Final LN + logits: `model::TinyGPT::forward_logits(...)` in `src/model.cpp`


## 5) Cross entropy (next-token prediction)

We flatten `[B,T,V]` into `[N,V]` where $N=B\cdot T$.

For each row $n$ with target class $y_n$:

$$p_{n,v} = \mathrm{softmax}(z_{n,:})_v$$
$$\ell_n = -\log(p_{n,y_n})$$
$$L = \frac{1}{N} \sum_{n=0}^{N-1} \ell_n$$

Where in code:
- `nn::cross_entropy(...)` in `src/ops.cpp`


## 6) AdamW optimizer update

This repo implements AdamW in the common “decoupled weight decay” style.

Given gradient $g_t$ and parameters $\theta_t$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

Bias correction:

$$\hat m_t = \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

Update:

$$\theta \leftarrow \theta - \alpha \left( \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta \right)$$

Where in code:
- `optim::AdamW::step(...)` in `src/optim.cpp`


## References (papers)

- “Attention Is All You Need” (Transformer)
- “Language Models are Unsupervised Multitask Learners” (GPT-2)

(We reference paper names for orientation; this document is an original mapping to this repo’s code.)
