# MHA extension: what changed vs the core baseline (and why)

This doc explains what the repo’s **multi-head attention (MHA) extension** added, what it intentionally did *not* change, and how the implementation maps to the standard transformer equations.

It is written for learning from the C++ structures/modules perspective.

## 0) Big picture

Baseline/core:
- The runnable model in `model::TinyGPT` uses **single-head causal self-attention**.
- That path is implemented by `nn::self_attention_1h` in `src/ops.cpp`.

MHA extension (this repo):
- We added MHA as a **side-by-side variant module** under `src/variants/...`.
- We did **not** rewrite the baseline model.

This keeps the core easy to study, while still giving you a working template for “new feature = code + docs + test”.

## 1) What we changed in C++ structures/classes

### 1.1 Core model types: unchanged

No changes were required to:
- `model::Config`
- `model::TinyGPT`
- `TinyGPT::Block`

So the core model’s members remain:
- token embedding `wte_` and positional embedding `wpe_`
- per-block attention weights: `w_qkv`, `b_qkv`, `w_proj`, `b_proj`
- per-block MLP weights: `w_fc`, `b_fc`, `w_out`, `b_out`
- final LM head: `w_lm_`, `b_lm_`

### 1.2 New variant API: added

We added a new function (not a new model class):
- `nn::variants::mha::self_attention_mha(...)`

Files:
- `src/variants/mha/mha_attention.h`
- `src/variants/mha/mha_attention.cpp`

Signature (conceptually):
- same inputs/weights as the baseline attention
- plus one extra integer: `n_heads`

## 2) What we changed in parameters / member shapes

Intentionally **nothing** (for the first template).

Both the baseline and MHA variant use the same parameter layout:
- `w_qkv: [C, 3C]`, `b_qkv: [3C]`
- `w_proj: [C, C]`, `b_proj: [C]`

This is realistic: many production implementations store QKV and output projection this way.

The difference is not the parameter shapes—it’s the **interpretation of the channel dimension**.

## 3) What changed in method implementation (math → code)

### 3.1 Baseline single-head attention (core)

The baseline score is:

$$
S[i,j] = \frac{Q[i]\cdot K[j]}{\sqrt{C}} + \text{causalMask}(i,j)
$$

$$
P[i,:] = \mathrm{softmax}(S[i,:])
$$

$$
Y[i] = \sum_j P[i,j] V[j]
$$

$$
\mathrm{out} = Y W_{proj} + b_{proj}
$$

Implemented in:
- `nn::self_attention_1h` (`src/ops.cpp`)

### 3.2 Multi-head attention (variant)

Multi-head attention splits the channel dimension into heads.

Choose:
- number of heads $H$ (called `n_heads`)
- per-head dimension $D$

Require:

$$
C = H\cdot D
$$

After the QKV projection (which still produces `[B,T,C]` for each of Q,K,V), we reshape:

- $Q, K, V$ from `[B,T,C]` into `[B,T,H,D]`

Then attention is computed independently per head:

$$
S_h[i,j] = \frac{Q_h[i]\cdot K_h[j]}{\sqrt{D}} + \text{causalMask}(i,j)
$$

$$
P_h[i,:] = \mathrm{softmax}(S_h[i,:])
$$

$$
Y_h[i] = \sum_j P_h[i,j] V_h[j]
$$

Finally concatenate heads back to model width and project:

$$
Y = \mathrm{concat}_h(Y_h) \in \mathbb{R}^{B\times T\times C}
$$

$$
\mathrm{out} = Y W_{proj} + b_{proj}
$$

Implemented in:
- `nn::variants::mha::self_attention_mha` (`src/variants/mha/mha_attention.cpp`)

Key differences vs baseline:
- scale uses $\sqrt{D}$ instead of $\sqrt{C}$
- scores/probs are conceptually `[B,H,T,T]` (one matrix per head)
- output is `[B,T,H,D]` then reshaped to `[B,T,C]` before projection

## 4) What changed in build targets (CMake)

We refactored the build into composable targets:

- `llm_core` (library)
  - baseline implementation: tensor/ops/model/optim/checkpoint/data/util

- `llm_variant_mha` (library)
  - compiles `src/variants/mha/mha_attention.cpp`
  - links against `llm_core`

- `train_gpt` (executable)
  - links only `llm_core`
  - so it still runs the baseline attention by default

- `test_build_llm` (executable)
  - links `llm_core` + `llm_variant_mha`

This makes variants easy to add without destabilizing the baseline.

## 5) What changed in tests

We added one focused test proving that the variant is consistent with the baseline when `n_heads=1`:

- In `tests/test_main.cpp`: `test_mha_matches_1h_when_single_head()`
  - forward output matches `nn::self_attention_1h`
  - backward gradients match via a scalar cross-entropy loss

This is a strong “sanity anchor”: MHA reduces to the single-head formula when $H=1$.

## 6) Paper mapping (names only)

- “Attention Is All You Need” — scaled dot-product attention + multi-head attention
- GPT-style decoder-only masking — causal mask uses the rule `j > i` is masked

(We reference paper names for orientation; this doc is an original explanation of this repo’s code.)

## 7) What’s *next* if you want MHA in the runnable model

Right now, `train_gpt` is intentionally baseline-only.

Two clean ways to enable MHA in an executable:

1) Add a config field `n_heads` to `model::Config` (default 1), add CLI flag `--heads`, and in the block call:
   - baseline when `n_heads == 1`
   - MHA variant when `n_heads > 1`

2) Create a separate executable (e.g. `train_gpt_mha`) that links `llm_variant_mha` and uses MHA unconditionally.

Option (1) keeps one binary and preserves baseline behavior by default.
Option (2) keeps an even cleaner separation for learning.
