> [简体中文](why_multi_head_attention.zh-CN.md)

# Why multi-head attention (MHA)? (beyond parallelism)

This is a teaching note answering:
- “If I have enough compute/memory, isn’t a single head enough?”
- “What does MHA buy us in practice?”

We tie the intuition to this repo’s baseline implementation and our MHA variant.

Relevant code:
- Baseline (single head): `nn::self_attention_1h` in `src/ops.cpp`
- Variant (multi head): `nn::variants::mha::self_attention_mha` in `src/variants/mha/mha_attention.cpp`

Paper name for orientation:
- “Attention Is All You Need” (scaled dot-product attention + multi-head attention)

## 1) What single-head attention can express in one layer

In this repo’s baseline, for each query token position `i` we compute one attention distribution:

- scores: `S[b,i,j]` with shape `[B,T,T]`
- probs:  `P[b,i,j]` with shape `[B,T,T]`

Math (single head):

$$
S[i,j] = \frac{Q[i]\cdot K[j]}{\sqrt{C}} + \text{causalMask}(i,j)
$$

$$
P[i,:] = \mathrm{softmax}(S[i,:])
$$

$$
Y[i] = \sum_j P[i,j] V[j]
$$

Key limitation:
- For each `i`, there is **exactly one** row vector `P[i,:]`.
- That means the token `i` can “look at the sequence” in **one way** at that layer.

Even if you increase `C` (more channels), you still only produce one distribution `P[i,:]` per token per layer.

## 2) What MHA adds: multiple simultaneous attention patterns

Multi-head attention splits the channel dimension:

- choose `H` heads
- per-head dim `D`
- require $C = H\cdot D$

Then for each head $h$ you compute its own attention distribution:

$$
S_h[i,j] = \frac{Q_h[i]\cdot K_h[j]}{\sqrt{D}} + \text{causalMask}(i,j)
$$

$$
P_h[i,:] = \mathrm{softmax}(S_h[i,:])
$$

$$
Y_h[i] = \sum_j P_h[i,j] V_h[j]
$$

Then concatenate heads:

$$
Y[i] = \mathrm{concat}_h(Y_h[i]) \in \mathbb{R}^{C}
$$

And project:

$$
\mathrm{out} = Y W_{proj} + b_{proj}
$$

Interpretation:
- With $H$ heads, token `i` can have **H different attention distributions** over the same context.
- Those heads can specialize (e.g., one head tracks recent tokens, another tracks punctuation, another tracks subject/object, etc.).

This is representational power, not just compute parallelism.

## 3) Why “one head with bigger C” is not equivalent

A common intuition is: “If I have more resources, I can just make `C` larger.”

But the core bottleneck is the *number of attention maps per layer*:
- single head: 1 map `P[i,:]`
- multi head: $H$ maps `P_h[i,:]`

If the model needs to attend strongly to multiple different places for different purposes, a single map must compromise by mixing probability mass.

MHA lets those different “reasons to attend” exist as separate heads in the same layer.

## 4) How this shows up in our code shapes

Baseline (`self_attention_1h`):
- Q,K,V are `[B,T,C]`
- scores/probs are `[B,T,T]`

MHA variant (`self_attention_mha`):
- Q,K,V are reshaped to `[B,T,H,D]`
- scores/probs are conceptually `[B,H,T,T]`

So the extra `H` dimension corresponds directly to “multiple attention maps”.

Also note the scaling:
- baseline uses $1/\sqrt{C}$
- MHA uses $1/\sqrt{D}$

This keeps score magnitudes in a similar range even though each head uses fewer channels.

## 5) Training dynamics (practical benefit)

MHA often trains more reliably because heads can form distinct gradient pathways:
- each head has its own scores/softmax outputs
- each head produces its own weighted sum before concatenation

In our variant, that is reflected by explicit `[B,H,T,T]` score/prob buffers and per-head loops.

## 6) When single-head is fine

- For educational baselines, small tasks, or very small models, single-head is totally fine.
- This repo keeps the baseline single-head path because it’s easier to read and verify.

When you want better quality per layer for realistic sizes/contexts, MHA is a standard building block.
