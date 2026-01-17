# Hidden states `[B,T,C]` vs attention matrices `[B,T,T]` (with flat-buffer offsets)

This note is a quick reference for two very common tensor shapes in the decoder:

- **Hidden states**: `X[b,t,c]` with shape `[B,T,C]`
- **Attention scores/probs**: `S[b,i,j]` / `P[b,i,j]` with shape `[B,T,T]`

The goal is to make the indexing feel “obvious” before you read the nested loops in `nn::self_attention_1h`.

Notation:
- `b` = batch index
- `t` = token position (time)
- `c` = channel (feature dimension)
- `i` = query position
- `j` = key position

All tensors are stored in row-major order (last dimension is contiguous).

## 1) Hidden states: `[B,T,C]`

Meaning:
- For each batch element `b`
- for each token position `t`
- store a **vector** of length `C` (the token embedding / hidden state)

Index:
- `X[b,t,c]`

Flat offset:

$$
\text{offset}(b,t,c) = (b\cdot T + t)\cdot C + c
$$

### Example A (easy numbers)

Let `B=1`, `T=3`, `C=4`.

`X` contains `1*3*4 = 12` floats.

- Token 0 vector is `X[0,0,:]` and occupies offsets `0..3`
- Token 1 vector is `X[0,1,:]` and occupies offsets `4..7`
- Token 2 vector is `X[0,2,:]` and occupies offsets `8..11`

Concrete offsets:
- `X[0,1,2]` → `(0*3 + 1)*4 + 2 = 6`
- `X[0,2,0]` → `(0*3 + 2)*4 + 0 = 8`

So: `[B,T,C]` is “per token, per channel”.

### Example B (two batches)

Let `B=2`, `T=3`, `C=4`.

- Batch 0 occupies offsets `0..11`
- Batch 1 occupies offsets `12..23`

For example:
- `X[1,0,0]` → `(1*3 + 0)*4 + 0 = 12`
- `X[1,2,3]` → `(1*3 + 2)*4 + 3 = 23`

## 2) Attention scores/probabilities: `[B,T,T]`

Meaning:
- For each batch element `b`
- for each query token position `i`
- store a scalar score/probability for **each key position `j`**

So each `S[b,i,:]` is a length-`T` row: how much token `i` attends to every token `j`.

Index:
- `S[b,i,j]`

Flat offset:

$$
\text{offset}(b,i,j) = (b\cdot T + i)\cdot T + j
$$

### Example C (matrix view)

Let `B=1`, `T=3`.

`S` contains `1*3*3 = 9` floats.

For batch `b=0`, think of a 3×3 matrix:

- Row `i=0` is offsets `0..2` (columns `j=0..2`)
- Row `i=1` is offsets `3..5`
- Row `i=2` is offsets `6..8`

Concrete offsets:
- `S[0,2,1]` → `(0*3 + 2)*3 + 1 = 7`
- `S[0,0,2]` → `(0*3 + 0)*3 + 2 = 2`

This tensor is “token-to-token”: one scalar per pair `(i,j)`.

## 3) Why attention matrices have no `c` dimension

In attention, scores are computed by a **dot product** of two length-`C` vectors (or length-`D` per head):

$$
S[b,i,j] = \frac{Q[b,i,:] \cdot K[b,j,:]}{\sqrt{C}} + \text{mask}(i,j)
$$

Expanded dot product:

$$
Q[b,i,:] \cdot K[b,j,:] = \sum_{c=0}^{C-1} Q[b,i,c] \; K[b,j,c]
$$

That summation **reduces the channel dimension** `c` away, producing **one scalar** for each `(b,i,j)`.

Then softmax produces probabilities:

$$
P[b,i,:] = \mathrm{softmax}(S[b,i,:])
$$

Still: one scalar per `(b,i,j)`.

So yes: attention score/prob matrices are “for each batch, each query token, each key token”.

## 4) Mini end-to-end sample (shapes only)

Suppose `B=1, T=3, C=4`.

- Hidden states: `X` is `[1,3,4]`
- Q/K/V after projection are each `[1,3,4]`
- Scores: `S` is `[1,3,3]`
- Probs:  `P` is `[1,3,3]`
- Attention output: `Y` is `[1,3,4]`

The key mental move:
- `C` is the “vector width” per token.
- `T` is the number of tokens.
- Scores/probs live over token pairs `(i,j)` so they are `T×T`.

## 5) Where this shows up in code

Baseline attention is implemented in:
- `nn::self_attention_1h` in `src/ops.cpp`

The scoring loops literally follow the indexing described above:
- They compute `qi` and `kj` base offsets for `[B,T,C]` buffers
- They write the score into the `[B,T,T]` buffer using the `(b,i,j)` formula

Next step (when you’re ready): pick specific `B,T,C,b,i,j` and trace one `S[b,i,j]` from `Q`/`K` memory offsets to the final score offset.
