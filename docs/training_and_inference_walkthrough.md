# Training + inference walkthrough

This is a “what happens when I run it?” guide.

## Entry point

- Program: `train_gpt` in `src/main.cpp`
- Key modes:
  - Train: `--data ... --steps N`
  - Save/load checkpoints: `--save PREFIX` / `--load PREFIX`
  - Generate from prompt: `--prompt "..." --gen N` (optionally after training)


## Training loop (high-level)

Given a byte dataset, the loop is:

1) Sample a batch of random substrings
- `data::ByteDataset::sample_batch(B,T, rng)`
- Returns:
  - `x[b,t] = bytes[start+t]`
  - `y[b,t] = bytes[start+t+1]`

2) Forward pass → loss
- `model::TinyGPT::loss(x, y, B, T)`
  - `forward_logits(x)` produces logits `[B,T,V]`
  - reshape to `[B*T,V]`
  - `nn::cross_entropy(logits, y)` produces scalar loss

3) Backward pass
- `loss.backward()` triggers reverse-mode autodiff via `nn::Tensor` nodes.

4) Optimizer update
- `optim::AdamW::step(gpt.parameters().tensors)` updates weights in-place.


## Generation (sampling)

If `--prompt` is provided:

1) Convert prompt string → byte tokens
- Each character byte becomes a token id in `[0..255]`.

2) Repeatedly
- Take the last `Tmax` tokens as context
- `gpt.forward_logits(ctx, 1, T)`
- Read logits at the last time step
- Sample next token:
  - temperature scaling
  - optional top-k
  - optional ASCII-only filter
- Append token, print it

This is implemented in `generate(...)` in `src/main.cpp`.


## Checkpoint format (overview)

- `<prefix>.json`: human-readable config (model dims, optimizer hypers, step)
- `<prefix>.bin`: weights + optional optimizer state

See `src/checkpoint.cpp` for the exact layout.
