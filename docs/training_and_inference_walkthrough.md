> [简体中文](training_and_inference_walkthrough.zh-CN.md)

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


## Sanity checks (dataset continuation)

These modes are meant to validate basic “next token” behavior against the *ground truth next byte* in the dataset file.

Important: this project is byte-level (vocab=256), so “token” here means “byte”.

Important note:
- With only ~60 steps on `the-verdict.txt`, it’s normal to get mixed results.
- For this check to become consistently correct, increase training steps (e.g. `1000–5000+`) and/or increase model size.

### Random dataset samples

CLI flags:
- `--sanity-next-from-data N`: sample `N` random contexts from the dataset.
- `--sanity-ctx T`: context length in bytes (defaults to model `seq_len`).
- `--sanity-top K`: prints top-K next-token probabilities (temperature=1, no ASCII filtering).

For each trial, the program picks an offset `start` in the dataset and constructs:
- `ctx[i] = bytes[start+i]` for `i in [0..T-1]`
- `expected = bytes[start+T]`

Then it runs `gpt.forward_logits(ctx, 1, T)` and compares the argmax at the last time step to `expected`.

### One specific offset

CLI flags:
- `--sanity-offset OFF`: use a fixed dataset offset.
- `--sanity-preview N`: print the last `N` bytes of the selected context (for human inspection).

This is the “I selected a specific substring/sentence; does the model predict the very next byte from that exact position?” mode.

Tip (PowerShell): if invoking via a quoted exe path, use `& <exe> --% ...` so `--flags` pass through.
