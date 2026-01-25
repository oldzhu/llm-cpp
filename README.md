> [简体中文](README.zh-CN.md)

# build-llm-using-cpp

A small, from-scratch (CPU FP32) GPT-style Transformer training project in C++.

Goals:
- Training-first (forward + backward + optimizer)
- CPU-only FP32 baseline for correctness
- Minimal dependencies (standard library only)

## Build (Windows)

Prereqs:
- CMake (>= 3.20 recommended)
- MSVC Build Tools (Visual Studio 2022 recommended)

Commands (PowerShell):

```powershell
cmake -S . -B build
cmake --build build --config Release -j
```

## Build (Linux)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Notes:
- This repo is intended to build from the same sources on Windows and Linux.
- Cross-compiling Linux binaries on Windows (or Windows binaries on Linux) is possible with toolchains, but it’s out of scope for the baseline.

## Run

By default it trains a tiny byte-level language model on a text file.

Example:

```powershell
.\\build\\Release\\train_gpt.exe --data .\\data\\the-verdict.txt --steps 200 --batch 4 --seq 64 --dmodel 64 --layers 1
```

If the relative path doesn’t work, pass an absolute path to `--data`.

## Generate ("chat" / sampling)

You can optionally train first and then generate from a prompt in the same run:

```powershell
.\\build\\Debug\\train_gpt.exe --data .\\data\\the-verdict.txt --steps 400 --batch 4 --seq 64 --dmodel 64 --layers 1 --lr 0.0003 --prompt "Hello, I'm a tiny GPT. " --gen 200 --temp 1.0 --topk 40
```

If you want readable output early in training, use ASCII-only sampling and/or escaping:

```powershell
.\\build\\Debug\\train_gpt.exe --load .\\data\\ckpt_tiny --steps 0 --prompt "Hello: " --gen 200 --temp 0.8 --topk 40 --ascii-only 1 --escape-bytes 1
```

Notes:
- Tokenization is pluggable: use --tokenizer byte (default) or --tokenizer bpe (with vocab/merges). Embedding layer and model config use tokenizer.vocab_size().
- This is a tiny baseline model; quality improves with more steps and larger `--dmodel/--layers/--seq`.

## Sanity checks (next-byte vs dataset)

These modes help answer: “given a context pulled from the dataset, does the model predict the *actual next byte* in that dataset?”

Important note:
- With only ~60 steps on `the-verdict.txt`, it’s normal to get mixed results.
- For this check to become consistently correct, increase training steps (e.g. `1000–5000+`) and/or increase model size.

### Random contexts from the dataset

Sample `N` random offsets from `--data`, run the model on each context, and compare the model’s top-1 prediction to the ground-truth next byte:

```powershell
.\build\Debug\train_gpt.exe --data .\data\the-verdict.txt --steps 0 --load .\data\ckpt_tiny --sanity-next-from-data 5 --sanity-ctx 64 --sanity-top 10 --escape-bytes 0
```

Flags:
- `--sanity-next-from-data N`: number of random trials.
- `--sanity-ctx T`: context length in bytes (defaults to model `seq_len`).
- `--sanity-top K`: also prints the top-K next-token distribution.
- `--escape-bytes 1`: prints bytes like `\xNN` for non-printables.

### One specific offset (exact selection check)

Check a specific byte offset `OFF` in the dataset:

- Context = `bytes[OFF .. OFF+T-1]`
- Expected next byte = `bytes[OFF+T]`

```powershell
& .\build\Debug\train_gpt.exe --% --data .\data\the-verdict.txt --steps 0 --load .\data\ckpt_tiny --sanity-offset 0 --sanity-ctx 64 --sanity-preview 120 --sanity-top 10 --escape-bytes 0
```

Notes:
- In PowerShell, if you run the executable via a quoted path, prefer `& <exe> --% ...` to avoid PowerShell interpreting `--flags`.
- `--sanity-preview N` prints the last `N` bytes of the context (with `...` prefix if truncated).

## Checkpoints (save/load)

Save a checkpoint (writes `PREFIX.json` + `PREFIX.bin`):

```powershell
.\build\Debug\train_gpt.exe --data .\data\the-verdict.txt --steps 1000 --batch 4 --seq 64 --dmodel 64 --layers 1 --save .\data\ckpt_tiny
```

Resume training from a checkpoint (adds more steps):

```powershell
.\build\Debug\train_gpt.exe --load .\data\ckpt_tiny --data .\data\the-verdict.txt --steps 500 --save .\data\ckpt_tiny
```

Generate from a checkpoint (no training needed):

```powershell
.\build\Debug\train_gpt.exe --load .\data\ckpt_tiny --steps 0 --prompt "Hello: " --gen 200 --temp 1.0 --topk 40
```

## Notes

- Vocabulary is 256 bytes (0-255). This is intentionally simple to get training working end-to-end.
- Next steps after this baseline: SIMD, mixed precision, CUDA backend, ROCm/HIP backend.

## Learning docs

If your goal is to learn how modern GPT-style training/inference works, start with:
- `docs/README.md` (docs index)
- `docs/transformer_math_to_code.md` (equations → functions/files)
- `docs/training_and_inference_walkthrough.md` (what happens in a run)

Project principles and workflow:
- `docs/project_lifecycle_guidelines.md`
- `.github/copilot-instructions.md` (persistent Copilot guidance for this repo)
