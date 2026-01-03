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
- Tokenization is byte-level (vocab=256). Any text file works.
- This is a tiny baseline model; quality improves with more steps and larger `--dmodel/--layers/--seq`.

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
