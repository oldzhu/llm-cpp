> [简体中文](README.zh-CN.md)

# data

Put training/eval datasets here.

Notes:
- The current training data pipeline is byte-level (vocab=256), so any text file works.
- Subword/BPE end-to-end training would require a token-id dataset path (not implemented yet).
- Large datasets should stay local; `data/*.txt` is gitignored.

Example (PowerShell):

```powershell
.\build\Debug\train_gpt.exe --data .\data\the-verdict.txt --steps 200 --batch 4 --seq 64 --dmodel 64 --layers 1
```
