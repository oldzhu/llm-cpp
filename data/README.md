# data

Put training/eval datasets here.

Notes:
- This project defaults to byte-level (vocab=256), so any text file works.
- Large datasets should stay local; `data/*.txt` is gitignored.

Example (PowerShell):

```powershell
.\build\Debug\train_gpt.exe --data .\data\the-verdict.txt --steps 200 --batch 4 --seq 64 --dmodel 64 --layers 1
```
