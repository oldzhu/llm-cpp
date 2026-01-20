> [English](README.md)

# data

把训练/评估数据集放在这里。

说明：
- 本项目默认是字节级 token（`vocab=256`），因此任意文本文件都可用于训练。
- 大数据集建议保留在本地；`data/*.txt` 在仓库里是 gitignored。

示例（PowerShell）：

```powershell
.\build\Debug\train_gpt.exe --data .\data\the-verdict.txt --steps 200 --batch 4 --seq 64 --dmodel 64 --layers 1
```
