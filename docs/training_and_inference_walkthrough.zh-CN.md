> [English](training_and_inference_walkthrough.md)

# 训练 + 推理流程（walkthrough）

这是一份“我运行它时到底发生了什么？”的指南。

## 入口

- 程序：`src/main.cpp` 中的 `train_gpt`
- 主要模式：
  - 训练：`--data ... --steps N`
  - 保存/加载：`--save PREFIX` / `--load PREFIX`
  - 生成：`--prompt "..." --gen N`（可在训练后执行）

## 训练循环（高层）

给定一个字节数据集（byte dataset），训练循环是：

1) 从数据集中抽样一个 batch 的随机子串
- `data::ByteDataset::sample_batch(B,T, rng)`
- 得到：
  - `x[b,t] = bytes[start+t]`
  - `y[b,t] = bytes[start+t+1]`

2) 前向传播得到 loss
- `model::TinyGPT::loss(x, y, B, T)`
  - 内部调用 `forward_logits(x)` 产生 logits `[B,T,V]`
  - reshape 成 `[B*T,V]`
  - `nn::cross_entropy(logits, y)` 输出一个标量 loss

3) 反向传播
- `loss.backward()` 触发 `nn::Tensor` 的 reverse-mode autograd

4) 优化器更新
- `optim::AdamW::step(gpt.parameters().tensors)` 原地更新权重

## 生成（采样）

当提供 `--prompt` 时：

1) 把 prompt 字符串转换为 byte tokens
- 每个字符的 byte 直接作为 token id（范围 `[0..255]`）。

2) 重复执行：
- 取最后最多 `Tmax` 个 token 作为上下文
- `gpt.forward_logits(ctx, 1, T)`
- 读取最后一个时间步的 logits
- 采样下一个 token：
  - temperature 缩放
  - 可选 top-k
  - 可选 ASCII-only 过滤
- 追加 token 并打印

实现位置：`src/main.cpp` 中的 `generate(...)`。


## Checkpoint 格式（概览）

- `<prefix>.json`：人类可读的配置（模型维度、优化器超参、step）
- `<prefix>.bin`：权重 +（可选）优化器状态

精确布局请看 `src/checkpoint.cpp`。

## Sanity 检查（数据集续写）

这些模式用于把模型的“下一 token”预测与数据集文件中的 **真实下一字节** 对齐验证。

重要说明：
- 本项目是 byte-level（`vocab=256`），所以这里的 token 就是 byte。
- 仅在 `the-verdict.txt` 上训练约 ~60 step 得到“有时对、有时错”是正常的。
- 若希望更稳定地预测正确：增加训练步数（例如 `1000–5000+`）和/或增大模型规模。

### 随机抽样数据集位置

CLI 参数：
- `--sanity-next-from-data N`：从数据集中随机抽取 `N` 个上下文。
- `--sanity-ctx T`：上下文长度（单位：字节），默认等于模型的 `seq_len`。
- `--sanity-top K`：打印 top-K 的下一 token 概率（temperature=1，不做 ASCII 过滤）。

对每一次 trial，程序会随机选择一个 offset `start`，并构造：
- `ctx[i] = bytes[start+i]`（`i in [0..T-1]`）
- `expected = bytes[start+T]`

然后运行 `gpt.forward_logits(ctx, 1, T)`，取最后一个时间步的 argmax，与 `expected` 比较。

### 固定一个特定 offset

CLI 参数：
- `--sanity-offset OFF`：使用固定的数据集 offset。
- `--sanity-preview N`：打印所选上下文最后 `N` 个字节（便于肉眼核对）。

这对应“我手动选了一段具体句子/片段；模型能否预测该位置的下一个字节？”这个问题。

PowerShell 小技巧：如果通过带引号的 exe 路径调用，使用 `& <exe> --% ...` 可确保 `--flags` 原样传入。

模式：
- `--sanity-next-from-data N`：随机抽样 N 次 offset
- `--sanity-offset OFF`：固定 offset（用于你手动选定的片段）
- `--sanity-ctx T`：上下文长度
- `--sanity-preview N`：打印上下文末尾 N 个字节，便于肉眼确认

PowerShell 提示：若通过 `& <exe>` 调用，建议使用 `--%` 传递 `--flags`。
