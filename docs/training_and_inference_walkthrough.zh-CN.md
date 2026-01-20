> [English](training_and_inference_walkthrough.md)

# 训练 + 推理流程（walkthrough）

> 说明：本中文版本翻译进行中；细节请优先参考英文版。

这篇文档解释“运行一次 `train_gpt` 时发生了什么”。

## 入口

- 程序：`src/main.cpp` 中的 `train_gpt`
- 主要模式：
  - 训练：`--data ... --steps N`
  - 保存/加载：`--save PREFIX` / `--load PREFIX`
  - 生成：`--prompt "..." --gen N`（可在训练后执行）

## 训练循环（高层）

1) 从数据集中抽样 batch
- `data::ByteDataset::sample_batch(B,T, rng)`
- 得到：
  - `x[b,t] = bytes[start+t]`
  - `y[b,t] = bytes[start+t+1]`

2) 前向得到 logits 并计算 loss
- `model::TinyGPT::loss(x, y, B, T)`

3) 反向传播
- `loss.backward()` 触发反向自动微分

4) 优化器更新
- `optim::AdamW::step(...)`

## 生成（采样）

- prompt 字符串会被转为 byte token（0–255）。
- 每一步：取最后 `Tmax` 个 token 作为上下文，前向得到 logits，然后按 temperature/top-k/可选过滤采样下一个 token。

## Sanity 检查（数据集续写）

这些模式用于验证：给定数据集中某段上下文，模型预测的“下一 token”是否与数据集中的真实下一字节一致。

重要说明：
- 只训练 ~60 step 得到混合正确率是正常的。
- 想更稳定正确：增加训练步数（如 `1000–5000+`）和/或增大模型。

模式：
- `--sanity-next-from-data N`：随机抽样 N 次 offset
- `--sanity-offset OFF`：固定 offset（用于你手动选定的片段）
- `--sanity-ctx T`：上下文长度
- `--sanity-preview N`：打印上下文末尾 N 个字节，便于肉眼确认

PowerShell 提示：若通过 `& <exe>` 调用，建议使用 `--%` 传递 `--flags`。
