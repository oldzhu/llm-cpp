> [English](2026-02-26_rl_stage_overview.md)

# 2026-02-26 — GPT 训练中的 RL 阶段（概览）

## 摘要

解释了 RL（RLHF/RLAIF）在典型 GPT 训练流程中的位置：

1) 预训练（next-token prediction）
2) 监督微调（instruction-following）
3) 偏好优化（RL 或 RL-like）

## 关键点

- RL 阶段把 GPT 模型当作 policy。
- Reward model 由偏好数据训练，并对输出打分。
- PPO 等方法优化期望奖励，同时用 KL 约束保持与 SFT policy 接近。
- DPO 等替代方法通过偏好数据直接构造损失，避免显式 RL。
