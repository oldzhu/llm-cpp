> [简体中文](2026-02-26_rl_stage_overview.zh-CN.md)

# 2026-02-26 — RL stage in GPT training (overview)

## Summary

Explained where RL (RLHF/RLAIF) fits in a typical GPT training pipeline:

1) Pretraining (next-token prediction)
2) Supervised fine-tuning (instruction-following)
3) Preference optimization (RL or RL-like)

## Key points

- RL stage uses the GPT model as the policy.
- A reward model is trained on preference pairs and scores outputs.
- PPO or similar optimizes expected reward while penalizing KL divergence from the SFT policy.
- Alternatives like DPO remove explicit RL by converting preferences into a direct loss.
