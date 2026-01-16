# build-llm-using-cpp docs

This folder is the **learning layer** of the project: short, practical notes that map the math and algorithms to the exact C++ implementation.

Start here:
- `transformer_math_to_code.md` — equations + where they live in code
- `training_and_inference_walkthrough.md` — how a full run (train / save / load / generate) flows through the code
- `autograd_and_memory_layout.md` — how tensors are indexed + how backward works in this repo
- `learn_by_hand_walkthrough.md` — tiny numeric examples you can compute on paper
- `attention_indexing_trace.md` — trace attention scores to flat-buffer offsets

Notes:
- `notes/2026-01-16_project_goal_and_next_steps.md` — project intent + recommended roadmap

Conventions:
- We prefer pointing to **functions and shapes** (e.g. `[B,T,C]`) rather than copying long paper text.
- When we reference papers, we only reference their public names/links; the content here is an original explanation.

Docs contract (important):
- Every time we add a major feature (multi-head attention, KV-cache, new optimizers, RLHF-style training, etc.), we also add/update docs so learners can map **paper concept → formula → code**.
