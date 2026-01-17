# build-llm-using-cpp docs

This folder is the **learning layer** of the project: short, practical notes that map the math and algorithms to the exact C++ implementation.

Start here:
- `transformer_math_to_code.md` — equations + where they live in code
- `training_and_inference_walkthrough.md` — how a full run (train / save / load / generate) flows through the code
- `autograd_and_memory_layout.md` — how tensors are indexed + how backward works in this repo
- `learn_by_hand_walkthrough.md` — tiny numeric examples you can compute on paper
- `attention_indexing_trace.md` — trace attention scores to flat-buffer offsets
- `project_lifecycle_guidelines.md` — project principles + extension workflow

Architecture:
- `architecture/folder_layout_and_targets.md` — proposed folder layout + CMake target scheme
- `architecture/cpp_structures_to_gpt_decoder_math.md` — C++ types/members/methods mapped to GPT decoder math
- `architecture/hidden_states_vs_attention_matrices.md` — `[B,T,C]` vs `[B,T,T]` meaning + flat-buffer offsets

Variants:
- `variants/mha/README.md` — multi-head attention variant overview
- `variants/mha/learn_by_hand.md` — small hand-computable MHA example
- `variants/mha/what_changed_vs_core.md` — what the MHA extension added/changed vs the baseline

Notes:
- `notes/2026-01-16_project_goal_and_next_steps.md` — project intent + recommended roadmap

Conventions:
- We prefer pointing to **functions and shapes** (e.g. `[B,T,C]`) rather than copying long paper text.
- When we reference papers, we only reference their public names/links; the content here is an original explanation.

Docs contract (important):
- Every time we add a major feature (multi-head attention, KV-cache, new optimizers, RLHF-style training, etc.), we also add/update docs so learners can map **paper concept → formula → code**.
