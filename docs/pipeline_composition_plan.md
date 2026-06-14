# Pipeline Composition Architecture — Design & Plan

> [简体中文](pipeline_composition_plan.zh-CN.md)

## Goal

Make the GPT pipeline **composable** — each stage is a swappable component selected via config, enabling any combination of tokenizer + embedding + attention + norm + MLP to be assembled through the CLI or Web UI without code changes.

## Motivation

The current architecture has implementations for multiple variants (MHA, GQA, RoPE, RMSNorm, SwiGLU, MoE) but they exist as **isolated standalone functions**, not wired into the model through config. To create a LLaMA-style model (RoPE + GQA + RMSNorm + SwiGLU), you'd need to manually edit `model.cpp`'s forward pass.

The LLM Architecture Gallery (81+ models) shows that modern LLMs are all variations of the same decoder-only pipeline with different choices at each stage. Our project should enable experimenting with ANY combination.

## Design: Component Pipeline

```
Pipeline Config:
  tokenizer:  { type: "byte" | "bpe" }                    (already configurable)
  position:   { type: "wpe" | "rope" }                    (NEW config field)
  attention:  { type: "1head" | "mha" | "gqa" }          (NEW config field)
  norm:       { type: "layernorm" | "rmsnorm" }           (already configurable)
  mlp:        { type: "gelu" | "swiglu" | "moe" }        (already configurable)
```

## Config Changes

```cpp
struct Config {
  // ... existing fields ...
  int norm_type = 0;    // 0=LayerNorm, 1=RMSNorm     (existing)
  int mlp_type = 0;     // 0=GELU, 1=SwiGLU, 2=MoE    (existing)
  int attn_type = 0;    // 0=1-head, 1=MHA, 2=GQA      (NEW)
  int pos_type = 0;     // 0=wpe, 1=RoPE               (NEW)
  int attn_n_heads = 1; // num attention heads          (NEW)
  int attn_n_kv = 1;    // num KV heads (for GQA)       (NEW)
};
```

## Model Changes

`forward_logits()` switches on `attn_type` and `pos_type`:

```cpp
// Position encoding
if (cfg_.pos_type == 0) {
  x = add_positional(x, B, T);  // wpe
} else {
  // RoPE applied inside attention (Q,K rotated)
}

// Attention sublayer
switch (cfg_.attn_type) {
  case 0: a = nn::self_attention_1h(h, w_qkv, b_qkv, w_proj, b_proj); break;
  case 1: a = nn::variants::mha::self_attention_mha(h, w_qkv, b_qkv, w_proj, b_proj, cfg_.attn_n_heads); break;
  case 2: a = nn::variants::gqa::self_attention_gqa(...).reshape(...); break;
}
```

## Backward Compatibility

- detault attn_type=0, pos_type=0 → identical behavior to current
- Checkpoint v5 stores new fields; v4/v3/v2/v1 load with defaults (attn_type=0, pos_type=0)
- KV-cache variant reads attn_type from config

## Future Extensions

| Component | Future types |
|-----------|-------------|
| attention | `3 = MLA` (DeepSeek-style latent attention) |
| position | `2 = ALiBi`, `3 = NoPE`, `4 = YaRN` |
| mlp | `3 = GeGLU`, `4 = ReGLU` |
| tokenizer | `2 = SentencePiece` |

## Implementation Phases

| Phase | What | Status |
|:---:|------|:---:|
| 1 | attn_type + pos_type config + model wiring | In progress |
| 2 | Sub-config structs (AttentionCfg, NormCfg, MLPCfg) | Planned |
| 3 | MLA attention variant | Planned |
| 4 | Shared MoE experts | Planned |
| 5 | MTP (Multi-Token Prediction) training | Planned |

## Key Decisions

1. **Flat config fields** (not nested JSON) for checkpoint backward compat
2. **New params only when used** (n_heads, n_kv not allocated if attn_type=0)
3. **Checkpoint v5** with auto-defaults for v1-v4
4. **Web UI** adds attention/position fields with show_if dependencies
