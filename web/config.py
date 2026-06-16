"""Training config schema, validation, and presets."""

from typing import Any

CONFIG = {
    "data_path":  {"type":"path","default":"data/alice.txt","label":"Training Data","group":"Data"},
    "tokenizer":  {"type":"choice","options":["byte","bpe","sp"],"default":"byte","label":"Tokenizer","group":"Data"},
    "bpe_vocab":  {"type":"path","default":"data/bpe_vocab.json","label":"BPE Vocab File","group":"Data","show_if":{"tokenizer":"bpe"}},
    "bpe_merges": {"type":"path","default":"data/bpe_merges.txt","label":"BPE Merges File","group":"Data","show_if":{"tokenizer":"bpe"}},
    "token_data": {"type":"path","default":"","label":"Token Data (.bin)","group":"Data","show_if":{"tokenizer":"bpe"}},
    "steps":       {"type":"int","default":500,"min":1,"max":100000,"label":"Training Steps","group":"Training"},
    "batch":       {"type":"int","default":4,"min":1,"max":256,"label":"Batch Size","group":"Training"},
    "seq":         {"type":"int","default":64,"min":8,"max":2048,"label":"Sequence Length","group":"Architecture"},
    "dmodel":      {"type":"int","default":64,"min":8,"max":1024,"label":"Model Dimension (C)","group":"Architecture"},
    "layers":      {"type":"int","default":1,"min":1,"max":48,"label":"Layers","group":"Architecture"},
    "lr":          {"type":"float","default":0.0003,"min":1e-6,"max":0.1,"label":"Learning Rate","group":"Training"},
    "seed":        {"type":"int","default":1,"min":0,"max":999999,"label":"Random Seed","group":"Training"},
    "norm_type":   {"type":"choice","options":["layernorm","rmsnorm"],"default":"layernorm","label":"Normalization","group":"Architecture"},
    "mlp_type":    {"type":"choice","options":["gelu","swiglu","moe"],"default":"gelu","label":"MLP Type","group":"Architecture"},
    "attn_type":   {"type":"choice","options":["1head","mha","gqa","mla"],"default":"1head","label":"Attention Type","group":"Architecture"},
    "n_heads":     {"type":"int","default":1,"min":1,"max":64,"label":"Number of Heads","group":"Architecture","show_if":{"attn_type":["mha","gqa","mla"]}},
    "n_kv":        {"type":"int","default":1,"min":1,"max":64,"label":"KV Heads (GQA)","group":"Architecture","show_if":{"attn_type":"gqa"}},
    "pos_type":    {"type":"choice","options":["wpe","rope","alibi","nope"],"default":"wpe","label":"Position Encoding","group":"Architecture"},
    "qk_norm":     {"type":"choice","options":["0","1"],"default":"0","label":"QK-Norm (0=off, 1=on)","group":"Architecture"},
    "swin_win":    {"type":"int","default":0,"min":0,"max":8192,"label":"Sliding Window (0=off)","group":"Architecture"},
    "mla_latent":  {"type":"int","default":0,"min":0,"max":2048,"label":"MLA Latent Dim (0=C/4)","group":"Architecture","show_if":{"attn_type":"mla"}},
    "n_mtp":       {"type":"int","default":1,"min":1,"max":8,"label":"MTP Heads (1=off)","group":"Architecture"},
    "n_shared":    {"type":"int","default":0,"min":0,"max":8,"label":"Shared MoE Experts","group":"Architecture","show_if":{"mlp_type":"moe"}},
    "save_prefix": {"type":"path","default":"data/ckpt_web","label":"Checkpoint Prefix","group":"Output"},
    "save_interval":{"type":"int","default":50,"min":0,"max":100000,"label":"Save Every N Steps (0=end only)","group":"Output"},
    "temperature": {"type":"float","default":0.8,"min":0.1,"max":2.0,"label":"Temperature","group":"Generation"},
    "topk":        {"type":"int","default":40,"min":0,"max":1000,"label":"Top-K Sampling","group":"Generation"},
    "kvcache":     {"type":"bool","default":True,"label":"Use KV-Cache","group":"Generation"},
}

PRESETS: dict[str, dict[str, Any]] = {
    "Tiny (32M)":    {"dmodel":32, "layers":1, "batch":8, "seq":64, "steps":200},
    "Small (64M)":   {"dmodel":64, "layers":1, "batch":8, "seq":80, "steps":500},
    "Medium (100M)": {"dmodel":100,"layers":2, "batch":4, "seq":100,"steps":1000},
    "BPE Tiny":      {"dmodel":32, "layers":1, "batch":4, "seq":32, "steps":200, "tokenizer":"bpe",
                      "bpe_vocab":"data/bpe_vocab.json","bpe_merges":"data/bpe_merges.txt","token_data":"data/verdict_tokens.bin"},
}

def validate(config: dict) -> dict[str, str]:
    errors: dict[str, str] = {}
    for key, spec in CONFIG.items():
        if key not in config:
            continue
        val = config[key]
        t = spec["type"]
        if t == "int" and not isinstance(val, int):
            try: val = int(val)
            except: errors[key] = f"Must be an integer"; continue
            config[key] = val
        if t == "float" and not isinstance(val, (int, float)):
            try: val = float(val)
            except: errors[key] = f"Must be a number"; continue
            config[key] = val
        if t == "int":
            if "min" in spec and val < spec["min"]:
                errors[key] = f"Min {spec['min']}"
            if "max" in spec and val > spec["max"]:
                errors[key] = f"Max {spec['max']}"
        if t == "choice" and val not in spec.get("options", []):
            errors[key] = f"Must be one of {spec['options']}"
    return errors

def apply_preset(config: dict, preset_name: str) -> dict:
    c = dict(config)
    c.update(PRESETS.get(preset_name, {}))
    return c
