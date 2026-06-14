#!/usr/bin/env python3
"""Import HuggingFace safetensors model → build-llm-using-cpp checkpoint.

Usage:
  python scripts/import_safetensors.py distilgpt2/ data/ckpt_distilgpt2 --mapping gpt2
  python scripts/import_safetensors.py Llama-3.2-1B/ data/ckpt_llama --mapping llama [--copy-tokenizer]
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import safetensors.torch
    HAVE_SAFETENSORS = True
except ImportError:
    HAVE_SAFETENSORS = False


def parse_shape(shape_spec, cfg):
    """Resolve shape specification like ["C","3*C"] or "3*C" to actual ints."""
    if shape_spec is None:
        return [0]
    C, T, V = cfg.get("d_model", 0), cfg.get("seq_len", 0), cfg.get("vocab_size", 0)
    interm = cfg.get("swiglu_interm", 0) or (3 * C) if C > 0 else 0
    if isinstance(shape_spec, list):
        result = []
        for s in shape_spec:
            if isinstance(s, int): result.append(s)
            else: result.append(eval(s, {"C": C, "T": T, "V": V, "interm": interm}))
        return result
    return [eval(shape_spec, {"C": C, "T": T, "V": V, "interm": interm})]


def resolve_shape_var(name, cfg):
    """Return shape for a named variable like 'C', 'V', '3C'."""
    C, T, V = cfg.get("d_model", 0), cfg.get("seq_len", 0), cfg.get("vocab_size", 0)
    if name == "C": return [C]
    if name == "T": return [T]
    if name == "V": return [V]
    if name == "3C": return [3 * C]
    if name == "4C": return [4 * C]
    return [0]


def get_tensor(hf_weights, hf_name, cfg):
    """Resolve HF tensor name, return (name_used, numpy_array) or (None, None)."""
    if hf_name is None:
        return None, None
    result = safetensors.torch.load_file(hf_weights) if isinstance(hf_weights, str) else None
    # We use a different API below
    return None, None


def load_safetensors(model_dir):
    """Load all tensors from model_dir/model.safetensors (or .safetensors file directly)."""
    if os.path.isfile(model_dir):
        path = model_dir
    else:
        # Try model.safetensors first, then check for sharded
        candidates = [
            os.path.join(model_dir, "model.safetensors"),
            os.path.join(model_dir, "model.safetensors.index.json"),
        ]
        path = None
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        if path is None:
            # Try any .safetensors file
            for f in os.listdir(model_dir):
                if f.endswith(".safetensors"):
                    path = os.path.join(model_dir, f)
                    break

    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"No safetensors file found in {model_dir}")

    if path.endswith(".index.json"):
        # Sharded model
        with open(path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = set(weight_map.values())
        tensors = {}
        for shard in shard_files:
            shard_path = os.path.join(model_dir, shard)
            with safetensors.safe_open(shard_path, framework="pt") as sf:
                for key in sf.keys():
                    tensors[key] = sf.get_tensor(key)
        return tensors
    else:
        tensors = {}
        with safetensors.safe_open(path, framework="pt") as sf:
            for key in sf.keys():
                tensors[key] = sf.get_tensor(key)
        return tensors


def apply_op(tensor, op, shape, cfg, extra=None):
    """Apply an operation to convert HF tensor to our tensor.
    
    ops: identity, transpose, zeros, transpose_or_wte, llama_qkv
    """
    C = cfg.get("d_model", 0)
    
    if op == "identity":
        if tensor is None:
            shape_vals = parse_shape(shape, cfg) if shape else [0]
            return np.zeros(shape_vals, dtype=np.float32)
        arr = tensor.float().numpy().astype(np.float32)
        # Ensure contiguous
        return np.ascontiguousarray(arr)
    
    elif op == "transpose":
        if tensor is None:
            shape_vals = parse_shape(shape, cfg)
            return np.zeros(shape_vals, dtype=np.float32)
        arr = tensor.float().numpy().astype(np.float32)
        return np.ascontiguousarray(arr.T)
    
    elif op == "zeros":
        if isinstance(shape, list):
            shape_vals = parse_shape(shape, cfg)  # resolve "C", "3*C" etc in list
        elif isinstance(shape, str):
            shape_vals = resolve_shape_var(shape, cfg)
        else:
            shape_vals = parse_shape(shape, cfg)
        return np.zeros(shape_vals, dtype=np.float32)
    
    elif op == "transpose_or_wte":
        # Used for lm_head.weight: if tied to wte, copy wte; otherwise transpose
        tie_name = extra.get("tied") if extra else None
        if tensor is not None:
            arr = tensor.float().numpy().astype(np.float32)
            return np.ascontiguousarray(arr.T)
        elif tie_name and extra is not None and extra.get("_tied") is not None and extra["_tied"].size > 0:
            return np.ascontiguousarray(extra["_tied"].copy())
        else:
            shape_vals = parse_shape(shape, cfg)
            return np.zeros(shape_vals, dtype=np.float32)
    
    elif op == "llama_qkv":
        # Concat separate Q, K, V projections into w_qkv [C, 3C]
        # HF stores weights as [heads*head_dim, C], we need [C, heads*head_dim] then concat
        if extra is None:
            raise ValueError("llama_qkv op requires extra dict with hf_q/hf_k/hf_v tensors")
        
        q_tensor = extra.get("q")
        k_tensor = extra.get("k")
        v_tensor = extra.get("v")
        n_kv_heads = extra.get("n_kv_heads", 1)
        
        if q_tensor is None:
            return np.zeros((C, 3 * C), dtype=np.float32)
        
        q = q_tensor.float().numpy().astype(np.float32)  # [n_heads*hd, C]
        q = np.ascontiguousarray(q.T)  # [C, n_heads*hd] where n_heads*hd == C
        
        if k_tensor is not None:
            k = k_tensor.float().numpy().astype(np.float32)  # [n_kv*hd, C]
            k = np.ascontiguousarray(k.T)  # [C, n_kv*hd]
        else:
            k = np.zeros((C, 0), dtype=np.float32)
        
        if v_tensor is not None:
            v = v_tensor.float().numpy().astype(np.float32)  # [n_kv*hd, C]
            v = np.ascontiguousarray(v.T)  # [C, n_kv*hd]
        else:
            v = np.zeros((C, 0), dtype=np.float32)
        
        k_dim = k.shape[1]  # n_kv_heads * head_dim
        v_dim = v.shape[1]
        
        # Pad K and V to C columns with zeros (for GQA when n_kv_heads < n_heads)
        k_pad = C - k_dim
        v_pad = C - v_dim
        
        if k_pad > 0:
            k = np.hstack([k, np.zeros((C, k_pad), dtype=np.float32)])
        if v_pad > 0:
            v = np.hstack([v, np.zeros((C, v_pad), dtype=np.float32)])
        
        result = np.hstack([q, k, v])  # [C, 3C]
        return np.ascontiguousarray(result.astype(np.float32))
    
    else:
        raise ValueError(f"Unknown op: {op}")


def write_checkpoint(output_prefix, cfg, tensors_dict):
    """Write our native checkpoint format (.json + .bin)."""
    # We need tensors in the EXACT order of parameters().
    # tensors_dict is a list of (name, numpy_array) in correct order.
    
    json_path = output_prefix + ".json"
    bin_path = output_prefix + ".bin"
    
    # JSON config
    json_obj = {
        "format": "build-llm-using-cpp-checkpoint",
        "version": 4,
        "vocab_size": cfg.get("vocab_size", 256),
        "seq_len": cfg.get("seq_len", 64),
        "d_model": cfg.get("d_model", 64),
        "n_layers": cfg.get("n_layers", 1),
        "norm_type": cfg.get("norm_type", 0),
        "mlp_type": cfg.get("mlp_type", 0),
        "swiglu_interm": cfg.get("swiglu_interm", 0),
        "n_experts": cfg.get("n_experts", 4),
        "top_k": cfg.get("top_k", 2),
        "lr": cfg.get("lr", 0.0003),
        "beta1": cfg.get("beta1", 0.9),
        "beta2": cfg.get("beta2", 0.999),
        "eps": cfg.get("eps", 1e-8),
        "weight_decay": cfg.get("weight_decay", 0.01),
        "step": 0,
        "has_optim_state": False,
        "imported_from": cfg.get("imported_from", ""),
    }
    
    with open(json_path, "w") as f:
        json.dump(json_obj, f, indent=2)
    
    # Binary
    with open(bin_path, "wb") as f:
        f.write(b"BGPTCKPT")
        f.write(struct.pack("<I", 4))  # version
        f.write(struct.pack("<I", 0))  # has_opt = no
        f.write(struct.pack("<Q", 0))  # step = 0
        f.write(struct.pack("<I", len(tensors_dict)))  # nparams
        
        for name, arr in tensors_dict:
            arr = np.ascontiguousarray(arr.flatten().astype(np.float32))
            f.write(struct.pack("<Q", arr.shape[0]))
            f.write(arr.tobytes())
    
    print(f"Wrote {len(tensors_dict)} tensors -> {bin_path}")
    print(f"Config → {json_path}")


def import_model(model_dir, output_prefix, mapping_name, copy_tokenizer=False):
    mapping_path = os.path.join(os.path.dirname(__file__), "mappings", f"{mapping_name}.json")
    with open(mapping_path) as f:
        mapping = json.load(f)
    
    print(f"Loading model from {model_dir}")
    hf_tensors = load_safetensors(model_dir)
    print(f"  Found {len(hf_tensors)} tensors")
    
    # Read HF config.json
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        cfg_path = None
        for fname in os.listdir(model_dir):
            if fname == "config.json":
                cfg_path = os.path.join(model_dir, fname)
                break
    if cfg_path is None:
        raise FileNotFoundError(f"No config.json found in {model_dir}")
    
    with open(cfg_path) as f:
        hf_cfg = json.load(f)
    
    # Build reverse map: our_key → hf_key  (config_map is hf_key → our_key)
    cm = mapping["config_map"]
    rev_map = {v: k for k, v in cm.items()}
    def hf_val(our_key, default_val=0):
        hf_key = rev_map.get(our_key, our_key)
        return hf_cfg.get(hf_key, default_val)
    
    C = hf_val("d_model", 0)
    V = hf_val("vocab_size", 256)
    T = hf_val("seq_len", 64)
    n_layers = hf_val("n_layers", 1)
    interm_key = rev_map.get("swiglu_interm", None)
    if interm_key:
        interm = hf_cfg.get(interm_key, 0) or (4 * C if C > 0 else 0)
    else:
        interm = 4 * C if C > 0 else 0
    
    cfg = {
        "vocab_size": V, "seq_len": T, "d_model": C, "n_layers": n_layers,
        "swiglu_interm": interm,
        "norm_type": mapping.get("fixed_config", {}).get("norm_type", 0),
        "mlp_type": mapping.get("fixed_config", {}).get("mlp_type", 0),
        "n_experts": 4, "top_k": 2,
        "lr": 0.0003, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01,
        "imported_from": os.path.basename(model_dir.rstrip("/\\")),
    }
    
    # Override with HF config if mapping has config_from_key
    if "config_from_key" in mapping:
        ck = mapping["config_from_key"]
        if "norm_type" in ck:
            for trigger_key, spec in ck["norm_type"].items():
                if trigger_key == "set": continue
                if trigger_key == "else": continue
                if trigger_key in hf_cfg:
                    cfg["norm_type"] = 1
                elif spec is True and trigger_key in hf_cfg:
                    cfg["norm_type"] = 1


    # GQA detection
    n_kv_heads = hf_cfg.get("num_key_value_heads", 1)
    n_heads = hf_cfg.get("num_attention_heads", n_kv_heads)
    has_gqa = n_kv_heads < n_heads
    if has_gqa:
        print(f"  GQA detected: n_kv_heads={n_kv_heads}, n_heads={n_heads}")
    
    print(f"  Config: C={C}, V={V}, T={T}, n_layers={n_layers}")
    
    # Build tensor list in parameters() order
    tensors = []  # list of (name, numpy_array)
    
    def get_hf(name_template, layer_idx=None):
        """Resolve HF tensor name, with {L} replaced by layer index."""
        if name_template is None:
            return None
        tname = name_template
        if layer_idx is not None:
            tname = tname.replace("{L}", str(layer_idx))
        return hf_tensors.get(tname)
    
    # 0-1: wte, wpe
    for p in mapping["params"]:
        t = get_hf(p["hf"])
        arr = apply_op(t, p["op"], p.get("shape"), cfg)
        tensors.append((p["name"], arr))
    
    # Layer params
    for li in range(n_layers):
        for p in mapping["layer_params"]:
            if p["op"] == "llama_qkv":
                q = get_hf(p["hf_q"], li)
                k = get_hf(p["hf_k"], li)
                v = get_hf(p["hf_v"], li)
                extra = {"q": q, "k": k, "v": v, "n_kv_heads": n_kv_heads}
                arr = apply_op(None, "llama_qkv", None, cfg, extra)
                tensors.append((p["name"], arr))
            else:
                t = get_hf(p["hf"], li)
                shape = None
                if "shape_var" in p:
                    shape = p["shape_var"]
                elif "shape" in p:
                    shape = p["shape"]
                arr = apply_op(t, p["op"], shape, cfg)
                tensors.append((p["name"], arr))
    
    # Post-layer params (w_lm, b_lm, ln_final_gamma, ln_final_beta)
    wte_arr = tensors[0][1]  # wte for tie detection
    for p in mapping.get("post_params", []):
        t = get_hf(p["hf"])
        shape = None
        if "shape_var" in p:
            shape = p["shape_var"]
        elif "shape" in p:
            shape = p["shape"]
        extra = None
        if p["op"] == "transpose_or_wte":
            extra = {"tied": p.get("tied"), "_tied": wte_arr if t is None else None}
        arr = apply_op(t, p["op"], shape, cfg, extra)
        tensors.append((p["name"], arr))
    
    # SwiGLU per-layer (for LLaMA)
    if "swiglu_per_layer" in mapping:
        for li in range(n_layers):
            for p in mapping["swiglu_per_layer"]["params"]:
                t = get_hf(p["hf"], li) if p.get("hf") else None
                shape = None
                if "shape_var" in p:
                    shape = p["shape_var"]
                elif "shape" in p:
                    shape = p["shape"]
                arr = apply_op(t, p["op"], shape, cfg)
                tensors.append((p["name"], arr))
    
    # Extra tensors per layer (SwiGLU zeros for GPT-2, MoE zeros)
    for p in mapping.get("extra_tensors_per_layer", []):
        shape = p.get("shape_var") or p.get("shape")  # support both keys
        shape_vals = parse_shape(shape, cfg)
        for li in range(n_layers):
            arr = np.zeros(shape_vals, dtype=np.float32)
            tensors.append((p["name"], arr))
    
    # MoE expert tensors
    n_exp = cfg["n_experts"]
    for p in mapping.get("extra_expert_tensors", []):
        shape = p.get("shape_var") or p.get("shape")
        shape_vals = parse_shape(shape, cfg)
        for li in range(n_layers):
            for e in range(n_exp):
                arr = np.zeros(shape_vals, dtype=np.float32)
                tensors.append((p["name"], arr))
    
    write_checkpoint(output_prefix, cfg, tensors)
    
    # Copy tokenizer if requested
    if copy_tokenizer:
        tok_info = mapping.get("tokenizer", {})
        model_base = os.path.dirname(model_dir.rstrip("/\\")) if not os.path.isfile(model_dir) else os.path.dirname(model_dir)
        model_dir_path = model_dir if os.path.isdir(model_dir) else os.path.dirname(model_dir)
        
        for fname in [tok_info.get("vocab_file"), tok_info.get("merges_file"), tok_info.get("model_file")]:
            if not fname: continue
            src = os.path.join(model_dir_path, fname)
            if os.path.exists(src):
                dst = f"{output_prefix}_{os.path.basename(fname)}"
                import shutil
                shutil.copy2(src, dst)
                print(f"  Copied {fname} → {dst}")
        
        print(f"\nTo train with this model:")
        bpe_prefix = f"{output_prefix}_"
        print(f"  train_gpt --load {output_prefix} --tokenizer bpe --bpe-vocab {bpe_prefix}vocab.json --bpe-merges {bpe_prefix}merges.txt --token-data <tokenized_data.bin>")


def main():
    parser = argparse.ArgumentParser(description="Import HuggingFace safetensors model")
    parser.add_argument("model_dir", help="Path to model directory (with config.json + .safetensors)")
    parser.add_argument("output_prefix", help="Output checkpoint prefix")
    parser.add_argument("--mapping", required=True, help="Mapping name (gpt2, llama)")
    parser.add_argument("--copy-tokenizer", action="store_true", help="Copy tokenizer files")
    args = parser.parse_args()
    
    if not HAVE_SAFETENSORS:
        print("Error: safetensors package not installed.")
        print("  pip install safetensors numpy")
        sys.exit(1)
    
    import_model(args.model_dir, args.output_prefix, args.mapping, args.copy_tokenizer)
    print("Done.")


if __name__ == "__main__":
    main()
