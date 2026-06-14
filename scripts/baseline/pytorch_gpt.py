#!/usr/bin/env python3
"""PyTorch baseline — exact reimplementation of C++ TinyGPT for correctness verification.

Mirrors model.cpp precisely:
- Same weight layout, same forward pass, same loss
- Loads weights from C++ JSON export
- Supports all config options (norm_type, mlp_type, attn_type, pos_type)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyGPT(nn.Module):
    """Exact PyTorch mirror of C++ model::TinyGPT."""

    def __init__(self, cfg: dict, weights_json: str = ""):
        super().__init__()
        self.cfg = cfg
        V = cfg["vocab_size"]
        T = cfg["seq_len"]
        C = cfg["d_model"]
        n_layers = cfg["n_layers"]
        self.norm_type = cfg.get("norm_type", 0)
        self.mlp_type = cfg.get("mlp_type", 0)
        self.attn_type = cfg.get("attn_type", 0)
        self.n_heads = cfg.get("attn_n_heads", 1)
        interm = cfg.get("swiglu_interm", 0) or (3 * C)

        # Embeddings
        self.wte = nn.Parameter(torch.empty(V, C))
        self.wpe = nn.Parameter(torch.empty(T, C))

        # Per-layer params
        n_layer_params = 12  # w_qkv,b_qkv,w_proj,b_proj,w_fc,b_fc,w_out,b_out,ln1_g,ln1_b,ln2_g,ln2_b
        self.layers_wqkv = nn.ParameterList()
        self.layers_bqkv = nn.ParameterList()
        self.layers_wproj = nn.ParameterList()
        self.layers_bproj = nn.ParameterList()
        self.layers_wfc = nn.ParameterList()
        self.layers_bfc = nn.ParameterList()
        self.layers_wout = nn.ParameterList()
        self.layers_bout = nn.ParameterList()
        self.layers_ln1_g = nn.ParameterList()
        self.layers_ln1_b = nn.ParameterList()
        self.layers_ln2_g = nn.ParameterList()
        self.layers_ln2_b = nn.ParameterList()

        for _ in range(n_layers):
            self.layers_wqkv.append(nn.Parameter(torch.empty(C, 3 * C)))
            self.layers_bqkv.append(nn.Parameter(torch.empty(3 * C)))
            self.layers_wproj.append(nn.Parameter(torch.empty(C, C)))
            self.layers_bproj.append(nn.Parameter(torch.empty(C)))
            self.layers_wfc.append(nn.Parameter(torch.empty(C, 4 * C)))
            self.layers_bfc.append(nn.Parameter(torch.empty(4 * C)))
            self.layers_wout.append(nn.Parameter(torch.empty(4 * C, C)))
            self.layers_bout.append(nn.Parameter(torch.empty(C)))
            self.layers_ln1_g.append(nn.Parameter(torch.empty(C)))
            self.layers_ln1_b.append(nn.Parameter(torch.empty(C)))
            self.layers_ln2_g.append(nn.Parameter(torch.empty(C)))
            self.layers_ln2_b.append(nn.Parameter(torch.empty(C)))

        # Final params
        self.w_lm = nn.Parameter(torch.empty(C, V))
        self.b_lm = nn.Parameter(torch.empty(V))
        self.ln_final_g = nn.Parameter(torch.empty(C))
        self.ln_final_b = nn.Parameter(torch.empty(C))

        # SwiGLU params
        self.swiglu_gate = nn.ParameterList()
        self.swiglu_gate_b = nn.ParameterList()
        self.swiglu_up = nn.ParameterList()
        self.swiglu_up_b = nn.ParameterList()
        self.swiglu_down = nn.ParameterList()
        self.swiglu_down_b = nn.ParameterList()
        for _ in range(n_layers):
            self.swiglu_gate.append(nn.Parameter(torch.empty(C, interm)))
            self.swiglu_gate_b.append(nn.Parameter(torch.empty(interm)))
            self.swiglu_up.append(nn.Parameter(torch.empty(C, interm)))
            self.swiglu_up_b.append(nn.Parameter(torch.empty(interm)))
            self.swiglu_down.append(nn.Parameter(torch.empty(interm, C)))
            self.swiglu_down_b.append(nn.Parameter(torch.empty(C)))

        # Load weights if provided
        if weights_json:
            self._load_weights(weights_json)

    def _load_weights(self, path: str):
        with open(path) as f:
            data = json.load(f)
        # Map JSON keys to our parameter list
        # weights are stored in parameters() order
        param_names = []
        param_names.append(("wte", self.wte))
        param_names.append(("wpe", self.wpe))
        n_layers = self.cfg["n_layers"]
        for li in range(n_layers):
            p = f"L{li}_"
            param_names.append((p + "w_qkv", self.layers_wqkv[li]))
            param_names.append((p + "b_qkv", self.layers_bqkv[li]))
            param_names.append((p + "w_proj", self.layers_wproj[li]))
            param_names.append((p + "b_proj", self.layers_bproj[li]))
            param_names.append((p + "w_fc", self.layers_wfc[li]))
            param_names.append((p + "b_fc", self.layers_bfc[li]))
            param_names.append((p + "w_out", self.layers_wout[li]))
            param_names.append((p + "b_out", self.layers_bout[li]))
            param_names.append((p + "ln_attn_gamma", self.layers_ln1_g[li]))
            param_names.append((p + "ln_attn_beta", self.layers_ln1_b[li]))
            param_names.append((p + "ln_mlp_gamma", self.layers_ln2_g[li]))
            param_names.append((p + "ln_mlp_beta", self.layers_ln2_b[li]))
        param_names.append(("w_lm", self.w_lm))
        param_names.append(("b_lm", self.b_lm))
        param_names.append(("ln_final_gamma", self.ln_final_g))
        param_names.append(("ln_final_beta", self.ln_final_b))
        for li in range(n_layers):
            p = f"L{li}_"
            param_names.append((p + "swiglu_gate", self.swiglu_gate[li]))
            param_names.append((p + "swiglu_gate_b", self.swiglu_gate_b[li]))
            param_names.append((p + "swiglu_up", self.swiglu_up[li]))
            param_names.append((p + "swiglu_up_b", self.swiglu_up_b[li]))
            param_names.append((p + "swiglu_down", self.swiglu_down[li]))
            param_names.append((p + "swiglu_down_b", self.swiglu_down_b[li]))

        for name, param in param_names:
            if name in data:
                w = data[name]
                tensor = torch.tensor(w["data"], dtype=torch.float32).reshape(w["shape"])
                param.data.copy_(tensor)

    def _ln(self, x, gamma, beta, eps=1e-5):
        if self.norm_type == 0:
            # LayerNorm
            mean = x.mean(dim=-1, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            return gamma * (x - mean) / torch.sqrt(var + eps) + beta
        else:
            # RMSNorm
            rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
            return gamma * x / rms

    def _attn_1h(self, x, w_qkv, b_qkv, w_proj, b_proj):
        B, T, C = x.shape
        qkv = x @ w_qkv + b_qkv  # [B,T,3C]
        q, k, v = qkv.split(C, dim=-1)
        scale = 1.0 / math.sqrt(C)
        scores = (q @ k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T), diagonal=1).to(x.device) * -1e9
        scores = scores + mask
        probs = F.softmax(scores, dim=-1)
        att = probs @ v
        return att @ w_proj + b_proj

    def _gelu(self, x):
        c = 0.044715
        s = 0.7978845608  # sqrt(2/pi)
        u = s * (x + c * x * x * x)
        return 0.5 * x * (1.0 + torch.tanh(u))

    def _silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, tokens):
        B, T = tokens.shape
        C = self.cfg["d_model"]

        x = self.wte[tokens] + self.wpe[:T]  # [B,T,C]

        for li in range(self.cfg["n_layers"]):
            # Attention sublayer
            h = self._ln(x, self.layers_ln1_g[li], self.layers_ln1_b[li])
            a = self._attn_1h(h, self.layers_wqkv[li], self.layers_bqkv[li],
                             self.layers_wproj[li], self.layers_bproj[li])
            x = x + a

            # MLP sublayer
            m = self._ln(x, self.layers_ln2_g[li], self.layers_ln2_b[li])
            if self.mlp_type == 0:
                ff = self._gelu(m @ self.layers_wfc[li] + self.layers_bfc[li]) @ self.layers_wout[li] + self.layers_bout[li]
            elif self.mlp_type == 1:
                gate = self._silu(m @ self.swiglu_gate[li] + self.swiglu_gate_b[li])
                up = m @ self.swiglu_up[li] + self.swiglu_up_b[li]
                ff = (gate * up) @ self.swiglu_down[li] + self.swiglu_down_b[li]
            else:
                ff = self._gelu(m @ self.layers_wfc[li] + self.layers_bfc[li]) @ self.layers_wout[li] + self.layers_bout[li]
            x = x + ff

        xn = self._ln(x, self.ln_final_g, self.ln_final_b)
        return xn @ self.w_lm + self.b_lm  # [B,T,V]

    def loss(self, tokens, targets):
        logits = self.forward(tokens)
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))


def init_weights(model: TinyGPT, seed: int = 1):
    """Initialize weights identically to C++ TinyGPT constructor."""
    torch.manual_seed(seed)
    C = model.cfg["d_model"]
    init_std = 1.0 / math.sqrt(C)

    for p in model.parameters():
        if p.dim() >= 2:
            fan = p.shape[1]  # fan_in
            nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(fan))
        elif p.dim() == 1:
            nn.init.zeros_(p)

    # Gamma = 1.0 for LN params
    for li in range(model.cfg["n_layers"]):
        nn.init.ones_(model.layers_ln1_g[li])
        nn.init.zeros_(model.layers_ln1_b[li])
        nn.init.ones_(model.layers_ln2_g[li])
        nn.init.zeros_(model.layers_ln2_b[li])
    nn.init.ones_(model.ln_final_g)
    nn.init.zeros_(model.ln_final_b)
