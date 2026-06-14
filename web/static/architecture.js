"use strict";

const ARCH_COMPONENTS = {
  tokenizer: {
    label_en: "Tokenizer", label_zh: "分词器",
    desc_en: "Converts text into token IDs. Byte tokenizer maps each byte (0-255) to its numeric value. BPE tokenizer uses merges to create subword tokens.",
    desc_zh: "将文本转换为token ID。字节分词器将每个字节(0-255)映射为其数值。BPE分词器使用合并规则创建子词token。",
    formula: "",
    file: "src/tokenizer/byte_tokenizer.cpp",
    shapes: "V=256 (byte) or V=50257 (BPE)"
  },
  embedding: {
    label_en: "Token + Position Embedding", label_zh: "Token + 位置嵌入",
    desc_en: "Token embedding: lookup wte[token_id] → [C] vector. Position embedding: add wpe[pos] for sequential order. Result: X [B,T,C].",
    desc_zh: "Token嵌入: 查表 wte[token_id] → [C]向量。位置嵌入: 加上wpe[pos]表示序列顺序。结果: X [B,T,C]。",
    formula: "X[b,t] = Wte[tokens[b,t]] + Wpe[t]",
    file: "src/model.cpp",
    shapes: "wte: [V,C], wpe: [Tmax,C], X: [B,T,C]"
  },
  layernorm: {
    label_en: "LayerNorm / RMSNorm", label_zh: "LayerNorm / RMSNorm",
    desc_en: "Normalizes each token's vector to zero mean and unit variance (LayerNorm) or unit RMS (RMSNorm), then applies learnable scale (γ) and shift (β).",
    desc_zh: "将每个token的向量归一化为零均值单位方差(LayerNorm)或单位RMS(RMSNorm)，然后应用可学习的缩放(γ)和平移(β)。",
    formula: "LN: y = (x-μ)/σ·γ+β\nRMS: y = x/rms·γ",
    file: "src/ops.cpp",
    shapes: "γ: [C], β: [C] (LN only)"
  },
  attention: {
    label_en: "Causal Self-Attention", label_zh: "因果自注意力",
    desc_en: "Each token looks at itself and all previous tokens (causal mask). Q,K,V projected from input, scores via dot product, weighted sum of values.",
    desc_zh: "每个token关注自身及之前所有token(因果掩码)。Q,K,V从输入投影，通过点积计算分数，加权求和值。",
    formula: "Q,K,V = X·Wqkv + bqkv\nS[i,j] = Q[i]·K[j]/√C + mask(j>i→-∞)\nP = softmax(S)\nY[i] = Σⱼ P[i,j]·V[j]",
    file: "src/ops.cpp:504 self_attention_1h()",
    shapes: "Wqkv: [C,3C], Q/K/V: [B,T,C], S: [B,T,T], P: [B,T,T]"
  },
  mlp: {
    label_en: "MLP / Feed-Forward", label_zh: "MLP / 前馈网络",
    desc_en: "Two-layer network with activation. Each token processed independently. GELU (standard), SwiGLU (gated), or MoE (multiple expert FFNs with router).",
    desc_zh: "带激活的双层网络。每个token独立处理。GELU(标准)、SwiGLU(门控)或MoE(多专家FFN+路由器)。",
    formula: "GELU: FF = GELU(X·Wfc)·Wout\nSwiGLU: gate = SiLU(X·Wg), up = X·Wu, FF = (gate⊙up)·Wd\nMoE: gate = topK(softmax(X·Wr)), FF = Σₑ gateₑ·Expertₑ(X)",
    file: "src/model.cpp",
    shapes: "GELU: Wfc[C,4C], Wout[4C,C]\nSwiGLU: Wgate[C,3C], Wup[C,3C], Wdown[3C,C]"
  },
  residual: {
    label_en: "Residual Connection (+)", label_zh: "残差连接 (+)",
    desc_en: "Adds the input of a sublayer to its output: X_new = X + Sublayer(X). Helps gradient flow in deep networks — prevents vanishing gradients.",
    desc_zh: "将子层的输入加到输出上: X_new = X + Sublayer(X)。有助于深层网络的梯度流动，防止梯度消失。",
    formula: "X ← X + Sublayer(LN(X))",
    file: "src/model.cpp",
    shapes: ""
  },
  lm_head: {
    label_en: "LM Head (Output)", label_zh: "LM头 (输出)",
    desc_en: "Projects the final hidden state to vocabulary-size logits. Each logit[i] represents the model's confidence that token i is the next token.",
    desc_zh: "将最终隐藏状态映射为词表大小的logits。每个logit[i]表示模型对token i为下一个token的置信度。",
    formula: "Logits = LN(X)·Wlm + blm",
    file: "src/model.cpp",
    shapes: "Wlm: [C,V], blm: [V], Logits: [B,T,V]"
  },
  loss: {
    label_en: "Cross-Entropy Loss", label_zh: "交叉熵损失",
    desc_en: "Measures how well the model predicts the actual next token. L = -log(p_correct). Lower = better. Random = ln(V).",
    desc_zh: "衡量模型预测实际下一个token的准确度。L = -log(p_正确)。越低越好。随机≈ln(V)。",
    formula: "L = -1/N Σₙ log(p_yn), where p = softmax(logits[n])",
    file: "src/ops.cpp:654 cross_entropy()",
    shapes: ""
  }
};

class ArchitectureDiagram {
  constructor() {
    this.svgNS = "http://www.w3.org/2000/svg";
    this.selectedComponent = null;
  }

  render(container) {
    container.innerHTML = "";
    const svg = document.createElementNS(this.svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 640 720");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "auto");
    svg.classList.add("arch-svg");
    container.appendChild(svg);

    let y = 15;
    const w = 600, cx = 320;

    // Title
    this._addText(svg, cx, y, t("arch.title"), "arch-title");
    y += 30;

    // === Tokenizer ===
    this._addBox(svg, 200, y, 240, 44, "tokenizer");
    y += 60;

    // Arrow
    this._addArrow(svg, cx, y-16, cx, y);
    y += 10;

    // === Embedding ===
    this._addBox(svg, 180, y, 280, 44, "embedding");
    y += 60;

    // Arrow
    this._addArrow(svg, cx, y-16, cx, y);
    y += 10;

    // === Layer Block ===
    this._addBox(svg, 30, y, 580, 24, null, t("arch.transformer_block"), "arch-subtitle", "arch-block-header");
    y += 36;

    // Attention sublayer
    this._addBox(svg, 50, y, 100, 36, "layernorm");
    this._addArrowSVG(svg, 155, y+18, 185, y+18);
    this._addBox(svg, 190, y, 230, 36, "attention");
    this._addArrowSVG(svg, 425, y+18, 460, y+18);
    this._addBox(svg, 465, y, 24, 36, "residual");
    y += 52;

    // MLP sublayer
    this._addBox(svg, 50, y, 100, 36, "layernorm");
    this._addArrowSVG(svg, 155, y+18, 185, y+18);
    this._addBox(svg, 190, y, 230, 36, "mlp");
    this._addArrowSVG(svg, 425, y+18, 460, y+18);
    this._addBox(svg, 465, y, 24, 36, "residual");
    y += 60;

    // Exit arrow from block
    this._addArrow(svg, cx, y-8, cx, y);
    y += 10;

    // === Final LN + LM Head ===
    this._addBox(svg, 200, y, 100, 36, "layernorm");
    this._addArrowSVG(svg, 305, y+18, 335, y+18);
    this._addBox(svg, 340, y, 100, 36, "lm_head");
    y += 56;

    // Arrow to loss
    this._addArrow(svg, cx, y-8, cx, y);
    y += 10;

    // === Loss ===
    this._addBox(svg, 220, y, 200, 36, "loss");

    // === Legend ===
    y += 60;
    this._addText(svg, cx, y, t("arch.legend"), "arch-subtitle");
    y += 20;
    this._addText(svg, cx, y, t("arch.legend_text"), "arch-legend");
  }

  _addBox(svg, x, y, w, h, componentId, label, cls, extraCls) {
    const g = document.createElementNS(this.svgNS, "g");
    g.setAttribute("transform", `translate(${x},${y})`);
    if (componentId) {
      g.classList.add("arch-component");
      g.setAttribute("data-component", componentId);
      g.addEventListener("click", () => this._onClick(componentId));
      g.addEventListener("mouseenter", (e) => this._showTooltip(e, componentId));
      g.addEventListener("mouseleave", () => this._hideTooltip());

      const comp = ARCH_COMPONENTS[componentId];
      const name = currentLang === "zh" ? comp.label_zh : comp.label_en;
      // Colored rect
      const rect = document.createElementNS(this.svgNS, "rect");
      rect.setAttribute("x", 0); rect.setAttribute("y", 0);
      rect.setAttribute("width", w); rect.setAttribute("height", h);
      rect.setAttribute("rx", 6); rect.classList.add("arch-rect");
      g.appendChild(rect);

      // Text
      const text = document.createElementNS(this.svgNS, "text");
      text.setAttribute("x", w/2); text.setAttribute("y", h/2 + 5);
      text.classList.add("arch-text");
      text.textContent = name;
      g.appendChild(text);
    } else if (cls === "arch-subtitle") {
      const text = document.createElementNS(this.svgNS, "text");
      text.setAttribute("x", w/2); text.setAttribute("y", h/2 + 5);
      text.classList.add(cls, extraCls || "");
      text.textContent = label || "";
      g.appendChild(text);
    }
    svg.appendChild(g);
  }

  _addText(svg, x, y, label, cls) {
    const text = document.createElementNS(this.svgNS, "text");
    text.setAttribute("x", x); text.setAttribute("y", y);
    text.setAttribute("text-anchor", "middle");
    text.classList.add(cls);
    text.textContent = label;
    svg.appendChild(text);
  }

  _addArrow(svg, x1, y1, x2, y2) {
    const line = document.createElementNS(this.svgNS, "line");
    line.setAttribute("x1", x1); line.setAttribute("y1", y1);
    line.setAttribute("x2", x2); line.setAttribute("y2", y2);
    line.classList.add("arch-arrow");
    svg.appendChild(line);
    // arrowhead
    const poly = document.createElementNS(this.svgNS, "polygon");
    poly.setAttribute("points", `${x2-5},${y2-4} ${x2},${y2} ${x2-5},${y2+4}`);
    poly.classList.add("arch-arrowhead");
    svg.appendChild(poly);
  }

  _addArrowSVG(svg, x1, y1, x2, y2) {
    this._addArrow(svg, x1, y1, x2, y2);
  }

  _onClick(id) {
    const comp = ARCH_COMPONENTS[id];
    if (!comp || !comp.file) return;
    // Extract file path and line number
    const parts = comp.file.split(":");
    const path = parts[0];
    const line = parts[1] ? parseInt(parts[1]) : 0;
    if (typeof codeUI !== "undefined" && codeUI.loadFile) {
      codeUI.loadFile(path).then(() => {
        if (line > 0) {
          const el = document.getElementById("L"+line);
          if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      });
    }
  }

  _showTooltip(e, id) {
    let tip = document.getElementById("arch-tooltip");
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "arch-tooltip";
      tip.className = "arch-tooltip-popup";
      document.body.appendChild(tip);
    }
    const comp = ARCH_COMPONENTS[id];
    if (!comp) return;
    const lang = currentLang || "en";
    tip.innerHTML = `
      <div class="arch-tooltip-title">${lang === "zh" ? comp.label_zh : comp.label_en}</div>
      <div class="arch-tooltip-desc">${lang === "zh" ? comp.desc_zh : comp.desc_en}</div>
      ${comp.formula ? `<div class="arch-tooltip-formula"><code>${comp.formula.replace(/\n/g,"<br>")}</code></div>` : ""}
      ${comp.shapes ? `<div class="arch-tooltip-shapes"><span class="arch-tooltip-label">Shapes:</span> ${comp.shapes}</div>` : ""}
      <div class="arch-tooltip-file">📄 ${comp.file}</div>
    `;
    tip.style.left = (e.pageX + 15) + "px";
    tip.style.top = (e.pageY - 10) + "px";
    tip.style.display = "block";
  }

  _hideTooltip() {
    const tip = document.getElementById("arch-tooltip");
    if (tip) tip.style.display = "none";
  }
}

const archDiagram = new ArchitectureDiagram();
