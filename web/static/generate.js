"use strict";

class GenUI {
  constructor() {
    this.modelLoaded = false;
    this.init();
  }

  async init() {
    await this.loadCheckpoints();
    document.getElementById("gen-temp").addEventListener("input", e => {
      document.getElementById("gen-temp-val").textContent = e.target.value;
    });
    document.getElementById("gen-topk").addEventListener("input", e => {
      document.getElementById("gen-topk-val").textContent = e.target.value;
    });
    document.getElementById("gen-tokens").addEventListener("input", e => {
      document.getElementById("gen-tokens-val").textContent = e.target.value;
    });
  }

  async loadCheckpoints() {
    try {
      const resp = await fetch("/api/play/checkpoints");
      const list = await resp.json();
      const sel = document.getElementById("gen-model-select");
      sel.innerHTML = '<option value="">-- Select checkpoint --</option>';
      for (const ckpt of list) {
        sel.innerHTML += `<option value="${ckpt.prefix}">${ckpt.name} (C=${ckpt.d_model}, L=${ckpt.n_layers}, step=${ckpt.step})</option>`;
      }
    } catch (e) { console.error("Failed to load checkpoints", e); }
  }

  async loadModel() {
    const prefix = document.getElementById("gen-model-select").value;
    if (!prefix) return alert("Select a checkpoint first");
    const btn = document.getElementById("gen-load-btn");
    btn.textContent = "Loading..."; btn.disabled = true;
    try {
      const resp = await fetch("/api/play/load", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint: prefix }),
      });
      const data = await resp.json();
      if (data.error) { alert(data.error); btn.disabled = false; btn.textContent = "Load"; return; }
      this.modelLoaded = true;
      btn.textContent = "Loaded ✓";
      document.getElementById("gen-generate-btn").disabled = false;
    } catch (e) { alert("Failed to load model"); btn.disabled = false; btn.textContent = "Load"; }
  }

  async generate() {
    const prompt = document.getElementById("gen-prompt").value.trim();
    if (!prompt) return alert("Enter a prompt");
    const temp = parseFloat(document.getElementById("gen-temp").value);
    const topk = parseInt(document.getElementById("gen-topk").value);
    const gen = parseInt(document.getElementById("gen-tokens").value);

    const output = document.getElementById("gen-output-text");
    output.textContent = "";
    document.getElementById("gen-stats").style.display = "none";

    const t0 = performance.now();
    let tokenCount = 0;

    try {
      const resp = await fetch("/api/play/generate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, temp, topk, gen }),
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        output.textContent += chunk;
        tokenCount++;
      }
    } catch (e) {
      output.textContent = "[Error generating]";
    }

    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    document.getElementById("gen-token-count").textContent = tokenCount;
    document.getElementById("gen-time").textContent = elapsed + "s";
    document.getElementById("gen-stats").style.display = "";
  }
}

const genUI = new GenUI();
