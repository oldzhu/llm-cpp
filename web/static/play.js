"use strict";

class PlayUI {
  constructor() {
    this.ws = null;
    this.init();
  }

  async init() {
    await this.loadCheckpoints();
    document.getElementById("play-temp").addEventListener("input", (e) => {
      document.getElementById("play-temp-val").textContent = e.target.value;
    });
    document.getElementById("play-topk").addEventListener("input", (e) => {
      document.getElementById("play-topk-val").textContent = e.target.value;
    });
    document.getElementById("play-gen").addEventListener("input", (e) => {
      document.getElementById("play-gen-val").textContent = e.target.value;
    });
    document.getElementById("play-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); this.send(); }
    });
  }

  async loadCheckpoints() {
    try {
      const resp = await fetch("/api/play/checkpoints");
      const list = await resp.json();
      const sel = document.getElementById("play-model-select");
      sel.innerHTML = '<option value="">-- Select checkpoint --</option>';
      for (const ckpt of list) {
        sel.innerHTML += `<option value="${ckpt.prefix}">${ckpt.name} (C=${ckpt.d_model}, L=${ckpt.n_layers}, step=${ckpt.step})</option>`;
      }
      sel.addEventListener("change", () => {
        const ckpt = list.find(c => c.prefix === sel.value);
        if (ckpt) {
          document.getElementById("play-model-info").innerHTML =
            `d_model=${ckpt.d_model}, layers=${ckpt.n_layers}, vocab=${ckpt.vocab_size}`;
        }
      });
    } catch (e) {
      console.error("Failed to load checkpoints", e);
    }
  }

  async loadModel() {
    const prefix = document.getElementById("play-model-select").value;
    if (!prefix) return alert("Select a checkpoint first");
    document.getElementById("play-load-btn").textContent = "Loading...";
    document.getElementById("play-load-btn").disabled = true;
    try {
      const resp = await fetch("/api/play/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint: prefix }),
      });
      const data = await resp.json();
      if (data.error) { alert(data.error); document.getElementById("play-load-btn").disabled = false; return; }
      document.getElementById("play-load-btn").textContent = "Loaded ✓";
      document.getElementById("play-send-btn").disabled = false;
      document.querySelector(".play-placeholder")?.remove();
    } catch (e) {
      alert("Failed to load model");
      document.getElementById("play-load-btn").textContent = "Load Model";
      document.getElementById("play-load-btn").disabled = false;
    }
  }

  async send() {
    const input = document.getElementById("play-input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    input.disabled = true;

    this.addMessage("user", text);
    const msgDiv = this.addMessage("assistant", "");

    const temp = parseFloat(document.getElementById("play-temp").value);
    const topk = parseInt(document.getElementById("play-topk").value);
    const gen = parseInt(document.getElementById("play-gen").value);

    try {
      const resp = await fetch("/api/play/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text, temp, topk, gen }),
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let full = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        full += chunk;
        msgDiv.querySelector(".play-text").textContent = full;
      }
    } catch (e) {
      msgDiv.querySelector(".play-text").textContent = "[Error generating]";
    }

    input.disabled = false;
    input.focus();
  }

  addMessage(role, content) {
    const container = document.getElementById("play-messages");
    const div = document.createElement("div");
    div.className = "play-message play-" + role;
    div.innerHTML = `<div class="play-role">${role === "user" ? "🧑 You" : "🤖 Model"}</div><div class="play-text">${content.replace(/&/g,"&amp;").replace(/</g,"&lt;")}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
  }

  clear() {
    document.getElementById("play-messages").innerHTML = '<div class="play-placeholder">Select a checkpoint and click Load Model to start chatting</div>';
  }
}

const playUI = new PlayUI();
