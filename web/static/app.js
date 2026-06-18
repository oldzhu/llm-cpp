"use strict";

class WebUI {
  constructor() {
    this.ws = null;
    this.configSchema = {};
    this.presets = {};
    this.stepTimes = [];
    this.init();
  }

  async init() {
    await this.loadSchema();
    this.buildConfigForm();
    this.setupTabs();
    this.setupButtons();
    this.connectWebSocket();
    this.updateStatus("idle");
  }

  async loadSchema() {
    const [s, p] = await Promise.all([
      fetch("/api/config/schema").then(r => r.json()),
      fetch("/api/config/presets").then(r => r.json()),
    ]);
    this.configSchema = s;
    this.presets = p;
  }

  buildConfigForm() {
    const groups = {};
    for (const [key, spec] of Object.entries(this.configSchema)) {
      const g = spec.group || "General";
      if (!groups[g]) groups[g] = [];
      groups[g].push({ key, ...spec });
    }
    const container = document.getElementById("config-groups");
    container.innerHTML = "";
    for (const [groupName, fields] of Object.entries(groups)) {
      const div = document.createElement("div");
      div.className = "config-group";
      const h4 = document.createElement("h4");
      h4.textContent = t("group." + groupName);
      div.appendChild(h4);
      for (const f of fields) {
        const row = document.createElement("div");
        row.className = "config-row";
        if (f.show_if) {
          const cond = Object.entries(f.show_if).map(([k,v]) => {
            const val = Array.isArray(v) ? v.join("|") : v;
            return `[name="${k}"][value="${val}"]`;
          }).join(",");
          row.dataset.showIf = cond;
        }
        let input = "";
        if (f.type === "choice") {
          input = `<select name="${f.key}">${f.options.map(o => `<option value="${o}" ${o===f.default?'selected':''}>${o}</option>`).join("")}</select>`;
        } else if (f.type === "bool") {
          input = `<input type="checkbox" name="${f.key}" ${f.default ? 'checked' : ''}>`;
        } else {
          input = `<input type="${f.type==='int'||f.type==='float'?'number':'text'}" name="${f.key}" value="${f.default}" ${f.min!==undefined?`min="${f.min}"`:''} ${f.max!==undefined?`max="${f.max}"`:''} step="${f.type==='float'?'0.0001':'1'}">`;
        }
        const label = document.createElement("label");
        label.textContent = t("field." + f.key);
        row.appendChild(label);
        row.insertAdjacentHTML("beforeend", input);
        div.appendChild(row);
      }
      container.appendChild(div);
    }
    this.setupConditionalFields();
    this.applyTranslations();
    this.injectTooltips();
  }

  injectTooltips() {
    // Add tooltips to config form labels
    document.querySelectorAll(".config-row label").forEach(lbl => {
      const input = lbl.nextElementSibling;
      const key = input?.name;
      if (key) {
        const tipKey = "tooltip." + key;
        const tipText = t(tipKey);
        if (tipText !== tipKey) {
          lbl.parentElement.classList.add("tooltip-container");
          lbl.setAttribute("data-tooltip", tipText);
        }
      }
    });
    // Add tooltips to metric cards
    document.querySelectorAll(".metric-label").forEach(lbl => {
      const text = lbl.textContent.trim();
      const enMap = { "Loss": "loss", "Loss/sec": "loss", "Step": "", "Time": "" };
      const zhMap = { "损失": "loss", "损失/秒": "loss", "步数": "", "时间": "" };
      const mkey = enMap[text] || zhMap[text] || "";
      if (mkey) {
        const tipText = t("tooltip.monitor." + mkey);
        if (tipText !== "tooltip.monitor." + mkey) {
          lbl.parentElement.classList.add("tooltip-container");
          lbl.setAttribute("data-tooltip", tipText);
        }
      }
    });
    // Add tooltips to weight table headers
    document.querySelectorAll("#weights-table th").forEach(th => {
      const text = th.textContent.trim();
      const map = { "Mean": "mean", "均值": "mean", "RMS": "rms", "Min": "", "Max": "", "最小": "", "最大": "", "Parameter": "", "参数": "" };
      const wkey = map[text];
      if (wkey) {
        const tipText = t("tooltip.monitor." + wkey);
        if (tipText !== "tooltip.monitor." + wkey) {
          th.classList.add("tooltip-container");
          th.setAttribute("data-tooltip", tipText);
        }
      }
    });
    // Create tooltip elements for hover
    document.querySelectorAll("[data-tooltip]").forEach(el => {
      if (el.querySelector(".tooltip-popup")) return;
      const popup = document.createElement("span");
      popup.className = "tooltip-popup";
      popup.textContent = el.getAttribute("data-tooltip");
      el.appendChild(popup);
    });
  }

  applyTranslations() {
    // Re-render config group headers and field labels
    document.querySelectorAll(".config-group h4").forEach(h4 => {
      const groupName = h4.textContent;
      // Don't re-translate if it was already translated
      h4.textContent = t("group." + Object.keys(I18N.en).find(k => I18N.en[k] === groupName && k.startsWith("group.")) || groupName);
    });
    // Force re-render config form
    if (Object.keys(this.configSchema).length > 0) {
      const groups = {};
      for (const [key, spec] of Object.entries(this.configSchema)) {
        const g = spec.group || "General";
        if (!groups[g]) groups[g] = [];
        groups[g].push({ key, ...spec });
      }
      const h4s = document.querySelectorAll(".config-group h4");
      let idx = 0;
      for (const [groupName] of Object.entries(groups)) {
        if (idx < h4s.length) h4s[idx].textContent = t("group." + groupName);
        idx++;
      }
      // Update field labels
      document.querySelectorAll(".config-row label").forEach(lbl => {
        // Find the matching field key by looking at the next sibling input/select
        const input = lbl.nextElementSibling;
        if (input && input.name) {
          lbl.textContent = t("field." + input.name);
        }
      });
    }
  }

  setupConditionalFields() {
    const selects = document.querySelectorAll("#config-form select");
    selects.forEach(sel => {
      sel.addEventListener("change", () => {
        document.querySelectorAll(".config-row[data-show-if]").forEach(row => {
          const cond = row.dataset.showIf;
          // Simple check: does any select match the condition
          let visible = false;
          if (cond) {
            const parts = cond.split(",");
            for (const part of parts) {
              const m = part.match(/\[name="(\w+)"\]\[value="(\w+)"\]/);
              if (m) {
                const el = document.querySelector(`[name="${m[1]}"]`);
                if (el) {
                  const vals = m[2].split("|");
                  if (vals.includes(el.value)) { visible = true; break; }
                }
              }
            }
          }
          row.classList.toggle("hidden", !visible);
        });
      });
      sel.dispatchEvent(new Event("change"));
    });
  }

  setupTabs() {
    document.querySelectorAll(".tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
        tab.classList.add("active");
        const target = document.getElementById(`tab-${tab.dataset.tab}`);
        if (target) target.classList.add("active");
        // Show/hide gear icons
        const gcode = document.getElementById("settings-gear-code");
        const gchat = document.getElementById("settings-gear-chat");
        if (gcode) gcode.style.display = (tab.dataset.tab === "code") ? "" : "none";
        if (gchat) gchat.style.display = (tab.dataset.tab === "chat") ? "" : "none";
      });
    });
    // Gear icons open settings
    document.getElementById("settings-gear-code")?.addEventListener("click", () => aiSettings.showModal());
    document.getElementById("settings-gear-chat")?.addEventListener("click", () => aiSettings.showModal());
    document.querySelectorAll(".preset-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const preset = this.presets[btn.dataset.preset];
        if (preset) {
          for (const [key, val] of Object.entries(preset)) {
            const el = document.querySelector(`[name="${key}"]`);
            if (el) {
              if (el.type === "checkbox") el.checked = val;
              else el.value = val;
            }
          }
          // trigger conditional field update
          document.querySelector("#config-form select")?.dispatchEvent(new Event("change"));
        }
      });
    });
  }

  setupButtons() {
    document.getElementById("btn-start").addEventListener("click", () => this.startTraining());
    document.getElementById("btn-stop").addEventListener("click", () => this.stopTraining());
  }

  getConfig() {
    const cfg = {};
    document.querySelectorAll("#config-form [name]").forEach(el => {
      if (el.type === "checkbox") cfg[el.name] = el.checked;
      else if (el.type === "number") cfg[el.name] = parseFloat(el.value) || 0;
      else cfg[el.name] = el.value;
    });
    return cfg;
  }

  async startTraining() {
    const config = this.getConfig();
    const resp = await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    });
    const result = await resp.json();
    if (result.error) {
      alert(result.error);
      return;
    }
    this.updateStatus("training");
    document.getElementById("btn-start").disabled = true;
    document.getElementById("btn-stop").disabled = false;
    document.getElementById("train-status").textContent = t("status.launching");
    this.stepTimes = [];
    window.chart?.reset();
  }

  async stopTraining() {
    document.getElementById("train-status").textContent = t("status.stopping");
    const resp = await fetch("/api/train/stop", { method: "POST" });
    const result = await resp.json();
    document.getElementById("train-status").textContent = result.error
      ? `Error: ${result.error}`
      : t("status.saved", {step: result.step});
    this.updateStatus("idle");
    document.getElementById("btn-start").disabled = false;
    document.getElementById("btn-stop").disabled = true;
  }

  connectWebSocket() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws/train`);
    this.ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === "step") {
        this.onStep(data);
      } else if (data.type === "status") {
        this.updateStatus(data.state);
        if (data.state === "idle") {
          document.getElementById("btn-start").disabled = false;
          document.getElementById("btn-stop").disabled = true;
        }
      }
    };
    this.ws.onclose = () => {
      this.updateStatus("idle");
      setTimeout(() => this.connectWebSocket(), 2000);
    };
  }

  onStep(data) {
    const step = data.step || 0;
    const total = data.total_steps || 100;
    const loss = data.loss || 0;
    const time = data.time || 0;
    const pct = Math.min(100, Math.round((step / total) * 100));

    document.getElementById("train-progress").value = pct;
    document.getElementById("step-label").textContent = `${step} / ${total}`;
    document.getElementById("loss-value").textContent = loss.toFixed(4);
    document.getElementById("step-value").textContent = step;
    document.getElementById("time-value").textContent = time.toFixed(1) + "s";
    document.getElementById("steps-sec").textContent = (step / Math.max(time, 0.01)).toFixed(1);

    this.stepTimes.push({ step, time });
    if (this.stepTimes.length > 2) {
      const dt = this.stepTimes[this.stepTimes.length - 1].time - this.stepTimes[0].time;
      const ds = this.stepTimes[this.stepTimes.length - 1].step - this.stepTimes[0].step;
      const rate = ds / Math.max(dt, 0.01);
      const remaining = total - step;
      const eta = remaining / Math.max(rate, 0.01);
      const min = Math.floor(eta / 60);
      const sec = Math.floor(eta % 60);
      document.getElementById("eta-label").textContent = t("monitor.eta") + ` ${min}m ${sec}s`;
    }

    document.getElementById("train-status").textContent = `Training... loss=${loss.toFixed(4)}`;

    if (data.w_stats) {
      this.updateWeightTable(data.w_stats);
    }

    window.chart?.update(step, loss);
  }

  updateWeightTable(stats) {
    const tbody = document.querySelector("#weights-table tbody");
    tbody.innerHTML = "";
    for (const [name, s] of Object.entries(stats)) {
      if (!name) continue;
      tbody.innerHTML += `<tr>
        <td>${name}</td>
        <td>${s.mean.toFixed(6)}</td>
        <td>${s.rms.toFixed(6)}</td>
        <td>${s.min.toFixed(6)}</td>
        <td>${s.max.toFixed(6)}</td>
      </tr>`;
    }
  }

  updateStatus(state) {
    const dot = document.getElementById("status-indicator");
    const text = document.getElementById("status-text");
    dot.className = "dot " + state;
    if (state === "idle") state = "ready";
    text.textContent = t("status." + state);
  }
}

// Global zoom functionality
(function() {
  const content = document.getElementById("content");
  const levelEl = document.getElementById("zoom-level");
  let zoom = parseFloat(localStorage.getItem("zoom") || "1.0");
  const STEP = 0.1, MIN = 0.5, MAX = 2.5;

  function applyZoom() {
    if (!content || !levelEl) return;
    content.style.transform = `scale(${zoom})`;
    levelEl.textContent = Math.round(zoom * 100) + "%";
    localStorage.setItem("zoom", zoom.toFixed(1));
  }
  applyZoom();

  document.getElementById("zoom-in")?.addEventListener("click", () => {
    if (zoom < MAX) { zoom += STEP; applyZoom(); }
  });
  document.getElementById("zoom-out")?.addEventListener("click", () => {
    if (zoom > MIN) { zoom -= STEP; applyZoom(); }
  });
  document.getElementById("zoom-reset")?.addEventListener("click", () => {
    zoom = 1.0; applyZoom();
  });

  // Shift + mouse wheel zoom
  window.addEventListener("wheel", (e) => {
    if (!e.shiftKey) return;
    e.preventDefault();
    if (e.deltaY < 0 && zoom < MAX) zoom += STEP;
    else if (e.deltaY > 0 && zoom > MIN) zoom -= STEP;
    applyZoom();
  }, { passive: false });
})();

// Global helper: switch to a tab and load a code file
function openInCode(filePath) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  const codeTab = document.querySelector('[data-tab="code"]');
  const codePanel = document.getElementById("tab-code");
  if (codeTab) codeTab.classList.add("active");
  if (codePanel) codePanel.classList.add("active");
  // Also show gear icon
  const gcode = document.getElementById("settings-gear-code");
  const gchat = document.getElementById("settings-gear-chat");
  if (gcode) gcode.style.display = "";
  if (gchat) gchat.style.display = "none";
  // Load file
  if (typeof codeUI !== "undefined" && filePath) {
    codeUI.loadFile(filePath);
  }
}
window.openInCode = openInCode;

const ui = new WebUI();
