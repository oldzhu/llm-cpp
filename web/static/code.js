"use strict";

class CodeUI {
  constructor() {
    this.currentFile = "";
    this.currentLines = [];
    this.selectedStart = -1;
    this.selectedEnd = -1;
    this.init();
  }

  async init() {
    await this.loadTree();
    this.renderDiagram();
    this.setupSelection();
  }

  renderDiagram() {
    const container = document.getElementById("arch-diagram-container");
    if (container && typeof archDiagram !== "undefined") {
      archDiagram.render(container);
    }
  }

  _aiSettings() { return aiSettings ? aiSettings.data : {}; }

  async loadTree() {
    try {
      const tree = await fetch("/api/code/tree").then(r => r.json());
      this.renderTree(tree, document.getElementById("file-tree"));
    } catch (e) {
      document.getElementById("file-tree").innerHTML = "Error loading file tree";
    }
  }

  renderTree(tree, container) {
    container.innerHTML = "";
    if (!Array.isArray(tree)) {
      for (const [group, items] of Object.entries(tree)) {
        const gdiv = document.createElement("div");
        gdiv.className = "tree-group";
        gdiv.innerHTML = `<div class="tree-group-name">▼ ${group}</div>`;
        const children = document.createElement("div");
        children.className = "tree-children";
        this.renderEntries(Array.isArray(items) ? items : [], children);
        gdiv.appendChild(children);
        gdiv.querySelector(".tree-group-name").addEventListener("click", function() {
          const c = this.nextElementSibling;
          c.classList.toggle("hidden");
          this.textContent = c.classList.contains("hidden")
            ? this.textContent.replace("▼", "▶") : this.textContent.replace("▶", "▼");
        });
        container.appendChild(gdiv);
      }
      return;
    }
    this.renderEntries(tree, container);
  }

  renderEntries(entries, container) {
    for (const entry of entries) {
      const div = document.createElement("div");
      div.className = "tree-entry";
      if (entry.type === "dir") {
        div.innerHTML = `<span class="tree-toggle">▶</span> 📁 ${entry.name}`;
        const children = document.createElement("div");
        children.className = "tree-children hidden";
        if (entry.children) this.renderEntries(entry.children, children);
        div.appendChild(children);
        div.querySelector(".tree-toggle").addEventListener("click", (e) => {
          e.stopPropagation();
          children.classList.toggle("hidden");
          div.querySelector(".tree-toggle").textContent = children.classList.contains("hidden") ? "▶" : "▼";
        });
      } else {
        div.innerHTML = `📄 ${entry.name}`;
        div.addEventListener("click", () => this.loadFile(entry.path));
      }
      container.appendChild(div);
    }
  }

  async loadFile(path) {
    try {
      const data = await fetch(`/api/code/file?path=${encodeURIComponent(path)}`).then(r => r.json());
      if (data.error) { alert(data.error); return; }
      this.currentFile = data.path;
      this.currentLines = data.lines;
      document.getElementById("code-filename").textContent = path;
      document.getElementById("code-lines-info").textContent = `${data.total_lines} lines`;
      document.getElementById("code-actions").style.display = "flex";
      document.getElementById("code-viewer").innerHTML = `<pre class="code-content">${this.renderHighlighted(data.lines)}</pre>`;
      document.getElementById("selection-info").textContent = t("code.selectHelp");
      this.selectedStart = -1; this.selectedEnd = -1;
    } catch (e) { alert("Failed to load file"); }
  }

  renderHighlighted(lines) {
    let html = "";
    for (let i = 0; i < lines.length; i++) {
      const num = i + 1;
      const esc = lines[i].replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
      html += `<span class="code-line" data-line="${num}" id="L${num}"><span class="line-num">${String(num).padStart(4)}</span> ${esc}\n</span>`;
    }
    return html;
  }

  setupSelection() {
    document.addEventListener("click", (e) => {
      const lineEl = e.target.closest(".code-line");
      if (!lineEl) return;
      const num = parseInt(lineEl.dataset.line);
      if (e.shiftKey && this.selectedStart > 0) {
        this.selectedEnd = num; this.highlightSelection();
      } else {
        this.selectedStart = num; this.selectedEnd = num; this.highlightSelection();
      }
    });
  }

  highlightSelection() {
    if (this.selectedStart < 0) return;
    const s = Math.min(this.selectedStart, this.selectedEnd);
    const e = Math.max(this.selectedStart, this.selectedEnd);
    document.querySelectorAll(".code-line.selected").forEach(el => el.classList.remove("selected"));
    for (let i = s; i <= e; i++) { const el = document.getElementById("L"+i); if (el) el.classList.add("selected"); }
    document.getElementById("selection-info").textContent = t("code.selected", {s:String(s), e:String(e)});
  }

  renderMarkdown(text) {
    return text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
      .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="md-code"><code>$2</code></pre>')
      .replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/\n/g,'<br>');
  }

  async _aiCall(url, body) {
    body.settings = this._aiSettings();
    document.getElementById("ai-response").innerHTML = `<div class="ai-loading">${t("code.thinking")}</div>`;
    try {
      const r = await fetch(url, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) });
      const d = await r.json();
      if (d.error) document.getElementById("ai-response").innerHTML = `<div class="ai-error">${d.error}</div>`;
      else document.getElementById("ai-response").innerHTML = `<div class="ai-message"><div class="ai-role">🤖 AI</div><div class="ai-text">${this.renderMarkdown(d.content)}</div></div>`;
    } catch(e) {
      document.getElementById("ai-response").innerHTML = `<div class="ai-error">${t("code.connErr")}</div>`;
    }
  }

  async explainSelection() {
    if (this.selectedStart < 0) { alert(t("code.select")); return; }
    const s = Math.min(this.selectedStart, this.selectedEnd), e = Math.max(this.selectedStart, this.selectedEnd);
    const question = document.getElementById("ai-question").value;
    await this._aiCall("/api/chat/explain", { file:this.currentFile, start_line:s, end_line:e, question });
  }

  async askQuestion() {
    if (this.selectedStart < 0) {
      const q = document.getElementById("ai-question").value;
      if (!q) return;
      await this._aiCall("/api/chat/send", { messages:[{role:"user",content:q}] });
    } else {
      this.explainSelection();
    }
  }
}

const codeUI = new CodeUI();
