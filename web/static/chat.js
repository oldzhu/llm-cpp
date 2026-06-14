"use strict";

class ChatUI {
  constructor() {
    this.messages = [];
    this.init();
  }

  _aiSettings() { return aiSettings ? aiSettings.data : {}; }

  init() {
    document.getElementById("chat-input")?.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); this.send(); }
    });
  }

  async send() {
    const input = document.getElementById("chat-input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    this.addMessage("user", text);
    this.messages.push({ role: "user", content: text });

    const container = document.getElementById("chat-messages");
    const loading = document.createElement("div");
    loading.className = "chat-loading";
    loading.textContent = t("chat.thinking");
    container.appendChild(loading);
    container.scrollTop = container.scrollHeight;

    try {
      const resp = await fetch("/api/chat/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: this.messages, settings: this._aiSettings() }),
      });
      const data = await resp.json();
      loading.remove();
      if (data.error) { this.addMessage("assistant", data.error); }
      else {
        this.addMessage("assistant", data.content);
        this.messages.push({ role: "assistant", content: data.content });
      }
    } catch (e) { loading.remove(); this.addMessage("assistant", t("code.connErr")); }
  }

  addMessage(role, content) {
    const container = document.getElementById("chat-messages");
    const div = document.createElement("div");
    div.className = "chat-message chat-" + role;
    const rendered = content
      .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
      .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="chat-code"><code>$2</code></pre>')
      .replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/\n/g,'<br>');
    div.innerHTML = `<div class="chat-role">${role==="user"?t("chat.you"):t("chat.ai")}</div><div class="chat-text">${rendered}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }
}

const chatUI = new ChatUI();
