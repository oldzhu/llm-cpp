"use strict";

const AI_SETTINGS_DEFAULTS = {
  api_url: "https://api.deepseek.com/anthropic",
  api_key: "",
  model: "deepseek-v4-flash",
};

class AISettings {
  constructor() {
    this.data = this.load();
  }

  load() {
    try {
      const raw = localStorage.getItem("ai_settings");
      if (raw) {
        const parsed = JSON.parse(raw);
        return { ...AI_SETTINGS_DEFAULTS, ...parsed };
      }
    } catch (e) { /* ignore */ }
    return { ...AI_SETTINGS_DEFAULTS };
  }

  save(data) {
    this.data = { ...this.data, ...data };
    localStorage.setItem("ai_settings", JSON.stringify(this.data));
  }

  showModal() {
    const existing = document.getElementById("settings-modal");
    if (existing) existing.remove();

    const s = this.data;
    const html = `
    <div id="settings-modal" class="modal-overlay">
      <div class="modal-box">
        <h3 data-i18n="settings.title">AI Settings</h3>
        <div class="modal-field">
          <label data-i18n="settings.url">API URL</label>
          <input type="text" id="set-url" value="${this.escape(s.api_url)}" data-i18n="settings.url">
        </div>
        <div class="modal-field">
          <label data-i18n="settings.key">API Key</label>
          <input type="password" id="set-key" value="${this.escape(s.api_key)}" data-i18n="settings.key" placeholder="sk-xxxxx">
        </div>
        <div class="modal-field">
          <label data-i18n="settings.model">Model</label>
          <input type="text" id="set-model" value="${this.escape(s.model)}" data-i18n="settings.model">
        </div>
        <p class="modal-note" data-i18n="settings.note">Settings are stored in your browser only.</p>
        <div class="modal-actions">
          <button class="btn" id="set-save" data-i18n="settings.save">Save</button>
          <button class="btn btn-cancel" id="set-cancel" data-i18n="settings.cancel">Cancel</button>
        </div>
      </div>
    </div>`;
    document.body.insertAdjacentHTML("beforeend", html);
    document.getElementById("set-save").addEventListener("click", () => {
      this.save({
        api_url: document.getElementById("set-url").value.trim(),
        api_key: document.getElementById("set-key").value.trim(),
        model: document.getElementById("set-model").value.trim(),
      });
      document.getElementById("settings-modal").remove();
      this.showToast(t("settings.saved"));
    });
    document.getElementById("set-cancel").addEventListener("click", () => {
      document.getElementById("settings-modal").remove();
    });
    applyI18n();
  }

  escape(s) { return s.replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

  showToast(msg) {
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 2000);
  }
}

const aiSettings = new AISettings();
