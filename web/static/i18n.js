"use strict";

const I18N = {
  en: {
    "app.title": "build-llm-using-cpp Trainer",
    "nav.config": "⚙ Config",
    "nav.monitor": "📊 Monitor",
    "nav.code": "📁 Code",
    "nav.chat": "🤖 AI Chat",
    "nav.learn": "📖 Learn",
    "nav.play": "💬 Play",
    "status.ready": "Ready — Configure and start training",
    "status.training": "Training in progress",
    "status.error": "Error — check server logs",
    "status.saved": "Saved at step {step}",
    "status.launching": "Launching...",
    "status.stopping": "Stopping & saving...",
    "config.presets": "Presets:",
    "config.start": "▶ Start Training",
    "config.stop": "⏹ Stop & Save",
    "config.title": "Training Configuration",
    "monitor.progress": "Progress:",
    "monitor.loss": "Loss",
    "monitor.step": "Step",
    "monitor.time": "Time",
    "monitor.lossPerSec": "Loss/sec",
    "monitor.weights": "Weight Statistics",
    "monitor.param": "Parameter",
    "monitor.mean": "Mean",
    "monitor.rms": "RMS",
    "monitor.min": "Min",
    "monitor.max": "Max",
    "monitor.eta": "ETA:",
    "code.files": "Files",
    "code.select": "Select a file to view",
    "code.selected": "Lines {s}-{e} selected",
    "code.selectHelp": "Click and drag to select lines",
    "code.explain": "🤖 Explain Selected Lines",
    "code.ask": "Ask a question about the code...",
    "code.askBtn": "Ask",
    "code.analyzing": "Analyzing...",
    "code.thinking": "Thinking...",
    "code.connErr": "Connection error",
    "code.noKey": "No API key configured",
    "chat.title": "AI Assistant",
    "chat.placeholder": "Ask anything about build-llm-using-cpp...",
    "chat.send": "Send",
    "chat.you": "🧑 You",
    "chat.ai": "🤖 AI",
    "chat.thinking": "Thinking...",
    "settings.title": "AI Settings",
    "settings.url": "API URL",
    "settings.key": "API Key",
    "settings.model": "Model",
    "settings.save": "Save",
    "settings.cancel": "Cancel",
    "settings.saved": "Settings saved",
    "settings.note": "Settings stored in browser only. Not sent to server except during AI chat.",
    "lang.toggle": "中文",
    // Config field labels
    "field.data_path": "Training Data",
    "field.tokenizer": "Tokenizer",
    "field.bpe_vocab": "BPE Vocab File",
    "field.bpe_merges": "BPE Merges File",
    "field.token_data": "Token Data (.bin)",
    "field.steps": "Training Steps",
    "field.batch": "Batch Size",
    "field.seq": "Sequence Length",
    "field.dmodel": "Model Dimension (C)",
    "field.layers": "Layers",
    "field.lr": "Learning Rate",
    "field.seed": "Random Seed",
    "field.norm_type": "Normalization",
    "field.mlp_type": "MLP Type",
    "field.save_prefix": "Checkpoint Prefix",
    "field.save_interval": "Save Every N Steps (0=end)",
    "field.temperature": "Temperature",
    "field.topk": "Top-K Sampling",
    "field.kvcache": "Use KV-Cache",
    // Group labels
    "group.Data": "Data",
    "group.Training": "Training",
    "group.Architecture": "Architecture",
    "group.Output": "Output",
    "group.Generation": "Generation",
    // Preset names
    "preset.Tiny (32M)": "Tiny",
    "preset.Small (64M)": "Small",
    "preset.Medium (100M)": "Medium",
    "preset.BPE Tiny": "BPE Tiny",
    "chart.loss": "Loss",
    // Architecture diagram
    "arch.title": "Model Architecture (click any component → view code)",
    "arch.transformer_block": "Transformer Block × N (pre-norm, residual connections)",
    "arch.legend": "Hover for details • Click to view code",
    "arch.legend_text": "",
    // Tooltips for config fields
    "tooltip.data_path": "Text file used for training. Each byte is a token (byte vocab=256). Larger files = more training data.",
    "tooltip.tokenizer": "Byte: each byte (0-255) is a token. BPE: subword tokens (e.g. GPT-2 with 50K vocab). BPE is more efficient but slower to train.",
    "tooltip.steps": "Number of training iterations. Each step = one batch. Model sees (steps × batch × seq) tokens total.",
    "tooltip.batch": "Number of independent sequences trained simultaneously. Larger = more stable gradients, more memory.",
    "tooltip.seq": "Maximum sequence length. Model can only see this many tokens at once. Longer = more context, more memory.",
    "tooltip.dmodel": "Model dimension (C). Each token is a C-dimensional vector. Larger = more capacity, slower training. Typical: 32-1024.",
    "tooltip.layers": "Transformer blocks stacked sequentially. Each = Attention + MLP with residual connections. More layers = deeper model.",
    "tooltip.lr": "Learning rate — how fast weights update each step. Too high = unstable (NaN loss). Too low = very slow. Typical: 1e-4 to 1e-2.",
    "tooltip.seed": "Random seed for reproducible training. Same seed = same initialization + same data order = same results.",
    "tooltip.norm_type": "LayerNorm: (x-μ)/σ·γ+β — subtracts mean, divides by std. RMSNorm: x/rms·γ — simpler, no centering. LLaMA uses RMSNorm.",
    "tooltip.mlp_type": "GELU: standard 2-layer MLP. SwiGLU: gated with SiLU. MoE: mixture of experts with top-K router. SwiGLU is used in LLaMA.",
    "tooltip.save_prefix": "Base filename for checkpoint files (.json + .bin). Resume training later with --load.",
    "tooltip.save_interval": "Save checkpoint every N steps. 0 = save only at end. Saves allow resuming if training stops.",
    "tooltip.temperature": "Controls randomness in generation. T=1.0 = normal, T>1 = more random, T<0.5 = more deterministic.",
    "tooltip.topk": "Only sample from top-K most likely tokens. 0 = disabled (sample from all). 40 = common for readable text.",
    "tooltip.kvcache": "Use KV-cache for faster generation. Caches keys/values from previous tokens instead of recomputing.",
    "tooltip.monitor.loss": "Cross-entropy between predicted and actual next token. Lower = better. Random start ≈ ln(vocab_size) ≈ 5.5 (byte) or 10.8 (BPE). Should decrease over time.",
    "tooltip.monitor.rms": "Root Mean Square — weight magnitude. Healthy range: 0.05-0.3 for weight matrices, ~1.0 for gamma. Explosion >1.0 or vanishing <0.001 may indicate LR issues.",
    "tooltip.monitor.mean": "Average weight value. Should stay near 0. Large deviations may indicate dead neurons or poor initialization.",
  },
  zh: {
    "app.title": "build-llm-using-cpp 训练器",
    "nav.config": "⚙ 配置",
    "nav.monitor": "📊 监控",
    "nav.code": "📁 代码",
    "nav.chat": "🤖 AI 对话",
    "nav.learn": "📖 学习",
    "nav.play": "💬 对话",
    "status.ready": "就绪 — 配置并开始训练",
    "status.training": "训练进行中",
    "status.error": "错误 — 请检查服务器日志",
    "status.saved": "已保存至第 {step} 步",
    "status.launching": "启动中...",
    "status.stopping": "停止并保存中...",
    "config.presets": "预设：",
    "config.start": "▶ 开始训练",
    "config.stop": "⏹ 停止并保存",
    "config.title": "训练配置",
    "monitor.progress": "进度：",
    "monitor.loss": "损失",
    "monitor.step": "步数",
    "monitor.time": "时间",
    "monitor.lossPerSec": "损失/秒",
    "monitor.weights": "权重统计",
    "monitor.param": "参数",
    "monitor.mean": "均值",
    "monitor.rms": "RMS",
    "monitor.min": "最小",
    "monitor.max": "最大",
    "monitor.eta": "预计剩余：",
    "code.files": "文件",
    "code.select": "选择文件以浏览源代码",
    "code.selected": "已选择第 {s}-{e} 行",
    "code.selectHelp": "点击并拖拽选择代码行",
    "code.explain": "🤖 解释所选代码",
    "code.ask": "询问有关代码的问题...",
    "code.askBtn": "询问",
    "code.analyzing": "分析中...",
    "code.thinking": "思考中...",
    "code.connErr": "连接错误",
    "code.noKey": "未配置 API 密钥",
    "chat.title": "AI 助手",
    "chat.placeholder": "询问关于 build-llm-using-cpp 的任何问题...",
    "chat.send": "发送",
    "chat.you": "🧑 你",
    "chat.ai": "🤖 AI",
    "chat.thinking": "思考中...",
    "settings.title": "AI 设置",
    "settings.url": "API 地址",
    "settings.key": "API 密钥",
    "settings.model": "模型",
    "settings.save": "保存",
    "settings.cancel": "取消",
    "settings.saved": "设置已保存",
    "settings.note": "设置仅存储在浏览器中。仅 AI 对话时发送到服务器。",
    "lang.toggle": "English",
    // Config field labels (zh)
    "field.data_path": "训练数据",
    "field.tokenizer": "分词器",
    "field.bpe_vocab": "BPE 词表文件",
    "field.bpe_merges": "BPE 合并文件",
    "field.token_data": "Token 数据 (.bin)",
    "field.steps": "训练步数",
    "field.batch": "批次大小",
    "field.seq": "序列长度",
    "field.dmodel": "模型维度 (C)",
    "field.layers": "层数",
    "field.lr": "学习率",
    "field.seed": "随机种子",
    "field.norm_type": "归一化类型",
    "field.mlp_type": "MLP 类型",
    "field.save_prefix": "检查点前缀",
    "field.save_interval": "保存间隔 (0=仅结束时)",
    "field.temperature": "温度",
    "field.topk": "Top-K 采样",
    "field.kvcache": "使用 KV-Cache",
    // Group labels (zh)
    "group.Data": "数据",
    "group.Training": "训练",
    "group.Architecture": "架构",
    "group.Output": "输出",
    "group.Generation": "生成",
    // Preset names (zh)
    "preset.Tiny (32M)": "微型",
    "preset.Small (64M)": "小型",
    "preset.Medium (100M)": "中型",
    "preset.BPE Tiny": "BPE 微型",
    "chart.loss": "损失",
    // Architecture diagram (zh)
    "arch.title": "模型架构（点击任意组件 → 查看代码）",
    "arch.transformer_block": "Transformer 块 × N（预归一化，残差连接）",
    "arch.legend": "悬停查看详情 • 点击查看代码",
    "arch.legend_text": "",
    // Tooltips (zh)
    "tooltip.data_path": "用于训练的文本文件。每个字节是一个token。文件越大，训练数据越多。",
    "tooltip.tokenizer": "Byte: 每个字节(0-255)是一个token。BPE: 子词token(如GPT-2有50K词表)。BPE效率更高但训练更慢。",
    "tooltip.steps": "训练迭代次数。每步处理一个批次。模型总共看到(steps × batch × seq)个token。",
    "tooltip.batch": "同时训练的独立序列数。越大梯度越稳定，内存需求也越大。",
    "tooltip.seq": "最大序列长度。模型一次只能看到这么多个token。更长=更多上下文，内存需求更大。",
    "tooltip.dmodel": "模型维度(C)。每个token表示为一个C维向量。越大能力越强，训练越慢。典型值: 32-1024。",
    "tooltip.layers": "顺序堆叠的Transformer块。每块=注意力+MLP+残差连接。更多层=更深的模型。",
    "tooltip.lr": "学习率—每步权重的更新速度。太高=不稳定(NaN损失)。太低=非常慢。典型: 1e-4到1e-2。",
    "tooltip.seed": "用于可复现训练的随机种子。相同种子=相同初始化+相同数据顺序=相同结果。",
    "tooltip.norm_type": "LayerNorm: (x-μ)/σ·γ+β—减去均值除以标准差。RMSNorm: x/rms·γ—更简单无需中心化。LLaMA使用RMSNorm。",
    "tooltip.mlp_type": "GELU: 标准双层MLP。SwiGLU: 带SiLU门控。MoE: 带top-K路由器的混合专家。LLaMA使用SwiGLU。",
    "tooltip.save_prefix": "检查点文件的基本文件名(.json+.bin)。之后可用--load恢复训练。",
    "tooltip.save_interval": "每N步保存一次检查点。0=仅在结束时保存。保存可在训练中断后恢复。",
    "tooltip.temperature": "控制生成的随机性。T=1.0=正常，T>1=更随机，T<0.5=更确定性。",
    "tooltip.topk": "仅从概率最高的K个token中采样。0=禁用(全部采样)。40=常见值可读性好。",
    "tooltip.kvcache": "使用KV-cache加速生成。缓存之前token的keys/values而非重新计算。",
    "tooltip.monitor.loss": "预测token与实际下一个token之间的交叉熵。越低越好。随机初始≈ln(词表大小)≈5.5(byte)或10.8(BPE)。应随时间降低。",
    "tooltip.monitor.rms": "均方根—权重幅度。健康范围: 权重矩阵0.05-0.3, gamma≈1.0。爆涨>1.0或消失<0.001可能表明学习率问题。",
    "tooltip.monitor.mean": "权重均值。应接近0。大幅偏离可能表明死神经元或初始化问题。",
  }
};

let currentLang = localStorage.getItem("lang") || "en";

function t(key, params) {
  let str = I18N[currentLang]?.[key] || I18N.en[key] || key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      str = str.replace(`{${k}}`, v);
    }
  }
  return str;
}

function toggleLang() {
  currentLang = currentLang === "en" ? "zh" : "en";
  localStorage.setItem("lang", currentLang);
  location.reload();
}

function applyI18n() {
  document.title = t("app.title");
  document.querySelectorAll("[data-i18n-nav]").forEach(el => el.textContent = t("nav." + el.dataset.i18nNav));
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.dataset.i18n;
    if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
      el.placeholder = t(key);
    } else {
      el.textContent = t(key);
    }
  });
  const lt = document.getElementById("lang-toggle");
  if (lt) lt.textContent = t("lang.toggle");
  // Re-apply dynamic labels (chart, config form, monitor)
  if (typeof ui !== "undefined" && ui.applyTranslations) ui.applyTranslations();
  if (typeof window.chart !== "undefined" && window.chart.applyTranslations) window.chart.applyTranslations();
}

document.addEventListener("DOMContentLoaded", applyI18n);
