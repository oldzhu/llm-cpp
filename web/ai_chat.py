"""AI chat proxy — forwards messages to OpenAI-compatible API with project context."""

import os

import httpx

from code_explorer import build_project_context

# Pre-load project context once
_PROJECT_CONTEXT = ""

def get_project_context() -> str:
    global _PROJECT_CONTEXT
    if not _PROJECT_CONTEXT:
        _PROJECT_CONTEXT = build_project_context()
    return _PROJECT_CONTEXT


async def chat(messages: list[dict], selected_code: str = "", selected_file: str = "",
               settings: dict | None = None) -> dict:
    """Send chat with project-aware system prompt, using provided or env settings."""
    settings = settings or {}
    api_key = settings.get("api_key") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = settings.get("api_url") or os.getenv("AI_BASE_URL", "https://api.openai.com/v1")
    model = settings.get("model") or os.getenv("AI_MODEL", "gpt-4o-mini")

    if not api_key:
        return {"error": "No API key configured. Click the gear icon to set your API key and model."}

    system = f"""You are an AI assistant for the build-llm-using-cpp project.
This is a C++ from-scratch GPT-style Transformer implementation for learning/teaching.
You help users understand the code, architecture, and training concepts.

Project structure:
{get_project_context()}

Key concepts:
- Tensor: flat float buffer with autograd Node-based backward graph
- TinyGPT: byte-level or BPE token-level GPT with configurable layers, norms, MLP types
- Variants: MHA, KV-cache, RoPE, GQA, MoE — side-by-side implementations
- Backend: KernelBackend seam for CPU/BlockedSimd/Vulkan matmul
- Configurable: norm_type (layernorm/rmsnorm), mlp_type (gelu/swiglu/moe)

Answer concisely. Reference specific source files and line numbers when helpful.
Use shapes like [B,T,C] for tensor dimensions."""

    if selected_code:
        system += f"\n\nThe user selected code from {selected_file}:\n```cpp\n{selected_code}\n```\nExplain clearly."

    full_messages = [{"role": "system", "content": system}]
    for m in messages:
        full_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    if not base_url.endswith("/chat/completions"):
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        base_url += "/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                base_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": full_messages, "max_tokens": 1024, "temperature": 0.3},
            )
            if resp.status_code != 200:
                return {"error": f"API error {resp.status_code}: {resp.text[:300]}"}
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {"role": "assistant", "content": content}
    except Exception as e:
        return {"error": str(e)}


async def explain_code(file_path: str, start_line: int, end_line: int, question: str = "",
                       settings: dict | None = None) -> dict:
    """Explain selected code lines."""
    from code_explorer import read_file_content
    content = read_file_content(file_path)
    if "error" in content: return {"error": content["error"]}
    lines = content["lines"]
    if start_line < 1: start_line = 1
    if end_line > len(lines): end_line = len(lines)
    selected = "\n".join(lines[start_line - 1 : end_line])
    prompt = f"Explain this code from {file_path} (lines {start_line}-{end_line}):\n```cpp\n{selected}\n```"
    if question: prompt += f"\n\nUser question: {question}"
    return await chat([{"role": "user", "content": prompt}],
                      selected_code=selected, selected_file=file_path, settings=settings)
