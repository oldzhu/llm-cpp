"""FastAPI server — renders pre-built HTML directly to avoid Jinja2 cache bug."""

import os
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import CONFIG, PRESETS, apply_preset, validate
from session import TrainingSession
from websocket import ProgressBroadcaster
from code_explorer import get_file_tree, read_file_content
from ai_chat import chat, explain_code
from play_session import PlaySession

app = FastAPI(title="build-llm-using-cpp Trainer")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

session = TrainingSession()
broadcaster = ProgressBroadcaster()
session.set_ws_clients(broadcaster.clients)
play = PlaySession()


def _render_page():
    base = (TEMPLATE_DIR / "base.html").read_text(encoding="utf-8")
    config = (TEMPLATE_DIR / "config.html").read_text(encoding="utf-8")
    monitor = (TEMPLATE_DIR / "monitor.html").read_text(encoding="utf-8")
    code = (TEMPLATE_DIR / "code.html").read_text(encoding="utf-8")
    chat_t = (TEMPLATE_DIR / "chat.html").read_text(encoding="utf-8")
    learn = (TEMPLATE_DIR / "learn.html").read_text(encoding="utf-8")
    play_t = (TEMPLATE_DIR / "play.html").read_text(encoding="utf-8")
    base = base.replace("{% include 'config.html' %}", config)
    base = base.replace("{% include 'monitor.html' %}", monitor)
    base = base.replace("{% include 'code.html' %}", code)
    base = base.replace("{% include 'chat.html' %}", chat_t)
    base = base.replace("{% include 'learn.html' %}", learn)
    base = base.replace("{% include 'play.html' %}", play_t)
    return base


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return HTMLResponse(_render_page())


@app.get("/code", response_class=HTMLResponse)
async def code_page(request: Request):
    return HTMLResponse(_render_page())


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return HTMLResponse(_render_page())


@app.get("/api/config/schema")
async def get_schema():
    return CONFIG


@app.get("/api/config/presets")
async def get_presets():
    return PRESETS


@app.post("/api/config/validate")
async def validate_config(data: dict):
    errors = validate(data)
    return {"valid": len(errors) == 0, "errors": errors}


@app.post("/api/train/start")
async def train_start(data: dict):
    config = data.get("config", data)
    errors = validate(config)
    if errors:
        return {"error": "Invalid config", "errors": errors}
    result = await session.start(config)
    return result


@app.post("/api/train/stop")
async def train_stop():
    return await session.stop()


@app.get("/api/train/status")
async def train_status():
    return {
        "state": session.state,
        "step": session.current_step,
        "loss": session.current_loss,
        "time": session.current_time,
        "total_steps": session.total_steps,
        "weight_stats": session.weight_stats,
    }


@app.websocket("/ws/train")
async def ws_train(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
    except Exception:
        broadcaster.disconnect(websocket)


@app.get("/api/generate")
async def generate(prompt: str = "Hello", tokens: int = 50, temp: float = 0.8, topk: int = 40):
    return {"status": "not yet implemented"}


@app.get("/api/code/tree")
async def code_tree():
    return get_file_tree()


@app.get("/api/code/file")
async def code_file(path: str = ""):
    return read_file_content(path)


@app.post("/api/chat/send")
async def chat_send(data: dict):
    messages = data.get("messages", [])
    selected_code = data.get("selected_code", "")
    selected_file = data.get("selected_file", "")
    settings = data.get("settings", {})
    return await chat(messages, selected_code, selected_file, settings)


@app.post("/api/chat/explain")
async def chat_explain(data: dict):
    file_path = data.get("file", "")
    start_line = data.get("start_line", 1)
    end_line = data.get("end_line", 1)
    question = data.get("question", "")
    settings = data.get("settings", {})
    return await explain_code(file_path, start_line, end_line, question, settings)


# === Play tab routes ===

@app.get("/api/play/checkpoints")
async def play_checkpoints():
    return play.list_checkpoints()


@app.post("/api/play/load")
async def play_load(data: dict):
    cp = data.get("checkpoint", "")
    if not cp:
        return {"error": "No checkpoint specified"}
    try:
        await play.start(cp)
        return {"status": "loaded"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/play/generate")
async def play_generate(data: dict):
    from fastapi.responses import StreamingResponse
    prompt = data.get("prompt", "")
    temp = float(data.get("temp", 0.8))
    topk = int(data.get("topk", 40))
    gen = int(data.get("gen", 50))
    if not prompt:
        return {"error": "No prompt"}

    async def token_stream():
        async for token_data in play.generate(prompt, temp, topk, gen):
            yield token_data.get("text", "") + " "
    return StreamingResponse(token_stream(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
