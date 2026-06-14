"""Training session manager — launches and monitors train_gpt.exe."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXE_PATH = PROJECT_ROOT / "build" / "Release" / "train_gpt.exe"
if not EXE_PATH.exists():
    EXE_PATH = PROJECT_ROOT / "build" / "Debug" / "train_gpt.exe"


class TrainingSession:
    def __init__(self):
        self.process: asyncio.subprocess.Process | None = None
        self.state: str = "idle"
        self.config: dict = {}
        self.current_step: int = 0
        self.current_loss: float = 0.0
        self.current_time: float = 0.0
        self.weight_stats: dict = {}
        self.total_steps: int = 0
        self._ws_clients: list = []

    def set_ws_clients(self, clients: list):
        self._ws_clients = clients

    async def _broadcast(self, data: dict):
        for ws in self._ws_clients:
            try:
                await ws.send_json(data)
            except Exception:
                pass

    def _build_args(self, config: dict) -> list[str]:
        args = [str(EXE_PATH)]
        if config.get("data_path"):
            args += ["--data", config["data_path"]]
        args += ["--steps", str(config.get("steps", 200))]
        args += ["--batch", str(config.get("batch", 4))]
        args += ["--seq", str(config.get("seq", 64))]
        args += ["--dmodel", str(config.get("dmodel", 64))]
        args += ["--layers", str(config.get("layers", 1))]
        args += ["--lr", str(config.get("lr", 0.0003))]
        args += ["--seed", str(config.get("seed", 1))]
        if config.get("save_prefix"):
            args += ["--save", config["save_prefix"]]
        if config.get("save_interval", 0) > 0:
            args += ["--save-interval", str(config["save_interval"])]
        if config.get("mlp_type"):
            args += ["--mlp", config["mlp_type"]]
        if config.get("norm_type"):
            args += ["--norm", config["norm_type"]]
        if config.get("tokenizer"):
            args += ["--tokenizer", config["tokenizer"]]
            if config["tokenizer"] == "bpe":
                if config.get("bpe_vocab"):
                    args += ["--bpe-vocab", config["bpe_vocab"]]
                if config.get("bpe_merges"):
                    args += ["--bpe-merges", config["bpe_merges"]]
                if config.get("token_data"):
                    args += ["--token-data", config["token_data"]]

        args += ["--progress-json"]
        args += ["--progress-grads", "10"]
        args += ["--pipe-stdin"]
        return args

    async def start(self, config: dict):
        if self.state != "idle":
            return {"error": "Already running"}

        self.config = config
        self.state = "training"
        self.current_step = 0
        self.total_steps = config.get("steps", 200)
        self.weight_stats = {}

        args = self._build_args(config)
        try:
            self.process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
            )
        except Exception as e:
            self.state = "error"
            return {"error": str(e)}

        await self._broadcast({"type": "status", "state": "training", "step": 0})
        asyncio.create_task(self._read_stdout())
        return {"status": "started"}

    async def stop(self):
        if self.state != "training" or self.process is None:
            return {"error": "Not training"}

        try:
            if self.process.stdin:
                self.process.stdin.write(b"EXIT\n")
                await self.process.stdin.drain()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        except Exception:
            pass

        self.state = "idle"
        await self._broadcast({"type": "status", "state": "idle"})
        return {"status": "stopped", "step": self.current_step}

    async def _read_stdout(self):
        try:
            while self.process and self.process.stdout:
                line = await self.process.stdout.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "step":
                    self.current_step = int(data.get("step", 0))
                    self.current_loss = float(data.get("loss", 0))
                    self.current_time = float(data.get("time", 0))
                    if "w_stats" in data:
                        self.weight_stats = data["w_stats"]
                    await self._broadcast({
                        "type": "step",
                        "step": self.current_step,
                        "loss": self.current_loss,
                        "time": self.current_time,
                        "total_steps": self.total_steps,
                        "w_stats": self.weight_stats,
                    })
                elif data.get("type") in ("stopped", "saved"):
                    await self._broadcast(data)
                    self.state = "idle"
        except Exception:
            pass
        finally:
            self.state = "idle"
