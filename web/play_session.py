"""Play session — manages a persistent train_gpt --serve process."""

import asyncio
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXE_PATH = PROJECT_ROOT / "build" / "Release" / "train_gpt.exe"
if not EXE_PATH.exists():
    EXE_PATH = PROJECT_ROOT / "build" / "Debug" / "train_gpt.exe"


class PlaySession:
    """Manages a train_gpt --serve process for interactive generation."""

    def __init__(self):
        self.process = None
        self.state = "idle"
        self.current_model = ""

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints in data/."""
        ckpts = []
        data_dir = PROJECT_ROOT / "data"
        seen = set()
        for f in sorted(data_dir.glob("*.json")):
            try:
                import json as j
                cfg = j.loads(f.read_text(encoding="utf-8"))
                if cfg.get("format") == "build-llm-using-cpp-checkpoint":
                    prefix = str(f).replace(".json", "")
                    name = Path(prefix).name
                    if name not in seen:
                        seen.add(name)
                        ckpts.append({
                            "prefix": prefix,
                            "name": name,
                            "d_model": cfg.get("d_model", "?"),
                            "n_layers": cfg.get("n_layers", "?"),
                            "vocab_size": cfg.get("vocab_size", "?"),
                            "step": cfg.get("step", "?"),
                        })
            except Exception:
                pass
        return ckpts

    async def start(self, checkpoint_prefix: str, tokenizer_type: str = "byte",
                    bpe_vocab: str = "", bpe_merges: str = ""):
        """Launch train_gpt --serve with the given checkpoint."""
        if self.state != "idle":
            await self.stop()

        args = [str(EXE_PATH), "--load", checkpoint_prefix, "--steps", "0", "--serve"]
        if tokenizer_type == "bpe" and bpe_vocab and bpe_merges:
            args += ["--tokenizer", "bpe", "--bpe-vocab", bpe_vocab, "--bpe-merges", bpe_merges]

        self.process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )
        self.state = "ready"
        self.current_model = checkpoint_prefix
        # Read past the "loaded checkpoint" line
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout=10.0)
        return True

    async def generate(self, prompt: str, temp: float = 0.8, topk: int = 40, gen: int = 50):
        """Send a generation request and yield tokens."""
        if not self.process or self.state != "ready":
            yield {"error": "Model not loaded"}
            return

        req = json.dumps({"prompt": prompt, "temp": temp, "topk": topk, "gen": gen})
        self.process.stdin.write(req.encode() + b"\n")
        await self.process.stdin.drain()

        while True:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=30.0)
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("done"):
                break
            yield data

    async def stop(self):
        if self.process and self.process.stdin:
            self.process.stdin.write(b"EXIT\n")
            await self.process.stdin.drain()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.state = "idle"
        self.current_model = ""
