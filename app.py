"""Web app: python -m uvicorn app:app --port 8000  then open http://127.0.0.1:8000"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from taxbot import llm
from taxbot.bot import TaxBot

app = FastAPI(title="TaxBot")
bot = TaxBot()
STATIC = Path(__file__).parent / "static"


class Question(BaseModel):
    question: str
    model: str | None = None
    explain: bool = False  # let the model write the explanation for calculations (slower)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/models")
def models():
    chat_models = [m for m in llm.list_models() if "embed" not in m and not m.startswith("bge")]
    return {"default": bot.model, "models": chat_models}


def friendly(error: str) -> str:
    if "alloc" in error or "memory" in error:
        return ("Not enough free memory to load the model. Close other heavy apps (browser tabs, another "
                "TaxBot or eval run) or choose qwen3:1.7b, then ask again.")
    if "connection refused" in error.lower() or "10061" in error:
        return "Ollama isn't running. Start the Ollama app, then ask again."
    return error


@app.post("/api/ask")
def ask(q: Question):
    # Plain def (not async): FastAPI runs it in a thread, so a slow CPU generation doesn't block the server.
    def events():
        try:
            for ev in bot.ask(q.question.strip(), q.model, q.explain):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except (RuntimeError, OSError) as e:  # OSError covers urllib's URLError when Ollama is down
            yield json.dumps({"type": "error", "message": friendly(str(e))}, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
