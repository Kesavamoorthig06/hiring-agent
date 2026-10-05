"""Web server: resume upload, pipeline jobs, and a coaching chat.

Jobs are held in memory with a TTL. A bounded worker pool keeps LLM calls and
memory in check when many people upload at once; extra uploads wait in a queue
and the client sees their position.
"""

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp import pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("hiring-agent")

MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
WORKERS = int(os.getenv("PIPELINE_WORKERS", "24"))
JOB_TTL = 60 * 60
STATIC = Path(__file__).parent / "webapp" / "static"

app = FastAPI(title="Hiring Agent", docs_url=None, redoc_url=None)
pool = ThreadPoolExecutor(max_workers=WORKERS)
chat_pool = ThreadPoolExecutor(max_workers=WORKERS)
jobs: dict = {}
lock = threading.Lock()


def _gc():
    cutoff = time.time() - JOB_TTL
    with lock:
        for jid in [k for k, v in jobs.items() if v["created"] < cutoff]:
            jobs.pop(jid, None)


def _run(jid: str, data: bytes):
    def progress(stage, label):
        jobs[jid].update(stage=stage, label=label, status="running")

    try:
        jobs[jid]["result"] = pipeline.analyze_pdf(data, progress)
        jobs[jid].update(status="done", stage="done", label="Done")
    except ValueError as exc:
        jobs[jid].update(status="error", error=str(exc))
    except Exception as exc:
        log.exception("pipeline failed")
        jobs[jid].update(
            status="error",
            error="The analysis service is busy or failed. Please try again in a minute.",
        )


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    _gc()
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(
            413, f"File is larger than {MAX_UPLOAD // (1024 * 1024)} MB"
        )
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "Please upload a PDF file")
    jid = uuid.uuid4().hex
    jobs[jid] = {
        "created": time.time(),
        "status": "queued",
        "stage": "queued",
        "label": "Waiting for a free worker",
    }
    pool.submit(_run, jid, data)
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "Job not found or expired")
    out = {k: j.get(k) for k in ("status", "stage", "label", "error")}
    if j["status"] == "done":
        out["result"] = j["result"]
    return out


class ChatIn(BaseModel):
    messages: List[dict]
    resume_text: str
    evaluation: dict
    suggestions: dict


@app.post("/api/chat")
def chat(body: ChatIn):
    if not body.messages or body.messages[-1].get("role") != "user":
        raise HTTPException(400, "Last message must be from the user")
    try:
        reply = chat_pool.submit(
            pipeline.chat_reply,
            body.messages,
            body.resume_text,
            body.evaluation,
            body.suggestions,
        ).result(timeout=120)
    except Exception:
        log.exception("chat failed")
        raise HTTPException(502, "The coach is busy. Try again in a moment.")
    return {"reply": reply}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
