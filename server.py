"""Web server: resume upload, pipeline jobs, and a coaching chat.

Jobs are held in memory with a TTL. Resumes are processed strictly one at a
time (and each resume's LLM prompts run one after another), so a free-tier API
key is never overloaded. Extra uploads wait in a FIFO queue; the client sees
its live position and an estimated wait.
"""

import logging
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp import limiter, pipeline

limiter.install()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("hiring-agent")

MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "60"))
AVG_SECONDS = [float(os.getenv("EST_JOB_SECONDS", "120"))]
JOB_TTL = 60 * 60
FRIENDLY = "We could not finish this one just now. Please upload it again in a moment."
STATIC = Path(__file__).parent / "webapp" / "static"

app = FastAPI(title="Hiring Agent", docs_url=None, redoc_url=None)
jobs: dict = {}
lock = threading.Lock()
waiting: deque = deque()  # job ids in FIFO order
wake = threading.Condition(lock)
current: set = set()


def _gc():
    cutoff = time.time() - JOB_TTL
    with lock:
        for jid in [k for k, v in jobs.items() if v["created"] < cutoff]:
            jobs.pop(jid, None)


def _run(jid: str, data: bytes):
    started = time.time()
    def progress(stage, label, detail=None):
        print(f"[job {jid[:8]}] +{time.time() - started:.0f}s stage={stage} {detail or ''}".rstrip(), flush=True)
        jobs[jid].update(stage=stage, label=label, detail=detail, status="running")

    jobs[jid].update(status="running", stage="parse", label="Reading your resume")
    try:
        jobs[jid]["result"] = pipeline.analyze_pdf(data, progress)
        jobs[jid].update(status="done", stage="done", label="Done")
        print(f"[job {jid[:8]}] done in {time.time() - started:.0f}s", flush=True)
        AVG_SECONDS[0] = 0.7 * AVG_SECONDS[0] + 0.3 * (time.time() - started)
    except ValueError as exc:
        msg = str(exc)
        low = msg.lower()
        if any(w in low for w in ("model", "api", "key", "provider", "quota", "429", "http")):
            log.error("pipeline value error: %s", msg)
            msg = FRIENDLY
        jobs[jid].update(status="error", error=msg)
    except Exception:
        log.exception("pipeline failed")
        jobs[jid].update(
            status="error",
            error=FRIENDLY,
        )
    finally:
        jobs[jid].pop("data", None)


def _worker():
    while True:
        with wake:
            while not waiting:
                wake.wait()
            jid = waiting.popleft()
            current.add(jid)
        try:
            _run(jid, jobs[jid]["data"])
        except Exception:
            log.exception("worker error")
        finally:
            current.discard(jid)


# One worker per API key: each key still serves one prompt at a time.
for _i in range(limiter.key_count()):
    threading.Thread(target=_worker, name=f"queue-worker-{_i}", daemon=True).start()


def _queue_info(jid: str) -> dict:
    with lock:
        ids = list(waiting)
        ahead = ids.index(jid) if jid in ids else 0
        running = len(current)
        total = len(ids)
    pos = ahead + running + 1
    return {
        "position": pos,
        "queue_size": total + len(current),
        "eta_seconds": int(AVG_SECONDS[0] * (ahead + running + 1)),
    }


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
    with wake:
        if len(waiting) >= MAX_QUEUE:
            raise HTTPException(503, "The queue is full right now. Please try again shortly.")
        jid = uuid.uuid4().hex
        jobs[jid] = {
            "created": time.time(),
            "status": "queued",
            "stage": "queued",
            "label": "Waiting in line",
            "data": data,
        }
        waiting.append(jid)
        wake.notify()
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "Job not found or expired")
    out = {k: j.get(k) for k in ("status", "stage", "label", "detail", "error")}
    if j["status"] == "queued":  # the queue is internal: show the first stage
        out.update(status="running", stage="parse", label="Reading your resume")
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
        reply = pipeline.chat_reply(
            body.messages, body.resume_text, body.evaluation, body.suggestions
        )
    except Exception:
        log.exception("chat failed")
        raise HTTPException(502, "The coach is busy. Try again in a moment.")
    return {"reply": reply}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
