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

import hmac
import re
import secrets

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp import bedrock, limiter, pipeline, runctx

limiter.install()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("hiring-agent")

_REDACT: dict = {}  # live own keys (refcounted), scrubbed from any log record


def _track(k):
    with lock:
        _REDACT[k] = _REDACT.get(k, 0) + 1


def _untrack(k):
    with lock:
        n = _REDACT.get(k, 0) - 1
        if n <= 0:
            _REDACT.pop(k, None)
        else:
            _REDACT[k] = n


class _Scrub(logging.Filter):
    def filter(self, record):
        try:
            text = record.getMessage()
            if record.exc_info:
                text += " " + logging.Formatter().formatException(record.exc_info)
        except Exception:
            return True
        if any(k in text for k in _REDACT):
            record.msg = "[log line withheld]"
            record.args = ()
            record.exc_info = None
        return True


for _h in logging.getLogger().handlers:
    _h.addFilter(_Scrub())
logging.getLogger("uvicorn.access").addFilter(_Scrub())

MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "60"))
AVG_SECONDS = [float(os.getenv("EST_JOB_SECONDS", "120"))]
JOB_TTL = 60 * 60
FRIENDLY = "We could not finish this one just now. Please upload it again in a moment."
STATIC = Path(__file__).parent / "webapp" / "static"

app = FastAPI(title="Hiring Agent", docs_url=None, redoc_url=None)
jobs: dict = {}
lock = threading.Lock()
LANES = ("gemini", "claude", "own")
waiting = {name: deque() for name in LANES}  # job ids in FIFO order, per lane
wake = threading.Condition(lock)
current = {name: set() for name in LANES}
lane_of: dict = {}


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
    runctx.set_run(jobs[jid].get("run"))
    try:
        jobs[jid]["result"] = pipeline.analyze_pdf(data, progress)
        jobs[jid].update(status="done", stage="done", label="Done")
        print(f"[job {jid[:8]}] done in {time.time() - started:.0f}s", flush=True)
        AVG_SECONDS[0] = 0.7 * AVG_SECONDS[0] + 0.3 * (time.time() - started)
    except (limiter.OwnKeyRejected, bedrock.ClaudeUnavailable) as exc:
        jobs[jid].update(status="error", error=str(exc))
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
        r = jobs[jid].pop("run", None) or {}  # drops a visitor's own key with the job
        if r.get("own_key"):
            _untrack(r["own_key"])
            r["own_key"] = None
        lane_of.pop(jid, None)


def _worker(lane):
    while True:
        with wake:
            while not waiting[lane]:
                wake.wait()
            jid = waiting[lane].popleft()
            current[lane].add(jid)
        try:
            _run(jid, jobs[jid]["data"])
        except Exception:
            log.exception("worker error")
        finally:
            current[lane].discard(jid)


# Gemini lane: one worker per pooled key, so each key serves one prompt at a time.
# Claude and own-key lanes have their own workers, so no lane starves another.
_WORKERS = {
    "gemini": limiter.key_count(),
    "claude": int(os.getenv("CLAUDE_WORKERS", "4")),
    "own": int(os.getenv("OWN_KEY_WORKERS", "8")),
}
for _lane, _n in _WORKERS.items():
    for _i in range(_n):
        threading.Thread(target=_worker, args=(_lane,), name=f"queue-{_lane}-{_i}", daemon=True).start()


def _queue_info(jid: str) -> dict:
    lane = lane_of.get(jid, "gemini")
    with lock:
        ids = list(waiting[lane])
        ahead = ids.index(jid) if jid in ids else 0
        running = len(current[lane])
        total = len(ids)
    pos = ahead + running + 1
    return {
        "position": pos,
        "queue_size": total + running,
        "eta_seconds": int(AVG_SECONDS[0] * (ahead + running + 1)),
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    provider: str = Form("gemini"),
    own_key: Optional[str] = Form(None),
):
    _gc()
    provider = provider if provider in ("gemini", "claude") else "gemini"
    own_key = (own_key or "").strip()
    run = {"provider": provider}
    lane = provider
    if provider == "claude":
        if not bedrock.available():
            raise HTTPException(503, "Claude is temporarily unavailable. Pick Gemini for now.")
    elif own_key:
        if not re.fullmatch(r"[A-Za-z0-9_\-]{20,200}", own_key):
            raise HTTPException(400, "That does not look like a Gemini key. Check it and try again.")
        run["own_key"] = own_key
        lane = "own"
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(
            413, f"File is larger than {MAX_UPLOAD // (1024 * 1024)} MB"
        )
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "Please upload a PDF file")
    with wake:
        if len(waiting[lane]) >= MAX_QUEUE:
            raise HTTPException(503, "The queue is full right now. Please try again shortly.")
        jid = uuid.uuid4().hex
        jobs[jid] = {
            "created": time.time(),
            "status": "queued",
            "stage": "queued",
            "label": "Waiting in line",
            "data": data,
            "run": run,
        }
        lane_of[jid] = lane
        if run.get("own_key"):
            _REDACT[run["own_key"]] = _REDACT.get(run["own_key"], 0) + 1
        waiting[lane].append(jid)
        wake.notify_all()
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
    provider: Optional[str] = "gemini"
    own_key: Optional[str] = None
    messages: List[dict]
    resume_text: str
    evaluation: dict
    suggestions: dict


@app.post("/api/chat")
def chat(body: ChatIn):
    if not body.messages or body.messages[-1].get("role") != "user":
        raise HTTPException(400, "Last message must be from the user")
    run = {"provider": "claude" if body.provider == "claude" else "gemini"}
    own = (body.own_key or "").strip()
    if run["provider"] == "gemini" and own and re.fullmatch(r"[A-Za-z0-9_\-]{20,200}", own):
        run["own_key"] = own
        _track(own)
    runctx.set_run(run)
    try:
        reply = pipeline.chat_reply(
            body.messages, body.resume_text, body.evaluation, body.suggestions
        )
    except (limiter.OwnKeyRejected, bedrock.ClaudeUnavailable) as exc:
        raise HTTPException(502, str(exc))
    except Exception:
        log.exception("chat failed")
        raise HTTPException(502, "The coach is busy. Try again in a moment.")
    finally:
        if run.get("own_key"):
            _untrack(run["own_key"])
        runctx.set_run(None)
    return {"reply": reply}


CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'; object-src 'none'"
)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = CSP
    resp.headers["Strict-Transport-Security"] = "max-age=31536000"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith(("/api", "/admin")):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.exception_handler(RequestValidationError)
async def _invalid(request: Request, exc: RequestValidationError):
    # The default handler echoes the submitted input, which could hold a key.
    return JSONResponse({"detail": "That request was not valid."}, status_code=422)


@app.get("/api/providers")
def providers():
    return {"claude": {"available": bedrock.available()}}


# ---------------------------------------------------------------- admin ----
ADMIN_HASH = os.getenv("ADMIN_PASSCODE_HASH", "")  # scrypt$N$r$p$salt$hash, never the passcode itself
ADMIN_PASSCODE = ADMIN_HASH  # truthy when the settings page is enabled
_sessions: dict = {}  # session token -> expiry
_fails: dict = {}  # client -> list of failure times
SESSION_TTL = 2 * 3600


def _passcode_ok(candidate: str) -> bool:
    import hashlib

    try:
        scheme, n, r, p, salt, want = ADMIN_HASH.split("$")
        got = hashlib.scrypt(
            candidate.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
            dklen=len(want) // 2, maxmem=128 * 1024 * 1024,
        )
        return scheme == "scrypt" and hmac.compare_digest(got.hex(), want)
    except Exception:
        return False


def _client_id(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() or (request.client.host if request.client else "?"))


def _authed(request: Request) -> bool:
    tok = request.cookies.get("ha_admin", "")
    exp = _sessions.get(tok)
    if not exp or exp < time.time():
        _sessions.pop(tok, None)
        return False
    return True


def _require(request: Request):
    if not ADMIN_PASSCODE or not _authed(request):
        raise HTTPException(404, "Not found")


@app.get("/admin")
def admin_page():
    if not ADMIN_PASSCODE:
        raise HTTPException(404, "Not found")
    return FileResponse(
        STATIC / "admin.html",
        headers={"X-Robots-Tag": "noindex, nofollow", "Cache-Control": "no-store"},
    )


@app.get("/robots.txt")
def robots():
    return Response("User-agent: *\nDisallow: /admin\n", media_type="text/plain")


class LoginIn(BaseModel):
    passcode: str


@app.post("/admin/api/login")
def admin_login(body: LoginIn, request: Request):
    if not ADMIN_PASSCODE:
        raise HTTPException(404, "Not found")
    who, now = _client_id(request), time.time()
    recent = [t for t in _fails.get(who, []) if now - t < 600]
    if len(recent) >= 8:
        raise HTTPException(429, "Too many tries. Wait a few minutes.")
    if not _passcode_ok(body.passcode):
        recent.append(now)
        _fails[who] = recent
        time.sleep(1.0)
        raise HTTPException(401, "Wrong passcode.")
    _fails.pop(who, None)
    tok = secrets.token_urlsafe(32)
    _sessions[tok] = now + SESSION_TTL
    resp = JSONResponse({"ok": True})
    resp.set_cookie("ha_admin", tok, max_age=SESSION_TTL, httponly=True, secure=True, samesite="strict", path="/admin")
    return resp


@app.get("/admin/api/status")
def admin_status(request: Request):
    _require(request)
    return bedrock.status()


class CredsIn(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = ""
    region: str
    model_id: str


@app.post("/admin/api/creds")
def admin_creds(body: CredsIn, request: Request):
    _require(request)
    vals = [v.strip() for v in (body.access_key_id, body.secret_access_key, body.session_token or "", body.region, body.model_id)]
    problem = bedrock.validate(*vals)
    if problem:
        raise HTTPException(400, problem)
    bedrock.set_creds(*vals)
    return bedrock.status()


@app.post("/admin/api/test")
def admin_test(request: Request):
    _require(request)
    ok, message = bedrock.test_invoke()
    return {"ok": ok, "message": message, **bedrock.status()}


@app.post("/admin/api/clear")
def admin_clear(request: Request):
    _require(request)
    bedrock.clear()
    return bedrock.status()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
