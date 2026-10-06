"""Pool of (API key, model) slots for LLM calls.

Each call picks a healthy slot. A key serves one prompt at a time and is paced
by LLM_MIN_INTERVAL. A 429 on a slot marks it cooling down (hours when the
daily quota is gone, about a minute otherwise) and the call moves to another
slot, so callers never see a rate limit unless every slot is exhausted.
Keys: GEMINI_API_KEYS (comma separated) or GEMINI_API_KEY.
Models, in preference order: POOL_MODELS (comma separated) or DEFAULT_MODEL.
"""

import copy
import os
import sys
import threading
import time

import requests

import models
from webapp import bedrock, runctx

MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "6"))
CALL_TIMEOUT = float(os.getenv("LLM_CALL_TIMEOUT", "15"))
MAX_PARK_WAIT = float(os.getenv("POOL_MAX_PARK_WAIT", "90"))
ACTIVE_KEYS = int(os.getenv("ACTIVE_KEYS", "10"))
_state = threading.local()
_installed = [False]
_mu = threading.Lock()
_slots: list = []
_key_locks: dict = {}
_key_next: dict = {}
_usage: dict = {}  # ki -> {"day": "YYYY-MM-DD" (Pacific), "req": int, "tok": int}

# Free-tier quota guards. Google resets daily quota at midnight Pacific time.
# Set these to match the model's free-tier limits; keys are rotated out at
# ROTATE_AT of the daily request limit, before Google starts returning 429.
KEY_DAILY_REQUESTS = int(os.getenv("KEY_DAILY_REQUESTS", "200"))
KEY_DAILY_TOKENS = int(os.getenv("KEY_DAILY_TOKENS", "0"))  # 0 = do not track tokens
ROTATE_AT = float(os.getenv("KEY_ROTATE_AT", "0.85"))
WARN_FRACTION = float(os.getenv("POOL_WARN_FRACTION", "0.25"))
_warned = [False]


class OwnKeyRejected(Exception):
    """Safe to show to the visitor."""


class PoolExhausted(Exception):
    """Every pooled Gemini slot is parked (quota gone or key rejected). Safe to show."""


class PoolRateLimited(Exception):
    def __init__(self, daily, seconds=None):
        self.daily = daily
        self.seconds = seconds


def _today():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")


def _secs_to_reset():
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    n = datetime.now(ZoneInfo("America/Los_Angeles"))
    nxt = (n + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    return max(60, int((nxt - n).total_seconds()))


def _u(ki):
    u = _usage.get(ki)
    t = _today()
    if u is None or u["day"] != t:
        u = _usage[ki] = {"day": t, "req": 0, "tok": 0}
    return u


def _fraction_used(ki):
    u = _u(ki)
    f = u["req"] / KEY_DAILY_REQUESTS if KEY_DAILY_REQUESTS > 0 else 0
    if KEY_DAILY_TOKENS > 0:
        f = max(f, u["tok"] / KEY_DAILY_TOKENS)
    return f


def _record(ki, tokens=0):
    with _mu:
        u = _u(ki)
        u["req"] += 1
        u["tok"] += tokens
    f = _fraction_used(ki)
    if f >= ROTATE_AT and f - 1.0 / max(KEY_DAILY_REQUESTS, 1) < ROTATE_AT:
        print(f"[pool] key#{ki} reached {int(f * 100)}% of its daily quota, rotating to a backup key", flush=True)
    _check_health()


def status():
    """Per-key capacity for the admin page and logs. Never contains key material."""
    now = time.monotonic()
    keys = []
    total_left = 0.0
    n = len(_keys())
    for ki in range(n):
        ms = [x for x in _slots if x["ki"] == ki]
        parked = max((x["dead_until"] for x in ms), default=0) - now if ms and all(x["dead_until"] > now for x in ms) else 0
        u = _u(ki)
        used = _fraction_used(ki)
        left = 0.0 if parked > 0 else max(0.0, 1.0 - used)
        total_left += left
        state = "parked" if parked > 0 else ("rotated-out" if used >= ROTATE_AT else "active")
        keys.append({
            "key": f"key#{ki}", "state": state,
            "requests_today": u["req"], "daily_request_limit": KEY_DAILY_REQUESTS,
            "tokens_today": u["tok"], "remaining_pct": round(left * 100),
            "parked_for_seconds": int(parked) if parked > 0 else 0,
        })
    capacity = (total_left / n) if n else 0.0
    return {
        "keys": keys, "key_count": n,
        "capacity_pct": round(capacity * 100),
        "warning": (n == 0) or capacity < WARN_FRACTION,
        "message": ("No Gemini keys configured." if n == 0 else
                    f"Gemini pool low: {round(capacity * 100)}% of today's free quota left across {n} keys. Add keys to GEMINI_API_KEYS."
                    if capacity < WARN_FRACTION else "ok"),
    }


def _check_health():
    st = status()
    if st["warning"] and not _warned[0]:
        _warned[0] = True
        print(f"[pool] WARNING {st['message']}", flush=True)
    elif not st["warning"]:
        _warned[0] = False


def available():
    """True when at least one shared Gemini key can still take work."""
    return any(k["state"] != "parked" for k in status()["keys"])


def key_count():
    return max(1, len(_keys()))


def _keys():
    raw = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


def _models():
    raw = os.getenv("POOL_MODELS") or os.getenv("DEFAULT_MODEL", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _build():
    keys, mods = _keys(), _models()
    for mi, m in enumerate(mods):
        for ki, k in enumerate(keys):
            _slots.append({"key": k, "ki": ki, "model": m, "rank": mi, "dead_until": 0.0})
    for ki in range(len(keys)):
        _key_locks[ki] = threading.Lock()
        _key_next[ki] = 0.0


def _pick():
    while True:
        now = time.monotonic()
        with _mu:
            alive = [s for s in _slots if s["dead_until"] <= now]
            # Active set = the first ACTIVE_KEYS keys that are not daily-parked.
            # Keys past that stay in reserve and are promoted one by one as
            # active keys run out of daily quota. Within the active set the
            # least recently used key goes next, so load spreads evenly.
            usable = []
            for ki in sorted(_key_locks):
                if any(x["ki"] == ki and x["dead_until"] - now < 1800 for x in _slots):
                    usable.append(ki)
            # Keys at ROTATE_AT of their daily quota step back while any other
            # key still has room, so a key is never driven into a 429.
            fresh = [ki for ki in usable if _fraction_used(ki) < ROTATE_AT]
            if fresh:
                usable = fresh
            active = set(usable[:ACTIVE_KEYS])
            alive = [s for s in alive if s["ki"] in active]
            if not alive:
                soonest = min((s["dead_until"] for s in _slots), default=now) - now
                if soonest > MAX_PARK_WAIT:
                    raise PoolExhausted(
                        "The shared Gemini keys are out of quota right now. "
                        "Try the Claude option, use your own key, or retry later."
                    )
            else:
                soonest = None
                alive.sort(key=lambda s: (s["rank"], _key_next[s["ki"]]))
                for s in alive:
                    if _key_locks[s["ki"]].acquire(blocking=False):
                        return s
        time.sleep(0.4 if soonest is None else min(15.0, max(1.0, soonest)))


def _stage_name():
    f = sys._getframe(2)
    while f is not None:
        sn = f.f_locals.get("section_name")
        if isinstance(sn, str):
            return "parse:" + sn
        if f.f_code.co_name in ("evaluate_resume", "analyze_pdf", "chat_reply", "_github_data", "select_projects"):
            return f.f_code.co_name
        f = f.f_back
    return "unknown"


def install():
    if _installed[0]:
        return
    _installed[0] = True
    _build()
    real_post = requests.post

    def shim(*a, **kw):
        t0 = time.monotonic()
        if getattr(_state, "active", False):
            print(f"[call-start] key#{getattr(_state, 'ki', '?')} stage={getattr(_state, 'stage', '?')} model={getattr(_state, 'model', '?')}", flush=True)
        if getattr(_state, "active", False):
            kw["timeout"] = CALL_TIMEOUT  # a slow model must not hold a call for minutes
        try:
            r = real_post(*a, **kw)
        except Exception as e:
            if getattr(_state, "active", False):
                print(f"[call] key#{getattr(_state, 'ki', '?')} stage={getattr(_state, 'stage', '?')} error={type(e).__name__} after {time.monotonic() - t0:.1f}s", flush=True)
                if isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
                    raise PoolRateLimited(False, 30)  # fail over to the next model/key at once
            raise
        if getattr(_state, "active", False):
            print(f"[call] key#{getattr(_state, 'ki', '?')} stage={getattr(_state, 'stage', '?')} model={getattr(_state, 'model', '?')} status={r.status_code} in {time.monotonic() - t0:.1f}s", flush=True)
            ki_ = getattr(_state, "ki", None)
            if isinstance(ki_, int):
                tok = 0
                try:
                    tok = int((r.json().get("usage") or {}).get("total_tokens") or 0)
                except Exception:
                    pass
                _record(ki_, tok)
            if r.status_code == 429:
                if "PerDay" in r.text:
                    raise PoolRateLimited(True, _secs_to_reset())
                raise PoolRateLimited(False)
            if r.status_code in (401, 403):
                raise PoolRateLimited(True, 24 * 3600)  # bad/revoked key: park the slot
            if r.status_code == 503:
                raise PoolRateLimited(False, 30)  # model overloaded: try another slot
        return r

    requests.post = shim
    original = models.OpenAICompatibleProvider.chat

    def pooled(self, model, messages, options=None, **kwargs):
        if not _slots:
            return original(self, model, messages, options, **kwargs)
        run = runctx.get_run() or {}
        if run.get("provider") == "claude":
            return bedrock.chat(messages, options, kwargs.get("format"))
        if run.get("own_key"):
            return own_key_call(self, messages, options, kwargs, run["own_key"], _stage_name())
        attempts = 0
        while True:
            slot = _pick()
            ki = slot["ki"]
            try:
                wait = _key_next[ki] - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                prov = copy.copy(self)
                prov.api_key = slot["key"]
                _state.active = True
                _state.ki = ki
                _state.stage = _stage_name()
                _state.model = slot['model']
                try:
                    return original(prov, slot["model"], messages, options, **kwargs)
                finally:
                    _state.active = False
                    _key_next[ki] = time.monotonic() + MIN_INTERVAL
            except PoolRateLimited as e:
                attempts += 1
                slot["dead_until"] = time.monotonic() + (e.seconds or (6 * 3600 if e.daily else 70))
                print(f"[pool] slot key#{ki} {slot['model']} cooling ({'daily' if e.daily else 'rpm'}, {int(e.seconds or (6*3600 if e.daily else 70))}s)")
                if attempts > 60:
                    raise
            finally:
                _key_locks[ki].release()

    def own_key_call(self, messages, options, kwargs, own_key, stage):
        # A visitor's own Gemini key: used for this run only, never pooled or stored.
        for rnd in range(2):
            for m in _models():
                prov = copy.copy(self)
                prov.api_key = own_key
                _state.active = True
                _state.ki = "own"
                _state.stage = stage
                _state.model = m
                try:
                    return original(prov, m, messages, options, **kwargs)
                except PoolRateLimited as e:
                    if e.seconds == 24 * 3600:
                        raise OwnKeyRejected("That Gemini key was not accepted. Check it and try again.")
                except requests.HTTPError as e:
                    if "API_KEY_INVALID" in str(e) or "API key not valid" in str(e):
                        raise OwnKeyRejected("That Gemini key was not accepted. Check it and try again.")
                    raise
                finally:
                    _state.active = False
            time.sleep(3)
        raise OwnKeyRejected("Your Gemini key is out of free quota right now. Try again later or use the shared option.")

    models.OpenAICompatibleProvider.chat = pooled
    _check_health()
    print(f"[pool] installed: {len(_keys())} keys x {len(_models())} models = {len(_slots)} slots, active {min(ACTIVE_KEYS, len(_keys()))} + reserve {max(0, len(_keys()) - ACTIVE_KEYS)}", flush=True)
