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
import threading
import time

import requests

import models

MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "6"))
ACTIVE_KEYS = int(os.getenv("ACTIVE_KEYS", "10"))
_state = threading.local()
_installed = [False]
_mu = threading.Lock()
_slots: list = []
_key_locks: dict = {}
_key_next: dict = {}


class PoolRateLimited(Exception):
    def __init__(self, daily, seconds=None):
        self.daily = daily
        self.seconds = seconds


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
            active = set(usable[:ACTIVE_KEYS])
            alive = [s for s in alive if s["ki"] in active]
            if not alive:
                soonest = min((s["dead_until"] for s in _slots), default=now) - now
            else:
                soonest = None
                alive.sort(key=lambda s: (s["rank"], _key_next[s["ki"]]))
                for s in alive:
                    if _key_locks[s["ki"]].acquire(blocking=False):
                        return s
        time.sleep(0.4 if soonest is None else min(15.0, max(1.0, soonest)))


def install():
    if _installed[0]:
        return
    _installed[0] = True
    _build()
    real_post = requests.post

    def shim(*a, **kw):
        r = real_post(*a, **kw)
        if getattr(_state, "active", False):
            if r.status_code == 429:
                raise PoolRateLimited("PerDay" in r.text)
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

    models.OpenAICompatibleProvider.chat = pooled
    print(f"[pool] installed: {len(_keys())} keys x {len(_models())} models = {len(_slots)} slots, active {min(ACTIVE_KEYS, len(_keys()))} + reserve {max(0, len(_keys()) - ACTIVE_KEYS)}", flush=True)
