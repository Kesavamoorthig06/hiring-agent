"""Global, process-wide pacing for LLM calls.

Every provider call takes the same lock and waits until at least
LLM_MIN_INTERVAL seconds have passed since the previous call finished
starting. Calls therefore run one at a time, which keeps a free-tier API key
under its per-minute quota. The repo's own retry/backoff still applies on top.
"""

import os
import threading
import time

import models

_lock = threading.Lock()
_last = [0.0]
MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "12"))
_installed = [False]


def install():
    if _installed[0]:
        return
    _installed[0] = True
    original = models.OpenAICompatibleProvider.chat

    def paced(self, *args, **kwargs):
        with _lock:
            wait = _last[0] + MIN_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                return original(self, *args, **kwargs)
            finally:
                _last[0] = time.monotonic()

    models.OpenAICompatibleProvider.chat = paced
