"""Per-run settings carried through the worker and its helper threads."""

import contextvars

_run = contextvars.ContextVar("run_settings", default=None)


def set_run(settings):
    return _run.set(settings)


def get_run():
    return _run.get()


def spawn(fn, *args):
    """Return a callable that runs fn inside a copy of the current context."""
    ctx = contextvars.copy_context()
    return lambda: ctx.run(fn, *args)
