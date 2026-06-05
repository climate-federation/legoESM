"""Core infrastructure for legoESM."""

from legoesm.core.cfl import cfl_check_and_adjust, cfl_max_dt
from legoesm.core.field import Field
from legoesm.core.smooth import sigmoid_switch, smooth_max, smooth_min, smooth_clamp

# Canonical backend utilities live in legoesm.runtime.backend; prefer importing
# them from there in new code.  They are re-exported here LAZILY (module-level
# __getattr__) so that importing legoesm.core does NOT eagerly import
# legoesm.runtime: a top-level core->runtime import re-enters runtime/__init__
# mid-initialisation and breaks isolated pytest of core modules (CLAUDE.md).
_RUNTIME_BACKEND_REEXPORTS = ("check_spectral_backend", "get_backend")


def __getattr__(name):
    if name in _RUNTIME_BACKEND_REEXPORTS:
        from legoesm.runtime import backend
        return getattr(backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
