"""Backwards-compat shim.

The canonical Held-Suarez forcing and initialization live in
``legoesm.atmosphere.held_suarez`` so that production entry points
(``run_amip.py``, ``ModelDriver``, benchmark scripts) can import them
without requiring the ``tests/`` directory to be on ``sys.path``.

Keep this shim so existing ``from tests.test_cases.held_suarez import ...``
call sites (other test modules, notebooks) continue to work. To
accommodate raw-checkout usage where the package is not pip-installed
in editable mode, insert ``<repo>/src`` onto ``sys.path`` as a last
resort before the delegating import — this preserves the old
contract for callers that only have the repo root on their path.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


def _ensure_legoesm_importable() -> None:
    # Check the *specific* submodule we need, not just top-level
    # ``legoesm``. A stale ``legoesm`` installed in site-packages (e.g.
    # an older wheel that predates the held_suarez move in #188) will
    # satisfy ``import legoesm`` but still miss this submodule, so we
    # must confirm the target can actually be located before deciding
    # not to bootstrap ``<repo>/src``.
    import importlib.util

    try:
        spec = importlib.util.find_spec("legoesm.atmosphere.held_suarez")
    except (ModuleNotFoundError, ImportError):
        # find_spec raises when an intermediate parent package is
        # missing (e.g. stale install has ``legoesm`` but no
        # ``legoesm.atmosphere``). Treat that as "not importable" so
        # the fallback fires.
        spec = None
    if spec is None:
        src_dir = _Path(__file__).resolve().parents[2] / "src"
        if not src_dir.is_dir():
            return
        src_str = str(src_dir)
        # Remove any existing copies of src_dir from sys.path (including
        # equivalently-normalized paths) and prepend a fresh entry so
        # the checkout package always wins against a stale ``legoesm``
        # that is already earlier on ``sys.path`` or cached in
        # ``sys.modules``. This covers the mixed case where the
        # editable-install entry is on the path but shadowed by a
        # stale package resolved earlier in the process.
        src_resolved = _Path(src_str).resolve()
        _sys.path = [
            p for p in _sys.path
            if p and _Path(p).resolve() != src_resolved
        ]
        _sys.path.insert(0, src_str)
        importlib.invalidate_caches()
        for mod in [
            m for m in list(_sys.modules)
            if m == "legoesm" or m.startswith("legoesm.")
        ]:
            del _sys.modules[mod]


_ensure_legoesm_importable()
del _ensure_legoesm_importable, _sys, _Path


from legoesm.atmosphere.held_suarez import *  # noqa: E402,F401,F403
from legoesm.atmosphere.held_suarez import (  # noqa: E402,F401
    K_A,
    K_S,
    K_F,
    SIGMA_B,
    DELTA_T_Y,
    DELTA_THETA_Z,
    T_MIN,
    P_0,
    held_suarez_equilibrium_temperature,
    held_suarez_forcing,
    held_suarez_forcing_latlon,
    held_suarez_forcing_mpas,
    held_suarez_forcing_spectral,
    held_suarez_init,
    held_suarez_init_latlon,
    held_suarez_init_mpas,
)
