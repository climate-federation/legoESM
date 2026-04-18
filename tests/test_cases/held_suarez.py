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
    # Decide whether to bootstrap ``<repo>/src`` using the *origin* of
    # the target submodule — not just whether it can be found. Two
    # distinct failure modes motivate this:
    #
    #   A. Stale install missing the submodule: a site-packages
    #      ``legoesm`` predates the #188 move and does not ship
    #      ``legoesm.atmosphere.held_suarez``. ``find_spec`` either
    #      returns ``None`` or raises ``ModuleNotFoundError`` on the
    #      missing parent package.
    #
    #   B. Stale install *with* the submodule: a different (older)
    #      copy of ``legoesm.atmosphere.held_suarez`` sits earlier on
    #      ``sys.path`` than this repo. ``find_spec`` returns a valid
    #      spec, but its origin is not under this repo's ``src/``, so
    #      the delegating import below would bind tests and notebooks
    #      to the wrong code. Force the checkout to win.
    import importlib.util

    src_dir = _Path(__file__).resolve().parents[2] / "src"
    if not src_dir.is_dir():
        return
    src_resolved = src_dir.resolve()

    try:
        spec = importlib.util.find_spec("legoesm.atmosphere.held_suarez")
    except (ModuleNotFoundError, ImportError):
        spec = None

    needs_bootstrap = spec is None
    if spec is not None:
        origin = getattr(spec, "origin", None)
        if not origin:
            needs_bootstrap = True
        else:
            try:
                origin_path = _Path(origin).resolve()
            except (OSError, ValueError):
                needs_bootstrap = True
            else:
                # Mismatch: the found submodule does not live under the
                # current checkout's ``src/``.
                try:
                    origin_path.relative_to(src_resolved)
                except ValueError:
                    needs_bootstrap = True

    if not needs_bootstrap:
        return

    src_str = str(src_dir)
    # Always remove any existing copies of ``src_dir`` from ``sys.path``
    # (including equivalently-normalized paths) and re-insert a single
    # fresh entry at position 0 so the checkout package wins against
    # stale installs earlier on the path *and* against cached
    # ``legoesm*`` modules already loaded from them.
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
