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
    try:
        import legoesm  # noqa: F401
    except ModuleNotFoundError:
        src_dir = _Path(__file__).resolve().parents[2] / "src"
        if src_dir.is_dir() and str(src_dir) not in _sys.path:
            _sys.path.insert(0, str(src_dir))


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
