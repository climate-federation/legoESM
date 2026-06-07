"""Checkpoint save/restore for the RCEMIP plane-CRM run.

Saves the full prognostic state (u, v, w, theta', rho', tracers, phis) + the global
step and sim-time to a compressed npz, so a long run can RESUME after a crash, a
fix, or a manual stop instead of re-spinning from t=0. Kept separate from the
driver; the driver adds ``--checkpoint-every-days`` + ``--restart <path>`` and two
small hooks.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

_FIELDS = ("u", "v", "w", "theta_prime", "rho_prime", "phis", "tracers")


def save(ckpt_dir, step: int, t_s: float, state, keep_last: int = 3) -> Path:
    """Write ``ckpt_<step>.npz`` (and prune to the most recent ``keep_last``)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    arrays = {f: np.asarray(getattr(state, f).data) for f in _FIELDS}
    path = ckpt_dir / f"ckpt_{step:09d}.npz"
    np.savez(path, step=step, t_s=t_s, **arrays)
    # prune
    existing = sorted(ckpt_dir.glob("ckpt_*.npz"))
    for old in existing[:-keep_last]:
        old.unlink()
    return path


def load(path, state):
    """Restore arrays from ``path`` onto ``state`` (a freshly-built template of the
    SAME shapes/dtype). Returns ``(state, step, t_s)``."""
    d = np.load(path)
    dtype = state.u.data.dtype
    repl = {f: getattr(state, f).replace(data=jnp.asarray(d[f], dtype=dtype))
            for f in _FIELDS}
    return state._replace(**repl), int(d["step"]), float(d["t_s"])


def latest(ckpt_dir):
    """Path to the newest checkpoint in ``ckpt_dir`` (or None)."""
    files = sorted(Path(ckpt_dir).glob("ckpt_*.npz"))
    return files[-1] if files else None
