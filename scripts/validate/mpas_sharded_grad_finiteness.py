"""Is the unstructured atmosphere's reverse-mode gradient finite when sharded?

The serial step's gradient is finite. The sharded step's is not: entries go
non-finite on a small set of interior cells, and the count grows with the
device count. Since end-to-end ``jax.grad`` compatibility is a goal of this
model, that is a defect of the sharded path, not of the physics -- the same
initial state, the same time step and the same loss give a clean gradient on
one device.

This probe exists because the number was found by accident, while gating an
unrelated halo change, and an uncommitted number is not a measurement.

    python scripts/validate/mpas_sharded_grad_finiteness.py --devices 1 2 4

It prints one row per device count: how many entries of dL/dT are non-finite,
and how many distinct cells they sit on. A clean sharded path prints zero on
every row.

Run on CPU with enough fake devices, e.g.::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 python ...

Deliberately NOT a pytest gate: it currently fails, and a red gate nobody can
turn green gets deleted or skipped. Fix the sharded gradient first, then
promote it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class _Env:
    """The builder below takes pytest's monkeypatch; this is the same
    interface for a script, and it deliberately does not undo itself -- the
    process exits."""

    def setenv(self, key: str, value: str) -> None:
        import os

        os.environ[key] = value


def _grad_non_finite(devices: int) -> tuple[int, int, int]:
    """Return (non-finite entries, distinct cells, total entries)."""
    sys.path.insert(0, str(_REPO / "tests" / "parallel"))
    from test_mpas_halo_merge_scatter import _build  # noqa: PLC0415

    step, state, dt = _build(devices, merge="0", monkeypatch=_Env())

    def loss(field):
        stepped = step(state._replace(T=state.T.replace(data=field)), dt)
        return jnp.sum(stepped.T.data ** 2)

    grad = np.asarray(jax.grad(loss)(state.T.data))
    bad = ~np.isfinite(grad)
    return int(bad.sum()), int(np.unique(np.where(bad)[0]).size), int(grad.size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    print(f"{'devices':>8} {'non-finite':>11} {'cells':>7} {'entries':>9}")
    worst = 0
    for devices in args.devices:
        if devices > jax.device_count():
            print(f"{devices:>8} {'SKIPPED, only':>11} "
                  f"{jax.device_count()} devices visible")
            continue
        n_bad, n_cells, n_total = _grad_non_finite(devices)
        worst = max(worst, n_bad)
        print(f"{devices:>8} {n_bad:>11} {n_cells:>7} {n_total:>9}")
    if worst:
        print("\nThe sharded gradient is not finite. See the module docstring.")
    return 1 if worst else 0


if __name__ == "__main__":
    raise SystemExit(main())
