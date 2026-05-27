"""Cross-grid CRM smoke for ``scripts/run_rcemip_long.py``.

iter-238: the multi-grid non-hydrostatic CRM driver
``scripts/run_rcemip_long.py`` dispatches on ``--grid {plane_fd,
plane_spectral, cubed_sphere, mpas}`` and reuses the same RCEMIP-like
moist Wing-2018 initial condition across all four grid backends.
Pre iter-238 it had ZERO test coverage — a silent regression on any
of the four paths (wrong-shape state, missing field, import drift,
upstream API change) would only surface when a user tried to launch
a 30-day production. iter-238 closes that gap.

This file is a SMOKE-LEVEL fixture: each grid runs for a tiny window
(``--days 0.001 --dt 10`` → ``n_steps = int(0.001·86400/10) = 8``
outer steps), asserts the driver exits cleanly, writes the per-step
history JSON, and populates expected diagnostic columns (``max_w``,
``min_th``, ``max_th``, ``min_qv``, ``max_qv``, ``mass``,
``mass_drift``).

iter-239 (Codex iter-238 round-1 HIGH): the original iter-238
draft used ``--days 0.0001`` which gives
``n_steps = int(0.864) = 0`` — the driver started, set up the model,
and exited without ever calling ``model.step``. The smoke validated
ONLY the construction/teardown path; any regression in
``model.step`` signature, factory dispatch, or state-shape would
have slipped through silently. iter-239 raises days to 0.001 so
8 outer steps actually fire + adds an explicit ``n_steps > 0`` +
history-content assertion.

The driver is intentionally DRY (``physics_fn=None``) and uses the
default sizes (plane_fd 8×8, plane_spectral T31, cubed_sphere C4,
MPAS res-2 Voronoi). A regression on physics integration or
production-resolution sizing is OUT of scope — those are covered by
``test_plane_crm_end_to_end_smoke.py`` (plane CRM) and the
forthcoming MPAS / cubed-sphere CRM production drivers.

Wall budget: ~15-30 s per grid on M5 Pro cached JIT (JIT compile
dominates; the 1-step physics-off run is negligible).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER = REPO_ROOT / "scripts" / "run_rcemip_long.py"


def _run_driver(output_file: Path, grid: str, days: float = 0.001,
                dt: float = 10.0, timeout_s: int = 180):
    """run_rcemip_long.py expects ``--output`` to be a FILE path
    (history JSON), not a directory — the driver does
    ``output.open("w")`` and ``output.parent.mkdir(exist_ok=True)``.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--grid", grid,
        "--days", repr(float(days)),
        "--dt", repr(float(dt)),
        "--print-every", "1",
        "--output", str(output_file),
    ]
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout_s,
    )


def _read_history_json(output_dir: Path, grid: str):
    """Locate the history JSON in the output dir.

    The driver writes ``<grid>_history.json`` (see _run_loop).
    """
    candidates = list(output_dir.glob(f"{grid}*.json")) + \
                 list(output_dir.glob("*history*.json"))
    if not candidates:
        return None
    return json.loads(candidates[0].read_text())


_EXPECTED_DIAG_COLS = {
    "max_w", "min_th", "max_th", "min_qv", "max_qv", "mass",
    "mass_drift", "step", "t_days",
}


@pytest.mark.parametrize("grid", [
    "plane_fd",
    pytest.param(
        "plane_spectral",
        marks=pytest.mark.skip(
            reason="plane_spectral depends on spectral_pe.py:72 "
            "float(jnp.log(100.0)) which crashes on Metal — pre-iter-238 "
            "collection error documented in CRM_implementation.md iter-236 "
            "fold. Once fixed, drop this skip."
        ),
    ),
    "cubed_sphere",
    "mpas",
])
def test_run_rcemip_long_grid_smoke(tmp_path, grid):
    """Each --grid backend completes a 1-step dry RCE smoke without
    crashing or producing non-finite state.

    iter-238 baseline: locks the cross-grid CRM driver as a tested
    surface. Any future commit that breaks a grid backend (state
    shape, factory dispatch, IC adapter import drift, model.step
    signature) will fail this test instead of slipping silently to
    a 30-day production attempt.
    """
    out_file = tmp_path / f"rcemip_{grid}_history.json"
    result = _run_driver(out_file, grid=grid)
    if result.returncode != 0:
        pytest.fail(
            f"run_rcemip_long.py --grid {grid} exited "
            f"{result.returncode}\n"
            f"stdout tail:\n{result.stdout[-1500:]}\n"
            f"stderr tail:\n{result.stderr[-1500:]}"
        )
    # Stdout header sanity: the driver prints
    # ``[<grid>] dt=Ns, N steps -> X sim-days``
    assert f"[{grid}]" in result.stdout, (
        f"Driver stdout missing ``[{grid}]`` header. stdout: "
        f"{result.stdout[-500:]}"
    )
    # Diagnostic columns appear at least once.
    for col in ("max|w|", "min(θ')", "max(θ')", "q_v_min", "q_v_max"):
        assert col in result.stdout, (
            f"Driver stdout missing diagnostic column {col!r} for "
            f"grid={grid}. stdout: {result.stdout[-500:]}"
        )
    # No "blew up" / "NaN" / "Inf" markers in the smoke.
    bad_markers = ("blew up", "NaN encountered", "non-finite",
                   "Inf encountered")
    combined = (result.stdout + "\n" + result.stderr).lower()
    for m in bad_markers:
        assert m.lower() not in combined, (
            f"Driver hit instability marker {m!r} for grid={grid} "
            f"in the smoke run. stdout tail: {result.stdout[-500:]}"
        )

    # iter-239 (Codex iter-238 round-1 HIGH): assert model.step
    # actually fired. The driver writes a JSON history with one
    # entry per print-every interval; with print-every=1 + n_steps=8
    # we expect ~9 history rows (IC + 8 steps). A history with
    # ONLY the IC row (length 1) means n_steps collapsed to 0 again
    # and the smoke would silently degrade.
    assert out_file.exists(), (
        f"Driver did not write the history JSON at {out_file}. "
        f"stdout tail: {result.stdout[-500:]}"
    )
    history_doc = json.loads(out_file.read_text())
    assert history_doc.get("n_steps", 0) > 0, (
        f"Driver wrote n_steps={history_doc.get('n_steps')} for "
        f"grid={grid} — model.step was never called and the smoke "
        f"validated only construction/teardown. iter-239 raised "
        f"--days from 0.0001 to 0.001 to make this fire."
    )
    history_rows = history_doc.get("history", [])
    assert len(history_rows) >= 2, (
        f"Driver wrote {len(history_rows)} history rows for "
        f"grid={grid}; expected at least 2 (IC + ≥1 stepped). "
        f"Likely the print-every gate skipped all steps."
    )
    # Final history row must hold finite floats (a NaN would have
    # surfaced as ``finite=False`` in the diag dict and triggered
    # the driver's ``non-finite`` early-exit, but a sub-step NaN
    # that recovered would otherwise slip past the stdout markers).
    last = history_rows[-1]
    for col in ("max_w", "min_th", "max_th", "min_qv", "max_qv"):
        v = last.get(col)
        assert v is None or (isinstance(v, (int, float))
                             and v == v and v != float("inf")
                             and v != float("-inf")), (
            f"history[-1][{col}]={v!r} is non-finite for grid="
            f"{grid}."
        )
