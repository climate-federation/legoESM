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
from pathlib import Path

import pytest

from tests.atmosphere.nonhydrostatic.integration._bench_smoke_helpers import (
    REPO_ROOT, run_bench,
)


DRIVER = REPO_ROOT / "scripts" / "run_rcemip_long.py"


def _run_driver(output_file: Path, grid: str, days: float = 0.001,
                dt: float = 10.0, timeout_s: int = 180):
    """run_rcemip_long.py expects ``--output`` to be a FILE path
    (history JSON), not a directory — the driver does
    ``output.open("w")`` and ``output.parent.mkdir(exist_ok=True)``.
    """
    return run_bench(DRIVER, [
        "--grid", grid,
        "--days", repr(float(days)),
        "--dt", repr(float(dt)),
        "--print-every", "1",
        "--output", str(output_file),
    ], timeout_s=timeout_s)


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
    "cubed_sphere",
    "mpas",
])
def test_run_rcemip_long_cross_grid_moist_smoke(tmp_path, grid):
    """iter-275: opt-in ``--moist`` flag for the cubed_sphere + mpas
    paths composes Kessler microphysics + gray radiation into a
    single physics_fn via ``_compose_nh_moist_physics``. Pre-iter-275
    these paths were dry-only (``physics_fn=None``).

    Single-process smoke at small mesh + 8 outer steps verifies:
    * --moist parses + composes the moist physics_fn cleanly
    * 8 outer steps complete (no NaN, no compose error)
    * label includes ``_moist`` suffix in the history JSON
    """
    # iter-276 (Codex iter-275 round-1 MEDIUM): the iter-275 draft
    # called _run_driver() WITHOUT --moist before the moist
    # run_bench, paying for an extra subprocess/JIT and silently
    # ignoring any nonzero exit. Dropped the redundant dry pre-run;
    # the dry path is already covered by the parametrise above.
    out_file_moist = tmp_path / f"rcemip_{grid}_moist_v2.json"
    result = run_bench(DRIVER, [
        "--grid", grid,
        "--days", "0.001",
        "--dt", "10.0",
        "--print-every", "1",
        "--moist",
        "--output", str(out_file_moist),
    ])
    if result.returncode != 0:
        pytest.fail(
            f"run_rcemip_long.py --moist --grid {grid} exited "
            f"{result.returncode}\n"
            f"stdout tail:\n{result.stdout[-1500:]}\n"
            f"stderr tail:\n{result.stderr[-1500:]}"
        )
    # The label in stdout should be ``<grid>_moist`` (per iter-275
    # branch).
    assert f"[{grid}_moist]" in result.stdout, (
        f"Driver stdout missing ``[{grid}_moist]`` marker. "
        f"Did the --moist flag dispatch?\n"
        f"stdout tail:\n{result.stdout[-500:]}"
    )
    # JSON history must exist + show n_steps=8 (= int(0.001*86400/10)).
    assert out_file_moist.exists()
    doc = json.loads(out_file_moist.read_text())
    assert doc["n_steps"] == 8, (
        f"Driver n_steps={doc['n_steps']}, expected 8. moist "
        f"path may have early-aborted."
    )
    # iter-276 (Codex iter-275 round-1 LOW): also assert the
    # JSON history label — not just the stdout marker — so a
    # label-schema regression that breaks the JSON side without
    # touching stdout fires.
    assert doc.get("label") == f"{grid}_moist", (
        f"history JSON label={doc.get('label')!r}, expected "
        f"{grid}_moist."
    )
    # Last history row diag fields finite + present.
    history_rows = doc["history"]
    assert len(history_rows) >= 2
    last = history_rows[-1]
    for col in ("max_w", "min_th", "max_th", "min_qv", "max_qv"):
        assert col in last
        v = last[col]
        assert isinstance(v, (int, float)) and v == v
    # iter-276 (Codex iter-275 round-1 LOW): tightened q_v upper
    # bound. IC max q_v = Q_V_SFC * exp(-z/4000) at z=1 km
    # ~ 0.012 * 0.78 ~ 9.35e-3. The iter-275 bound 0.05 was 5x
    # over IC — would silently miss a 4x moisture inflation.
    # Tightened to (-1e-10, 0.015) = IC + 60% margin.
    # iter-277 (Codex iter-276 round-2 MEDIUM-1): scan ALL history
    # rows, not just the last. A transient q_v spike at step 3
    # that recovers by step 8 would slip through a last-row-only
    # check.
    qv_max_over_run = max(float(r["max_qv"]) for r in history_rows)
    qv_min_over_run = min(float(r["min_qv"]) for r in history_rows)
    assert 0.0 <= qv_max_over_run < 0.015, (
        f"max_qv over the run = {qv_max_over_run} outside "
        f"(0, 0.015) — IC max ~9.35e-3 at z=1 km; >0.015 means "
        f"moisture inflation regression (transient or final)."
    )
    assert qv_min_over_run >= -1e-10, (
        f"min_qv over the run = {qv_min_over_run} below -1e-10 "
        f"floor — negative-bias regression in moist tracer "
        f"positivity filter."
    )


@pytest.mark.parametrize("grid", [
    "plane_fd",
    # iter-241: removed the iter-240 xfail-strict marker after
    # cherry-picking SpectralPlanePhysicsState +
    # SpectralPlanePhysicsTendencies from the unmerged
    # feature/crm-plane-spectral branch (edbae138) into
    # src/legoesm/core/state.py. The subprocess JAX_PLATFORMS=cpu
    # wrapper insulates against the unrelated
    # spectral_pe.py:72 Metal bug, so plane_spectral now PASSES.
    # The xfail-strict design did its job — XPASS fired the
    # moment the fix was in place.
    "plane_spectral",
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

    # iter-239 (Codex iter-238 round-1 HIGH) + iter-240
    # (Codex iter-239 round-2 MEDIUM#1+#2): assert model.step
    # actually fired with the EXACT contract — print-every=1 +
    # days=0.001 + dt=10 → n_steps=8 → history len = 9 (IC + 8
    # steps with step indices 0,1,2,...,8).
    assert out_file.exists(), (
        f"Driver did not write the history JSON at {out_file}. "
        f"stdout tail: {result.stdout[-500:]}"
    )
    history_doc = json.loads(out_file.read_text())
    expected_n_steps = 8  # int(0.001 * 86400 / 10)
    assert history_doc.get("n_steps") == expected_n_steps, (
        f"Driver wrote n_steps={history_doc.get('n_steps')!r} for "
        f"grid={grid}; expected exactly {expected_n_steps} (= "
        f"int(0.001·86400/10)). A mismatch means the driver step "
        f"calc regressed (n_steps=0 would mean model.step never "
        f"fires — Codex iter-238 round-1 HIGH)."
    )
    history_rows = history_doc.get("history", [])
    assert len(history_rows) == expected_n_steps + 1, (
        f"Driver wrote {len(history_rows)} history rows for "
        f"grid={grid}; expected exactly {expected_n_steps + 1} "
        f"(IC + {expected_n_steps} steps at print-every=1). A "
        f"short history means a step was skipped or the print "
        f"gate regressed."
    )
    actual_steps = [row.get("step") for row in history_rows]
    expected_steps = list(range(expected_n_steps + 1))
    assert actual_steps == expected_steps, (
        f"Driver wrote step sequence {actual_steps!r} for "
        f"grid={grid}; expected exactly {expected_steps!r} "
        f"(monotonic 0..{expected_n_steps}). A gap means the "
        f"driver skipped a step; a duplicate means double-logging."
    )
    # iter-240 (Codex round-2 MEDIUM#2): require each diag field
    # present + finite — NOT ``v is None or finite``. Missing
    # diagnostic fields are the schema-drift this smoke is meant
    # to catch.
    # iter-240 (Codex round-2 LOW#1): scoped claim — this catches
    # a NaN at the FINAL outer step, not sub-step NaNs that
    # recovered inside model.step.
    last = history_rows[-1]
    for col in ("max_w", "min_th", "max_th", "min_qv", "max_qv"):
        assert col in last, (
            f"history[-1] for grid={grid} is missing required "
            f"diagnostic field {col!r}. Driver schema regressed."
        )
        v = last[col]
        assert (isinstance(v, (int, float))
                and v == v
                and v not in (float("inf"), float("-inf"))), (
            f"history[-1][{col}]={v!r} is non-finite for grid="
            f"{grid} (final outer-step diag). A sub-step NaN that "
            f"recovered before the diag fire would slip past this — "
            f"out of scope for the 8-step smoke."
        )
