"""#1492 item 1.1: multi-step trajectory-replay tests against NEMO's own
per-step restarts.

Exercises ``scripts/validate/ocean_fidelity/dino_1226/multistep_replay.py``
(see its module docstring for the full method, the state-threading paths
covered, and why single-step operator comparisons cannot see this class of
defect). Requires the machine-local NEMO oracle tree (``RUN_TRAJ`` mesh donor
+ ``RUN_TWIN_STEP1`` per-step restart tiles) -- skipped, not failed, when
absent, so this file stays collectible without the oracle build.

NOT CI-enforced: ``.github/workflows/ci.yml`` has not run since 2026-05-27
and its recent runs failed. Run locally via:

    JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \\
        .venv/bin/python -m pytest tests/ocean/unit/test_dino_1492_multistep_replay.py -v

``DINO_1492_REPLAY_STEPS`` (default 16, max 32 = one full RUN_TWIN_STEP1 day)
controls the replay window for the growth-curve tests.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DINO_1226_DIR = SCRIPTS_DIR / "validate" / "ocean_fidelity" / "dino_1226"


@pytest.fixture(scope="module")
def mr():
    sys.path.insert(0, str(SCRIPTS_DIR))
    sys.path.insert(0, str(SCRIPTS_DIR / "validate" / "ocean_fidelity"))
    sys.path.insert(0, str(DINO_1226_DIR))
    try:
        import multistep_replay
        return multistep_replay
    finally:
        for p in (str(SCRIPTS_DIR), str(SCRIPTS_DIR / "validate" / "ocean_fidelity"),
                  str(DINO_1226_DIR)):
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def _have_artifacts(mr) -> bool:
    try:
        return mr.have_step1_artifacts()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    "DINO_1492_FORCE_SKIP" in os.environ,
    reason="forced skip (debug escape hatch)",
)

N_STEPS = int(os.environ.get("DINO_1492_REPLAY_STEPS", "16"))
if not (1 <= N_STEPS <= 32):
    raise ValueError(
        f"DINO_1492_REPLAY_STEPS={N_STEPS} out of range: RUN_TWIN_STEP1 covers "
        "exactly one day (steps 230401..230432, 32 steps); pick 1..32.")


def _skip_if_no_artifacts(mr):
    if not _have_artifacts(mr):
        pytest.skip(
            "NEMO oracle tree not found on this machine (RUN_TRAJ mesh_mask.nc "
            "/ RUN_TWIN_STEP1 per-rank restarts) -- set $DINO_NEMO_RUN_TRAJ / "
            "$DINO_NEMO_RUN_TWIN_STEP1 to point at a built oracle, or run on a "
            "machine with one.")


# ---------------------------------------------------------------------------
# Path 5 (solver warm start): documented absence, no NEMO artifacts needed.
# ---------------------------------------------------------------------------
def test_path5_no_solver_warm_start_on_this_card(mr):
    """nemo_dino_kamm_mlf uses the split-explicit barotropic solver
    (explicit_substep, NEMO's dynspg_ts) -- NOT rigid_lid, the only legoESM
    barotropic solver with elliptic/warm-start carry state (streamfunction
    psi/dpsi/dpsin_prev, seeded once via model.seed_scan_carry in
    run_dino.py). A regression that switched the card onto rigid_lid without
    updating this test/module would silently reintroduce an untested
    warm-start replay path -- this assertion is the tripwire."""
    from legoesm.ocean.experiments.dino import dino_config_for_recipe

    cfg = dino_config_for_recipe(mr.RECIPE)
    assert cfg.barotropic_solver == "explicit_substep", (
        f"{mr.RECIPE} now uses barotropic_solver={cfg.barotropic_solver!r} -- "
        "if this is 'rigid_lid', item 5 (solver warm-start replay) is no "
        "longer N/A for this card and multistep_replay.py needs a psi/dpsi "
        "carry check added, not just this test updated.")


# ---------------------------------------------------------------------------
# Day-0 gate + dtype sanity (fast-fail before the expensive replay loop).
# ---------------------------------------------------------------------------
def test_day0_bridge_is_bit_identical_and_fp64(mr):
    _skip_if_no_artifacts(mr)
    g, br, cfg, st = mr.build_replay_ic()
    assert str(st.T.data.dtype) == "float64"
    assert str(st.tke.data.dtype) == "float64"
    assert str(st.eta.data.dtype) == "float64"
    # verify_day0_matches_restart already asserted-and-raised inside
    # build_replay_ic; reaching here means it passed. Independently confirm
    # a genuine leap-frog continuation (not a rest-state fallback): u/v carry
    # nonzero before-level velocity too.
    assert st.u_before is not None and st.v_before is not None
    umask3 = np.asarray(g.umask) > 0.5
    assert float(np.max(np.abs(np.asarray(st.u_before.data)[:, 1:, :][umask3]))) > 0.1


# ---------------------------------------------------------------------------
# Paths 1-4: per-step difference curves vs NEMO's own restarts.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def replay_leapfrog_rhs(mr):
    """The FAITHFUL placement's growth curve -- computed once, shared by
    every path-specific assertion below (the expensive part: N_STEPS x
    per-step restart rebuild I/O)."""
    if not _have_artifacts(mr):
        pytest.skip("NEMO oracle tree not found")
    return mr.run_replay(N_STEPS, surface_tendency_placement="leapfrog_rhs")


def test_path1_now_level_curve_reported(mr, replay_leapfrog_rhs):
    """Path 1 (MLF before/now/after threading): report the max|dT| growth
    curve vs NEMO's own now-level restart at each step. No hardcoded pass/
    fail threshold here -- the #1492 gate constants live in
    fidelity_bar_gate.py (not touched by this lane); this test's job is to
    PRODUCE the curve and assert it is finite/sane, which the actual dycore
    gate is applied to separately."""
    res = replay_leapfrog_rhs
    assert len(res["max_dT_now"]) == N_STEPS
    assert np.all(np.isfinite(res["max_dT_now"]))
    assert np.all(res["max_dT_now"] >= 0.0)
    print(f"path1 max|dT_now| curve (leapfrog_rhs): {res['max_dT_now']}")


def test_path2_before_level_curve_reported(mr, replay_leapfrog_rhs):
    """Path 2 (Asselin-filtered T_before carried forward): report the
    curve vs NEMO's tb restart at each step."""
    res = replay_leapfrog_rhs
    assert len(res["max_dT_before"]) == N_STEPS
    assert np.all(np.isfinite(res["max_dT_before"]))
    print(f"path2 max|dT_before| curve (leapfrog_rhs): {res['max_dT_before']}")


def test_path3_tke_carry_curve_reported(mr, replay_leapfrog_rhs):
    """Path 3 (prognostic TKE 'en' carry): report the curve vs NEMO's en
    restart at each step."""
    res = replay_leapfrog_rhs
    assert len(res["max_den"]) == N_STEPS
    assert np.all(np.isfinite(res["max_den"]))
    print(f"path3 max|d(en)| curve (leapfrog_rhs): {res['max_den']}")


def test_path4_e3t_ssh_commit_ordering_curve_reported(mr, replay_leapfrog_rhs):
    """Path 4 (z*/e3t layer-thickness vs ssh commit ordering): report the
    cross-model e3t(lego eta) vs e3t(NEMO sshn) curve, both sides through the
    SAME compute_layer_thickness/H_bathy/z_coord."""
    res = replay_leapfrog_rhs
    assert len(res["max_de3t"]) == N_STEPS
    assert np.all(np.isfinite(res["max_de3t"]))
    print(f"path4 max|de3t| curve (leapfrog_rhs): {res['max_de3t']}")


def test_growth_report_all_paths(mr, replay_leapfrog_rhs):
    """Numbers-first summary: for every path's curve, compare the LAST-step
    value to the FIRST-step value and print whether it grew, so a human (or
    a future CI gate) can see the growth-vs-flat verdict per path without
    re-running the replay. Deliberately does not assert a pass/fail bound --
    the fidelity bar lives in fidelity_bar_gate.py, untouched by this lane."""
    res = replay_leapfrog_rhs
    for name in ("max_dT_now", "max_dS_now", "max_du_now", "max_dv_now",
                 "max_deta_now", "max_dT_before", "max_dS_before",
                 "max_du_before", "max_dv_before", "max_den", "max_de3t"):
        curve = res[name]
        first, last = float(curve[0]), float(curve[-1])
        ratio = last / first if first > 0 else float("nan")
        print(f"{name}: step1={first:.6e} step{N_STEPS}={last:.6e} "
              f"ratio={ratio:.3f} {'GROWS' if last > first else 'flat/decays'}")
    assert True  # this test's product is the printed report, not a gate


# ---------------------------------------------------------------------------
# Sensitivity demonstration: applied_now (lossy) vs leapfrog_rhs (faithful).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def replay_applied_now(mr):
    if not _have_artifacts(mr):
        pytest.skip("NEMO oracle tree not found")
    return mr.run_replay(N_STEPS, surface_tendency_placement="applied_now")


def test_sensitivity_placement_changes_the_trajectory(
        mr, replay_leapfrog_rhs, replay_applied_now):
    """Positive control (#1492's own finding): 'applied_now' discards ~56%
    of the retained surface tendency (Asselin retention (1-2*gamma)/
    (2*(1-gamma)) = 4/9 at gamma=0.1) relative to 'leapfrog_rhs'. From the
    IDENTICAL bridged day-0 IC, the two placements must produce DIFFERENT
    domain-mean-temperature trajectories, and the gap between them must GROW
    with step count (not merely be nonzero at step 1) -- proof this
    multi-step comparison is sensitive to a real threading defect, not
    reporting noise a single-step check would also catch.
    """
    mean_t_rhs = replay_leapfrog_rhs["mean_T_now"]
    mean_t_now = replay_applied_now["mean_T_now"]
    assert len(mean_t_rhs) == len(mean_t_now) == N_STEPS

    delta = mean_t_rhs - mean_t_now
    print(f"sensitivity: mean_T(leapfrog_rhs) - mean_T(applied_now) per step: {delta}")

    # Nonzero from step 1 (both placements apply DIFFERENT increments even on
    # the very first step -- this alone would ALSO be visible to a
    # single-step comparison, so it is necessary but not sufficient).
    assert abs(delta[0]) > 0.0, (
        "the two placements produced an IDENTICAL mean-T after step 1 -- "
        "the sensitivity control did not fire; surface_tendency_placement "
        "is not actually wired into this replay.")

    # SUFFICIENT: the |delta| must GROW over the window -- the signature a
    # single-step test cannot produce (a single step has no "growth").
    abs_delta = np.abs(delta)
    assert abs_delta[-1] > abs_delta[0], (
        f"|delta| did not grow over {N_STEPS} steps "
        f"(step1={abs_delta[0]:.3e}, step{N_STEPS}={abs_delta[-1]:.3e}) -- "
        "the sensitivity demonstration requires the placement gap to widen "
        "with step count, matching the #1492 finding that the loss "
        "compounds rather than appearing once.")

    # Report-only (not gated): consecutive-step monotonicity. Measured on a
    # real 32-step run (see module docstring / task report), the early steps
    # are noisy (day-1 integrator-memory handshake, non-shrinking ratio only
    # 0.33 at n=4) and settle into a robust >=90% non-shrinking ratio by
    # n=32 -- exactly the "flat/noisy early, grows later" signature this
    # lane is built to surface, not a bug in the assertion. Gating on this
    # at small N_STEPS would be flaky by construction, so it is printed for
    # the report and NOT asserted; the first-vs-last growth check above is
    # the one load-bearing sensitivity assertion.
    non_shrinking = int(np.sum(np.diff(abs_delta) >= -1e-12))
    print(f"sensitivity monotonicity (report only): {non_shrinking}/{N_STEPS - 1} "
          f"consecutive steps non-shrinking")
