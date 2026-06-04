"""End-to-end smoke test for the plane CRM production stack.

Mirrors the hydrostatic ``test_rce_cross_grid_smoke.py`` regression
for the plane CRM (non-hydrostatic). Runs ``run_rce_mpi_long.py``
on a tiny 12x12x20 mesh for a short window with the F8/F10/iter-13
production-default config (clean Wing IC, no bubble, no qv noise,
hyperdiff=5e6, dt=5 s) and asserts:

1. The driver exits cleanly with finite diagnostics.
2. ``max|w|`` stays below 0.5 m/s (production envelope at F8 IC is
   ~6e-3 m/s; >0.5 m/s in a short smoke is a smoking gun for an
   in-flight numerical instability).
3. ``MSE`` drift stays below 1e-3 relative (production envelope is
   ~7e-5 over 28 min sim at 132x132; the same per-step drift on a
   12x12 mesh extrapolates well below 1e-3 over the smoke window).
4. ``CWV`` doesn't drift more than 0.1 mm from IC (49.78 mm for
   the 12x12@nlev=20 smoke; 49.9413 mm for the 132x132@nlev=30
   production-scale slow tests after the iter-95 hydrostatic-BC
   fix; legacy bug values were 55.001 / 55.550 mm) — the F8 IC is
   dynamically frozen until surface flux + radiation warm the
   column on hour-day timescales.

Wall budget: ~30 s on M5 Pro (rough estimate). Single-rank, no MPI,
no JAX JIT pre-compile.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER = REPO_ROOT / "scripts" / "run" / "run_rce_mpi_long.py"


def _run_driver(output_dir):
    """Invoke run_rce_mpi_long.py with the F8/F10 production defaults.

    iter-41: switched the dycore-only path from
    ``--rad-call-interval-s 1e9`` (Codex iter-39 HIGH#2: fires once at
    step 1 + caches) to the real ``--no-radiation`` flag. The smoke
    now truly excludes radiation.
    """
    env = os.environ.copy()
    # FORCE JAX_PLATFORMS=cpu (override any exported value). iter-7
    # found that JAX_PLATFORMS=metal triggers MLIR legalisation
    # crashes on spectral / voronoi / latlon-cgrid paths; the same
    # backend can also break the plane CRM in subtle ways. Tests must
    # always run on CPU regardless of the developer's shell env.
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", "0.005",        # ~7 min sim = 86 outer steps
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", "0.1",
        "--n-acoustic-substeps", "12",
        "--advection", "upwind1",
        "--hyperdiff", "5e6",
        "--bubble-theta-pert", "0.0",
        "--qv-noise-amp", "0.0",
        "--log-every-steps", "20",
        "--n-physics-substeps", "1",
        "--no-radiation",
        "--output", str(output_dir),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=300,
    )
    return result


# iter-84: ``_read_log`` moved to ``_plane_crm_helpers.py`` (mirror
# of iter-79 ``_parse_rad_call_count`` extraction) so the parser
# can be unit-tested directly. All 4 use sites in this file now
# import via the helpers module.


# iter-79: ``_parse_rad_call_count`` moved to sibling
# ``_plane_crm_helpers.py`` so the iter-53/54 unit-test file can
# import it without triggering pytest collection of THIS file as
# an import side effect. Mirrors iter-78 hydrostatic refactor.
from tests.atmosphere.nonhydrostatic.integration._plane_crm_helpers import (
    _parse_rad_call_count,
    _read_log,
)


def test_plane_crm_short_smoke_clean_ic(tmp_path):
    """Plane CRM with F8 clean IC + F10 dt=5 s + iter-13 hardened
    defaults must complete a 7-minute smoke window with all
    diagnostics in the production envelope.

    iter-41: now uses ``--no-radiation`` (was previously
    ``--rad-call-interval-s 1e9`` which Codex iter-39 HIGH#2 showed
    actually fires radiation once at step 1 + caches the tendency
    for the full run). The dycore-only contract is now genuine.
    """
    out_dir = tmp_path / "rce_plane_smoke"
    result = _run_driver(out_dir)
    if result.returncode != 0:
        pytest.fail(
            "run_rce_mpi_long.py exited nonzero "
            f"({result.returncode})\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-1000:]}"
        )
    rows = _read_log(out_dir)
    assert rows, "log.txt produced no diagnostic rows"

    # iter-41 dycore-only contract: --no-radiation must yield
    # rad_calls=0. If a future regression strips the --no-radiation
    # gate or drops the rad_calls counter from the Done. line, this
    # fires before the envelope checks below.
    rad_calls = _parse_rad_call_count(result.stdout)
    assert rad_calls == 0, (
        f"plane CRM dycore-only smoke: driver reported "
        f"rad_calls={rad_calls!r}, expected 0 under --no-radiation. "
        f"Either the --no-radiation gate regressed or the rad_calls "
        f"counter is missing from the Done. line.\n"
        f"stdout tail:\n{result.stdout[-500:]}"
    )

    # iter-41 anchor re-verification (Codex iter-41 HIGH fix): the
    # numerical anchors below were originally measured with
    # ``--rad-call-interval-s 1e9`` (one cached radiation tendency
    # held for 86 steps). iter-95 also shifted the IC CWV by ~5.5 mm
    # downward (legacy top-down hydrostatic BC with T_avg=250 K vs
    # iter-95 p_sfc=101480 bottom-up BC). Current 12x12 / no-rad /
    # no-mass-fixer measurements at this dt/dz are:
    #   step  1: CWV=49.78 mm, MSE=~3.52e9, max|w|=0.0
    #   step 80: CWV=49.78 mm, max|w|<1e-3 m/s
    # Verified iter-95h run.

    max_w_final = float(rows[-1]["max|w|"])
    cwv_first = float(rows[0]["CWV_mean"])
    cwv_final = float(rows[-1]["CWV_mean"])

    # Anchor the IC CWV: the 12x12@nlev=20 Wing 2018 IC carries
    # 49.78 mm at this nlev/H after the iter-95 hydrostatic-BC fix.
    # (Was 49.78 mm when pinned at #315, where the driver called
    # make_wing2018_theta_ref_fn(T_sfc=300) against the old actual-
    # temperature parameterization. That surface-temp parameter was
    # since renamed T_sfc -> T_v0 = surface VIRTUAL temperature, so the
    # driver now passes T_v0=300 -> a slightly cooler actual surface T
    # and +0.31 mm CWV. The driver crashed continuously between that
    # rename and the kwarg fix, so this sentinel was never re-validated;
    # 49.78 mm is the correct IC under the current, more physical T_v0
    # parameterization.) If a future commit silently shifts the Wing
    # profile coefficients or reverts the BC, the drift assertion below
    # would still pass against the new IC and miss the regression — this
    # gate makes the IC itself part of the contract. Tolerance 0.01 mm
    # (~2e-4 relative).
    assert abs(cwv_first - 49.78) < 0.01, (
        f"plane CRM smoke: IC CWV={cwv_first:.4f} mm != 49.78 ± 0.01. "
        f"The Wing 2018 reference profile or its area weighting "
        f"changed — update this test's expected value if intentional, "
        f"otherwise diagnose the regression."
    )
    mse_first = float(rows[0]["MSE_mean"])
    mse_final = float(rows[-1]["MSE_mean"])

    assert max_w_final < 0.5, (
        f"plane CRM smoke: max|w|={max_w_final:.3e} m/s at end "
        f"exceeds 0.5 m/s production envelope (F8 measured ~6e-3 m/s "
        f"on 132x132). Check CRM_implementation.md F8/F10."
    )
    # CWV is measured against its OWN IC (not a hardcoded constant);
    # the small-grid IC carries a slightly different mean than the
    # 132x132 production grid (49.78 mm at 12x12@nlev=20 vs
    # 49.9413 mm at 132x132@nlev=30 after the iter-95 BC fix; the
    # diff is driven by vertical-grid spacing, not horizontal).
    cwv_drift = abs(cwv_final - cwv_first)
    assert cwv_drift < 0.1, (
        f"plane CRM smoke: CWV drifted {cwv_drift:.4f} mm in the "
        f"smoke window (IC={cwv_first:.4f}, final={cwv_final:.4f}). "
        f"F8 says CWV stays pinned at IC until convection spins up "
        f"on hour-day timescale; >0.1 mm drift in 86 outer steps "
        f"means radiation is unexpectedly active or the moist-mass "
        f"fixer regressed."
    )
    rel_mse_drift = abs(mse_final - mse_first) / mse_first
    assert rel_mse_drift < 1e-3, (
        f"plane CRM smoke: MSE drift {rel_mse_drift:.3e} relative "
        f"exceeds 1e-3 cap. F8/F10 production measurements: 7e-5 "
        f"relative over 28 min sim at 132x132."
    )


def _run_driver_with_radiation(output_dir):
    """Variant of _run_driver that ENABLES gray radiation. Catches a
    regression where radiation is broken in a way the dycore-only
    smoke would miss (e.g. NaN in the column-water-vapor reduction
    used by gray rad, or a crash in the first radiation tendency
    application)."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", "0.002",       # ~3 min sim = ~35 outer steps
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", "0.1",
        "--n-acoustic-substeps", "12",
        "--advection", "upwind1",
        "--hyperdiff", "5e6",
        "--bubble-theta-pert", "0.0",
        "--qv-noise-amp", "0.0",
        "--log-every-steps", "10",
        "--n-physics-substeps", "1",
        "--rad-call-interval-s", "30.0",  # fire radiation every 30s
        "--output", str(output_dir),
    ]
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=300,
    )


def test_plane_crm_short_smoke_with_radiation(tmp_path):
    """Plane CRM with gray radiation actively called every 30 s sim
    time. Codex iter-16 gap: the main dycore-only smoke disables
    radiation via --rad-call-interval-s=1e9, so a radiation
    regression (NaN in CWV reduction, broken first tendency
    application, etc.) would slip through CI.

    This shorter smoke exercises the radiation path with a tight
    call interval. Only asserts the driver exits cleanly + ``max|w|``
    stays bounded — radiation can drive larger drift than the
    dycore-only case over the same window, so the CWV/MSE caps
    above are not appropriate here.

    iter-41: also asserts ``rad_calls > 0`` so a silently-disabled
    radiation path on the 12x12 smoke fails immediately. With
    days=0.002 → 34 outer steps and rad-interval=30s/dt=5s
    (every=6), the tick fires at steps 1, 7, 13, 19, 25, 31 = 6
    calls. The lower-bound assertion (>0) keeps the test robust to
    minor day/dt fiddling; the schedule arithmetic itself is
    pinned by the iter-40 production-scale tests.
    """
    out_dir = tmp_path / "rce_plane_smoke_rad"
    result = _run_driver_with_radiation(out_dir)
    if result.returncode != 0:
        pytest.fail(
            "run_rce_mpi_long.py exited nonzero with radiation enabled "
            f"({result.returncode})\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-1000:]}"
        )
    # iter-41 hardening: radiation tick must fire the derived
    # expected number of times. Codex iter-41 MEDIUM fix: tighten
    # from rad_calls > 0 (a broken cadence pinning rad_calls=1
    # would still pass) to an exact count derived from CLI args.
    # days=0.002 → total_steps=34; rad-interval=30, dt=5 → every=6;
    # expected = 6 (fires at steps 1, 7, 13, 19, 25, 31).
    # iter-42: use the production helper instead of re-deriving the
    # arithmetic locally — keeps test contract anchored to the same
    # source the driver uses.
    from legoesm.driver.physics_schedule import radiation_call_schedule
    dt_s = 5.0
    rad_interval_s = 30.0
    days = 0.002
    total_steps = int(days * 86400.0 / dt_s)
    schedule = radiation_call_schedule(rad_interval_s, dt_s, total_steps)
    rad_calls = _parse_rad_call_count(result.stdout)
    assert rad_calls == schedule.num_calls, (
        f"plane CRM radiation smoke: driver reported "
        f"rad_calls={rad_calls!r}, expected {schedule.num_calls} "
        f"(dt={dt_s}, rad-interval={rad_interval_s}, "
        f"total_steps={total_steps} → every={schedule.every_steps}, "
        f"fires at {schedule.fire_step_indices}). Either the "
        f"radiation tick gate regressed or the rad_calls counter "
        f"is missing.\n"
        f"stdout tail:\n{result.stdout[-500:]}"
    )
    rows = _read_log(out_dir)
    assert rows, "log.txt produced no diagnostic rows with radiation on"
    max_w_final = float(rows[-1]["max|w|"])
    assert max_w_final < 0.5, (
        f"plane CRM smoke with radiation: max|w|={max_w_final:.3e} "
        f"m/s at end exceeds 0.5 m/s production envelope. Possibly a "
        f"radiation tendency injecting too much energy too quickly — "
        f"check make_radiation_physics output."
    )


def _run_driver_no_mass_fixer(output_dir):
    """iter-95k: variant that disables fix_moist_mass_plane via
    ``--no-mass-fixer``. Surface flux is allowed to NET ADD moisture,
    so CWV should GROW from the IC over the smoke window. The default
    test (``test_plane_crm_short_smoke_clean_ic``) verifies the
    opposite: with the fixer ON, CWV stays pinned. Together the two
    tests pin both halves of iter-95b's Bug 2 fix.

    Window: ~7 min sim = 86 outer steps on the 12x12 mesh — enough
    time for surface flux to inject ~mm-level moisture.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", "0.005",
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", "0.1",
        "--n-acoustic-substeps", "12",
        "--advection", "upwind1",
        "--hyperdiff", "5e6",
        "--bubble-theta-pert", "0.0",
        "--qv-noise-amp", "0.0",
        "--log-every-steps", "20",
        "--n-physics-substeps", "1",
        "--no-radiation",
        "--no-mass-fixer",
        "--output", str(output_dir),
    ]
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=300,
    )


def test_plane_crm_no_mass_fixer_lets_cwv_grow(tmp_path):
    """iter-95k: pin the behavioural half of iter-95b's Bug 2 fix.

    With ``--no-mass-fixer`` set, fix_moist_mass_plane is NOT called
    after surface flux deposits q_v in the lowest model level. CWV
    must therefore GROW from the IC (49.78 mm at 12x12@nlev=20).
    The matching default-config test pins the inverse: with the
    fixer ON, CWV stays pinned within 0.1 mm.

    Failure modes this catches:
    * Someone removes the gate ``if not args.no_mass_fixer:`` (CWV
      would pin again and this test fails).
    * Someone flips the argparse default to True (the default
      test would start failing because CWV would grow there too).
    * Surface flux scheme stops actually depositing moisture
      (CWV would not grow with the flag set).
    """
    out_dir = tmp_path / "rce_plane_smoke_no_mass_fixer"
    result = _run_driver_no_mass_fixer(out_dir)
    if result.returncode != 0:
        pytest.fail(
            f"run_rce_mpi_long.py --no-mass-fixer exited "
            f"{result.returncode}\n"
            f"stdout tail:\n{result.stdout[-1000:]}\n"
            f"stderr tail:\n{result.stderr[-500:]}"
        )
    rows = _read_log(out_dir)
    assert rows, "log.txt produced no diagnostic rows with --no-mass-fixer"
    cwv_first = float(rows[0]["CWV_mean"])
    cwv_final = float(rows[-1]["CWV_mean"])
    # IC anchor (same as the default-config smoke): iter-95 12x12 = 49.78.
    assert abs(cwv_first - 49.78) < 0.01, (
        f"plane CRM --no-mass-fixer smoke: IC CWV={cwv_first:.4f} mm "
        f"!= 49.78 ± 0.01. Either the Wing IC drifted or the "
        f"iter-95 hydrostatic-BC fix regressed."
    )
    # Behavioural check: CWV must grow. The default-config test
    # asserts drift < 0.1 mm; here we assert drift > 0.01 mm (a
    # solid margin above the noise floor in 86 outer steps and
    # above the default-config drift limit at <1e-4). Measured
    # growth in this 86-step / dt=5s / 12x12 window is ~0.026 mm;
    # the 0.01 mm gate has a 2.5x cushion. On the iter-95 v3 run
    # at 32x32 / dt=10s CWV grew ~0.4 mm in the first 7 minutes —
    # consistent with this measurement.
    cwv_growth = cwv_final - cwv_first
    assert cwv_growth > 0.01, (
        f"plane CRM --no-mass-fixer smoke: CWV grew only "
        f"{cwv_growth:.4f} mm (IC={cwv_first:.4f}, "
        f"final={cwv_final:.4f}). Expected > 0.01 mm growth — "
        f"either --no-mass-fixer regressed (fixer still rescaling "
        f"back to IC), surface flux scheme stopped depositing "
        f"moisture, or both."
    )


def _run_driver_production_scale(output_dir, *, n_outer_steps=60,
                                 log_every_steps=15,
                                 rad_call_interval_s=None,
                                 timeout_s=900,
                                 dt=5.0,
                                 advection="upwind1",
                                 acoustic_off_centering=0.1,
                                 no_mass_fixer=False):
    """Invoke run_rce_mpi_long.py at the iter-14 production scale
    (132x132x30 dx=2km dt=5s) for ``n_outer_steps`` outer steps.

    Defaults reproduce the iter-14/iter-15 measurement that locked
    in the plane CRM production envelope: clean Wing IC (no bubble,
    no qv noise), hyperdiff=5e6, Smag c_s=0.2 (passed explicitly so
    a future driver default change can't silently shift this
    regression), SI acoustic with off-centering=0.1 + 12 substeps,
    mass fixer on (driver default).

    ``dt``, ``advection``, ``acoustic_off_centering`` (iter-230 fix):
    parameterised so the iter-183 production contract (dt=20,
    van_leer, beta=0.2) gets its own regression envelope without
    a fork of this helper. iter-14 envelope: dt=5/upwind1/0.1;
    iter-183 envelope: dt=20/van_leer/0.2.

    ``rad_call_interval_s=None`` (default) passes ``--no-radiation``
    to the driver — radiation is FULLY skipped (cached_rad_tend
    stays None). Codex iter-39 review pointed out that the prior
    ``--rad-call-interval-s 1e9`` still fired one radiation call at
    step 1 and cached it for the rest of the run, so "dycore-only"
    was a misnomer. The iter-39 fix wires a real disable flag.
    Pass a positive float (e.g. 60.0) to exercise the radiation
    path with refreshes every N seconds of sim time.

    ``--days`` carries +0.5*dt padding so ``int(total_t / dt)`` in
    the driver always lands at ``n_outer_steps`` exactly (without
    padding, float-roundoff truncates 60 → 59 silently, leaving
    ``rows[-1]`` on step 40 instead of step 60 when
    ``log_every_steps=20``).  ``log_every_steps`` defaults to 15 so
    steps 1, 15, 30, 45, 60 all log — transient spikes between
    log_every=20 rows would otherwise hide a CFL crash that
    self-recovers in 19 steps.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    days = (n_outer_steps + 0.5) * dt / 86400.0
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "132", "--ny", "132", "--nlev", "30",
        "--dx", "2000.0", "--dt", repr(float(dt)),
        "--days", f"{days:.8f}",
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", repr(float(acoustic_off_centering)),
        "--n-acoustic-substeps", "12",
        "--advection", str(advection),
        "--hyperdiff", "5e6",
        "--smag-cs", "0.2",
        "--bubble-theta-pert", "0.0",
        "--qv-noise-amp", "0.0",
        "--log-every-steps", str(log_every_steps),
        "--n-physics-substeps", "1",
        "--output", str(output_dir),
    ]
    if rad_call_interval_s is None:
        cmd.append("--no-radiation")
    else:
        cmd.extend(["--rad-call-interval-s",
                    repr(float(rad_call_interval_s))])
    if no_mass_fixer:
        cmd.append("--no-mass-fixer")
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout_s,
    )


@pytest.mark.slow
def test_plane_crm_production_scale_132x132_envelope(tmp_path):
    """Nightly slow regression for the plane CRM 132x132 production
    config (5-min-sim sub-envelope of the iter-14 1-sim-hour smoke,
    truly dycore-only — radiation fully disabled via --no-radiation).

    iter-14 measured (725 steps = 1 sim-hour, 132x132x30 dx=2km dt=5s,
    clean Wing IC, mass fixer + Smag c_s=0.2 + SI acoustic + hyperdiff
    5e6, ``--rad-call-interval-s 1e9``):

    | step | CWV [mm]   | MSE [J/kg] | max|w| [m/s] |
    |------|------------|------------|--------------|
    |   1  |  49.9413   |  3.52e9    |  0.0e+00     |
    | 100  |  49.9413   |  3.52e9    |  3.9e-3      |
    | 300  |  49.9413   |  3.52e9    |  5.5e-3      |
    | 700  |  49.9413   |  3.52e9    |  6.1e-3      |
    (iter-95 IC anchor; pre-iter-95 the legacy top-down BC bug
    inflated this to ~55.550 mm / ~4.21e9 — see CRM_implementation.md)

    Codex iter-39 review pointed out that ``--rad-call-interval-s
    1e9`` still fires one radiation call at step 1 and caches the
    tendency for the entire run — so iter-14 / iter-38 first-cut
    was actually "dycore + 1 cached radiation tendency", not pure
    dycore. iter-39 added a real ``--no-radiation`` flag to the
    driver and this test now uses it, so the iter-38 envelope is
    now genuinely radiation-free. The expected MSE drift drops vs
    the original iter-38 cached-radiation measurement; we hold the
    same 5e-5 cap because it remains a valid UPPER bound (caches
    the per-step dycore/fast-physics drift rate, not the radiation
    contribution).

    The matching ``test_plane_crm_production_scale_132x132_with_radiation``
    (iter-39) exercises radiation refreshing every 60 s — its
    MSE-drift FLOOR assertion (>5e-7 relative) catches a
    silently-disabled radiation path that this --no-radiation
    dycore-only test would not.

    Wall budget ~2.5 min on M5 Pro (1.35 s/step × 60 + JIT compile).
    Marked ``slow``; runs via ``pytest -m slow``. Default skips it.
    """
    n_steps = 60
    log_every = 15
    out_dir = tmp_path / "rce_plane_prod"
    result = _run_driver_production_scale(
        out_dir, n_outer_steps=n_steps, log_every_steps=log_every,
    )
    if result.returncode != 0:
        pytest.fail(
            "run_rce_mpi_long.py exited nonzero at production scale "
            f"({result.returncode})\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-1000:]}"
        )
    rows = _read_log(out_dir)
    assert rows, "log.txt produced no diagnostic rows at production scale"

    # iter-40 hardening: --no-radiation must produce rad_calls=0
    # on the driver's "Done." line. A regression where the
    # _maybe_fire_radiation gate degrades silently (e.g. the
    # ``args.no_radiation`` early return is removed) would
    # otherwise be invisible because the MSE-drift cap on this
    # test is an UPPER bound only.
    rad_calls = _parse_rad_call_count(result.stdout)
    assert rad_calls == 0, (
        f"plane CRM production smoke (--no-radiation): driver "
        f"reported rad_calls={rad_calls!r}, expected 0. Either the "
        f"--no-radiation gate regressed or the rad_calls= counter "
        f"is missing from the Done. line. "
        f"stdout tail:\n{result.stdout[-500:]}"
    )

    # Schema + step-count sanity: log_every=15 + n=60 yields exact rows
    # at steps 1, 15, 30, 45, 60. A short run (driver miscount) would
    # leave the final row at an earlier step and all envelope asserts
    # would silently evaluate the wrong state.
    expected_logged_steps = [1, 15, 30, 45, 60]
    logged_steps = [int(r["step"]) for r in rows]
    assert logged_steps == expected_logged_steps, (
        f"plane CRM production smoke: logged steps={logged_steps} "
        f"!= expected {expected_logged_steps}. Driver step counter or "
        f"log-every cadence regressed."
    )

    cwv_first = float(rows[0]["CWV_mean"])
    cwv_final = float(rows[-1]["CWV_mean"])
    mse_first = float(rows[0]["MSE_mean"])
    mse_final = float(rows[-1]["MSE_mean"])

    # iter-95 measured IC CWV = 49.9413 mm at 132x132@nlev=30 Wing
    # 2018 (after the hydrostatic-BC fix; pre-iter-95 the legacy
    # T_avg=250 K top-down BC inflated this to 55.550 mm).
    assert abs(cwv_first - 49.9413) < 0.01, (
        f"plane CRM production smoke: IC CWV={cwv_first:.4f} mm "
        f"!= 49.9413 ± 0.01. iter-95 anchored this value at "
        f"132x132@nlev=30 post hydrostatic-BC fix. If intentional, "
        f"update the test."
    )
    # max|w| envelope on EVERY logged row, not just the last — a
    # transient CFL spike that self-damps in <log_every steps would
    # otherwise pass silently. iter-14 saw monotone w growth up to
    # 6.1e-3 at step 700; at step 60 actual was 2.08e-3. Cap at
    # 0.05 m/s is 24x over that — catches any genuine in-flight CFL
    # crash without false positives from spinup noise.
    max_w_per_row = [float(r["max|w|"]) for r in rows]
    for step_idx, mw in zip(logged_steps, max_w_per_row):
        assert mw < 0.05, (
            f"plane CRM production smoke: max|w|={mw:.3e} m/s at step "
            f"{step_idx} exceeds 0.05 m/s safety cap. iter-14 measured "
            f"~6e-3 m/s peak across step 1..700; this is a smoking gun "
            f"for an in-flight CFL crash or stabilizer regression."
        )
    # Activity floor: by step 15 the SI acoustic substeps should have
    # produced non-zero max|w|. A dead/no-op simulation (e.g. step
    # function elided by a future refactor, frozen state pytree, JIT
    # eating the loop body) would pass every envelope above. Anchor
    # at step 15 because step 1 is logged BEFORE any time advance.
    step15_max_w = max_w_per_row[1]
    assert step15_max_w > 1e-6, (
        f"plane CRM production smoke: max|w|={step15_max_w:.3e} m/s "
        f"at step 15 — the dycore appears inactive. iter-14 saw "
        f"~2e-3 m/s by step 20 from baseline thermo gradient + SI "
        f"acoustic. <1e-6 means step() is a no-op."
    )
    # CWV pinned in iter-14 within rounding; 0.01 mm drift cap.
    cwv_drift = abs(cwv_final - cwv_first)
    assert cwv_drift < 0.01, (
        f"plane CRM production smoke: CWV drifted {cwv_drift:.4f} mm "
        f"(IC={cwv_first:.4f}, final={cwv_final:.4f}) in {n_steps} "
        f"steps. iter-14 saw zero drift at 132x132 over 700 steps. "
        f"Possibly the moist-mass fixer regressed."
    )
    # iter-14 saw 1.7e-4 relative drift over 725 steps WITH one cached
    # radiation tendency (Codex iter-39 review: ``--rad-call-interval-s
    # 1e9`` still fires once at step 1 + caches). iter-39 fix switches
    # this test to ``--no-radiation`` so the drift now reflects dycore
    # + fast physics only — likely smaller per-step rate. The 5e-5 cap
    # remains a valid UPPER bound and still catches a 3x per-step
    # regression of the dycore/fast-physics drift rate; the original
    # iter-14 reference is now a CEILING (the true dycore-only rate
    # must be < the cached-radiation rate, since radiation cools).
    rel_mse_drift = abs(mse_final - mse_first) / mse_first
    assert rel_mse_drift < 5e-5, (
        f"plane CRM production smoke (truly no-radiation): MSE drift "
        f"{rel_mse_drift:.3e} relative exceeds 5e-5 sub-envelope cap. "
        f"iter-14 measured 1.7e-4 over 725 steps WITH 1 cached "
        f"radiation tendency (per-step ≈ 2.3e-7); pure dycore + fast "
        f"physics at 60 steps should land BELOW that. Investigate "
        f"dycore energy budget if exceeded."
    )


@pytest.mark.slow
def test_plane_crm_iter183_production_scale_132x132_envelope(tmp_path):
    """iter-230: nightly slow regression for the iter-183 plane CRM
    production contract (dt=20 s + van_leer + beta=0.2 + Smag c_s=0.2
    + hyperdiff 5e6 + NO mass fixer, no radiation). The
    --no-mass-fixer flag matches the 30-day wrapper default
    (run_rce_30day.sh: NO_MASS_FIXER=${NO_MASS_FIXER:-1}); without
    that the test would silently pin CWV at IC via the fixer and
    miss a surface-flux regression. The iter-14 envelope above
    keeps the fixer ON because the iter-14 contract relies on it.

    iter-229 verified the iter-183 contract at full 30-day scale:
    DOD PASS with log max|w|=1.06e-02 m/s, CWV evolution
    49.94→53.33 mm (Wing 2018 plateau), MSE drift -1.6%. This test
    runs the SAME wrapper-equivalent configuration for 60 outer
    steps (= 20 sim-min) and acts as an early-stability +
    config-fingerprint smoke. It catches CFL crashes, dycore
    no-ops, and silent kwarg fallbacks in any of
    {dt=20, van_leer, beta=0.2, no-mass-fixer, n_acoustic=12,
    hyperdiff=5e6}. It does NOT cover late-time effects
    (precipitation cycles starting day 16, slow MSE drift, late
    convective amplification) — those remain covered by the
    30-day production runs themselves, not by this smoke.

    Empirical measurement (iter-230 smoke, dt=20, n=60, NMF):

    | step | day       | CWV_mean[mm] | MSE_mean[J/m²] | max|w|[m/s] | Ca |
    |------|-----------|--------------|----------------|-------------|----|
    |  1   | 0.000231  | 49.942       | 3.5247e9       | 0.0e+00     |0.87|
    | 15   | 0.003472  | 49.958       | 3.5247e9       | 3.81e-04    |0.87|
    | 30   | 0.006944  | 49.975       | 3.5248e9       | 5.41e-04    |0.87|
    | 45   | 0.010417  | 49.992       | 3.5248e9       | 6.22e-04    |0.87|
    | 60   | 0.013889  | 50.008       | 3.5249e9       | 6.68e-04    |0.87|

    Bounds + dt fingerprints (each catches a different regression):
    * max|w| < 0.05 m/s on every logged row — generic CFL safety net.
    * activity floor max|w| > 1e-6 m/s at step 15 — catches no-op step().
    * |CWV drift| < 0.15 mm (2.3× measured 0.066 mm) — catches a
      surface-flux regression OR an accidental --mass-fixer flip
      (fixer would pin near zero).
    * |MSE drift| < 1.5e-4 relative (2.6× measured 5.7e-5) — catches
      energy-budget drift; tighter than iter-39 with-rad envelope
      because no radiation cooling source.
    * Final logged day ≈ 60·20/86400 = 0.01389 (rejects a dt=5
      fallback at 4× too-low day count).
    * Ca_substep > 0.5 — fingerprints dt=20 (iter-14 dt=5 gives
      Ca_substep ≈ 0.22; iter-183 dt=20 gives ≈ 0.87). A silent
      kwarg-default fallback to dt=5 would fail this.

    Wall budget ~30 s on M5 Pro cached JIT, ~1.5 min cold (4× fewer
    steps than the iter-14 envelope's dt=5 equivalent for the same
    20-min sim window). Marked ``slow``.
    """
    n_steps = 60
    log_every = 15
    dt = 20.0
    out_dir = tmp_path / "rce_plane_iter183_prod"
    result = _run_driver_production_scale(
        out_dir,
        n_outer_steps=n_steps,
        log_every_steps=log_every,
        dt=dt,
        advection="van_leer",
        acoustic_off_centering=0.2,
        no_mass_fixer=True,
    )
    if result.returncode != 0:
        pytest.fail(
            "run_rce_mpi_long.py exited nonzero at iter-183 "
            f"production scale ({result.returncode})\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-1000:]}"
        )
    rows = _read_log(out_dir)
    assert rows, (
        "log.txt produced no diagnostic rows at iter-183 production "
        "scale"
    )

    rad_calls = _parse_rad_call_count(result.stdout)
    assert rad_calls == 0, (
        f"plane CRM iter-183 envelope (--no-radiation): driver "
        f"reported rad_calls={rad_calls!r}, expected 0."
    )

    expected_logged_steps = [1, 15, 30, 45, 60]
    logged_steps = [int(r["step"]) for r in rows]
    assert logged_steps == expected_logged_steps, (
        f"plane CRM iter-183 envelope: logged steps={logged_steps} "
        f"!= expected {expected_logged_steps}."
    )

    # iter-231 (Codex round-2 MEDIUM#1) + iter-232 (Codex round-3
    # MEDIUM): exact-token fingerprint of advection + beta +
    # mass-fixer + n_acoustic + hyperdiff from the driver's
    # ``# config:`` log header line. Round-2 used ``in
    # config_line`` substring matching which would pass
    # ``acoustic_off_centering=0.25`` as containing
    # ``acoustic_off_centering=0.2`` and ``n_acoustic_substeps=120``
    # as containing ``n_acoustic_substeps=12``. Round-3 fix: parse
    # the line into an exact ``key=value`` dict and compare typed
    # values (float for off-centering / hyperdiff, int for
    # n_acoustic_substeps, str for advection / on/off flags).
    log_path = out_dir / "log.txt"
    config_line = None
    with open(log_path) as fh:
        for line in fh:
            if line.startswith("# config:"):
                config_line = line.strip()
                break
    assert config_line is not None, (
        "plane CRM iter-183 envelope: driver log.txt is missing the "
        "iter-231 ``# config:`` line — either the driver regressed "
        "(removed the header) or the log path changed."
    )
    # Parse ``# config: k1=v1 k2=v2 ...`` into a dict. Tokens are
    # space-separated after the leading ``# config:`` marker; each
    # token is a single ``key=value`` (no spaces in values for the
    # current driver — driver writes floats/ints/strings only).
    config_body = config_line[len("# config:"):].strip()
    config_kv: dict[str, str] = {}
    for tok in config_body.split():
        if "=" not in tok:
            pytest.fail(
                f"plane CRM iter-183 envelope: malformed ``# config:`` "
                f"token {tok!r} (no ``=``). Full line: {config_line!r}"
            )
        k, v = tok.split("=", 1)
        config_kv[k] = v
    # Required exact matches (typed where applicable). Each line
    # asserts ONE contract element so a regression message tells
    # the dev which kwarg flipped, not just "fingerprint mismatch".
    assert config_kv.get("advection") == "van_leer", (
        f"plane CRM iter-183 envelope: advection="
        f"{config_kv.get('advection')!r}, expected 'van_leer'. "
        f"Silent helper-kwarg fallback to upwind1?"
    )
    assert float(config_kv.get("acoustic_off_centering", "nan")) == 0.2, (
        f"plane CRM iter-183 envelope: acoustic_off_centering="
        f"{config_kv.get('acoustic_off_centering')!r}, expected '0.2'. "
        f"iter-183's beta=0.2 contract regressed."
    )
    assert config_kv.get("mass_fixer") == "off", (
        f"plane CRM iter-183 envelope: mass_fixer="
        f"{config_kv.get('mass_fixer')!r}, expected 'off'. The "
        f"--no-mass-fixer flag silently flipped — CWV growth check "
        f"is now meaningless."
    )
    assert config_kv.get("si_acoustic") == "on", (
        f"plane CRM iter-183 envelope: si_acoustic="
        f"{config_kv.get('si_acoustic')!r}, expected 'on'. The "
        f"--semi-implicit-acoustic flag regressed."
    )
    assert int(config_kv.get("n_acoustic_substeps", "0")) == 12, (
        f"plane CRM iter-183 envelope: n_acoustic_substeps="
        f"{config_kv.get('n_acoustic_substeps')!r}, expected 12. "
        f"Ca_substep would still pass with n_acoustic=120 — exact "
        f"match required."
    )
    # iter-232: add hyperdiff fingerprint (was on the contract but
    # not on the round-2 assertion list). 5e6 is the iter-14 +
    # iter-183 production default; a regression to 1e6 (pre-iter-9
    # default) would slip past every numerical check at 60 steps.
    assert float(config_kv.get("hyperdiff", "nan")) == 5e6, (
        f"plane CRM iter-183 envelope: hyperdiff="
        f"{config_kv.get('hyperdiff')!r}, expected '5e6'. The "
        f"pre-iter-9 default 1e6 (5× too weak) would pass the "
        f"20-sim-min max|w| envelope but fail at 30-day scale."
    )

    # dt=20 fingerprint #1: final logged sim day matches 60·dt/86400.
    # A silent fallback to dt=5 (iter-14 default) would land at
    # 0.00347 instead of 0.01389 — 4× too low.
    expected_final_day = n_steps * dt / 86400.0
    final_day = float(rows[-1]["day"])
    assert abs(final_day - expected_final_day) < 1e-5, (
        f"plane CRM iter-183 envelope: final day={final_day:.6f} "
        f"!= expected {expected_final_day:.6f} (= n_steps·dt/86400 "
        f"at dt={dt}). Likely the helper's dt kwarg defaulted to "
        f"5.0 instead of 20.0 — silent path regression."
    )

    # dt=20 fingerprint #2: Ca_substep > 0.5. iter-183's dt=20 +
    # n_acoustic=12 yields Ca_substep ≈ 0.87; iter-14 dt=5 + same
    # n_acoustic gives ≈ 0.22. A silent dt-fallback would drop
    # this well under 0.5.
    ca_substep_first = float(rows[0]["Ca_substep"])
    assert ca_substep_first > 0.5, (
        f"plane CRM iter-183 envelope: Ca_substep={ca_substep_first:.4f} "
        f"at step 1 below 0.5 lower bound. iter-183's dt=20 + "
        f"n_acoustic=12 must give ≈0.87; a silent dt fallback to 5 "
        f"would drop this to ≈0.22. See run_rce_mpi_long.py iter-183 "
        f"docstring for the contract."
    )

    cwv_first = float(rows[0]["CWV_mean"])
    cwv_final = float(rows[-1]["CWV_mean"])
    mse_first = float(rows[0]["MSE_mean"])
    mse_final = float(rows[-1]["MSE_mean"])

    assert abs(cwv_first - 49.9413) < 0.01, (
        f"plane CRM iter-183 envelope: IC CWV={cwv_first:.4f} mm "
        f"!= 49.9413 ± 0.01. The IC anchor must match the iter-14 "
        f"envelope (same Wing 2018 IC); a drift here means the IC "
        f"build path regressed."
    )

    max_w_per_row = [float(r["max|w|"]) for r in rows]
    for step_idx, mw in zip(logged_steps, max_w_per_row):
        assert mw < 0.05, (
            f"plane CRM iter-183 envelope: max|w|={mw:.3e} m/s at "
            f"step {step_idx} exceeds 0.05 m/s safety cap. iter-229 "
            f"measured 1.06e-2 m/s end-state at 30-day scale; >5e-2 "
            f"at 20 sim-min is a smoking gun for a dt=20 + van_leer "
            f"regression."
        )
    step15_max_w = max_w_per_row[1]
    assert step15_max_w > 1e-6, (
        f"plane CRM iter-183 envelope: max|w|={step15_max_w:.3e} m/s "
        f"at step 15 — dycore appears inactive."
    )
    # iter-231 (Codex round-2 MEDIUM#2): signed CWV GROWTH (not
    # absolute drift). The wrapper-equivalent --no-mass-fixer path
    # must show MOISTURE GAIN early in RCE spinup (surface flux >
    # column losses); a drying run with cwv_final - cwv_first ≈
    # -0.066 mm would pass an abs() check but indicates a flipped
    # surface-flux sign or broken evaporation. Empirical iter-230
    # measurement = +0.066 mm gain over 60 steps.
    cwv_growth = cwv_final - cwv_first
    assert cwv_growth < 0.15, (
        f"plane CRM iter-183 envelope: CWV grew {cwv_growth:+.4f} mm "
        f"(IC={cwv_first:.4f}, final={cwv_final:.4f}) in {n_steps} "
        f"steps. iter-230 measured +0.066 mm; >+0.15 means a "
        f"surface-flux regression (over-firing evaporation)."
    )
    # Inverted fixer-flip floor: empirical gain = +0.066 mm; with
    # fixer ON the gain would be <+0.005 mm. A floor of +1e-3 mm
    # catches the no_mass_fixer kwarg silently defaulting to False
    # AND a flipped surface-flux sign (drying run would land
    # negative, failing the floor).
    assert cwv_growth > 1e-3, (
        f"plane CRM iter-183 envelope: CWV growth {cwv_growth:+.5f} mm "
        f"below +1e-3 floor. The wrapper-equivalent --no-mass-fixer "
        f"path must SHOW MOISTURE GAIN at 20 sim-min (~+0.066 mm "
        f"measured); a negative or near-zero value means either the "
        f"helper's no_mass_fixer kwarg silently defaulted to False "
        f"(fixer pins near zero) OR the surface-flux sign flipped "
        f"(drying instead of moistening)."
    )
    rel_mse_drift = abs(mse_final - mse_first) / mse_first
    assert rel_mse_drift < 1.5e-4, (
        f"plane CRM iter-183 envelope: MSE drift {rel_mse_drift:.3e} "
        f"relative exceeds 1.5e-4 cap. iter-230 measured 5.7e-5 at "
        f"this config; >1.5e-4 is an energy-budget regression."
    )


@pytest.mark.slow
def test_plane_crm_production_scale_132x132_with_radiation(tmp_path):
    """Nightly slow regression for the plane CRM 132x132 production
    config WITH gray radiation refreshing every 60 s sim time
    (iter-39 — radiation analog of iter-38).

    iter-16 added the radiation smoke at 12x12x20 to catch a
    radiation-tendency regression the dycore-only smoke would miss
    (NaN in CWV reduction, crash in first tendency application,
    etc.). iter-38 added the dycore-only smoke at 132x132x30
    production scale. This test closes the symmetry: production
    scale WITH radiation. A radiation regression that only
    triggers at production-scale grid resolution (cwv reduction
    underflow at 132x132 area weighting, gray-tau column integral
    dtype regression, etc.) would slip through iter-16's 12x12
    smoke but be caught here.

    Config: same as iter-38 (132x132x30, dx=2km, dt=5s, hyperdiff
    5e6, Smag c_s=0.2, SI acoustic off-centering=0.1 + 12 substeps,
    mass fixer on, clean Wing IC) but with ``rad_call_interval_s
    =60.0`` — 5 radiation refreshes (steps 1, 13, 25, 37, 49)
    across the 5-min sim window.

    Asserts: logged-steps schema (iter-38 contract), IC CWV anchored
    to 49.9413 ± 0.01 mm (iter-95 post-BC-fix value), max|w| < 0.05 m/s on EVERY logged row
    (catches transient CFL crash from a radiation tendency that
    over-fires), activity floor at step 15 (catches dead-sim
    regression), MSE COOLING — Codex iter-39 HIGH#1 fix: assert
    ``mse_final < mse_first`` AND ``rel_mse_drift > 5e-7`` so a
    silently-disabled radiation path (cached_rad_tend stays None;
    iter-38 truly-dycore-only would land at ≪ 5e-7 drift) cannot
    pass this test. The drift floor combined with the iter-38
    ``--no-radiation`` flag makes the two tests genuinely
    distinguishable.

    Wall budget ~2.5 min on M5 Pro. Marked ``slow``.
    """
    n_steps = 60
    log_every = 15
    out_dir = tmp_path / "rce_plane_prod_rad"
    result = _run_driver_production_scale(
        out_dir, n_outer_steps=n_steps, log_every_steps=log_every,
        rad_call_interval_s=60.0,
    )
    if result.returncode != 0:
        pytest.fail(
            "run_rce_mpi_long.py exited nonzero at production scale "
            f"with radiation ({result.returncode})\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-1000:]}"
        )
    rows = _read_log(out_dir)
    assert rows, (
        "log.txt produced no diagnostic rows at production scale "
        "with radiation"
    )

    # iter-40 hardening (Codex iter-39 MEDIUM#1 fix; iter-40
    # Codex LOW#5 fix: derive expected count from CLI args, don't
    # hardcode). The driver computes
    # ``rad_call_every_steps = max(1, round(rad_interval / dt))``
    # and fires at outer steps where ``(step - 1) % every == 0``,
    # so the expected fire count over ``n_steps`` outer steps is
    # encoded by ``radiation_call_schedule``. For this config
    # (dt=5, interval=60, n_steps=60): every=12, count=5
    # (steps 1, 13, 25, 37, 49). iter-42 routes the test through
    # the same helper the driver uses so a refactor of the
    # schedule formula updates BOTH sides in lock-step.
    from legoesm.driver.physics_schedule import radiation_call_schedule
    dt_s = 5.0
    rad_interval_s = 60.0
    schedule = radiation_call_schedule(rad_interval_s, dt_s, n_steps)
    rad_calls = _parse_rad_call_count(result.stdout)
    assert rad_calls == schedule.num_calls, (
        f"plane CRM production+rad smoke: driver reported "
        f"rad_calls={rad_calls!r}, expected {schedule.num_calls} "
        f"(dt={dt_s}, rad-interval={rad_interval_s}, n_steps={n_steps} "
        f"→ every={schedule.every_steps}, "
        f"fires at {schedule.fire_step_indices}). Either the "
        f"radiation tick gate regressed or the rad_calls counter is "
        f"missing from the Done. line. "
        f"stdout tail:\n{result.stdout[-500:]}"
    )

    expected_logged_steps = [1, 15, 30, 45, 60]
    logged_steps = [int(r["step"]) for r in rows]
    assert logged_steps == expected_logged_steps, (
        f"plane CRM production+rad smoke: logged steps={logged_steps} "
        f"!= expected {expected_logged_steps}. Driver step counter or "
        f"log-every cadence regressed (possibly the radiation tick "
        f"branch broke the outer loop)."
    )

    cwv_first = float(rows[0]["CWV_mean"])
    # iter-95 anchor: IC CWV at 132x132@nlev=30 Wing 2018 =
    # 49.9413 mm post hydrostatic-BC fix (pre-iter-95: 55.550 mm
    # under the legacy T_avg=250 K BC bug). The IC is the same
    # whether radiation is on or off (radiation fires only at t > 0);
    # locking this catches IC-profile regressions on the
    # with-radiation path too.
    assert abs(cwv_first - 49.9413) < 0.01, (
        f"plane CRM production+rad smoke: IC CWV={cwv_first:.4f} mm "
        f"!= 49.9413 ± 0.01. iter-95 anchored this value at "
        f"132x132@nlev=30 post hydrostatic-BC fix."
    )

    max_w_per_row = [float(r["max|w|"]) for r in rows]
    for step_idx, mw in zip(logged_steps, max_w_per_row):
        assert mw < 0.05, (
            f"plane CRM production+rad smoke: max|w|={mw:.3e} m/s "
            f"at step {step_idx} exceeds 0.05 m/s safety cap. "
            f"iter-14 dycore-only envelope is ~6e-3 m/s; radiation "
            f"can drive larger drift but >5e-2 m/s is a CFL crash "
            f"likely caused by an over-firing radiation tendency."
        )
    # Activity floor at step 15 (iter-38 logic): catches dead-sim.
    step15_max_w = max_w_per_row[1]
    assert step15_max_w > 1e-6, (
        f"plane CRM production+rad smoke: max|w|={step15_max_w:.3e} "
        f"m/s at step 15 — dycore appears inactive. iter-14 saw "
        f"~2e-3 m/s by step 20 from baseline thermo gradient + SI "
        f"acoustic. <1e-6 means step() is a no-op."
    )
    # Finite-value sanity on every numeric column for every logged
    # row — radiation NaN typically appears as inf in MSE_mean or
    # negative CWV from a column-integral underflow. iter-16 chose
    # not to assert finiteness; this is a small hardening over that
    # baseline.
    numeric_cols = ("CWV_mean", "CWV_max", "MSE_mean", "max|w|",
                    "max(qc)", "max(qr)", "max(precip_mm_day)",
                    "Ca_substep")
    import math
    for step_idx, row in zip(logged_steps, rows):
        for col in numeric_cols:
            val = float(row[col])
            assert math.isfinite(val), (
                f"plane CRM production+rad smoke: non-finite "
                f"{col}={val!r} at step {step_idx}. Radiation NaN "
                f"or column-integral underflow on the 132x132 "
                f"path."
            )
    # Radiation-specific assertion (Codex iter-39 HIGH#1 fix). Gray
    # radiation cools the column at ~1 K/day in the upper troposphere
    # at this IC. Over a 5-min sim window with 5 tendency refreshes,
    # the integrated MSE drift must be (a) NEGATIVE (cooling, not
    # heating — sign-check catches a flipped flux convention) and
    # (b) ABOVE a small relative floor (catches the silently-disabled
    # radiation path: if cached_rad_tend stays None for the whole run,
    # MSE drift falls to surface-flux + dycore noise floor ≪ 5e-7).
    # The matching iter-38 ``--no-radiation`` test now lands at
    # ~zero MSE drift, so this assertion genuinely distinguishes the
    # two paths.
    mse_first = float(rows[0]["MSE_mean"])
    mse_final = float(rows[-1]["MSE_mean"])
    mse_drift_signed = (mse_final - mse_first) / mse_first
    rel_mse_drift = abs(mse_drift_signed)
    assert mse_drift_signed < 0, (
        f"plane CRM production+rad smoke: MSE drift "
        f"{mse_drift_signed:+.3e} relative is non-negative. Gray "
        f"radiation should cool the column over 5 min sim; positive "
        f"drift means a flipped flux convention or broken sign in "
        f"the LW tendency."
    )
    assert rel_mse_drift > 5e-7, (
        f"plane CRM production+rad smoke: |MSE drift| "
        f"{rel_mse_drift:.3e} relative below 5e-7 floor. Radiation "
        f"appears inactive — cached_rad_tend likely stays None or "
        f"the tendency is zero. iter-38 measures comparable drift "
        f"~2e-5 with cached radiation; truly dycore-only "
        f"(--no-radiation) lands well below 5e-7."
    )
    # Upper bound — radiation over-firing/over-amplification would
    # spike MSE drift past the iter-14 1-sim-hour ceiling. Cap at
    # 5e-4 (3x over iter-14's 1.7e-4) keeps room for radiation
    # cooling at production scale without false positives.
    assert rel_mse_drift < 5e-4, (
        f"plane CRM production+rad smoke: |MSE drift| "
        f"{rel_mse_drift:.3e} relative exceeds 5e-4 cap. Radiation "
        f"tendency is over-firing — check rad_call_every_steps "
        f"calc and cached_rad_tend sign."
    )


@pytest.mark.parametrize("flag,value,expected_err", [
    # iter-67/68 float NaN/inf cases.
    ("--dt", "nan", "must be finite"),
    ("--dt", "inf", "must be finite"),
    ("--dt", "0.0", "must be positive"),
    ("--dt", "-1.0", "must be positive"),
    ("--hyperdiff", "nan", "must be finite"),
    ("--smag-cs", "inf", "must be finite"),
    ("--acoustic-off-centering", "nan", "must be finite"),
    # iter-69 positive-int guards: grid dims + substep counts.
    ("--nx", "0", "must be positive integer"),
    ("--ny", "-1", "must be positive integer"),
    ("--nlev", "0", "must be positive integer"),
    ("--n-acoustic-substeps", "0", "must be positive integer"),
    ("--n-physics-substeps", "-3", "must be positive integer"),
    ("--log-every-steps", "0", "must be positive integer"),
    # iter-70 Codex HIGH: range guards beyond NaN/inf.
    # Positive-floats (zero invalid — div-by-zero or empty loop).
    ("--days", "-1.0", "must be positive"),
    ("--days", "0.0", "must be positive"),
    ("--dx", "-100.0", "must be positive"),
    ("--H", "0.0", "must be positive"),
    ("--dz-sfc", "-50.0", "must be positive"),
    ("--snapshot-hours", "0.0", "must be positive"),
    ("--profile-days", "-5.0", "must be positive"),
    # Non-negative floats (zero is a meaningful disabled sentinel).
    ("--bubble-theta-pert", "-1.0", "must be non-negative"),
    # argparse interprets ``-1e-5`` as a new flag (starts with ``-``);
    # use ``-0.001`` to test negative qv-noise rejection.
    ("--qv-noise-amp", "-0.001", "must be non-negative"),
    # iter-188: --theta-noise-amp added in iter-181 but missing from
    # the iter-70 non-negative validator until iter-188. A negative
    # amplitude used to silently no-op via the ``if amp > 0.0`` gate.
    ("--theta-noise-amp", "-0.01", "must be non-negative"),
    ("--smag-cs", "-0.1", "must be non-negative"),
    ("--hyperdiff", "-1.0", "must be non-negative"),
    ("--sponge-coeff", "-0.01", "must be non-negative"),
    # Acoustic off-centering range [0, 1) — Skamarock-Klemp constraint.
    ("--acoustic-off-centering", "-0.05", "must be in"),
    ("--acoustic-off-centering", "1.0", "must be in"),
    ("--acoustic-off-centering", "1.5", "must be in"),
    # iter-75 post-derivation guard: huge dt / tiny days → total_steps=0
    # silent-pass. argparse type=float accepts 1e10; the guard fires
    # after total_steps computation.
    ("--dt", "1e10", "total_steps=0"),
])
def test_plane_crm_driver_rejects_nan_inf_numeric_args(
    tmp_path, flag, value, expected_err,
):
    """iter-67: numeric CLI args reject NaN/inf with concise
    SystemExit + non-zero exit code, NOT a raw Python traceback.

    iter-65 caught the --rad-call-interval-s case via try/except
    around the physics_schedule helper. iter-67 extends to all
    numeric args via a finiteness validation block in main().

    Without iter-67, --dt nan would crash with
    ``ValueError: cannot convert float NaN to integer`` at
    ``int(total_t / args.dt)`` — opaque trace deep in the driver.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", "0.0005",
        "--no-radiation",
        flag, value,
        "--output", str(tmp_path / f"rce_bad_{flag.strip('-')}"),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0, (
        f"driver exited 0 with {flag}={value}; expected non-zero.\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
    combined = result.stdout + result.stderr
    assert "rejected" in combined and expected_err in combined, (
        f"driver exit message missing 'rejected' + '{expected_err}'.\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
    assert "Traceback" not in combined, (
        f"driver emitted Python traceback for {flag}={value}; "
        f"iter-67 contract: clean SystemExit.\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )


def test_plane_crm_driver_rejects_implicit_buoyancy_without_si(tmp_path):
    """iter-76: ``--implicit-buoyancy`` without
    ``--semi-implicit-acoustic`` is invalid (KW78 substitution lives
    inside the SI substep). iter-1 added this guard with a different
    message style; iter-76 unified it with the iter-67+ pattern
    (``error: <flag> rejected: <reason>``).

    Catches a future silent revert that removes the validation OR
    reverts the message format.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", "0.0005",
        "--no-radiation",
        "--implicit-buoyancy",  # without --semi-implicit-acoustic
        "--output", str(tmp_path / "rce_implicit_no_si"),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0, (
        f"driver exited 0 with --implicit-buoyancy alone; expected "
        f"non-zero.\nstdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )
    combined = result.stdout + result.stderr
    assert "rejected" in combined and "semi-implicit-acoustic" in combined, (
        f"driver exit message missing 'rejected' + "
        f"'semi-implicit-acoustic'.\nstdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )
    assert "Traceback" not in combined, (
        f"driver emitted Python traceback for --implicit-buoyancy "
        f"alone; iter-76 contract: clean SystemExit.\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )


def test_plane_crm_driver_rejects_nan_rad_interval_with_clean_exit(tmp_path):
    """iter-65: passing ``--rad-call-interval-s nan`` (without
    ``--no-radiation``) must produce a SystemExit with a concise
    CLI-style error message + non-zero exit code, NOT a raw
    Python traceback.

    iter-43 physics_schedule.radiation_call_every_steps rejects
    NaN/inf with ``ValueError``. iter-55 added --no-radiation that
    skips the helper. Codex iter-55 LOW#2 noted the ValueError
    still propagates as raw traceback when radiation is enabled.
    iter-65 wraps it in SystemExit.

    Fast — argparse + import + ValueError fires before any compute.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "12", "--ny", "12", "--nlev", "20",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", "0.0005",
        "--rad-call-interval-s", "nan",
        "--output", str(tmp_path / "rce_nan"),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0, (
        f"driver exited 0 with --rad-call-interval-s=nan; expected "
        f"non-zero (NaN must be rejected loudly).\n"
        f"stdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )
    combined = result.stdout + result.stderr
    assert "rejected" in combined, (
        f"driver exit message missing 'rejected' marker. iter-65 "
        f"contract: ``error: --rad-call-interval-s rejected: ...``\n"
        f"stdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )
    assert "Traceback" not in combined, (
        f"driver emitted a raw Python traceback for nan input. "
        f"iter-65 contract: catch ValueError → SystemExit.\n"
        f"stdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )


@pytest.mark.slow
def test_plane_crm_production_scale_132x132_one_hour_envelope(tmp_path):
    """SLOW nightly regression for the plane CRM 132x132 production
    config at near-full iter-14 reference (720 steps = exactly
    1 sim-hour at dt=5s, vs iter-38's 60-step 5-min sub-envelope).

    iter-14 measured 725 outer steps at dt=5s (3625 s sim ≈ 1 sim-hour
    + 25 s) at 132x132x30 dx=2km. iter-14 ran WITH the legacy
    ``--rad-call-interval-s 1e9`` "disable" trick that Codex iter-39
    later found actually fires radiation ONCE at step 1 then caches.
    iter-63 runs at the EXACT 720-step / 3600-s window with the
    iter-39 ``--no-radiation`` flag for a true dycore-only run:

    | step | CWV [mm]   | MSE [J/kg] | max|w| [m/s] |
    |------|------------|------------|--------------|
    |   1  |  49.9413   |  3.52e9    |  0.0e+00     |
    | 100  |  49.9413   |  3.52e9    |  3.9e-3      |
    | 300  |  49.9413   |  3.52e9    |  5.5e-3      |
    | 700  |  49.9413   |  3.52e9    |  6.1e-3      |
    (iter-95 IC anchor; pre-iter-95 the legacy top-down BC bug
    inflated this to ~55.550 mm / ~4.21e9 — see CRM_implementation.md)
    (iter-14 cached-rad values; iter-63 ``--no-radiation`` lands
    LOWER per iter-38's 60-step measurement of 2.4e-5 MSE drift)

    iter-38 only locks the 5-min sub-envelope. iter-63 closes the
    empirical gap between step 60 and step 720 — a regression that
    destabilises later (slow CFL drift, halo edge accumulation,
    mass-fixer convergence issue) slips iter-38 but trips here.

    Wall budget: ~17 min on M5 Pro (1.35 s/step × 720 steps + JIT
    compile). timeout_s=1800 (1.8× cushion). Marked ``slow``;
    runs via ``pytest -m slow``.

    Codex iter-63 HIGH#2 fix: pass timeout_s=1800 explicitly. The
    helper used to hardcode timeout=900 which would have killed
    the 1-hour run before completion.
    """
    n_steps = 720  # 720 * 5 = 3600 s = 1 sim-hour (exact)
    log_every = 60  # logs at steps 1, 60, 120, ..., 720 (13 rows)
    out_dir = tmp_path / "rce_plane_prod_1hr"
    result = _run_driver_production_scale(
        out_dir, n_outer_steps=n_steps, log_every_steps=log_every,
        timeout_s=1800,
    )
    if result.returncode != 0:
        pytest.fail(
            "run_rce_mpi_long.py exited nonzero at production scale "
            f"1-hour ({result.returncode})\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-1000:]}"
        )
    rows = _read_log(out_dir)
    assert rows, "log.txt produced no diagnostic rows at production 1-hour"

    # iter-40 contract: --no-radiation must produce rad_calls=0.
    rad_calls = _parse_rad_call_count(result.stdout)
    assert rad_calls == 0, (
        f"plane CRM 1-hour smoke (--no-radiation): driver "
        f"reported rad_calls={rad_calls!r}, expected 0."
    )

    # Schema sanity: with log_every=60 + n=720, expect rows at
    # steps {1, 60, 120, ..., 720} = 13 rows.
    expected_logged_steps = [1] + list(range(60, 721, 60))
    logged_steps = [int(r["step"]) for r in rows]
    assert logged_steps == expected_logged_steps, (
        f"plane CRM 1-hour smoke: logged steps={logged_steps} != "
        f"expected {expected_logged_steps}. Driver step counter "
        f"or log-every cadence regressed."
    )

    cwv_first = float(rows[0]["CWV_mean"])
    cwv_final = float(rows[-1]["CWV_mean"])
    mse_first = float(rows[0]["MSE_mean"])
    mse_final = float(rows[-1]["MSE_mean"])

    # iter-95 IC anchor at 132x132@nlev=30 post hydrostatic-BC fix
    # (pre-iter-95: 55.550 mm under the legacy T_avg=250 K bug).
    assert abs(cwv_first - 49.9413) < 0.01, (
        f"plane CRM 1-hour smoke: IC CWV={cwv_first:.4f} mm "
        f"!= 49.9413 ± 0.01."
    )

    # max|w| cap on every logged row. iter-14 measured 6.1e-3 m/s
    # at the peak (step 700); cap at 0.05 = 8x cushion vs that.
    max_w_per_row = [float(r["max|w|"]) for r in rows]
    for step_idx, mw in zip(logged_steps, max_w_per_row):
        assert mw < 0.05, (
            f"plane CRM 1-hour smoke: max|w|={mw:.3e} m/s at "
            f"step {step_idx} exceeds 0.05 m/s safety cap. "
            f"iter-14 measured ~6e-3 m/s peak across step 1..700; "
            f"this is a smoking gun for an in-flight CFL crash "
            f"between step 60 (iter-38 boundary) and step 720."
        )

    # Activity floor (same as iter-38).
    step60_max_w = max_w_per_row[1]
    assert step60_max_w > 1e-6, (
        f"plane CRM 1-hour smoke: max|w|={step60_max_w:.3e} m/s "
        f"at step 60 — dycore appears inactive."
    )

    # CWV drift cap. iter-14 saw zero drift to 4 sig figs;
    # 0.01 mm cushion absorbs any rounding/integration noise.
    cwv_drift = abs(cwv_final - cwv_first)
    assert cwv_drift < 0.01, (
        f"plane CRM 1-hour smoke: CWV drifted {cwv_drift:.4f} mm "
        f"in 720 steps. iter-14 saw zero drift over 725 steps."
    )

    # MSE drift cap. iter-14 measured 1.7e-4 relative over 725
    # steps WITH the cached step-1 radiation tendency (Codex
    # iter-39 found this); with --no-radiation the actual drift
    # is much smaller. iter-38 with --no-radiation at 60 steps
    # measured 2.4e-5. Extrapolating per-step rate to 720 steps:
    # ~3e-4. Cap at 5e-4 gives ~1.7x margin over the expected
    # 1-sim-hour drift without radiation.
    rel_mse_drift = abs(mse_final - mse_first) / mse_first
    assert rel_mse_drift < 5e-4, (
        f"plane CRM 1-hour smoke: MSE drift {rel_mse_drift:.3e} "
        f"relative exceeds 5e-4 cap. The iter-14 ceiling was "
        f"1.7e-4 WITH cached radiation; --no-radiation should "
        f"land BELOW that. Investigate dycore + fast-physics "
        f"energy budget at the 1-sim-hour scale."
    )


def test_plane_crm_dt20_van_leer_stability_smoke(tmp_path):
    """iter-183 regression: the new production combination
    (dt=20 + Van Leer + beta=0.2) must run stably with radiation.

    Smoke measurements at 32x32x30 dx=4 km, single-rank, clean
    Wing IC, no bubble, no qv noise, gray radiation every 600 s,
    --no-mass-fixer, --hyperdiff 5e6, --smag-cs 0.2,
    --semi-implicit-acoustic, --acoustic-off-centering 0.2,
    --advection van_leer: 432 steps in 0.6 min wall (M5 Pro CPU)
    with max|w| <= 1.78e-3 m/s and Ca_substep = 0.87. iter-183
    measured 3x wall-time speedup over the dt=10+upwind1 baseline
    at the same sim time.

    A regression that destabilises this combination (e.g. a silent
    revert of the iter-179 Van Leer dispatch wiring, the iter-180
    dt/beta refresh, or the iter-183 advection scheme refresh)
    would surface here as a NaN at < 100 steps. Fast enough
    (~30 s wall) to run in the regular suite, not under ``slow``.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    out_dir = tmp_path / "rce_dt20_van_leer"
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "32", "--ny", "32", "--nlev", "30",
        "--dx", "4000.0", "--dt", "20.0",
        "--days", "0.025",  # 100 outer steps
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", "0.2",
        "--n-acoustic-substeps", "12",
        "--advection", "van_leer",
        "--hyperdiff", "5e6",
        "--smag-cs", "0.2",
        "--bubble-theta-pert", "0.0",
        "--qv-noise-amp", "0.0",
        "--no-mass-fixer",
        "--rad-call-interval-s", "600.0",
        "--log-every-steps", "25",
        "--n-physics-substeps", "1",
        "--output", str(out_dir),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(
            "dt=20 + Van Leer + beta=0.2 driver exited nonzero "
            f"({result.returncode})\n"
            f"stdout tail:\n{result.stdout[-1500:]}\n"
            f"stderr tail:\n{result.stderr[-800:]}"
        )
    rows = _read_log(out_dir)
    assert rows, (
        "dt=20 + Van Leer smoke: log.txt empty. The driver compiled "
        "but exited without writing any data rows."
    )
    # Every row must be finite — any NaN means the new combination
    # destabilised. iter-183 baseline smoke saw max|w| <= 1.8e-3
    # at 432 steps with radiation; 100-step max should land well
    # under 5e-3.
    for r in rows:
        for key in ("CWV_mean", "max|w|", "MSE_mean"):
            val = r[key]
            assert val not in ("nan", "inf", "-inf"), (
                f"dt=20 + Van Leer: {key}={val!r} at step {r['step']!r} "
                f"— iter-183 production combination destabilised."
            )
    final = rows[-1]
    max_w_final = float(final["max|w|"])
    assert max_w_final < 5e-3, (
        f"dt=20 + Van Leer smoke: final max|w|={max_w_final:.3e} "
        f"exceeds 5e-3 cap (iter-183 baseline measurement was "
        f"1.78e-3 at 432 steps with radiation)."
    )
    # Acoustic CFL ratio should stay well below the SI-relaxed
    # bound. Pre-iter-183 the smoke read Ca_substep = 0.87 at
    # dt=20 + N_ACOUSTIC=12. Cap at 1.5 leaves room for minor
    # variations across hardware while catching a config that
    # silently doubles the inner CFL.
    ca_final = float(final["Ca_substep"])
    assert ca_final < 1.5, (
        f"dt=20 + Van Leer smoke: Ca_substep={ca_final:.3f} exceeds "
        f"1.5 — N_ACOUSTIC may have been reduced silently."
    )
