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
4. ``CWV`` doesn't drift more than 0.1 mm from IC (55.001 mm for the
   12x12 smoke; 55.550 mm for the 132x132 production-scale slow
   tests) — the F8 IC is dynamically frozen until surface flux +
   radiation warm the column on hour-day timescales.

Wall budget: ~30 s on M5 Pro (rough estimate). Single-rank, no MPI,
no JAX JIT pre-compile.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER = REPO_ROOT / "scripts" / "run_rce_mpi_long.py"


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


def _read_log(output_dir):
    """Read log.txt rows as a list of dicts."""
    log_path = Path(output_dir) / "log.txt"
    if not log_path.exists():
        return []
    rows = []
    with open(log_path) as fh:
        header = None
        for line in fh:
            line = line.strip()
            if line.startswith("# step,"):
                header = line.lstrip("# ").split(",")
            elif line.startswith("#") or not line:
                continue
            elif header is not None:
                parts = line.split(",")
                if len(parts) == len(header):
                    rows.append(dict(zip(header, parts)))
    return rows


def _parse_rad_call_count(stdout: str) -> int | None:
    """Extract ``rad_calls=N`` from the driver's final ``Done.`` line.

    Driver format (``scripts/run_rce_mpi_long.py:914``):
        ``f"rad_calls={rad_call_count}."`` — always integer + trailing
        period (the Done.-line punctuation).

    Regex requirements (anchored to that exact format):
    * line starts with ``Done.`` (multiline + ``\\b`` word-bound
      avoids ``total_rad_calls=5`` substring drift — iter-40 Codex
      LOW#4 fix).
    * ``\\d+\\.[^\\S\\n]*$`` (multiline) requires integer-then-
      period as the LAST non-whitespace token on the Done. line.
      The character class ``[^\\S\\n]*`` is "any whitespace except
      newline" — so trailing spaces / tabs / CR (before \\n) are
      tolerated, but a future schema drift like
      ``rad_calls=5. (cached)`` or ``rad_calls=5.0.`` is rejected
      (iter-54 hardening; closes the iter-53-pinned gap).

    Returns the count when exactly one ``Done.`` line matches, else
    ``None`` (older driver / parser-side regression / unexpected
    multiple matches).
    """
    import re
    matches = re.findall(
        r"(?m)^Done\..*\brad_calls=(\d+)\.[^\S\n]*$", stdout,
    )
    if len(matches) != 1:
        return None
    return int(matches[0])


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
    # held for 86 steps). After switching to ``--no-radiation`` the
    # 12x12 measurements at this dt/dz are:
    #   step  1: CWV=55.001 mm, MSE=4.2049e9, max|w|=0.0
    #   step 80: CWV=55.001 mm, MSE=4.2049e9, max|w|=7.07e-4 m/s
    # Anchors below remain valid because (a) radiative cooling over
    # 7 min sim on the 12x12 Wing IC is below the measurement
    # precision in the log (4 sig figs) so CWV/MSE are unchanged,
    # and (b) max|w| at step 80 is 7e-4 m/s vs the 0.5 cap = 700x
    # safety margin. Verified iter-41 run: 2 PASS in 33 s.

    max_w_final = float(rows[-1]["max|w|"])
    cwv_first = float(rows[0]["CWV_mean"])
    cwv_final = float(rows[-1]["CWV_mean"])

    # Anchor the IC CWV: the 12x12 Wing 2018 IC carries 55.001 mm at
    # this nlev/H. If a future commit silently shifts the Wing profile
    # coefficients, the drift assertion below would still pass against
    # the new IC and miss the regression — this gate makes the IC
    # itself part of the contract. Tolerance 0.01 mm (5e-4 relative)
    # is tight enough to detect any meaningful profile change but
    # loose enough for the area-weighted-integration roundoff.
    assert abs(cwv_first - 55.001) < 0.01, (
        f"plane CRM smoke: IC CWV={cwv_first:.4f} mm != 55.001 ± 0.01. "
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
    # 132x132 production grid (55.001 mm at 12x12 vs 55.550 mm at
    # 132x132 due to area-weighted integration of the Wing profile).
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


def _run_driver_production_scale(output_dir, *, n_outer_steps=60,
                                 log_every_steps=15,
                                 rad_call_interval_s=None,
                                 timeout_s=900):
    """Invoke run_rce_mpi_long.py at the iter-14 production scale
    (132x132x30 dx=2km dt=5s) for ``n_outer_steps`` outer steps.

    Defaults reproduce the iter-14/iter-15 measurement that locked
    in the plane CRM production envelope: clean Wing IC (no bubble,
    no qv noise), hyperdiff=5e6, Smag c_s=0.2 (passed explicitly so
    a future driver default change can't silently shift this
    regression), SI acoustic with off-centering=0.1 + 12 substeps,
    mass fixer on (driver default).

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
    days = (n_outer_steps + 0.5) * 5.0 / 86400.0
    cmd = [
        sys.executable, str(DRIVER),
        "--nx", "132", "--ny", "132", "--nlev", "30",
        "--dx", "2000.0", "--dt", "5.0",
        "--days", f"{days:.8f}",
        "--semi-implicit-acoustic",
        "--acoustic-off-centering", "0.1",
        "--n-acoustic-substeps", "12",
        "--advection", "upwind1",
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

    | step | CWV [mm] | MSE [J/kg] | max|w| [m/s] |
    |------|----------|------------|--------------|
    |   1  |  55.550  |  4.2132e9  |  0.0e+00     |
    | 100  |  55.550  |  4.2131e9  |  3.9e-3      |
    | 300  |  55.550  |  4.2129e9  |  5.5e-3      |
    | 700  |  55.550  |  4.2125e9  |  6.1e-3      |

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

    # iter-14 measured IC CWV = 55.550 mm at 132x132 Wing 2018.
    assert abs(cwv_first - 55.550) < 0.01, (
        f"plane CRM production smoke: IC CWV={cwv_first:.4f} mm "
        f"!= 55.550 ± 0.01. iter-14 anchored this value at 132x132. "
        f"If intentional, update the test."
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
    to 55.550 ± 0.01 mm, max|w| < 0.05 m/s on EVERY logged row
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
    # iter-14/38 anchor: IC CWV at 132x132 Wing 2018 = 55.550 mm. The
    # IC is the same whether radiation is on or off (radiation fires
    # only at t > 0); locking this catches IC-profile regressions on
    # the with-radiation path too.
    assert abs(cwv_first - 55.550) < 0.01, (
        f"plane CRM production+rad smoke: IC CWV={cwv_first:.4f} mm "
        f"!= 55.550 ± 0.01. iter-14 anchored this value at 132x132."
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
    ("--dt", "nan", "must be finite"),
    ("--dt", "inf", "must be finite"),
    ("--dt", "0.0", "must be positive"),
    ("--dt", "-1.0", "must be positive"),
    ("--hyperdiff", "nan", "must be finite"),
    ("--smag-cs", "inf", "must be finite"),
    ("--acoustic-off-centering", "nan", "must be finite"),
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

    | step | CWV [mm] | MSE [J/kg] | max|w| [m/s] |
    |------|----------|------------|--------------|
    |   1  |  55.550  |  4.2132e9  |  0.0e+00     |
    | 100  |  55.550  |  4.2131e9  |  3.9e-3      |
    | 300  |  55.550  |  4.2129e9  |  5.5e-3      |
    | 700  |  55.550  |  4.2125e9  |  6.1e-3      |
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

    # IC anchor (same as iter-38).
    assert abs(cwv_first - 55.550) < 0.01, (
        f"plane CRM 1-hour smoke: IC CWV={cwv_first:.4f} mm "
        f"!= 55.550 ± 0.01."
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
