"""The two run_rcemip_plane.py defaults that used to fail silently.

1. ``--theta-noise-amp`` defaulted to 0, so the rest state stayed horizontally
   uniform and the run produced a laminar radiative-equilibrium column while
   exiting 0 with a clean mass budget. Nothing said so (CONV-TRIGGER #83).
2. ``--hyperdiff`` was a FIXED 1.0e6 regardless of dx, unlike the sibling
   GATE/LBA drivers which call ``dx_aware_hyperdiff(dx)``. Since both the
   biharmonic CFL and the 2*dx damping rate go as K/dx^4, a fixed coefficient is
   wrong at every dx but one — ~4 orders too weak at dx = 4 km.

These are behavioural checks: they run the driver on a tiny grid — mostly for a
couple of steps with physics off — and read what it reports. The
startup-transient test needs real radiation and a few hundred steps, so it is
the slow one (~40 s).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts" / "run" / "run_rcemip_plane.py"

# The driver runs in a SCRUBBED environment (no inherited JAX/XLA flags), but
# PYTHONPATH must survive: on a checkout that is not pip-installed — the
# cluster layout, where legoesm resolves from packages/*/ via PYTHONPATH — a
# scrubbed env makes every one of these tests fail with
# "ImportError: cannot import name 'constants' from 'legoesm'", which is an
# environment artefact and not the footgun each test exists to catch.
_SUBPROC_ENV = {"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin",
                **({"PYTHONPATH": os.environ["PYTHONPATH"]}
                   if os.environ.get("PYTHONPATH") else {})}


def _run(tmp_path, *extra, dx="4000.0", steps="2"):
    cmd = [sys.executable, str(DRIVER),
           "--nx", "8", "--ny", "8", "--nlev", "10", "--H", "20000.0",
           "--dx", dx, "--dt", "2.0", "--steps", str(steps),
           "--radiation", "none", "--microphysics", "none",
           # substep_horizontal_acoustic defaults ON and requires the SI path.
           "--semi-implicit",
           *(() if any(a == "--print-every" for a in extra)
             else ("--print-every", "1")),
           "--output", str(tmp_path / "out"), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                       timeout=1800, env=_SUBPROC_ENV)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_hyperdiff_defaults_to_the_dx_aware_value(tmp_path):
    """At dx = 4 km the dx-aware coefficient is 1e8*(4)^4 = 2.56e10, not 1e6."""
    out = _run(tmp_path)
    assert "dx-aware auto" in out, out[-2000:]
    assert "hyperdiff=2.56e+10" in out, out[-2000:]


def test_explicit_hyperdiff_still_wins(tmp_path):
    out = _run(tmp_path, "--hyperdiff", "1.0e6")
    assert "hyperdiff=1.00e+06" in out, out[-2000:]
    assert "explicit" in out, out[-2000:]


def test_dx_aware_scales_with_dx(tmp_path):
    """K/dx^4 is the invariant: halving dx must drop K by 16x."""
    out = _run(tmp_path, dx="2000.0")
    assert "hyperdiff=1.60e+09" in out, out[-2000:]


def test_unseeded_run_is_announced_up_front(tmp_path):
    out = _run(tmp_path)
    assert "NO IC PERTURBATION" in out, out[-2000:]
    assert "--theta-noise-amp 0.1" in out, out[-2000:]


def test_unseeded_run_reports_a_laminar_verdict_at_the_end(tmp_path):
    """The whole point: a laminar run must SAY it never convected instead of
    exiting 0 and looking like a successful integration."""
    out = _run(tmp_path)
    assert "LAMINAR RUN" in out, out[-2000:]
    assert "never convected" in out, out[-2000:]


def test_laminar_verdict_does_not_scold_a_no_radiation_dycore_smoke(tmp_path):
    """Not every use of this driver wants convection. A no-radiation dycore
    smoke (and a DNS/LES closure probe) is legitimately laminar, so the verdict
    must not prescribe an RCE seed fix there."""
    out = _run(tmp_path)  # --radiation none
    assert "not a problem there" in out, out[-2000:]
    assert "--theta-noise-amp 0.1 --seed-kind band_noise" not in out.split(
        "LAMINAR RUN")[-1], out[-2000:]


def test_reported_peak_excludes_the_startup_transient(tmp_path):
    """Regression: with radiation ON the rest state's initial hydrostatic
    adjustment reaches ~0.8 m/s at step 1 even when the domain is uniform and
    stays uniform. A whole-run peak would clear any convective threshold, so the
    laminar check would never fire. Assert the reported figure is the SECOND-HALF
    peak of the printed table, not the global one."""
    import re as _re
    # This grid/timestep is the one where the transient genuinely dominates:
    # step 1 reaches ~0.8 m/s and the uniform column then decays to ~4e-3 m/s.
    n_steps = 400
    cmd = [sys.executable, str(DRIVER),
           "--nx", "32", "--ny", "32", "--nlev", "30",
           "--dx", "2000.0", "--dt", "5.0", "--steps", str(n_steps),
           "--radiation", "gray", "--microphysics", "kessler",
           "--semi-implicit", "--print-every", "50",
           "--output", str(tmp_path / "out")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                       timeout=1800, env=_SUBPROC_ENV)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    # A uniform column that never convected must be called out — and because
    # radiation is ON here, the seed diagnosis and fix must be offered.
    assert "LAMINAR RUN" in r.stdout, r.stdout[-3000:]
    assert "--theta-noise-amp is 0" in r.stdout, r.stdout[-3000:]
    assert "--seed-kind band_noise" in r.stdout, r.stdout[-3000:]

    # Printed rows: "<step>  <t>  <max|w|>  ..."
    rows = {}
    for line in r.stdout.splitlines():
        m = _re.match(r"\s*(\d+)\s+[\d.]+\s+([\d.eE+-]+)\s+[-\d.]", line)
        if m:
            rows[int(m.group(1))] = float(m.group(2))
    assert len(rows) > 4, f"could not parse the diagnostic table:\n{r.stdout}"

    reported = float(_re.search(
        r"peak \|w\| over the second half of the integration was "
        r"([\d.eE+-]+) m/s", r.stdout).group(1))
    second_half = [w for s, w in rows.items() if s >= n_steps // 2]
    assert reported == pytest.approx(max(second_half), rel=1e-2), (
        f"reported {reported}, second-half peak {max(second_half)}")
    # The step-1 transient is NOT part of the contract, and asserting that it
    # dominates makes this test fail on a physically-fine configuration.
    # Measured on clean main (dd4bf62a0): step-1 5.963e-05 m/s vs second-half
    # peak 3.525e-04 — i.e. THIS ASSERTION IS RED ON MAIN TODAY. Both parents'
    # ICs sit near 80 % RH, so neither produces the step-1 condensation kick the
    # assertion assumed (that came from an older IC that pinned T_v0 = 295 K and
    # left the column 39 % supersaturated). What this test exists to pin is the
    # WINDOWING arithmetic — reported == second-half max — which the assertions
    # above already cover.
    assert rows[1] > 0.0


def test_no_false_laminar_verdict_when_print_every_is_too_coarse(tmp_path):
    """|w| is sampled only on print steps. If --print-every is coarse enough
    that none land in the second half, the accumulator is still 0.0 — emitting
    a LAMINAR verdict there would libel a perfectly good convecting run."""
    out = _run(tmp_path, "--print-every", "4000", steps="10")
    assert "no |w| samples in the second half" in out, out[-2000:]
    assert "LAMINAR RUN" not in out, out[-2000:]


def test_smooth_k1_seed_is_flagged_as_single_circulation(tmp_path):
    out = _run(tmp_path, "--theta-noise-amp", "0.1", "--seed-kind", "smooth_k1")
    assert "smooth_k1 puts all seed energy" in out, out[-2000:]


def test_negative_hyperdiff_is_rejected(tmp_path):
    cmd = [sys.executable, str(DRIVER), "--nx", "8", "--ny", "8", "--nlev", "10",
           "--steps", "1", "--hyperdiff", "-1.0",
           "--output", str(tmp_path / "out")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                       timeout=600, env=_SUBPROC_ENV)
    assert r.returncode != 0
    assert "must be non-negative" in (r.stdout + r.stderr)
