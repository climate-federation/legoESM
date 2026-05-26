"""Pin the auto-dt heuristic used by ``scripts/run_rce.py``.

iter-24: now imports ``legoesm.driver.rce_dt.auto_dt_rce`` directly
instead of hand-copying the production logic into a test mirror
(Codex iter-19 LOW finding). Any change to the production heuristic
that lands without updating this test will trip on the explicit
boundary assertions below.

Empirical lineage of each ladder boundary:
* voronoi V4 → dt=300 (iter-8 fix; dt>=450 blows up at day 1)
* C24 → 600 (iter-12 30-day PASS)
* C48 → 150 (iter-13/15 30-day PASS; dt=300 blows up at day 25)
* C72 → 75  (iter-13 small-N end; C72 30-day in flight at iter-22/23)
* C96 → 37  (iter-20 fix after dt=75 BLOWUP at day 20)
* N>96 → ValueError (iter-21 hard refusal; silent extrapolation
  produced the iter-13/20 cascade of BLOWUPs).
"""
from __future__ import annotations

import pytest

from legoesm.driver.rce_dt import auto_dt_rce


def _resolve_dt(grid_type, resolution, dt_override=None):
    """Reproduce the run_rce.py contract: explicit --dt wins, else
    auto-dt. Tests this contract here so we don't have to subprocess
    out to the script just to verify the precedence."""
    if dt_override is not None:
        return float(dt_override)
    return auto_dt_rce(grid_type, resolution)


def test_voronoi_auto_dt_is_300s():
    """Voronoi must default to 300 s regardless of resolution
    (R10 cross-grid finding 2026-05)."""
    for N in (4, 5, 8, 12):
        assert auto_dt_rce("voronoi", N) == 300.0, (
            f"voronoi at N={N} must default to 300 s; got "
            f"{auto_dt_rce('voronoi', N)}"
        )


def test_cubed_sphere_auto_dt():
    assert auto_dt_rce("cubed_sphere", 24) == 600.0
    assert auto_dt_rce("cubed_sphere", 25) == 150.0  # iter-13: C25-48 needs 150
    assert auto_dt_rce("cubed_sphere", 48) == 150.0
    assert auto_dt_rce("cubed_sphere", 49) == 75.0
    assert auto_dt_rce("cubed_sphere", 72) == 75.0
    assert auto_dt_rce("cubed_sphere", 73) == 37.0   # iter-20: C96 needs ≤37
    assert auto_dt_rce("cubed_sphere", 96) == 37.0
    # iter-21: N>96 RAISES instead of silently picking dt=20.
    with pytest.raises(ValueError, match=">96"):
        auto_dt_rce("cubed_sphere", 97)
    with pytest.raises(ValueError):
        auto_dt_rce("cubed_sphere", 192)


def test_latlon_auto_dt():
    assert auto_dt_rce("latlon", 16) == 600.0
    assert auto_dt_rce("latlon", 32) == 150.0


def test_gaussian_auto_dt():
    assert auto_dt_rce("gaussian", 21) == 600.0
    assert auto_dt_rce("gaussian", 42) == 150.0


def test_override_takes_precedence():
    """Explicit --dt always wins over auto-dt, even for high-N where
    the auto path would raise."""
    assert _resolve_dt("voronoi", 4, dt_override=60.0) == 60.0
    assert _resolve_dt("cubed_sphere", 24, dt_override=900.0) == 900.0
    # The override path bypasses the N>96 ValueError gate (this is
    # the documented escape hatch for high-resolution runs).
    assert _resolve_dt("cubed_sphere", 192, dt_override=10.0) == 10.0


def test_auto_dt_rce_lies_inside_cfl_envelope():
    """Sanity: every empirical ladder value must satisfy
    ``auto_dt_rce(...) <= 2.0 * gravity_wave_cfl(dx)``.

    iter-23 CFL advisory measurements showed our ladder picks dt
    between 0.33× and 1.32× the gravity-wave CFL formula. A future
    bump that pushes past 2.0× would be a strong signal that the
    ladder is back in BLOWUP territory — the iter-13/20 mistakes
    that produced the C48/C96 crashes both started with ratios ≥
    1.32×. 2.0× gives a comfortable buffer above the empirical
    envelope while still catching gross regression.

    Voronoi/MPAS is excluded because its CFL profile is different
    (the dx_min estimator uses a different formula and the actual
    stability bound is not gravity-wave-CFL-limited).

    Latlon is also excluded: ``auto_dt_rce`` returns the un-clamped
    ladder value, but ``run_rce.py`` runs a SECOND pole-cell-CFL
    clamp afterwards (see scripts/run_rce.py:229-235) that drops
    the effective dt below the latlon CFL formula. Measuring the
    auto-dt ladder against the formula directly is therefore not
    meaningful for latlon — the effective dt is the clamped value.
    """
    from legoesm.core.cfl import (
        cfl_max_dt, estimate_min_dx_cubed_sphere,
        estimate_min_dx_gaussian,
    )
    cases = [
        ("cubed_sphere", estimate_min_dx_cubed_sphere, [24, 48, 72, 96]),
        ("gaussian", estimate_min_dx_gaussian, [21, 42]),
    ]
    for grid_type, dx_fn, resolutions in cases:
        for N in resolutions:
            dx_min = dx_fn(N)
            dt_cfl = float(cfl_max_dt(dx_min, 300.0, cfl_number=0.8, ndim=2))
            dt_ladder = auto_dt_rce(grid_type, N)
            ratio = dt_ladder / dt_cfl
            assert ratio < 2.0, (
                f"{grid_type}/N={N}: ladder dt={dt_ladder} > 2.0× CFL "
                f"formula ({dt_cfl:.1f} s, ratio={ratio:.2f}). iter-13/20 "
                f"showed crossings of 1.3× already produce BLOWUP at "
                f"30-day; ratio={ratio:.2f} is firmly in the danger zone."
            )


def test_print_rce_auto_dt_table_script_runs():
    """Smoke: the iter-31 diagnostic table script
    (scripts/print_rce_auto_dt_table.py) must run cleanly and print
    rows for every cubed_sphere row in the parametrise above.

    Lightweight — no JAX dycore, no run_rce.py invocation. Catches
    breakage of the script (e.g. an estimator removed, the
    ``auto_dt_rce`` import broken) without spending wall time.
    """
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[3] / "scripts" / "print_rce_auto_dt_table.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"print_rce_auto_dt_table.py exited {result.returncode}:\n"
        f"{result.stdout}\n---stderr---\n{result.stderr}"
    )
    # Every cubed_sphere case must appear in the table.
    for token in ("C24", "C48", "C72", "C96"):
        assert token in result.stdout, (
            f"diagnostic table missing {token} row:\n{result.stdout}"
        )
    # The header must include the expected columns.
    assert "ladder dt" in result.stdout
    assert "CFL ratio" in result.stdout
    assert "fit ratio" in result.stdout


def test_ladder_matches_empirical_dt_dx2_fit():
    """The iter-12..26 measurements give a clean ``dt ∝ dx²`` fit on
    the cubed_sphere branch (iter-28 finding). This test asserts the
    ladder values agree with that fit within 30%, locking in the
    structural scaling.

    Tightens the iter-28 "2×-CFL envelope" check from a sanity
    backstop into a structural regression: a future ladder change
    that breaks the dx² scaling (e.g. a halving past N=96 instead
    of quartering) will FAIL here.

    30% tolerance reflects iter-12..26 measurements: actual ratios
    of ladder dt to the empirical fit are 1.00 (C24 anchor), 1.20
    (C48), 0.98 (C72), 1.18 (C96). 30% is double the largest
    observed deviation.
    """
    from legoesm.core.cfl import estimate_min_dx_cubed_sphere
    from legoesm.driver.rce_dt import empirical_dt_dx2
    for N in (24, 48, 72, 96):
        dx = estimate_min_dx_cubed_sphere(N)
        fit_dt = empirical_dt_dx2(dx)
        ladder_dt = auto_dt_rce("cubed_sphere", N)
        rel = ladder_dt / fit_dt
        assert 0.70 < rel < 1.30, (
            f"C{N}: ladder dt={ladder_dt}, empirical-fit dt={fit_dt:.1f}, "
            f"ratio={rel:.2f} outside [0.7, 1.3]. iter-28 documented the "
            "dt ∝ dx² scaling; a 30%+ deviation means either the ladder "
            "or the fit anchor has shifted — bisect against rce_dt.py."
        )


def test_auto_dt_rce_is_public_api():
    """auto_dt_rce must be re-exported from ``legoesm.driver`` so
    callers can do ``from legoesm.driver import auto_dt_rce`` instead
    of digging into the submodule. iter-24 Codex HIGH caught the
    missing re-export when iter-24 first landed."""
    import legoesm.driver
    assert hasattr(legoesm.driver, "auto_dt_rce"), (
        "legoesm.driver does not re-export auto_dt_rce — iter-24 "
        "follow-up missing? Update src/legoesm/driver/__init__.py."
    )
    # And the re-export must be the SAME function object (not a wrapper).
    from legoesm.driver.rce_dt import auto_dt_rce as _from_submodule
    assert legoesm.driver.auto_dt_rce is _from_submodule
