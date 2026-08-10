"""Pin the lat-lon polar-filter / dt gating in the atmosphere matrix runner.

The runner enables the CAM-FV polar Fourier filter (and the cube-like dt=200)
for lat-lon hydrostatic PE cases, EXCEPT the ``rotated_*`` (DCMIP-2008) cases
whose tilted solid-body jet flows across the grid poles — there the zonal
filter would damp the physical cross-polar flow, so it must stay OFF with the
legacy pole-limited dt.  Codex review asked for a test pinning the exact
exception set, the env override, and dt/flag agreement.
"""
import importlib
import sys
from pathlib import Path

import pytest

_MATRIX_DIR = Path(__file__).resolve().parents[2] / "scripts" / "matrix"


@pytest.fixture(scope="module")
def mat():
    sys.path.insert(0, str(_MATRIX_DIR))
    try:
        m = importlib.import_module("run_atmosphere_test_matrix")
    finally:
        # leave it importable for other tests; cheap idempotent insert
        pass
    return m


# Every lat-lon hydrostatic case the runner builds, and whether the polar
# filter must be ON.  Cross-polar-flow (rotated) cases -> OFF.
_LATLON_HYDRO_CASES = {
    "held_suarez": True,
    "held_suarez_topo": True,
    "baroclinic": True,
    "baroclinic_steady": True,   # no "rotated" in the name -> filter ON, same
                                 # as its perturbed partner, so the pair stays
                                 # one-variable (same filter, same dt).
    "rotated_baroclinic": False,
    "rotated_steady": False,
    "rest_state_topo": True,
    "gravity_wave_3_1": True,
    "inertio_gravity_3_2": True,
    "mountain_rossby_5_0": True,
    "rossby_haurwitz_6_0": True,
    "amip": True,
}


def test_only_rotated_cases_disable_filter(mat, monkeypatch):
    monkeypatch.delenv("LEGOESM_LATLON_POLAR_FILTER", raising=False)
    for case, expected_on in _LATLON_HYDRO_CASES.items():
        assert mat._latlon_polar_filter_on(case) is expected_on, (
            f"{case}: filter_on={mat._latlon_polar_filter_on(case)}, "
            f"expected {expected_on}")


def test_rotated_set_matches_case_families(mat, monkeypatch):
    """The string gate ``'rotated' in case`` catches exactly the case names
    that legoESM tags as rotated (no over/under-match)."""
    monkeypatch.delenv("LEGOESM_LATLON_POLAR_FILTER", raising=False)
    rotated = {c for c in mat._CASE_FAMILIES if "rotated" in c}
    assert rotated == {"rotated_baroclinic", "rotated_steady"}
    for c in mat._CASE_FAMILIES:
        # Only the rotated names disable the filter via the substring rule.
        disabled = not mat._latlon_polar_filter_on(c)
        assert disabled == (c in rotated) or "rotated" not in c


def test_env_override_forces_filter_off(mat, monkeypatch):
    monkeypatch.setenv("LEGOESM_LATLON_POLAR_FILTER", "0")
    for case in _LATLON_HYDRO_CASES:
        assert mat._latlon_polar_filter_on(case) is False


def test_dt_agrees_with_filter_flag(mat, monkeypatch):
    """dt == dt_cap exactly when the filter is on; else pole-limited (<dt_cap)."""
    monkeypatch.delenv("LEGOESM_LATLON_POLAR_FILTER", raising=False)
    dx_pole = 6.0e3  # ~ 72x144 polar cell; pole-limited dt = 0.5*6e3/300 = 10 s
    for case, expected_on in _LATLON_HYDRO_CASES.items():
        dt = mat._latlon_dt(dx_pole, 200.0, case)
        if expected_on:
            assert dt == 200.0, f"{case}: filter on but dt={dt}"
        else:
            assert dt < 200.0, f"{case}: filter off but dt={dt} not pole-limited"
            assert dt == pytest.approx(0.5 * dx_pole / 300.0)


def test_latlon_polar_filter_stable_at_relaxed_dt(mat):
    """End-to-end: the polar filter actually makes the relaxed dt stable.

    The gating tests above only check the helper booleans.  This pins the
    SHIPPED operating point — ``use_polar_filter=True`` at the relaxed
    ``dt=_latlon_dt(...)`` (=200 s, ~20x the legacy pole-limited ~10 s) — by
    running the real lat-lon hydrostatic PE model and asserting it stays finite
    and bounded.  Without the filter that dt violates the polar CFL and the run
    blows up within a few steps (the filter damps the offending zonal
    wavenumbers so the mid-latitude grid sets dt).  Conservation is the mass
    fixer's job; the new risk this guards is a polar-CFL blowup, not drift.
    """
    import math
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid)
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_forcing_latlon, held_suarez_init_latlon)

    n_lat, n_lon, nlev = 32, 64, 20
    grid = create_latlon_grid(n_lat, n_lon)
    sigma = create_sigma_coordinate(nlev)
    dx_pole = float(grid.radius) * grid.dlon * math.cos(
        math.pi / 2 - grid.dlat / 2)
    dt = mat._latlon_dt(dx_pole, 200.0, "held_suarez")
    assert dt == 200.0, "filter case should relax dt to the cap"

    # Replicate the runner's HS lat-lon config exactly: horizontal diffusion
    # capped at the polar-cell stability limit so A_h itself is not the blowup.
    dy = math.pi * float(grid.radius) / n_lat
    ah = min(0.1 * math.sqrt(constants.R_d * 300.0) * dy,
             0.4 * dx_pole ** 2 / dt)
    cfg = CGridLatLonPrimitiveEquationConfig(
        A_h=ah, fix_mass=True, anchor_mass_to_initial=True,
        use_polar_filter=mat._latlon_polar_filter_on("held_suarez"))
    assert cfg.use_polar_filter is True
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=dt)
    state = hydrostatic_to_cgrid(held_suarez_init_latlon(grid, sigma), grid)

    p_s0 = float(jnp.sum(state.p_s))
    for _ in range(40):
        state = model.step_with_physics(state, dt, held_suarez_forcing_latlon)

    for name in ("u", "v", "T", "p_s"):
        arr = getattr(state, name)
        assert bool(jnp.all(jnp.isfinite(arr))), (
            f"{name} went non-finite at dt={dt} with the polar filter on")
    max_wind = float(jnp.max(jnp.abs(state.u)))
    assert max_wind < 60.0, (
        f"max|u|={max_wind:.1f} m/s after 40 steps — polar-CFL blowup despite "
        "the filter")
    drift = abs(float(jnp.sum(state.p_s)) - p_s0) / abs(p_s0)
    assert drift < 1e-3, f"surface-pressure mass drift {drift:.2e} too large"
