"""Energy/moisture budget diagnostics must use the coordinate's real layer mass.

`column_moist_static_energy` assumed a PURE-SIGMA column (`dp = p_s*dsigma`,
`p_full = p_s*sigma_full`), but `vertical_coord` DEFAULTS to hybrid, where the
truth is `dA*p_ref + dB*p_s` and `A*p_ref + B*p_s`. Measured error in column
MSE on a hybrid grid: up to 3.6%, largest where p_s departs from p_ref.

Conservation diagnostics are first-class here, so both the sigma no-op and the
hybrid correction are pinned.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.diagnostics.energy_budget import (
    column_dry_static_energy,
    column_moist_static_energy,
)
from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels

NLEV = 20
NCOL = 6


def _state():
    r = np.random.default_rng(0)
    return dict(
        T=jnp.asarray(np.tile(np.linspace(230.0, 290.0, NLEV), (NCOL, 1))),
        q_v=jnp.asarray(r.uniform(1.0e-4, 1.0e-2, (NCOL, NLEV))),
        u=jnp.zeros((NCOL, NLEV)),
        v=jnp.zeros((NCOL, NLEV)),
        phis=jnp.zeros(NCOL),
        # spans sea level down to elevated terrain, where hybrid departs most
        p_s=jnp.asarray(np.linspace(7.0e4, 1.01e5, NCOL)),
    )


def _call(fn, coord, s, explicit):
    # Signatures read from source, not guessed:
    #   column_moist_static_energy(T, q_v, u, v, phis, p_s, dsigma, sigma_full)
    #   column_dry_static_energy  (T,           phis, p_s, dsigma, sigma_full)
    a = ((s["T"], s["q_v"], s["u"], s["v"], s["phis"], s["p_s"])
         if fn is column_moist_static_energy
         else (s["T"], s["phis"], s["p_s"]))
    kw = {}
    if explicit:
        kw = dict(dp=coord.layer_thickness_dp(s["p_s"]),
                  p_full=coord.pressure_at_full(s["p_s"]))
    return np.asarray(fn(*a, coord.dsigma, coord.sigma_full, **kw))


def test_pure_sigma_is_exactly_unchanged():
    """Passing the coordinate's own dp/p_full on a SIGMA grid must reproduce
    the legacy p_s*dsigma path to the bit — it is the same arithmetic."""
    s = _state()
    coord = create_sigma_coordinate(NLEV)
    legacy = _call(column_moist_static_energy, coord, s, explicit=False)
    explicit = _call(column_moist_static_energy, coord, s, explicit=True)
    np.testing.assert_array_equal(legacy, explicit)


def test_hybrid_correction_is_material_and_finite():
    """On hybrid the legacy path is WRONG; pin the size so a regression shows."""
    s = _state()
    coord = make_hybrid_levels(NLEV)
    legacy = _call(column_moist_static_energy, coord, s, explicit=False)
    fixed = _call(column_moist_static_energy, coord, s, explicit=True)
    assert np.all(np.isfinite(fixed)) and np.all(fixed > 0.0)
    rel = np.abs(legacy - fixed) / np.abs(fixed)
    assert rel.max() > 0.01, f"correction vanished ({rel.max():.4f})"
    assert rel.max() < 0.20, f"implausibly large ({rel.max():.4f})"
    # largest where p_s departs most from p_ref (the lowest-p_s column)
    assert rel[0] > rel[-1]


def test_dry_static_energy_gets_the_same_treatment():
    """The sibling function carries the identical defect; fixing only one
    would leave a wrong diagnostic behind."""
    s = _state()
    sig, hyb = create_sigma_coordinate(NLEV), make_hybrid_levels(NLEV)
    np.testing.assert_array_equal(
        _call(column_dry_static_energy, sig, s, explicit=False),
        _call(column_dry_static_energy, sig, s, explicit=True))
    lo = _call(column_dry_static_energy, hyb, s, explicit=False)
    hi = _call(column_dry_static_energy, hyb, s, explicit=True)
    assert np.all(np.isfinite(hi))
    assert np.abs(lo - hi).max() / np.abs(hi).max() > 0.005


def test_no_nan_from_the_out_of_bounds_first_scan_step():
    """Regression: the hydrostatic scan's first step indexes j_below = nlev,
    which is OUT OF BOUNDS. The original relied on JAX __getitem__ CLAMPING to
    make it a no-op; `jnp.take` defaults to mode="fill" and returns NaN, which
    poisoned every column. mode="clip" restores the original semantics.
    """
    s = _state()
    for coord in (create_sigma_coordinate(NLEV), make_hybrid_levels(NLEV)):
        out = _call(column_moist_static_energy, coord, s, explicit=True)
        assert np.all(np.isfinite(out)), f"NaN in {coord.__class__.__name__}"


def test_moisture_tracker_accepts_dp():
    """The moisture budget integrates q_v, which is BOTTOM-HEAVY, so the
    hybrid layer-mass error does not cancel the way it does for a uniform
    tracer."""
    from legoesm.diagnostics.energy_budget import MoistureBudgetTracker

    coord = make_hybrid_levels(NLEV)
    s = _state()
    t = MoistureBudgetTracker()
    zero = jnp.zeros(NCOL)
    a = t.update(s["q_v"], s["p_s"], coord.dsigma, zero, zero,
                 elapsed_seconds=0.0)
    t2 = MoistureBudgetTracker()
    b = t2.update(s["q_v"], s["p_s"], coord.dsigma, zero, zero,
                  elapsed_seconds=0.0, dp=coord.layer_thickness_dp(s["p_s"]))
    assert np.isfinite(b.column_water)
    assert abs(a.column_water - b.column_water) > 1.0e-3, (
        "dp made no difference — it is probably not being forwarded")
