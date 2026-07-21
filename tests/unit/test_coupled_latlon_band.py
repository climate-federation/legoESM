"""Single-process contract tests for the coupled atm + slab-ocean surface
exchange (`coupler.coupled_latlon_band`).

Pins the load-bearing property — EXACT per-cell energy conservation of the
sensible-heat handshake — without a launcher.  The serial==band-MPI parity +
global conservation gate is `tests/distributed/test_coupled_latlon_mpi.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (  # noqa: E402
    CGridLatLonHydrostaticState,
)
from legoesm.coupler.coupled_latlon_band import (  # noqa: E402
    CoupledSlabConfig,
    apply_surface_coupling,
    coupling_energy,
    make_coupled_latlon_band_step,
)

_NLAT, _NLON, _NLEV = 8, 12, 6


def _fake_state(seed=0):
    rng = np.random.default_rng(seed)
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(rng.standard_normal((_NLAT, _NLON + 1, _NLEV))),
        v=jnp.asarray(rng.standard_normal((_NLAT + 1, _NLON, _NLEV))),
        T=jnp.asarray(280.0 + 20.0 * rng.standard_normal((_NLAT, _NLON, _NLEV))),
        p_s=jnp.asarray(1.0e5 + rng.standard_normal((_NLAT, _NLON))),
        phis=jnp.zeros((_NLAT, _NLON)),
        tracers={},
    )


def test_surface_coupling_conserves_energy_exactly():
    """c_atm·ΔT_sfc_air + c_ocean·ΔSST == 0 at every cell (no freezing clamp
    active — warm SST), for the pure sensible exchange."""
    cfg = CoupledSlabConfig()
    st = _fake_state(1)
    rng = np.random.default_rng(2)
    sst = jnp.asarray(290.0 + 5.0 * rng.standard_normal((_NLAT, _NLON)))  # warm
    dt = 3600.0

    st2, sst2, clamp_e = apply_surface_coupling(st, sst, cfg, dt)
    # Warm SST => the clamp never fires => zero injected energy.
    np.testing.assert_array_equal(np.asarray(clamp_e), np.zeros((_NLAT, _NLON)))
    d_air = np.asarray(st2.T[..., -1] - st.T[..., -1])
    d_sst = np.asarray(sst2 - sst)
    resid = cfg.c_atm_area * d_air + cfg.c_ocean_area * d_sst
    # Exact in exact arithmetic (equal-and-opposite flux); the residual is
    # pure float64 round-off of the divide-then-multiply.  Measure it RELATIVE
    # to the per-cell coupled ENERGY scale (c_atm·T_air + c_ocean·SST) — the
    # conservation reference — not the flux magnitude (which is ~0 for
    # near-equilibrium cells and would spuriously inflate the ratio).
    e_scale = np.abs(cfg.c_atm_area * np.asarray(st.T[..., -1])
                     + cfg.c_ocean_area * np.asarray(sst))
    rel = np.abs(resid) / e_scale
    assert float(np.max(rel)) < 1e-14, (
        f"per-cell coupling energy not conserved (max relative "
        f"residual {float(np.max(rel)):.2e})")
    # Only the surface level moved; interior atm levels untouched.
    np.testing.assert_array_equal(
        np.asarray(st2.T[..., :-1]), np.asarray(st.T[..., :-1]))


def test_coupling_energy_invariant_over_repeated_exchange():
    """Repeated exchanges keep the area-weighted coupled surface energy fixed
    (warm SST, no clamp) — the conservation gate's serial analogue."""
    cfg = CoupledSlabConfig()
    st = _fake_state(3)
    rng = np.random.default_rng(4)
    sst = jnp.asarray(295.0 + 3.0 * rng.standard_normal((_NLAT, _NLON)))
    area = jnp.asarray(1.0 + 0.1 * rng.random((_NLAT, _NLON)))  # per-cell m^2
    e0 = float(coupling_energy(st, sst, cfg, area))
    for _ in range(20):
        st, sst, _ = apply_surface_coupling(st, sst, cfg, 1800.0)
    e1 = float(coupling_energy(st, sst, cfg, area))
    assert abs(e1 - e0) / abs(e0) < 1e-12, (e0, e1)


def test_freezing_clamp_floors_sst():
    """A cold-forced SST is clamped at the ocean freezing point (the only
    non-conservative term)."""
    cfg = CoupledSlabConfig()
    st = _fake_state(5)
    # Very cold air everywhere → the exchange pulls SST down; start near freeze.
    st = st._replace(T=st.T.at[..., -1].set(200.0))
    sst = jnp.full((_NLAT, _NLON), cfg.t_freeze_ocean_K + 0.05)
    for _ in range(50):
        st, sst, _ = apply_surface_coupling(st, sst, cfg, 3600.0)
    assert float(jnp.min(sst)) >= cfg.t_freeze_ocean_K - 1e-9


def test_freezing_clamp_energy_bookkeeping():
    """When the clamp FIRES it is a SOURCE to the diagnosed atm+water budget,
    and `apply_surface_coupling` reports exactly how much: the returned
    `clamp_energy` closes the budget so `Δ(c_atm·T_air + c_ocean·SST)` equals
    `Σ clamp_energy` per cell (to round-off), making the non-conservation
    auditable rather than hidden."""
    cfg = CoupledSlabConfig()
    st = _fake_state(6)
    # Cold air pulls SST below freezing on this single exchange => clamp active.
    st = st._replace(T=st.T.at[..., -1].set(200.0))
    sst = jnp.full((_NLAT, _NLON), cfg.t_freeze_ocean_K + 0.02)
    dt = 3600.0

    st2, sst2, clamp_e = apply_surface_coupling(st, sst, cfg, dt)
    clamp_np = np.asarray(clamp_e)
    # The clamp only ever ADDS energy (raises a would-be sub-freezing SST).
    assert float(np.min(clamp_np)) >= 0.0
    assert float(np.max(clamp_np)) > 0.0, "clamp should fire in this cold setup"

    # Per-cell diagnosed-budget change == clamp_energy (the sensible exchange is
    # conservative; the clamp is the entire residual, now accounted for).
    d_budget = (cfg.c_atm_area * np.asarray(st2.T[..., -1] - st.T[..., -1])
                + cfg.c_ocean_area * np.asarray(sst2 - sst))
    e_scale = np.abs(cfg.c_atm_area * np.asarray(st.T[..., -1])
                     + cfg.c_ocean_area * np.asarray(sst))
    rel = np.abs(d_budget - clamp_np) / e_scale
    assert float(np.max(rel)) < 1e-14, (
        f"clamp energy does not close the diagnosed budget (max relative "
        f"residual {float(np.max(rel)):.2e})")


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_rejects_nonpositive_or_nonintegral_substeps(bad):
    """`n_atm_substeps < 1` (or non-integral) must raise, not silently build a
    no-op 'coupled' loop that takes no atm step (dispatch-hardening).  The guard
    runs before the model is touched, so `None` placeholders are fine."""
    with pytest.raises(ValueError, match="n_atm_substeps"):
        make_coupled_latlon_band_step(
            None, None, CoupledSlabConfig(), n_atm_substeps=bad)
