"""Leaf column water/enthalpy conservation for Bechtold (P1.3+P1.4 gate).

The 2026-07-22 per-process budget ledger convicted the production Bechtold
configuration of (i) fabricating column heat (equilibrium: +185 W/m² of
heating vs 1.25 mm/day ≈ 36 W/m² of water-consistent conversion) and
(ii) removing column water that never reaches the surface-precip booking
(CMOR pr ≈ the microphysics share only).  This test pins BOTH budgets at
the scheme boundary, on the production-resolved flag set.

Sign conventions in scope (stated per the sign-convention gate):
- tendencies are SOURCES: state += dt * tendency; level index surface-LAST;
  dp_full > 0.
- The pipeline routes ``dq_c_conv_dt`` into the grid cloud (stays in the
  column) and books ``surface precip P = sum(dq_r_conv_dt * dp) / g``
  (rain leaves the column immediately, positive-down).
- WATER closure therefore requires
      integral(dq_v_dt + dq_c_conv_dt + dq_r_conv_dt) dp/g = 0
  (what the vapor loses beyond the retained cloud is exactly the rain
  that precipitates; in - out - d(storage) = 0).
- ENTHALPY closure on h = c_pd*T + L_v*q_v requires
      integral(c_pd*dT_dt + L_v*dq_v_dt) dp/g = (L_f books)
  and the scheme's L_f books (formation-side freezing + in-march melt +
  surface fold) are documented to cancel EXACTLY, CMT carries no heat
  booking, so the moist-enthalpy residual must be ~0 for any flag set.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
from legoesm.atmosphere.physics.convection.config import BechtoldConfig

from tests.unit.test_bechtold import _column


# Production-resolved flag set (scripts/tmp/_probe_amip_bechtold_config.py,
# 2026-07-22: experiment_config.json of the 2-yr pilot / decade v3).
PROD_CONFIG = BechtoldConfig(
    cape_threshold=70.0,
    M_b_max=0.02,
    cmt_c_u=0.7,
    cmt_c_d=0.7,
    p_conv_top_pa=15000.0,
    precip_split_scheme="constant",
    precip_efficiency=0.8,
    autoconv_q_c_crit=5.0e-4,
    autoconv_pe_max=0.9,
    downdraft_evap_efficiency=0.05,
    downdraft_alpha=0.3,
    downdraft_RH_min=0.2,
    downdraft_transport=False,
    downdraft_entrain_rate=3.0e-4,
    downdraft_detrain_scale_m=700.0,
    use_ifs_cape_closure=True,
    use_ifs_subcloud_evap=True,
    use_ifs_inplume_precip=True,
    dx_m=0.0,
    use_ifs_downdraft=True,
    use_ifs_shallow_closure=False,
    use_ifs_capdcycl=True,
    use_ifs_land_rhebc=True,
    use_ifs_snow_melt=True,
)

DT = 300.0  # production time step


def _spun_up_tendencies(config, n_spin=20):
    """Run the scheme to a quasi-steady plume (M_u carry fed back) and
    return the LAST call's outputs + the column geometry."""
    T, q, pf, ph, u, v = _column(ncol=4, nlev=20)
    cpp = jnp.zeros_like(T)
    stoch = jnp.zeros((T.shape[0],))
    out = None
    for _ in range(n_spin):
        out, cpp, stoch = bechtold_convection(
            T, q, pf, ph, u, v, cpp, stoch, None, dt=DT, config=config,
        )
    dp = ph[:, 1:] - ph[:, :-1]           # > 0, surface-last
    return out, dp


def _budgets(out, dp):
    """(water residual [kg/m²/s], moist-enthalpy residual [W/m²],
    heating [W/m²], L_v*vapor sink [W/m²], booked precip [kg/m²/s])
    as per-column arrays."""
    g = constants.g
    dq_r = out.dq_r_conv_dt if out.dq_r_conv_dt is not None else 0.0
    water = jnp.sum(
        (out.dq_v_dt + out.dq_c_conv_dt + dq_r) * dp, axis=-1) / g
    heat = constants.c_pd * jnp.sum(out.dT_dt * dp, axis=-1) / g
    lv_sink = -constants.L_v * jnp.sum(out.dq_v_dt * dp, axis=-1) / g
    enthalpy = heat - lv_sink   # = ∫(c_p dT + L_v dq_v) dp/g
    precip = jnp.sum(jnp.asarray(dq_r) * jnp.ones_like(out.dq_v_dt)
                     * dp, axis=-1) / g if out.dq_r_conv_dt is not None \
        else jnp.zeros(out.dT_dt.shape[0])
    return (np.asarray(water), np.asarray(enthalpy), np.asarray(heat),
            np.asarray(lv_sink), np.asarray(precip))


def test_production_column_water_closes():
    """∫(dq_v+dq_c+dq_r) dp/g = 0: whatever vapor the scheme removes beyond
    retained cloud must be exactly the rain it books to surface precip."""
    out, dp = _spun_up_tendencies(PROD_CONFIG)
    water, _, heat, _, _ = _budgets(out, dp)
    assert float(np.max(np.abs(np.asarray(out.convective_mask)))) > 0.5, \
        "fixture must convect"
    assert float(np.max(heat)) > 20.0, "fixture must drive O(100 W/m²)"
    # mm/day equivalent for readability; tolerance 0.02 mm/day on an
    # O(5-10 mm/day) active column (~0.3% — numerics, not physics).
    water_mm_day = water * 86400.0
    np.testing.assert_allclose(water_mm_day, 0.0, atol=2e-2)


def test_production_column_enthalpy_closes():
    """∫(c_pd dT + L_v dq_v) dp/g ≈ 0: heating must be paired with actual
    vapor conversion (L_f books cancel by construction; CMT books no heat).
    The fabricated-heat defect makes this residual O(+100 W/m²)."""
    out, dp = _spun_up_tendencies(PROD_CONFIG)
    _, enthalpy, heat, lv_sink, _ = _budgets(out, dp)
    assert float(np.max(heat)) > 20.0, "fixture must drive O(100 W/m²)"
    np.testing.assert_allclose(enthalpy, 0.0, atol=3.0)  # W/m²


def test_legacy_flagset_budgets_close():
    """The all-IFS-off legacy path must satisfy the same two budgets —
    the conservation contract is flag-independent."""
    legacy = PROD_CONFIG._replace(
        use_ifs_cape_closure=False, use_ifs_subcloud_evap=False,
        use_ifs_inplume_precip=False, use_ifs_downdraft=False,
        use_ifs_capdcycl=False, use_ifs_land_rhebc=False,
        use_ifs_snow_melt=False, use_convective_turnover_tau=False,
    )
    out, dp = _spun_up_tendencies(legacy)
    water, enthalpy, heat, _, _ = _budgets(out, dp)
    assert float(np.max(heat)) > 20.0, "fixture must drive O(100 W/m²)"
    np.testing.assert_allclose(water * 86400.0, 0.0, atol=2e-2)
    np.testing.assert_allclose(enthalpy, 0.0, atol=3.0)


# ---------------------------------------------------------------------------
# Penetrative-downdraft transport: the per-level CFL cap binds AND conserves
# ---------------------------------------------------------------------------

def _thin_layer_column(ncol=2, nlev=16):
    """A column with one 2 hPa layer just above the boundary layer, the shape
    that blew the uncapped transport up on a restart."""
    T, q, p_full, p_half, _u, _v = _column(ncol=ncol, nlev=nlev)
    p_half = np.array(p_half, dtype=np.float64)
    # squeeze level nlev-4 to 200 Pa by moving its upper interface down
    k = nlev - 4
    p_half[:, k] = p_half[:, k + 1] - 200.0
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dp_full = p_half[:, 1:] - p_half[:, :-1]
    z = -8500.0 * np.log(p_full / p_half[:, -1:])
    return (jnp.asarray(T), jnp.asarray(q), jnp.asarray(z),
            jnp.asarray(dp_full))


def _dd_transport(M_d_mag, dt=112.5, nlev=16):
    from legoesm.atmosphere.physics.convection.bechtold import (
        _DD_CFL_FRAC, _penetrative_downdraft_transport)
    T, q, z, dp = _thin_layer_column(nlev=nlev)
    ncol = T.shape[0]
    k_lcl = jnp.full((ncol,), nlev - 3.0)          # LCL just above the surface
    levels = jnp.arange(nlev, dtype=z.dtype)
    dT, dq = _penetrative_downdraft_transport(
        T, q, z, dp, k_lcl, levels, jnp.full((ncol,), M_d_mag),
        entrain_rate=3.0e-4, detrain_scale_m=700.0, dt=dt)
    return dT, dq, q, dp, dt, _DD_CFL_FRAC


def test_downdraft_cfl_cap_is_the_neighbour_minimum():
    from legoesm.atmosphere.physics.convection import bechtold as b
    dp = jnp.asarray([[6000.0, 6000.0, 200.0, 6000.0, 6000.0]])
    cap = np.asarray(b._downdraft_cfl_cap(dp, 100.0))[0]
    want = b._DD_CFL_FRAC * np.array([6000.0, 200.0, 200.0, 200.0, 6000.0]) / (constants.g * 100.0)
    np.testing.assert_allclose(cap, want, rtol=1e-12)


def test_downdraft_transport_cap_is_on_the_path_and_conserves(monkeypatch):
    from legoesm.atmosphere.physics.convection import bechtold as b
    dT_cap, dq_cap, q, dp, dt, frac = _dd_transport(1.0)
    monkeypatch.setattr(b, "_downdraft_cfl_cap",
                        lambda dp_full, dt: jnp.full_like(dp_full, 1.0e30))
    dT_off, dq_off, *_ = _dd_transport(1.0)
    for a in (dT_cap, dq_cap, dT_off, dq_off):
        assert bool(jnp.all(jnp.isfinite(a)))
    # The cap binds at the 2 hPa layer: removing it changes the transport.
    k = 16 - 4
    assert float(jnp.max(jnp.abs(dq_off[:, k] - dq_cap[:, k]))) > 1e-3 * float(
        jnp.max(jnp.abs(dq_cap)))
    assert float(jnp.max(jnp.abs(dq_cap))) > 0.0
    # Conservation survives the level-varying cap: the dp-weighted column
    # integrals of the vapour and dry-static-energy tendencies vanish.
    col_q = jnp.sum(dq_cap * dp, axis=1) / jnp.sum(jnp.abs(dq_cap) * dp, axis=1)
    col_s = jnp.sum(dT_cap * dp, axis=1) / jnp.sum(jnp.abs(dT_cap) * dp, axis=1)
    assert float(jnp.max(jnp.abs(col_q))) < 1e-10
    assert float(jnp.max(jnp.abs(col_s))) < 1e-10
    # The column limiter still holds: no level loses more than _DD_CFL_FRAC
    # of its vapour in one step, with no per-level clamp.
    assert bool(jnp.all(dq_cap * dt >= -frac * q * (1.0 + 1e-6)))
