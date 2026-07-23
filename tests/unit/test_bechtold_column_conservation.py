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
    cape_sink_heating_ratio=5.0,
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
