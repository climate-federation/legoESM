"""Canopy Vcmax25 depth profile switch (PR4): kn vs coordination.

The "coordination" option sets the nitrogen-extinction exponent to a tunable
fraction of the two-stream's diffuse-PAR light envelope
(kn_eff = vcmax_light_frac * 0.72 * LAI in the kn*CI*x convention) and leaves
every integral, the sun/shade split, and the leaf-top Vcmax25 unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy.config import CanopyConfig, VALID_VCMAX_PROFILES
from legoesm.land.canopy.radiative_transfer import (
    canopy_shortwave_rt, coordination_kn)
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state, step_multilayer_land)


def _forcing(ncol):
    ones = jnp.ones(ncol)
    vals = dict(
        sw_down=400.0, lw_down=300.0, precip_total=1e-5, precip_snow=0.0,
        T_lowest=292.0, q_lowest=6e-3, u_lowest=3.0, v_lowest=1.0,
        p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.2, cos_zenith=0.6,
        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
    )
    return AtmToSurface(**{k: v * ones for k, v in vals.items()})


def test_coordination_kn_oracle():
    # kn convention: exponent kn*CI*x, x = cumulative LAI fraction, so the
    # per-unit-LAI diffuse extinction 0.72 maps to kn = f * 0.72 * LAI.
    got = float(coordination_kn(jnp.asarray(5.0), 0.35))
    assert got == pytest.approx(0.35 * 0.72 * 5.0, rel=1e-12)
    assert float(coordination_kn(jnp.asarray(0.0), 0.35)) == 0.0


def test_config_validates_profile():
    assert VALID_VCMAX_PROFILES == ("kn", "coordination")
    CanopyConfig(vcmax_profile="coordination").validate()
    with pytest.raises(ValueError, match="vcmax_profile"):
        CanopyConfig(vcmax_profile="light").validate()


def test_rt_canopy_total_drops_under_coordination_at_high_lai():
    """At LAI=5 the coordination exponent (0.35*0.72*5 = 1.26) is steeper
    than the default kn=0.3, so the canopy-total Vcmax integral must drop
    while the leaf-top value is held fixed (no-rescale is the design)."""
    args = [jnp.full((1,), v) for v in
            (300.0, 90.0, 320.0, 95.0, 12.0, 30.0, 5.0, 0.8, 0.08, 0.25,
             60.0, 30.0)]
    lai = args[6]
    out_kn = canopy_shortwave_rt(*args, jnp.full((1,), 0.3))
    out_co = canopy_shortwave_rt(*args, coordination_kn(lai, 0.35))
    tot_kn = float(out_kn.Vcmax25_C3Sun[0] + out_kn.Vcmax25_C3Sh[0])
    tot_co = float(out_co.Vcmax25_C3Sun[0] + out_co.Vcmax25_C3Sh[0])
    assert tot_co < tot_kn
    # strict light proportionality (f=1) is steeper still
    out_f1 = canopy_shortwave_rt(*args, coordination_kn(lai, 1.0))
    assert float(out_f1.Vcmax25_C3Sun[0] + out_f1.Vcmax25_C3Sh[0]) < tot_co


def test_step_switch_binds_and_typo_raises():
    """The two-leaf step must produce different canopy fluxes under the
    coordination profile (switch binds), and an unknown profile smuggled
    past validate() must raise at the flux entry, not silently run kn."""
    base = CanopyConfig(stomatal_model="medlyn")
    cfg_kn = MultiLayerLandConfig(surface_scheme=base.validate())
    cfg_co = MultiLayerLandConfig(
        surface_scheme=base._replace(vcmax_profile="coordination").validate())
    st = init_multilayer_land_state(2, cfg_kn)
    _, r_kn, _ = step_multilayer_land(st, _forcing(2), cfg_kn, U_min=1.0,
                                      dt=1800.0)
    _, r_co, _ = step_multilayer_land(st, _forcing(2), cfg_co, U_min=1.0,
                                      dt=1800.0)
    assert np.all(np.isfinite(np.asarray(r_co.lhflx)))
    assert not np.allclose(np.asarray(r_kn.lhflx), np.asarray(r_co.lhflx))

    cfg_bad = MultiLayerLandConfig(
        surface_scheme=base._replace(vcmax_profile="light"))
    with pytest.raises(ValueError, match="vcmax_profile"):
        step_multilayer_land(st, _forcing(2), cfg_bad, U_min=1.0, dt=1800.0)
