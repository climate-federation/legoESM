"""C3 light response of electron transport: CLM vs P-model phi0(T) (step D).

The P-model sizes Vcmax/Jmax with phi0(T) = kphio x Bernacchi (2003); the
CLM kernel's light response has a temperature-independent initial yield of
0.5 * 0.85 / 4 = 0.10625 mol C per mol photon. Under
light_response='pmodel_phi0' the kernel uses J = 4 phi0(T) I /
sqrt(1 + (4 phi0 I / Jmax)^2) instead. Pinned here:

* the initial light-use slope ratio P-model/CLM equals phi0(T)/0.10625 at a
  cool and a warm leaf temperature (the temperature dependence is real);
* at saturating light both forms approach the same Jmax-limited rate;
* the default (phi0_ref=None) is the CLM form; finite gradients in the dark;
* configuration refusals.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land.canopy.photosynthesis import c3_assimilation
from legoesm.land.p_model import phi0_c3

_ENTRY_X64 = jax.config.read("jax_enable_x64")
CLM_INITIAL_YIELD = 0.5 * 0.85 / 4.0     # mol C per mol photon (CLM5 form)


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", _ENTRY_X64)


def _a(T_K, apar, phi0_ref=None, vcmax25=500.0):
    """Gross assimilation of a Rubisco-rich leaf (light-limited regime)."""
    T = jnp.asarray(T_K, dtype=jnp.float64)
    return c3_assimilation(T, jnp.asarray(300.0), jnp.asarray(apar, dtype=jnp.float64),
                           jnp.asarray(vcmax25), jnp.asarray(20.0),
                           rjv25=jnp.asarray(2.0), phi0_ref=phi0_ref).a_gross


@pytest.mark.parametrize("t_c", [10.0, 28.0])
def test_initial_light_use_slope_ratio_equals_phi0_over_clm_yield(t_c):
    T = constants.T_freeze + t_c
    slope = lambda ref: jax.grad(lambda I: _a(T, I, ref))(1e-3)  # noqa: E731
    ratio = float(slope(jnp.asarray(0.125)) / slope(None))
    expected = float(phi0_c3(jnp.asarray(T), 0.125)) / CLM_INITIAL_YIELD
    assert ratio == pytest.approx(expected, rel=2e-3)


def test_light_saturated_rates_agree():
    T = constants.T_freeze + 25.0
    a_clm = float(_a(T, 2e5, None))
    a_pm = float(_a(T, 2e5, jnp.asarray(0.125)))
    assert a_pm == pytest.approx(a_clm, rel=0.02)


def test_default_is_clm_form_and_dark_gradients_finite():
    """Independent pin of the default (CLM) light response: the initial
    light-use slope of a Rubisco-rich leaf is 0.5 * 0.85 / 4 * mj with
    mj = (Ci - Gamma*)/(Ci + 2 Gamma*) (no reference to the code path)."""
    from legoesm.land.canopy.photosynthesis import co2_compensation_point
    T = constants.T_freeze + 20.0
    gs_ = float(co2_compensation_point(jnp.asarray(T)))
    mj = (300.0 - gs_) / (300.0 + 2.0 * gs_)
    slope = float(jax.grad(lambda I: _a(T, I, None))(1e-3))
    assert slope == pytest.approx(CLM_INITIAL_YIELD * mj, rel=2e-3)
    g = jax.grad(lambda I, ref: _a(T, I, ref), argnums=(0, 1))(
        0.0, jnp.asarray(0.0818))
    assert all(np.isfinite(float(x)) for x in g)


def test_config_refusals():
    from legoesm.land.canopy.config import CanopyConfig
    with pytest.raises(ValueError, match="unknown light_response"):
        CanopyConfig(light_response="smith").validate()
    with pytest.raises(ValueError, match="requires capacity_scheme='p_model'"):
        CanopyConfig(light_response="pmodel_phi0").validate()
    ok = CanopyConfig(light_response="pmodel_phi0", capacity_scheme="p_model",
                      g1_source="p_model", stomatal_model="medlyn").validate()
    assert ok.light_response == "pmodel_phi0"
    assert CanopyConfig().light_response == "clm"


def test_two_leaf_step_runs_and_lowers_light_limited_gpp():
    """End to end through the two-leaf canopy at dim light: with kphio =
    0.0818 the P-model initial yield at 20 C (~0.06) is below CLM's 0.106,
    so light-limited GPP must fall when the switch is on."""
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.multilayer_land import (
        init_multilayer_land_state, step_multilayer_land_with_diagnostics)
    from legoesm.land.canopy.config import CanopyConfig
    from legoesm.core.coupling_fields import AtmToSurface
    n = 2
    # every AtmToSurface field (co2 included) as an (ncol,) array
    f = AtmToSurface(**{k: jnp.full(n, v) for k, v in dict(
        sw_down=60.0, lw_down=320.0, precip_total=0.0, precip_snow=0.0,
        T_lowest=constants.T_freeze + 20.0, q_lowest=0.008, u_lowest=2.0,
        v_lowest=0.0, p_lowest=100000.0, p_surface=101000.0, rho_lowest=1.2,
        cos_zenith=0.5, co2_ppmv=410.0, has_radiation=1.0,
        has_precipitation=1.0).items()})
    gpp = {}
    for light in ("clm", "pmodel_phi0"):
        cfg = MultiLayerLandConfig(surface_scheme=CanopyConfig(
            stomatal_model="medlyn", capacity_scheme="p_model",
            g1_source="p_model", light_response=light).validate())
        st = init_multilayer_land_state(n, cfg, T_init=constants.T_freeze + 20.0)
        _, _, _, out = step_multilayer_land_with_diagnostics(
            st, f, cfg, U_min=1.0, dt=1800.0)
        gpp[light] = np.asarray(out.gpp)
        assert np.all(np.isfinite(gpp[light]))
    assert np.all(gpp["pmodel_phi0"] < gpp["clm"])
