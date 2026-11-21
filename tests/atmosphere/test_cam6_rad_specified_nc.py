"""Radiation droplet number under specified droplets (cam6_rad_specified_nc).

With prognostic droplets off, ZM detrainment still books droplet number into
the N_c tracer and nothing removes it, and the radiation read that tracer.
With the flag on the radiation takes MG2's own specified number, as CAM6 does
with micro_mg_nccons: MG2 leaves ncnst/rho * lcldm in the state
(micro_mg2_0.F90:2824, lcldm = max(alst, mincld)) and REL takes it in cloud
over max(ast, mincld) (micro_mg_cam.F90:2539).  The REL routine itself is
pinned by the reff_incloud oracle.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics._shared import compute_rho
from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.radiation import integration as radint
from tests.unit import _cam6_rad_columns as cc

jax.config.update("jax_enable_x64", True)

MICRO = dict(mg2_ice_dcs_m=500e-6, mg2_ice_slope_log_width=0.01)


def _props(cfg, c, n_cloud, **kw):
    return compute_cloud_properties(
        c["T"], c["p_full"], c["q_v"], c["p_half"][:, 1:] - c["p_half"][:, :-1],
        cfg, q_cloud=c["q_c"], q_ice=c["q_i"], n_ice=c["n_i"], n_cloud=n_cloud,
        cloud_fraction_override=c["alst"], lat=c["lat"],
        conv_mass_flux_up=c["conv_mass_flux_up"], conv_icwmr=c["conv_icwmr"],
        p_half=c["p_half"], clubb_config=cc.CLUBB, **MICRO, **kw)


def test_radius_ignores_the_tracer_and_uses_mg2_ncnst():
    c = cc.columns(6)
    on = cc.cloud_config(cam6_rad_specified_nc=True)
    a = _props(on, c, c["n_c_tracer"], n_cloud_specified=1e8)
    b = _props(on, c, 50.0 * c["n_c_tracer"], n_cloud_specified=1e8)
    np.testing.assert_array_equal(np.asarray(a.r_eff_liq), np.asarray(b.r_eff_liq))
    # the number handed to REL: ncnst/rho_dry * max(alst,1e-4)/max(ast,1e-4)
    # per kg, i.e. that times the moist density per volume, in cloud
    T, p, qv = c["T"], c["p_full"], c["q_v"]
    ncic = (1e8 / compute_rho(T, p) * jnp.maximum(a.alst, 1e-4)
            / jnp.maximum(a.ast, 1e-4))
    ref = _props(cc.cloud_config(), c, ncic * compute_rho(T, p, qv), n_cloud_in_cloud=True)
    np.testing.assert_allclose(np.asarray(a.r_eff_liq), np.asarray(ref.r_eff_liq),
                               rtol=1e-12, atol=0)
    # ... and the ice radius is left as it was (tracer route unchanged)
    off = _props(cc.cloud_config(), c, c["n_c_tracer"])
    np.testing.assert_array_equal(np.asarray(a.r_eff_ice), np.asarray(off.r_eff_ice))


def test_default_still_reads_the_tracer():
    c = cc.columns(6)
    a = _props(cc.cloud_config(), c, c["n_c_tracer"])
    b = _props(cc.cloud_config(), c, 50.0 * c["n_c_tracer"])
    liq = np.asarray(c["q_c"]) > 1e-14
    assert not np.allclose(np.asarray(a.r_eff_liq)[liq], np.asarray(b.r_eff_liq)[liq])


def test_specified_number_is_mg2_nc0_and_refused_for_other_droplets():
    rc = cc.radiation_config(cc.cloud_config(cam6_rad_specified_nc=True))
    assert radint._specified_droplet_number(rc, MorrisonConfig(), False) == MorrisonConfig().Nc_0
    assert radint._specified_droplet_number(
        cc.radiation_config(cc.cloud_config()), MorrisonConfig(), False) is None
    for leaf, aer in ((MorrisonConfig(prognostic_droplet_number=True), False),
                      (MorrisonConfig(), True), (None, False)):
        with pytest.raises(ValueError):
            radint._specified_droplet_number(rc, leaf, aer)
    c = cc.columns(6)
    with pytest.raises(ValueError, match="n_cloud_specified"):
        _props(cc.cloud_config(cam6_rad_specified_nc=True), c, None)


def test_deck_key_reaches_the_cloud_config():
    from legoesm.driver.model_driver import _standalone_cloud_config
    from tests.unit.test_cam6_cloud_fraction import _exp_cfg
    cfg = _exp_cfg(microphysics="morrison", cloud_cam6_rad_specified_nc=True)
    cfg.validate_strict()
    assert _standalone_cloud_config(cfg, "cam6_clubb").cam6_rad_specified_nc
    with pytest.raises(ValueError, match="specified droplets"):
        _exp_cfg(cloud_cam6_rad_specified_nc=True).validate_strict()
