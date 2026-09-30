"""NEMO SI3 lead heat budget (``SeaIceConfig.lead_freeze_source="nemo_qlead"``).

icesbc.F90:357-405 (NEMO 5.0.1): zqld = (1-A)*q_open*dt, zqfr = rho0*cp*
dz*(Tf - SST), qlead = MIN(0, zqld - zqfr).  Lead ice forms from qlead and its
latent heat returns to the ocean, so the coupler's (1-A)*q_open cooling nets to
at most the freezing deficit and a supercooled cell refreezes toward Tf.
Each case below is an analytic ledger the kernel must reproduce exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.sea_ice import _thermo_v2

jax.config.update("jax_enable_x64", True)

_SHAPE = (3, 3)
_DT = 3600.0
_DZ = 1.0
_Q_OPEN = -300.0          # W/m2 per open-water area (winter lead)


def _forcing():
    s = _SHAPE
    return AtmToSurface(
        sw_down=jnp.zeros(s), lw_down=jnp.full(s, 150.0),
        precip_total=jnp.zeros(s), precip_snow=jnp.zeros(s),
        T_lowest=jnp.full(s, 245.0), q_lowest=jnp.full(s, 1e-4),
        u_lowest=jnp.full(s, 2.0), v_lowest=jnp.zeros(s),
        p_lowest=jnp.full(s, 9.5e4), p_surface=jnp.full(s, 1e5),
        rho_lowest=jnp.full(s, 1.3), cos_zenith=jnp.zeros(s),
        co2_ppmv=jnp.full(s, 400.0),
        has_radiation=jnp.ones(s), has_precipitation=jnp.zeros(s),
    )


def _run(source, conc, sst_offset_K, q_open=_Q_OPEN):
    cfg = SeaIceConfig(lead_freeze_source=source)
    s = _SHAPE
    sst = jnp.full(s, cfg.T_freeze_ocean + sst_offset_K)
    return cfg, _thermo_v2(
        jnp.full(s, 0.5), jnp.full(s, 250.0), jnp.full(s, conc),
        jnp.zeros(s), jnp.full(s, 5.0), jnp.zeros(s), jnp.zeros(s),
        _forcing(), sst, cfg, 1.0, _DT,
        q_open_top=None if q_open is None else jnp.full(s, q_open),
        ocean_dz_top_m=_DZ)


def _lead_term(cfg, out):
    return np.asarray(out["delta_V_lead_freeze"]) * cfg.rho_ice * cfg.L_f / _DT


def test_above_freezing_forms_no_lead_ice():
    """zqld - zqfr > 0: the open-water cooling only brings the cell toward
    Tf, no ice (the legacy ice_skin path would also be gated off here)."""
    cfg, out = _run("nemo_qlead", conc=0.5, sst_offset_K=+0.5)
    assert float(np.abs(np.asarray(out["delta_V_lead_freeze"])).max()) == 0.0


def test_at_freezing_lead_ice_returns_exactly_the_open_water_cooling():
    """zqfr = 0 -> qlead = zqld = (1-A)*q_open*dt; the ocean lead term is
    qlead/dt = (1-A)*q_open (a GAIN cancelling the coupler's open-water
    cooling), and the ice volume is -qlead/(rho_i L_f)."""
    cfg, out = _run("nemo_qlead", conc=0.5, sst_offset_K=0.0)
    cfg0, out0 = _run("nemo_qlead", conc=0.5, sst_offset_K=0.0, q_open=0.0)
    dv = np.asarray(out["delta_V_lead_freeze"])
    expect_dv = -(0.5 * _Q_OPEN * _DT) / (cfg.rho_ice * cfg.L_f)
    np.testing.assert_allclose(dv, expect_dv, rtol=1e-12)
    # only the lead term differs between q_open = -300 and q_open = 0
    d_ext = (np.asarray(out["ocean_heat_extraction"])
             - np.asarray(out0["ocean_heat_extraction"]))
    np.testing.assert_allclose(d_ext, 0.5 * _Q_OPEN, rtol=1e-12)  # = -150
    assert float(np.abs(np.asarray(out0["delta_V_lead_freeze"])).max()) == 0.0


def test_supercooled_full_cover_frazil_restores_freezing():
    """A = 1 (no leads), SST = Tf - 0.5: zqld = 0, zqfr > 0 -> qlead = -zqfr.
    Ice forms from the supercooling and the ocean GAINS exactly rho0 cp dz dT
    per step — the frazil closure the legacy path lacks at A = 1."""
    dT = 0.5
    cfg, out = _run("nemo_qlead", conc=1.0, sst_offset_K=-dT)
    cfg0, out0 = _run("nemo_qlead", conc=1.0, sst_offset_K=0.0)
    zqfr = constants.rho_ocean * constants.c_p_seawater * _DZ * dT
    np.testing.assert_allclose(np.asarray(out["delta_V_lead_freeze"]),
                               zqfr / (cfg.rho_ice * cfg.L_f), rtol=1e-12)
    d_ext = (np.asarray(out["ocean_heat_extraction"])
             - np.asarray(out0["ocean_heat_extraction"]))
    # F_ocean is zero in both (SST <= Tf), so the difference is the lead term
    np.testing.assert_allclose(d_ext, -zqfr / _DT, rtol=1e-12)
    # the conc-1 cell keeps A = 1: the volume thickens the pack
    assert float(np.asarray(out["conc"]).max()) <= 1.0 + 1e-12


def test_ice_skin_ignores_the_new_inputs_and_debits_the_ocean():
    """Legacy path: kwargs inert (bit-identical with/without), and the lead
    latent is a positive extraction (the ocean is debited)."""
    cfg, out = _run("ice_skin", conc=0.5, sst_offset_K=0.0)
    _, out_none = _run("ice_skin", conc=0.5, sst_offset_K=0.0, q_open=None)
    for k in ("h", "conc", "ocean_heat_extraction", "delta_V_lead_freeze"):
        np.testing.assert_array_equal(np.asarray(out[k]),
                                      np.asarray(out_none[k]))
    assert float(np.asarray(out["delta_V_lead_freeze"]).max()) > 0.0
    _, out0 = _run("ice_skin", conc=0.5, sst_offset_K=+0.5)  # gate closed
    assert float(np.abs(np.asarray(out0["delta_V_lead_freeze"])).max()) == 0.0


def test_nemo_qlead_requires_its_inputs_and_unknown_literal_raises():
    with pytest.raises(ValueError, match="q_open_top"):
        _run("nemo_qlead", conc=0.5, sst_offset_K=0.0, q_open=None)
    with pytest.raises(ValueError, match="lead_freeze_source"):
        _run("frazil", conc=0.5, sst_offset_K=0.0)


def test_default_is_legacy_ice_skin():
    assert SeaIceConfig().lead_freeze_source == "ice_skin"


def test_top_layer_absorbed_fraction_is_a_fraction():
    """The lead budget's SW share: RGB (chl) and two-band paths both give a
    fraction in (0, 1) for a 1 m top cell, larger for a thicker cell."""
    from legoesm.ocean.physics.shortwave_penetration import (
        top_layer_absorbed_fraction,
    )
    f1 = np.asarray(top_layer_absorbed_fraction(1.0, chl_surface=jnp.full((4,), 0.1)))
    f5 = np.asarray(top_layer_absorbed_fraction(5.0, chl_surface=jnp.full((4,), 0.1)))
    assert f1.shape == (4,) and np.all((f1 > 0.05) & (f1 < 0.95))
    assert np.all(f5 > f1)
    g1 = np.asarray(top_layer_absorbed_fraction(jnp.ones((3,))))
    assert g1.shape == (3,) and np.all((g1 > 0.05) & (g1 < 0.95))
