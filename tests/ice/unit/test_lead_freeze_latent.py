"""Lead-freeze latent-heat routing (``SeaIceConfig.lead_freeze_latent``).

The supercooling-latch fix (2026-07-21 Antarctic coastal ice-runaway bisect:
window → sea_ice.py → the lead-freeze SST gate; ll11 gate-revert BOUNDED):

* ``"charge"`` (default, legacy): the ocean is DEBITED L_f per kg of new lead
  ice — correct for the slab/implicit-ocean surrogates, where this term IS the
  atmospheric cooling delivered through the ice budget.
* ``"credit"``: freezing RELEASES L_f into the water (frazil convention) —
  required under a PROGNOSTIC ocean that separately receives the open-water
  atmospheric q_net: the legacy charge then double-counts the same heat
  (atm −Q through the lead AND the charge −Q again), the ocean supercools
  without bound, and the lead-freeze SST gate latches open.

These pin the EXACT ledger: the two modes differ ONLY in the sign of the
lead-freeze latent term (2·ΔV·ρ_i·L_f/dt apart), everything else bit-identical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.sea_ice import _thermo_v2, step_sea_ice

jax.config.update("jax_enable_x64", True)

_SHAPE = (4, 4)
_DT = 3600.0


def _freezing_forcing():
    """Cold, dark, calm atmosphere over partial ice — drives lead freezing."""
    s = _SHAPE
    from legoesm.coupler.coupler import AtmToSurface
    return AtmToSurface(
        sw_down=jnp.zeros(s), lw_down=jnp.full(s, 150.0),   # strongly cooling
        precip_total=jnp.zeros(s), precip_snow=jnp.zeros(s),
        T_lowest=jnp.full(s, 245.0), q_lowest=jnp.full(s, 1e-4),
        u_lowest=jnp.full(s, 2.0), v_lowest=jnp.zeros(s),
        p_lowest=jnp.full(s, 9.5e4), p_surface=jnp.full(s, 1e5),
        rho_lowest=jnp.full(s, 1.3), cos_zenith=jnp.zeros(s),
        co2_ppmv=jnp.full(s, 400.0),
        has_radiation=jnp.ones(s), has_precipitation=jnp.zeros(s),
    )


def _run(mode):
    cfg = SeaIceConfig(lead_freeze_latent=mode)
    s = _SHAPE
    h0 = jnp.full(s, 0.5)
    T0 = jnp.full(s, 250.0)
    conc0 = jnp.full(s, 0.5)          # 50% leads -> lead freeze active
    hs0 = jnp.zeros(s)
    S0 = jnp.full(s, 5.0)
    pa0 = jnp.zeros(s)
    pd0 = jnp.zeros(s)
    sst = jnp.full(s, constants.T_freeze_ocean)   # AT freezing -> gate open
    return _thermo_v2(h0, T0, conc0, hs0, S0, pa0, pd0,
                      _freezing_forcing(), sst, cfg, 1.0, _DT)


def test_modes_differ_by_exactly_twice_the_lead_latent():
    """extraction(charge) − extraction(credit) == 2·ΔV_lead·ρ_i·L_f/dt,
    with ΔV_lead > 0 (non-vacuous) and identical between the runs."""
    out_c = _run("charge")
    out_f = _run("credit")
    dv_c = np.asarray(out_c["delta_V_lead_freeze"])
    dv_f = np.asarray(out_f["delta_V_lead_freeze"])
    np.testing.assert_array_equal(dv_c, dv_f)      # sign never feeds back to ΔV
    assert float(dv_c.max()) > 0.0                 # lead freeze actually fired
    cfg = SeaIceConfig()
    expect = 2.0 * dv_c * cfg.rho_ice * cfg.L_f / _DT
    got = (np.asarray(out_c["ocean_heat_extraction"])
           - np.asarray(out_f["ocean_heat_extraction"]))
    np.testing.assert_allclose(got, expect, rtol=1e-12)


def test_everything_else_bit_identical_between_modes():
    """The sign routes ONLY the latent term: state evolution (h, conc, T,
    salt, freshwater) must be bit-identical between charge and credit."""
    out_c = _run("charge")
    out_f = _run("credit")
    for k in ("h", "T", "conc", "h_snow", "S_ice",
              "salt_flux_to_ocean", "freshwater_to_ocean",
              "delta_V_lead_freeze"):
        np.testing.assert_array_equal(
            np.asarray(out_c[k]), np.asarray(out_f[k]),
            err_msg=f"field {k} differs between charge/credit")


def test_credit_makes_lead_freeze_warm_the_ocean():
    """Under credit, the lead-freeze contribution is NEGATIVE extraction
    (ocean GAINS the released latent) — the self-limiting frazil feedback."""
    out_c = _run("charge")
    out_f = _run("credit")
    # charge >= credit everywhere; strictly greater where lead freeze fired.
    diff = (np.asarray(out_c["ocean_heat_extraction"])
            - np.asarray(out_f["ocean_heat_extraction"]))
    assert float(diff.min()) >= 0.0
    assert float(diff.max()) > 0.0


def test_default_is_legacy_charge():
    assert SeaIceConfig().lead_freeze_latent == "charge"


def test_slab_path_honors_the_literal_too():
    """The SLAB thermodynamics path is also reachable from prognostic-ocean
    callers (omip_sea_ice_surface_forcing, the coupled dynamic-ocean driver),
    so it must honor lead_freeze_latent as well (codex latch r1 HIGH: a
    slab-only always-charge made the credit a silent no-op there).  Same
    exact-ledger contract as the v2 test: charge − credit ==
    2·ρ_i·L_f·vlead_freeze, everything else identical."""
    from legoesm.ice.sea_ice import _step_slab
    from legoesm.ice.state import SeaIceState
    from legoesm.core.field import Field
    s = _SHAPE
    dims = ("lat", "lon")
    state = SeaIceState(
        h_ice=Field(data=jnp.full(s, 0.5), name="h_ice", dims=dims, units="m"),
        T_ice=Field(data=jnp.full(s, 250.0), name="T_ice", dims=dims,
                    units="K"),
        concentration=Field(data=jnp.full(s, 0.5), name="conc", dims=dims,
                            units="1"),
    )
    sst = jnp.full(s, constants.T_freeze_ocean)
    zeros = jnp.zeros(s)
    outs = {}
    for mode in ("charge", "credit"):
        cfg = SeaIceConfig(lead_freeze_latent=mode)
        _, resp = _step_slab(state, _freezing_forcing(), sst, zeros, zeros,
                             cfg, 1.0, _DT)
        outs[mode] = resp
    diff = (np.asarray(outs["charge"].ocean_heat_extraction)
            - np.asarray(outs["credit"].ocean_heat_extraction))
    assert float(diff.max()) > 0.0          # lead freeze fired (non-vacuous)
    assert float(diff.min()) >= 0.0         # charge >= credit everywhere
    # freshwater/salt identical (the sign routes only the latent term)
    np.testing.assert_array_equal(
        np.asarray(outs["charge"].freshwater_to_ocean),
        np.asarray(outs["credit"].freshwater_to_ocean))


def test_unknown_literal_raises_at_step_entry():
    cfg = SeaIceConfig(lead_freeze_latent="frazil")   # typo-like value
    with pytest.raises(ValueError, match="lead_freeze_latent"):
        step_sea_ice(None, None, None, None, None, cfg, 0.0, _DT)
