"""stomatal_model='pmodel_chi' (step E): ci = chi(xi) Ca at every step.

JULES stomata_pmodel / pyrealm SubdailyPModel: chi = Gamma*/Ca + (1 -
Gamma*/Ca) xi / (xi + sqrt(D)) from the acclimated (lagged) xi and the
current VPD; gs = 1.6 An / (Ca - ci); closed stomata (An <= 0 or gs below
it) take the residual conductance (JULES glmin; legoESM b0). Pinned:

* open leaf: the diffusion ci equals chi Ca exactly and gs = 1.6 An /
  (Ca (1 - chi)); chi does not depend on An (hence not on soil stress);
* closed leaf: gs = b0;
* configuration refusals;
* a two-leaf step runs, and sunlit and shaded leaves share almost the same
  ci/ca (they differ only through Gamma*(T_leaf)), unlike a slope model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land.canopy.energy_balance import _compute_gs_and_ci
from legoesm.land.canopy.photosynthesis import co2_compensation_point
from legoesm.land.p_model import chi_core

_ENTRY_X64 = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", _ENTRY_X64)


def _gs_ci(An, xi=60.0, vpd=1200.0, Tf=constants.T_freeze + 25.0, b0=0.01):
    a = lambda v: jnp.asarray(v, dtype=jnp.float64)  # noqa: E731
    rs, gs, ci = _compute_gs_and_ci(a(An), a(0.6), a(vpd), a(410.0), a(Tf), a(1e5),
                                    a(xi), a(b0), "pmodel_chi")
    return rs, gs, ci


@pytest.mark.parametrize("An", [2.0, 10.0, 25.0])
def test_open_leaf_ci_equals_chi_ca(An):
    Tf = constants.T_freeze + 25.0
    chi = float(chi_core(60.0, co2_compensation_point(jnp.asarray(Tf)) / 410.0,
                         np.sqrt(1200.0)))
    _, _, ci = _gs_ci(An)
    # the smooth max(gs_open, b0) adds O(w exp(-(gs_open-b0)/w)) to gs
    assert float(ci) == pytest.approx(chi * 410.0, rel=1e-6)
    # the same ci for every An: chi depends on xi, D and T only
    assert 0.5 < chi < 0.95


def test_low_chi_is_not_clipped():
    """chi < 0.5 (C4-like small xi in dry air) must still give ci = chi Ca:
    no 0.5 Ca floor under pmodel_chi (codex review round 1)."""
    Tf = constants.T_freeze + 30.0
    chi = float(chi_core(15.0, co2_compensation_point(jnp.asarray(Tf)) / 410.0,
                         np.sqrt(3000.0)))
    assert chi < 0.5
    _, _, ci = _gs_ci(20.0, xi=15.0, vpd=3000.0, Tf=Tf)
    assert float(ci) == pytest.approx(chi * 410.0, rel=1e-6)


def test_closed_leaf_takes_residual_conductance():
    from legoesm.land.canopy.energy_balance import gs_mol_per_ms
    Tf, Ps = constants.T_freeze + 25.0, 1e5
    _, gs, _ = _gs_ci(-1.0, b0=0.02)
    assert float(gs) * float(gs_mol_per_ms(jnp.asarray(Tf), jnp.asarray(Ps))) == pytest.approx(0.02)


def test_drier_air_lowers_chi_and_conductance():
    _, gs_wet, ci_wet = _gs_ci(10.0, vpd=500.0)
    _, gs_dry, ci_dry = _gs_ci(10.0, vpd=3000.0)
    assert float(ci_dry) < float(ci_wet) and float(gs_dry) < float(gs_wet)


def test_config_refusals():
    from legoesm.land.canopy.config import CanopyConfig
    with pytest.raises(ValueError, match="pmodel_chi"):
        CanopyConfig(stomatal_model="pmodel_chi").validate()
    with pytest.raises(ValueError, match="pmodel_chi"):
        CanopyConfig(stomatal_model="pmodel_chi", capacity_scheme="p_model",
                     g1_source="p_model").validate()
    ok = CanopyConfig(stomatal_model="pmodel_chi", capacity_scheme="p_model",
                      g1_source="table").validate()
    assert ok.stomatal_model == "pmodel_chi"


def test_two_leaf_step_sunlit_and_shaded_share_chi():
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.canopy.config import CanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.multilayer_land import (
        init_multilayer_land_state, step_multilayer_land_with_diagnostics)
    n = 2
    f = AtmToSurface(**{k: jnp.full(n, v) for k, v in dict(
        sw_down=600.0, lw_down=330.0, precip_total=0.0, precip_snow=0.0,
        T_lowest=constants.T_freeze + 24.0, q_lowest=0.009, u_lowest=2.5,
        v_lowest=0.0, p_lowest=100000.0, p_surface=101000.0, rho_lowest=1.2,
        cos_zenith=0.8, co2_ppmv=410.0, has_radiation=1.0,
        has_precipitation=1.0).items()})
    out = {}
    for sm, g1 in (("pmodel_chi", "table"), ("medlyn", "p_model")):
        cfg = MultiLayerLandConfig(surface_scheme=CanopyConfig(
            stomatal_model=sm, capacity_scheme="p_model", g1_source=g1).validate())
        st = init_multilayer_land_state(n, cfg, T_init=constants.T_freeze + 22.0)
        _, _, _, o = step_multilayer_land_with_diagnostics(st, f, cfg, U_min=1.0, dt=1800.0)
        sun, sh = np.asarray(o.ci_ca_sun), np.asarray(o.ci_ca_sh)
        assert np.all(np.isfinite(sun)) and np.all(np.isfinite(np.asarray(o.gpp)))
        out[sm] = np.abs(sun - sh)
    assert np.all(out["pmodel_chi"] < 0.02)
    assert np.all(out["pmodel_chi"] < out["medlyn"])


def _production_gs_mol(gs_open, b0, xi=60.0, vpd=1000.0,
                       Tf=constants.T_freeze + 20.0, Ca=410.0):
    """Conductance [mol m-2 s-1] returned by the PRODUCTION pmodel_chi branch
    for the An that gives the requested gs_open (codex review round 3)."""
    from legoesm.land.canopy.energy_balance import gs_mol_per_ms
    chi = float(chi_core(xi, co2_compensation_point(jnp.asarray(Tf)) / Ca, np.sqrt(vpd)))
    An = gs_open * Ca * (1.0 - chi) / 1.6
    _, gs, _ = _gs_ci(An, xi=xi, vpd=vpd, Tf=Tf, b0=b0)
    return float(gs) * float(gs_mol_per_ms(jnp.asarray(Tf), jnp.asarray(1e5)))


def test_smooth_switch_departure_bounds():
    """Production conductance against the documented bounds: b0 + w ln 2 at
    gs_open = b0 (a hard max would give b0), < 5e-8 mol m-2 s-1 above
    max(gs_open, b0) at 10 w either side, w ln 2 for a closed leaf with b0 = 0
    (above the 1e-4 floor)."""
    from legoesm.land.canopy import energy_balance as eb
    w, b0 = eb._PMODEL_CHI_SWITCH_MOL, 0.01
    assert _production_gs_mol(b0, b0) - b0 == pytest.approx(w * np.log(2.0), rel=1e-6)
    for d in (10.0, -10.0):
        dep = _production_gs_mol(b0 + d * w, b0) - max(b0 + d * w, b0)
        assert 0.0 < dep < 5e-8
    assert _production_gs_mol(0.0, 0.0) == pytest.approx(w * np.log(2.0), rel=1e-6)


def test_open_leaf_ci_close_to_switch_is_bounded():
    """0 <= ci - chi Ca <= (1 - chi) Ca (gs - gs_open) / gs_open, with
    gs - gs_open <= w exp(-10) at gs_open = b0 + 10 w and w ln 2 at b0."""
    from legoesm.land.canopy import energy_balance as eb
    w = eb._PMODEL_CHI_SWITCH_MOL
    Tf, Ca, xi, vpd, b0 = constants.T_freeze + 20.0, 410.0, 60.0, 1000.0, 0.01
    chi = float(chi_core(xi, co2_compensation_point(jnp.asarray(Tf)) / Ca, np.sqrt(vpd)))
    for gs_open, dgs_max in ((b0 + 10 * w, w * np.exp(-10.0)), (b0, w * np.log(2.0))):
        An = gs_open * Ca * (1.0 - chi) / 1.6       # An that gives this gs_open
        _, _, ci = _gs_ci(An, xi=xi, vpd=vpd, Tf=Tf, b0=b0)
        err = float(ci) - chi * Ca
        assert 0.0 <= err <= (1.0 - chi) * Ca * dgs_max / gs_open * (1 + 1e-9)
