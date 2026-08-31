"""P-model full set on the big-leaf (SimpleSEB) lane.

Injection equivalence (the switch reproduces a hand-injected config), refusal
matrix (validate typos, switch on disabled stomata, Jarvis fallback, configured
C4 data, missing state), and default-path neutrality.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.p_model import acclimated_capacities, init_pmodel_acclim
from legoesm.land.stomata import StomataConfig
from legoesm.land.stomata_utils import compute_effective_beta


def _forcing(ncol):
    ones = jnp.ones(ncol)
    vals = dict(
        sw_down=500.0, lw_down=300.0, precip_total=0.0, precip_snow=0.0,
        T_lowest=295.0, q_lowest=8e-3, u_lowest=3.0, v_lowest=0.0,
        p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.2, cos_zenith=0.7,
        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
    )
    return AtmToSurface(**{k: v * ones for k, v in vals.items()})


def _cfg(**stomata_kw):
    return MultiLayerLandConfig(
        stomata=StomataConfig(enabled=True, stomata_model="medlyn",
                              **stomata_kw),
        carbon=CarbonConfig(scheme="differland"),
    )


def _acclim(ncol, cfg):
    return init_pmodel_acclim(ncol, t_init_K=295.0, ps_init_pa=101325.0,
                              cfg=cfg.p_model)


def test_switch_reproduces_hand_injected_config():
    ncol = 2
    cfg = _cfg(capacity_scheme="p_model", g1_source="p_model")
    carbon = init_carbon_state((ncol,), cfg.carbon)
    acclim = _acclim(ncol, cfg)
    beta_a, gpp_a, _ = compute_effective_beta(
        jnp.full(ncol, 295.0), _forcing(ncol), jnp.ones(ncol), cfg, carbon,
        1800.0, pmodel_acclim=acclim)
    caps = acclimated_capacities(acclim, cfg.p_model)
    tgc = acclim.t_mean_K - constants.T_freeze
    from legoesm.land import stomata as sto
    hand = cfg._replace(stomata=cfg.stomata._replace(
        Vc_max25=caps.vcmax25_leaf, g1_med=caps.g1_kpa,
        capacity_scheme="prescribed", g1_source="table"))
    # Hand path: same capacities + rjv25 + growth T via the solver directly.
    LAI = carbon.C_fol / hand.carbon.LCMA
    leaf = sto.solve_coupled_farquhar_ci(
        jnp.full(ncol, 295.0), _forcing(ncol).sw_down, 400.0,
        _forcing(ncol).q_lowest, _forcing(ncol).p_surface, LAI,
        jnp.ones(ncol), hand.stomata, 0.0, rjv25=caps.rjv25, TgC_C=tgc)
    beta_b = sto.compute_stomatal_beta(leaf.gs, LAI, jnp.ones(ncol), hand.stomata)
    np.testing.assert_allclose(np.asarray(beta_a), np.asarray(beta_b), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(gpp_a), np.asarray(leaf.gpp), rtol=1e-6)


def test_defaults_bit_identical_passthrough():
    cfg = MultiLayerLandConfig()  # stomata disabled, everything default
    beta, gpp, sif = compute_effective_beta(
        jnp.full(2, 290.0), _forcing(2), jnp.full(2, 0.6), cfg, None, 1800.0)
    np.testing.assert_array_equal(np.asarray(beta), np.full(2, 0.6))
    assert gpp is None and sif is None


def test_validate_matrix():
    with pytest.raises(ValueError, match="capacity_scheme"):
        StomataConfig(enabled=True, capacity_scheme="pmodel").validate()
    with pytest.raises(ValueError, match="medlyn"):
        StomataConfig(enabled=True, stomata_model="ball_berry",
                      g1_source="p_model").validate()
    with pytest.raises(ValueError, match="inert"):
        StomataConfig(enabled=False, capacity_scheme="p_model").validate()
    StomataConfig(enabled=True, stomata_model="leuning").validate()
    # capacities + leuning stomata composes (only the g1 source needs medlyn)
    StomataConfig(enabled=True, stomata_model="leuning",
                  capacity_scheme="p_model").validate()


def test_jarvis_fallback_with_switch_raises():
    cfg = MultiLayerLandConfig(
        stomata=StomataConfig(enabled=True, stomata_model="medlyn",
                              capacity_scheme="p_model"),
        carbon=CarbonConfig(scheme="none"))
    with pytest.raises(ValueError, match="Jarvis|coupled Farquhar"):
        compute_effective_beta(
            jnp.full(1, 295.0), _forcing(1), jnp.ones(1), cfg, None, 1800.0,
            pmodel_acclim=_acclim(1, cfg))


def test_c4_column_uses_the_c4_optimum():
    """fC4=1 column: the injected capacity/slope equal the rpmodel-c4 optimum
    (endpoint-exact; the kernel's single-capacity blend covers 0<fC4<1)."""
    from legoesm.land.p_model import acclimated_capacities_c4
    from legoesm.land.surface_params import LandSurfaceParams
    from legoesm.land import stomata as sto
    cfg = _cfg(capacity_scheme="p_model", g1_source="p_model")
    carbon = init_carbon_state((1,), cfg.carbon)
    acclim = _acclim(1, cfg)
    lp = LandSurfaceParams(
        albedo_veg=jnp.full(1, 0.15), emissivity=jnp.full(1, 0.98),
        z0=jnp.full(1, 0.1), W_max=jnp.full(1, 150.0),
        C_soil=jnp.full(1, 2e6), d_soil=jnp.full(1, 1.0),
        root_depth=jnp.full(1, 1.0), theta_wp=jnp.full(1, 0.15),
        theta_fc=jnp.full(1, 0.3), Vc_max25=jnp.full(1, 60.0),
        LCMA=jnp.full(1, 50.0), g1=jnp.full(1, 9.0), fC4=jnp.ones(1))
    beta_a, gpp_a, _ = compute_effective_beta(
        jnp.full(1, 295.0), _forcing(1), jnp.ones(1), cfg, carbon,
        1800.0, land_params=lp, pmodel_acclim=acclim)
    caps4 = acclimated_capacities_c4(acclim, cfg.p_model)
    tgc = acclim.t_mean_K - constants.T_freeze
    hand = cfg.stomata._replace(
        Vc_max25=caps4.vcmax25_c4_leaf, g1_med=caps4.g1_c4_kpa,
        capacity_scheme="prescribed", g1_source="table")
    LAI = carbon.C_fol / lp.LCMA
    leaf = sto.solve_coupled_farquhar_ci(
        jnp.full(1, 295.0), _forcing(1).sw_down, 400.0,
        _forcing(1).q_lowest, _forcing(1).p_surface, LAI,
        jnp.ones(1), hand._replace(g1_bb=lp.g1), 1.0,
        rjv25=None, TgC_C=tgc)
    np.testing.assert_allclose(np.asarray(gpp_a), np.asarray(leaf.gpp),
                               rtol=1e-5)


def test_switch_without_state_raises():
    cfg = _cfg(capacity_scheme="p_model")
    carbon = init_carbon_state((1,), cfg.carbon)
    with pytest.raises(ValueError, match="acclimation state"):
        compute_effective_beta(
            jnp.full(1, 295.0), _forcing(1), jnp.ones(1), cfg, carbon, 1800.0)
