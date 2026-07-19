"""Invariants on `legoesm.constants` — fail loudly if a value drifts."""

from __future__ import annotations

import math

from legoesm import constants


def test_M_dry_matches_M_air():
    assert constants.M_dry == constants.M_air * 1e-3


def test_M_h2o_matches_M_H2O():
    assert constants.M_h2o == constants.M_H2O * 1e-3


def test_cpd_cvd_Rd_identity():
    assert abs(constants.c_pd - constants.c_vd - constants.R_d) < 1e-9


def test_kappa_derived():
    assert abs(constants.kappa - constants.R_d / constants.c_pd) < 1e-12


def test_epsilon_derived():
    assert abs(constants.epsilon - constants.R_d / constants.R_v) < 1e-12


def test_avogadro_codata_2019_exact():
    assert constants.N_A == 6.02214076e23


def test_M_air_codata():
    assert constants.M_air == 28.96546


def test_all_constants_finite_and_positive():
    names = (
        "g", "Omega", "R_earth",
        "R_d", "R_v", "c_pd", "c_vd", "c_pv", "c_pw", "c_pi",
        "L_v", "L_s", "L_f",
        "rho_water", "rho_ice", "rho_air", "rho_ocean", "c_sw",
        "T_freeze", "T_freeze_ocean", "T_freshwater_max_density",
        "rho_freshwater_curvature",
        "sigma_sb", "S_0",
        "emissivity_ocean", "emissivity_ice", "emissivity_land",
        "kappa_vk", "nu_air",
        "M_air", "M_CO2", "M_H2O",
        "M_dry", "M_o3", "M_h2o",
        "N_A",
        "p_ref", "H_MEAN",
    )
    for name in names:
        val = float(getattr(constants, name))
        assert math.isfinite(val), f"{name}={val} is not finite"
        assert val > 0, f"{name}={val} is not positive"


def test_rrtmgp_constants_re_export_canonical():
    """Phase B ensures rrtmgp re-exports the canonical legoesm.constants values."""
    from legoesm.atmosphere.physics.radiation.rrtmgp import constants as rrtmgp_constants

    assert rrtmgp_constants.G == constants.g
    assert rrtmgp_constants.R_D == constants.R_d
    assert rrtmgp_constants.R_V == constants.R_v
    assert rrtmgp_constants.CP_D == constants.c_pd
    # iter-47 dropped the CV_D (c_vd) and CP_V (c_pv) re-exports from rrtmgp
    # constants — they are moist-thermo dycore values, not radiation constants;
    # the canonical definitions live in legoesm.constants.
    assert rrtmgp_constants.DRY_AIR_MOL_MASS == constants.M_dry
    assert rrtmgp_constants.WATER_MOL_MASS == constants.M_h2o
    assert rrtmgp_constants.AVOGADRO == constants.N_A


def test_surface_emissivity_defaults_reference_constants():
    """Config emissivity defaults MUST equal ``legoesm.constants`` (single
    source of truth).  Tripwire for the drift that had ``ExperimentConfig``
    ice=0.95 (vs 0.97) and land=0.96 (vs 0.95) — and a builder that dropped
    ``emissivity_land`` entirely — silently diverge from the canonical values.
    """
    from legoesm.driver.config import ExperimentConfig
    from legoesm.land.config import LandConfig, MultiLayerLandConfig

    exp = ExperimentConfig()
    assert exp.emissivity_ice == constants.emissivity_ice
    assert exp.emissivity_land == constants.emissivity_land
    assert exp.sfc_emissivity == constants.emissivity_ocean
    assert LandConfig().emissivity_land == constants.emissivity_land
    assert MultiLayerLandConfig().emissivity_land == constants.emissivity_land


def test_builder_forwards_emissivity_land_to_pipeline():
    """The pipeline builder MUST forward ``config.emissivity_land`` (regression
    guard for the dropped-field bug this reconciliation fixed: land emissivity
    was silently ignored by the builder and fell back to the constructor
    default).  A default-only check would not catch re-dropping the kwarg.
    """
    import jax.numpy as jnp

    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    sentinel = 0.42  # non-default, distinct from any emissivity constant
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray",
        emissivity_land=sentinel,
    )
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    pipe = build_physics_pipeline(grid, sigma, cfg)
    assert pipe.emissivity_land == sentinel
