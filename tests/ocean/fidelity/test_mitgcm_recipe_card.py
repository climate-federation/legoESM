"""Tests for the MITgcm recipe card + the MITgcm<->legoESM wiring diagram
(``mitgcm_recipe.py``)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")


def test_canonical_config_builds_a_valid_model():
    """The shared MITgcm-faithful block choices build a config the model accepts."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity import mitgcm_baroclinic_gyre_recipe as bg
    from legoesm.ocean.fidelity.mitgcm_recipe import mitgcm_canonical_ocean_config

    r = bg.build_baroclinic_gyre_recipe()        # reuse a real EOS + physics block
    c = r.config
    cfg = mitgcm_canonical_ocean_config(
        A_h=c.A_h, K_h=c.K_h, A_v=c.A_v, K_v=c.K_v,
        eos_linear=c.eos_linear, physics=c.physics,
    )
    # canonical card pins MITgcm's actual algorithm
    assert cfg.barotropic_solver == "implicit_unsplit"
    assert cfg.momentum_advection == "flux_form"
    assert cfg.coriolis_scheme == "explicit_ab2"
    assert cfg.lateral_viscosity_operator == "flux_divergence"
    assert cfg.constants.rho_0 == 999.8
    # builds a model (passes the model's config validation)
    LatLonCGridOceanModel(r.geometry, r.z_coord, cfg)


def test_block_mapping_references_real_config_fields():
    """Every legoESM config field named in the wiring diagram must be a real field
    (a top-level LatLonCGridOceanConfig attribute) — keeps the diagram honest."""
    from legoesm.ocean.fidelity.mitgcm_recipe import MITGCM_BLOCK_MAPPING
    from legoesm.ocean.state import LatLonCGridOceanConfig

    cfg = LatLonCGridOceanConfig()
    # TOP-LEVEL config fields the wiring diagram references (surface_forcing/sponge
    # are physics sub-fields / step args, not top-level — excluded here).
    top_level = {
        "eos", "eos_linear", "momentum_advection", "momentum_flux_scheme",
        "ke_gradient_scheme", "coriolis_scheme", "barotropic_solver",
        "tracer_advection", "A_h", "lateral_viscosity_operator", "K_h",
        "A_v", "K_v", "implicit_vertical_mixing", "gm_redi", "lateral_side_bc",
        "outer_integrator", "ab2_epsilon", "physics", "constants",
    }
    assert len(MITGCM_BLOCK_MAPPING) >= 15
    for concept, field, note in MITGCM_BLOCK_MAPPING:
        assert concept and note          # documented
        # every top-level token the field column names must be a real config field
        for tok in field.replace("+", " ").replace("/", " ").replace(".", " ").split():
            if tok in top_level:
                assert hasattr(cfg, tok), tok


def test_card_choices_match_the_actual_recipes():
    """The card's faithful block choices match what the tutorial recipes actually use
    (the card documents reality, not an aspiration)."""
    from legoesm.ocean.fidelity import mitgcm_baroclinic_gyre_recipe as bg
    from legoesm.ocean.fidelity.mitgcm_recipe import mitgcm_canonical_ocean_config

    r = bg.build_baroclinic_gyre_recipe()
    c = r.config
    card = mitgcm_canonical_ocean_config(
        A_h=c.A_h, K_h=c.K_h, A_v=c.A_v, K_v=c.K_v,
        eos_linear=c.eos_linear, physics=c.physics,
    )
    for f in ("barotropic_solver", "momentum_advection", "momentum_flux_scheme",
              "tracer_advection", "coriolis_scheme", "lateral_viscosity_operator",
              "lateral_side_bc", "outer_integrator", "ab2_scope", "eos"):
        assert getattr(card, f) == getattr(c, f), f
