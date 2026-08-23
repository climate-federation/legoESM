"""Unit tests for the proven OMIP NEMO-match recipe cards (#500).

Direct tests for ``legoesm.ocean.fidelity.nemo_match_recipe`` — the single
source of truth for the two PROVEN OMIP dycores (MPAS ico6 SST RMSE 0.84,
tripole eORCA025 SST RMSE 1.15).  The catalog<->factory<->_create_setup drift
guards live in ``test_recipes.py``; this file exercises the factory's own
behaviour: block selection, model construction (so the shared model validators
run), the GM/Redi enable/disable branch, the physics passthrough, and the
unknown-dispatch raise.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import pytest

from legoesm.ocean.fidelity.nemo_match_recipe import (
    NEMO_MATCH_BLOCK_MAPPING,
    NEMOMatchMPASRecipeConfig,
    NEMOMatchTripoleRecipeConfig,
    nemo_match_mpas_model_config,
    nemo_match_tripole_model_config,
)


def test_mpas_factory_selects_canonical_blocks():
    mc = nemo_match_mpas_model_config()
    assert mc.eos == "wright"
    assert mc.tracer_advection == "tvd"
    assert mc.pgf_scheme == "adcroft"
    assert mc.barotropic_solver == "implicit_cn"  # MPAS: flat (not grouped)
    assert mc.pv_scheme == "enstrophy"
    assert mc.implicit_vertical_mixing is True
    assert mc.A_h == pytest.approx(1.0e5)
    assert mc.C_smag_lap == pytest.approx(0.33)
    assert mc.K_zeta_bih == pytest.approx(1.0e14)
    assert mc.barotropic_implicit_pcg_maxiter == 300  # MPAS: flat (not grouped)
    assert mc.normalize_freshwater is True
    assert mc.gm_redi is not None
    assert mc.gm_redi.kappa_GM == pytest.approx(600.0)
    assert mc.gm_redi.kappa_Redi == pytest.approx(600.0)
    assert mc.gm_redi.slope_scheme == "centered"
    assert mc.gm_redi.visbeck.enabled is False


def test_tripole_factory_selects_canonical_blocks():
    mc = nemo_match_tripole_model_config()
    assert mc.eos == "wright"
    assert mc.momentum_advection == "vector_invariant"
    assert mc.tracer_advection == "tvd"
    assert mc.pgf_scheme == "adcroft"
    # ke_gradient intentionally the centered default (NOT fold-aware hollingsworth).
    assert mc.ke_gradient_scheme == "centered"
    assert mc.barotropic.barotropic_solver == "implicit_cn"
    assert mc.coriolis_scheme == "matsuno_split"
    assert mc.outer_integrator == "forward_euler"
    assert mc.tracer_time_integrator == "euler"
    assert mc.barotropic.n_barotropic_substeps == 30
    assert mc.freshwater_closure == "virtual_salt_flux"
    assert mc.gm_redi is not None
    assert mc.gm_redi.kappa_GM == pytest.approx(600.0)
    assert mc.gm_redi.slope_scheme == "centered"


def test_gm_redi_disable_branch():
    mc = nemo_match_mpas_model_config(
        NEMOMatchMPASRecipeConfig(gm_redi=False))
    assert mc.gm_redi is None
    tri = nemo_match_tripole_model_config(
        NEMOMatchTripoleRecipeConfig(gm_redi=False))
    assert tri.gm_redi is None


def test_gm_treguier_branch_defaults_off_and_is_byte_identical():
    """The recipe default keeps the CONSTANT kappa_GM; the Treguier block is
    present but disabled, so existing cards are unaffected."""
    mc = nemo_match_tripole_model_config(NEMOMatchTripoleRecipeConfig())
    assert mc.gm_redi.treguier.enabled is False
    assert mc.gm_redi.visbeck.enabled is False
    assert mc.gm_redi.kappa_GM == pytest.approx(600.0)


def test_gm_treguier_threads_the_kappa_min_floor():
    """REGRESSION: the recipe built ``TreguierConfig(enabled=True, aei0=...)``
    with NO kappa_min field at all, so the floor was UNREACHABLE from the
    recipe and silently 0.0 while ``run_omip_core2.py --gm-kappa-min``
    defaulted to 200 -- one scheme, two defaults, nobody aware.

    The floor is now threaded. The recipe default stays 0.0 because this is a
    NEMO-MATCH card and raw NEMO is capped-only; the run script's 200.0 is a
    production stability knob. That divergence is DELIBERATE and asserted at
    both ends (see tests/unit/test_run_omip_core2_gm_treguier.py)."""
    cfg = NEMOMatchTripoleRecipeConfig(gm_treguier=True)
    assert cfg.gm_kappa_min == 0.0          # oracle default = raw NEMO
    mc = nemo_match_tripole_model_config(cfg)
    assert mc.gm_redi.treguier.enabled is True
    # 900 = 1/2*rn_Ue*rn_Le (laplacian prefactor, ldftra.F90:290-293); NEMO's
    # emitted aeiu_2d maxes at exactly 900 on eORCA1.  The old 1800 here
    # encoded the bilaplacian prefactor, which NEMO rejects for EIV.
    assert mc.gm_redi.treguier.aei0 == pytest.approx(900.0)
    assert mc.gm_redi.treguier.kappa_min == 0.0
    # Visbeck stays off (mutually exclusive), Redi/S_max untouched
    assert mc.gm_redi.visbeck.enabled is False
    assert mc.gm_redi.kappa_Redi == pytest.approx(600.0)
    assert mc.gm_redi.S_max == pytest.approx(0.005)
    # the floor is now REACHABLE from the recipe (the actual bug fixed here)
    floored = nemo_match_tripole_model_config(
        NEMOMatchTripoleRecipeConfig(gm_treguier=True, gm_kappa_min=200.0))
    assert floored.gm_redi.treguier.kappa_min == pytest.approx(200.0)


def test_gm_treguier_requires_gm_redi():
    """gm_redi=False returns None early, which would silently discard both the
    selected scheme and its floor."""
    with pytest.raises(ValueError, match="cannot apply when GM/Redi is"):
        nemo_match_tripole_model_config(
            NEMOMatchTripoleRecipeConfig(gm_redi=False, gm_treguier=True))


def test_mpas_recipe_rejects_gm_treguier_at_config_build():
    """gm_redi_mpas raises NotImplementedError for the Treguier block, but only
    inside the first GM tendency -- after a full model build. The recipe must
    reject the unsupported flag up front."""
    with pytest.raises(NotImplementedError, match="lat-lon C-grid GM/Redi"):
        nemo_match_mpas_model_config(
            NEMOMatchMPASRecipeConfig(gm_treguier=True))


def test_gm_treguier_rejects_floor_above_cap():
    with pytest.raises(ValueError, match="exceeds the NEMO cap"):
        nemo_match_tripole_model_config(
            NEMOMatchTripoleRecipeConfig(gm_treguier=True, gm_aei0=1800.0,
                                         gm_kappa_min=5000.0))


def test_physics_passthrough_is_setup():
    """A caller-supplied physics is used verbatim (SETUP), and the default
    standalone physics is built when none is given (recipe-only use)."""
    sentinel = object()
    mc = nemo_match_mpas_model_config(physics=sentinel)
    assert mc.physics is sentinel
    # Default standalone physics: the proven restoring-mode SETUP.
    default = nemo_match_tripole_model_config()
    assert default.physics is not None
    assert default.physics.surface_forcing.scheme == "combined"
    assert default.physics.vertical_mixing.scheme == "kpp"
    assert default.physics.convection.scheme == "enhanced_diffusion"
    assert default.physics.convection.enhanced_diffusion.K_conv == pytest.approx(1.0)


def test_default_physics_jra55_mode_uses_none_surface_forcing():
    from legoesm.ocean.fidelity import nemo_match_recipe as M
    phys = M._default_match_physics(forcing_mode="jra55_do_tropical")
    assert phys.surface_forcing.scheme == "none"
    assert phys.vertical_mixing.scheme == "kpp"


def test_default_physics_rejects_unknown_forcing_mode():
    from legoesm.ocean.fidelity import nemo_match_recipe as M
    with pytest.raises(ValueError, match="unknown NEMO-match forcing_mode"):
        M._default_match_physics(forcing_mode="bogus")


def test_mpas_factory_builds_valid_model_and_one_step_is_finite():
    """The MPAS factory config constructs a valid model and a step stays finite
    (exercises the shared model validators on a tiny ico2 mesh)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    mesh = create_voronoi_mesh(2)
    z_coord = create_ocean_z_star(n_levels=3, H_max=4000.0)
    config = nemo_match_mpas_model_config()
    model = MPASOceanModel(mesh, z_coord, config)
    state = rest_state_mpas_ocean(
        mesh, z_coord, H_max=4000.0, land_lat_threshold=90.0)
    new_state = model.step(state, dt=60.0)
    assert bool(jnp.all(jnp.isfinite(new_state.T.data)))
    assert bool(jnp.all(jnp.isfinite(new_state.u.data)))
    assert bool(jnp.all(jnp.isfinite(new_state.eta.data)))


def test_tripole_factory_builds_valid_model_and_one_step_is_finite():
    """The tripole factory config constructs a valid lat-lon C-grid model on a
    small synthetic grid (no eORCA mesh file needed for the dycore validators)
    AND a step stays finite — the tripole card was previously only constructed,
    never stepped (issue #501).  The eORCA north-fold is not exercised on this
    synthetic grid; this gates the dycore scheme stack the card selects."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(8, 16)
    z_coord = create_ocean_z_star(n_levels=3, H_max=4000.0)
    config = nemo_match_tripole_model_config()
    model = LatLonCGridOceanModel(grid, z_coord, config)
    state = rest_state_latlon_cgrid_ocean(grid, z_coord)
    new_state = model.step(state, dt=60.0)
    assert bool(jnp.all(jnp.isfinite(new_state.T.data)))
    assert bool(jnp.all(jnp.isfinite(new_state.u.data)))
    assert bool(jnp.all(jnp.isfinite(new_state.eta.data)))


def test_block_mapping_is_explicit():
    names = {name for name, _, _ in NEMO_MATCH_BLOCK_MAPPING}
    assert "barotropic solver" in names
    assert "GM/Redi" in names
    assert any(
        name == "barotropic solver" and "implicit_cn" in sel
        for name, _, sel in NEMO_MATCH_BLOCK_MAPPING
    )


def test_nemo_match_recipe_is_lazy_registered():
    import legoesm.ocean.fidelity as fidelity

    assert "nemo_match_recipe" in fidelity.__all__
    # Lazy __getattr__ resolves the module.
    assert fidelity.nemo_match_recipe is not None
