"""NEMO-match recipe flag routing: GM/Redi off and the biharmonic / Leith viscosity fields."""

from legoesm.ocean.fidelity.nemo_match_recipe import (
    NEMOMatchMPASRecipeConfig,
    NEMOMatchTripoleRecipeConfig,
    nemo_match_mpas_model_config,
    nemo_match_tripole_model_config,
)


def test_mpas_gm_redi_flag_routing():
    off = nemo_match_mpas_model_config(NEMOMatchMPASRecipeConfig(gm_redi=False))
    assert off.gm_redi is None
    on = nemo_match_mpas_model_config()
    assert on.gm_redi.kappa_GM == 600.0


def test_tripole_gm_redi_flag_routing():
    off = nemo_match_tripole_model_config(NEMOMatchTripoleRecipeConfig(gm_redi=False))
    assert off.gm_redi is None
    on = nemo_match_tripole_model_config()
    assert on.gm_redi.kappa_GM == 600.0


def test_mpas_biharmonic_fields_route_through():
    cfg = nemo_match_mpas_model_config(
        NEMOMatchMPASRecipeConfig(A_h=0.0, B_h=1e10, C_smag=0.1, C_smag_lap=0.0, C_leith=0.0)
    )
    assert cfg.A_h == 0.0
    assert cfg.B_h == 1e10
    assert cfg.C_smag == 0.1
    assert cfg.C_smag_lap == 0.0


def test_tripole_laplacian_fields_route_and_biharmonic_is_refused():
    import pytest

    cfg = nemo_match_tripole_model_config(
        NEMOMatchTripoleRecipeConfig(A_h=0.0, C_smag_lap=0.0)
    )
    assert cfg.lateral_viscosity.A_h == 0.0
    assert cfg.lateral_viscosity.C_smag_lap == 0.0
    for kw in ({"B_h": 1e10}, {"C_smag": 0.1}, {"C_leith": 0.5}):
        with pytest.raises(ValueError, match="tripole recipe"):
            nemo_match_tripole_model_config(NEMOMatchTripoleRecipeConfig(**kw))


def test_recipe_defaults():
    cfg = nemo_match_mpas_model_config()
    assert cfg.A_h == 1e5
    assert cfg.B_h == 0.0
    assert cfg.C_smag == 0.0
    assert cfg.C_smag_lap == 0.33
    assert cfg.C_leith == 0.0

    tripole = nemo_match_tripole_model_config()
    assert tripole.lateral_viscosity.B_h == 0.0
    assert tripole.lateral_viscosity.C_smag == 0.0
    assert tripole.lateral_viscosity.C_smag_lap == 0.33


def test_mpas_create_setup_routes_flags():
    from scripts.run import run_omip as R

    _, _, config, _, kind = R._create_setup(
        grid_type="mpas", resolution="ico2", nlev=3, H_max=4000.0,
        physics_preset="none", water_type="jerlov_1", forcing_mode="restoring",
        A_h_override=0.0, B_h_override=1e10, C_smag=0.3, C_smag_lap=0.0,
        no_gm_redi=True,
    )
    assert kind == "mpas"
    assert config.A_h == 0.0
    assert config.B_h == 1e10
    assert config.C_smag == 0.3
    assert config.C_smag_lap == 0.0
    assert config.gm_redi is None

    _, _, dflt, _, _ = R._create_setup(
        grid_type="mpas", resolution="ico2", nlev=3, H_max=4000.0,
        physics_preset="none", water_type="jerlov_1", forcing_mode="restoring",
    )
    assert dflt.A_h == 1e5
    assert dflt.C_smag_lap == 0.33
    assert dflt.gm_redi is not None


def test_tripole_refuses_biharmonic_flags(tmp_path):
    import sys
    from pathlib import Path

    import pytest
    from scripts.run import run_omip as R

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "grids"))
    from test_tripole_multifile_mesh import _write_tripole_like_mesh

    mesh = tmp_path / "mesh.nc"
    _write_tripole_like_mesh(mesh, 8, 12, dead_north_row=False)
    common = dict(tripole_mesh=str(mesh), tripole_fold_convention="(n_lon-i)%n_lon")

    with pytest.raises(ValueError, match="tripole recipe"):
        R._create_setup("tripole", "eorca1", 3, 1000.0, "none", "type1",
                        B_h_override=1e9, **common)

    _, _, config, _, _ = R._create_setup("tripole", "eorca1", 3, 1000.0, "none", "type1",
                                         no_gm_redi=True, **common)
    assert config.gm_redi is None
