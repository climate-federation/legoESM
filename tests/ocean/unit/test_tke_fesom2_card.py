"""The FESOM2 TKE card: constants, param-spec coverage, and the prognostic
carry on the MPAS lane (seeded before the step, finite after stepping)."""

import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing import config as vm_config
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig,
    VerticalMixingConfig,
    tke_fesom2_card,
)


def test_card_constants_and_default_bit_identity():
    card = tke_fesom2_card()
    assert card.prognostic is True
    assert card.surface_flux_coeff == 3.75
    assert card.prandtl_mode == "richardson"
    assert card.enable_kappaH_profile is False
    assert (card.kappaM_min, card.kappaH_min, card.kappaM_max) == (1e-4, 1e-5, 100.0)
    assert TKEConfig().surface_flux_coeff == 1.0
    assert TKEConfig().prognostic is False
    params = vm_config.__param_spec__["TKEConfig"]["params"]
    assert params["surface_flux_coeff"]["bounds"] == (0.5, 10.0)


@pytest.fixture(autouse=True)
def _fp64_policy_restored():
    """The driver's precision entry point sets a GLOBAL policy; restore the
    previous one so a same-worker test that expects the default is not
    polluted (xdist)."""
    import argparse

    from legoesm.core.precision import get_policy, set_policy
    from scripts.run import run_omip as R

    prev = get_policy()
    R.apply_run_precision(argparse.Namespace(precision="fp64"))
    try:
        yield
    finally:
        set_policy(prev)


def _build_mpas(tke_cfg, H_max=4000.0):
    from scripts.run import run_omip as R

    vm = VerticalMixingConfig(scheme="tke", tke=tke_cfg)
    mesh, z_coord, _, model, kind = R._create_setup(
        grid_type="mpas", resolution="ico2", nlev=3, H_max=H_max,
        physics_preset="none", water_type="jerlov_1", forcing_mode="restoring",
        vertical_mixing=vm,
    )
    assert kind == "mpas"
    state = R._init_rest_state("mpas", mesh, z_coord, H_max=H_max)
    return mesh, model, state


def test_prognostic_carry_seeded_and_finite_on_mpas():
    mesh, model, state = _build_mpas(tke_fesom2_card())
    assert state.tke is None
    state = model.seed_tke(state)
    assert state.tke is not None
    assert state.tke.data.shape == (mesh.nCells, 2)
    for _ in range(2):
        state = model.step(state, 600.0)
        tke = np.asarray(state.tke.data)
        assert np.all(np.isfinite(tke))
        assert np.all(tke >= 0.0)


def test_diagnostic_card_carries_no_tke_on_mpas():
    _, model, state = _build_mpas(TKEConfig())
    state = model.seed_tke(state)
    assert state.tke is None
    state = model.step(state, 600.0)
    assert state.tke is None


def test_surface_flux_coeff_changes_the_tke_under_wind():
    """Non-vacuous: the coefficient multiplies the wind-work TKE injection, so
    two cards differing only in it must produce different TKE under a nonzero
    wind stress (0.1 Pa zonal, passed as the external surface forcing)."""
    from legoesm.ocean.state import OceanSurfaceForcing

    # 100 m layers and a 1 Pa wind: the injected TKE, cd (tau/rho0)^1.5 dt/dz
    # ~ 7e-4 (cd 3.75) vs 1.9e-4 (cd 1), clears the 1e-4 surface floor in both.
    mesh, model_a, state_a = _build_mpas(tke_fesom2_card(), H_max=300.0)
    _, model_b, state_b = _build_mpas(tke_fesom2_card()._replace(surface_flux_coeff=1.0),
                                      H_max=300.0)
    state_a = model_a.seed_tke(state_a)
    state_b = model_b.seed_tke(state_b)
    tau = 1.0 * np.ones(mesh.nCells)
    sf = OceanSurfaceForcing(tau_x=tau, tau_y=0.0 * tau)
    for _ in range(2):
        state_a = model_a.step(state_a, 600.0, surface_forcing=sf)
        state_b = model_b.step(state_b, 600.0, surface_forcing=sf)
    tke_a = np.asarray(state_a.tke.data)
    tke_b = np.asarray(state_b.tke.data)
    assert np.all(np.isfinite(tke_a)) and np.all(np.isfinite(tke_b))
    assert np.any(tke_a != tke_b)
    assert float(np.max(tke_a)) > float(np.max(tke_b)) > 1.0e-4   # both above the floor


def test_prognostic_carry_seeded_and_finite_on_tripole(tmp_path):
    """Same contract on the tripole lane: the driver seeds the scan carry with
    model.seed_scan_carry (idempotent) before the first step; the stepped TKE
    is carried and finite."""
    import sys
    from pathlib import Path

    from scripts.run import run_omip as R

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "grids"))
    from test_tripole_multifile_mesh import _write_tripole_like_mesh

    mesh = tmp_path / "mesh.nc"
    _write_tripole_like_mesh(mesh, 8, 12, dead_north_row=False)
    vm = VerticalMixingConfig(scheme="tke", tke=tke_fesom2_card())
    grid, z_coord, _, model, kind = R._create_setup(
        "tripole", "eorca1", 3, 1000.0, "none", "type1",
        tripole_mesh=str(mesh), tripole_fold_convention="(n_lon-i)%n_lon",
        vertical_mixing=vm,
    )
    assert kind == "tripole"
    assert model._tke_prognostic_active()
    state = R._init_rest_state("tripole", grid, z_coord, H_max=1000.0)
    assert state.tke is None
    state = model.seed_scan_carry(state, 600.0)
    assert state.tke is not None
    # idempotent: an evolved carry must pass through untouched
    evolved = state._replace(tke=state.tke.replace(data=state.tke.data + 3.0e-3))
    state2 = model.seed_scan_carry(evolved, 600.0)
    assert np.array_equal(np.asarray(state2.tke.data), np.asarray(evolved.tke.data))
    state = model.step(state, 600.0)
    assert np.all(np.isfinite(np.asarray(state.tke.data)))


def test_restart_round_trip_keeps_the_tke_carry(tmp_path):
    """The driver seeds the template BEFORE loading a restart, so a saved
    prognostic TKE field is restored instead of being reset to background."""
    from scripts.run import run_omip as R

    mesh, model, state = _build_mpas(tke_fesom2_card(), H_max=300.0)
    state = model.seed_tke(state)
    evolved = state._replace(tke=state.tke.replace(data=state.tke.data + 2.0e-3))
    fname = R._save_restart(evolved, day=1.0, step=1, output_dir=tmp_path, grid_type="mpas")
    R._join_restart_writer()
    loaded, _, _ = R._load_restart(fname, state, grid_type="mpas")   # seeded template
    assert np.array_equal(np.asarray(loaded.tke.data), np.asarray(evolved.tke.data))
    unseeded = state._replace(tke=None)
    lost, _, _ = R._load_restart(fname, unseeded, grid_type="mpas")
    assert lost.tke is None   # why the seeding must precede the load


def test_driver_seeds_the_carry_before_loading_the_restart():
    """Order pin on run_omip_single's source: the seeding block must precede
    the restart load, or a saved prognostic TKE is silently dropped (see
    test_restart_round_trip_keeps_the_tke_carry for the mechanism)."""
    import inspect

    from scripts.run import run_omip as R

    src = inspect.getsource(R.run_omip_single)
    i_seed = src.index("model.seed_tke(state)")
    i_scan = src.index("model.seed_scan_carry(state, dt)")
    i_load = src.index("_load_restart(")
    i_spmd = src.index("if run_config.enable_latlon_spmd:")
    assert max(i_seed, i_scan) < i_load < i_spmd
