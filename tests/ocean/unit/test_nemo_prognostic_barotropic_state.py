"""Decision-8 guards for NEMO's prognostic ``uu_b/vv_b`` state pair."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.restart import load_run_restart, save_run_restart
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


def _case(*, carried: bool):
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=3, H_max=300.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=10.0, H_max=300.0,
        land_lat_threshold=70.0,
        nemo_prognostic_barotropic_velocity=carried,
    )
    return grid, z, state


def test_non_nemo_state_has_no_depth_mean_array_leaves():
    """The approved state expansion is confined to NEMO-identity recipes."""
    _, _, plain = _case(carried=False)
    _, _, nemo = _case(carried=True)
    assert plain.uu_b is None and plain.vv_b is None
    assert len(jax.tree_util.tree_leaves(nemo)) == (
        len(jax.tree_util.tree_leaves(plain)) + 2)
    assert nemo.uu_b.data.shape == nemo.u.data.shape[:-1]
    assert nemo.vv_b.data.shape == nemo.v.data.shape[:-1]
    assert np.count_nonzero(np.asarray(nemo.uu_b.data)) == 0
    assert np.count_nonzero(np.asarray(nemo.vv_b.data)) == 0


def test_external_mode_reads_and_rewrites_carried_pair():
    """A changed Kbb pair changes the production seed despite identical 3-D u/v."""
    grid, z, state = _case(carried=True)
    cfg = LatLonCGridOceanConfig(fix_eta_drift=False)
    cfg = cfg._replace(barotropic=cfg.barotropic._replace(
        barotropic_solver="explicit_substep",
        bebt=0.0,
        maxvel_barotropic=0.0,
        barotropic_diffusion_alpha=0.0,
        barotropic_local_subcycle_clamp=False,
    ))
    u_seed = 1.0e-4 * state.u_mask.data
    seeded = state._replace(uu_b=state.uu_b.replace(data=u_seed))
    out_zero, _ = barotropic_substeps_latlon_cgrid(
        state, 1.0, 1, grid, z, cfg, add_barotropic_coriolis=False)
    out_seed, _ = barotropic_substeps_latlon_cgrid(
        seeded, 1.0, 1, grid, z, cfg, add_barotropic_coriolis=False)
    assert np.max(np.abs(np.asarray(
        out_seed.uu_b.data - out_zero.uu_b.data))) > 0.0
    assert out_seed.uu_b is not seeded.uu_b
    with pytest.raises(ValueError, match="both uu_b and vv_b"):
        barotropic_substeps_latlon_cgrid(
            seeded._replace(vv_b=None), 1.0, 1, grid, z, cfg,
            add_barotropic_coriolis=False)


def test_run_restart_round_trips_depth_mean_bits(tmp_path):
    """The production restart inventory persists the pair as prognostic state."""
    _, _, state = _case(carried=True)
    ub = jnp.arange(state.uu_b.data.size, dtype=jnp.float64).reshape(
        state.uu_b.data.shape) * jnp.float64(2.0**-40)
    vb = jnp.arange(state.vv_b.data.size, dtype=jnp.float64).reshape(
        state.vv_b.data.shape) * jnp.float64(-2.0**-41)
    marked = state._replace(
        uu_b=state.uu_b.replace(data=ub), vv_b=state.vv_b.replace(data=vb))
    path = tmp_path / "nemo_identity.npz"
    save_run_restart(path, marked, step=5, time_days=5.0,
                     grid_type="latlon")
    got, _, meta = load_run_restart(
        path, state, grid_type="latlon")
    assert meta["step"] == 5
    np.testing.assert_array_equal(np.asarray(got.uu_b.data), np.asarray(ub))
    np.testing.assert_array_equal(np.asarray(got.vv_b.data), np.asarray(vb))


def test_nemo_identity_kt5_restart_matches_unbroken_kt6_to_10(tmp_path):
    """NEMO-style prognostics, including uu_b/vv_b, restart without a cold start."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    old_policy = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        card = build_nemo_testcase_card("LOCK_EXCHANGE-zco")
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
        unbroken = card.recipe.initial_state
        checkpoint = tmp_path / "kt5.npz"
        after = {}
        for kt in range(1, 11):
            unbroken = model.step(unbroken, dt=card.dt_s)
            if kt == 5:
                save_run_restart(
                    checkpoint, unbroken, step=5,
                    time_days=5.0 * card.dt_s / 86400.0,
                    grid_type="latlon")
            elif kt >= 6:
                after[kt] = unbroken

        resumed, _, meta = load_run_restart(
            checkpoint, card.recipe.initial_state, grid_type="latlon")
        assert meta["step"] == 5
        for kt in range(6, 11):
            resumed = model.step(resumed, dt=card.dt_s)
            want_leaves = jax.tree_util.tree_leaves(after[kt])
            got_leaves = jax.tree_util.tree_leaves(resumed)
            assert len(got_leaves) == len(want_leaves)
            for got, want in zip(got_leaves, want_leaves):
                np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    finally:
        set_policy(old_policy)
