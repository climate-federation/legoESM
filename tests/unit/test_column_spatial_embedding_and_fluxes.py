"""Tests for the WeatherBench training-lane additions: (1) era5_surface_fluxes
prescribed surface-flux lower boundary condition, and (2) spatial_embedding for
the learned arms. Covers input-size arithmetic, error paths, gradient flow
through pos_embed, sensitivity of outputs to flux keys, registry wiring, and
the turbulence-only prescribed-stress path. Flag-off paths are checked to
match the legacy layout."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from legoesm.atmosphere.physics import neural_physics as nph
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.training.model_registry import (
    VARIANT_DEFAULTS,
    build_variant,
    sfno_arch_config,
    sfno_extra_input_channels,
)
from legoesm.training.neural_gcm_spectral import (
    carry_to_spectral_state,
    make_sfno_spectral_physics,
    make_turbulence_only_spectral_physics,
)

from tests.unit.test_learned_column import _mini_spectral_state

NLEV = 4
HIDDEN = 6
LAYERS = 1


@pytest.fixture(scope="module")
def grid():
    return create_gaussian_grid(n_max=10)


@pytest.fixture(scope="module")
def state(grid):
    carry, sigma = _mini_spectral_state(grid, NLEV)
    return carry_to_spectral_state(carry, grid), sigma


def _mini_net(**flags):
    return nph.build_column_physics(
        NLEV, HIDDEN, LAYERS, 0.1, key=jax.random.PRNGKey(0), **flags
    )


def _base_forcing(ncol):
    return {
        "T_sfc": jnp.full((ncol,), 290.0),
        "sic": jnp.zeros((ncol,)),
        "day_of_year": jnp.asarray(180.0),
        "seconds_of_day": jnp.asarray(43200.0),
    }


def _flux_forcing(ncol, val=0.5):
    f = _base_forcing(ncol)
    f["land_frac"] = jnp.full((ncol,), 0.3)
    for k in nph.SFC_FLUX_FORCING_KEYS:
        f[k] = jnp.full((ncol,), float(val))
    return f


# ---------------------------------------------------------------- sizes


def test_flag_off_input_size_matches_legacy():
    net = _mini_net()
    assert net.n_static == 0
    assert net.n_flux == 0
    assert net.n_embed == 0
    assert net.n_extra_dyn == 0
    assert net.pos_embed is None
    assert net.n_input == NLEV * 4 + 4


def test_flux_only_input_size():
    net = _mini_net(era5_surface_fluxes=True)
    assert net.n_flux == nph.N_SFC_FLUX_INPUT_CHANNELS
    assert net.n_static == 0
    assert net.n_embed == 0
    assert net.pos_embed is None
    assert net.n_input == NLEV * 4 + 4 + 6


def test_embedding_only_input_size_and_pos_embed():
    net = _mini_net(spatial_embedding=True, n_columns=7)
    assert net.n_embed == nph.N_COLUMN_POS_EMBED
    assert net.n_flux == 0
    assert net.pos_embed.shape == (7, nph.N_COLUMN_POS_EMBED)
    assert jnp.all(net.pos_embed == 0)
    assert net.n_input == NLEV * 4 + 4 + nph.N_COLUMN_STATIC_FEATURES + 8


def test_both_flags_input_size():
    net = _mini_net(spatial_embedding=True, era5_surface_fluxes=True, n_columns=7)
    assert net.n_input == NLEV * 4 + 4 + nph.N_COLUMN_STATIC_FEATURES + 6 + 8
    assert net.n_extra_dyn == nph.N_COLUMN_STATIC_FEATURES + 6


def test_embedding_requires_n_columns():
    with pytest.raises(ValueError):
        nph.build_column_physics(
            NLEV, HIDDEN, LAYERS, 0.1, key=jax.random.PRNGKey(1),
            spatial_embedding=True,
        )


# ------------------------------------------------------- forward errors


def _columns(net, ncol=5):
    return dict(
        T_col=jnp.ones((ncol, NLEV)),
        u_col=jnp.ones((ncol, NLEV)),
        v_col=jnp.ones((ncol, NLEV)),
        q_col=jnp.full((ncol, NLEV), 1e-3),
        p_s_col=jnp.full((ncol,), 1e5),
        solar_col=jnp.zeros((ncol,)),
        sst_col=jnp.full((ncol,), 290.0),
        sic_col=jnp.zeros((ncol,)),
    )


def test_flag_on_forward_without_extra_raises():
    net = _mini_net(spatial_embedding=True, era5_surface_fluxes=True, n_columns=5)
    with pytest.raises(ValueError):
        nph.neural_column_forward(net, **_columns(net))


def test_flag_on_forward_wrong_width_raises():
    net = _mini_net(spatial_embedding=True, era5_surface_fluxes=True, n_columns=5)
    extra = jnp.zeros((5, net.n_static + 3))
    with pytest.raises(ValueError):
        nph.neural_column_forward(net, extra_col=extra, **_columns(net))


def test_flag_off_forward_with_none_is_legacy_path():
    net = _mini_net()
    out = nph.neural_column_forward(net, extra_col=None, **_columns(net))
    assert out.shape == (5, NLEV * 4 + 6)
    assert jnp.all(out == 0)  # zero-initialised last layer


def test_flag_on_forward_correct_width_runs():
    net = _mini_net(spatial_embedding=True, era5_surface_fluxes=True, n_columns=5)
    extra = jnp.zeros((5, net.n_extra_dyn))
    out = nph.neural_column_forward(net, extra_col=extra, **_columns(net))
    assert out.shape == (5, NLEV * 4 + 6)


# ------------------------------------------------- physics_fn assembly


def _rand_last_layer(net, seed=3):
    w = jax.random.normal(jax.random.PRNGKey(seed), net.layers[-1].weight.shape) * 0.05
    return eqx.tree_at(lambda t: t.layers[-1].weight, net, w)


def test_physics_fn_flag_on_without_forcing_raises(grid, state):
    st, sigma = state
    net = _mini_net(spatial_embedding=True, era5_surface_fluxes=True,
                    n_columns=len(grid.lat) * len(grid.lon))
    fn = nph.make_column_physics_fn(net, grid)
    with pytest.raises(ValueError):
        fn(st, grid, sigma, forcing=None)


def test_physics_fn_missing_land_frac_names_key(grid, state):
    st, sigma = state
    net = _mini_net(spatial_embedding=True, n_columns=3)
    fn = nph.make_column_physics_fn(net, grid)
    forcing = _flux_forcing(3)
    del forcing["land_frac"]
    with pytest.raises(KeyError, match="land_frac"):
        fn(st, grid, sigma, forcing=forcing)


def test_physics_fn_missing_flux_key_names_key(grid, state):
    st, sigma = state
    net = _mini_net(era5_surface_fluxes=True)
    fn = nph.make_column_physics_fn(net, grid)
    forcing = _flux_forcing(3)
    del forcing["sfc_lw_up"]
    with pytest.raises(KeyError, match="sfc_lw_up"):
        fn(st, grid, sigma, forcing=forcing)


def test_pos_embed_gets_finite_nonzero_gradient(grid, state):
    st, sigma = state
    ncol = len(grid.lat) * len(grid.lon)
    net = _rand_last_layer(
        _mini_net(spatial_embedding=True, n_columns=ncol), seed=11)
    net = eqx.tree_at(
        lambda t: t.pos_embed,
        net,
        0.01 * jax.random.normal(jax.random.PRNGKey(12), (ncol, 8)),
    )
    def loss(p):
        net_p = eqx.tree_at(lambda t: t.pos_embed, net, p)
        out = nph.make_column_physics_fn(net_p, grid)(st, grid, sigma,
                                                       forcing=_flux_forcing(ncol))
        return jnp.sum(jnp.abs(out.T_hat.data) ** 2)

    g = eqx.filter_grad(loss)(net.pos_embed)
    assert g.shape == (ncol, 8)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_output_sensitive_to_sfc_shf(grid, state):
    st, sigma = state
    ncol = len(grid.lat) * len(grid.lon)
    net = _rand_last_layer(_mini_net(era5_surface_fluxes=True), seed=21)
    fn = nph.make_column_physics_fn(net, grid)
    f1 = _flux_forcing(ncol, val=0.5)
    f2 = _flux_forcing(ncol, val=0.5)
    f2["sfc_shf"] = jnp.full((ncol,), 37.0)
    o1 = fn(st, grid, sigma, forcing=f1)
    o2 = fn(st, grid, sigma, forcing=f2)
    assert float(jnp.max(jnp.abs(o2.T_hat.data - o1.T_hat.data))) > 0.0


# ------------------------------------------------------------- registry


def test_variant_defaults_gain_new_flags():
    for v in ("column_nn", "sfno_physics"):
        d = VARIANT_DEFAULTS[v]
        assert d["spatial_embedding"] is False
        assert d["era5_surface_fluxes"] is False


def test_sfno_extra_input_channels_matrix():
    assert sfno_extra_input_channels(spatial_embedding=False,
                                     era5_surface_fluxes=False) == 0
    assert sfno_extra_input_channels(spatial_embedding=True,
                                     era5_surface_fluxes=False) == 1
    assert sfno_extra_input_channels(spatial_embedding=False,
                                     era5_surface_fluxes=True) == 6
    assert sfno_extra_input_channels(spatial_embedding=True,
                                     era5_surface_fluxes=True) == 7


def test_sfno_arch_config_in_channels():
    cfg = sfno_arch_config(
        "sfno_physics", nlev=NLEV,
        overrides={"sfno_embed_dim": 8, "sfno_n_blocks": 1,
                   "spatial_embedding": True, "era5_surface_fluxes": True},
    )
    assert cfg.in_channels == cfg.out_channels + 3 + 7


def test_build_variant_embedding_requires_grid():
    with pytest.raises(ValueError):
        build_variant("column_nn", nlev=NLEV,
                      overrides={"spatial_embedding": True})


def test_build_variant_embedding_with_grid_runs():
    grid8 = create_gaussian_grid(n_max=8)
    model = build_variant("column_nn", nlev=NLEV, grid=grid8,
                          overrides={"spatial_embedding": True})
    assert model.pos_embed.shape == (len(grid8.lat) * len(grid8.lon), 8)


# ----------------------------------------------------------------- SFNO


def test_sfno_flag_mismatch_raises(grid):
    sfno = build_variant(
        "sfno_physics", nlev=NLEV, grid=grid,
        overrides={"sfno_embed_dim": 8, "sfno_n_blocks": 1},
    )
    with pytest.raises(ValueError):
        make_sfno_spectral_physics(sfno, grid, spatial_embedding=True)
    with pytest.raises(ValueError):
        make_sfno_spectral_physics(sfno, grid, era5_surface_fluxes=True)


def test_sfno_flag_on_missing_flux_key_names_key(grid, state):
    st, sigma = state
    sfno = build_variant(
        "sfno_physics", nlev=NLEV, grid=grid,
        overrides={"sfno_embed_dim": 8, "sfno_n_blocks": 1,
                   "spatial_embedding": True, "era5_surface_fluxes": True},
    )
    fn = make_sfno_spectral_physics(sfno, grid, spatial_embedding=True,
                                    era5_surface_fluxes=True)
    ncol = len(grid.lat) * len(grid.lon)
    forcing = _flux_forcing(ncol)
    del forcing["sfc_sw_up"]
    with pytest.raises(KeyError, match="sfc_sw_up"):
        fn(st, grid, sigma, forcing=forcing)


def test_sfno_sensitive_to_flux_plane(grid, state):
    st, sigma = state
    sfno = build_variant(
        "sfno_physics", nlev=NLEV, grid=grid,
        overrides={"sfno_embed_dim": 8, "sfno_n_blocks": 1,
                   "spatial_embedding": True, "era5_surface_fluxes": True},
    )
    sfno = eqx.tree_at(
        lambda t: t.decoder.weight, sfno,
        0.05 * jax.random.normal(jax.random.PRNGKey(31),
                                 sfno.decoder.weight.shape),
    )
    fn = make_sfno_spectral_physics(sfno, grid, spatial_embedding=True,
                                    era5_surface_fluxes=True)
    ncol = len(grid.lat) * len(grid.lon)
    f1 = _flux_forcing(ncol, val=0.2)
    f2 = _flux_forcing(ncol, val=0.2)
    f2["sfc_lhf"] = jnp.full((ncol,), 55.0)
    o1 = fn(st, grid, sigma, forcing=f1)
    o2 = fn(st, grid, sigma, forcing=f2)
    assert float(jnp.max(jnp.abs(o2.T_hat.data - o1.T_hat.data))) > 0.0


# ------------------------------------------------- turbulence-only fluxes


def _sheared(state, seed=5):
    pert = 1e-3 * jax.random.normal(jax.random.PRNGKey(seed),
                                    state.vor_hat.data.shape)
    return eqx.tree_at(lambda s: s.vor_hat.data, state,
                       state.vor_hat.data + pert)


def test_turbulence_only_stress_changes_tendency(state):
    st, sigma = state
    st_s = _sheared(st)
    fn = make_turbulence_only_spectral_physics(dt=900.0)
    grid = create_gaussian_grid(n_max=10)
    ncol = len(grid.lat) * len(grid.lon)
    out0 = fn(st_s, grid, sigma, forcing=None)
    f1 = _flux_forcing(ncol)
    f1.update({"sfc_tau_x": jnp.full((ncol,), 0.1),
               "sfc_tau_y": jnp.full((ncol,), 0.0)})
    out1 = fn(st_s, grid, sigma, forcing=f1)
    assert float(jnp.max(jnp.abs(out1.vor_hat.data - out0.vor_hat.data))) > 0.0
    f2 = dict(f1, sfc_tau_x=jnp.full((ncol,), 0.9))
    out2 = fn(st_s, grid, sigma, forcing=f2)
    assert float(jnp.max(jnp.abs(out2.vor_hat.data - out1.vor_hat.data))) > 0.0


def test_turbulence_only_single_stress_key_raises(state):
    st, sigma = state
    grid = create_gaussian_grid(n_max=10)
    ncol = len(grid.lat) * len(grid.lon)
    fn = make_turbulence_only_spectral_physics(dt=900.0)
    forcing = {"sfc_tau_x": jnp.full((ncol,), 0.1)}
    with pytest.raises(ValueError):
        fn(st, grid, sigma, forcing=forcing)


def test_prescribed_eastward_stress_accelerates_the_lowest_level_eastward(state):
    """SIGN through the scheme: a stress ON THE ATMOSPHERE of +0.1 Pa
    (eastward) must give du/dt > 0 at the lowest level under Louis."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_to_grid

    st, sigma = state
    grid = create_gaussian_grid(n_max=10)
    ncol = len(grid.lat) * len(grid.lon)
    fn = make_turbulence_only_spectral_physics(dt=900.0, turbulence_scheme="louis")
    f = _base_forcing(ncol)
    f.update({"sfc_tau_x": jnp.full((ncol,), 0.1),
              "sfc_tau_y": jnp.zeros((ncol,))})
    tend = fn(st, grid, sigma, forcing=f)
    du_dt = spectral_pe_to_grid(tend, grid, sigma)["u"][..., -1]
    assert float(jnp.min(du_dt)) > 0.0
    f["sfc_tau_x"] = jnp.full((ncol,), -0.1)
    du_dt_w = spectral_pe_to_grid(fn(st, grid, sigma, forcing=f), grid, sigma)["u"][..., -1]
    assert float(jnp.max(du_dt_w)) < 0.0
