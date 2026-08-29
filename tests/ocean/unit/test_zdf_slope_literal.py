"""Red-capable unit receipts for the row-30 literal ldf_slp inputs."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.eos import NemoSEOSConfig, nemo_seos_prd_literal
from legoesm.ocean.experiments.dino import (
    DINOConfig,
    DINO_RECIPES,
    dino_lat_lon_grid,
    dino_lat_lon_model_config,
    dino_config_for_recipe,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig


def test_literal_prd_matches_hand_computed_source_order_and_red_roundtrip():
    cfg = NemoSEOSConfig()
    temperature = jnp.asarray([8.25, 12.75], dtype=jnp.float64)
    salinity = jnp.asarray([34.8, 35.3], dtype=jnp.float64)
    depth = jnp.asarray([127.125, 3987.75], dtype=jnp.float64)
    got = np.asarray(nemo_seos_prd_literal(
        temperature, salinity, depth, cfg))

    zt = np.asarray(temperature) - np.float64(cfg.T0)
    zs = np.asarray(salinity) - np.float64(cfg.S0)
    t_linear = np.float64(0.5) * np.float64(cfg.lambda1) * zt
    t_depth = np.float64(cfg.mu1) * np.asarray(depth)
    t_factor = np.float64(1.0) + t_linear + t_depth
    t_term = -np.float64(cfg.a0) * t_factor * zt
    s_linear = np.float64(0.5) * np.float64(cfg.lambda2) * zs
    s_depth = np.float64(cfg.mu2) * np.asarray(depth)
    s_factor = np.float64(1.0) - s_linear - s_depth
    s_term = np.float64(cfg.b0) * s_factor * zs
    cross = np.float64(cfg.nu) * zt * zs
    want = (t_term + s_term - cross) * (
        np.float64(1.0) / np.float64(cfg.rho0))
    np.testing.assert_array_equal(got, want)

    # Planted legacy construction: adding rho0 and then subtracting it again
    # must be distinguishable for this fixture, or the test cannot go red.
    roundtrip = (np.float64(cfg.rho0) + (t_term + s_term - cross)) \
        / np.float64(cfg.rho0) - np.float64(1.0)
    assert np.any(got.view(np.uint64) != roundtrip.view(np.uint64))


def test_literal_prd_is_jittable_and_differentiable():
    fn = lambda t: nemo_seos_prd_literal(
        t, jnp.asarray(35.1), jnp.asarray(1400.0), NemoSEOSConfig())
    x = jnp.asarray(9.25, dtype=jnp.float64)
    np.testing.assert_array_equal(np.asarray(jax.jit(fn)(x)), np.asarray(fn(x)))
    assert np.isfinite(np.asarray(jax.grad(fn)(x)))


def test_full_literal_slope_with_carried_w_bundle_has_finite_jit_gradient():
    """The unused restored surface W slot must not inject 0*inf into AD."""
    from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_nemo_native_slopes,
    )
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate,
        create_z_star_from_thicknesses,
    )

    nlat, nlon, nlev = 4, 4, 4
    dz = np.asarray([1.0, 2.0, 3.0, 4.0])
    shape = (nlat, nlon, nlev)
    horizontal = np.ones((nlat, nlon))
    gdept = np.broadcast_to(np.asarray([0.5, 2.0, 4.5, 8.0]), shape)
    gdepw = np.broadcast_to(np.asarray([0.0, 1.0, 3.0, 6.0]), shape)
    e3w_mesh = np.broadcast_to(np.asarray([1.0, 1.5, 2.5, 3.5]), shape)
    raw = create_z_star_from_thicknesses(
        dz, nemo_gdept_0_m=gdept, nemo_gdepw_0_m=gdepw,
        nemo_e3t_0_m=np.broadcast_to(dz, shape),
        nemo_e3w_0_m=e3w_mesh,
        nemo_hu_0_m=horizontal, nemo_hv_0_m=horizontal,
        nemo_e1e2t_m=horizontal, nemo_e1e2u_m=horizontal,
        nemo_e1e2v_m=horizontal)
    z_coord = create_partial_cell_coordinate(
        raw, jnp.full((nlat, nlon), float(dz.sum())))
    grid = ensure_geometry(create_latlon_grid(n_lat=nlat, n_lon=nlon))
    lat = jnp.arange(nlat, dtype=jnp.float64)[:, None, None]
    lon = jnp.arange(nlon, dtype=jnp.float64)[None, :, None]
    lev = jnp.arange(nlev, dtype=jnp.float64)[None, None, :]
    T = 12.0 - 0.4 * lev + 0.03 * lat + 0.02 * lon
    S = 35.0 + 0.01 * lev - 0.002 * lat
    mask = jnp.ones((nlat, nlon), dtype=jnp.float64)
    umask = jnp.ones((nlat, nlon + 1), dtype=jnp.float64)
    vmask = jnp.ones((nlat + 1, nlon), dtype=jnp.float64)
    cfg = GMRediConfig(slope_prd_evaluation="nemo_literal")
    eos_fn = make_eos_fn("nemo_seos", None, rho0=1026.0)
    carried_n2 = jnp.full((nlat, nlon, nlev - 1), 1.0e-5)
    carried_e3w = jnp.broadcast_to(
        jnp.asarray([1.5, 2.5, 3.5]), carried_n2.shape)

    def objective(temperature):
        rho = 1026.0 + 0.2 * (10.0 - temperature)
        slopes = compute_nemo_native_slopes(
            rho, temperature, S, mask, umask, vmask, z_coord, grid, cfg,
            eos_fn, active_3d=z_coord.is_active,
            pn2_override=carried_n2, e3w_override=carried_e3w)
        return sum(jnp.sum(field) for field in slopes)

    eager = jax.grad(objective)(T)
    compiled = jax.jit(jax.grad(objective))(T)
    assert np.all(np.isfinite(np.asarray(eager)))
    assert np.all(np.isfinite(np.asarray(compiled)))
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager),
                               rtol=1.0e-12, atol=1.0e-14)


def test_row30_selectors_are_scoped_to_the_two_dino_nemo_cards():
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name in DINO_RECIPES:
        cfg = dino_config_for_recipe(name)
        if name in faithful:
            assert cfg.gm_redi_slope_n2_evaluation == "carried_step_entry"
            assert cfg.gm_redi_slope_prd_geometry_stage == "before_step"
            assert cfg.gm_redi_slope_prd_evaluation == "nemo_literal"
        else:
            assert cfg.gm_redi_slope_n2_evaluation == "recompute", name
            assert cfg.gm_redi_slope_prd_geometry_stage == "current_step", name
            assert cfg.gm_redi_slope_prd_evaluation == "density_roundtrip", name

    assert GMRediConfig().slope_n2_evaluation == "recompute"
    assert GMRediConfig().slope_prd_geometry_stage == "current_step"
    assert GMRediConfig().slope_prd_evaluation == "density_roundtrip"
    explicit_legacy = dataclasses.replace(
        DINOConfig(), gm_redi_slope_n2_evaluation="recompute",
        gm_redi_slope_prd_geometry_stage="current_step",
        gm_redi_slope_prd_evaluation="density_roundtrip")
    assert explicit_legacy == DINOConfig()

    fe = dino_config_for_recipe("nemo_dino_kamm")
    mlf = dino_config_for_recipe("nemo_dino_kamm_mlf")
    assert dino_lat_lon_model_config(
        dino_lat_lon_grid(fe, n_lon=8), fe, physics=True)[0].outer_integrator \
        == "forward_euler"
    assert dino_lat_lon_model_config(
        dino_lat_lon_grid(mlf, n_lon=8), mlf, physics=True)[0].outer_integrator \
        == "leapfrog"
