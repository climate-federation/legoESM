"""Red-capable unit receipts for the row-30 literal ldf_slp inputs."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def test_full_literal_slope_with_carried_w_bundle_has_finite_jit_gradient(
        monkeypatch):
    """Kmm literal geometry supersedes a poisoned later-state Jacobian."""
    from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_nemo_native_slopes,
    )
    from legoesm.ocean.physics.lateral_mixing import gm_redi_latlon_cgrid as gm
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
    cfg = GMRediConfig(
        slope_prd_evaluation="nemo_literal",
        slope_metric_evaluation="nemo_reciprocal")
    literal_cfg = cfg._replace(
        slope_n2="nemo_bn2",
        slope_depth_evaluation="nemo_qco_live_literal")
    eos_fn = make_eos_fn("nemo_seos", None, rho0=1026.0)
    carried_n2 = jnp.full((nlat, nlon, nlev - 1), 1.0e-5)
    carried_e3w = jnp.broadcast_to(
        jnp.asarray([1.5, 2.5, 3.5]), carried_n2.shape)
    eta_kmm = jnp.full((nlat, nlon), 0.125, dtype=jnp.float64)
    H_bathy = jnp.full((nlat, nlon), float(dz.sum()), dtype=jnp.float64)
    poisoned_later_jacobian = jnp.full(
        (nlat, nlon), jnp.nan, dtype=jnp.float64)
    assert not np.isfinite(np.asarray(poisoned_later_jacobian)).any()

    def objective(temperature):
        rho = 1026.0 + 0.2 * (10.0 - temperature)
        slopes = compute_nemo_native_slopes(
            rho, temperature, S, mask, umask, vmask, z_coord, grid,
            literal_cfg,
            eos_fn, active_3d=z_coord.is_active,
            jacobian=poisoned_later_jacobian,
            eta=eta_kmm, H_bathy=H_bathy,
            pn2_override=carried_n2, e3w_override=carried_e3w)
        return sum(jnp.sum(field) for field in slopes)

    eager = jax.grad(objective)(T)
    compiled = jax.jit(jax.grad(objective))(T)
    assert np.all(np.isfinite(np.asarray(eager)))
    assert np.all(np.isfinite(np.asarray(compiled)))
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager),
                               rtol=1.0e-12, atol=1.0e-14)

    # Scope receipt: the historical implicit default and its explicit selector
    # must remain numerically byte-identical through the active slope path.
    rho = 1026.0 + 0.2 * (10.0 - T)
    explicit_static = cfg._replace(
        slope_face_thickness_evaluation="static_face",
        slope_depth_evaluation="legacy_jacobian_t_surface")
    default_out = compute_nemo_native_slopes(
        rho, T, S, mask, umask, vmask, z_coord, grid, cfg, eos_fn,
        active_3d=z_coord.is_active, pn2_override=carried_n2,
        e3w_override=carried_e3w)
    explicit_out = compute_nemo_native_slopes(
        rho, T, S, mask, umask, vmask, z_coord, grid, explicit_static,
        eos_fn, active_3d=z_coord.is_active, pn2_override=carried_n2,
        e3w_override=carried_e3w)
    for got, want in zip(explicit_out, default_out):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    # Every card outside the two DINO oracle cards must stay on the static
    # path.  Make accidental live-helper reachability a hard red failure and
    # exercise the complete numerical entry point for each unchanged card.
    def forbidden_live_helper(*_args, **_kwargs):
        raise AssertionError("unchanged card reached nemo_qco_live helper")

    monkeypatch.setattr(
        gm, "_nemo_qco_live_slope_face_thicknesses", forbidden_live_helper)
    monkeypatch.setattr(
        gm, "_nemo_qco_live_slope_depths", forbidden_live_helper)
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name in DINO_RECIPES:
        if name in faithful:
            continue
        card = dino_config_for_recipe(name)
        card_cfg = cfg._replace(
            slope_face_thickness_evaluation=
            card.gm_redi_slope_face_thickness_evaluation,
            slope_depth_evaluation=card.gm_redi_slope_depth_evaluation)
        card_out = compute_nemo_native_slopes(
            rho, T, S, mask, umask, vmask, z_coord, grid, card_cfg,
            eos_fn, active_3d=z_coord.is_active, pn2_override=carried_n2,
            e3w_override=carried_e3w)
        for got, want in zip(card_out, default_out):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want),
                                          err_msg=name)


def test_row30_selectors_are_scoped_to_the_two_dino_nemo_cards():
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name in DINO_RECIPES:
        cfg = dino_config_for_recipe(name)
        if name in faithful:
            assert cfg.gm_redi_slope_n2_evaluation == "carried_step_entry"
            assert cfg.gm_redi_slope_prd_geometry_stage == "before_step"
            assert cfg.gm_redi_slope_prd_evaluation == "nemo_literal"
            assert cfg.gm_redi_slope_metric_evaluation == "nemo_reciprocal"
            assert cfg.gm_redi_slope_face_thickness_evaluation == "nemo_qco_live"
            assert cfg.gm_redi_flux_face_thickness_evaluation == "nemo_qco_live"
            assert cfg.gm_redi_slope_depth_evaluation == "nemo_qco_live_literal"
            assert cfg.gm_treguier_vertical_reduction_evaluation == "nemo_left"
            assert cfg.gm_treguier_sqrt_evaluation == "nemo_forward_exact"
        else:
            assert cfg.gm_redi_slope_n2_evaluation == "recompute", name
            assert cfg.gm_redi_slope_prd_geometry_stage == "current_step", name
            assert cfg.gm_redi_slope_prd_evaluation == "density_roundtrip", name
            assert cfg.gm_redi_slope_metric_evaluation == "division", name
            assert cfg.gm_redi_slope_face_thickness_evaluation == "static_face", name
            assert cfg.gm_redi_flux_face_thickness_evaluation == \
                "tpoint_jacobian", name
            assert cfg.gm_redi_slope_depth_evaluation == \
                "legacy_jacobian_t_surface", name
            assert cfg.gm_treguier_vertical_reduction_evaluation == "tree", name
            assert cfg.gm_treguier_sqrt_evaluation == "guarded_floor", name

    assert GMRediConfig().slope_n2_evaluation == "recompute"
    assert GMRediConfig().slope_prd_geometry_stage == "current_step"
    assert GMRediConfig().slope_prd_evaluation == "density_roundtrip"
    assert GMRediConfig().slope_metric_evaluation == "division"
    assert GMRediConfig().slope_face_thickness_evaluation == "static_face"
    assert GMRediConfig().redi_flux_face_thickness_evaluation == \
        "tpoint_jacobian"
    assert GMRediConfig().slope_depth_evaluation == \
        "legacy_jacobian_t_surface"
    assert GMRediConfig().treguier_vertical_reduction_evaluation == "tree"
    assert GMRediConfig().treguier_sqrt_evaluation == "guarded_floor"
    explicit_legacy = dataclasses.replace(
        DINOConfig(), gm_redi_slope_n2_evaluation="recompute",
        gm_redi_slope_prd_geometry_stage="current_step",
        gm_redi_slope_prd_evaluation="density_roundtrip",
        gm_redi_slope_metric_evaluation="division",
        gm_redi_slope_face_thickness_evaluation="static_face",
        gm_redi_flux_face_thickness_evaluation="tpoint_jacobian",
        gm_redi_slope_depth_evaluation="legacy_jacobian_t_surface",
        gm_treguier_vertical_reduction_evaluation="tree",
        gm_treguier_sqrt_evaluation="guarded_floor")
    assert explicit_legacy == DINOConfig()

    fe = dino_config_for_recipe("nemo_dino_kamm")
    mlf = dino_config_for_recipe("nemo_dino_kamm_mlf")
    assert dino_lat_lon_model_config(
        dino_lat_lon_grid(fe, n_lon=8), fe, physics=True)[0].outer_integrator \
        == "forward_euler"
    assert dino_lat_lon_model_config(
        dino_lat_lon_grid(mlf, n_lon=8), mlf, physics=True)[0].outer_integrator \
        == "leapfrog"
    bad = dataclasses.replace(
        DINOConfig(), gm_redi_slope_depth_evaluation="silent_typo")
    with pytest.raises(ValueError, match="gm_redi_slope_depth_evaluation"):
        dino_lat_lon_model_config(dino_lat_lon_grid(bad, n_lon=8), bad)
    bad_flux_face = dataclasses.replace(
        DINOConfig(), gm_redi_flux_face_thickness_evaluation="silent_typo")
    with pytest.raises(ValueError, match="gm_redi_flux_face_thickness"):
        dino_lat_lon_model_config(
            dino_lat_lon_grid(bad_flux_face, n_lon=8), bad_flux_face)
    bad_reduction = dataclasses.replace(
        DINOConfig(), gm_treguier_vertical_reduction_evaluation="silent_typo")
    with pytest.raises(ValueError, match="gm_treguier_vertical_reduction"):
        dino_lat_lon_model_config(
            dino_lat_lon_grid(bad_reduction, n_lon=8), bad_reduction)
    bad_sqrt = dataclasses.replace(
        DINOConfig(), gm_treguier_sqrt_evaluation="silent_typo")
    with pytest.raises(ValueError, match="gm_treguier_sqrt_evaluation"):
        dino_lat_lon_model_config(
            dino_lat_lon_grid(bad_sqrt, n_lon=8), bad_sqrt)


def test_nemo_qco_live_face_thickness_matches_hand_source_order_and_is_red():
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        _nemo_qco_live_slope_face_thicknesses,
    )

    eta = jnp.asarray([[0.25, -0.125], [0.375, 0.0625]], dtype=jnp.float64)
    hu0 = np.asarray([[10.0, 12.0], [15.0, 20.0]], dtype=np.float64)
    hv0 = np.asarray([[11.0, 13.0], [17.0, 19.0]], dtype=np.float64)
    area_t = np.asarray([[2.0, 3.0], [5.0, 7.0]], dtype=np.float64)
    area_u = np.asarray([[11.0, 13.0], [17.0, 23.0]], dtype=np.float64)
    area_v = np.asarray([[19.0, 29.0], [31.0, 37.0]], dtype=np.float64)
    z_coord = SimpleNamespace(
        nemo_hu_0=jnp.asarray(hu0), nemo_hv_0=jnp.asarray(hv0),
        nemo_e1e2t=jnp.asarray(area_t), nemo_e1e2u=jnp.asarray(area_u),
        nemo_e1e2v=jnp.asarray(area_v))
    e3u0 = jnp.asarray(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=jnp.float64)
    e3v0 = e3u0 + jnp.float64(0.5)
    umask = jnp.ones_like(e3u0)
    vmask = jnp.ones_like(e3v0)

    def live(ssh):
        return _nemo_qco_live_slope_face_thicknesses(
            ssh, z_coord, e3u0, e3v0, umask, vmask)

    got_u, got_v = jax.jit(live)(eta)
    eta_np = np.asarray(eta)
    weighted = area_t * eta_np
    num_u = np.float64(0.5) * (weighted + np.roll(weighted, -1, axis=1))
    num_v = np.float64(0.5) * (weighted + np.roll(weighted, -1, axis=0))
    r1_hu0 = np.float64(1.0) / hu0
    r1_hv0 = np.float64(1.0) / hv0
    r1_area_u = np.float64(1.0) / area_u
    r1_area_v = np.float64(1.0) / area_v
    r3u = (num_u * r1_hu0) * r1_area_u
    r3v = (num_v * r1_hv0) * r1_area_v
    want_u = np.asarray(e3u0) * (np.float64(1.0) + r3u[..., None])
    want_v = np.asarray(e3v0) * (np.float64(1.0) + r3v[..., None])
    np.testing.assert_array_equal(np.asarray(got_u), want_u)
    np.testing.assert_array_equal(np.asarray(got_v), want_v)
    assert np.any(np.asarray(got_u).view(np.uint64) != np.asarray(e3u0).view(np.uint64))
    tangent = jax.grad(lambda ssh: jnp.sum(live(ssh)[0]))(eta)
    assert np.all(np.isfinite(np.asarray(tangent)))


def test_nemo_qco_live_depth_and_face_accumulation_are_literal_jit_ad_and_red():
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        _nemo_literal_slope_face_depth,
        _nemo_qco_live_slope_depths,
    )

    eta = jnp.asarray([[0.25, -0.125], [0.375, 0.0625]], dtype=jnp.float64)
    H = jnp.asarray([[10.0, 12.0], [15.0, 20.0]], dtype=jnp.float64)
    gdept0 = np.asarray(
        [[[0.5, 2.0, 4.5], [0.6, 2.1, 4.6]],
         [[0.7, 2.2, 4.7], [0.8, 2.3, 4.8]]], dtype=np.float64)
    gdepw0 = np.asarray(
        [[[0.0, 1.0, 3.0], [0.0, 1.1, 3.1]],
         [[0.0, 1.2, 3.2], [0.0, 1.3, 3.3]]], dtype=np.float64)
    z_coord = SimpleNamespace(
        nemo_gdept_0=jnp.asarray(gdept0), nemo_gdepw_0=jnp.asarray(gdepw0),
        linear_free_surface=False)
    e3u = jnp.asarray(
        [[[1.2, 2.0, 3.0], [1.3, 2.0, 3.0]],
         [[1.4, 2.0, 3.0], [1.5, 2.0, 3.0]]], dtype=jnp.float64)

    def live(ssh):
        gd, gw, stretch = _nemo_qco_live_slope_depths(
            ssh, H, z_coord, jnp.float64)
        zu = _nemo_literal_slope_face_depth(gd, e3u, axis=1)
        return gd, gw, stretch, zu

    gd, gw, stretch, zu = jax.jit(live)(eta)
    eta_np, H_np = np.asarray(eta), np.asarray(H)
    r1_H = np.float64(1.0) / H_np
    want_stretch = np.maximum(
        np.float64(1.0) + eta_np * r1_H, np.float64(1.0e-6))
    want_gd = gdept0 * want_stretch[..., None]
    want_gw = gdepw0 * want_stretch[..., None]
    pair = want_gd + np.roll(want_gd, -1, axis=1)
    want_zu = np.float64(0.5) * (pair - np.asarray(e3u)[..., :1])
    np.testing.assert_array_equal(np.asarray(stretch), want_stretch)
    np.testing.assert_array_equal(np.asarray(gd), want_gd)
    np.testing.assert_array_equal(np.asarray(gw), want_gw)
    eager_zu = np.asarray(live(eta)[3])
    np.testing.assert_array_max_ulp(np.asarray(zu), eager_zu, maxulp=2)
    np.testing.assert_allclose(
        np.asarray(zu), want_zu, rtol=0.0, atol=np.float64(5.0e-16))

    # Planted old association: subtracting the full surface thickness after
    # the half multiply must differ at every represented wet column.
    wrong = np.float64(0.5) * pair - np.asarray(e3u)[..., :1]
    assert np.any(np.asarray(zu).view(np.uint64) != wrong.view(np.uint64))
    tangent = jax.grad(lambda ssh: jnp.sum(live(ssh)[3]))(eta)
    assert np.all(np.isfinite(np.asarray(tangent)))


def test_nemo_reciprocal_metric_matches_hand_computed_rounding_and_is_red():
    """0.1/7 differs by one ULP from 0.1*(1/7) in binary64."""
    zg = jnp.asarray([[[0.1]]], dtype=jnp.float64)
    metric = jnp.asarray([[7.0]], dtype=jnp.float64)

    def literal(z, e):
        reciprocal = jax.lax.optimization_barrier(jnp.float64(1.0) / e)
        return jax.lax.optimization_barrier(z * reciprocal[:, :, None])

    want = np.float64(0.1) * (np.float64(1.0) / np.float64(7.0))
    got = np.asarray(jax.jit(literal)(zg, metric))[0, 0, 0]
    divided = np.float64(0.1) / np.float64(7.0)
    assert got == want
    assert got.view(np.uint64) != divided.view(np.uint64)
    assert np.isfinite(np.asarray(jax.grad(
        lambda x: jnp.sum(literal(x, metric)))(zg))).all()
