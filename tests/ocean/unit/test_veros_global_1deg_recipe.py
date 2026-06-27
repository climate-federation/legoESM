"""Tests for the legoESM-Veros global_1deg transfer recipe.

THE STOCK-RESOLUTION GLOBAL TRANSFER TEST: maps every
``veros/setups/global_1deg/global_1deg.py`` setting onto existing canonical
config options (SCOPING_three_setups.md SS C;
``packages/ocean/legoesm/ocean/fidelity/veros_global_1deg_recipe.py``).
This file verifies:

  1. Config mapping — each Veros setting -> legoESM field, including the
     EKE FLIP BACK (``enable_eke_isopycnal_diffusion=True``, unlike
     global_4deg), the GM/Redi deltas (K_iso_steep=50, S_max=5e-3,
     frac=1.0), the A_h=5e4 setup literal, ``tke_mxl_choice=1`` (THE one
     TKE delta vs the 4deg/flexible block), the penetrative-shortwave
     flux_feedback channel, and the STOCK synchronous dt (1800, ratio 1).
  2. Grid/vertical builders — uniform-axis placement on the transcribed
     Veros xt/yt (x_origin=91, y_origin=-79, cyclic ghosts), dz-from-file
     orientation + the loud guards.
  3. Data-prep pure helpers — the salt zero-count kbot rule + bathymetry
     mask + kbot<nz fixup + the THREE channel closures (transcription
     MUTATION-PINNED cell-for-cell), the interface-snap full-cell
     degeneration, the MIT-grid tau shift, layout bridges, area weights.
  4. The recipe constructs (and, slow-marked, steps ONCE finite) on
     SYNTHETIC full-size data (no netCDF) with the full faithful stack
     including the q_solar column.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.precision import PrecisionPolicy, set_policy

set_policy(PrecisionPolicy.fp64())

from legoesm.core.field import Field
from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
    GLOBAL4_EKE_CONFIG,
    GLOBAL4_GM_REDI_CONFIG,
    GLOBAL4_TKE_CONFIG,
    VEROS_GLOBAL4_CP0,
)
from legoesm.ocean.fidelity.veros_global_1deg_recipe import (
    CHANNEL_CLOSURES,
    DT_MOM_RATIO, DT_S,
    GLOBAL_1DEG_A_H, GLOBAL_1DEG_EKE_CONFIG, GLOBAL_1DEG_GM_REDI_CONFIG,
    GLOBAL_1DEG_TKE_CONFIG,
    NX, NY, NZ,
    build_global_1deg_grid,
    build_global_1deg_model_config,
    build_global_1deg_recipe,
    build_global_1deg_z_coord,
    kbot_to_mask_and_h_bathy_1deg,
    replicate_veros_kbot_1deg,
    veros_area_t_1deg,
    veros_full_axes_1deg,
    veros_mit_tau_shift_1deg,
    veros_xy_to_legoesm_1deg,
    veros_xyz_to_legoesm_1deg,
)
from legoesm.ocean.fidelity.veros_state_bridge import veros_u_centered_z_centres


def _synthetic_dz() -> np.ndarray:
    """A surface-first 115-level dz profile in the file's class (10 m
    surface cells growing to ~83 m; the REAL file is 10 m x ~40 then
    growth to 83.33 m, H = 5500.13 m)."""
    dz = np.concatenate([
        np.full(40, 10.0),
        np.geomspace(10.0, 83.333336, 75),
    ])
    return dz


# ---------------------------------------------------------------------------
# 1. Config mapping
# ---------------------------------------------------------------------------


def test_gm_redi_mapping():
    """K_iso_0=1000, K_iso_steep=50, iso_dslope=iso_slopec=0.005 ->
    S_max=5e-3, taper_width_frac=1.0 (the global_flexible values)."""
    gm = GLOBAL_1DEG_GM_REDI_CONFIG
    assert gm.kappa_Redi == 1000.0          # K_iso_0 (EKE-driven; fallback)
    assert gm.kappa_GM == 1000.0            # EKE-off fallback
    assert gm.K_iso_steep == 50.0           # *** delta vs 4deg's 1000 ***
    assert gm.S_max == 5.0e-3               # iso_slopec
    assert gm.taper_width_frac == 1.0       # iso_dslope / iso_slopec
    assert gm.slope_density == "neutral"
    assert gm.implicit_K33 is True
    assert gm.veros_triad_weights is True
    assert gm.double_redi_diagonal is True
    assert gm.eke is not None               # enable_eke = True


def test_gm_redi_shares_4deg_invariants():
    """Fields the setup does NOT change must equal the matched 4deg block."""
    for f in GLOBAL_1DEG_GM_REDI_CONFIG._fields:
        if f in ("S_max", "taper_width_frac", "K_iso_steep", "eke"):
            continue
        assert getattr(GLOBAL_1DEG_GM_REDI_CONFIG, f) == \
            getattr(GLOBAL4_GM_REDI_CONFIG, f), f


def test_eke_isopycnal_diffusion_flip_back():
    """THE FLIP BACK: enable_eke_isopycnal_diffusion=True in THIS setup
    (global_1deg.py:79) — unlike global_4deg's settings-default False."""
    assert GLOBAL_1DEG_EKE_CONFIG.isopycnal_diffusion is True
    assert GLOBAL4_EKE_CONFIG.isopycnal_diffusion is False   # the contrast
    for f in GLOBAL_1DEG_EKE_CONFIG._fields:
        if f == "isopycnal_diffusion":
            continue
        assert getattr(GLOBAL_1DEG_EKE_CONFIG, f) == \
            getattr(GLOBAL4_EKE_CONFIG, f), f


def test_tke_mxl_choice_is_the_one_delta():
    """tke_mxl_choice=1 (global_1deg.py:64) is the ONLY TKE field that
    differs from the verbatim 4deg/flexible block — mutation-pinned."""
    t = GLOBAL_1DEG_TKE_CONFIG
    assert t.tke_mxl_choice == 1            # *** THE delta ***
    assert GLOBAL4_TKE_CONFIG.tke_mxl_choice == 2
    for f in t._fields:
        if f == "tke_mxl_choice":
            continue
        assert getattr(t, f) == getattr(GLOBAL4_TKE_CONFIG, f), f
    # the shared knobs the setup repeats
    assert t.c_k == 0.1 and t.c_eps == 0.7 and t.alpha_tke == 30.0
    assert t.advection_scheme == "superbee"
    assert t.source_bottom_drag_diss is False     # r_bot = 0
    assert t.prognostic is True


def test_mxl_choice_1_post_mixing_is_accepted_and_debt_safe():
    """The faithful tke_mxl_choice=1 config (post-mixing Veros surface
    correction) is now ACCEPTED by _validate_post_mixing_cfg — the
    distance-to-boundary cap (veros_mxl_choice1_boundary_cap) makes it
    debt-safe, so no fallback is needed. Both choices validate; only an
    UNKNOWN choice raises (in compute_mixing_lengths, not the timing
    guard)."""
    from legoesm.ocean.physics.vertical_mixing.tke import (
        _validate_post_mixing_cfg,
    )
    tke1 = GLOBAL_1DEG_TKE_CONFIG
    assert tke1.tke_mxl_choice == 1                       # recipe is faithful
    assert tke1.buoyancy_timing == "post_mixing_veros"
    assert tke1.positivity == "veros_surface_correction"
    # No longer raises — choice=1 is debt-safe under post-mixing.
    _validate_post_mixing_cfg(tke1)
    _validate_post_mixing_cfg(tke1._replace(tke_mxl_choice=2))


def test_mxl_choice1_boundary_cap_bounds_the_surface_blowup():
    """veros_mxl_choice1_boundary_cap reproduces Veros tke.py:43-47: the
    raw buoyancy length overflows where N²→0 at the surface, and the cap
    clamps it to the distance-to-surface (a few metres). Non-vacuous: the
    UNCAPPED length is orders of magnitude larger."""
    import jax.numpy as jnp
    import numpy as np
    from legoesm.ocean.physics.vertical_mixing.tke import (
        veros_mxl_choice1_boundary_cap, compute_mixing_lengths,
    )
    dz_ref = jnp.array([10., 30., 60., 100., 150., 250.])
    z_half = jnp.concatenate([jnp.array([0.]), -jnp.cumsum(dz_ref)])
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    dz_half_ref = z_full[:-1] - z_full[1:]
    z_interface = z_half[1:-1]
    column_depth = jnp.sum(dz_ref)
    cap = veros_mxl_choice1_boundary_cap(z_interface, dz_half_ref, column_depth)
    # cap == min(-zw + dzw/2, ht + zw), all positive on wet interfaces
    np.testing.assert_allclose(
        np.asarray(cap),
        np.minimum(np.asarray(-z_interface + 0.5 * dz_half_ref),
                   np.asarray(column_depth + z_interface)))
    assert bool(np.all(np.asarray(cap) > 0.0))
    # N²→0 at the surface interface ⇒ raw length blows up; cap clamps it
    N2 = jnp.array([1e-13, 1e-5, 1e-4, 1e-4, 1e-4])
    e = jnp.array([1e-2, 1e-3, 1e-4, 1e-4, 1e-4])
    cfg1 = GLOBAL_1DEG_TKE_CONFIG
    l_raw, _ = compute_mixing_lengths(e, N2, dz_half_ref, cfg1, signed_n2=True)
    l_cap, l_eps = compute_mixing_lengths(
        e, N2, dz_half_ref, cfg1, signed_n2=True, boundary_cap=cap)
    assert float(np.max(l_raw)) > 1e4               # the uncapped blowup
    assert float(np.max(l_cap)) <= float(np.max(np.asarray(cap))) + 1e-9
    assert bool(np.all(np.asarray(l_cap) >= cfg1.mxl_min))
    assert bool(np.all(np.asarray(l_cap) <= np.asarray(cap) + 1e-9))
    np.testing.assert_allclose(np.asarray(l_cap), np.asarray(l_eps))


def test_dycore_and_stepping_mapping():
    cfg = build_global_1deg_model_config()
    # STOCK synchronous dt pair — this row mirrors Veros exactly.
    assert DT_S == 1800.0
    assert cfg.dt_mom_ratio == 1.0 == DT_MOM_RATIO
    assert cfg.eos == "veros_gsw"           # eq_of_state_type = 5
    assert cfg.bottom_drag.bottom_drag_r == 0.0         # Veros r_bot default
    # A_h is a setup LITERAL (5e4), not the degtom**3 formula.
    assert cfg.A_h == 5.0e4 == GLOBAL_1DEG_A_H
    assert cfg.A_h_lat_scaling is True and cfg.A_h_cos_power == 1
    assert cfg.lateral_viscosity_operator == "flux_divergence"
    assert cfg.outer_integrator == "ab2"
    assert cfg.ab2_scope == "advective"
    assert cfg.barotropic.barotropic_solver == "rigid_lid"
    assert cfg.coriolis_scheme == "explicit_ab2"
    assert cfg.momentum_friction_additive is True
    assert cfg.vertical_momentum_scheme == "centered_full"
    assert cfg.momentum_advection == "flux_form"
    assert cfg.momentum_flux_scheme == "centered"
    assert cfg.tracer_advection == "centered"
    assert cfg.implicit_vertical_mixing is True
    assert cfg.K_v == 0.0
    assert cfg.surface_forcing_implicit is True
    assert cfg.constants is VEROS_CONSTANTS_CONFIG
    assert cfg.rho_0 == 1024.0


def test_flux_feedback_solar_mapping():
    """The penetrative-shortwave channel is ON with Jerlov type I (== the
    setup literals 0.58/0.35/23.0) and the kernel cp_0 literal."""
    cfg = build_global_1deg_model_config()
    sf = cfg.physics.surface_forcing
    assert sf.scheme == "flux_feedback"
    ff = sf.flux_feedback
    assert ff.penetrative_shortwave is True
    assert ff.shortwave_water_type == "I"
    from legoesm.ocean.physics.shortwave_penetration import JERLOV_TYPES
    R, z1, z2 = JERLOV_TYPES["I"]
    assert (R, z1, z2) == (0.58, 0.35, 23.0)
    assert ff.c_sw == 3991.86795711963 == VEROS_GLOBAL4_CP0
    assert ff.rho_0 == VEROS_CONSTANTS_CONFIG.rho_0 == cfg.rho_0
    assert ff.tau_restore_s == 30.0 * 86400.0
    assert ff.ice_mask is True
    assert cfg.physics.shortwave_penetration is None   # double-count guard


def test_explicit_ab2_coriolis_margin_at_stock_dt():
    """|f|*dt_mom at the most poleward INTERIOR centre (79.5 deg) sits
    comfortably inside the explicit-AB2 margin at the STOCK dt=1800 —
    the documented reason NO dt deviation is needed on this row."""
    _, yt, _ = veros_full_axes_1deg()
    lat_max = np.max(np.abs(yt[2:-2]))
    assert lat_max == pytest.approx(79.5, abs=1e-12)
    f_max = 2.0 * VEROS_CONSTANTS_CONFIG.Omega * np.sin(np.deg2rad(lat_max))
    assert f_max * DT_S < 0.30
    assert f_max * DT_S == pytest.approx(0.258, abs=0.005)


# ---------------------------------------------------------------------------
# 2. Grid / vertical builders
# ---------------------------------------------------------------------------


def test_full_axes_uniform_and_cyclic_ghosts():
    xt, yt, yu = veros_full_axes_1deg()
    assert xt.shape == (NX + 4,) and yt.shape == (NY + 4,)
    # interior: uniform 1 degree, first centre = origin - 1/2
    assert xt[2] == pytest.approx(90.5, abs=1e-12)
    assert xt[-3] == pytest.approx(449.5, abs=1e-12)
    np.testing.assert_allclose(np.diff(xt[2:-2]), 1.0, atol=1e-12)
    # cyclic x ghost VALUES copied from the far side (calc_grid)
    np.testing.assert_allclose(xt[:2], xt[-4:-2])
    np.testing.assert_allclose(xt[-2:], xt[2:4])
    # y_origin is the north face of the first interior cell
    assert yu[2] == pytest.approx(-79.0, abs=1e-12)
    assert yt[2] == pytest.approx(-79.5, abs=1e-12)
    assert yt[-3] == pytest.approx(79.5, abs=1e-12)
    np.testing.assert_allclose(np.diff(yt), 1.0, atol=1e-12)


def test_grid_alignment_veros_xt_yt():
    """Interior centres land EXACTLY on the transcribed Veros calc_grid
    axes; wall rows coincide with Veros's first ghost rows."""
    grid = build_global_1deg_grid()
    assert grid.n_lat == NY + 2 and grid.n_lon == NX
    xt, yt, _ = veros_full_axes_1deg()
    lat = np.degrees(np.asarray(grid.lat))
    lon = np.degrees(np.asarray(grid.lon))
    np.testing.assert_allclose(lat[1:-1], yt[2:-2], rtol=0, atol=1e-9)
    assert lat[0] == pytest.approx(yt[1], abs=1e-9)
    assert lat[-1] == pytest.approx(yt[-2], abs=1e-9)
    np.testing.assert_allclose(lon, xt[2:-2], rtol=0, atol=1e-9)


def test_grid_coriolis_full_sin_lat():
    grid = build_global_1deg_grid()
    f = np.asarray(grid.f)[:, 0]
    lat = np.asarray(grid.lat)
    np.testing.assert_allclose(
        f, 2.0 * VEROS_CONSTANTS_CONFIG.Omega * np.sin(lat), rtol=1e-12)


def test_z_coord_from_file_dz():
    dz = _synthetic_dz()
    zc = build_global_1deg_z_coord(dz)
    assert zc.n_levels == NZ
    np.testing.assert_allclose(np.asarray(zc.dz_ref), dz)     # k=0 surface
    assert float(zc.dz_ref[0]) == 10.0
    assert zc.H_max == pytest.approx(dz.sum())
    # centres are the u_centered recursion (Veros dzw metric), interleaving
    zf = np.asarray(zc.z_full_ref)
    zh = np.asarray(zc.z_half_ref)
    assert np.all(zf < zh[:-1]) and np.all(zf > zh[1:])
    assert np.all(np.asarray(zc.dz_half_ref) > 0)
    np.testing.assert_allclose(zf, veros_u_centered_z_centres(dz))


def test_z_coord_loud_guards():
    dz = _synthetic_dz()
    with pytest.raises(ValueError, match="SURFACE-FIRST"):
        build_global_1deg_z_coord(dz[::-1])          # Veros-order input
    with pytest.raises(ValueError, match="profile"):
        build_global_1deg_z_coord(dz[:-1])           # wrong level count
    bad = dz.copy()
    bad[3] = -1.0
    with pytest.raises(ValueError, match="positive"):
        build_global_1deg_z_coord(bad)


# ---------------------------------------------------------------------------
# 3. kbot / closures / mask
# ---------------------------------------------------------------------------


def _all_wet_world(n_zero: int = 0):
    """bathymetry < 0 everywhere; salt has ``n_zero`` zero (land-sentinel)
    levels at the BOTTOM of every column (Veros z-order: k=0 deepest)."""
    bathy = np.full((NX, NY), -5000.0)
    salt = np.full((NX, NY, NZ), 35.0)
    if n_zero:
        salt[:, :, :n_zero] = 0.0
    return bathy, salt


def test_kbot_salt_count_rule():
    """kbot = 1 + count(salt==0) over z, zeroed by the bathymetry mask and
    the kbot<nz fixup (order-faithful to set_topography)."""
    bathy, salt = _all_wet_world(n_zero=3)
    salt[7, 9, :10] = 0.0                    # a shallower column: kbot 11
    bathy[5, 5] = 0.0                        # bathymetry land
    salt[6, 6, : NZ - 1] = 0.0               # kbot = NZ -> only-surface fixup
    salt[6, 7, :] = 0.0                      # kbot = NZ+1 -> > nz -> 0
    kbot = replicate_veros_kbot_1deg(bathy, salt)
    assert kbot[0, 9] == 4                   # 1 + 3
    assert kbot[7, 9] == 11
    assert kbot[5, 5] == 0                   # bathymetry mask
    assert kbot[6, 6] == 0                   # kbot==nz fixup
    assert kbot[6, 7] == 0


def test_channel_closures_transcription():
    """The THREE set_topography closures, cell-for-cell: on an ALL-WET
    world the land set must be EXACTLY the transcribed index boxes
    (any mutation of an index bound flips a specific cell)."""
    bathy, salt = _all_wet_world()
    kbot = replicate_veros_kbot_1deg(bathy, salt)
    expected_land = np.zeros((NX, NY), dtype=bool)
    # Indonesian strip: i = 207..213, j = 0..4 (35 cells)
    expected_land[207:214, 0:5] = True
    # Aleutian cell: (104, 134)
    expected_land[104, 134] = True
    # English Channel: i = 269..270, j = 130 (2 cells)
    expected_land[269:271, 130] = True
    np.testing.assert_array_equal(kbot == 0, expected_land)
    assert int(expected_land.sum()) == 35 + 1 + 2
    # and the wet cells are untouched (kbot = 1 on the all-wet world)
    assert (kbot[~expected_land] == 1).all()
    # the module-level table matches the Veros source literals
    assert CHANNEL_CLOSURES == (
        (207, 214, 0, 5), (104, 105, 134, 135), (269, 271, 130, 131))


def test_kbot_to_mask_h_bathy_full_cell_snap():
    from legoesm.ocean.vertical import create_partial_cell_coordinate
    dz = _synthetic_dz()
    zc_ref = build_global_1deg_z_coord(dz)
    bathy, salt = _all_wet_world(n_zero=3)
    salt[3, 12, : NZ - 5] = 0.0              # a shelf column: kbot = NZ-4
    kbot = replicate_veros_kbot_1deg(bathy, salt)
    land_mask, H_bathy = kbot_to_mask_and_h_bathy_1deg(kbot, zc_ref)
    assert land_mask.shape == (NY + 2, NX)
    assert land_mask[0].sum() == 0 and land_mask[-1].sum() == 0
    iface = np.concatenate([[0.0], np.cumsum(dz)])
    wet = land_mask > 0
    assert np.all(np.isin(H_bathy[wet].round(9), iface.round(9)))
    zc = build_global_1deg_z_coord(dz)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H_bathy))
    ia = np.asarray(pc.is_active)
    hp = np.asarray(pc.h_partial)
    np.testing.assert_allclose(
        np.where(ia, hp, 0.0),
        np.where(ia, np.broadcast_to(dz, ia.shape), 0.0))
    # n_wet mapping: kbot k => nz-k+1 active cells (shelf: 5 wet cells)
    assert kbot[3, 12] == NZ - 4
    assert int(ia[13, 3].sum()) == 5
    # closures are land in the mask too
    assert land_mask[1 + 2, 210] == 0.0      # Indonesian strip (j=2)
    assert land_mask[1 + 134, 104] == 0.0    # Aleutian
    assert land_mask[1 + 130, 269] == 0.0    # English Channel


# ---------------------------------------------------------------------------
# 4. Layout bridges / tau shift / weights
# ---------------------------------------------------------------------------


def test_mit_tau_shift():
    """surface_taux[i] = taux[i+1] and surface_tauy[j] = tauy[j+1], BOTH
    pulling the ZERO (never-exchanged) Veros ghost into the last interior
    column/row — measured on the live oracle's t=0 surface_taux/y (a
    cyclic roll fails the harness gate by 0.2 N/m²)."""
    rng = np.random.default_rng(3)
    taux = rng.normal(size=(NX, NY, 12))
    tauy = rng.normal(size=(NX, NY, 12))
    tx, ty = veros_mit_tau_shift_1deg(taux, tauy)
    np.testing.assert_array_equal(tx[:-1], taux[1:])
    np.testing.assert_array_equal(tx[-1], 0.0)           # zero x ghost (!)
    np.testing.assert_array_equal(ty[:, :-1], tauy[:, 1:])
    np.testing.assert_array_equal(ty[:, -1], 0.0)        # zero y ghost


def test_layout_bridges():
    arr = np.zeros((NX, NY, NZ))
    arr[5, 7, NZ - 1] = 42.0            # SURFACE in Veros z-order
    arr[5, 7, 0] = -7.0                 # DEEPEST
    out = veros_xyz_to_legoesm_1deg(arr)
    assert out.shape == (NY + 2, NX, NZ)
    assert out[8, 5, 0] == 42.0
    assert out[8, 5, NZ - 1] == -7.0
    assert (out[0] == 0).all() and (out[-1] == 0).all()
    out2 = veros_xy_to_legoesm_1deg(
        np.arange(NX * NY, dtype=float).reshape(NX, NY))
    assert out2.shape == (NY + 2, NX)
    assert out2[1, 0] == 0.0 and out2[2, 0] == 1.0


def test_area_weights_uniform_1deg():
    _, yt, _ = veros_full_axes_1deg()
    a = veros_area_t_1deg(yt[2:-2])
    assert a.shape == (NY,)
    assert np.all(a > 0)
    degtom = VEROS_CONSTANTS_CONFIG.R_earth * np.pi / 180.0
    j = NY // 2                              # yt = +0.5 deg
    expected = degtom * degtom * np.cos(np.deg2rad(yt[2 + j]))
    assert a[j] == pytest.approx(expected, rel=1e-12)
    assert a[j] > a[0]                       # equator row biggest


# ---------------------------------------------------------------------------
# 5. Recipe constructs + steps once on synthetic data (no netCDF)
# ---------------------------------------------------------------------------


def _synthetic_inputs():
    """Fabricated full-size world: an open zonal band crossing the closure
    boxes (so the closures visibly bite) + shelf structure."""
    dz = _synthetic_dz()
    bathy = np.zeros((NX, NY))
    salt = np.zeros((NX, NY, NZ))
    temp = np.zeros((NX, NY, NZ))
    zt = veros_u_centered_z_centres(dz)      # k=0 surface, negative
    for j in range(2, NY - 2):
        bathy[:, j] = -5000.0
        n_land = 3 if (j % 5) else NZ - 10   # shelving rows
        salt[:, j, n_land:] = 35.0           # Veros z-order: wet ABOVE kbot
        for k in range(NZ):
            # surface-first profile written into Veros order (reverse k)
            temp[:, j, NZ - 1 - k] = 2.0 + 10.0 * np.exp(zt[k] / 800.0)
        temp[:, j, :n_land] = 0.0
    return dz, bathy, salt, temp


def test_recipe_constructs_on_synthetic_data():
    dz, bathy, salt, temp = _synthetic_inputs()
    recipe = build_global_1deg_recipe(dz, bathy, salt, temp)
    st = recipe.initial_state
    assert st.T.data.shape == (NY + 2, NX, NZ)
    ia = np.asarray(recipe.z_coord.is_active)
    T = np.asarray(st.T.data)
    assert (T[~ia.astype(bool)] == 0.0).all()
    assert st.tke is not None and st.dtke is not None
    assert st.eke is not None and st.eke_diss is not None
    assert st.tke.data.shape == (NY + 2, NX, NZ - 1)
    # closures cut the band: Indonesian strip cells are land
    lm = np.asarray(st.land_mask.data)
    assert lm[1 + 3, 210] == 0.0
    assert lm[1 + 134, 104] == 0.0


@pytest.mark.slow
def test_recipe_steps_once_finite_synthetic_with_solar():
    """Full stack + the q_solar column steps once finite at the STOCK
    360x160x115 size (slow: one jitted step of the full model).  Runs the
    FAITHFUL tke_mxl_choice=1 (no fallback) — the distance-to-boundary cap
    makes it debt-safe (see test_mxl_choice1_boundary_cap_bounds_the_surface
    _blowup + the recipe docstring)."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    dz, bathy, salt, temp = _synthetic_inputs()
    recipe = build_global_1deg_recipe(dz, bathy, salt, temp)
    cfg = recipe.model_config
    assert cfg.physics.vertical_mixing.tke.tke_mxl_choice == 1   # faithful
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    model.check_coriolis_stability(DT_S)

    state = recipe.initial_state
    _z = lambda d: Field(data=jnp.zeros_like(d.data),
                         name=d.name + "_incr_prev", dims=d.dims,
                         units=d.units)
    state = state._replace(
        T_incr_prev=_z(state.T), S_incr_prev=_z(state.S),
        u_incr_prev=_z(state.u), v_incr_prev=_z(state.v))
    rl = model._ensure_rigid_lid_data(state)
    zV = jnp.zeros((recipe.grid.n_lat + 1, recipe.grid.n_lon + 1),
                   dtype=state.u.data.dtype)
    zI = jnp.zeros((rl.nisle,), dtype=state.u.data.dtype)
    state = state._replace(psi=zV, dpsi=zV, dpsi_prev=zV, dpsin=zI,
                           dpsin_prev=zI)

    n_lat, n_lon = recipe.grid.n_lat, recipe.grid.n_lon
    sf = OceanSurfaceForcing(
        tau_x=jnp.full((n_lat, n_lon), -0.05),
        tau_y=jnp.zeros((n_lat, n_lon)),
        q_prescribed=jnp.full((n_lat, n_lon), 10.0),
        q_feedback=jnp.full((n_lat, n_lon), 20.0),
        T_feedback_target=jnp.full((n_lat, n_lon), 12.0),
        S_restore_target=jnp.full((n_lat, n_lon), 35.0),
        q_solar=jnp.full((n_lat, n_lon), 150.0),
    )
    after = model.step(state, dt=DT_S, surface_forcing=sf)
    jax.block_until_ready(after.u.data)
    for fld in ("u", "v", "T", "S", "tke", "eke"):
        arr = np.asarray(getattr(after, fld).data)
        assert np.all(np.isfinite(arr)), f"non-finite {fld} after one step"
    assert float(np.max(np.abs(np.asarray(after.eta.data)))) < 1e-6
    # the solar column actually warmed sub-surface water somewhere
    dT_sub = np.asarray(after.T.data - state.T.data)[1:-1, :, 1:4]
    assert float(np.max(dT_sub)) > 0.0
