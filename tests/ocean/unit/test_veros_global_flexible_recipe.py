"""Tests for the legoESM-Veros global_flexible transfer recipe.

THE STRETCHED-GRID GLOBAL TRANSFER TEST: maps every
``veros/setups/global_flexible/global_flexible.py`` setting onto existing
canonical config options (SCOPING_three_setups.md §B;
``packages/ocean/legoesm/ocean/fidelity/veros_global_flexible_recipe.py``).
This file verifies:

  1. Config mapping — each Veros setting → legoESM field, including the EKE
     FLIP BACK (``enable_eke_isopycnal_diffusion=True``, unlike global_4deg),
     the GM/Redi deltas (K_iso_steep=50, S_max=5e-3, frac=1.0), the A_h=5e4
     setup literal, the penetrative-shortwave flux_feedback channel
     (EXT-F2), and the documented rescaled dt pair (1800/14400, ratio 8).
  2. The Vinokur transcription — anchored on the Veros docstring example and
     (when a veros install is importable) bit-parity with the live
     ``veros.tools.get_vinokur_grid_steps``.
  3. Grid/vertical builders — stretched-grid placement on the transcribed
     Veros yt/yu, the z-coordinate orientation + the LOUD nz degeneracy
     guard (nz=15/18 measured-degenerate).
  4. Data-prep pure helpers — kbot nearest-level rule + kbot==nz fixup +
     marginal-sea morphology (incl. the cyclic ghost frame), the
     interface-snap full-cell degeneration, the MIT-grid tau shift, the
     interpolate/fill_holes transcription, layout bridges.
  5. The recipe constructs and steps ONCE finite on SYNTHETIC data (no
     netCDF) with the full faithful stack INCLUDING the q_solar column.
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
from legoesm.ocean.fidelity.veros_global_flexible_recipe import (
    DT_MOM_RATIO, DT_MOM_S, DT_TRACER_S,
    GLOBAL_FLEX_A_H, GLOBAL_FLEX_EKE_CONFIG, GLOBAL_FLEX_GM_REDI_CONFIG,
    GLOBAL_FLEX_TKE_CONFIG,
    NX, NY, NZ,
    build_global_flexible_grid,
    build_global_flexible_model_config,
    build_global_flexible_recipe,
    build_global_flexible_z_coord,
    global_flexible_dyt_deg,
    global_flexible_dzt_veros,
    kbot_to_mask_and_h_bathy_flexible,
    prepare_global_flexible_topography,
    replicate_veros_kbot_flexible,
    veros_area_t_flexible,
    veros_fill_holes,
    veros_full_axes,
    veros_interpolate,
    veros_mit_tau_shift,
    veros_vinokur_grid_steps,
    veros_xy_to_legoesm_flex,
    veros_xyz_to_legoesm_flex,
)
from legoesm.ocean.fidelity.veros_state_bridge import veros_u_centered_z_centres


# ---------------------------------------------------------------------------
# 1. Config mapping
# ---------------------------------------------------------------------------


def test_gm_redi_mapping():
    """K_iso_0=1000, K_iso_steep=50, iso_dslope=iso_slopec=0.005 →
    S_max=5e-3, taper_width_frac=1.0."""
    gm = GLOBAL_FLEX_GM_REDI_CONFIG
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


def test_eke_isopycnal_diffusion_flip_back():
    """THE FLIP BACK: enable_eke_isopycnal_diffusion=True in THIS setup
    (global_flexible.py:87) — unlike global_4deg's settings-default False."""
    assert GLOBAL_FLEX_EKE_CONFIG.isopycnal_diffusion is True
    assert GLOBAL4_EKE_CONFIG.isopycnal_diffusion is False   # the contrast
    # Every OTHER EKE field identical to the matched 4deg block.
    for f in GLOBAL_FLEX_EKE_CONFIG._fields:
        if f == "isopycnal_diffusion":
            continue
        assert getattr(GLOBAL_FLEX_EKE_CONFIG, f) == \
            getattr(GLOBAL4_EKE_CONFIG, f), f


def test_tke_block_reused_verbatim():
    """The setup repeats the global_4deg TKE block verbatim (incl. superbee
    advection, tke_mxl_choice=2, r_bot=0 ⇒ no bottom-drag source) — the
    recipe REUSES the same config object (factored, not copied)."""
    assert GLOBAL_FLEX_TKE_CONFIG is GLOBAL4_TKE_CONFIG
    t = GLOBAL_FLEX_TKE_CONFIG
    assert t.c_k == 0.1 and t.c_eps == 0.7 and t.alpha_tke == 30.0
    assert t.tke_mxl_choice == 2
    assert t.advection_scheme == "superbee"
    assert t.source_bottom_drag_diss is False
    assert t.prognostic is True


def test_gm_redi_shares_4deg_invariants():
    """Fields the setup does NOT change must equal the matched 4deg block."""
    for f in GLOBAL_FLEX_GM_REDI_CONFIG._fields:
        if f in ("S_max", "taper_width_frac", "K_iso_steep", "eke"):
            continue
        assert getattr(GLOBAL_FLEX_GM_REDI_CONFIG, f) == \
            getattr(GLOBAL4_GM_REDI_CONFIG, f), f


def test_dycore_and_stepping_mapping():
    cfg = build_global_flexible_model_config()
    # Documented rescaled dt pair (recipe docstring): 1800/14400, ratio 8.
    assert DT_MOM_S == 1800.0 and DT_TRACER_S == 14400.0
    assert cfg.dt_mom_ratio == 8.0 == DT_MOM_RATIO
    assert cfg.eos == "veros_gsw"           # eq_of_state_type = 5
    assert cfg.bottom_drag_r == 0.0         # Veros r_bot default
    # A_h is a setup LITERAL (5e4), not the degtom³ formula.
    assert cfg.A_h == 5.0e4 == GLOBAL_FLEX_A_H
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
    """EXT-F2: the penetrative-shortwave channel is ON with Jerlov type I
    (≡ the setup literals 0.58/0.35/23.0) and the kernel cp_0 literal."""
    cfg = build_global_flexible_model_config()
    sf = cfg.physics.surface_forcing
    assert sf.scheme == "flux_feedback"
    ff = sf.flux_feedback
    assert ff.penetrative_shortwave is True
    assert ff.shortwave_water_type == "I"
    # Jerlov "I" must be exactly the Veros setup literals.
    from legoesm.ocean.physics.shortwave_penetration import JERLOV_TYPES
    R, z1, z2 = JERLOV_TYPES["I"]
    assert (R, z1, z2) == (0.58, 0.35, 23.0)
    assert ff.c_sw == 3991.86795711963 == VEROS_GLOBAL4_CP0
    assert ff.rho_0 == VEROS_CONSTANTS_CONFIG.rho_0 == cfg.rho_0
    assert ff.tau_restore_s == 30.0 * 86400.0
    assert ff.ice_mask is True
    # The standalone shortwave_penetration scheme must be OFF (solar is
    # owned by flux_feedback — the double-count guard).
    assert cfg.physics.shortwave_penetration is None


def test_explicit_ab2_coriolis_margin():
    """|f|·dt_mom at the most poleward INTERIOR centre must sit inside the
    explicit-AB2 margin — the documented reason dt_mom=1800 (not the
    CFL-consistent 3600, which lands ~0.5 ON the margin)."""
    _, yt, _ = veros_full_axes()
    lat_max = np.max(np.abs(yt[2:-2]))
    f_max = 2.0 * VEROS_CONSTANTS_CONFIG.Omega * np.sin(np.deg2rad(lat_max))
    assert f_max * DT_MOM_S < 0.30
    assert f_max * 3600.0 > 0.45            # the rejected sync-3600 choice


# ---------------------------------------------------------------------------
# 2. Vinokur transcription
# ---------------------------------------------------------------------------


def test_vinokur_matches_veros_docstring_example():
    """Anchor: the values printed in the Veros get_vinokur_grid_steps
    docstring — ``(14, 180, 5, two_sided_grid=True)``."""
    ref = np.array([
        18.2451554, 17.23915939, 15.43744632, 13.17358802,
        10.78720589, 8.53852027, 6.57892471, 6.57892471,
        8.53852027, 10.78720589, 13.17358802, 15.43744632,
        17.23915939, 18.2451554,
    ])
    steps = veros_vinokur_grid_steps(14, 180.0, 5.0, two_sided_grid=True)
    np.testing.assert_allclose(steps, ref, atol=1e-7)
    assert abs(steps.sum() - 180.0) < 1e-10


@pytest.mark.parametrize("call", [
    dict(n_cells=NY, total_length=160.0, lower_stepsize=0.5 * 160.0 / NY,
         two_sided_grid=True),                                # dyt
    dict(n_cells=NZ, total_length=5400.0, lower_stepsize=10.0,
         refine_towards="lower"),                             # dzt
    dict(n_cells=160, total_length=160.0, lower_stepsize=0.5,
         two_sided_grid=True),                                # stock dyt
    dict(n_cells=60, total_length=5400.0, lower_stepsize=10.0,
         refine_towards="lower"),                             # stock dzt
])
def test_vinokur_bit_parity_with_live_veros(call):
    """When a veros install is importable, the transcription must agree
    with ``veros.tools.get_vinokur_grid_steps`` to float round-off for both
    setup calls at BOTH the comparison and the stock resolution."""
    veros_tools = pytest.importorskip("veros.tools")
    ours = veros_vinokur_grid_steps(**call)
    theirs = np.asarray(veros_tools.get_vinokur_grid_steps(**call))
    np.testing.assert_allclose(ours, theirs, rtol=1e-13, atol=0)


def test_vinokur_rejects_sinc_branch():
    with pytest.raises(ValueError, match="sinc branch"):
        veros_vinokur_grid_steps(10, 10.0, 2.0)   # s0 < 1


def test_vinokur_rejects_odd_two_sided():
    with pytest.raises(ValueError, match="even"):
        veros_vinokur_grid_steps(15, 160.0, 2.0, two_sided_grid=True)


def test_dyt_equator_refined_symmetric():
    d = global_flexible_dyt_deg()
    assert d.shape == (NY,)
    assert abs(d.sum() - 160.0) < 1e-9
    np.testing.assert_allclose(d, d[::-1], atol=1e-12)       # symmetric
    assert d.min() == pytest.approx(d[NY // 2], abs=0)        # finest at eq
    assert d[NY // 2] < 0.5 * 160.0 / NY * 1.25               # ~eq spacing
    assert d.max() == d[0] > d.min()                          # coarsest poles


def test_dzt_orientation_and_surface_cell():
    d = global_flexible_dzt_veros()
    assert d.shape == (NZ,)
    assert abs(d.sum() - 5400.0) < 1e-6
    assert np.all(np.diff(d) < 0)            # k=0 deepest = largest (Veros)
    assert 10.0 < d[-1] < 15.0               # ~min_depth surface cell


# ---------------------------------------------------------------------------
# 3. Grid / vertical builders
# ---------------------------------------------------------------------------


def test_grid_alignment_veros_xt_yt():
    """Interior centres land EXACTLY on the transcribed Veros calc_grid
    axes (u_centered recursion + yu[2]=y_origin / xu[2]=x_origin shifts)."""
    grid = build_global_flexible_grid()
    assert grid.n_lat == NY + 2 and grid.n_lon == NX
    xt, yt, yu = veros_full_axes()
    lat = np.degrees(np.asarray(grid.lat))
    lon = np.degrees(np.asarray(grid.lon))
    np.testing.assert_allclose(lat[1:-1], yt[2:-2], rtol=0, atol=1e-9)
    # wall rows coincide with Veros's first ghost rows (edge-extended dyt)
    assert lat[0] == pytest.approx(yt[1], abs=1e-9)
    assert lat[-1] == pytest.approx(yt[-2], abs=1e-9)
    # faces on yu
    lat_v = np.degrees(np.asarray(grid.lat_v))
    np.testing.assert_allclose(lat_v[1:-1], yu[1:-2], rtol=0, atol=1e-9)
    # 90-shifted uniform 4° lon axis: 88, 92, ..., 444
    np.testing.assert_allclose(lon, xt[2:-2], rtol=0, atol=1e-9)
    assert lon[0] == pytest.approx(90.0 - 0.5 * 360.0 / NX, abs=1e-9)


def test_grid_coriolis_full_sin_lat():
    grid = build_global_flexible_grid()
    f = np.asarray(grid.f)[:, 0]
    lat = np.asarray(grid.lat)
    np.testing.assert_allclose(
        f, 2.0 * VEROS_CONSTANTS_CONFIG.Omega * np.sin(lat), rtol=1e-12)


def test_z_coord_orientation_and_dzw():
    zc = build_global_flexible_z_coord()
    d_veros = global_flexible_dzt_veros()
    np.testing.assert_allclose(np.asarray(zc.dz_ref), d_veros[::-1])
    assert float(zc.dz_ref[0]) == pytest.approx(d_veros[-1])  # surface ~12 m
    assert zc.H_max == pytest.approx(5400.0, abs=1e-6)
    # centres are the u_centered recursion (Veros dzw metric), interleaving
    zf = np.asarray(zc.z_full_ref)
    zh = np.asarray(zc.z_half_ref)
    assert np.all(zf < zh[:-1]) and np.all(zf > zh[1:])
    assert np.all(np.asarray(zc.dz_half_ref) > 0)


@pytest.mark.parametrize("nz_bad", [15, 18])
def test_z_coord_degeneracy_guard(nz_bad):
    """The measured degeneracies: nz=15 (surface centre BELOW the next
    centre) and nz=18 (surface centre above z=0) must raise LOUDLY."""
    with pytest.raises(ValueError, match="degenerate"):
        build_global_flexible_z_coord(nz_bad)


def test_full_axes_cyclic_ghosts():
    xt, yt, yu = veros_full_axes()
    assert xt.shape == (NX + 4,) and yt.shape == (NY + 4,)
    # cyclic x ghost VALUES copied from the far side (calc_grid)
    np.testing.assert_allclose(xt[:2], xt[-4:-2])
    np.testing.assert_allclose(xt[-2:], xt[2:4])
    assert xt[2] == pytest.approx(90.0 - 0.5 * 360.0 / NX)
    # y_origin is the north face of the first interior cell
    assert yu[2] == pytest.approx(-80.0, abs=1e-12)
    assert np.all(np.diff(yt) > 0)


# ---------------------------------------------------------------------------
# 4. Data-prep pure helpers
# ---------------------------------------------------------------------------


def test_kbot_nearest_level_rule_synthetic():
    """kbot = 1 + argmin|z − zt| where z<0 (the nearest-LEVEL rule, not the
    4deg count rule); kbot==nz → 0 (only-surface-cell column is land)."""
    d_veros = global_flexible_dzt_veros()
    zt = veros_u_centered_z_centres(d_veros[::-1])[::-1]   # k=0 deepest
    z = np.zeros((NX, NY))
    # probes live inside an open zonal band so the marginal-sea morphology
    # (which removes ISOLATED wet cells — see the dedicated tests) keeps them
    z[:, 8:12] = zt[0]                  # deepest centre → kbot 1
    z[6, 9] = zt[3] + 0.01              # nearest level 3 → kbot 4
    z[7, 9] = -1.0                      # nearest the SURFACE centre → kbot=NZ
    kbot = replicate_veros_kbot_flexible(z)
    assert kbot[5, 9] == 1
    assert kbot[6, 9] == 4
    # z=-1 is nearest zt[NZ-1] (the ~-6 m surface centre) → kbot = NZ → 0
    # (the only-surface-cell fixup), and the morphology keeps it 0.
    assert kbot[7, 9] == 0
    assert (kbot[z >= 0] == 0).all()


def test_kbot_marginal_sea_removal():
    """An ENCLOSED interior sea is removed by the dilate→fill→erode chain;
    the cyclically-connected open ocean is kept (the ghost-frame semantics:
    cyclic x exchange happens BEFORE the morphology)."""
    d_veros = global_flexible_dzt_veros()
    zt = veros_u_centered_z_centres(d_veros[::-1])[::-1]
    deep = zt[0]
    z = np.zeros((NX, NY))
    z[:, 10:14] = deep                  # an open zonal band (wraps cyclically)
    z[30:34, 25:29] = deep              # 4x4 enclosed sea
    kbot = replicate_veros_kbot_flexible(z)
    assert (kbot[:, 10:14] == 1).all(), "cyclic open band must survive"
    assert (kbot[30:34, 25:29] == 0).all(), "enclosed sea must be removed"


def test_kbot_one_cell_passage_closed():
    """A side basin connected only through a ONE-CELL passage is sealed by
    the binary dilation and removed (the Gibraltar-class behaviour)."""
    d_veros = global_flexible_dzt_veros()
    zt = veros_u_centered_z_centres(d_veros[::-1])[::-1]
    deep = zt[0]
    z = np.zeros((NX, NY))
    z[:, 10:16] = deep                  # open band
    z[40:46, 22:28] = deep              # side basin
    z[42, 16:22] = deep                 # 1-cell-wide channel into the band
    kbot = replicate_veros_kbot_flexible(z)
    assert (kbot[40:46, 22:28] == 0).all(), "1-cell-passage basin removed"
    assert (kbot[:, 10:16] == 1).all()


def test_kbot_to_mask_h_bathy_full_cell_snap():
    from legoesm.ocean.vertical import create_partial_cell_coordinate
    d_veros = global_flexible_dzt_veros()
    zt = veros_u_centered_z_centres(d_veros[::-1])[::-1]
    z = np.zeros((NX, NY))
    z[:, 10:30] = zt[0]
    z[3, 12] = zt[5]                    # a shelf column
    kbot = replicate_veros_kbot_flexible(z)
    land_mask, H_bathy = kbot_to_mask_and_h_bathy_flexible(kbot)
    assert land_mask.shape == (NY + 2, NX)
    assert land_mask[0].sum() == 0 and land_mask[-1].sum() == 0
    dz_ref = d_veros[::-1]
    iface = np.concatenate([[0.0], np.cumsum(dz_ref)])
    wet = land_mask > 0
    assert np.all(np.isin(H_bathy[wet].round(9), iface.round(9)))
    zc = build_global_flexible_z_coord()
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H_bathy))
    ia = np.asarray(pc.is_active)
    hp = np.asarray(pc.h_partial)
    np.testing.assert_allclose(
        np.where(ia, hp, 0.0),
        np.where(ia, np.broadcast_to(dz_ref, ia.shape), 0.0))
    # n_wet mapping: kbot k ⇒ nz-k+1 active cells.  The shelf column sits
    # exactly ON zt[5] ⇒ argmin index 5 ⇒ kbot = 1+5 = 6 ⇒ 15 wet cells.
    assert kbot[3, 12] == 6
    assert int(ia[13, 3].sum()) == NZ - 6 + 1


def test_mit_tau_shift():
    """surface_taux[i] = taux[i+1] with the cyclic wrap; surface_tauy[j] =
    tauy[j+1] with the zero north ghost pulled into the last row."""
    rng = np.random.default_rng(3)
    taux = rng.normal(size=(NX, NY, 12))
    tauy = rng.normal(size=(NX, NY, 12))
    tx, ty = veros_mit_tau_shift(taux, tauy)
    np.testing.assert_array_equal(tx[:-1], taux[1:])
    np.testing.assert_array_equal(tx[-1], taux[0])       # cyclic wrap
    np.testing.assert_array_equal(ty[:, :-1], tauy[:, 1:])
    np.testing.assert_array_equal(ty[:, -1], 0.0)        # zero y ghost


def test_interpolate_and_fill_holes():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0])
    v = x[:, None] + 2.0 * y[None, :]
    xi = np.array([0.5, 1.5, 3.5])          # 3.5 outside → NaN → filled
    yi = np.array([0.25, 1.75])
    out = veros_interpolate((x, y), v, (xi, yi))
    np.testing.assert_allclose(out[0], 0.5 + 2.0 * yi)
    np.testing.assert_allclose(out[1], 1.5 + 2.0 * yi)
    np.testing.assert_allclose(out[2], out[1])           # nearest-filled...
    # fill_holes: nearest finite by axis sweeps
    a = np.array([[np.nan, 1.0], [2.0, np.nan]])
    f = veros_fill_holes(a)
    assert np.isfinite(f).all()
    # nearest interp (no fill) leaves NaN outside the hull
    out2 = veros_interpolate((x, y), v, (xi, yi), kind="nearest", fill=False)
    assert np.isnan(out2[2]).all()


def test_layout_bridges():
    arr = np.zeros((NX, NY, NZ))
    arr[5, 7, NZ - 1] = 42.0            # SURFACE in Veros z-order
    arr[5, 7, 0] = -7.0                 # DEEPEST
    out = veros_xyz_to_legoesm_flex(arr)
    assert out.shape == (NY + 2, NX, NZ)
    assert out[8, 5, 0] == 42.0
    assert out[8, 5, NZ - 1] == -7.0
    assert (out[0] == 0).all() and (out[-1] == 0).all()
    out2 = veros_xy_to_legoesm_flex(
        np.arange(NX * NY, dtype=float).reshape(NX, NY))
    assert out2.shape == (NY + 2, NX)
    assert out2[1, 0] == 0.0 and out2[2, 0] == 1.0


def test_area_weights_on_stretched_rows():
    _, yt, _ = veros_full_axes()
    dyt = global_flexible_dyt_deg()
    a = veros_area_t_flexible(yt[2:-2], dyt)
    assert a.shape == (NY,)
    assert np.all(a > 0)
    # equator rows have the SMALLEST dy but largest cos → compare exactly
    degtom = VEROS_CONSTANTS_CONFIG.R_earth * np.pi / 180.0
    j = NY // 2
    expected = (360.0 / NX * degtom) * (dyt[j] * degtom) * np.cos(
        np.deg2rad(yt[2 + j]))
    assert a[j] == pytest.approx(expected, rel=1e-12)


def test_topography_prep_threshold_and_nearest():
    """Raw topo ≥ −1 m forces the smoothed value to 0 (land) before the
    nearest interp; deep cells survive smoothing."""
    topo_x = np.arange(0.0, 360.0, 2.0)          # 180 points
    topo_y = np.linspace(-89.0, 89.0, 90)
    topo_z = np.full((180, 90), -4000.0)
    topo_z[:, :10] = 0.0                          # polar land
    xt, yt, _ = veros_full_axes()
    z = prepare_global_flexible_topography(
        topo_x, topo_y, topo_z, xt[2:-2], yt[2:-2])
    assert z.shape == (NX, NY)
    assert (z <= 0.0).all()
    assert z.min() < -3000.0                      # deep ocean survives


# ---------------------------------------------------------------------------
# 5. Recipe constructs + steps once on synthetic data (no netCDF)
# ---------------------------------------------------------------------------


def _synthetic_inputs():
    """Tiny fabricated world: an open zonal band + shelves + an enclosed
    sea that the morphology must remove."""
    d_veros = global_flexible_dzt_veros()
    zt = veros_u_centered_z_centres(d_veros[::-1])[::-1]
    z_interp = np.zeros((NX, NY))
    for i in range(NX):
        for j in range(6, NY - 6):
            z_interp[i, j] = -5300.0 if (i + j) % 7 else -800.0
    z_interp[10:12, 14:16] = 0.0                 # islands
    temp = np.zeros((NX, NY, NZ))
    salt = np.zeros((NX, NY, NZ))
    for k in range(NZ):
        temp[:, :, k] = 2.0 + 10.0 * np.exp(zt[k] / 800.0)
        salt[:, :, k] = 35.0
    return z_interp, temp, salt


def test_recipe_constructs_on_synthetic_data():
    z_interp, temp, salt = _synthetic_inputs()
    recipe = build_global_flexible_recipe(z_interp, temp, salt)
    st = recipe.initial_state
    assert st.T.data.shape == (NY + 2, NX, NZ)
    ia = np.asarray(recipe.z_coord.is_active)
    T = np.asarray(st.T.data)
    assert (T[~ia.astype(bool)] == 0.0).all()
    assert st.tke is not None and st.dtke is not None
    assert st.eke is not None and st.eke_diss is not None
    assert st.tke.data.shape == (NY + 2, NX, NZ - 1)


@pytest.mark.slow
def test_recipe_steps_once_finite_synthetic_with_solar():
    """Full faithful stack + the q_solar column steps once finite."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    z_interp, temp, salt = _synthetic_inputs()
    recipe = build_global_flexible_recipe(z_interp, temp, salt)
    cfg = recipe.model_config
    LatLonCGridOceanModel._validate_config(cfg)
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    model.check_coriolis_stability(DT_TRACER_S)

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
    after = model.step(state, dt=DT_TRACER_S, surface_forcing=sf)
    jax.block_until_ready(after.u.data)
    for fld in ("u", "v", "T", "S", "tke", "eke"):
        arr = np.asarray(getattr(after, fld).data)
        assert np.all(np.isfinite(arr)), f"non-finite {fld} after one step"
    assert float(np.max(np.abs(np.asarray(after.eta.data)))) < 1e-6
    # the solar column actually warmed sub-surface water somewhere
    dT_sub = np.asarray(after.T.data - state.T.data)[1:-1, :, 1:4]
    assert float(np.max(dT_sub)) > 0.0
