"""Tests for the legoESM-Veros global_4deg transfer recipe.

THE GLOBAL TRANSFER TEST: maps every ``veros/setups/global_4deg/global_4deg.py``
setting onto existing canonical config options (SCOPING §E table,
``.physics-validator/transfer_global_4deg/SCOPING.md``).  This file verifies:

  1. Config mapping — each Veros setting → legoESM field, including the LOUD
     EKE flip (``enable_eke_isopycnal_diffusion`` ABSENT ⇒ Veros default False
     ⇒ ``EKEConfig.isopycnal_diffusion=False``, unlike the ACC's True), zero
     bottom friction (Veros r_bot default), the gsw EOS, TKE superbee
     advection, and the Veros kernel ``cp_0`` in FluxFeedbackConfig.
  2. Grid/vertical/bathymetry builders — Veros xt/yt alignment, the
     u_centered_grid zt recursion, kbot replication semantics, interface
     snapping (full-cell degeneration).
  3. Data-prep pure helpers — sentinel masking, the annual-mean qnet
     imbalance removal (order-faithful), the 360-day get_periodic_interval
     weights vs hand-computed Veros values (incl. jit-traceability).
  4. The recipe constructs and steps ONCE finite on SYNTHETIC data (no netCDF
     dependency) with the full faithful stack (ab2 + rigid_lid + ratio-48 +
     additive friction + explicit_ab2 + partial cells).
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

# fp64 policy: the recipe targets Veros (f64) fidelity — same policy as the
# free-run harness; grid-centre alignment asserts at 1e-9.
from legoesm.core.precision import PrecisionPolicy, set_policy

set_policy(PrecisionPolicy.fp64())

from legoesm.core.field import Field
from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
from legoesm.ocean.fidelity.veros_acc_recipe import ACC_GM_REDI_CONFIG, ACC_TKE_CONFIG
from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
    DT_MOM_RATIO, DT_MOM_S, DT_TRACER_S,
    GLOBAL4_DDZ, GLOBAL4_EKE_CONFIG, GLOBAL4_GM_REDI_CONFIG, GLOBAL4_TKE_CONFIG,
    H_MAX, MONTH_S, NX, NY, NZ, T_REST_S, VEROS_GLOBAL4_CP0, YEAR_S,
    build_global_4deg_grid,
    build_global_4deg_model_config,
    build_global_4deg_recipe,
    build_global_4deg_z_coord,
    get_periodic_interval_weights,
    global_4deg_A_h,
    kbot_to_mask_and_h_bathy,
    mask_forcing_sentinel,
    remove_qnet_imbalance,
    replicate_veros_kbot,
    veros_area_t,
    veros_xy_to_legoesm,
    veros_xyz_to_legoesm,
    veros_zt_centres,
)


# ---------------------------------------------------------------------------
# 1. Config mapping (each Veros setting → legoESM field)
# ---------------------------------------------------------------------------


def test_gm_redi_mapping():
    """K_iso_0=1000, K_iso_steep=1000, iso_slopec=1e-3 → S_max,
    iso_dslope/iso_slopec = 4e-3/1e-3 → taper_width_frac=4.0."""
    gm = GLOBAL4_GM_REDI_CONFIG
    assert gm.kappa_Redi == 1000.0          # Veros K_iso_0 (constant! see flip)
    assert gm.kappa_GM == 1000.0            # EKE-off fallback (eke drives GM)
    assert gm.K_iso_steep == 1000.0         # Veros K_iso_steep (ACC: 500)
    assert gm.S_max == 1.0e-3               # Veros iso_slopec
    assert gm.taper_width_frac == 4.0       # iso_dslope / iso_slopec
    assert gm.slope_density == "neutral"
    assert gm.implicit_K33 is True
    assert gm.eke is not None               # enable_eke = True


def test_eke_isopycnal_diffusion_flip_vs_acc():
    """THE FLIP: enable_eke_isopycnal_diffusion ABSENT in the setup ⇒ Veros
    settings.py default False ⇒ isopycnal_diffusion=False (ACC: True)."""
    assert GLOBAL4_EKE_CONFIG.isopycnal_diffusion is False
    assert ACC_GM_REDI_CONFIG.eke.isopycnal_diffusion is True  # the contrast


def test_eke_params_match_veros_setup():
    """eke_k_max=1e4, eke_c_k=0.4, eke_c_eps=0.5, eke_cross=2.0,
    eke_crhin=1.0, eke_lmin=100, superbee advection, 3-D field."""
    e = GLOBAL4_EKE_CONFIG
    assert e.kappa_gm_max == 1.0e4
    assert e.c_k == 0.4
    assert e.c_eps == 0.5
    assert e.eke_cross == 2.0
    assert e.eke_crhin == 1.0
    assert e.l_min == 100.0
    assert e.advection_scheme == "superbee"
    assert e.eke_3d is True
    assert e.mixing_length_scheme == "rhines"
    assert e.alpha_eke == 1.0               # Veros default (setup leaves it)
    # Matched-ACC source options carry over (same Veros core).
    acc_e = ACC_GM_REDI_CONFIG.eke
    for f in ("source_kdiss_h", "kdiss_h_flux_form", "gm_source_mode",
              "source_p_diss_iso", "n2_mode", "mixing_length_scheme"):
        assert getattr(e, f) == getattr(acc_e, f), f


def test_tke_mapping():
    """TKE block identical to ACC except the two deltas: superbee advection
    (enable_tke_superbee_advection=True; ACC absent) and
    source_bottom_drag_diss=False (r_bot=0 ⇒ K_diss_bot ≡ 0)."""
    t = GLOBAL4_TKE_CONFIG
    # Veros setup values (repeated identically from the ACC setup).
    assert t.c_k == 0.1
    assert t.c_eps == 0.7
    assert t.alpha_tke == 30.0
    assert t.mxl_min == 1.0e-8
    assert t.tke_mxl_choice == 2
    assert t.kappaM_min == 2.0e-4
    assert t.kappaH_min == 2.0e-5
    assert t.enable_kappaH_profile is True
    assert t.prognostic is True
    assert t.prandtl_mode == "richardson"   # enable_Prandtl_tke default True
    # The two deltas vs ACC:
    assert t.advection_scheme == "superbee"
    assert ACC_TKE_CONFIG.advection_scheme == "none"
    assert t.source_bottom_drag_diss is False
    assert ACC_TKE_CONFIG.source_bottom_drag_diss is True
    # Everything else shared with ACC verbatim.
    for f in ("c_k", "c_eps", "alpha_tke", "mxl_min", "tke_mxl_choice",
              "kappaM_min", "kappaM_max", "kappaH_min",
              "enable_kappaH_profile", "n2_mode", "prandtl_mode",
              "prognostic", "source_eke_diss"):
        assert getattr(t, f) == getattr(ACC_TKE_CONFIG, f), f


def test_dycore_and_stepping_mapping():
    cfg = build_global_4deg_model_config()
    # dt_mom=1800 / dt_tracer=86400 asynchronous ratio 48.
    assert DT_MOM_S == 1800.0 and DT_TRACER_S == 86400.0
    assert cfg.dt_mom_ratio == 48.0 == DT_MOM_RATIO
    # eq_of_state_type=5 → gsw.
    assert cfg.eos == "veros_gsw"
    # NO bottom friction (Veros r_bot default 0.0; setup never enables it).
    assert cfg.bottom_drag_r == 0.0
    # A_h = (4·degtom)³·2e-11 with cos¹(lat) scaling.
    degtom = VEROS_CONSTANTS_CONFIG.R_earth * np.pi / 180.0
    assert np.isclose(cfg.A_h, (4.0 * degtom) ** 3 * 2.0e-11)
    assert np.isclose(global_4deg_A_h(), cfg.A_h)
    assert cfg.A_h_lat_scaling is True and cfg.A_h_cos_power == 1
    assert cfg.lateral_viscosity_operator == "flux_divergence"
    # Veros-faithful stack (matched-ACC kwargs, baked in — see recipe doc).
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
    # Constants pinned through config (G-C4).
    assert cfg.constants is VEROS_CONSTANTS_CONFIG
    assert cfg.rho_0 == 1024.0


def test_flux_feedback_config_mapping():
    cfg = build_global_4deg_model_config()
    sf = cfg.physics.surface_forcing
    assert sf.scheme == "flux_feedback"
    ff = sf.flux_feedback
    # The Veros set_forcing_kernel cp_0 HARDCODE (≠ constants.c_sw = 3994).
    assert ff.c_sw == 3991.86795711963 == VEROS_GLOBAL4_CP0
    # ONE consistent rho threaded from the same block the dycore reads.
    assert ff.rho_0 == VEROS_CONSTANTS_CONFIG.rho_0 == cfg.rho_0
    assert ff.tau_restore_s == 30.0 * 86400.0 == T_REST_S
    assert ff.ice_mask is True
    # Ice threshold from constants (−1.8 to float repr).
    assert np.isclose(ff.ice_threshold_C, -1.8)


def test_explicit_ab2_coriolis_margin():
    """|f|·dt_mom at the most poleward wet row (78°) must sit well inside the
    ≈0.5 explicit-AB2 margin — the documented reason the global domain is
    safe at dt_mom=1800."""
    f78 = 2.0 * VEROS_CONSTANTS_CONFIG.Omega * np.sin(np.deg2rad(78.0))
    assert f78 * DT_MOM_S < 0.30


# ---------------------------------------------------------------------------
# 2. Grid / vertical / bathymetry builders
# ---------------------------------------------------------------------------


def test_grid_alignment_veros_xt_yt():
    grid = build_global_4deg_grid()
    assert grid.n_lat == NY + 2 and grid.n_lon == NX      # 2 wall rows
    lat = np.degrees(np.asarray(grid.lat))
    lon = np.degrees(np.asarray(grid.lon))
    np.testing.assert_allclose(lat[1:-1], -78.0 + 4.0 * np.arange(NY), atol=1e-9)
    np.testing.assert_allclose(lon % 360.0, (2.0 + 4.0 * np.arange(NX)) % 360.0,
                               atol=1e-9)


def test_z_coord_orientation():
    """legoESM dz_ref = Veros ddz AS PUBLISHED (k=0 surface = 50 m); Veros
    stores the reversed ddz[::-1] (k=0 deepest = 690 m)."""
    zc = build_global_4deg_z_coord()
    np.testing.assert_allclose(np.asarray(zc.dz_ref), GLOBAL4_DDZ)
    assert float(zc.dz_ref[0]) == 50.0      # surface
    assert float(zc.dz_ref[-1]) == 690.0    # deepest
    assert zc.H_max == 5200.0 == H_MAX


def test_veros_zt_is_u_centered_not_midpoints():
    """Veros builds zt with u_centered_grid (reflected recursion), NOT plain
    midpoints — using midpoints mis-buckets 56 columns' kbot.  Verify the
    recursion's hand-computable anchors."""
    zt = veros_zt_centres()
    assert zt.shape == (NZ,)
    # Deepest cell: bottom −5200, top −4510 ⇒ centre −4855 (recursion start).
    assert np.isclose(zt[0], -4855.0)
    # Surface cell: the recursion gives −35 (midpoint would give −25).
    assert np.isclose(zt[-1], -35.0)
    # Reflection identity: zt[k+1] = 2·zw[k] − zt[k] with zw from cumsum.
    dzt = GLOBAL4_DDZ[::-1]
    zw = np.zeros(NZ)
    zw[1:] = np.cumsum(dzt[1:])
    zw_shifted = zw - zw[-1]
    for k in range(NZ - 1):
        assert np.isclose(zt[k + 1] + zt[k], 2.0 * zw_shifted[k] + 2.0 * (zw[-1] - zw[-1]))


def test_kbot_replication_rules_synthetic():
    """Veros set_topography verbatim: land where zt <= bathymetry OR salt==0;
    kbot = 1 + count; all-land fixup (bathymetry==0 | kbot==nz → 0)."""
    zt = veros_zt_centres()
    bath = np.zeros((NX, NY))
    salt = np.full((NX, NY, NZ), 35.0)
    # Column A: full-depth ocean (bathymetry below the deepest centre).
    bath[0, 0] = -5300.0
    # Column B: bathymetry at −1000 → levels with zt <= −1000 are land.
    bath[1, 0] = -1000.0
    # Column C: bathymetry == 0 → all-land fixup.
    bath[2, 0] = 0.0
    # Column D: deep bathymetry but salt sentinel zero at the bottom 3 levels.
    bath[3, 0] = -5300.0
    salt[3, 0, :3] = 0.0
    # All other columns: land (bath 0).
    kbot = replicate_veros_kbot(bath, salt)
    assert kbot[0, 0] == 1                               # fully wet
    expected_b = 1 + int(np.sum(zt <= -1000.0))
    assert kbot[1, 0] == expected_b
    assert kbot[2, 0] == 0                               # all-land fixup
    assert kbot[3, 0] == 4                               # 3 sentinel levels
    assert (kbot[4:, :] == 0).all()


def test_kbot_snapping_full_cells():
    """H_bathy snapped to interface depths ⇒ partial-cell coordinate
    degenerates to FULL cells; wall rows land."""
    from legoesm.ocean.vertical import create_partial_cell_coordinate
    zt = veros_zt_centres()
    bath = np.zeros((NX, NY)); salt = np.full((NX, NY, NZ), 35.0)
    bath[0, 0] = -5300.0     # full depth
    bath[1, 0] = -1000.0
    kbot = replicate_veros_kbot(bath, salt)
    land_mask, H_bathy = kbot_to_mask_and_h_bathy(kbot, NY + 2)
    assert land_mask.shape == (NY + 2, NX)
    assert land_mask[0].sum() == 0 and land_mask[-1].sum() == 0   # walls land
    assert H_bathy[1, 0] == 5200.0                                # full column
    # Interface snapping: H equals a partial sum of dz_ref exactly.
    iface = np.concatenate([[0.0], np.cumsum(GLOBAL4_DDZ)])
    assert H_bathy[1, 1] in iface
    zc = build_global_4deg_z_coord()
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H_bathy))
    ia = np.asarray(pc.is_active)
    hp = np.asarray(pc.h_partial)
    np.testing.assert_allclose(
        np.where(ia, hp, 0.0),
        np.where(ia, np.broadcast_to(GLOBAL4_DDZ, ia.shape), 0.0))


def test_xyz_bridge_orientation():
    """(x, y, z k=0 deepest) → (lat, lon, k=0 surface) + wall rows."""
    arr = np.zeros((NX, NY, NZ))
    arr[5, 7, NZ - 1] = 42.0          # SURFACE value in Veros z-order
    arr[5, 7, 0] = -7.0               # DEEPEST value
    out = veros_xyz_to_legoesm(arr, NY + 2)
    assert out.shape == (NY + 2, NX, NZ)
    assert out[8, 5, 0] == 42.0       # row 7+1 (wall offset), surface k=0
    assert out[8, 5, NZ - 1] == -7.0
    assert (out[0] == 0).all() and (out[-1] == 0).all()
    out2 = veros_xy_to_legoesm(np.arange(NX * NY, dtype=float).reshape(NX, NY))
    assert out2.shape == (NY + 2, NX)
    assert out2[1, 0] == 0.0 and out2[2, 0] == 1.0       # (x=0, y=1) → row 2


# ---------------------------------------------------------------------------
# 3. Data-prep pure helpers
# ---------------------------------------------------------------------------


def test_mask_forcing_sentinel():
    a = np.array([-2.0e10, -1.0e10, -0.9e10, 5.0])
    out = mask_forcing_sentinel(a)
    np.testing.assert_allclose(out, [0.0, 0.0, -0.9e10, 5.0])  # <= -1e10 only


def test_remove_qnet_imbalance_order_faithful():
    """The mean is over ALL interior cells (land included), THEN the wet
    mask — Veros set_initial_conditions order (the residual after masking is
    deliberately nonzero)."""
    qnet = np.zeros((2, 2, 12))
    qnet[0, 0, :] = 10.0          # wet cell
    qnet[1, 1, :] = -2.0          # land cell still enters the MEAN
    area = np.ones((2, 2))
    wet = np.array([[1.0, 0.0], [0.0, 0.0]])
    adj, mean_flux = remove_qnet_imbalance(qnet, area, wet)
    # mean = (10 - 2)·12months/(12·4cells) = 2.0
    assert np.isclose(mean_flux, 2.0)
    np.testing.assert_allclose(adj[0, 0, :], 8.0)        # wet: 10 − 2
    np.testing.assert_allclose(adj[1, 1, :], 0.0)        # masked after removal
    # Area weighting matters: double one cell's area.
    area2 = np.array([[3.0, 1.0], [1.0, 1.0]])
    _, mf2 = remove_qnet_imbalance(qnet, area2, wet)
    assert np.isclose(mf2, (10.0 * 3.0 - 2.0) / 6.0)


def test_veros_area_t():
    a = veros_area_t(np.array([0.0, 60.0]))
    degtom = VEROS_CONSTANTS_CONFIG.R_earth * np.pi / 180.0
    assert np.isclose(a[0], (4 * degtom) ** 2)
    assert np.isclose(a[1], (4 * degtom) ** 2 * 0.5)


def test_get_periodic_interval_weights_vs_veros():
    """Hand-computed Veros get_periodic_interval values (360-day year,
    month-START anchoring: t=0 → 100% January)."""
    cases = [
        # (t_days, n1, f1, n2, f2)
        (0.0,   0, 1.0, 1, 0.0),
        (15.0,  0, 0.5, 1, 0.5),
        (30.0,  1, 1.0, 2, 0.0),
        (45.0,  1, 0.5, 2, 0.5),
        (345.0, 11, 0.5, 0, 0.5),       # December → wraps to January
        (360.0, 0, 1.0, 1, 0.0),        # periodic
        (725.0, 0, 1.0 - 5.0 / 30.0, 1, 5.0 / 30.0),   # 2nd year, day 5
    ]
    for t_days, n1e, f1e, n2e, f2e in cases:
        n1, f1, n2, f2 = get_periodic_interval_weights(t_days * 86400.0)
        assert int(n1) == n1e, t_days
        assert int(n2) == n2e, t_days
        assert np.isclose(float(f1), f1e), t_days
        assert np.isclose(float(f2), f2e), t_days
    # jit-traceable from a traced time (the in-scan SegmentForcing pattern).
    n1, f1, n2, f2 = jax.jit(get_periodic_interval_weights)(
        jnp.asarray(45.0 * 86400.0))
    assert int(n1) == 1 and np.isclose(float(f2), 0.5)
    assert np.isclose(MONTH_S * 12, YEAR_S)


# ---------------------------------------------------------------------------
# 4. Recipe constructs + steps once on synthetic data (no netCDF)
# ---------------------------------------------------------------------------


def _synthetic_inputs():
    """Tiny fabricated world: a deep basin with a shelf and some land."""
    rng = np.random.default_rng(7)
    bath = np.zeros((NX, NY))
    salt = np.zeros((NX, NY, NZ))
    temp = np.zeros((NX, NY, NZ))
    zt = veros_zt_centres()
    for i in range(NX):
        for j in range(4, NY - 4):
            depth = 5300.0 if (i + j) % 7 else 800.0   # basin + shelves
            bath[i, j] = -depth
            wet = zt > -depth
            salt[i, j, :] = np.where(wet, 35.0, 0.0)
            temp[i, j, :] = np.where(
                wet, 2.0 + 10.0 * np.exp(zt / 800.0), 0.0)
    # a couple of mid-basin land islands
    bath[10:12, 12:14] = 0.0
    salt[10:12, 12:14, :] = 0.0
    return bath, salt, temp


def test_recipe_constructs_on_synthetic_data():
    bath, salt, temp = _synthetic_inputs()
    recipe = build_global_4deg_recipe(bath, salt, temp)
    st = recipe.initial_state
    assert st.T.data.shape == (NY + 2, NX, NZ)
    # tracer fields masked by the active cells
    ia = np.asarray(recipe.z_coord.is_active)
    T = np.asarray(st.T.data)
    assert (T[~ia.astype(bool)] == 0.0).all()
    # carry fields seeded (constant lax.scan pytree from step 0)
    assert st.tke is not None and st.dtke is not None
    assert st.eke is not None and st.eke_diss is not None
    assert st.tke.data.shape == (NY + 2, NX, NZ - 1)


@pytest.mark.slow
def test_recipe_steps_once_finite_synthetic():
    """Full faithful stack steps once finite on the synthetic world."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    bath, salt, temp = _synthetic_inputs()
    recipe = build_global_4deg_recipe(bath, salt, temp)
    cfg = recipe.model_config
    LatLonCGridOceanModel._validate_config(cfg)          # must not raise
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    model.check_coriolis_stability(DT_TRACER_S)

    state = recipe.initial_state
    _z = lambda d: Field(data=jnp.zeros_like(d.data),
                         name=d.name + "_incr_prev", dims=d.dims, units=d.units)
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
    )
    after = model.step(state, dt=DT_TRACER_S, surface_forcing=sf)
    jax.block_until_ready(after.u.data)
    for fld in ("u", "v", "T", "S", "tke", "eke"):
        arr = np.asarray(getattr(after, fld).data)
        assert np.all(np.isfinite(arr)), f"non-finite {fld} after one step"
    assert float(np.max(np.abs(np.asarray(after.eta.data)))) < 1e-6  # rigid lid


def test_veros_u_centered_z_centres_recursion():
    """Direct unit test for the bridge's u_centered_grid construction (review
    hygiene item): hand-computed pyOM recursion on a tiny stretched grid, the
    uniform-grid midpoint degeneration, and the global_4deg reference values.
    """
    import numpy as np
    from legoesm.ocean.fidelity.veros_state_bridge import veros_u_centered_z_centres

    # Hand-computed: dzt top-down [10, 20, 40] (bottom-up [40, 20, 10]).
    # pyOM (bottom-up): zt[0] = dz[0]/2 = 20; zw[0] = 40
    # zt[k] = 2*zw[k-1] - zt[k-1]: zt[1] = 80-20 = 60; zw[1] = 60
    # zt[2] = 120-60 = 60 ... recompute carefully against numerics.py:
    #   zw_raw = cumsum(bottom-up dz) = [40, 60, 70]
    #   zt[0] = 40 - 40/2 = 20; zt[1] = 2*40 - 20 = 60; zt[2] = 2*60 - 60 = 60
    # shift so surface interface = 0 (total depth 70): depths top-down =
    #   70 - [60, 60, 20] reversed -> [10, 10, 50]
    # (z is NEGATIVE-down: surface interface 0, centres below.)
    z = veros_u_centered_z_centres(np.array([10.0, 20.0, 40.0]))
    np.testing.assert_allclose(z, np.array([-10.0, -10.0, -50.0]), atol=1e-12)
    # NOT the midpoints [-5, -20, -50]: the recursion differs on stretched grids.
    assert abs(z[0] - (-5.0)) > 1.0

    # Uniform grid degenerates to midpoints exactly.
    zu = veros_u_centered_z_centres(np.full(6, 100.0))
    np.testing.assert_allclose(zu, -(np.arange(6) * 100.0 + 50.0), atol=1e-12)

    # global_4deg dzt (top-down): first interior centre-spacings dzw must be
    # the documented alternating sequence [30, 110, 90, 190, 190] (NOT the
    # midpoint [60, 85, 120, 165, 215]) — provenance: Veros numerics.py:9-22
    # applied to the global_4deg ddz, verified bit-identical vs the banked
    # oracle zt during review.
    ddz_td = np.array([50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0,
                       390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0])
    zg = veros_u_centered_z_centres(ddz_td)
    dzw = -np.diff(zg)   # positive spacings, top-down
    np.testing.assert_allclose(dzw[:5], [30.0, 110.0, 90.0, 190.0, 190.0],
                               atol=1e-9)
