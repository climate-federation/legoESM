"""Smoke tests for the legoESM-Veros ACC recipe and Veros↔legoESM
state bridge (Phase G.0c).

Designed to exercise the recipe + bridge infrastructure end-to-end
without requiring Veros to be installed. Uses a synthetic
``VerosResult``-like object to simulate a Veros snapshot.
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

from legoesm.ocean.fidelity.tendency_probe import (
    build_region_masks, probe_latlon_cgrid,
)
from legoesm.ocean.fidelity.veros_acc_recipe import (
    ACC_DZT, ACC_GM_REDI_CONFIG, ACC_TKE_CONFIG, NX, NY, NZ,
    T_RESTORING_DAYS, VEROS_BLOCK_MAPPING, acc_A_h, build_acc_grid,
    build_acc_model_config, build_acc_recipe, build_acc_restoring_config,
    build_acc_t_star, build_acc_wind_stress, build_acc_z_coord,
)
from legoesm.ocean.fidelity.veros_runner import VerosResult
from legoesm.ocean.fidelity.veros_state_bridge import (
    VEROS_TENDENCY_CAPTURE_VARS,
    extract_veros_tendencies,
    veros_snapshot_to_legoesm_state,
)


# ---------------------------------------------------------------------------
# Recipe construction
# ---------------------------------------------------------------------------


def test_recipe_resolution_matches_veros_acc():
    """nx=30, ny=42, nz=15 verbatim from veros/setups/acc/acc.py."""
    assert NX == 30
    assert NY == 42
    assert NZ == 15


def test_recipe_dzt_sums_to_2080m():
    """sum(ddz)/2.5 = 5200/2.5 = 2080 m total depth, exactly."""
    total = float(np.sum(ACC_DZT))
    np.testing.assert_allclose(total, 2080.0, atol=1e-9)


def test_acc_A_h_value():
    """A_h = (2 * R_earth · π/180)^3 · 2e-11. Veros canonical."""
    R = 6.370e6   # Veros's radius; legoESM default differs slightly
    A_h = acc_A_h(R)   # R_earth passed explicitly (recipe pins via config, not the patch)
    degtom = R * np.pi / 180.0
    expected = (2 * degtom) ** 3 * 2.0e-11
    np.testing.assert_allclose(A_h, expected, rtol=1e-12)


def test_tke_and_gm_redi_configs_match_veros():
    """Verbatim from veros/setups/acc/acc.py."""
    assert ACC_TKE_CONFIG.c_k == 0.1
    assert ACC_TKE_CONFIG.c_eps == 0.7
    assert ACC_TKE_CONFIG.alpha_tke == 30.0
    assert ACC_TKE_CONFIG.mxl_min == 1e-8
    assert ACC_TKE_CONFIG.tke_mxl_choice == 2
    assert ACC_TKE_CONFIG.kappaM_min == 2e-4
    assert ACC_TKE_CONFIG.kappaH_min == 2e-5

    assert ACC_GM_REDI_CONFIG.kappa_GM == 1000.0
    assert ACC_GM_REDI_CONFIG.kappa_Redi == 1000.0
    # iso_slopec=0.01, iso_dslope=0.005 → S_max=0.01, taper_width_frac=0.5.
    assert ACC_GM_REDI_CONFIG.S_max == 0.01
    assert ACC_GM_REDI_CONFIG.taper_width_frac == 0.5

    # EKE adopted (Veros ACC enable_eke=True, acc.py:67-75): Rhines-limited mixing
    # length with eke_cross=2.0; other params match the EKEConfig ACC defaults.
    eke = ACC_GM_REDI_CONFIG.eke
    assert eke is not None, "ACC recipe must run prognostic EKE (Veros enable_eke=True)"
    assert eke.mixing_length_scheme == "rhines"
    assert eke.eke_cross == 2.0 and eke.eke_crhin == 1.0   # acc.py:71-72
    assert eke.c_k == 0.4 and eke.c_eps == 0.5             # acc.py:69-70
    assert eke.l_min == 100.0 and eke.kappa_gm_max == 1.0e4  # acc.py:73,68
    # K_iso = K_gm: Veros enable_eke_isopycnal_diffusion=True (acc.py:75).
    assert eke.isopycnal_diffusion is True


def test_z_coord_dz_ref_in_legoesm_order():
    """legoESM convention: dz_ref[0] = surface = 20 m, dz_ref[-1] =
    bottom = 276 m (Veros stores the same values reversed)."""
    z = build_acc_z_coord()
    dz = np.asarray(z.dz_ref)
    assert dz[0] == pytest.approx(20.0)
    assert dz[-1] == pytest.approx(276.0)
    np.testing.assert_allclose(float(np.sum(dz)), z.H_max, atol=1e-10)


def test_full_recipe_builds():
    """The complete builder must produce a usable
    (model_config, physics_config, grid, z_coord, state) tuple with all
    Veros constants pinned via config (no monkey-patch)."""
    recipe = build_acc_recipe()
    assert recipe.model_config.eos == "veros_nonlin2"
    assert recipe.model_config.lateral_viscosity.A_h_lat_scaling is True
    assert recipe.model_config.lateral_viscosity.A_h_cos_power == 1
    assert recipe.model_config.implicit_vertical_mixing is True
    assert recipe.physics_config.vertical_mixing.scheme == "tke"
    # GM/Redi on lat-lon is a top-level (dynamics) field, NOT physics-pathway
    # lateral mixing (which is cubed-sphere-only). The physics lateral_mixing
    # is "none"; the GM/Redi config lives at model_config.gm_redi.
    assert recipe.physics_config.lateral_mixing.scheme == "none"
    assert recipe.model_config.gm_redi is ACC_GM_REDI_CONFIG
    assert recipe.z_coord.n_levels == NZ
    assert recipe.initial_state.T.data.shape[-1] == NZ
    assert recipe.initial_state.land_mask.data.dtype.kind == "f"


# ---------------------------------------------------------------------------
# Veros ↔ legoESM state bridge
# ---------------------------------------------------------------------------


def _make_synthetic_veros_result(nx=NX, ny=NY, nz=NZ):
    """Construct a VerosResult-like object with halos + tau dim.

    Field shape conventions used by Veros:
    - 3-D vars (u, v, temp, salt, tendencies):
        (nx+4, ny+4, nz, 3)  with 2-cell halos and a (taum1, tau, taup1) dim
    - 2-D vars (surface_taux, surface_tauy):
        (nx+4, ny+4)
    """
    rng = np.random.default_rng(42)
    halo = 2
    n_tau = 3
    var3d_shape = (nx + 2 * halo, ny + 2 * halo, nz, n_tau)
    var2d_shape = (nx + 2 * halo, ny + 2 * halo)

    variables = {}
    # State vars
    for name in ("u", "v", "temp", "salt", "rho"):
        variables[name] = 0.01 * rng.standard_normal(var3d_shape)
    variables["surface_taux"] = 0.05 * rng.standard_normal(var2d_shape)
    variables["surface_tauy"] = 0.01 * rng.standard_normal(var2d_shape)
    # Per-process tendencies
    for veros_name in VEROS_TENDENCY_CAPTURE_VARS:
        if veros_name in variables:
            continue
        if veros_name.startswith(("du_", "dv_", "dtemp_", "dsalt_")):
            variables[veros_name] = 1e-7 * rng.standard_normal(var3d_shape)

    return VerosResult(
        case_name="acc_channel",
        times_s=np.array([4800.0]),
        variables=variables,
        grid_metadata={"nx": nx, "ny": ny, "nz": nz},
        provenance={"veros_version": "synthetic-mock-for-test"},
    )


def test_state_bridge_strips_halos_and_reverses_z():
    """A synthetic VerosResult must round-trip through the bridge:
    halos stripped, z-axis reversed, axes transposed (x↔lat,y↔lon)."""
    recipe = build_acc_recipe()
    result = _make_synthetic_veros_result()

    bridged = veros_snapshot_to_legoesm_state(result, recipe.initial_state)
    state = bridged.state

    # T must have legoESM shape (n_lat, n_lon, n_lev).
    assert state.T.data.shape == recipe.initial_state.T.data.shape
    assert state.S.data.shape == recipe.initial_state.S.data.shape
    assert state.u.data.shape == recipe.initial_state.u.data.shape
    assert state.v.data.shape == recipe.initial_state.v.data.shape

    # Veros's temp[2, 2, 0, 1] (after halo strip → (0, 0, 0), tau=1)
    # must end up at legoESM T[0, 0, n_lev-1] after the z-reverse
    # (Veros k=0 deep → legoESM k=n_lev-1 deep) but also after the
    # x↔y transpose (Veros (x=0, y=0) → legoESM (lat=0, lon=0) since
    # x is column 0 = lon 0 and y is row 0 = lat 0).
    #
    # With y-wall padding (legoESM has 2 extra lat rows for N/S walls
    # at ACC's recipe geometry), Veros row 0 → legoESM row 1.
    # Verify the round-trip preserves the magnitudes.
    veros_t = result.variables["temp"][2:-2, 2:-2, :, 1]   # (nx, ny, nz)
    # Bridged should match veros_t up to (y-pad + swap + z-reverse).
    n_lat = state.T.data.shape[0]
    n_lon = state.T.data.shape[1]
    # interior lat rows = 1 .. n_lat-2 (if padded), or 0..n_lat-1 (if not).
    # In this test the recipe builder uses periodic_x and create_regional_latlon_grid
    # which yields n_lat = NY + 2 (walls); so interior is rows 1..NY.
    if n_lat == NY + 2:
        interior_T = np.asarray(state.T.data[1:-1, :, :])
    else:
        interior_T = np.asarray(state.T.data)
    # Veros (x, y, z) → legoESM (lat=y, lon=x, level=z-reversed)
    expected = np.swapaxes(veros_t, 0, 1)[..., ::-1]
    np.testing.assert_allclose(interior_T, expected, rtol=1e-12)


def test_extract_veros_tendencies_returns_mapped_dict():
    """The tendency extractor must produce a dict keyed by the legoESM-side
    names from the VEROS_TO_LEGOESM_* maps. Default behavior pads y-walls
    so shapes match legoESM's lat-lon C-grid convention."""
    result = _make_synthetic_veros_result()
    tend = extract_veros_tendencies(result)   # pad_y_walls=True default
    assert "coriolis_u" in tend
    assert "coriolis_v" in tend
    assert "veros_du_adv" in tend
    assert "veros_du_mix" in tend
    assert "veros_dT_hmix" in tend
    assert "veros_dT_vmix" in tend
    assert "veros_dT_iso" in tend
    assert "veros_dS_hmix" in tend
    # With pad_y_walls=True: shape (NY + 2, NX, NZ) — matches legoESM
    # land_mask grid.
    for k, v in tend.items():
        assert v.shape == (NY + 2, NX, NZ), f"{k}: {v.shape}"


def test_extract_veros_tendencies_without_pad():
    """``pad_y_walls=False`` returns Veros interior shape (NY, NX, NZ)."""
    result = _make_synthetic_veros_result()
    tend = extract_veros_tendencies(result, pad_y_walls=False)
    for k, v in tend.items():
        assert v.shape == (NY, NX, NZ), f"{k}: {v.shape}"


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_end_to_end_recipe_probe_round_trip():
    """Build the recipe, build a Veros-like snapshot, bridge it,
    probe the bridged state, and verify all probe fields are finite."""
    recipe = build_acc_recipe()
    result = _make_synthetic_veros_result()
    bridged = veros_snapshot_to_legoesm_state(result, recipe.initial_state)
    probe = probe_latlon_cgrid(
        bridged.state, recipe.grid, recipe.z_coord, recipe.model_config,
        dt=4800.0,
    )
    masks = build_region_masks(
        recipe.grid, recipe.z_coord, bridged.state,
    )

    for name in probe._fields:
        arr = np.asarray(getattr(probe, name))
        assert np.all(np.isfinite(arr)), (
            f"Probe field {name!r} contains non-finite values."
        )
    # Region masks have the cell-centre shape.
    assert masks.interior.shape == bridged.state.T.data.shape


# ---------------------------------------------------------------------------
# Q1 regression: out-of-domain padding rows must be land (Phase G residual)
# ---------------------------------------------------------------------------


def test_acc_grid_centres_match_veros_xt_yt():
    """legoESM build_acc_grid interior centres must coincide with Veros's
    u-centred xt/yt (verified from a live ACCSetup): xt=[-1,1,...,57],
    yt=[-41,...,41]. Getting this wrong (half-cell lon offset, one-row lat
    offset) was the Phase G tier-2 residual — density matched point-wise but
    the wall mis-aligned and momentum metrics f(lat) were wrong."""
    from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_grid

    grid = build_acc_grid()
    lon = np.degrees(np.asarray(grid.lon))
    lat = np.degrees(np.asarray(grid.lat))
    # lon (periodic, no walls) matches Veros xt. atol=1e-3 accommodates the
    # float32 grid storage while firmly catching any real (>=1.0 half-cell)
    # offset — the bug this guards against.
    np.testing.assert_allclose(lon, np.arange(NX) * 2.0 - 1.0, atol=1e-3)
    # interior lat rows (1..NY; rows 0/-1 are N/S walls) match Veros yt.
    np.testing.assert_allclose(lat[1:-1], np.arange(NY) * 2.0 - 41.0, atol=1e-3)


def test_acc_land_mask_matches_veros_ocean_footprint():
    """With Veros-coincident centres, the analytic kbot must reproduce Veros's
    ACC ocean footprint EXACTLY: the western wall is 2 lon-columns wide
    (xt=-1,1 both <=1 -> land for lat>=-20), ocean from col 2; the channel
    (lat<-20) is fully zonal. Over legoESM's 44 lat rows (incl 2 N/S walls)
    the wet count per lon-col is [11, 11, 42, 42, ...]. Regression lock for the
    Phase G grid-alignment + wall-contamination bug (no Veros dependency)."""
    from legoesm.ocean.fidelity.veros_acc_recipe import (
        build_acc_grid, build_acc_land_mask,
    )

    grid = build_acc_grid()
    lm = np.asarray(build_acc_land_mask(grid))
    # N/S wall rows are fully land.
    assert float(lm[0].max()) == 0.0 and float(lm[-1].max()) == 0.0
    # Wet count per lon-col matches Veros's ocean footprint.
    wet_per_col = lm.sum(axis=0).astype(int)
    expected = np.array([11, 11] + [42] * (NX - 2))
    np.testing.assert_array_equal(wet_per_col, expected)


def test_bridged_acc_state_has_no_zero_TS_wet_cells():
    """Q1 acceptance gate (a): after the land-mask fix, NO wet cell may carry
    the bridge's zero-padded T=S=0 (the rho ~ 997 contamination). Regression
    lock for the Phase G interior density residual."""
    recipe = build_acc_recipe()
    result = _make_synthetic_veros_result()
    bridged = veros_snapshot_to_legoesm_state(result, recipe.initial_state)

    T = np.asarray(bridged.state.T.data)
    S = np.asarray(bridged.state.S.data)
    wet = np.broadcast_to(
        np.asarray(bridged.state.land_mask.data)[:, :, None] > 0.5, T.shape
    )
    zero_TS = (T == 0.0) & (S == 0.0)
    n_bad = int((wet & zero_TS).sum())
    assert n_bad == 0, (
        f"{n_bad} wet cells carry zero-padded T=S=0 (wall contamination)"
    )


# ---------------------------------------------------------------------------
# Q2: per-process momentum comparison (face->centre interpolation)
# ---------------------------------------------------------------------------


def test_face_to_centre_interpolation_reduces_shape_and_averages():
    """Face->centre helpers drop one along the staggered axis and average
    adjacent faces."""
    from legoesm.ocean.fidelity.tendency_probe import (
        u_face_to_centre, v_face_to_centre,
    )

    uf = jnp.asarray(np.arange(2 * 4 * 1, dtype=float).reshape(2, 4, 1))
    uc = np.asarray(u_face_to_centre(uf))
    assert uc.shape == (2, 3, 1)
    np.testing.assert_allclose(
        uc, 0.5 * (np.asarray(uf)[:, :-1, :] + np.asarray(uf)[:, 1:, :]),
    )
    vf = jnp.asarray(np.arange(3 * 2 * 1, dtype=float).reshape(3, 2, 1))
    vc = np.asarray(v_face_to_centre(vf))
    assert vc.shape == (2, 2, 1)
    np.testing.assert_allclose(
        vc, 0.5 * (np.asarray(vf)[:-1, :, :] + np.asarray(vf)[1:, :, :]),
    )


def test_compare_momentum_emits_all_processes():
    """Q2 deliverable: the momentum comparison must emit per-region metrics
    (incl. the magnitude-weighted sign-match) for all 6 Veros momentum
    processes — no more 'shape mismatch / deferred' rows."""
    from legoesm.ocean.fidelity.tendency_probe import compare_momentum_at_centres

    recipe = build_acc_recipe()
    result = _make_synthetic_veros_result()
    bridged = veros_snapshot_to_legoesm_state(result, recipe.initial_state)
    probe = probe_latlon_cgrid(
        bridged.state, recipe.grid, recipe.z_coord, recipe.model_config,
        dt=4800.0,
    )
    masks = build_region_masks(recipe.grid, recipe.z_coord, bridged.state)
    vt = extract_veros_tendencies(result)
    mom = compare_momentum_at_centres(probe, vt, masks)
    for proc in ("coriolis_u", "coriolis_v", "du_adv", "dv_adv", "du_mix", "dv_mix"):
        assert proc in mom, f"missing momentum process {proc}"
        it = mom[proc]["interior"]
        assert np.isfinite(it["L2"]), proc
        assert "weighted_sign_match" in it, proc


# ---------------------------------------------------------------------------
# Q3: fidelity recipe modules must be registered for discovery
# ---------------------------------------------------------------------------


def test_fidelity_recipe_modules_are_registered():
    """The recipe/bridge/probe modules must be discoverable via the fidelity
    package's lazy loader. They existed on disk but were absent from __all__,
    so attribute access / test discovery failed (Phase G open issue)."""
    import legoesm.ocean.fidelity as fid

    for name in (
        "veros_acc_recipe", "veros_state_bridge", "tendency_probe",
    ):
        assert name in fid.__all__, f"{name} not registered in fidelity __all__"
        mod = getattr(fid, name)  # exercises the lazy __getattr__
        assert mod is not None, name
    assert hasattr(fid.veros_acc_recipe, "build_acc_recipe")
    assert hasattr(fid.tendency_probe, "compare_momentum_at_centres")


# ---------------------------------------------------------------------------
# Q6: bridge equivariance tier (the bridge is a physics-preserving bijection)
# ---------------------------------------------------------------------------


def test_bridge_halo_strip_invariance():
    """φ = the Veros halo-cell values. The bridge strips the 2-cell halos, so
    the bridged INTERIOR must be independent of them — garbage in the halos
    must not change the bridged state. Locks strip-halo as a true bijection on
    the physical field (doctrine §4)."""
    recipe = build_acc_recipe()
    res_clean = _make_synthetic_veros_result()
    res_garbage = _make_synthetic_veros_result()  # same seed -> identical interior
    halo = 2
    for name in ("u", "v", "temp", "salt", "rho"):
        a = np.array(res_garbage.variables[name])
        a[:halo] = 1e30
        a[-halo:] = 1e30
        a[:, :halo] = 1e30
        a[:, -halo:] = 1e30
        res_garbage.variables[name] = a

    b_clean = veros_snapshot_to_legoesm_state(res_clean, recipe.initial_state)
    b_garb = veros_snapshot_to_legoesm_state(res_garbage, recipe.initial_state)
    for fld in ("T", "S", "u", "v"):
        np.testing.assert_array_equal(
            np.asarray(getattr(b_garb.state, fld).data),
            np.asarray(getattr(b_clean.state, fld).data),
        )


def test_bridge_roundtrip_salt_index_mapping():
    """Extend the temp round-trip to salinity: strip-halo ∘ transpose(x,y ->
    lat,lon) ∘ reverse-z must place Veros cell (i,j,k) at legoESM
    (lat=j, lon=i, level=nz-1-k)."""
    result = _make_synthetic_veros_result()
    recipe = build_acc_recipe()
    bridged = veros_snapshot_to_legoesm_state(result, recipe.initial_state)
    veros_s = result.variables["salt"][2:-2, 2:-2, :, 1]   # (nx, ny, nz), tau=1
    expected = np.swapaxes(veros_s, 0, 1)[..., ::-1]        # (lat, lon, z-reversed)
    n_lat = bridged.state.S.data.shape[0]
    interior_S = (
        np.asarray(bridged.state.S.data[1:-1, :, :]) if n_lat == NY + 2
        else np.asarray(bridged.state.S.data)
    )
    np.testing.assert_allclose(interior_S, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# GM/Redi wiring: lat-lon GM/Redi is dynamics-level (top-level config.gm_redi),
# NOT physics-pathway lateral mixing (which is cubed-sphere-only)
# ---------------------------------------------------------------------------


def test_recipe_gm_redi_wired_at_top_level_not_physics():
    """The lat-lon model applies GM/Redi from config.gm_redi (top-level); the
    physics-pathway lateral-mixing factory is cubed-sphere-only. The recipe
    must set GM/Redi at the top level (else GM/Redi is silently INACTIVE:
    config.gm_redi defaults None -> model skips it, probe never runs physics)."""
    recipe = build_acc_recipe()
    assert recipe.model_config.gm_redi is ACC_GM_REDI_CONFIG, (
        "GM/Redi must be at model_config.gm_redi (what the lat-lon model reads)"
    )
    assert recipe.model_config.physics.lateral_mixing.scheme == "none"


def test_recipe_self_pins_constants_via_config():
    """G-C4: build_acc_recipe() pins g/rho_0/constants (and grid radius/Omega +
    A_h via R_earth) to Veros values purely through config — no monkey-patch,
    no module-level constant mutation. This is what let the old
    override_constants context manager be deleted entirely."""
    from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG

    recipe = build_acc_recipe()
    assert recipe.model_config.g == VEROS_CONSTANTS_CONFIG.g == 9.81
    assert recipe.model_config.rho_0 == VEROS_CONSTANTS_CONFIG.rho_0 == 1024.0
    assert recipe.model_config.constants is VEROS_CONSTANTS_CONFIG


def test_latlon_model_rejects_physics_lateral_mixing():
    """Guard: a non-'none' physics.lateral_mixing scheme on the lat-lon C-grid
    is a mis-wiring (the physics factory is cubed-sphere-only). It must raise at
    construction with a clear message; the fixed recipe must pass."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.state import LatLonCGridOceanConfig

    bad = LatLonCGridOceanConfig.from_flat(
        physics=OceanPhysicsConfig(
            lateral_mixing=LateralMixingConfig(
                scheme="gm_redi", gm_redi=ACC_GM_REDI_CONFIG,
            ),
        ),
    )
    with pytest.raises(ValueError, match="lat-lon C-grid"):
        LatLonCGridOceanModel._validate_config(bad)

    recipe = build_acc_recipe()
    LatLonCGridOceanModel._validate_config(recipe.model_config)  # must not raise


# ---------------------------------------------------------------------------
# Free-run surface forcing (Veros ACC wind stress + T* restoring)
# ---------------------------------------------------------------------------


def test_acc_t_star_profile_matches_veros_bands():
    """T* = 15 degC in [-20, 20], ramping to ~0 at the meridional walls."""
    grid = build_acc_grid()
    lat = np.degrees(np.asarray(grid.lat))
    ts = np.asarray(build_acc_t_star(grid))
    assert ts.shape == (grid.n_lat, grid.n_lon)
    # constant across longitude (zonally symmetric forcing)
    assert np.allclose(ts, ts[:, :1])
    prof = ts[:, 0]
    mid = (lat >= -20) & (lat <= 20)
    assert np.allclose(prof[mid], 15.0, atol=1e-5)
    # ramps DOWN toward the walls (monotone away from the plateau)
    assert float(prof[lat > 20].max()) <= 15.0 + 1e-6
    assert float(prof[lat < -20].max()) <= 15.0 + 1e-6
    assert float(prof.min()) >= -1e-6        # never negative


def test_acc_wind_stress_sign_and_bands():
    """Wind: zero in the tropical band [-20, 10]; the supplied tau_x is NEGATIVE
    in the southern westerly band so legoESM's internal -tau_x flip yields an
    EASTWARD ocean stress (drives the ACC the right way)."""
    grid = build_acc_grid()
    lat = np.degrees(np.asarray(grid.lat))
    sf = build_acc_wind_stress(grid)
    tau_x = np.asarray(sf.tau_x)
    assert tau_x.shape == (grid.n_lat, grid.n_lon)
    assert np.all(np.asarray(sf.tau_y) == 0.0)
    assert np.all(np.isfinite(tau_x))
    # tropical band: no wind
    trop = (lat >= -20) & (lat <= 10)
    assert np.allclose(tau_x[trop, :], 0.0, atol=1e-12)
    # southern westerly band: ocean-side stress (= -tau_x) is eastward (>0)
    band = (lat < -20) & (lat > -42)
    assert np.all(-tau_x[band, :] > 0.0), "ACC westerlies must push the ocean eastward"


def test_acc_restoring_config_matches_veros():
    grid = build_acc_grid()
    cfg = build_acc_restoring_config(grid)
    assert cfg.tau_T == T_RESTORING_DAYS * 86400.0      # 30-day heat restoring
    assert cfg.tau_S >= 1.0e29                          # salinity effectively unrestored
    assert cfg.T_star_array is not None
    assert np.asarray(cfg.T_star_array).shape == (grid.n_lat, grid.n_lon)
    assert cfg.implicit is False                        # match Veros explicit restoring


def test_build_acc_recipe_surface_forcing_flag():
    """Default recipe = no forcing (frozen-state probe unchanged). With the flag,
    the recipe carries the wind OceanSurfaceForcing + a restoring physics scheme."""
    off = build_acc_recipe()
    assert off.wind_forcing is None
    assert off.physics_config.surface_forcing.scheme == "prescribed"

    on = build_acc_recipe(with_surface_forcing=True)
    assert on.wind_forcing is not None
    assert on.wind_forcing.tau_x is not None
    assert on.physics_config.surface_forcing.scheme == "restoring"
    assert on.physics_config.surface_forcing.restoring.T_star_array is not None
    # dynamics/grid/IC unchanged by the forcing flag
    assert on.model_config.eos == off.model_config.eos
    assert on.grid.n_lat == off.grid.n_lat


def test_free_run_ships_faithful_stepping_composition():
    """The free-run recipe (with_surface_forcing=True) must ship the
    Veros-faithful stepping composition WITHOUT driver overrides. The
    matsuno_split/"total" composition leaks ~+1.8 GW of spurious KE
    (0.6% of wind throughput; realized 90-day energy audit vs Veros's
    built-in energy diagnostics, 2026-06-12) and equilibrated at KE 1.53x /
    mean_eke 4.6x the Veros equilibrium; this bundle closes the budget
    channel-by-channel to ~1-3% and lands the 30-yr free run at KE 0.950x.
    The frozen-state probe path (with_surface_forcing=False) keeps the
    legoESM defaults — bit-identical for every tendency probe."""
    from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_model_config

    free = build_acc_model_config(with_surface_forcing=True)
    assert free.outer_integrator == "ab2"
    assert free.dt_mom_ratio == 9.0          # dt_tracer 43200 / dt_mom 4800
    assert free.barotropic.barotropic_solver == "rigid_lid"
    assert free.coriolis_scheme == "explicit_ab2"   # the energy lever
    assert free.ab2_scope == "advective"
    assert free.momentum_friction_additive is True
    assert free.ab2_epsilon == 0.1           # Veros AB_eps (config default)
    assert free.surface_forcing_implicit is True

    probe = build_acc_model_config(with_surface_forcing=False)
    assert probe.outer_integrator == "forward_euler"
    assert probe.dt_mom_ratio == 1.0
    assert probe.barotropic.barotropic_solver == "explicit_substep"
    assert probe.coriolis_scheme == "matsuno_split"
    assert probe.ab2_scope == "total"
    assert probe.momentum_friction_additive is False


def test_initial_condition_is_veros_exact_linear():
    """The recipe IC must be the LITERAL Veros ACC profile
    temp = (1 - zt/zw[0])*15 with Veros's bottom-first zw[0] = the top face
    of the BOTTOM cell (~-1724 m) — NOT the bottom interface -H_max. The
    earlier 'linear' transcription normalised by -H_max, starting the abyss
    +2.15 K warm (lego +1.00 vs Veros -1.15 C at t=0); with the ~3 Sv deep
    ventilation that IC offset persisted as essentially the entire 30-yr
    abyssal warm bias (2.33 vs 0.98 C). Matched-IC 5-yr abyss trajectories
    agree to ~0.01 K."""
    import numpy as np
    recipe = build_acc_recipe(with_surface_forcing=True)
    z = recipe.z_coord
    st = recipe.initial_state
    lm = np.asarray(st.land_mask.data)
    T = np.asarray(st.T.data)
    zt = np.asarray(z.z_full_ref)
    zw0 = float(np.asarray(z.z_half_ref)[-2])     # Veros bottom-first zw[0]
    expected = 15.0 * (1.0 - zt / zw0)
    # the profile is z-only: check a wet column exactly
    j, i = np.argwhere(lm > 0.5)[0]
    np.testing.assert_allclose(T[j, i, :], expected, rtol=1e-6, atol=0)  # storage-precision (float32) tolerance
    # the defining signatures: a COLD (negative) bottom cell, matching Veros's
    # measured t=0 abyss of -1.15 C on this grid
    assert expected[-1] < 0.0
    assert abs(expected[-1] - (-1.147)) < 0.05
    # surface ~14.9 C (not exactly 15: zt[0] != 0)
    assert 14.5 < expected[0] < 15.0


# ---------------------------------------------------------------------------
# Oracle-card wiring diagram (VEROS_BLOCK_MAPPING) — the card<->catalog link
# ---------------------------------------------------------------------------


def test_veros_block_mapping_fields_are_real_config_fields():
    """Every bare top-level field the wiring diagram names must be a real
    ``LatLonCGridOceanConfig`` attribute (keeps the audit diagram honest, like
    NEMO_BLOCK_MAPPING). Non-bare entries (``ConstantsConfig``, ``physics.*``)
    are sub-config/documentation rows, skipped here."""
    from legoesm.ocean.state import LatLonCGridOceanConfig

    cfg = LatLonCGridOceanConfig.from_flat()
    # documentation rows whose middle column names a sub-config / dotted path
    # rather than a bare top-level scheme field (skipped by the field check).
    doc_rows = {"ConstantsConfig"}
    assert len(VEROS_BLOCK_MAPPING) >= 15
    for name, field, note in VEROS_BLOCK_MAPPING:
        assert name and note                       # documented
        if "." in field or field in doc_rows:      # sub-config / documentation row
            continue
        assert hasattr(cfg, field) or field in type(cfg).flat_fields(), field  # #501: grouped members          # bare top-level field must be real


def test_veros_block_mapping_covers_the_full_dycore_identity():
    """The wiring diagram documents EVERY scheme field in the ``veros_faithful_v1``
    catalog recipe — so the card's audit trail is complete w.r.t. the catalog's
    dycore identity (a new scheme key in the recipe without a mapping row fails)."""
    from legoesm.ocean.recipes import get_recipe

    identity_keys = set(get_recipe("veros_faithful_v1"))
    mapped_fields = {field for _, field, _ in VEROS_BLOCK_MAPPING}
    missing = identity_keys - mapped_fields
    assert not missing, f"dycore-identity keys absent from VEROS_BLOCK_MAPPING: {missing}"


def test_veros_block_mapping_matches_catalog_and_card():
    """The explicit card<->catalog link: for every ``veros_faithful_v1`` selector,
    the catalog value == the Veros card's assembled config value. This is the
    Veros analogue of test_recipes.TestCatalogMatchesFactories, asserted next to
    the card so the card module itself guards the link (non-vacuous: 16 keys)."""
    from legoesm.ocean.recipes import get_recipe

    identity = get_recipe("veros_faithful_v1")
    assert len(identity) >= 16
    mc = build_acc_model_config(with_surface_forcing=True)
    for key, catalog_value in identity.items():
        assert mc.flat_get(key) == catalog_value, key  # #501: grouped names via flat_get
