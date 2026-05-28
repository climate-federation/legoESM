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

from legoesm.ocean.fidelity.recipe_constants import (
    VEROS_CONSTANTS, override_constants,
)
from legoesm.ocean.fidelity.tendency_probe import (
    build_region_masks, probe_latlon_cgrid,
)
from legoesm.ocean.fidelity.veros_acc_recipe import (
    ACC_DZT, ACC_GM_REDI_CONFIG, ACC_TKE_CONFIG, NX, NY, NZ,
    acc_A_h, build_acc_grid, build_acc_recipe, build_acc_z_coord,
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
    with override_constants(R_earth=R):
        A_h = acc_A_h()
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


def test_z_coord_dz_ref_in_legoesm_order():
    """legoESM convention: dz_ref[0] = surface = 20 m, dz_ref[-1] =
    bottom = 276 m (Veros stores the same values reversed)."""
    z = build_acc_z_coord()
    dz = np.asarray(z.dz_ref)
    assert dz[0] == pytest.approx(20.0)
    assert dz[-1] == pytest.approx(276.0)
    np.testing.assert_allclose(float(np.sum(dz)), z.H_max, atol=1e-10)


def test_full_recipe_builds_under_constants_override():
    """The complete builder must produce a usable
    (model_config, physics_config, grid, z_coord, state) tuple inside
    ``override_constants(**VEROS_CONSTANTS)``."""
    with override_constants(**VEROS_CONSTANTS):
        recipe = build_acc_recipe()
    assert recipe.model_config.eos == "veros_nonlin2"
    assert recipe.model_config.A_h_lat_scaling is True
    assert recipe.model_config.A_h_cos_power == 1
    assert recipe.model_config.implicit_vertical_mixing is True
    assert recipe.physics_config.vertical_mixing.scheme == "tke"
    assert recipe.physics_config.lateral_mixing.scheme == "gm_redi"
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
    with override_constants(**VEROS_CONSTANTS):
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
    with override_constants(**VEROS_CONSTANTS):
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
