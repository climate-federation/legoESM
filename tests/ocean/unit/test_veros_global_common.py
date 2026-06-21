"""Direct tests for the shared VEROS-global recipe builders.

Covers :mod:`legoesm.ocean.fidelity.veros_global_common` and confirms the
three recipes still expose byte-identical results after routing through it.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
from legoesm.ocean.fidelity.veros_global_common import (
    build_veros_global_state,
    gm_redi_eke_isopycnal_on,
    veros_area_t_generic,
)


def test_area_generic_matches_hand_formula():
    yt = np.array([-60.0, 0.0, 30.0, 75.0])
    dxt, dyt = 1.0, 1.0
    r = VEROS_CONSTANTS_CONFIG.R_earth
    degtom = r * np.pi / 180.0
    expected = (dxt * degtom) * (dyt * degtom) * np.cos(np.deg2rad(yt))
    got = veros_area_t_generic(yt, dxt, dyt, r)
    np.testing.assert_array_equal(got, expected)


def test_area_generic_accepts_per_row_dyt():
    """Stretched grids pass a per-row dyt array; broadcasting must hold."""
    yt = np.array([0.0, 45.0])
    dyt = np.array([2.0, 3.0])
    dxt = 4.0
    r = VEROS_CONSTANTS_CONFIG.R_earth
    degtom = r * np.pi / 180.0
    expected = (dxt * degtom) * (dyt * degtom) * np.cos(np.deg2rad(yt))
    np.testing.assert_array_equal(veros_area_t_generic(yt, dxt, dyt, r), expected)


def test_gm_redi_eke_isopycnal_on_matches_recipes():
    from legoesm.ocean.fidelity.veros_global_1deg_recipe import (
        GLOBAL_1DEG_EKE_CONFIG,
        GLOBAL_1DEG_GM_REDI_CONFIG,
    )
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
        GLOBAL4_EKE_CONFIG,
        GLOBAL4_GM_REDI_CONFIG,
    )
    from legoesm.ocean.fidelity.veros_global_flexible_recipe import (
        GLOBAL_FLEX_EKE_CONFIG,
        GLOBAL_FLEX_GM_REDI_CONFIG,
    )

    gm, eke = gm_redi_eke_isopycnal_on(GLOBAL4_GM_REDI_CONFIG, GLOBAL4_EKE_CONFIG)
    # The shared delta == both recipes' public configs (1deg/flex identical).
    assert gm == GLOBAL_1DEG_GM_REDI_CONFIG
    assert gm == GLOBAL_FLEX_GM_REDI_CONFIG
    assert eke == GLOBAL_1DEG_EKE_CONFIG
    assert eke == GLOBAL_FLEX_EKE_CONFIG
    # Specific deltas vs the 4deg base.
    assert eke.isopycnal_diffusion is True
    assert GLOBAL4_EKE_CONFIG.isopycnal_diffusion is False
    assert gm.S_max == pytest.approx(5.0e-3)
    assert gm.taper_width_frac == pytest.approx(1.0)
    assert gm.K_iso_steep == pytest.approx(50.0)
    assert gm.eke == eke


def _build_4deg_inputs():
    """A valid (grid, partial-cell z_coord, land_mask, H_bathy) — wall rows
    land, interior fully wet at full column depth (mirrors the recipe path)."""
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
        GLOBAL4_DDZ,
        NX,
        NY,
        build_global_4deg_grid,
        build_global_4deg_z_coord,
    )
    from legoesm.ocean.vertical import create_partial_cell_coordinate

    grid = build_global_4deg_grid()
    zc = build_global_4deg_z_coord()
    H_full = float(GLOBAL4_DDZ.sum())
    land_mask = np.ones((NY + 2, NX))
    land_mask[0, :] = 0.0
    land_mask[-1, :] = 0.0
    H_bathy = np.full((NY + 2, NX), H_full)
    H_bathy[0, :] = 0.0
    H_bathy[-1, :] = 0.0
    z = create_partial_cell_coordinate(zc, jnp.asarray(H_bathy))
    return grid, z, land_mask, H_bathy


def test_build_state_seeds_tke_eke_and_masks_tracers():
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
        GLOBAL4_EKE_CONFIG,
        GLOBAL4_TKE_CONFIG,
    )

    grid, z, land_mask, H_bathy = _build_4deg_inputs()
    nz = z.n_levels

    state = build_veros_global_state(
        grid, z, land_mask, H_bathy,
        tke_config=GLOBAL4_TKE_CONFIG, eke_config=GLOBAL4_EKE_CONFIG,
    )
    # Rest velocity + zero eta.
    assert float(np.max(np.abs(np.asarray(state.u.data)))) == 0.0
    assert float(np.max(np.abs(np.asarray(state.eta.data)))) == 0.0
    # TKE/EKE seeded at config floors over wet cells, zeroed over land rows.
    tke = np.asarray(state.tke.data)
    eke = np.asarray(state.eke.data)
    assert tke.shape[-1] == nz - 1
    assert np.allclose(tke[1:-1].max(), GLOBAL4_TKE_CONFIG.tke_background)
    assert np.allclose(eke[1:-1].max(), GLOBAL4_EKE_CONFIG.e_min)
    assert np.all(tke[0] == 0.0) and np.all(tke[-1] == 0.0)
    # AB2 histories seed at zero.
    assert float(np.max(np.abs(np.asarray(state.dtke.data)))) == 0.0
    assert float(np.max(np.abs(np.asarray(state.eke_diss.data)))) == 0.0


def test_4deg_literals_equal_zcoord_fields():
    """The OLD 4° builder passed the module literals ``H_max=H_MAX`` and used
    ``NZ - 1``; the shared builder reads ``float(z_coord.H_max)`` and
    ``z_coord.n_levels - 1``.  The migrated call site is byte-identical ONLY if
    these are exactly equal, so assert it directly against the recipe's own
    ``H_MAX``/``NZ`` constants (not a re-derived value)."""
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import H_MAX, NZ

    _grid, z, _lm, _hb = _build_4deg_inputs()
    assert z.n_levels == NZ                      # NZ - 1 == n_levels - 1
    assert float(z.H_max) == H_MAX               # the exact float the old call passed


def _old_reference_build_state(grid, z_coord, land_mask, H_bathy,
                               tke_config, eke_config, T_init, S_init):
    """The pre-refactor 4° inline seeding body, VERBATIM — using the module
    literals ``H_max=H_MAX`` and ``NZ`` exactly as the old 4° builder did, so
    the comparison exercises the true old call path (not a re-derivation)."""
    import jax.numpy as jnp

    from legoesm.core.field import Field
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import H_MAX, NZ
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean

    nz = NZ
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        S_uniform=35.0, H_max=H_MAX,
        land_mask_override=jnp.asarray(land_mask),
        H_bathy_override=jnp.asarray(H_bathy),
    )
    is_active = jnp.asarray(z_coord.is_active, dtype=state.T.data.dtype)
    if T_init is not None:
        state = state._replace(
            T=state.T.replace(data=jnp.asarray(T_init) * is_active))
    if S_init is not None:
        state = state._replace(
            S=state.S.replace(data=jnp.asarray(S_init) * is_active))
    lm = state.land_mask.data
    dtype = state.T.data.dtype
    wet3 = (lm[:, :, jnp.newaxis] > 0.5) * jnp.ones((1, 1, nz - 1), dtype=dtype)
    tke0 = tke_config.tke_background * wet3
    state = state._replace(
        tke=Field(data=tke0, name="tke", dims=("lat", "lon", "level"),
                  units="m^2/s^2"),
        dtke=Field(data=jnp.zeros_like(tke0), name="dtke",
                   dims=("lat", "lon", "level"), units="m^2/s^3"))
    eke0 = eke_config.e_min * wet3
    state = state._replace(
        eke=Field(data=eke0, name="eke", dims=("lat", "lon", "level"),
                  units="m^2/s^2"),
        eke_diss=Field(data=jnp.zeros_like(eke0), name="eke_diss",
                       dims=("lat", "lon", "level"), units="m^2/s^3"))
    return state


def test_build_state_byte_identical_to_old_inline_body():
    """The shared builder reproduces the pre-refactor inline seeding exactly
    (every prognostic + carry field array-equal)."""
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
        GLOBAL4_EKE_CONFIG,
        GLOBAL4_TKE_CONFIG,
    )

    grid, z, land_mask, H_bathy = _build_4deg_inputs()
    n_lat, n_lon = land_mask.shape
    nz = z.n_levels
    rng = np.random.default_rng(11)
    T_init = rng.standard_normal((n_lat, n_lon, nz)) + 4.0
    S_init = rng.standard_normal((n_lat, n_lon, nz)) + 34.0

    new = build_veros_global_state(
        grid, z, land_mask, H_bathy,
        tke_config=GLOBAL4_TKE_CONFIG, eke_config=GLOBAL4_EKE_CONFIG,
        T_init=T_init, S_init=S_init)
    old = _old_reference_build_state(
        grid, z, land_mask, H_bathy,
        GLOBAL4_TKE_CONFIG, GLOBAL4_EKE_CONFIG, T_init, S_init)
    for f in ("T", "S", "u", "v", "eta", "tke", "dtke", "eke", "eke_diss"):
        np.testing.assert_array_equal(
            np.asarray(getattr(new, f).data),
            np.asarray(getattr(old, f).data), err_msg=f)


def test_build_state_t_s_masked_by_active():
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
        GLOBAL4_EKE_CONFIG,
        GLOBAL4_TKE_CONFIG,
    )

    grid, z, land_mask, H_bathy = _build_4deg_inputs()
    n_lat, n_lon = land_mask.shape
    nz = z.n_levels
    T_init = np.full((n_lat, n_lon, nz), 5.0)
    S_init = np.full((n_lat, n_lon, nz), 34.0)

    state = build_veros_global_state(
        grid, z, land_mask, H_bathy,
        tke_config=GLOBAL4_TKE_CONFIG, eke_config=GLOBAL4_EKE_CONFIG,
        T_init=T_init, S_init=S_init,
    )
    is_active = np.asarray(z.is_active)
    np.testing.assert_array_equal(np.asarray(state.T.data), T_init * is_active)
    np.testing.assert_array_equal(np.asarray(state.S.data), S_init * is_active)
