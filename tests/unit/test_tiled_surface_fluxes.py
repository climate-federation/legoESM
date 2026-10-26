"""Tiled (mosaic) surface fluxes — :func:`compute_tiled_surface_fluxes`.

The AMIP surface used to blend ocean / sea-ice / land surface TEMPERATURES
into one number and then apply a single bulk scheme to the blend.  With
COARE 3.0 selected, that ran the ocean air-sea scheme over land columns
(Charnock wave roughness + gustiness over a forest) — the bm_v3 land-tile
energy blowup.

``compute_tiled_surface_fluxes`` instead computes the fluxes SEPARATELY per
tile (COARE3 on the ocean, the fixed-roughness land Monin-Obukhov ``"most"``
scheme on land) and AREA-WEIGHTS the resulting fluxes.  These tests pin the
contract that makes that correct:

* single-tile limits reduce to the per-tile scheme (and the land/ocean
  schemes are genuinely different — the split is not cosmetic),
* heat/momentum fluxes aggregate LINEARLY by area fraction,
* the land roughness ``z0`` is LIVE under the land tile (a real gradient),
* the helper is JIT- and grad-safe.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    SurfaceTileSpec,
    compute_surface_fluxes,
    compute_tiled_surface_fluxes,
)
from legoesm.core.bulk_flux import compute_most_fluxes


_OCEAN_CFG = SurfaceLayerConfig(bulk_scheme="coare3", z0=1e-4, z_ref=10.0, bulk_n_iter=5)
_ICE_CFG = SurfaceLayerConfig(bulk_scheme="constant", Cd_neutral=1.5e-3, Ch_neutral=1.5e-3)
_LAND_CFG = SurfaceLayerConfig(bulk_scheme="most", z0=0.1, z_ref=10.0, bulk_n_iter=5)


def _atm(ncol: int = 6):
    """A representative lowest-level atmospheric column state."""
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.uniform(-8.0, 8.0, ncol))
    v = jnp.asarray(rng.uniform(-8.0, 8.0, ncol))
    T = jnp.asarray(rng.uniform(280.0, 300.0, ncol))
    q_v = jnp.asarray(rng.uniform(2e-3, 1.5e-2, ncol))
    rho = jnp.asarray(rng.uniform(1.0, 1.25, ncol))
    return u, v, T, q_v, rho


def _tiles(frac_ocean, frac_ice, frac_land, *, ncol=6, T_ocean=298.0,
           T_ice=271.0, T_land=305.0, q_o=1.6e-2, q_i=3e-3, q_l=1.8e-2):
    one = jnp.ones(ncol)
    return SurfaceTileSpec(
        frac_ocean=one * frac_ocean,
        frac_ice=one * frac_ice,
        frac_land=one * frac_land,
        T_ocean=one * T_ocean,
        T_ice=one * T_ice,
        T_land=one * T_land,
        q_sfc_ocean=one * q_o,
        q_sfc_ice=one * q_i,
        q_sfc_land=one * q_l,
    )


def _blend_call(u, v, T, q_v, rho, tiles):
    """Invoke the helper with the standard ocean/ice/land scheme configs."""
    return compute_tiled_surface_fluxes(
        u, v, T, q_v, rho, tiles, _OCEAN_CFG, _ICE_CFG, _LAND_CFG,
    )


def test_all_ocean_reduces_to_coare3():
    """frac_ocean=1 ⇒ blended flux == bare COARE3 ocean flux."""
    u, v, T, q_v, rho = _atm()
    tiles = _tiles(1.0, 0.0, 0.0)
    blended = _blend_call(u, v, T, q_v, rho, tiles)

    ref = compute_most_fluxes(
        u, v, T, q_v, tiles.T_ocean, tiles.q_sfc_ocean, rho,
        z_ref=_OCEAN_CFG.z_ref, z0_init=_OCEAN_CFG.z0,
        scheme="coare3", n_iter=_OCEAN_CFG.bulk_n_iter,
    )
    # tau/SH/LH match exactly; u* is re-derived from blended stress (consistent
    # with |tau|=rho u*^2) so compare it against the same relation.
    for got, exp in zip(blended[:4], ref[:4]):
        np.testing.assert_allclose(np.asarray(got), np.asarray(exp), rtol=1e-10)
    tau_mag = jnp.sqrt(ref[0] ** 2 + ref[1] ** 2)
    np.testing.assert_allclose(
        np.asarray(blended[4]), np.asarray(jnp.sqrt(tau_mag / rho)), rtol=1e-10,
    )


def test_all_land_uses_iterative_most_not_constant():
    """frac_land=1 ⇒ blended flux == fixed-roughness MOST (z0 live), and it
    differs from the constant-coefficient path (the land scheme is real)."""
    u, v, T, q_v, rho = _atm()
    tiles = _tiles(0.0, 0.0, 1.0)
    blended = _blend_call(u, v, T, q_v, rho, tiles)

    most_ref = compute_most_fluxes(
        u, v, T, q_v, tiles.T_land, tiles.q_sfc_land, rho,
        z_ref=_LAND_CFG.z_ref, z0_init=_LAND_CFG.z0,
        scheme="most", n_iter=_LAND_CFG.bulk_n_iter,
    )
    for got, exp in zip(blended[:4], most_ref[:4]):
        np.testing.assert_allclose(np.asarray(got), np.asarray(exp), rtol=1e-10)

    # Constant-coefficient surface flux at the same land surface must DIFFER —
    # otherwise "most" silently degraded to constant (the surface_layer.py bug).
    const_cfg = SurfaceLayerConfig(bulk_scheme="constant")
    const = compute_surface_fluxes(
        u, v, T, q_v, tiles.T_land, tiles.q_sfc_land, rho, const_cfg,
    )
    assert not np.allclose(np.asarray(blended[2]), np.asarray(const[2]), rtol=1e-3)


def test_fluxes_aggregate_linearly():
    """Heat/momentum fluxes are an exact area-weighted blend of the tiles."""
    u, v, T, q_v, rho = _atm()
    fo, fi, fl = 0.5, 0.2, 0.3
    tiles = _tiles(fo, fi, fl)
    blended = _blend_call(u, v, T, q_v, rho, tiles)

    pure_o = _blend_call(u, v, T, q_v, rho, _tiles(1.0, 0.0, 0.0))
    pure_i = _blend_call(u, v, T, q_v, rho, _tiles(0.0, 1.0, 0.0))
    pure_l = _blend_call(u, v, T, q_v, rho, _tiles(0.0, 0.0, 1.0))

    for idx in range(4):  # tau_x, tau_y, shflx, lhflx
        exp = fo * pure_o[idx] + fi * pure_i[idx] + fl * pure_l[idx]
        np.testing.assert_allclose(np.asarray(blended[idx]), np.asarray(exp), rtol=1e-10)


def test_partition_of_unity_independent_of_tile_count():
    """A 100%-land cell and a land cell whose 'ocean'/'ice' tiles carry zero
    weight give the same answer (zero-fraction tiles do not leak)."""
    u, v, T, q_v, rho = _atm()
    a = _blend_call(u, v, T, q_v, rho, _tiles(0.0, 0.0, 1.0))
    # Same land tile, different (irrelevant) ocean/ice surface states, weight 0.
    b_tiles = _tiles(0.0, 0.0, 1.0, T_ocean=320.0, T_ice=250.0)
    b = _blend_call(u, v, T, q_v, rho, b_tiles)
    for idx in range(5):
        np.testing.assert_allclose(np.asarray(a[idx]), np.asarray(b[idx]), rtol=1e-12)


def test_land_z0_is_live_under_grad():
    """∂(land sensible heat)/∂z0_land is finite and non-zero — the land
    roughness genuinely enters the flux (Pierre's land Monin-Obukhov)."""
    u, v, T, q_v, rho = _atm()

    tiles = _tiles(0.0, 0.0, 1.0)

    def loss(z0_land):
        cfg = _LAND_CFG._replace(z0=z0_land)
        _, _, shflx, lhflx, _ = compute_tiled_surface_fluxes(
            u, v, T, q_v, rho, tiles, _OCEAN_CFG, _ICE_CFG, cfg,
        )
        return jnp.sum(shflx + lhflx)

    g = jax.grad(loss)(0.1)
    assert np.isfinite(float(g))
    assert abs(float(g)) > 0.0


def test_jit_executes():
    u, v, T, q_v, rho = _atm()
    tiles = _tiles(0.4, 0.1, 0.5)
    # Configs carry the static bulk_scheme string -> static_argnums.
    jit_fn = jax.jit(compute_tiled_surface_fluxes, static_argnums=(6, 7, 8))
    out = jit_fn(u, v, T, q_v, rho, tiles, _OCEAN_CFG, _ICE_CFG, _LAND_CFG)
    eager = _blend_call(u, v, T, q_v, rho, tiles)
    for got, exp in zip(out, eager):
        np.testing.assert_allclose(np.asarray(got), np.asarray(exp), rtol=1e-10)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_return_water_is_the_sum_of_per_tile_inverses():
    """``return_water=True`` hands back the water each tile's law moved,
    inverted with the L that law CHARGED (every tile: Kirchhoff L_v at the
    tile's own temperature, since _single_tile_flux passes no L_latent) --
    not the blended heat over one L at one temperature."""
    from legoesm.thermo import latent_heat_vaporization as L
    u, v, T, q_v, rho = _atm()
    tiles = _tiles(0.5, 0.3, 0.2)
    (tau_x, tau_y, shflx, lhflx, ustar), water = compute_tiled_surface_fluxes(
        u, v, T, q_v, rho, tiles, _OCEAN_CFG, _ICE_CFG, _LAND_CFG, return_water=True)
    # The tuple is unchanged by the flag.
    for got, exp in zip((tau_x, tau_y, shflx, lhflx, ustar),
                        _blend_call(u, v, T, q_v, rho, tiles)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(exp))
    per_tile = []
    for frac, T_s, q_s, cfg in ((1.0, tiles.T_ocean, tiles.q_sfc_ocean, _OCEAN_CFG),
                                (1.0, tiles.T_ice, tiles.q_sfc_ice, _ICE_CFG),
                                (1.0, tiles.T_land, tiles.q_sfc_land, _LAND_CFG)):
        pure = SurfaceTileSpec(frac_ocean=jnp.zeros(6), frac_ice=jnp.zeros(6),
                               frac_land=jnp.zeros(6), T_ocean=T_s, T_ice=T_s,
                               T_land=T_s, q_sfc_ocean=q_s, q_sfc_ice=q_s,
                               q_sfc_land=q_s)
        # Run the single law alone through the same helper (one tile at 100 %).
        lh_i = compute_tiled_surface_fluxes(
            u, v, T, q_v, rho, pure._replace(frac_ocean=jnp.ones(6), T_ocean=T_s,
                                              q_sfc_ocean=q_s),
            cfg, cfg, cfg)[3]
        per_tile.append(lh_i / L(T_s))
    expected = (tiles.frac_ocean * per_tile[0] + tiles.frac_ice * per_tile[1]
                + tiles.frac_land * per_tile[2])
    np.testing.assert_allclose(np.asarray(water), np.asarray(expected), rtol=1e-10)
    # ...and it is NOT the blended heat over one L (the ice tile at 271 K vs
    # land at 305 K differ by ~3 % in L_v).
    T_blend = (tiles.frac_ocean * tiles.T_ocean + tiles.frac_ice * tiles.T_ice
               + tiles.frac_land * tiles.T_land)
    assert float(jnp.max(jnp.abs(water / (lhflx / L(T_blend)) - 1.0))) > 1e-4
