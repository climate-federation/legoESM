#!/usr/bin/env python
"""Test MOST bulk fluxes across all surface tiles and verify differentiability.

Tests the Obukhov length iteration (jax.lax.fori_loop) is differentiable
for land, ocean, sea ice, and lake tiles with all supported schemes.
"""

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.coupler.bulk_flux import compute_most_fluxes
from legoesm.coupler.config import CouplerConfig
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.coupler.coupler import ocean_tile_response
from legoesm.land.config import LandConfig
from legoesm.land.state import LandState
from legoesm.land.slab_land import step_land
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import SeaIceState
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState
from legoesm.coupler.lake.two_layer_lake import step_lake
from legoesm.core.field import Field


if __name__ == "__main__":
    shape = (6, 4, 4)
    dims = ("face", "x", "y")

    # Standard atmospheric forcing
    forcing = AtmToSurface(
        sw_down=jnp.full(shape, 200.0),
        lw_down=jnp.full(shape, 300.0),
        precip_total=jnp.full(shape, 1e-5),
        precip_snow=jnp.zeros(shape),
        T_lowest=jnp.full(shape, 285.0),
        q_lowest=jnp.full(shape, 0.006),
        u_lowest=jnp.full(shape, 7.0),
        v_lowest=jnp.full(shape, 3.0),
        p_lowest=jnp.full(shape, 95000.0),
        p_surface=jnp.full(shape, 101325.0),
        rho_lowest=jnp.full(shape, 1.2),
        cos_zenith=jnp.full(shape, 0.5),
        co2_ppmv=jnp.array(400.0),
        has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )

    print("=" * 70)
    print("MOST Bulk Flux — All Surface Tiles Differentiability Test")
    print("=" * 70)

    all_pass = True

    # =========================================================================
    # 1. Ocean tile
    # =========================================================================
    print("\n--- OCEAN ---")
    for scheme in ["constant", "coare3", "large_yeager"]:
        config = CouplerConfig(bulk_scheme=scheme)
        sst = jnp.full(shape, 295.0) + 3.0 * jnp.sin(jnp.linspace(0, 3, 4))[None, :, None]

        def ocean_loss(sst_in):
            r = ocean_tile_response(forcing, sst_in, jnp.zeros(shape), jnp.zeros(shape), config)
            return jnp.mean(r.shflx ** 2 + r.tau_x ** 2)

        grads = jax.grad(ocean_loss)(sst)
        ok = bool(jnp.all(jnp.isfinite(grads)) and not jnp.allclose(grads, 0.0))
        grad_norm = float(jnp.max(jnp.abs(grads)))
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {scheme:12s}  |grad|_max={grad_norm:.4e}")
        if not ok:
            all_pass = False

    # =========================================================================
    # 2. Land tile
    # =========================================================================
    print("\n--- LAND ---")
    for scheme in ["constant", "most"]:
        land_config = LandConfig(bulk_scheme=scheme)
        T_init = jnp.full(shape, 290.0) + 5.0 * jnp.sin(jnp.linspace(0, 3, 4))[None, :, None]
        land_state = LandState(
            T_soil=Field(data=T_init, name="T_soil", dims=dims, units="K"),
            W_bucket=Field(data=jnp.full(shape, 75.0), name="W_bucket", dims=dims, units="kg/m2"),
            snow_depth=Field(data=jnp.zeros(shape), name="snow_depth", dims=dims, units="kg/m2"),
            snow_age=Field(data=jnp.zeros(shape), name="snow_age", dims=dims, units="s"),
        )

        def land_loss(T_in):
            s = LandState(
                T_soil=land_state.T_soil.replace(data=T_in),
                W_bucket=land_state.W_bucket,
                snow_depth=land_state.snow_depth,
                snow_age=land_state.snow_age,
            )
            new_s, resp, _ = step_land(s, forcing, land_config, 1.0, 3600.0)
            return jnp.mean(resp.shflx ** 2 + resp.tau_x ** 2)

        grads = jax.grad(land_loss)(T_init)
        ok = bool(jnp.all(jnp.isfinite(grads)) and not jnp.allclose(grads, 0.0))
        grad_norm = float(jnp.max(jnp.abs(grads)))
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {scheme:12s}  |grad|_max={grad_norm:.4e}")
        if not ok:
            all_pass = False

    # =========================================================================
    # 3. Sea ice tile
    # =========================================================================
    print("\n--- SEA ICE ---")
    for scheme in ["constant", "most"]:
        ice_config = SeaIceConfig(bulk_scheme=scheme)
        T_ice_init = jnp.full(shape, 260.0) + 5.0 * jnp.sin(jnp.linspace(0, 3, 4))[None, :, None]
        ice_state = SeaIceState(
            h_ice=Field(data=jnp.full(shape, 1.0), name="h_ice", dims=dims, units="m"),
            T_ice=Field(data=T_ice_init, name="T_ice", dims=dims, units="K"),
            concentration=Field(data=jnp.full(shape, 0.8), name="conc", dims=dims, units="1"),
        )
        ocean_sst = jnp.full(shape, constants.T_freeze_ocean)

        def ice_loss(T_in):
            s = SeaIceState(
                h_ice=ice_state.h_ice,
                T_ice=ice_state.T_ice.replace(data=T_in),
                concentration=ice_state.concentration,
            )
            new_s, resp = step_sea_ice(
                s, forcing, ocean_sst, jnp.zeros(shape), jnp.zeros(shape),
                ice_config, 1.0, 3600.0,
            )
            return jnp.mean(resp.shflx ** 2 + resp.tau_x ** 2)

        grads = jax.grad(ice_loss)(T_ice_init)
        ok = bool(jnp.all(jnp.isfinite(grads)) and not jnp.allclose(grads, 0.0))
        grad_norm = float(jnp.max(jnp.abs(grads)))
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {scheme:12s}  |grad|_max={grad_norm:.4e}")
        if not ok:
            all_pass = False

    # =========================================================================
    # 4. Lake tile
    # =========================================================================
    print("\n--- LAKE ---")
    for scheme in ["constant", "most"]:
        lake_config = LakeConfig(bulk_scheme=scheme)
        T_epi_init = jnp.full(shape, 290.0) + 3.0 * jnp.sin(jnp.linspace(0, 3, 4))[None, :, None]
        lake_state = LakeState(
            T_epi=Field(data=T_epi_init, name="T_epi", dims=dims, units="K"),
            T_hypo=Field(data=jnp.full(shape, 280.0), name="T_hypo", dims=dims, units="K"),
        )

        def lake_loss(T_in):
            s = LakeState(
                T_epi=lake_state.T_epi.replace(data=T_in),
                T_hypo=lake_state.T_hypo,
            )
            new_s, resp = step_lake(s, forcing, lake_config, 1.0, 3600.0)
            return jnp.mean(resp.shflx ** 2 + resp.tau_x ** 2)

        grads = jax.grad(lake_loss)(T_epi_init)
        ok = bool(jnp.all(jnp.isfinite(grads)) and not jnp.allclose(grads, 0.0))
        grad_norm = float(jnp.max(jnp.abs(grads)))
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {scheme:12s}  |grad|_max={grad_norm:.4e}")
        if not ok:
            all_pass = False

    # =========================================================================
    # 5. Multi-step land differentiability (the real test)
    # =========================================================================
    print("\n--- LAND MULTI-STEP (5 steps with MOST) ---")
    land_config = LandConfig(bulk_scheme="most", z_ref=10.0, bulk_n_iter=5)
    T_init = jnp.full(shape, 290.0) + 5.0 * jnp.sin(jnp.linspace(0, 3, 4))[None, :, None]
    land_state = LandState(
        T_soil=Field(data=T_init, name="T_soil", dims=dims, units="K"),
        W_bucket=Field(data=jnp.full(shape, 75.0), name="W_bucket", dims=dims, units="kg/m2"),
        snow_depth=Field(data=jnp.zeros(shape), name="snow_depth", dims=dims, units="kg/m2"),
        snow_age=Field(data=jnp.zeros(shape), name="snow_age", dims=dims, units="s"),
    )

    def land_multistep_loss(T_in):
        s = LandState(
            T_soil=land_state.T_soil.replace(data=T_in),
            W_bucket=land_state.W_bucket,
            snow_depth=land_state.snow_depth,
            snow_age=land_state.snow_age,
        )
        for _ in range(5):
            s, _, _ = step_land(s, forcing, land_config, 1.0, 1800.0)
        return jnp.mean(s.T_soil.data ** 2)

    grads = jax.grad(land_multistep_loss)(T_init)
    ok = bool(jnp.all(jnp.isfinite(grads)) and not jnp.allclose(grads, 0.0))
    grad_norm = float(jnp.max(jnp.abs(grads)))
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  5-step grad  |grad|_max={grad_norm:.4e}")
    if not ok:
        all_pass = False

    print()
    if all_pass:
        print("All surface tile MOST bulk flux tests PASSED.")
    else:
        print("Some tests FAILED — see above.")
