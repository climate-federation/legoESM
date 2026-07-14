"""FV3_3D iter 219: sponge-layer-aware delt_max cap (refines iter 218).

iter-218 implemented a flat per-step cap on dissipative heating
that applied to every level uniformly.  FV3 ``dyn_core.F90:1764-
1786`` actually treats the top-2 sponge layers specially:

* ``cp_air`` branch (PE in our naming): ``if (k<3)`` skips the cap
  entirely and applies the full heat increment.  In legoESM's 0-
  based level convention (``k=0`` = model top) this means **k=0
  and k=1 receive uncapped heating** (FV3 1-based k=1,2).
* ``cv_air`` branch (NH): the cap is **tighter** at the top.
  ``k=1`` (FV3) → ``delt = 0.1 * bdt * delt_max``; ``k=2`` (FV3)
  → ``delt = 0.5 * bdt * delt_max``; else ``delt = bdt * delt_max``.
  In legoESM's 0-based: k=0 → 0.1×, k=1 → 0.5×, k≥2 → 1×.

The asymmetry between PE (no cap) and NH (tightest cap) at the
top mirrors FV3's two distinct sponge implementations: PE has a
zonal-mean Rayleigh sponge upstream that already moderates wind
extrema, so the d_con cap can be permissive; NH relies on the
delt cap itself to bound dissipative heating in the absorbing
layer.

Tests
-----

1. ``test_pe_top_layers_uncapped`` — PE k=0,1 receive the full
   uncapped d_con heat even when ``delt_max > 0``.
2. ``test_pe_interior_layers_capped`` — PE k≥2 are capped to
   ``dt * delt_max``.
3. ``test_nh_top_layer_tighter_cap`` — NH k=0 cap is 0.1× the
   interior cap; per-step |Δθ_p*Π| ≤ 0.1*dt*delt_max there.
4. ``test_nh_layer_one_half_cap`` — NH k=1 cap is 0.5× interior.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
    standard_hybrid_levels,
)


def _pe_state():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=219)
    n_corners = n + 1
    u_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, cdgrid, coord, state


def _nh_state():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=2190)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev + 1))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_p), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def test_pe_top_layers_uncapped():
    """PE k=0,1 receive the FULL uncapped d_con heat even when
    ``delt_max > 0``: ``T_capped[k=0,1] == T_uncapped[k=0,1]``."""
    grid, cdgrid, coord, state = _pe_state()
    dt = 100.0
    common = dict(
        damp_v=0.030, nord_v=0, damp_v_d_con=1.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_uncapped = CDGridPrimitiveEquationConfig(
        **common, delt_max=0.0,
    )
    cfg_capped = CDGridPrimitiveEquationConfig(
        **common, delt_max=1e-4,    # very tight cap
    )
    s_uncapped = CDGridPrimitiveEquationModel(
        grid, coord, cfg_uncapped,
    ).step(state, dt)
    s_capped = CDGridPrimitiveEquationModel(
        grid, coord, cfg_capped,
    ).step(state, dt)

    # Top 2 sponge layers should be IDENTICAL (no cap applied).
    np.testing.assert_array_equal(
        s_capped.T.data[..., 0], s_uncapped.T.data[..., 0],
        err_msg="PE k=0 (top sponge) must be uncapped.",
    )
    np.testing.assert_array_equal(
        s_capped.T.data[..., 1], s_uncapped.T.data[..., 1],
        err_msg="PE k=1 (sponge) must be uncapped.",
    )


def test_pe_interior_layers_capped():
    """PE interior k≥2 is capped: max|d_con-only ΔT| ≤ dt*delt_max
    in those layers."""
    grid, cdgrid, coord, state = _pe_state()
    dt = 100.0
    delt_max = 1e-4
    common = dict(
        damp_v=0.030, nord_v=0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_no_dcon = CDGridPrimitiveEquationConfig(
        **common, damp_v_d_con=0.0,
    )
    cfg_capped = CDGridPrimitiveEquationConfig(
        **common, damp_v_d_con=1.0, delt_max=delt_max,
    )
    s_no_dcon = CDGridPrimitiveEquationModel(
        grid, coord, cfg_no_dcon,
    ).step(state, dt)
    s_capped = CDGridPrimitiveEquationModel(
        grid, coord, cfg_capped,
    ).step(state, dt)

    d_con_only = s_capped.T.data - s_no_dcon.T.data
    cap = dt * delt_max

    # Interior layers (k>=2) must respect the cap.
    interior = d_con_only[..., 2:]
    max_interior = float(jnp.max(jnp.abs(interior)))
    assert max_interior <= cap + 1e-12, (
        f"PE interior d_con max|ΔT|={max_interior:.4e} > "
        f"cap={cap:.4e} — sponge-aware cap not bounding interior."
    )


def test_nh_top_layer_tighter_cap():
    """NH k=0 cap is 0.1× interior.  With damp_w_d_con on, the d_con
    contribution at k=0 must satisfy |Δθ_p * Π_ref| ≤ 0.1 * dt *
    delt_max."""
    grid, hc, tm, state = _nh_state()
    dt = 10.0
    delt_max = 1e-3    # so 0.1*dt*delt_max = 1e-3 K, tight
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.30, nord_w=0,
    )
    cfg_no_dcon = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=0.0,
    )
    cfg_capped = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=1.0, delt_max=delt_max,
    )
    s_no = CDGridCompressibleEulerModel(grid, hc, tm, cfg_no_dcon).step(
        state, dt,
    )
    s_cap = CDGridCompressibleEulerModel(grid, hc, tm, cfg_capped).step(
        state, dt,
    )
    d_con_only = s_cap.theta_prime.data - s_no.theta_prime.data

    # Convert dtheta_p back to dT-equivalent: ΔT ≈ Δθ_p * Π_ref.
    exner_ref = hc.exner_ref    # (nlev,)
    delta_T_eq = d_con_only * exner_ref[None, None, None, :]

    # k=0: cap = 0.1 * dt * delt_max
    cap_top = 0.1 * dt * delt_max
    max_top = float(jnp.max(jnp.abs(delta_T_eq[..., 0])))
    assert max_top <= cap_top + 1e-12, (
        f"NH k=0 sponge: |ΔT_eq|={max_top:.4e} > cap_top="
        f"{cap_top:.4e} (factor 0.1 not applied)."
    )


def test_nh_layer_one_half_cap():
    """NH k=1 cap is 0.5× interior."""
    grid, hc, tm, state = _nh_state()
    dt = 10.0
    delt_max = 1e-3
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.30, nord_w=0,
    )
    cfg_no_dcon = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=0.0,
    )
    cfg_capped = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=1.0, delt_max=delt_max,
    )
    s_no = CDGridCompressibleEulerModel(grid, hc, tm, cfg_no_dcon).step(
        state, dt,
    )
    s_cap = CDGridCompressibleEulerModel(grid, hc, tm, cfg_capped).step(
        state, dt,
    )
    d_con_only = s_cap.theta_prime.data - s_no.theta_prime.data
    exner_ref = hc.exner_ref

    delta_T_eq = d_con_only * exner_ref[None, None, None, :]

    # k=1: cap = 0.5 * dt * delt_max
    cap_lvl1 = 0.5 * dt * delt_max
    max_lvl1 = float(jnp.max(jnp.abs(delta_T_eq[..., 1])))
    assert max_lvl1 <= cap_lvl1 + 1e-12, (
        f"NH k=1 sponge: |ΔT_eq|={max_lvl1:.4e} > cap_lvl1="
        f"{cap_lvl1:.4e} (factor 0.5 not applied)."
    )

    # And k>=2: cap = 1.0 * dt * delt_max
    cap_full = dt * delt_max
    max_interior = float(
        jnp.max(jnp.abs(delta_T_eq[..., 2:])),
    )
    assert max_interior <= cap_full + 1e-12, (
        f"NH k>=2: |ΔT_eq|={max_interior:.4e} > cap_full="
        f"{cap_full:.4e}."
    )
