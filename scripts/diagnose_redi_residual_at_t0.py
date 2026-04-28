"""Diagnose where the Redi-only 1.14 K divergence at 120 days comes from.

Three checks, each with print output you can read top-to-bottom:

  (1) At t = 0 in the *production* Eady setup (n_lat=22, nlev=20, with
      walls and the actual model EOS path), call the GM/Redi
      orchestrator directly and report max|dT|.  If this is at
      machine precision the per-step cancellation is genuinely
      working in production — so the 1.14 K cannot come from the
      per-step Redi residual.

  (2) Inspect the F_x, F_y, F_z fluxes from the triad function at the
      wall-adjacent rows (i = 0, n_lat-1, 1, n_lat-2), at the surface
      (k = 0), and at the bottom (k = nlev-1).  This answers Dhruv's
      question: does the flux genuinely vanish at boundaries, or is
      there cross-isopycnal flux there that the volume integral hides?

  (3) Compare ``gm_redi=None`` (path turned OFF) vs
      ``gm_redi=GMRediConfig(kappa=0, slope=triads)`` (path called but
      returns zero) bit-for-bit at t = 0 after one ``model.step``.
      Any difference here means the *path itself* — not just the
      Redi flux — perturbs the dynamics.

Run with:
    JAX_ENABLE_X64=1 python scripts/diagnose_redi_residual_at_t0.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.experiments.eady_uniform import (
    EadyUniformConfig,
    create_initial_conditions as eu_ic,
    create_forcings as eu_forcings,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    gm_redi_tracer_tendency_latlon,
)
from legoesm.ocean.eos import LinearEOSConfig
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _neumann_fill_cgrid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    gradient_x_cgrid, gradient_y_cgrid, divergence_cgrid,
)
from legoesm.ocean.vertical import compute_ocean_jacobian

from ocean_test_matrix.experiments import _create_ocean_setup
from ocean_test_matrix.testcase import TestCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_eady(slope_scheme: str, kappa_Redi: float, gm_off: bool = False):
    """Return (grid, z_coord, model, state, gm_cfg, eu) at t = 0."""
    eu = EadyUniformConfig(
        T_perturbation_K=0.0, U_surface=0.5, N=1.2e-3,
        A_h=0.0, B_h=0.0, C_smag=0.0, K_h=0.0, K_bih=0.0,
        sponge_width_deg=2.0,
    )
    gm_cfg = (None if gm_off
              else GMRediConfig(
                  kappa_GM=0.0, kappa_Redi=kappa_Redi, S_max=0.01,
                  slope_scheme=slope_scheme,
              ))
    physics = eu_forcings("latlon_channel", None, eu)
    tc = TestCase(
        case="diag", grid_type="latlon_channel", resolution="20x10",
        duration_days=1.0, quick_days=0.5,
        run_kwargs={
            "lat_south": eu.lat_south, "lat_north": eu.lat_north,
            "lon_west": eu.lon_west, "lon_east": eu.lon_east,
        },
    )
    grid, z_coord, _, model, _, _, _ = _create_ocean_setup(
        tc, nlev=20, physics=physics,
        A_h=eu.A_h, B_h=eu.B_h, C_smag=eu.C_smag,
        K_h=eu.K_h, K_bih=eu.K_bih,
        A_v=eu.A_v, K_v=eu.K_v,
        bottom_drag_r=eu.bottom_drag_coeff,
        eos="linear",
        eos_linear=LinearEOSConfig(
            alpha_T=eu.alpha_T, rho_ref=eu.rho_0,
            T_ref=eu.T_ref, S_ref=eu.S_uniform,
        ),
        barotropic_diffusion_alpha=eu.barotropic_diffusion_alpha,
        barotropic_div_damp=eu.barotropic_div_damp,
        tracer_advection=eu.tracer_advection,
        gm_redi=gm_cfg,
    )
    state = eu_ic("latlon_channel", grid, z_coord, eu)
    return grid, z_coord, model, state, gm_cfg, eu


def _eos_lin(eu: EadyUniformConfig) -> LinearEOSConfig:
    return LinearEOSConfig(
        alpha_T=eu.alpha_T, rho_ref=eu.rho_0,
        T_ref=eu.T_ref, S_ref=eu.S_uniform,
    )


# ---------------------------------------------------------------------------
# (1) At-t=0 Redi tendency from the production orchestrator
# ---------------------------------------------------------------------------

def check_one_t0_tendency():
    print("=" * 78)
    print("  (1)  At-t=0 Redi tendency from production orchestrator")
    print("=" * 78)

    grid, z_coord, _, state, _, eu = _build_eady("triads", 5.0e4)
    eos = _eos_lin(eu)

    print(f"\n  Grid: {grid.n_lat} x {grid.n_lon}, nlev={z_coord.n_levels}")
    print(f"  Mask sum / total = {float(state.land_mask.data.sum())}/"
          f"{state.land_mask.data.size}  "
          f"(walls at i=0 and i={grid.n_lat - 1})")
    print(f"  T range: [{float(state.T.data.min()):.4f}, "
          f"{float(state.T.data.max()):.4f}],  "
          f"|S - S_ref| max: {float(jnp.max(jnp.abs(state.S.data - eu.S_uniform))):.2e}")

    T_scale = float(jnp.max(jnp.abs(state.T.data)))

    print()
    for sch, k in [("triads",   5.0e4),
                   ("centered", 5.0e4),
                   ("triads",   1.0e3),
                   ("triads",   0.0)]:
        cfg = GMRediConfig(
            kappa_GM=0.0, kappa_Redi=k, S_max=0.01, slope_scheme=sch,
        )
        dT, dS = gm_redi_tracer_tendency_latlon(
            state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
            grid, z_coord, cfg,
            eos="linear", eos_linear=eos,
            mask=state.land_mask.data,
            u_mask=state.u_mask.data, v_mask=state.v_mask.data,
        )
        max_dT = float(jnp.max(jnp.abs(dT)))
        max_dS = float(jnp.max(jnp.abs(dS)))
        per_kappa = max_dT / max(k, 1.0)
        print(f"   {sch:8s}  κ_R={k:>7.0e}   "
              f"max|dT|={max_dT:.4e} K/s   "
              f"per-κ={per_kappa:.4e}   "
              f"max|dS|={max_dS:.2e}   "
              f"rel(T)={max_dT/max(T_scale,1e-30):.2e}")

    print(
        "\n  Reading: per-κ value should be at the float64 round-off scale\n"
        "  (~1e-15..1e-19) for triads.  If it is, the 1.14 K integration\n"
        "  drift can NOT be the per-step Redi residual."
    )


# ---------------------------------------------------------------------------
# (2) Boundary inspection — fluxes, not tendencies
# ---------------------------------------------------------------------------

def check_two_boundary_fluxes():
    """Reproduce the triad fluxes inline so we can print F_x, F_y, F_z."""
    print("\n" + "=" * 78)
    print("  (2)  Per-face Redi flux at boundaries (triads, κ_R = 5e4)")
    print("=" * 78)

    grid, z_coord, _, state, _, eu = _build_eady("triads", 5.0e4)
    eos = _eos_lin(eu)

    # Build ρ exactly the way the orchestrator does.
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        iterate_eos_and_pressure_anomaly,
    )
    from legoesm.ocean.eos import make_eos_fn, rho_0 as _RHO_0
    from legoesm import constants

    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    eta = state.eta.data
    H_bathy = state.H_bathy.data
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)

    eos_fn = make_eos_fn("linear", eos)
    fill_fn = lambda f: _neumann_fill_cgrid(f, mask)
    rho, _, _ = iterate_eos_and_pressure_anomaly(
        state.T.data, state.S.data, mask, fill_fn, eos_fn,
        z_coord.dz_ref, _RHO_0, constants.g, n_iter=2,
    )

    # Rebuild the triad fluxes inline.  Same algebra as the function.
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        _to_uface_west, _to_uface_east, _to_vface_south, _to_vface_north,
        _triad_taper, _EPS_DIV,
    )
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        _EPS, vertical_flux_divergence,
    )

    n_lat, n_lon, nlev = state.T.data.shape
    dz_actual = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]
    S_max = 0.01
    kappa_Redi = 5.0e4
    kappa_GM_b = 0.0

    rho_filled = _neumann_fill_cgrid(rho, mask)
    q_filled = _neumann_fill_cgrid(state.T.data, mask)

    drho_dx_u = gradient_x_cgrid(rho_filled, grid)
    drho_dy_v = gradient_y_cgrid(rho_filled, grid)
    dq_dx_u = gradient_x_cgrid(q_filled, grid)
    dq_dy_v = gradient_y_cgrid(q_filled, grid)

    drho_dz_raw = (rho_filled[:, :, :-1] - rho_filled[:, :, 1:]) / jnp.maximum(dz_half, _EPS_DIV)
    drho_dz_w = jnp.minimum(drho_dz_raw, -_EPS_DIV)
    dq_dz_w = (q_filled[:, :, :-1] - q_filled[:, :, 1:]) / jnp.maximum(dz_half, _EPS_DIV)

    # ----- u-face fluxes ----------------------------------------------------
    sentinel_rho = jnp.full((n_lat, n_lon, 1), -_EPS_DIV, dtype=drho_dz_w.dtype)
    drho_dz_below = jnp.concatenate([drho_dz_w, sentinel_rho], axis=-1)
    drho_dz_above = jnp.concatenate([sentinel_rho, drho_dz_w], axis=-1)
    sentinel_q = jnp.zeros((n_lat, n_lon, 1), dtype=dq_dz_w.dtype)
    dq_dz_below = jnp.concatenate([dq_dz_w, sentinel_q], axis=-1)
    dq_dz_above = jnp.concatenate([sentinel_q, dq_dz_w], axis=-1)

    valid_below = jnp.concatenate([
        jnp.ones((n_lat, n_lon, nlev - 1), dtype=drho_dz_w.dtype),
        jnp.zeros((n_lat, n_lon, 1), dtype=drho_dz_w.dtype),
    ], axis=-1)
    valid_above = jnp.concatenate([
        jnp.zeros((n_lat, n_lon, 1), dtype=drho_dz_w.dtype),
        jnp.ones((n_lat, n_lon, nlev - 1), dtype=drho_dz_w.dtype),
    ], axis=-1)

    def _t(S):
        return S * _triad_taper(S, S_max)

    # u-face triads
    drho_dz_T1 = _to_uface_west(drho_dz_below)
    drho_dz_T2 = _to_uface_west(drho_dz_above)
    drho_dz_T3 = _to_uface_east(drho_dz_below)
    drho_dz_T4 = _to_uface_east(drho_dz_above)
    dq_dz_T1 = _to_uface_west(dq_dz_below)
    dq_dz_T2 = _to_uface_west(dq_dz_above)
    dq_dz_T3 = _to_uface_east(dq_dz_below)
    dq_dz_T4 = _to_uface_east(dq_dz_above)
    v_T1 = _to_uface_west(valid_below)
    v_T2 = _to_uface_west(valid_above)
    v_T3 = _to_uface_east(valid_below)
    v_T4 = _to_uface_east(valid_above)
    S_T1 = _t(jnp.clip(-drho_dx_u / drho_dz_T1, -S_max, S_max))
    S_T2 = _t(jnp.clip(-drho_dx_u / drho_dz_T2, -S_max, S_max))
    S_T3 = _t(jnp.clip(-drho_dx_u / drho_dz_T3, -S_max, S_max))
    S_T4 = _t(jnp.clip(-drho_dx_u / drho_dz_T4, -S_max, S_max))
    Nu = jnp.maximum(v_T1 + v_T2 + v_T3 + v_T4, 1.0)
    S_dq_dz_uface = (v_T1 / Nu * S_T1 * dq_dz_T1
                   + v_T2 / Nu * S_T2 * dq_dz_T2
                   + v_T3 / Nu * S_T3 * dq_dz_T3
                   + v_T4 / Nu * S_T4 * dq_dz_T4)
    F_x_u = (kappa_Redi * dq_dx_u
             + (kappa_Redi - kappa_GM_b) * S_dq_dz_uface) * u_mask[:, :, jnp.newaxis]

    # v-face triads
    drho_dz_V1 = _to_vface_south(drho_dz_below)
    drho_dz_V2 = _to_vface_south(drho_dz_above)
    drho_dz_V3 = _to_vface_north(drho_dz_below)
    drho_dz_V4 = _to_vface_north(drho_dz_above)
    dq_dz_V1 = _to_vface_south(dq_dz_below)
    dq_dz_V2 = _to_vface_south(dq_dz_above)
    dq_dz_V3 = _to_vface_north(dq_dz_below)
    dq_dz_V4 = _to_vface_north(dq_dz_above)
    v_V1 = _to_vface_south(valid_below)
    v_V2 = _to_vface_south(valid_above)
    v_V3 = _to_vface_north(valid_below)
    v_V4 = _to_vface_north(valid_above)
    S_V1 = _t(jnp.clip(-drho_dy_v / drho_dz_V1, -S_max, S_max))
    S_V2 = _t(jnp.clip(-drho_dy_v / drho_dz_V2, -S_max, S_max))
    S_V3 = _t(jnp.clip(-drho_dy_v / drho_dz_V3, -S_max, S_max))
    S_V4 = _t(jnp.clip(-drho_dy_v / drho_dz_V4, -S_max, S_max))
    Nv = jnp.maximum(v_V1 + v_V2 + v_V3 + v_V4, 1.0)
    S_dq_dz_vface = (v_V1 / Nv * S_V1 * dq_dz_V1
                   + v_V2 / Nv * S_V2 * dq_dz_V2
                   + v_V3 / Nv * S_V3 * dq_dz_V3
                   + v_V4 / Nv * S_V4 * dq_dz_V4)
    F_y_v = (kappa_Redi * dq_dy_v
             + (kappa_Redi - kappa_GM_b) * S_dq_dz_vface) * v_mask[:, :, jnp.newaxis]

    # w-face flux (x and y triads).  Skip GM contribution (kappa_GM = 0).
    drho_dx_west = drho_dx_u[:, :n_lon, :]
    drho_dx_east = drho_dx_u[:, 1:n_lon + 1, :]
    dq_dx_west = dq_dx_u[:, :n_lon, :]
    dq_dx_east = dq_dx_u[:, 1:n_lon + 1, :]
    drho_dy_south = drho_dy_v[:n_lat, :, :]
    drho_dy_north = drho_dy_v[1:n_lat + 1, :, :]
    dq_dy_south = dq_dy_v[:n_lat, :, :]
    dq_dy_north = dq_dy_v[1:n_lat + 1, :, :]

    def _AB(arr):
        return arr[:, :, :-1], arr[:, :, 1:]

    drho_dx_west_A, drho_dx_west_B = _AB(drho_dx_west)
    drho_dx_east_A, drho_dx_east_B = _AB(drho_dx_east)
    drho_dy_south_A, drho_dy_south_B = _AB(drho_dy_south)
    drho_dy_north_A, drho_dy_north_B = _AB(drho_dy_north)
    dq_dx_west_A, dq_dx_west_B = _AB(dq_dx_west)
    dq_dx_east_A, dq_dx_east_B = _AB(dq_dx_east)
    dq_dy_south_A, dq_dy_south_B = _AB(dq_dy_south)
    dq_dy_north_A, dq_dy_north_B = _AB(dq_dy_north)

    S_Wx1 = _t(jnp.clip(-drho_dx_west_A / drho_dz_w, -S_max, S_max))
    S_Wx2 = _t(jnp.clip(-drho_dx_east_A / drho_dz_w, -S_max, S_max))
    S_Wx3 = _t(jnp.clip(-drho_dx_west_B / drho_dz_w, -S_max, S_max))
    S_Wx4 = _t(jnp.clip(-drho_dx_east_B / drho_dz_w, -S_max, S_max))
    S_Wy1 = _t(jnp.clip(-drho_dy_south_A / drho_dz_w, -S_max, S_max))
    S_Wy2 = _t(jnp.clip(-drho_dy_north_A / drho_dz_w, -S_max, S_max))
    S_Wy3 = _t(jnp.clip(-drho_dy_south_B / drho_dz_w, -S_max, S_max))
    S_Wy4 = _t(jnp.clip(-drho_dy_north_B / drho_dz_w, -S_max, S_max))

    cross_x = 0.25 * (S_Wx1 * dq_dx_west_A + S_Wx2 * dq_dx_east_A
                     + S_Wx3 * dq_dx_west_B + S_Wx4 * dq_dx_east_B)
    cross_y = 0.25 * (S_Wy1 * dq_dy_south_A + S_Wy2 * dq_dy_north_A
                     + S_Wy3 * dq_dy_south_B + S_Wy4 * dq_dy_north_B)
    Sx2 = 0.25 * (S_Wx1**2 + S_Wx2**2 + S_Wx3**2 + S_Wx4**2)
    Sy2 = 0.25 * (S_Wy1**2 + S_Wy2**2 + S_Wy3**2 + S_Wy4**2)
    F_z = (kappa_Redi * (cross_x + cross_y)
           + kappa_Redi * (Sx2 + Sy2) * dq_dz_w)

    # ----- Print -----
    print(f"\n  v-face index 0 (south wall):           v_mask[0]={float(v_mask[0,0]):.0f}, "
          f"max|F_y[0,:,:]|={float(jnp.max(jnp.abs(F_y_v[0]))):.4e}")
    print(f"  v-face index 1 (wall-adjacent):        v_mask[1]={float(v_mask[1,0]):.0f}, "
          f"max|F_y[1,:,:]|={float(jnp.max(jnp.abs(F_y_v[1]))):.4e}")
    print(f"  v-face index 2 (first interior):       v_mask[2]={float(v_mask[2,0]):.0f}, "
          f"max|F_y[2,:,:]|={float(jnp.max(jnp.abs(F_y_v[2]))):.4e}")
    print(f"  v-face mid-domain (i={n_lat//2}):                        "
          f"max|F_y[{n_lat//2},:,:]|={float(jnp.max(jnp.abs(F_y_v[n_lat//2]))):.4e}")
    print(f"  v-face index {n_lat - 1} (wall-adjacent N):    v_mask[{n_lat-1}]={float(v_mask[n_lat-1,0]):.0f}, "
          f"max|F_y[{n_lat - 1},:,:]|={float(jnp.max(jnp.abs(F_y_v[n_lat - 1]))):.4e}")
    print(f"  v-face index {n_lat} (north wall):             v_mask[{n_lat}]={float(v_mask[n_lat,0]):.0f}, "
          f"max|F_y[{n_lat},:,:]|={float(jnp.max(jnp.abs(F_y_v[n_lat]))):.4e}")

    print(f"\n  Surface (k=0)            max|F_z[:,:,0]|     = {float(jnp.max(jnp.abs(F_z[:,:,0]))):.4e}")
    print(f"  Mid-column (k={(nlev-1)//2})        max|F_z[:,:,{(nlev-1)//2}]| = {float(jnp.max(jnp.abs(F_z[:,:,(nlev-1)//2]))):.4e}")
    print(f"  Bottom (k={nlev-2})           max|F_z[:,:,{nlev-2}]|     = {float(jnp.max(jnp.abs(F_z[:,:,nlev-2]))):.4e}")

    # And the masked tendency
    dq_h = divergence_cgrid(F_x_u, F_y_v, grid)
    dq_v = vertical_flux_divergence(F_z, dz_actual, _EPS)
    tend = (dq_h + dq_v) * mask[:, :, jnp.newaxis]
    print(f"\n  max|dq_h| (horiz divergence) = {float(jnp.max(jnp.abs(dq_h))):.4e}")
    print(f"  max|dq_v| (vert  divergence) = {float(jnp.max(jnp.abs(dq_v))):.4e}")
    print(f"  max|tendency|                = {float(jnp.max(jnp.abs(tend))):.4e}")

    # Top-5 cells where the tendency is largest — gives a sense of where
    # the residual *actually* lives.
    abs_t = jnp.abs(tend)
    flat = abs_t.flatten()
    idx = jnp.argsort(flat)[-5:]
    print(f"\n  Top 5 cells by |tendency|:")
    for k_idx in idx:
        i, j, k = jnp.unravel_index(k_idx, abs_t.shape)
        print(f"    (i={int(i):3d}, j={int(j):3d}, k={int(k):3d})  "
              f"tend={float(tend[i,j,k]):.4e}  "
              f"S_x_face={float(S_T1[i,j,k]):.2e} (T1) "
              f"{float(S_T3[i,j,k]):.2e} (T3)")


# ---------------------------------------------------------------------------
# (3)  GM/Redi=None vs GMRediConfig(κ=0, triads):  is the path itself inert?
# ---------------------------------------------------------------------------

def check_three_path_off_vs_kappa_zero():
    print("\n" + "=" * 78)
    print("  (3)  Path OFF vs path-on-with-zero-κ — single model.step")
    print("=" * 78)

    grid, z_coord, model_off, state, _, eu = _build_eady(
        slope_scheme="triads", kappa_Redi=0.0, gm_off=True,
    )
    grid2, z2, model_on, state2, _, _ = _build_eady(
        slope_scheme="triads", kappa_Redi=0.0, gm_off=False,
    )

    # Single physics+dycore step.
    dt = 300.0
    s_off = model_off.step(state, dt)
    s_on = model_on.step(state2, dt)

    for name, a, b in [
        ("T",   s_off.T.data, s_on.T.data),
        ("S",   s_off.S.data, s_on.S.data),
        ("u",   s_off.u.data, s_on.u.data),
        ("v",   s_off.v.data, s_on.v.data),
        ("eta", s_off.eta.data, s_on.eta.data),
    ]:
        d = float(jnp.max(jnp.abs(a - b)))
        scale = float(jnp.max(jnp.abs(a)) + 1e-30)
        print(f"  max|Δ{name}| after 1 step: {d:.4e}  (relative {d/scale:.2e})")

    print(
        "\n  Reading: if any |Δ| > round-off, the GM/Redi orchestrator path\n"
        "  itself is not bit-equivalent to no-path even at κ=0 — meaning a\n"
        "  fraction of the 1.14 K is from path ordering / EOS recomputation,\n"
        "  not from κ_Redi · (cancelling tensor)."
    )


def main():
    check_one_t0_tendency()
    check_two_boundary_fluxes()
    check_three_path_off_vs_kappa_zero()


if __name__ == "__main__":
    main()
