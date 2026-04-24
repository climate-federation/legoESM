"""Iter-795 diagnostic: test whether COMPONENT-CONSISTENT halo
for B reduces W2 mode-A at cube vertices.

Iter-793 showed W2 mode-A is INCOMPLETE cancellation of Cor+press
+KE gradients at cube vertices.  Hypothesis: because our current
code halos B = KE + g*(h + h_s) as a SCALAR (via `pad_halo(B)` in
`_arakawa_lamb_gradient`), the cube-vertex halo cells of B get a
2-pt scalar average that is INCONSISTENT with halo'd (u_cc, v_cc)
(used for Coriolis via zeta).

Hypothesis test: construct B at cube-vertex halo cells from halo'd
components:
  u_halo, v_halo = pad_halo_vector(u_cc, v_cc, ...)
  h_halo = pad_halo(h, ...)
  h_s_halo = pad_halo(h_s, ...)
  B_halo[cube-vertex cells] = 0.5*(u_halo² + v_halo²)
                              + g*(h_halo + h_s_halo)
Keep non-cube-vertex halo cells as the regular pad_halo(B).

If W2 v_ll_Linf drops, the hypothesis is confirmed and
`_arakawa_lamb_gradient` should use component-consistent halo by
default.  If unchanged or worse, the hypothesis is refuted.

Scope: diagnostic only.  A positive result would motivate a
Fortran-faithful source-code change to `_arakawa_lamb_gradient` or
its upstream B construction.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from legoesm.core import operators_cdgrid as ocd
from legoesm.core.operators_cdgrid import (
    _arakawa_lamb_gradient as orig_al_grad,
    fv3_d2cc)
from legoesm.grids.halo import pad_halo, pad_halo_vector
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _construct_consistent_B_halo(h, h_s, u_cc, v_cc, cdgrid, g):
    """Build B=KE+g*(h+h_s) with cube-vertex halo cells derived from
    component halos rather than scalar halo of B.

    Returns
    -------
    B_pad : (6, n+2, n+2) — padded B with 4 cube-vertex halo cells
        replaced by component-consistent values.
    """
    grid = cdgrid.base
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets

    # Vector halo for u_cc, v_cc.
    u_pad, v_pad = pad_halo_vector(
        u_cc, v_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    # Scalar halo for h, h_s.
    h_pad = pad_halo(h, halo=1, interp_offsets=offsets, duogrid=dg)
    h_s_pad = pad_halo(h_s, halo=1, interp_offsets=offsets, duogrid=dg)

    # Regular scalar halo for B.
    B = 0.5 * (u_cc ** 2 + v_cc ** 2) + g * (h + h_s)
    B_pad = pad_halo(B, halo=1, interp_offsets=offsets, duogrid=dg)

    # Overwrite ONLY the 4 cube-vertex halo cells per face with
    # component-derived B values.  Padded indices (halo=1, shape
    # (6, n+2, n+2)):
    #   SW corner: [0, 0]
    #   SE corner: [n+1, 0]
    #   NW corner: [0, n+1]
    #   NE corner: [n+1, n+1]
    #
    # The component halos at these cells give (u_pad_corner,
    # v_pad_corner, h_pad_corner, h_s_pad_corner).  Compute
    # B_corner = 0.5*(u² + v²) + g*(h + h_s).
    corner_indices = [(0, 0), (-1, 0), (0, -1), (-1, -1)]
    B_pad_new = B_pad
    for (i, j) in corner_indices:
        u_at = u_pad[:, i, j]
        v_at = v_pad[:, i, j]
        h_at = h_pad[:, i, j]
        hs_at = h_s_pad[:, i, j]
        B_at = 0.5 * (u_at ** 2 + v_at ** 2) + g * (h_at + hs_at)
        B_pad_new = B_pad_new.at[:, i, j].set(B_at)

    return B_pad_new


def _patched_al_grad(B, cdgrid, padded=None,
                     fortran_dir_aware_corners=False,
                     fortran_a2b_corner_avg=False,
                     _B_pad_consistent=None):
    """A-L gradient with optional consistent B-halo."""
    if _B_pad_consistent is not None:
        padded = _B_pad_consistent
    return orig_al_grad(
        B, cdgrid, padded=padded,
        fortran_dir_aware_corners=fortran_dir_aware_corners,
        fortran_a2b_corner_avg=fortran_a2b_corner_avg)


def _run_w2(n, use_consistent_B_halo, hours=24.0, dt=300.0):
    n_steps = int(round(hours * 3600 / dt))
    div_damp = 8.0 * _div_damp_cube(n)
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=div_damp,
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    orig_fn = ocd._arakawa_lamb_gradient
    if use_consistent_B_halo:
        g_val = cfg.g

        def _wrapped(B, cdgrid_arg, padded=None,
                     fortran_dir_aware_corners=False,
                     fortran_a2b_corner_avg=False):
            if padded is None:
                # Build component-consistent B halo.  The u_cc, v_cc, h, h_s
                # used must be reconstructable from cdgrid/state context;
                # use the fields threaded via `build_pad` closure.
                # Since this function runs INSIDE JIT on each step, we use
                # a heuristic: if B has cell-centre shape (6, n, n) and
                # matches the expected cell-centre shape, look up u_cc etc.
                # from closure.  Rather than doing complex introspection,
                # use the simple approach: override ONLY the primary B
                # (= KE + g*(h+h_s)) call, and leave div_damp's
                # _arakawa_lamb_gradient(div_field, ...) call untouched.
                #
                # The primary B call has a specific float32-B-dtype and is
                # called first.  Attempt to identify by shape.  Actually,
                # both B and div_field have shape (6, n, n), so we can't
                # distinguish.  Fall back to: compute B from components
                # given cdgrid (which has u_d/v_d indirectly).  But we
                # don't have u_d/v_d at this call site.
                #
                # Simpler alternative: use orig_fn default.
                pass
            return orig_fn(
                B, cdgrid_arg, padded=padded,
                fortran_dir_aware_corners=fortran_dir_aware_corners,
                fortran_a2b_corner_avg=fortran_a2b_corner_avg)

        # Monkey-patch, but actually the right approach is to intervene
        # at the call site in `fv3_sw_tendencies` rather than here.
        # Since the call site needs u_cc, v_cc, h, h_s context, the
        # cleanest diagnostic is to patch `fv3_sw_tendencies` itself
        # — which is beyond a short diagnostic.  Fall back to:
        # measure t=0 tendency with consistent B halo, NOT the full
        # 1-day run.
        print("  (note: monkey-patching _arakawa_lamb_gradient alone")
        print("   cannot distinguish B vs div_field at the call site")
        print("   because both are (6, n, n) scalars.  Falling back to")
        print("   t=0 tendency diagnostic.)")

    h0 = state.h
    for _ in range(n_steps):
        state = model.step(state, dt)

    h_mean = float(jnp.mean(jnp.abs(h0)))
    err = state.h - h0
    L2 = float(jnp.sqrt(jnp.mean(err ** 2)) / h_mean)

    from legoesm.grids.cubed_sphere_cdgrid import (
        cell_centre_angles_from_4edge)
    from legoesm.grids.regridding import (
        get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    return {
        'L2': L2,
        'v_ll_linf': float(np.max(np.abs(v_ll))),
    }


# Instead of the tricky monkey-patch, run a t=0 tendency diagnostic.
n = 36
grid = create_cubed_sphere(n=n, use_duogrid=False)
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0,
    div_damp=8.0 * _div_damp_cube(n),
    boundary_fix=True,
    damp_v=0.06,
    nord_v=2)
model = FV3EdgeShallowWaterModel(grid, config=cfg)
cdgrid = model.cdgrid

sw = williamson_test2(grid)
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
h = sw.h.data
h_s = sw.h_s.data
g = cfg.g

u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
B = 0.5 * (u_cc ** 2 + v_cc ** 2) + g * (h + h_s)

# Regular gradient via default halo.
dB_dx_def, dB_dy_def = orig_al_grad(B, cdgrid)

# Component-consistent B halo.
B_pad_cons = _construct_consistent_B_halo(h, h_s, u_cc, v_cc, cdgrid, g)
dB_dx_cons, dB_dy_cons = orig_al_grad(B, cdgrid, padded=B_pad_cons)

# Interpolate to cell centres.
from legoesm.core.operators_cdgrid import _interp_corner_to_center
dB_dy_cc_def = _interp_corner_to_center(dB_dy_def)
dB_dy_cc_cons = _interp_corner_to_center(dB_dy_cons)

diff = np.asarray(dB_dy_cc_cons - dB_dy_cc_def)
print(f"Iter-795 test: component-consistent B halo vs scalar halo")
print(f"C36 W2 IC, β=0")
print()
print(f"max |dB_dy_cc_cons - dB_dy_cc_def| = {float(np.max(np.abs(diff))):.3e} m/s²")
print(f"mean |diff| = {float(np.mean(np.abs(diff))):.3e} m/s²")
lat_cc = np.rad2deg(np.asarray(grid.lat))
lon_cc = np.rad2deg(np.asarray(grid.lon))
lon_cc = np.where(lon_cc > 180.0, lon_cc - 360.0, lon_cc)

idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
print(f"peak diff at ({face}, {ci}, {cj}) lat={lat_cc[face, ci, cj]:+.2f}° "
      f"lon={lon_cc[face, ci, cj]:+.2f}°")

# Now compute the full dv_cc residual under both settings.
from legoesm.core.operators_cdgrid import (
    cgrid_divergence, cgrid_mass_flux_divergence, dgrid_vorticity, fv3_cc2c)

u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)
u_cc_pad, v_cc_pad = pad_halo_vector(
    u_cc, v_cc,
    grid.cos_angle, grid.sin_angle,
    grid.cos_angle_padded, grid.sin_angle_padded,
    interp_offsets=grid.halo_interp_offsets, duogrid=None,
)
u_corner = 0.25 * (u_cc_pad[:, :-1, :-1] + u_cc_pad[:, 1:, :-1]
                    + u_cc_pad[:, :-1, 1:] + u_cc_pad[:, 1:, 1:])
v_corner = 0.25 * (v_cc_pad[:, :-1, :-1] + v_cc_pad[:, 1:, :-1]
                    + v_cc_pad[:, :-1, 1:] + v_cc_pad[:, 1:, 1:])
zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)
zeta_abs = zeta + cdgrid.base.f

dv_default = -zeta_abs * u_cc - dB_dy_cc_def
dv_consistent = -zeta_abs * u_cc - dB_dy_cc_cons

peak_d = float(np.max(np.abs(np.asarray(dv_default))))
peak_c = float(np.max(np.abs(np.asarray(dv_consistent))))

print()
print(f"dv_cc residual (WITHOUT div_damp):")
print(f"  default halo:    max |dv| = {peak_d:.3e}")
print(f"  consistent halo: max |dv| = {peak_c:.3e}")
print(f"  ratio (cons/def): {peak_c/peak_d:.3f}")

# Location of each.
for label, dv in (('default', np.asarray(dv_default)),
                   ('consistent', np.asarray(dv_consistent))):
    idx = np.unravel_index(np.argmax(np.abs(dv)), dv.shape)
    face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
    peak = float(np.abs(dv)[face, ci, cj])
    latd = float(lat_cc[face, ci, cj])
    lond = float(lon_cc[face, ci, cj])
    CUBE_VERTEX_LAT = np.arcsin(1.0 / np.sqrt(3.0))
    # quick GC
    from math import radians, asin, sin, cos, sqrt
    vlat = np.sign(latd) * np.rad2deg(CUBE_VERTEX_LAT)
    dlat = radians(latd - vlat)
    dlon = radians((lond % 360) - 45) if abs((lond % 360) - 45) < 90 else radians((lond % 360) - 135)
    # simpler: compute GC to (35.26°, 45°) approximately
    print(f"  {label:>12}: peak={peak:.3e} ({face},{ci},{cj}) lat={latd:+.2f}° lon={lond:+.2f}°")

print()
print("Interpretation cues (observational only):")
print("- If consistent B halo gives a SMALLER dv_cc residual:")
print("  halo inconsistency IS a mechanism for W2 mode-A.  Implement")
print("  `pad_halo_components` for B in fv3_sw_tendencies.")
print("- If consistent B halo gives LARGER or EQUAL dv_cc residual:")
print("  halo inconsistency is not the dominant mechanism.")

print()
print("What iter-795 DOES measure (observational only):")
print("- dv_cc residual at t=0 under default vs component-consistent")
print("  B halo.")
print("What it does NOT establish:")
print("- 1-day v_ll_Linf impact (would need intrusive source-code")
print("  change to fv3_sw_tendencies; deferred to future iter).")
print("- Whether the same change helps W5 or ocean rest state.")
