"""Iter-844 diagnostic: t=0 W2 LEGACY dv/dt decomposition on the A-L path.

Directly instruments the live `fv3_sw_tendencies()` algebra on the
current LEGACY production path (C36, use_duogrid=False) and decomposes
the returned D-grid `dv_d_dt` into:

  - planetary Coriolis        (-f * u)
  - relative-vorticity transport (-zeta * u)
  - pressure-gradient         (-grad_y g(h+h_s))
  - KE gradient               (-grad_y KE)
  - divergence damping
  - hyperdiffusion
  - boundary-fix increment

Each contribution is projected through the SAME vector halo/projection
path as the production `dv_d_dt` so the sum matches the actual returned
field to roundoff.
"""

import os
import sys

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (  # noqa: E402
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
)
from legoesm.core.operators import laplacian_compact  # noqa: E402
from legoesm.core.operators_cdgrid import (  # noqa: E402
    _arakawa_lamb_gradient,
    _fortran_agrid_vector_corner_fill,
    _interp_corner_to_center,
    cgrid_divergence,
    dgrid_vorticity,
    fv3_cc2c,
    fv3_d2cc,
    fv3_sw_tendencies,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere  # noqa: E402
from legoesm.grids.halo import pad_halo_vector  # noqa: E402
from tests.atmosphere.shallow_water.test_cases.williamson import (  # noqa: E402
    williamson_test2,
)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


CUBE_VERTEX_LAT_RAD = np.arcsin(1.0 / np.sqrt(3.0))
CUBE_VERTEX_LATLON_DEG = [
    (+np.rad2deg(CUBE_VERTEX_LAT_RAD), +45.0),
    (+np.rad2deg(CUBE_VERTEX_LAT_RAD), +135.0),
    (+np.rad2deg(CUBE_VERTEX_LAT_RAD), -45.0),
    (+np.rad2deg(CUBE_VERTEX_LAT_RAD), -135.0),
    (-np.rad2deg(CUBE_VERTEX_LAT_RAD), +45.0),
    (-np.rad2deg(CUBE_VERTEX_LAT_RAD), +135.0),
    (-np.rad2deg(CUBE_VERTEX_LAT_RAD), -45.0),
    (-np.rad2deg(CUBE_VERTEX_LAT_RAD), -135.0),
]


def _gc_dist_deg(lat_deg, lon_deg, vlat_deg, vlon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    vlat = np.deg2rad(vlat_deg)
    vlon = np.deg2rad(vlon_deg)
    dlat = lat - vlat
    dlon = lon - vlon
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat) * np.cos(vlat) * np.sin(dlon / 2.0) ** 2)
    return float(np.rad2deg(2.0 * np.arcsin(np.sqrt(max(a, 0.0)))))


def _nearest_cube_vertex(lat_deg, lon_deg):
    best_gc = np.inf
    best_vertex = None
    for vlat_deg, vlon_deg in CUBE_VERTEX_LATLON_DEG:
        gc = _gc_dist_deg(lat_deg, lon_deg, vlat_deg, vlon_deg)
        if gc < best_gc:
            best_gc = gc
            best_vertex = (vlat_deg, vlon_deg)
    return best_gc, best_vertex


def _cube_vertex_mask(lat_deg, lon_deg, max_gc_deg):
    out = np.zeros_like(lat_deg, dtype=bool)
    flat = out.reshape(-1)
    lat_flat = np.asarray(lat_deg).reshape(-1)
    lon_flat = np.asarray(lon_deg).reshape(-1)
    for idx, (lat_i, lon_i) in enumerate(zip(lat_flat, lon_flat, strict=False)):
        gc, _ = _nearest_cube_vertex(float(lat_i), float(lon_i))
        flat[idx] = gc <= max_gc_deg
    return out


def _apply_boundary_fix(du_cc, dv_cc, n, skip_corners):
    if n <= 2:
        return du_cc, dv_cc
    if skip_corners:
        du_out = du_cc.at[:, 0, 1:-1].set(
            0.5 * (du_cc[:, 0, 1:-1] + du_cc[:, 1, 1:-1]))
        du_out = du_out.at[:, n - 1, 1:-1].set(
            0.5 * (du_out[:, n - 1, 1:-1] + du_out[:, n - 2, 1:-1]))
        du_out = du_out.at[:, 1:-1, 0].set(
            0.5 * (du_out[:, 1:-1, 0] + du_out[:, 1:-1, 1]))
        du_out = du_out.at[:, 1:-1, n - 1].set(
            0.5 * (du_out[:, 1:-1, n - 1] + du_out[:, 1:-1, n - 2]))

        dv_out = dv_cc.at[:, 0, 1:-1].set(
            0.5 * (dv_cc[:, 0, 1:-1] + dv_cc[:, 1, 1:-1]))
        dv_out = dv_out.at[:, n - 1, 1:-1].set(
            0.5 * (dv_out[:, n - 1, 1:-1] + dv_out[:, n - 2, 1:-1]))
        dv_out = dv_out.at[:, 1:-1, 0].set(
            0.5 * (dv_out[:, 1:-1, 0] + dv_out[:, 1:-1, 1]))
        dv_out = dv_out.at[:, 1:-1, n - 1].set(
            0.5 * (dv_out[:, 1:-1, n - 1] + dv_out[:, 1:-1, n - 2]))
        return du_out, dv_out

    du_out = du_cc.at[:, 0, :].set(0.5 * (du_cc[:, 0, :] + du_cc[:, 1, :]))
    du_out = du_out.at[:, n - 1, :].set(
        0.5 * (du_out[:, n - 1, :] + du_out[:, n - 2, :]))
    du_out = du_out.at[:, :, 0].set(0.5 * (du_out[:, :, 0] + du_out[:, :, 1]))
    du_out = du_out.at[:, :, n - 1].set(
        0.5 * (du_out[:, :, n - 1] + du_out[:, :, n - 2]))

    dv_out = dv_cc.at[:, 0, :].set(0.5 * (dv_cc[:, 0, :] + dv_cc[:, 1, :]))
    dv_out = dv_out.at[:, n - 1, :].set(
        0.5 * (dv_out[:, n - 1, :] + dv_out[:, n - 2, :]))
    dv_out = dv_out.at[:, :, 0].set(0.5 * (dv_out[:, :, 0] + dv_out[:, :, 1]))
    dv_out = dv_out.at[:, :, n - 1].set(
        0.5 * (dv_out[:, :, n - 1] + dv_out[:, :, n - 2]))
    return du_out, dv_out


def _project_cc_vector_to_dgrid(du_cc, dv_cc, grid, dg, offsets, do_fortran_vec_fill):
    du_cc_pad, dv_cc_pad = pad_halo_vector(
        du_cc, dv_cc,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    if do_fortran_vec_fill:
        du_cc_pad, dv_cc_pad = _fortran_agrid_vector_corner_fill(
            du_cc_pad, dv_cc_pad)
    du_d = 0.5 * (du_cc_pad[:, 1:-1, :-1] + du_cc_pad[:, 1:-1, 1:])
    dv_d = 0.5 * (dv_cc_pad[:, :-1, 1:-1] + dv_cc_pad[:, 1:, 1:-1])
    return du_d, dv_d


def _zero_like_pair(shape):
    return jnp.zeros(shape), jnp.zeros(shape)


def _peak_info(field, lat_deg, lon_deg):
    arr = np.asarray(field)
    abs_arr = np.abs(arr)
    idx = np.unravel_index(np.argmax(abs_arr), abs_arr.shape)
    face, ii, jj = (int(idx[0]), int(idx[1]), int(idx[2]))
    peak = float(abs_arr[face, ii, jj])
    signed_val = float(arr[face, ii, jj])
    lat = float(lat_deg[face, ii, jj])
    lon = float(lon_deg[face, ii, jj])
    gc, vertex = _nearest_cube_vertex(lat, lon)
    return {
        "face": face,
        "i": ii,
        "j": jj,
        "peak_abs": peak,
        "value": signed_val,
        "lat_deg": lat,
        "lon_deg": lon,
        "gc_deg": gc,
        "vertex_lat_deg": float(vertex[0]),
        "vertex_lon_deg": float(vertex[1]),
    }


def main():
    n = 36
    cube_vertex_gc_deg = 5.0

    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    h = sw.h.data
    h_s = sw.h_s.data

    dh_ref, du_ref, dv_ref = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid,
        g=cfg.g,
        div_damp=cfg.div_damp,
        hyperdiff_coeff=cfg.hyperdiff_coeff,
        boundary_fix=cfg.boundary_fix,
        boundary_fix_skip_corners=cfg.boundary_fix_skip_corners,
        fortran_a2b_corner_avg=cfg.fortran_a2b_corner_avg,
        fortran_vector_corner_fill=cfg.fortran_vector_corner_fill,
    )

    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)

    geop = cfg.g * (h + h_s)
    ke = 0.5 * (u_cc ** 2 + v_cc ** 2)
    bernoulli = geop + ke

    grad_kwargs = dict(
        fortran_dir_aware_corners=False,
        fortran_a2b_corner_avg=cfg.fortran_a2b_corner_avg,
    )
    dbern_dx, dbern_dy_perp = _arakawa_lamb_gradient(bernoulli, cdgrid, **grad_kwargs)
    dke_dx, dke_dy_perp = _arakawa_lamb_gradient(ke, cdgrid, **grad_kwargs)
    dbern_dx_cc = _interp_corner_to_center(dbern_dx)
    dbern_dy_cc = _interp_corner_to_center(dbern_dy_perp)
    dke_dx_cc = _interp_corner_to_center(dke_dx)
    dke_dy_cc = _interp_corner_to_center(dke_dy_perp)

    base = cdgrid.base
    dg = base.duogrid
    offsets = None if dg is not None else base.halo_interp_offsets
    u_cc_pad, v_cc_pad = pad_halo_vector(
        u_cc, v_cc,
        base.cos_angle, base.sin_angle,
        base.cos_angle_padded, base.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg,
    )
    if cfg.fortran_vector_corner_fill:
        u_cc_pad, v_cc_pad = _fortran_agrid_vector_corner_fill(u_cc_pad, v_cc_pad)
    u_corner = 0.25 * (u_cc_pad[:, :-1, :-1] + u_cc_pad[:, 1:, :-1]
                       + u_cc_pad[:, :-1, 1:] + u_cc_pad[:, 1:, 1:])
    v_corner = 0.25 * (v_cc_pad[:, :-1, :-1] + v_cc_pad[:, 1:, :-1]
                       + v_cc_pad[:, :-1, 1:] + v_cc_pad[:, 1:, 1:])
    zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)

    f_cc = cdgrid.base.f
    planetary_du_cc = f_cc * v_cc
    planetary_dv_cc = -f_cc * u_cc
    rel_vort_du_cc = zeta * v_cc
    rel_vort_dv_cc = -zeta * u_cc
    # Define pressure as the exact residual of the production Bernoulli
    # gradient after removing KE so pressure + KE reproduces `-grad(B)`
    # to roundoff.
    pressure_du_cc = -(dbern_dx_cc - dke_dx_cc)
    pressure_dv_cc = -(dbern_dy_cc - dke_dy_cc)
    ke_du_cc = -dke_dx_cc
    ke_dv_cc = -dke_dy_cc

    if cfg.div_damp > 0.0:
        div_field = cgrid_divergence(u_c, v_c, cdgrid)
        da_min_c = jnp.min(cdgrid.area_corner)
        d2_bg = cfg.div_damp / da_min_c
        dddmp = 0.2
        adaptive_coeff = da_min_c * jnp.maximum(
            d2_bg, jnp.minimum(0.20, dddmp * jnp.abs(div_field)))
        ddiv_dx, ddiv_dy_perp = _arakawa_lamb_gradient(
            div_field, cdgrid, **grad_kwargs)
        div_du_cc = adaptive_coeff * _interp_corner_to_center(ddiv_dx)
        div_dv_cc = adaptive_coeff * _interp_corner_to_center(ddiv_dy_perp)
    else:
        div_du_cc, div_dv_cc = _zero_like_pair(u_cc.shape)

    if cfg.hyperdiff_coeff > 0.0:
        cos_a = jnp.cos(cdgrid.base.angle)
        sin_a = jnp.sin(cdgrid.base.angle)
        ue_cc = cos_a * u_cc - sin_a * v_cc
        vn_cc = sin_a * u_cc + cos_a * v_cc
        lap_ue = laplacian_compact(ue_cc, cdgrid.base)
        bilap_ue = laplacian_compact(lap_ue, cdgrid.base)
        lap_vn = laplacian_compact(vn_cc, cdgrid.base)
        bilap_vn = laplacian_compact(lap_vn, cdgrid.base)
        bilap_u_local = cos_a * bilap_ue + sin_a * bilap_vn
        bilap_v_local = -sin_a * bilap_ue + cos_a * bilap_vn
        hyper_du_cc = -cfg.hyperdiff_coeff * bilap_u_local
        hyper_dv_cc = -cfg.hyperdiff_coeff * bilap_v_local
    else:
        hyper_du_cc, hyper_dv_cc = _zero_like_pair(u_cc.shape)

    du_pre = (
        planetary_du_cc + rel_vort_du_cc + pressure_du_cc
        + ke_du_cc + div_du_cc + hyper_du_cc
    )
    dv_pre = (
        planetary_dv_cc + rel_vort_dv_cc + pressure_dv_cc
        + ke_dv_cc + div_dv_cc + hyper_dv_cc
    )

    if cfg.boundary_fix:
        du_post, dv_post = _apply_boundary_fix(
            du_pre, dv_pre, n, cfg.boundary_fix_skip_corners)
    else:
        du_post, dv_post = du_pre, dv_pre
    boundary_du_cc = du_post - du_pre
    boundary_dv_cc = dv_post - dv_pre

    term_pairs_cc = {
        "planetary_coriolis": (planetary_du_cc, planetary_dv_cc),
        "relative_vort_transport": (rel_vort_du_cc, rel_vort_dv_cc),
        "pressure_gradient": (pressure_du_cc, pressure_dv_cc),
        "ke_gradient": (ke_du_cc, ke_dv_cc),
        "div_damp": (div_du_cc, div_dv_cc),
        "hyperdiff": (hyper_du_cc, hyper_dv_cc),
        "boundary_fix": (boundary_du_cc, boundary_dv_cc),
    }

    term_pairs_d = {}
    for name, (du_cc_term, dv_cc_term) in term_pairs_cc.items():
        term_pairs_d[name] = _project_cc_vector_to_dgrid(
            du_cc_term, dv_cc_term, base, dg, offsets,
            cfg.fortran_vector_corner_fill,
        )

    du_proj_total, dv_proj_total = _project_cc_vector_to_dgrid(
        du_post, dv_post, base, dg, offsets, cfg.fortran_vector_corner_fill)

    du_sum_terms = jnp.zeros_like(du_ref)
    dv_sum_terms = jnp.zeros_like(dv_ref)
    for du_term, dv_term in term_pairs_d.values():
        du_sum_terms = du_sum_terms + du_term
        dv_sum_terms = dv_sum_terms + dv_term

    lat_v = np.rad2deg(np.asarray(cdgrid.lat_edge_y))
    lon_v = np.rad2deg(np.asarray(cdgrid.lon_edge_y))
    lon_v = np.where(lon_v > 180.0, lon_v - 360.0, lon_v)
    vertex_mask = _cube_vertex_mask(lat_v, lon_v, cube_vertex_gc_deg)

    peak = _peak_info(dv_ref, lat_v, lon_v)
    peak_idx = (peak["face"], peak["i"], peak["j"])

    peak_values = {}
    peak_abs_sum = 0.0
    for name, (_, dv_term) in term_pairs_d.items():
        value = float(np.asarray(dv_term)[peak_idx])
        peak_values[name] = value
        peak_abs_sum += abs(value)

    term_vertex_linf = {}
    for name, (_, dv_term) in term_pairs_d.items():
        dv_np = np.asarray(dv_term)
        term_vertex_linf[name] = float(np.max(np.abs(dv_np[vertex_mask])))

    total_vertex_linf = float(np.max(np.abs(np.asarray(dv_ref)[vertex_mask])))
    residual_total = float(np.max(np.abs(np.asarray(dv_ref - dv_proj_total))))
    residual_terms = float(np.max(np.abs(np.asarray(dv_ref - dv_sum_terms))))

    print(f"Iter-844 A-L LEGACY W2 t=0 dv/dt decomposition at C{n}")
    print("Config mirrors scripts/run_atmosphere_test_matrix.py W2/W5 canonical block.")
    print(
        f"use_duogrid={grid.duogrid is not None}  "
        f"hyperdiff_coeff={cfg.hyperdiff_coeff:.6e}  "
        f"div_damp={cfg.div_damp:.6e}  "
        f"boundary_fix={cfg.boundary_fix}  "
        f"damp_v={cfg.damp_v:.3f}  nord_v={cfg.nord_v}"
    )
    print()
    print("Notes:")
    print("- `dv_d_dt` is measured from `fv3_sw_tendencies()` only; the post-step")
    print("  `damp_v` del-n vorticity damping in `FV3EdgeShallowWaterModel.step()`")
    print("  is excluded because it is applied after RK3, not inside one tendency call.")
    print("- Iter-808 sign-flip flux sync exists in the codebase but is inactive here:")
    print("  LEGACY uses `use_duogrid=False`, so `cgrid_mass_flux_divergence()`")
    print("  does not call `synchronize_cgrid_fluxes()` on this path.")
    print()
    print("Peak |dv/dt|:")
    print(f"  peak |dv/dt| = {peak['peak_abs']:.12e} m s^-2")
    print(f"  signed dv/dt = {peak['value']:.12e} m s^-2")
    print(f"  face={peak['face']}  (i,j)=({peak['i']},{peak['j']})")
    print(f"  lat={peak['lat_deg']:+.6f} deg  lon={peak['lon_deg']:+.6f} deg")
    print(
        f"  nearest cube vertex = ({peak['vertex_lat_deg']:+.6f}, "
        f"{peak['vertex_lon_deg']:+.6f}) deg"
    )
    print(f"  GC-to-vertex = {peak['gc_deg']:.6f} deg")
    print()
    print(
        f"Cube-vertex stencil mask: GC <= {cube_vertex_gc_deg:.1f} deg "
        f"at D-grid v points ({int(np.sum(vertex_mask))} points)"
    )
    print(f"  total dv/dt vertex-mask L_inf = {total_vertex_linf:.12e} m s^-2")
    print()
    print(
        f"{'term':>26}  {'at peak':>16}  {'abs share':>10}  "
        f"{'vertex L_inf':>16}"
    )
    print("-" * 78)
    for name in (
        "planetary_coriolis",
        "pressure_gradient",
        "relative_vort_transport",
        "ke_gradient",
        "div_damp",
        "hyperdiff",
        "boundary_fix",
    ):
        share = (abs(peak_values[name]) / peak_abs_sum) if peak_abs_sum > 0.0 else 0.0
        print(
            f"{name:>26}  {peak_values[name]:>16.12e}  "
            f"{share:>10.4f}  {term_vertex_linf[name]:>16.12e}"
        )
    print("-" * 78)
    print(
        f"{'sum(projected terms)':>26}  "
        f"{float(np.asarray(dv_sum_terms)[peak_idx]):>16.12e}"
    )
    print(f"{'actual dv/dt':>26}  {peak['value']:>16.12e}")
    print()
    print("Balance checks:")
    print(f"  max|dv_ref - project(total_cc)| = {residual_total:.12e}")
    print(f"  max|dv_ref - sum(projected_terms)| = {residual_terms:.12e}")
    if residual_terms > 1.0e-10:
        print("  WARNING: decomposition residual exceeds 1e-10.")
    else:
        print("  OK: decomposition residual <= 1e-10.")
    if peak["peak_abs"] <= 0.0:
        print("  WARNING: peak |dv/dt| is zero/non-positive.")
    if peak["gc_deg"] > 5.0:
        print("  WARNING: peak is farther than 5 deg from a cube vertex.")
    else:
        print("  OK: peak lies within 5 deg of a cube vertex.")


if __name__ == "__main__":
    main()
