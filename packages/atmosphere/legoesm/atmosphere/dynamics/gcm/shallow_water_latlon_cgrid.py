"""Arakawa C-grid Shallow Water Equations on the latitude-longitude grid.

True staggered C-grid discretisation for the shallow water equations:

* u lives at longitude interfaces: shape (n_lat, n_lon+1)
* v lives at latitude interfaces:  shape (n_lat+1, n_lon)
* h, h_s at cell centers:         shape (n_lat, n_lon)

Key design choices (following MITgcm / Sadourny 1975):
- Vector-invariant momentum form with Bernoulli function at cell centers.
- Sadourny (1975) energy-conserving Coriolis from the ocean C-grid operators.
- Conservative FV mass flux divergence using compact C-grid stencils.
- Pole treatment: v = 0 at poles (wall BC), periodic longitude.
- No checkerboard null space (compact 1-cell gradient/divergence, not 2Δx).

Operator reuse:
- gradient_x/y_cgrid, divergence_cgrid, coriolis_cgrid, vector_laplacian_cgrid
  are imported from the ocean lat-lon C-grid operator module.
- Halo exchange uses the existing lat-lon periodic + pole-folding backend.

References
----------
- Sadourny (1975): The dynamics of finite-difference models of the
  shallow water equations. J. Atmos. Sci., 32, 680-689.
- Arakawa & Lamb (1977): Computational Design of the Basic Dynamical
  Processes of the UCLA General Circulation Model.
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core (sections on
  lat-lon C-grid shallow water).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.grids.operator_adapters import latlon_cgrid_operators
from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
from legoesm.grids.halo import (
    get_halo_backend, get_mpi_topology, get_spmd_mesh,
)
from legoesm.grids.operators_latlon_cgrid import (
    get_band_mpi_cut_layout,
    lat_ends_are_poles,
    pad_lon_cgrid,
)
from legoesm.parallel.latlon_mpi import LatLon2DLayout, LatLonBandLayout
from legoesm.parallel.latlon_spmd import spmd_pole_end_masks
from legoesm.parallel.reductions import global_sum_mpi
from legoesm.core.williamson_sw_analytic import (
    W5_H0_M,
    W5_UBAR_MS,
    solid_body_geopotential,
    solid_body_winds,
    williamson_5_mountain_height,
)
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.polar_filter import (
    compute_polar_filter_mask,
    fourier_filter,
)
from legoesm.atmosphere.dynamics.gcm.shallow_water_generic import cgrid_latlon_sw_tendencies
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm.core.precision import cast_pytree
from legoesm.core.conservation import conservation_accumulator
from legoesm import constants


# ==============================================================================
# State and Config
# ==============================================================================

class CGridLatLonShallowWaterState(NamedTuple):
    """Shallow water state on the lat-lon Arakawa C-grid.

    h   : (n_lat, n_lon)   -- fluid depth at cell centres [m]
    u   : (n_lat, n_lon+1)  -- zonal velocity at longitude interfaces [m/s]
    v   : (n_lat+1, n_lon)  -- meridional velocity at latitude interfaces [m/s]
    h_s : (n_lat, n_lon)   -- surface topography at cell centres [m]
    """
    h: jax.Array
    u: jax.Array
    v: jax.Array
    h_s: jax.Array


class CGridLatLonShallowWaterConfig(NamedTuple):
    """Configuration for the C-grid lat-lon shallow water model.

    Time integrator options: "ssp_rk3", "ssp_rk34", "ssp_rk54", "rk4".
    """
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
    time_integrator: str = "ssp_rk3"
    fix_mass: bool = True
    anchor_mass_to_initial: bool = False  # Mirror PE: anchor fixer to initial mass
    use_ppm_transport: bool = True  # PPM (4th-order) vs simple averaging for mass flux
    use_polar_filter: bool = False
    polar_filter_cutoff_deg: float = 60.0
    polar_filter_max_wave_speed: float = 300.0
    # --- New fields APPENDED (codex review: inserting before existing
    # fields breaks positional NamedTuple construction for callers) ---
    # Biharmonic (del-4) viscosity [m^4/s].  Scale-selective: damps
    # grid-scale noise ~ (k*dx)^4 while leaving resolved scales nearly
    # untouched — unlike A_h, whose k^2 law measurably damps planetary
    # waves at 2.5 deg (Rossby-Haurwitz wave-4 e-folds in ~6 days at
    # A_h ~ 8e5 m^2/s).  Requires passing ``dt`` to the tendency
    # function: the per-latitude-row coefficient is capped at
    # ``nu_del4_cfl_frac * 1/(dt*lam_row^2)`` (lam_row = max Laplacian
    # eigenvalue of the row) so the shrinking zonal spacing toward the
    # poles cannot violate the explicit del-4 stability bound.
    nu_del4: float = 0.0
    # Numerics safety cap fraction for the del-4 diffusive CFL (SSP-RK3
    # real-axis stability ~ 2.5; 0.25 gives 10x margin).  Not a tunable.
    nu_del4_cfl_frac: float = 0.25


# ==============================================================================
# C-grid interpolation helpers
# ==============================================================================

def latlon_coriolis_is_rotated(grid: LatLonGrid) -> bool:
    """Whether the planetary vorticity comes from the rotated-planet formula.

    A Python ``0``/``0.0`` tilt (every existing grid) keeps the historical
    per-row path, bit-identically.  Any other tilt -- concrete or traced (grid
    passed as a dynamic JIT argument) -- takes the rotated formula, which is
    exact for every value (and equals the historical rows at 0) on the serial
    AND band-decomposed lanes: each band evaluates it on its own vertex rows.
    """
    a = getattr(grid, "coriolis_alpha", 0.0)
    return not (isinstance(a, (int, float)) and a == 0.0)


def latlon_rotated_rows(grid: LatLonGrid, lat_rows: jnp.ndarray) -> jnp.ndarray:
    """``rotated_coriolis`` on vertex rows ``lat_rows`` x the u-face longitudes
    ``lon[j] - dlon/2``, shape ``(len(lat_rows), n_lon+1)``; column ``n_lon``
    is an exact copy of column 0 (periodic seam).  Pole rows are NOT clamped
    here -- callers set them to the exact scalars ``-+2 omega cos(alpha)``."""
    from legoesm.core.williamson_sw_analytic import rotated_coriolis
    # Evaluate in the widest available float, then round once to the grid
    # dtype, so the result does not depend on how XLA fuses the transcendentals.
    wide = jnp.result_type(float)
    lon_v = grid.lon.astype(wide) - 0.5 * grid.dlon
    f = rotated_coriolis(lon_v[None, :], lat_rows.astype(wide)[:, None],
                         omega=grid.omega, alpha=grid.coriolis_alpha,
                         xp=jnp).astype(lat_rows.dtype)
    return jnp.concatenate([f, f[:, :1]], axis=1)


def latlon_coriolis_pole_value(grid: LatLonGrid):
    """Exact north-pole planetary vorticity ``2 omega cos(alpha)``."""
    return 2.0 * grid.omega * jnp.cos(grid.coriolis_alpha)


def latlon_vertex_coriolis(grid: LatLonGrid) -> jnp.ndarray:
    """Serial-grid rotated planetary vorticity at all C-grid vertices,
    ``(n_lat+1, n_lon+1)``: interior rows at the latitude interfaces, pole rows
    the exact scalars ``-+2 omega cos(alpha)``."""
    lat = grid.lat
    f = latlon_rotated_rows(grid, 0.5 * (lat[:-1] + lat[1:]))
    pole = jnp.full((1, f.shape[1]), latlon_coriolis_pole_value(grid), dtype=f.dtype)
    return jnp.concatenate([-pole, f, pole], axis=0)


def latlon_pole_end_masks(grid: LatLonGrid):
    """``(south, north)`` booleans: is this array's end v-face row a
    GEOGRAPHIC pole?  Requires all three: the grid is a global pole-to-pole
    grid (``grid.global_poles``), the backend says this end is the physical
    end of the global domain (serial / pole-owning MPI band / SPMD end band --
    never an interior cut), and the face sits at the pole (``cos lat_v`` ~ 0).
    Regional, Mercator and Cartesian walls never match."""
    flag = getattr(grid, "global_poles", False)
    if flag is False:                      # static: not a global polar grid
        return jnp.asarray(False), jnp.asarray(False)
    flag = jnp.asarray(flag)               # True, or traced (grid as jit arg)
    south = flag & (grid.cos_lat_v[0] < _POLE_COS_TOL)
    north = flag & (grid.cos_lat_v[-1] < _POLE_COS_TOL)
    ends = spmd_pole_end_masks()
    if ends is None:
        ends = lat_ends_are_poles()
    return south & ends[0], north & ends[1]


# cos(lat_v) below this marks a v-face row at a geographic pole
# (compute_v_face_coords clamps the pole value to 1e-10).
_POLE_COS_TOL = 1e-6


def _thin_band_layout() -> bool:
    """Static and identical on every rank: can some latitude tile of the
    active MPI decomposition hold a single cell row?  (Then every rank joins
    the extra v-face halo exchange, so the collective stays matched.)"""
    if get_halo_backend() != "mpi":
        return False
    topo = get_mpi_topology()
    if isinstance(topo, LatLon2DLayout):
        n_bands = topo.proc_lat
    elif isinstance(topo, LatLonBandLayout):
        n_bands = topo.n_ranks
    else:
        return False
    return topo.n_lat_global // n_bands < 2


def _extrap_to_centre(v1, v2, l1, l2, lc, active):
    """v at latitude ``lc`` linearly extrapolated from faces (l1, v1) and
    (l2, v2); returns ``v1`` where the stencil is inactive or degenerate."""
    ok = active & (jnp.cos(l2) >= _POLE_COS_TOL) & (l2 != l1)
    w = jnp.where(ok, (l1 - lc) / jnp.where(ok, l2 - l1, 1.0), 0.0)
    v2 = jnp.where(ok, v2, v1)      # an invalid sample never enters the sum
    return v1 + (v1 - v2) * w.astype(v1.dtype)


def latlon_pole_cell_v(v_c, v, grid: LatLonGrid):
    """Cell-row v from lat-face v, ``v_c`` = plain face mean.

    On the pole-adjacent rows the polar v-face has zero length, so its wall
    v=0 is not a value -- the plain mean halves v there (spurious f*v and KE
    gradient under cross-polar flow).  Extrapolate v to the cell-centre
    latitude from the two nearest real faces instead (2nd order); other rows
    unchanged.  Used for the KE (the Coriolis term's pole rows use the
    face-1 transport instead, see ``absolute_vorticity_coriolis``).
    A one-row pole tile fetches its second real face (and its latitude) from
    the neighbouring tile through the lat halo, so the result does not depend
    on the decomposition.  Only a two-row GLOBAL grid, whose second face is
    the other pole's wall, falls back to the single real face.  Grids that
    are not global pole-to-pole grids return ``v_c`` untouched (statically,
    when ``global_poles`` is the stored Python bool; a grid passed as a
    traced jit argument still returns the same values but stages the
    pole-only exchanges)."""
    v_c, v = jnp.asarray(v_c), jnp.asarray(v)
    if getattr(grid, "global_poles", False) is False:
        return v_c          # static: no geographic pole, no extra halo traffic
    pole_s, pole_n = latlon_pole_end_masks(grid)
    lat, lat_v = grid.lat, grid.lat_v
    fetch = v.shape[0] == 2 or _thin_band_layout()
    if fetch:
        # v-face tiles share their end faces, so the neighbour's next real
        # face is the north ghost of v[1:] and the south ghost of v[:-1]
        # (cell-row-sized arrays, halo 1, so any tile size >= 1 row works).
        v2s_, l2s_ = (pad_with_pole_bc_lat(a[1:], halo=1)[-1] for a in (v, lat_v))
        v2n_, l2n_ = (pad_with_pole_bc_lat(a[:-1], halo=1)[0] for a in (v, lat_v))
    if v.shape[0] == 2:
        v2s, l2s, v2n, l2n = v2s_, l2s_, v2n_, l2n_
        v1s, l1s, v1n, l1n = v[1], lat_v[1], v[0], lat_v[0]
    else:
        v2s, l2s, v2n, l2n = v[2], lat_v[2], v[-3], lat_v[-3]
        v1s, l1s, v1n, l1n = v[1], lat_v[1], v[-2], lat_v[-2]
        if fetch:
            # Wider tile in a thin layout: keep a zero-weight dependence on the
            # fetched faces so this rank also enters the REVERSE-mode exchange
            # that returns the thin neighbour's cotangent (no DCE mismatch).
            # (non-finite ghosts, e.g. a NaN wall sample, are zeroed first).
            def _dep(x):
                return 0.0 * jnp.where(jnp.isfinite(x), x, 0.0)
            v2s, l2s = v2s + _dep(v2s_), l2s + _dep(l2s_)
            v2n, l2n = v2n + _dep(v2n_), l2n + _dep(l2n_)
    # A second sample that is itself a pole face (two-row global grid, local
    # or fetched through the halo) is a wall, not a value: use the single real
    # face there.  Inactive stencils get a unit denominator BEFORE dividing
    # (no inf/NaN in the forward value or the reverse-mode cotangent).
    v_s = _extrap_to_centre(v1s, v2s, l1s, l2s, lat[0], pole_s)
    v_n = _extrap_to_centre(v1n, v2n, l1n, l2n, lat[-1], pole_n)
    if v.shape[0] == 2:   # a one-cell tile owning BOTH poles has no real face
        both = pole_s & pole_n
        pole_s, pole_n = pole_s & ~both, pole_n & ~both
        v_c = jnp.where(both, jnp.zeros_like(v_c), v_c)   # v = 0: walls only
    v_c = v_c.at[0].set(jnp.where(pole_s, v_s, v_c[0]))
    return v_c.at[-1].set(jnp.where(pole_n, v_n, v_c[-1]))


def latlon_cell_ke(u_c, v_c, v, grid: LatLonGrid):
    """Cell-centre kinetic energy from cell winds ``(u_c, v_c)`` (plain face
    means) and the lat-face ``v``: pole rows use :func:`latlon_pole_cell_v`.
    The ONE KE used by the lat-lon legacy PE, generic PE (sb81_theta) and SW
    cores."""
    return 0.5 * (u_c**2 + latlon_pole_cell_v(v_c, v, grid)**2)


def _lon_ring_sum(x):
    """Complete a per-rank partial LONGITUDE sum over the whole circle.

    Local / lat-band: identity.  SPMD ``("lat", "lon")`` mesh: ``psum`` over
    ``"lon"``.  2-D MPI with ``proc_lon > 1``: each rank writes its partial
    into its lat-band slot and one AD-safe ``global_sum_mpi`` completes every
    band's row sum (no sub-communicator needed)."""
    backend = get_halo_backend()
    if backend == "spmd":
        mesh = get_spmd_mesh()
        if mesh is not None and "lon" in tuple(
                getattr(mesh, "axis_names", ())):
            return jax.lax.psum(x, "lon")
    elif backend == "mpi":
        topo = get_mpi_topology()
        if isinstance(topo, LatLon2DLayout) and topo.proc_lon > 1:
            slots = jnp.zeros((topo.proc_lat,) + x.shape, x.dtype)
            return global_sum_mpi(slots.at[topo.proc_row].set(x))[topo.proc_row]
    return x


def _pole_cap_vorticity(zeta, u, grid):
    """Overwrite the geographic-pole vertex rows of ``zeta`` with the
    polar-cap Stokes vorticity (see ``absolute_vorticity_coriolis``)."""
    if getattr(grid, "global_poles", False) is False:
        return zeta         # static: no geographic pole, no lon reduction
    # Per-rank partial sums of u over the owned u-faces (drop the wrap
    # column) and the face count, both completed over the longitude circle.
    sums = _lon_ring_sum(jnp.stack([
        jnp.sum(u[0, :-1], axis=0), jnp.sum(u[-1, :-1], axis=0),
        jnp.full(u.shape[2:], u.shape[1] - 1, u.dtype),
    ]))
    u_s, u_n = sums[0] / sums[2], sums[1] / sums[2]   # zonal-mean pole-row u
    pole_s, pole_n = latlon_pole_end_masks(grid)
    # Stokes: zeta = circulation / cap area, cap from the pole to the first
    # cell-centre row at angular distance d:  (-+ u R sin d 2pi) /
    # (R^2 2pi (1 - cos d)),  1 - cos d = 2 sin^2(d/2)  (fp32-safe near the
    # pole).  Inactive ends get a unit denominator (no inf/NaN, finite grads).
    d_s = grid.lat[0] + 0.5 * jnp.pi
    d_n = 0.5 * jnp.pi - grid.lat[-1]
    den_s = jnp.where(pole_s, 2.0 * jnp.sin(0.5 * d_s) ** 2, 1.0)
    den_n = jnp.where(pole_n, 2.0 * jnp.sin(0.5 * d_n) ** 2, 1.0)
    z_s = -u_s * grid.cos_lat[0] / (grid.radius * den_s)
    z_n = u_n * grid.cos_lat[-1] / (grid.radius * den_n)
    zeta = zeta.at[0].set(jnp.where(
        pole_s, jnp.broadcast_to(z_s, zeta[0].shape).astype(zeta.dtype), zeta[0]))
    return zeta.at[-1].set(jnp.where(
        pole_n, jnp.broadcast_to(z_n, zeta[-1].shape).astype(zeta.dtype), zeta[-1]))


def absolute_vorticity_coriolis(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    *,
    u_lat_pad: jnp.ndarray | None = None,
    lat_pad: jnp.ndarray | None = None,
    lat_pad_pole: jnp.ndarray | None = None,
    cos_lat_pad: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Coriolis-like acceleration using absolute vorticity (ζ+f).

    ``lat_pad`` (exactly ``pad_with_pole_bc_lat(grid.lat, halo=1, 0, 0)``),
    ``lat_pad_pole`` / ``cos_lat_pad`` (see ``curl_vertex_cgrid``): optional
    precomputed stage-invariant geometry pads (the SPMD band step's geometry
    stacks) so the per-stage scalar-row exchanges are skipped; ``None`` keeps
    the in-body backend-dispatched pads byte-identical.

    The vector-invariant momentum equation requires the full absolute
    vorticity η = ζ + f, not just the planetary vorticity f.  This
    function computes ζ at vertices via ``curl_vertex_cgrid``, adds the
    planetary vorticity, averages η to velocity faces, and returns the
    cross-product terms.

    Parameters
    ----------
    u : (n_lat, n_lon+1) or (n_lat, n_lon+1, nlev)
    v : (n_lat+1, n_lon) or (n_lat+1, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    cor_u, cor_v : same shapes as u, v
    """
    is_3d = u.ndim == 3

    # Static band-MPI dispatch (None ⟺ serial / single-rank band /
    # non-band topology — the historical code path runs verbatim).
    # Under latitude-band MPI with interior cuts, the band's end
    # vertex/face rows are NOT poles, so the two pole-constant
    # constructions below (planetary vorticity ±2Ω, u_at_v zero wall
    # rows) must instead use the neighbour rank's rows.
    _band = get_band_mpi_cut_layout()
    # Single-program SPMD twin: under the lat-band shard_map backend
    # ``get_band_mpi_cut_layout()`` is None (it recognizes only the 'mpi'
    # backend), so the f_vert / u_at_v pole constants must instead be selected
    # DATA-dependently per band — interior band cuts are NOT poles. Returns
    # ``(south_mask, north_mask)`` traced booleans inside the shard_map, or None
    # for serial/MPI/cube (those paths keep their static branch — additive).
    # Mirrors curl_vertex_cgrid's SPMD pole handling.
    _spmd_pm = spmd_pole_end_masks()
    _rotated = latlon_coriolis_is_rotated(grid)

    # --- Relative vorticity at vertices (via the B2 operator interface) ---
    # ``u_lat_pad`` (optional, exactly pad_with_pole_bc_lat(u, halo=1,
    # 0, 0)) is shared between the curl's circulation pad and the
    # 4-pt u->v-face average below — the atm tendency fuses it with
    # its other entry-level pads (census probe 8460424: u was padded
    # twice per RK stage).
    zeta = latlon_cgrid_operators(grid).vorticity(
        u, v, u_ext=u_lat_pad,
        lat_pad_pole=lat_pad_pole, cos_lat_pad=cos_lat_pad,
    )  # (n_lat+1, n_lon+1[, nlev])
    # Pole vertex rows: the curl leaves the free-slip WALL value 0 there, but
    # the pole is an interior point of the sphere.  Use Stokes over the polar
    # cap -- the zonal sum of the pole-row wedge circulations -u dx (south) /
    # +u dx (north) over the cap area -- so cross-polar flow sees its real
    # relative vorticity.  Applied only where the end v-face is a true pole
    # (cos lat_v ~ 0); interior band cuts and regional walls are untouched.
    zeta = _pole_cap_vorticity(zeta, u, grid)

    # --- Planetary vorticity at vertices ---
    lat = grid.lat  # cell-center latitudes
    # Rotation rate from the GRID, not constants.Omega: a non-rotating
    # grid (create_latlon_grid(..., omega=0.0) — the colliding-modons
    # case, #521) must yield f_vert = 0, but the hardcoded
    # constants.Omega silently kept the planet rotating regardless of
    # the grid omega (codex 2026-07-03 HIGH).  ``grid.omega`` is the
    # stored construction scalar (appended NamedTuple field), so this
    # is exact on any MPI band (no recovery from local ``f`` rows —
    # a band owning only the equator row has sin(lat) = 0), stays in
    # JAX space if the grid is ever passed as a dynamic pytree, and is
    # bit-identical for every rotating caller (grids built with the
    # default omega = constants.Omega).
    twoOmega = 2.0 * grid.omega
    if _band is None and _spmd_pm is None:
        # Serial / single-rank: sin(±π/2) = ±1 exactly, so build f_vert directly
        # from the interior sin via Pad with constant_values = ±2Ω.  Single
        # Pad HLO op replaces alloc-2-singletons + concatenate-of-three + sin.
        lat_int = 0.5 * (lat[:-1] + lat[1:])
        f_vert = jnp.pad(
            twoOmega * jnp.sin(lat_int),
            (1, 1), constant_values=(-twoOmega, twoOmega),
        )  # (n_lat+1,)
        if _rotated:
            f_vert_2d = latlon_vertex_coriolis(grid)
    else:
        # Band-decomposed (MPI or single-program SPMD): end vertex rows at
        # interior cuts sit at the midpoint latitude ACROSS the cut.  Pad
        # ``lat`` one row (backend-aware: MPI 1-D sendrecv / SPMD lat-band
        # ppermute at cuts; the constant ghost at a pole-touching end never
        # reaches the output — overwritten below), midpoint to vertex rows,
        # then restore the exact ±2Ω pole values at PHYSICAL pole ends only.
        # Interior vertex rows evaluate the identical
        # ``2Ω·sin(0.5·(lat[j-1]+lat[j]))`` chain — bit-identical to serial.
        from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
        if lat_pad is None:
            lat_pad = pad_with_pole_bc_lat(
                lat, halo=1, south_value=0.0, north_value=0.0,
            )
        lat_vert = 0.5 * (lat_pad[:-1] + lat_pad[1:])  # (n_lat_local+1,)
        f_vert = twoOmega * jnp.sin(lat_vert)
        if _spmd_pm is not None:
            # SPMD: select the ±2Ω clamp DATA-dependently per band (a static
            # ``if`` would clamp every interior cut). south then north,
            # sequenced like the MPI path so a single band clamps both ends.
            _south_m, _north_m = _spmd_pm
            f_vert = jnp.where(_south_m, f_vert.at[0].set(-twoOmega), f_vert)
            f_vert = jnp.where(_north_m, f_vert.at[-1].set(twoOmega), f_vert)
        else:
            # MPI: static per-rank pole answer (None ⟺ this rank owns the pole).
            if _band.south_rank is None:
                f_vert = f_vert.at[0].set(-twoOmega)
            if _band.north_rank is None:
                f_vert = f_vert.at[-1].set(twoOmega)
        if _rotated:
            # Rotated planet on a band: the formula on THIS band's vertex rows
            # (interior cuts are not poles), with the exact pole scalars
            # restored under the same physical-pole conditions as above.
            f_vert_2d = latlon_rotated_rows(grid, lat_vert)
            _fp = latlon_coriolis_pole_value(grid).astype(f_vert_2d.dtype)
            if _spmd_pm is not None:
                f_vert_2d = jnp.where(_south_m, f_vert_2d.at[0].set(-_fp), f_vert_2d)
                f_vert_2d = jnp.where(_north_m, f_vert_2d.at[-1].set(_fp), f_vert_2d)
            else:
                if _band.south_rank is None:
                    f_vert_2d = f_vert_2d.at[0].set(-_fp)
                if _band.north_rank is None:
                    f_vert_2d = f_vert_2d.at[-1].set(_fp)

    # Absolute vorticity at vertices
    if is_3d:
        eta = zeta + f_vert[:, jnp.newaxis, jnp.newaxis]
    else:
        eta = zeta + f_vert[:, jnp.newaxis]
    if _rotated:
        eta = zeta + (f_vert_2d[..., jnp.newaxis] if is_3d else f_vert_2d)

    # --- Average η to u-faces ---
    # u-face[i, j] is flanked by vertex[i, j] (south) and vertex[i+1, j] (north)
    eta_at_u = 0.5 * (eta[:-1] + eta[1:])  # (n_lat, n_lon+1[, nlev])

    # --- Average v to u-faces (4-point, same as coriolis_cgrid) ---
    # v sits on lat-faces, cell-aligned in lon; the u-face 4-pt average needs
    # v's lon WEST-neighbour cell, so pad-then-average through the dispatched
    # lon halo (local wrap / 2-D ring) -> the full n_lon+1 u-faces directly.
    # Bit-identical to ``roll(v,1)`` + wrap-column concat at proc_lon==1, and
    # spans lon partition cuts under a 2-D split (THIS local roll was the
    # missed lon op that made the 2-D u-momentum decomposition-dependent).
    v_pad = pad_lon_cgrid(v, halo=1)
    ve = v_pad[:, 1:]    # cell j   bordering u-face j
    vw = v_pad[:, :-1]   # cell j-1 bordering u-face j
    v_at_u = jnp.asarray(0.25 * (ve[:-1] + ve[1:] + vw[:-1] + vw[1:]))  # (n_lat, n_lon+1)
    # (Pole rows are overwritten below by the energy-conserving form.)

    # --- Average η to v-faces ---
    # v-face[i, j] is flanked by vertex[i, j] (west) and vertex[i, j+1] (east)
    # vertex[:, n_lon] wraps to vertex[:, 0]
    eta_at_v = 0.5 * (eta[:, :-1] + eta[:, 1:])  # (n_lat+1, n_lon[, nlev])

    # --- Average u to v-faces (4-point, same as coriolis_cgrid) ---
    if _band is None and _spmd_pm is None:
        # Serial / single-rank: ``jnp.pad`` on the leading axis (one HLO Pad op
        # vs alloc + concat).  The zero end rows are the pole wall BC.
        u_avg_interior = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
        if is_3d:
            u_at_v = jnp.pad(u_avg_interior, ((1, 1), (0, 0), (0, 0)))
        else:
            u_at_v = jnp.pad(u_avg_interior, ((1, 1), (0, 0)))
    else:
        # Band-decomposed (MPI or SPMD): the end v-face rows at interior cuts
        # carry the genuine 4-point average spanning the cut, not the pole wall
        # zero.  Pad ``u`` one lat row (backend-aware: MPI sendrecv / SPMD
        # lat-band ppermute; u-face fields with the n_lon+1 wrap column are
        # supported), average on the padded array — interior rows the identical
        # operand chain as serial — then restore the zero wall rows at PHYSICAL
        # pole ends only.
        from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
        if u_lat_pad is None:
            u_lat_pad = pad_with_pole_bc_lat(
                u, halo=1, south_value=0.0, north_value=0.0,
            )  # (n_lat_local+2, n_lon+1[, nlev])
        u_at_v = 0.25 * (
            u_lat_pad[:-1, :-1] + u_lat_pad[:-1, 1:]
            + u_lat_pad[1:, :-1] + u_lat_pad[1:, 1:]
        )  # (n_lat_local+1, n_lon[, nlev])
        if _spmd_pm is not None:
            # SPMD: zero the pole-wall v-face DATA-dependently per band.
            _south_m, _north_m = _spmd_pm
            u_at_v = jnp.where(
                _south_m, u_at_v.at[0].set(jnp.zeros_like(u_at_v[0])), u_at_v)
            u_at_v = jnp.where(
                _north_m, u_at_v.at[-1].set(jnp.zeros_like(u_at_v[-1])), u_at_v)
        else:
            # MPI: static per-rank pole answer.
            if _band.south_rank is None:
                u_at_v = u_at_v.at[0].set(jnp.zeros_like(u_at_v[0]))
            if _band.north_rank is None:
                u_at_v = u_at_v.at[-1].set(jnp.zeros_like(u_at_v[-1]))

    # --- Coriolis terms ---
    cor_u = eta_at_u * v_at_u
    cor_v = -eta_at_v * u_at_v

    # --- Pole rows: energy-conserving vorticity term (Sadourny 1975) ---
    # The product-of-averages form above is not antisymmetric: its work
    # sum A_u u cor_u + sum A_v v cor_v does not cancel.  In the interior
    # the residual is small (smooth eta, A_u ~ A_v), but on the pole rows the
    # pole-vertex eta multiplies v with NO transpose partner (the polar
    # v-face is a zero-length wall) and A_u(row 0) ~ A_v(face 1)/2, so the
    # term itself feeds a pole-trapped zonal-wavenumber-1 mode (Held-Suarez,
    # CAM6 L32, 72x144: this term alone grew the pole-row eddy KE at
    # +1.9/day; max wind 31 m/s at 7 hPa on the 88.75 deg row).  There the
    # term takes Sadourny's (1975) energy-conserving form -- averages of
    # PRODUCTS of vertex eta with face TRANSPORTS V = v dx_v, U = u dy
    # (Arakawa & Lamb 1981) -- so each (u, vertex, v) triple enters cor_u
    # and cor_v with weights whose area-weighted work cancels, and the
    # zero-length polar face carries zero transport rather than a wall value.
    # The pole-vertex (Stokes cap) eta is kept, folded into the
    # (pole-row u, first interior vertex, v_1) triples as eta_hat =
    # (eta_pole + eta_1)/2 on BOTH sides (the cancellation is kept); without
    # it cor_u would use eta half a cell off the pole row (tilted solid-body
    # cross-polar test: pole-row error 2.9% vs 0.4% with eta_hat).  Scope of
    # the guarantee: area-weighted (vector-invariant form, as in the
    # interior).  Only the pole-row u triples change: cor_u on row 0 and
    # the u_0 part of cor_v on face 1 (the u_1 part keeps the interior form;
    # north mirrored).  Accuracy cost: v enters through the face-1 transport,
    # so a zonally uniform (m = 0) meridional flow v ~ cos(lat) gets twice its
    # pointwise f v on the pole row (an O(dlat) absolute error, as v itself
    # is O(dlat) there); cross-polar (m = 1) flow is kept (above).
    # Needs >= 2 rows per latitude tile (a one-row tile keeps the plain
    # product-of-averages form above):
    # on a one-row pole tile face 1 is a cut face.  (Before this change the
    # operator already raised (2 x 1-row bands) or hung (2 + 1 rows) on
    # one-row tiles under latitude-band MPI.)
    if u.shape[0] < 2:      # static: a one-row tile has no interior v-face
        return cor_u, cor_v
    clv = grid.cos_lat_v
    pole_s, pole_n = latlon_pole_end_masks(grid)
    two_rows = u.shape[0] == 2      # static: row 1 may be the north pole row

    def _hat(eta_p, eta_1):
        return 0.5 * (eta_p + eta_1)

    def _cor_u_pole(eta_hat, ve_1, vw_1, clv_1, cl_0):
        """Pole-row cor_u: ``ve_1``/``vw_1`` are the face-1 v east/west of
        each u-face (the lon-padded rows built above -- no extra exchange)."""
        return 0.5 * eta_hat * (0.5 * (ve_1 + vw_1) * clv_1) / cl_0

    def _cor_v_face1(eta_hat, u_0, eta_1, u_1, u_1_pole):
        """cor_v on the first interior v-face.  Pole-row (u_0) triples: the
        energy-conserving form with eta_hat.  Interior-row (u_1) triples: the
        unchanged product-of-averages form, unless u_1 is itself a pole row
        (two-row grid; ``u_1_pole`` is then its eta_hat, else None)."""
        EU0 = 0.5 * eta_hat * u_0                            # vertex row 1
        out = -0.5 * (EU0[:-1] + EU0[1:])
        if u_1_pole is None:
            return out - 0.5 * (eta_1[:-1] + eta_1[1:]) * 0.25 * (u_1[:-1] + u_1[1:])
        EU1 = 0.5 * u_1_pole * u_1
        return out - 0.5 * (EU1[:-1] + EU1[1:])

    hat_s, hat_n = _hat(eta[0], eta[1]), _hat(eta[-1], eta[-2])
    cu_s = _cor_u_pole(hat_s, ve[1], vw[1], clv[1], grid.cos_lat[0])
    cu_n = _cor_u_pole(hat_n, ve[-2], vw[-2], clv[-2], grid.cos_lat[-1])
    # rows 0..1 / -2..-1 are tile-local (>= 2 rows): no lat-halo dependence
    cv_s = _cor_v_face1(hat_s, u[0], eta[1], u[1], None)
    cv_n = _cor_v_face1(hat_n, u[-1], eta[-2], u[-2], None)
    if two_rows:   # row 1 may be the north pole row (both rows pole rows)
        cv_s = jnp.where(pole_n, _cor_v_face1(hat_s, u[0], eta[1], u[1], hat_n), cv_s)
        cv_n = jnp.where(pole_s, _cor_v_face1(hat_n, u[-1], eta[-2], u[-2], hat_s), cv_n)
    cor_u = cor_u.at[0].set(jnp.where(pole_s, cu_s, cor_u[0]))
    cor_u = cor_u.at[-1].set(jnp.where(pole_n, cu_n, cor_u[-1]))
    cor_v = cor_v.at[1].set(jnp.where(pole_s, cv_s, cor_v[1]))
    cor_v = cor_v.at[-2].set(jnp.where(pole_n, cv_n, cor_v[-2]))

    return cor_u, cor_v


# ==============================================================================
# Tendency computation
# ==============================================================================

def nu_del4_row_profiles(
    grid: LatLonGrid,
    nu_del4: float,
    cfl_frac: float,
    dt,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-latitude-row biharmonic coefficients with pole stability cap.

    The zonal spacing ``dx = R*dlon*cos(lat)`` shrinks toward the poles,
    so a globally-constant ``nu_del4`` sized for the interior violates
    the explicit del-4 diffusive CFL near the poles by many orders of
    magnitude.  Cap the coefficient per row at

        nu_max(row) = cfl_frac / (dt * lam_row^2),
        lam_row     = 4/dx_row^2 + 4/dy^2

    (``lam_row`` = max eigenvalue of the discrete 5-point Laplacian on
    that row), which keeps ``nu*dt*lam^2 <= cfl_frac`` everywhere.  The
    interior rows keep the full ``nu_del4``; only rows poleward
    of the crossover (about 79 deg at 2.5 deg resolution, 48 h grid
    e-folding) are reduced.

    Returns
    -------
    (nu_u, nu_v) : ((n_lat, 1), (n_lat+1, 1)) coefficient columns for
        the u-face (cell-center-lat) and v-face (lat-interface) rows.
    """
    # Per-row cell height: ``grid.dy`` is the 2-cell distance (see
    # LatLonGrid docstring), so the single-cell height of row j is
    # dy[j]/2 — scalar ``grid.dlat`` is only a diagnostic minimum on
    # Mercator/stretched grids and would over-cap interior rows there
    # (codex review).
    dy_u = 0.5 * grid.dy                                 # (n_lat,)
    # v-face (interface) rows: mean of the adjacent cell heights;
    # boundary interfaces reuse the edge-row height.
    dy_v = jnp.concatenate([
        dy_u[0:1], 0.5 * (dy_u[:-1] + dy_u[1:]), dy_u[-1:],
    ])                                                   # (n_lat+1,)
    dx_u = grid.radius * grid.dlon * grid.cos_lat        # (n_lat,)
    dx_v = grid.radius * grid.dlon * grid.cos_lat_v      # (n_lat+1,)
    lam_u = 4.0 / dx_u ** 2 + 4.0 / dy_u ** 2
    lam_v = 4.0 / dx_v ** 2 + 4.0 / dy_v ** 2
    nu_u = jnp.minimum(nu_del4, cfl_frac / (dt * lam_u ** 2))
    nu_v = jnp.minimum(nu_del4, cfl_frac / (dt * lam_v ** 2))
    return nu_u[:, None], nu_v[:, None]


# ==============================================================================
# Model class
# ==============================================================================

class CGridLatLonShallowWaterModel(IntegrationMixin):
    """C-grid shallow water model on the latitude-longitude grid.

    Features:
    - Compact-stencil C-grid operators (no checkerboard mode).
    - Sadourny (1975) energy-conserving Coriolis.
    - Conservative mass flux divergence.
    - Wall BC at poles (v = 0), periodic longitude.
    - Optional Laplacian viscosity and mass conservation fixer.

    Parameters
    ----------
    grid : LatLonGrid
        Horizontal grid.
    config : CGridLatLonShallowWaterConfig, optional
    """

    def __init__(
        self,
        grid: LatLonGrid,
        config: CGridLatLonShallowWaterConfig | None = None,
        dt: float = 600.0,
    ):
        self.grid = grid
        self.config = config or CGridLatLonShallowWaterConfig()
        # Constructor dt, kept as the fallback for the no-argument
        # ``tendencies(state)`` path (the nu_del4 pole cap needs a dt).
        self._dt = dt
        # Precompute polar filter masks (cached, not traced): one for
        # cell-centered rows (dh, du after lon-trim) and one for v-face
        # rows (dv).  The v-face mask uses ``grid.cos_lat_v`` / the
        # half-cell-offset lat-interface coordinates so the wavenumber
        # cutoff matches the actual v-face CFL — mirrors the PE dycore
        # (``primitive_eq_latlon_cgrid``), which fixed exactly this:
        # filtering only dh/du leaves unfiltered high-k dv modes at the
        # pole-adjacent v rows to violate CFL at the relaxed dt.
        if self.config.use_polar_filter:
            self._polar_mask = compute_polar_filter_mask(
                grid, dt=dt,
                max_wave_speed=self.config.polar_filter_max_wave_speed,
                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
            )
            self._polar_mask_v = compute_polar_filter_mask(
                grid, dt=dt,
                max_wave_speed=self.config.polar_filter_max_wave_speed,
                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
                is_v_face=True,
            )
        else:
            self._polar_mask = None
            self._polar_mask_v = None

        # Iter-4: anchored mass target (fp64, lazy on first step()).
        # Mirrors ``CGridLatLonPrimitiveEquationModel._target_mass``.
        self._target_mass: jax.Array | None = None



    def compute_mass(self, state: CGridLatLonShallowWaterState) -> jax.Array:
        """Compute total mass (for conservation fixer target).

        Uses the fp64 conservation accumulator unconditionally — mass
        budgets need higher precision than the per-step compute dtype
        (``_accumulation_dtype`` may be fp32 on fp32 policies, which
        leaks ~10^-7 reduction noise into the anchored target and
        defeats the fixer; ``conservation_accumulator`` promotes to
        fp64 whenever x64 is enabled).
        """
        acc = conservation_accumulator()
        return jnp.sum(state.h.astype(acc) * self.grid.area.astype(acc))

    def tendencies(
        self, state: CGridLatLonShallowWaterState,
        dt: float | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute tendencies (pure function wrapper).

        ``dt`` is only consumed when ``config.nu_del4 > 0`` (the del-4
        pole stability cap is dt-dependent); when omitted it falls back
        to the constructor ``dt`` so the no-argument
        ``model.tendencies(state)`` path
        keeps working with the biharmonic enabled.
        """
        if dt is None:
            dt = self._dt
        return cgrid_latlon_sw_tendencies(state, self.grid, self.config, dt)

    def step(
        self,
        state: CGridLatLonShallowWaterState,
        dt: float,
        target_mass: jax.Array | None = None,
    ) -> CGridLatLonShallowWaterState:
        """Outer wrapper: snapshots initial mass on first call when
        ``anchor_mass_to_initial`` is on (fp64, outside JIT), then
        delegates to the JIT'd inner step."""
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None
                and target_mass is None):
            self._target_mass = self.compute_mass(state)
        if (target_mass is None
                and self.config.fix_mass
                and self.config.anchor_mass_to_initial):
            target_mass = self._target_mass
        return self._step_jit(state, dt, target_mass)

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(
        self,
        state: CGridLatLonShallowWaterState,
        dt: float,
        target_mass: jax.Array | None = None,
    ) -> CGridLatLonShallowWaterState:
        """Advance one time step.

        Parameters
        ----------
        state : CGridLatLonShallowWaterState
        dt : float
        target_mass : jax.Array or None
            If provided, the mass fixer corrects to this target.
            Compute once via ``model.compute_mass(state0)`` and pass
            as an explicit traced argument (avoids JIT recompilation
            when the target changes).
        """
        state_c = cast_pytree(state, None, "compute")

        def tendency_fn(s):
            dh, du, dv = cgrid_latlon_sw_tendencies(
                s, self.grid, self.config, dt,
            )
            # Polar filter: damp high-frequency modes near poles
            if self._polar_mask is not None:
                dh = fourier_filter(dh, self.grid, self._polar_mask)
                du_interior = fourier_filter(du[:, :-1], self.grid, self._polar_mask)
                du = jnp.concatenate([du_interior, du_interior[:, 0:1]], axis=1)
                # dv on lat-interface (v-face) rows, with its own mask —
                # mirrors the PE dycore's polar_mask_v treatment.
                dv = fourier_filter(dv, self.grid, self._polar_mask_v)
            return CGridLatLonShallowWaterState(
                h=dh, u=du, v=dv,
                h_s=jnp.zeros_like(s.h_s),
            )

        state_new = dispatch_integrator(
            state_c, tendency_fn, dt, self.config.time_integrator,
        )

        # Enforce v = 0 at poles via a single Pad HLO op.
        v_new = jnp.pad(state_new.v[1:-1, :], ((1, 1), (0, 0)))
        state_new = state_new._replace(v=v_new)

        # Conservation fixer — use the fp64 budget accumulator so the
        # mass integrals aren't contaminated by fp32 reduction noise
        # (fp32 ``jnp.sum`` over 16k+ cells leaks ~N·eps relative
        # error and produces spurious O(1e-6) drift even when the
        # anchored target is exact).
        if self.config.fix_mass:
            acc = conservation_accumulator()
            area = self.grid.area.astype(acc)
            total_area = jnp.sum(area)
            if target_mass is not None:
                mass_target = target_mass
                mass_new = jnp.sum(state_new.h.astype(acc) * area)
            else:
                # Both mass integrals share the ``* area`` weight on
                # the same horizontal axes — stack and reduce once.
                _h_pair = jnp.stack(
                    [state.h.astype(acc), state_new.h.astype(acc)], axis=-1,
                ) * area[..., None]
                _mass_pair = jnp.sum(
                    _h_pair, axis=tuple(range(area.ndim)),
                )
                mass_target, mass_new = _mass_pair[0], _mass_pair[1]
            correction = (mass_target - mass_new) / total_area
            # Iter-4: drop ``.astype(state_new.h.dtype)`` so the
            # fp64 correction promotes the add (matches
            # ``fix_ps_mass`` semantics on cubed-sphere PE).  The
            # end-of-step ``cast_pytree(..., "storage")`` rounds back
            # to fp32 once instead of compounding fp32 quantization
            # every step.
            h_fixed = state_new.h + correction
            # Re-clamp after mass correction to prevent negative depth
            h_fixed = jnp.maximum(h_fixed, 0.0)
            state_new = state_new._replace(h=h_fixed)

        return cast_pytree(state_new, None, "storage")


# ==============================================================================
# Initial condition helpers
# ==============================================================================

# cell_to_cgrid_winds is imported from
# legoesm.ocean.dynamics.latlon_cgrid_operators (shared with ocean).
# The shared version handles both 2D and 3D via ndim dispatch.


def williamson_test2_cgrid(
    grid: LatLonGrid,
) -> CGridLatLonShallowWaterState:
    """Williamson Test 2 initial condition on the C-grid.

    Global steady-state nonlinear zonal geostrophic flow (12-day rotation).
    Since the flow is purely zonal and zonally uniform, u at lon faces
    equals u at cell centers.
    """
    R = grid.radius
    Omega = constants.Omega
    g = constants.g

    u_0 = 2.0 * jnp.pi * R / (12.0 * 86400.0)
    gh_0 = 2.94e4
    h_0 = gh_0 / g

    lat = grid.lat2d   # (n_lat, n_lon)

    # Height at cell centers (geostrophic balance)
    h = h_0 - (R * Omega * u_0 + u_0**2 / 2.0) * jnp.sin(lat)**2 / g

    # Zonal velocity at cell centers (zonally uniform)
    u_cell = u_0 * jnp.cos(lat)

    # Convert to C-grid: u at lon faces, v = 0 at lat faces
    n_lat, n_lon = grid.n_lat, grid.n_lon
    # u at lon faces: same as cell centers since u is zonally uniform
    u_face = jnp.concatenate([u_cell, u_cell[:, 0:1]], axis=1)
    v_face = jnp.zeros((n_lat + 1, n_lon), dtype=h.dtype)

    return CGridLatLonShallowWaterState(
        h=h, u=u_face, v=v_face, h_s=jnp.zeros_like(h),
    )


def williamson_test2_exact_cgrid(
    grid: LatLonGrid,
    t: float,
) -> CGridLatLonShallowWaterState:
    """Exact solution for Williamson Test 2 at time t (steady-state)."""
    return williamson_test2_cgrid(grid)


def williamson_test5_cgrid(
    grid: LatLonGrid,
) -> CGridLatLonShallowWaterState:
    """Williamson Test 5: Zonal flow over isolated mountain on C-grid."""
    R = grid.radius
    g = constants.g

    u_0 = W5_UBAR_MS
    lat = grid.lat2d
    lon = grid.lon2d
    n_lat, n_lon = grid.n_lat, grid.n_lon

    # Analytic fields: ONE shared definition (williamson_sw_analytic).
    u_cell, _ = solid_body_winds(lon, lat, u0=u_0, xp=jnp)

    # ★ CHANGED: this used a GREAT-CIRCLE radius.  Williamson et al.
    # (1992) case 5 and test_cases.F90:1185 both specify the clipped
    # PLANAR (lon, lat) radius; the two differ by up to 295.3 m of
    # mountain height (codex r1, numerically maximised), so this lane
    # was not running the same case as its cubed-sphere sibling.
    h_s = williamson_5_mountain_height(lon, lat, xp=jnp)

    # Height: h = h_free - h_s
    h_free = solid_body_geopotential(lon, lat, radius=R,
                                     omega=constants.Omega, u0=u_0,
                                     gh0=W5_H0_M * g, xp=jnp) / g
    h = h_free - h_s

    # C-grid winds
    u_face = jnp.concatenate([u_cell, u_cell[:, 0:1]], axis=1)
    v_face = jnp.zeros((n_lat + 1, n_lon), dtype=h.dtype)

    return CGridLatLonShallowWaterState(
        h=h, u=u_face, v=v_face, h_s=h_s,
    )


def compute_error_norms_cgrid(
    state: CGridLatLonShallowWaterState,
    reference: CGridLatLonShallowWaterState,
    grid: LatLonGrid,
) -> dict[str, float]:
    """Compute L1, L2, L∞ error norms for height field."""
    err = state.h - reference.h
    ref = reference.h
    area = grid.area

    l1 = jnp.sum(jnp.abs(err) * area) / jnp.sum(jnp.abs(ref) * area)
    l2 = jnp.sqrt(jnp.sum(err**2 * area) / jnp.sum(ref**2 * area))
    linf = jnp.max(jnp.abs(err)) / jnp.max(jnp.abs(ref))

    return {"l1": float(l1), "l2": float(l2), "linf": float(linf)}
