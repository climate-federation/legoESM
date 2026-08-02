"""Oriented C-grid section transports and their time-mean accumulation.

Fills a real gap: the lat-lon / tripole C-grid had NO reusable "integrate a
flux through a set of faces" helper.  ``diagnostics_streamfunction`` owns
``_v_face_geometry`` (private, v-faces only) and ``barotropic_streamfunction``
inlines its own u-face min-rule; the genuine oriented-section logic existed
only inlined twice in the MPAS edge branch (``spinup.compute_amoc_from_state_
mpas`` / ``compute_acc_from_state_mpas``).  This module provides that helper
once, for both face families, and builds the Arctic gateway sections on top of
it so the model and the offline probes share ONE definition.

WHY BOUNDARY FACES AND NOT A LATITUDE ROW
-----------------------------------------
On the eORCA1 TRIPOLE the grid is curvilinear north of ~60N, so a "latitude
row" is NOT a grid row and integrating ``v`` along a j-index would not be a
geographic section.  Instead a face is selected when its two adjacent CELL
CENTRES straddle the region boundary.  Every boundary face is then counted
EXACTLY ONCE and oriented into the region, so corners cannot be double counted
and any longitude binning of those faces partitions the boundary exactly.

SIGN CONVENTION (stated at the term)
------------------------------------
Orientation signs are built so that a POSITIVE transport is INTO the region,
regardless of the face's own i/j orientation:

* u-face ``i`` lies between cell ``i-1`` (west) and cell ``i`` (east).  If the
  EAST cell is inside the region, an eastward (positive) ``mfu`` carries fluid
  INTO it, so the sign is ``+1``; if the WEST cell is inside, ``-1``.
* v-face ``j`` lies between cell ``j-1`` (south) and cell ``j`` (north).  If
  the NORTH cell is inside, a northward (positive) ``mfv`` enters, sign ``+1``.

FLUX CONVENTION
---------------
``mfu``/``mfv`` are THICKNESS-WEIGHTED face transports [m^2/s] (``h_face * u``)
built with the model's own operators, matching ``divergence_cgrid``'s metric
``div = (1/A)[u_e dy_e - u_w dy_w + v_n dx_n - v_s dx_s]``.  A face's volume
transport is therefore ``mfu * dy_u`` (u) or ``mfv * dx_v`` (v) in m^3/s.

The tracer value carried across a face is the UPWIND (donor-cell) value.  That
is a DIAGNOSTIC choice, stated rather than hidden: the model integrates tracers
with a flux limiter (e.g. superbee) which is not reproduced here.

*** THE VOLUME TRANSPORT IS RECONSTRUCTED AND IS MISSING THE BAROTROPIC
*** CORRECTION.  READ THIS BEFORE QUOTING ANY NUMBER FROM THIS MODULE.

It is built from the POST-STEP state with the model's own face operators
(``min_cell_to_uface`` / ``min_cell_to_vface`` + ``compute_face_masks_3d``),
which is the closest quantity recoverable from a state -- but it is NOT the
transport the model advected with, and the gap is not small.

``ocean/dynamics/ocean_model_latlon_cgrid.py:3547`` forms

    u_corrected = u_3d + delta_U,   delta_U = (Hu_avg - Hu_3d) / H_u_old

a DEPTH-UNIFORM (barotropic) increment that makes the depth-integrated
transport equal the barotropic solver's time-averaged ``Hu_avg`` exactly, and
line 3563 advects tracers with ``h_u_old * u_corrected``.  ``u_corrected`` is
used ONLY there -- it is never written back to the state (it appears at
3547 and 3563 and nowhere else).  So ``state.u`` lacks ``delta_U``, and this
module reconstructs ``h_new * state.u``, not ``h_old * (state.u + delta_U)``.

MEASURED SIZE (eORCA1 tripole, run nemolev_trp_icemelt70_d90, d30/d60/d90):
comparing the section transport against the net convergence implied by the
MEASURED sea-level change, the gap is 0.35-0.61 Sv at 60S/30S/0N/30N and
1.14-1.28 Sv at 60N/66N.  At 66N that is ~100% of the apparent net transport
(1.34 Sv reconstructed vs 0.06 Sv from eta).  The absolute gap is comparable
at all latitudes -- the mode is missing everywhere; it merely looks worse
where the true signal is small.

CONSEQUENCE FOR THE ACCUMULATOR: ``gateway_step`` reads the POST-STEP state,
so running it inside the driver's step loop does NOT recover ``delta_U``.
The in-model accumulator has exactly the same omission as an offline probe.
Fixing it requires the model to expose ``mass_flux_u``/``mass_flux_v`` (or
``Hu_avg``/``Hv_avg``) as a step output; see the tracking issue.

``Hu_avg`` itself is NOT on the state, so the correction cannot be recovered
offline.  Its DIVERGENCE is: the barotropic solver guarantees
``div(Hu_avg) == (eta_old - eta_new)/dt``
(``ocean/dynamics/barotropic_latlon_cgrid.py:1041``), so the NET volume
transport across a CLOSED section is recoverable from the eta tendency even
though the per-face flux is not.  Prefer that for net-volume questions.

Also omitted: the GM ``through_fct`` bolus flux and the adaptive implicit
vertical-advection split.  See ``mass_fluxes_from_state`` for the same caveat
at the point of use.

All functions are pure and JAX-traceable (no host callbacks, stable shapes).

SCOPE LIMIT: ``GatewayAccumulator`` carries the gateway NAMES (Python strings),
so it is NOT a valid ``lax.scan`` carry -- strings are not JAX types.  It is
designed for the HOST step loop that the tripole OMIP driver actually runs
(``--scan-block`` defaults to 0 and the scan path rejects the OMIP forcing
stack).  To accumulate inside a scan, carry the two float arrays alone and
re-attach the names on the host.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Exact unit conversion, not an empirical coefficient.
_M3_S_PER_SV = 1.0e6

# Northern boundary latitude of the Arctic region [deg N].  This is the SAME
# region the offline salt-budget probes integrate over, so the accumulated
# transports are directly comparable to them.
ARCTIC_LAT_DEG = 66.0

# Gateway longitude bins on the lat >= ARCTIC_LAT_DEG boundary, in [-180, 180).
# The bins TILE the circle, so binning the boundary faces by the longitude of
# the cell inside the region partitions the boundary exactly (asserted by
# ``arctic_gateway_transports``).
#
# NOTE: Fram Strait (~79N) and the Barents Sea Opening (20E, 70-76N) are
# INTERIOR to lat > 66N and are therefore NOT boundary gateways of this region.
# They are named here so their absence is not read as an omission.
# The bins MUST TILE [-180, 180) with no gap and no overlap; ``_assert_bins_
# tile`` enforces it at import.  An earlier draft left [-155, -100) uncovered
# (the Beaufort / Canadian mainland sector).  On eORCA1 that sector happens to
# have no wet boundary face at 66N so nothing was dropped in practice, but on
# any other grid those faces would have vanished SILENTLY from every gateway
# while the totals still looked self-consistent.
ARCTIC_GATEWAYS: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = (
    ("bering_pacific", ((-180.0, -140.0), (155.0, 180.0))),
    ("davis_caa", ((-140.0, -45.0),)),
    ("atlantic_nordic", ((-45.0, 70.0),)),
    ("siberian_other", ((70.0, 155.0),)),
)


def _assert_bins_tile(gateways) -> None:
    """Fail at IMPORT if the gateway bins do not tile [-180, 180) exactly."""
    edges = sorted(b for _n, bins in gateways for b in bins)
    if not edges:
        raise ValueError("no gateway bins defined")
    cursor = -180.0
    for lo, hi in edges:
        if lo != cursor:
            raise ValueError(
                f"gateway longitude bins do not tile [-180,180): expected the "
                f"next bin to start at {cursor}, got {lo}. A gap DROPS faces "
                f"silently; an overlap DOUBLE COUNTS them.")
        if hi <= lo:
            raise ValueError(f"empty or inverted gateway bin [{lo}, {hi})")
        cursor = hi
    if cursor != 180.0:
        raise ValueError(
            f"gateway longitude bins stop at {cursor}, not 180.0 — the "
            "remainder of the circle is uncovered and its faces would be "
            "dropped silently.")


_assert_bins_tile(ARCTIC_GATEWAYS)


class BoundaryFaces(NamedTuple):
    """Boundary-face selection + orientation for one region.

    u_sel : (n_lat, n_lon+1) bool  -- selected u-faces
    u_sign: (n_lat, n_lon+1)       -- +1 / -1, positive = INTO the region
    v_sel : (n_lat+1, n_lon) bool
    v_sign: (n_lat+1, n_lon)
    """

    u_sel: jnp.ndarray
    u_sign: jnp.ndarray
    v_sel: jnp.ndarray
    v_sign: jnp.ndarray


class SectionTransport(NamedTuple):
    """Volume [m^3/s] and tracer [tracer-units * m^3/s] transport, + into region."""

    volume: jnp.ndarray
    tracer: jnp.ndarray


def region_boundary_faces(region: jnp.ndarray,
                          *, periodic_x: bool = True) -> BoundaryFaces:
    """Faces whose two adjacent cell centres straddle ``region``.

    Parameters
    ----------
    region : (n_lat, n_lon) bool -- True inside the region (already wet-masked).
    periodic_x : the i direction wraps (true for a global ocean grid).  When
        False the i=0 face is treated as a wall and never selected.

    Returns
    -------
    BoundaryFaces.  The i = n_lon face is the periodic image of i = 0 and is
    NEVER selected, so a periodic boundary is counted once, not twice.
    """
    region = jnp.asarray(region, dtype=bool)
    n_lat, n_lon = region.shape

    # cell i-1 for each i (west neighbour); wrap when periodic
    west = jnp.roll(region, 1, axis=1)
    if not periodic_x:
        west = west.at[:, 0].set(region[:, 0])      # no straddle at the wall
    u_inner = jnp.logical_xor(region, west)
    # +1 when the EAST cell (i) is inside => eastward mfu enters the region
    u_sign_inner = jnp.where(region & ~west, 1.0,
                             jnp.where(west & ~region, -1.0, 0.0))
    u_sel = jnp.zeros((n_lat, n_lon + 1), dtype=bool).at[:, :n_lon].set(u_inner)
    u_sign = jnp.zeros((n_lat, n_lon + 1)).at[:, :n_lon].set(u_sign_inner)

    # v-face j between cell j-1 (south) and cell j (north); j=0 and j=n_lat are
    # domain walls and are never boundary faces of an interior region.
    south = region[:-1, :]
    north = region[1:, :]
    v_inner = jnp.logical_xor(south, north)
    v_sign_inner = jnp.where(north & ~south, 1.0,
                             jnp.where(south & ~north, -1.0, 0.0))
    v_sel = jnp.zeros((n_lat + 1, n_lon), dtype=bool).at[1:n_lat, :].set(v_inner)
    v_sign = jnp.zeros((n_lat + 1, n_lon)).at[1:n_lat, :].set(v_sign_inner)
    return BoundaryFaces(u_sel, u_sign, v_sel, v_sign)


def upwind_face_values(tracer: jnp.ndarray, mfu: jnp.ndarray,
                       mfv: jnp.ndarray, grid=None
                       ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Donor-cell (upwind) tracer values on u- and v-faces.

    Delegates to the CANONICAL ``upwind_cell_to_uface`` / ``upwind_cell_to_vface``
    (``legoesm.grids.operators_latlon_cgrid``) rather than re-deriving the
    donor selection -- those own the periodic wrap and the tripole-fold
    handling, and a local copy would drift from them (repo rule: no duplicate
    numerics).  ``grid`` is forwarded so the v-face helper can apply its fold
    treatment; pass the model grid whenever it is available.
    """
    from legoesm.grids.operators_latlon_cgrid import (
        upwind_cell_to_uface, upwind_cell_to_vface,
    )
    return (upwind_cell_to_uface(tracer, mfu),
            upwind_cell_to_vface(tracer, mfv, grid))


def section_transport(mfu: jnp.ndarray, mfv: jnp.ndarray,
                      dy_u: jnp.ndarray, dx_v: jnp.ndarray,
                      faces: BoundaryFaces,
                      tracer_u: jnp.ndarray | None = None,
                      tracer_v: jnp.ndarray | None = None) -> SectionTransport:
    """Oriented volume + tracer transport through the selected faces.

    POSITIVE = INTO the region (the orientation lives in ``faces``).
    ``mfu``/``mfv`` are thickness-weighted face transports [m^2/s]; the face
    length (``dy_u`` / ``dx_v``) turns them into m^3/s, matching the metric
    ``divergence_cgrid`` uses.  Summed over ALL levels.
    """
    wu = (faces.u_sel.astype(mfu.dtype) * faces.u_sign
          * dy_u.astype(mfu.dtype))[:, :, None]
    wv = (faces.v_sel.astype(mfv.dtype) * faces.v_sign
          * dx_v.astype(mfv.dtype))[:, :, None]
    vol = jnp.sum(wu * mfu) + jnp.sum(wv * mfv)
    if tracer_u is None or tracer_v is None:
        tr = jnp.zeros_like(vol)
    else:
        tr = jnp.sum(wu * mfu * tracer_u) + jnp.sum(wv * mfv * tracer_v)
    return SectionTransport(vol, tr)


def _lon_in_bins(lon_deg: jnp.ndarray,
                 bins: tuple[tuple[float, float], ...]) -> jnp.ndarray:
    """Membership of ``lon_deg`` (in [-180,180)) in a union of half-open bins."""
    out = jnp.zeros_like(lon_deg, dtype=bool)
    for lo, hi in bins:
        out = out | ((lon_deg >= lo) & (lon_deg < hi))
    return out


def gateway_face_masks(faces: BoundaryFaces, lon_deg: jnp.ndarray,
                       gateways=ARCTIC_GATEWAYS
                       ) -> dict[str, BoundaryFaces]:
    """Split ``faces`` into per-gateway subsets by the longitude of the cell
    INSIDE the region (unambiguous — no wrap arithmetic on a face midpoint).

    The bins tile [-180, 180), so the subsets PARTITION the boundary: every
    selected face lands in exactly one gateway.
    """
    lon180 = ((lon_deg + 180.0) % 360.0) - 180.0
    n_lat, n_lon = lon180.shape
    # longitude attributed to each face = the inside cell's longitude
    lon_u_inner = jnp.where(faces.u_sign[:, :n_lon] > 0.0,
                            lon180, jnp.roll(lon180, 1, axis=1))
    lon_u = jnp.zeros((n_lat, n_lon + 1)).at[:, :n_lon].set(lon_u_inner)
    lon_v_inner = jnp.where(faces.v_sign[1:n_lat, :] > 0.0,
                            lon180[1:, :], lon180[:-1, :])
    lon_v = jnp.zeros((n_lat + 1, n_lon)).at[1:n_lat, :].set(lon_v_inner)

    out: dict[str, BoundaryFaces] = {}
    for name, bins in gateways:
        us = faces.u_sel & _lon_in_bins(lon_u, bins)
        vs = faces.v_sel & _lon_in_bins(lon_v, bins)
        out[name] = BoundaryFaces(us, faces.u_sign, vs, faces.v_sign)
    return out


class GatewayAccumulator(NamedTuple):
    """Running SUMS of gateway transports; divide by ``n`` for the time mean.

    ``volume``/``tracer`` are (n_gateway,) arrays ordered as ``names``.
    ``n`` is a plain Python int held on the HOST (the tripole OMIP driver runs
    a Python step loop, so the count never needs to be traced).

    NOT a ``lax.scan`` carry: ``names`` are strings, which are not JAX types.
    """

    names: tuple[str, ...]
    volume: jnp.ndarray
    tracer: jnp.ndarray
    n: int

    @property
    def volume_sv(self) -> jnp.ndarray:
        """Time-mean volume transport per gateway [Sv], + INTO the region."""
        return self.volume / max(self.n, 1) / _M3_S_PER_SV

    @property
    def tracer_mean(self) -> jnp.ndarray:
        """Time-mean tracer transport per gateway [tracer-units * m^3/s]."""
        return self.tracer / max(self.n, 1)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        """{gateway: (volume_Sv, tracer_transport)} time means."""
        vs = self.volume_sv
        ts = self.tracer_mean
        return {nm: (float(vs[i]), float(ts[i]))
                for i, nm in enumerate(self.names)}


def new_gateway_accumulator(names: tuple[str, ...]) -> GatewayAccumulator:
    """Zeroed accumulator in the widest dtype JAX will give us.

    x64 is requested so a long run accumulates in float64, but under ``--fp32``
    the driver deliberately leaves ``jax_enable_x64`` OFF and JAX will silently
    return float32 here.  That is stated rather than claimed away: the
    per-step spatial reductions are then float32 too, and a very long fp32 run
    will lose precision in the running sum.  Check ``acc.volume.dtype`` if that
    matters for your run.
    """
    dt = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    z = jnp.zeros((len(names),), dtype=dt)
    return GatewayAccumulator(tuple(names), z, z, 0)


@jax.jit
def _accumulate_jit(vol_acc, tr_acc, mfu, mfv, dy_u, dx_v,
                    u_sel_stack, u_sign, v_sel_stack, v_sign,
                    tracer_u, tracer_v):
    """One accumulation step for all gateways at once (stable shapes)."""
    wu = (u_sel_stack.astype(mfu.dtype) * u_sign[None, :, :]
          * dy_u.astype(mfu.dtype)[None, :, :])[:, :, :, None]
    wv = (v_sel_stack.astype(mfv.dtype) * v_sign[None, :, :]
          * dx_v.astype(mfv.dtype)[None, :, :])[:, :, :, None]
    vol = (jnp.sum(wu * mfu[None], axis=(1, 2, 3))
           + jnp.sum(wv * mfv[None], axis=(1, 2, 3)))
    tr = (jnp.sum(wu * mfu[None] * tracer_u[None], axis=(1, 2, 3))
          + jnp.sum(wv * mfv[None] * tracer_v[None], axis=(1, 2, 3)))
    return (vol_acc + vol.astype(vol_acc.dtype),
            tr_acc + tr.astype(tr_acc.dtype))


class GatewayStack(NamedTuple):
    """Gateway face masks stacked ONCE for the hot path (static for a run)."""

    names: tuple[str, ...]
    u_sel: jnp.ndarray      # (n_gateway, n_lat, n_lon+1)
    v_sel: jnp.ndarray      # (n_gateway, n_lat+1, n_lon)
    u_sign: jnp.ndarray
    v_sign: jnp.ndarray


def prepare_gateway_stack(gates: dict[str, BoundaryFaces]) -> GatewayStack:
    """Stack the per-gateway masks once, before the step loop.

    Re-stacking every step would allocate and copy full mask arrays in the hot
    path; the selections are static for the whole run.
    """
    names = tuple(gates)
    first = gates[names[0]]
    return GatewayStack(
        names,
        jnp.stack([gates[nm].u_sel for nm in names]),
        jnp.stack([gates[nm].v_sel for nm in names]),
        first.u_sign, first.v_sign)


def accumulate_gateways(acc: GatewayAccumulator, stack: GatewayStack,
                        mfu: jnp.ndarray, mfv: jnp.ndarray,
                        dy_u: jnp.ndarray, dx_v: jnp.ndarray,
                        tracer_u: jnp.ndarray,
                        tracer_v: jnp.ndarray) -> GatewayAccumulator:
    """Add one timestep's transports to ``acc`` (pure; returns a new carry)."""
    if tuple(stack.names) != tuple(acc.names):
        raise ValueError(
            f"gateway stack names {stack.names} do not match the accumulator's "
            f"{acc.names} -- the columns would be silently mis-assigned.")
    n_g = len(acc.names)
    if stack.u_sel.shape[0] != n_g or stack.v_sel.shape[0] != n_g:
        raise ValueError(
            f"gateway stack has {stack.u_sel.shape[0]}/{stack.v_sel.shape[0]} "
            f"selector rows for {n_g} names -- matching names alone does NOT "
            "prove the selector arrays belong to them (codex L8).")
    if stack.u_sel.shape[1:] != mfu.shape[:2] or \
            stack.v_sel.shape[1:] != mfv.shape[:2]:
        raise ValueError(
            f"gateway stack selector shapes {stack.u_sel.shape[1:]}/"
            f"{stack.v_sel.shape[1:]} do not match the flux fields "
            f"{mfu.shape[:2]}/{mfv.shape[:2]}.")
    vol, tr = _accumulate_jit(
        acc.volume, acc.tracer, mfu, mfv, dy_u, dx_v,
        stack.u_sel, stack.u_sign, stack.v_sel, stack.v_sign,
        tracer_u, tracer_v)
    return GatewayAccumulator(acc.names, vol, tr, acc.n + 1)


def mass_fluxes_from_state(state, z_coord, grid, *,
                           min_water_column_m: float | None = None):
    """Thickness-weighted face transports (mfu, mfv) [m^2/s] from a state.

    Built from the model's own operators (``compute_layer_thickness``,
    ``min_cell_to_uface`` / ``min_cell_to_vface``, ``compute_face_masks_3d``).

    WHAT THIS IS NOT
    ----------------
    It is a DIAGNOSTIC RECONSTRUCTION from the POST-STEP state, not the model's
    own advecting transport.  ``_step_impl`` builds its tracer mass flux from
    the PRE-barotropic thickness and then applies the barotropic solver's
    time-averaged ``Hu_avg``/``Hv_avg`` correction, which is not exposed on the
    state and therefore cannot be recovered here.  Expect agreement to the size
    of the barotropic correction, NOT bit-equality; do not describe transports
    built from this as "the model's exact transport".

    Beyond the barotropic correction, the model may also add a GM bolus
    transport (``gm_bolus_advection="through_fct"``) and may rewrite w under
    adaptive implicit vertical advection AFTER the tracer flux is formed;
    neither is exposed on the state.

    PREFER THE STORED FLUX.  If the run set
    ``LatLonCGridOceanConfig.store_mass_flux=True`` (#1442), the state carries
    ``mass_flux_u``/``mass_flux_v`` -- the ACTUAL tracer-advecting flux, with
    the barotropic correction and the GM bolus already in it -- and this
    function returns those unchanged.  Everything said above about
    reconstruction then does not apply.  The reconstruction below is the
    fallback for a state that predates the flag (e.g. an archived snapshot).

    ``min_water_column_m`` is forwarded to ``compute_layer_thickness`` so a
    caller can match the model config's own floor.
    """
    stored_u = getattr(state, "mass_flux_u", None)
    stored_v = getattr(state, "mass_flux_v", None)
    if stored_u is not None and stored_v is not None:
        # Exact: what the model advected with. No reconstruction, no caveat.
        return stored_u.data, stored_v.data

    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d, min_cell_to_uface, min_cell_to_vface,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord,
                                  min_water_column_m=min_water_column_m)
    h_u = min_cell_to_uface(h_k)
    h_v = min_cell_to_vface(h_k, grid)
    # Partial-cell coords carry a 3-D activity mask; a plain z-star coord does
    # NOT have ``is_active`` (a default run would crash here), so mirror the
    # model's non-partial branch and broadcast the 2-D face masks instead.
    act = getattr(z_coord, "is_active", None)
    if act is not None:
        um3, vm3 = compute_face_masks_3d(act, grid)
        um3 = um3.astype(h_u.dtype)
        vm3 = vm3.astype(h_v.dtype)
    else:
        um3 = state.u_mask.data[..., None].astype(h_u.dtype)
        vm3 = state.v_mask.data[..., None].astype(h_v.dtype)
    return state.u.data * h_u * um3, state.v.data * h_v * vm3


# Grids whose C-grid v faces this diagnostic can integrate.  These are the
# ``app_grid_type`` strings the OMIP driver ACTUALLY sets (run_omip_core2.py
# :4640 "tripole", :4771 "latlon"); an earlier draft checked "latlon_bathy",
# which is the CLI value, not the resolved app_grid_type -- so it disabled the
# very grid it meant to support (codex H4).
SUPPORTED_APP_GRIDS = ("tripole", "latlon")


def validate_gateway_lanes(*, use_scan: bool, spmd_persistent: bool,
                           app_grid_type: str) -> None:
    """Reject the run lanes this diagnostic cannot serve.  Raises ValueError.

    Kept OUT of the driver so it is directly testable: the two bugs this
    guards against were both WIRING bugs that 36 green unit tests could not
    see -- a guard placed AFTER the ``use_scan`` early return (so it never
    executed) and a grid-type string that never matched.
    """
    if use_scan:
        raise ValueError(
            "--gateway-transports is incompatible with --scan-block > 0: the "
            "scan path returns before the host step loop, so the accumulator "
            "would never run and the flag would be SILENTLY ignored. Drop "
            "--scan-block (it is off by default).")
    if spmd_persistent:
        raise ValueError(
            "--gateway-transports is not supported with "
            "--spmd-persistent-state: the persistent carrier stores v as the "
            "n_lat-row v_lower field while the diagnostic needs the n_lat+1 "
            "staggered v, so the face shapes disagree.")
    if app_grid_type not in SUPPORTED_APP_GRIDS:
        raise ValueError(
            f"--gateway-transports supports app_grid_type in "
            f"{SUPPORTED_APP_GRIDS}; this run resolved {app_grid_type!r}.")


def setup_gateway_accumulator(lat_deg, lon_deg, land_mask,
                              *, lat_min_deg: float = ARCTIC_LAT_DEG):
    """Build the static face selection + a zeroed accumulator.

    Returns ``(accumulator, stack, boundary_faces)``.  Call ONCE before the
    step loop: the selections are static, so the hot path never re-stacks.
    """
    region = arctic_region_mask(lat_deg, land_mask, lat_min_deg=lat_min_deg)
    faces = region_boundary_faces(region)
    stack = prepare_gateway_stack(gateway_face_masks(faces, lon_deg))
    return new_gateway_accumulator(stack.names), stack, faces


def promote_gateway_geometry(grid, *, metric_convention: str = "exact"):
    """Promote a grid to ``LatLonCGridGeometry`` ONCE, before the step loop.

    ``ensure_geometry``'s own docstring says it "should be called **once** at
    model-construction time ... not per operator call", so calling it inside
    ``gateway_step`` rebuilt every metric array on every timestep AND silently
    defaulted to ``metric_convention="exact"`` even for a ``nemo_isotropic``
    run -- the diagnostic would then integrate over different face lengths
    than the model (codex round-3 finding 1).

    Prefer passing ``model.grid``: the ocean model already ran
    ``ensure_geometry(grid, metric_convention=config.metric_convention)`` in
    its constructor, so ``model.grid`` is promoted with the RIGHT convention
    and this call is a no-op early return.  ``metric_convention`` here only
    matters when a raw ``LatLonGrid`` is passed instead.
    """
    from legoesm.grids.latlon import ensure_geometry

    return ensure_geometry(grid, metric_convention=metric_convention)


def gateway_step(acc: GatewayAccumulator, stack: GatewayStack, state, z_coord,
                 geom, *, min_water_column_m: float | None = None
                 ) -> GatewayAccumulator:
    """Accumulate ONE timestep from a post-step state.  Pure: reads only.

    This is the single entry point the driver calls per step, so a test that
    exercises it is testing what the driver runs.

    ``geom`` MUST already be a promoted ``LatLonCGridGeometry`` (pass
    ``model.grid``, or ``promote_gateway_geometry(grid)`` once before the
    loop).  A raw ``LatLonGrid`` -- what ``_create_setup`` returns for
    ``--grid latlon`` -- has no ``dy_u``/``dx_v`` and is REJECTED rather than
    silently re-promoted per step with a possibly wrong metric convention.
    """
    from legoesm.grids.latlon import LatLonCGridGeometry

    if not isinstance(geom, LatLonCGridGeometry):
        # Attribute presence alone is NOT enough: a duck-typed object with the
        # right names but the wrong staggering or units would silently
        # mis-integrate every section (codex round-4 finding 2).  Accept a
        # non-LatLonCGridGeometry only if it carries BOTH face metrics with
        # the exact C-grid shapes implied by the state, which is what pins the
        # staggering.
        for _attr in ("dy_u", "dx_v"):
            if not hasattr(geom, _attr):
                raise TypeError(
                    f"gateway_step needs a promoted LatLonCGridGeometry "
                    f"(missing {_attr!r}); pass model.grid, or call "
                    "promote_gateway_geometry(grid, metric_convention=...) "
                    "ONCE before the step loop. Promoting per step rebuilds "
                    "every metric array and defaults the convention to "
                    "'exact'.")
        n_lat, n_lon = jnp.shape(state.S.data)[0], jnp.shape(state.S.data)[1]
        for _attr, _want in (("dy_u", (n_lat, n_lon + 1)),
                             ("dx_v", (n_lat + 1, n_lon))):
            _got = tuple(jnp.shape(getattr(geom, _attr)))
            if _got != _want:
                raise TypeError(
                    f"gateway_step: {type(geom).__name__}.{_attr} has shape "
                    f"{_got}, expected {_want} for a C-grid with tracer shape "
                    f"({n_lat}, {n_lon}). The face metrics are mis-staggered; "
                    "pass model.grid or promote_gateway_geometry(grid).")
    mfu, mfv = mass_fluxes_from_state(
        state, z_coord, geom, min_water_column_m=min_water_column_m)
    tr_u, tr_v = upwind_face_values(state.S.data, mfu, mfv, geom)
    return accumulate_gateways(acc, stack, mfu, mfv,
                               jnp.asarray(geom.dy_u), jnp.asarray(geom.dx_v),
                               tr_u, tr_v)


def arctic_region_mask(lat_deg: jnp.ndarray, land_mask: jnp.ndarray,
                       *, lat_min_deg: float = ARCTIC_LAT_DEG) -> jnp.ndarray:
    """Wet cells north of ``lat_min_deg`` (the salt-budget region)."""
    return (jnp.asarray(lat_deg) >= lat_min_deg) & (jnp.asarray(land_mask) > 0.5)


__all__ = [
    "ARCTIC_GATEWAYS",
    "ARCTIC_LAT_DEG",
    "BoundaryFaces",
    "GatewayAccumulator",
    "SectionTransport",
    "accumulate_gateways",
    "SUPPORTED_APP_GRIDS",
    "arctic_region_mask",
    "GatewayStack",
    "gateway_face_masks",
    "mass_fluxes_from_state",
    "prepare_gateway_stack",
    "new_gateway_accumulator",
    "gateway_step",
    "promote_gateway_geometry",
    "region_boundary_faces",
    "setup_gateway_accumulator",
    "validate_gateway_lanes",
    "section_transport",
    "upwind_face_values",
]
