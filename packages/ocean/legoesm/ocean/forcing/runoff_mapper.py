"""Route river runoff from dry cells to wet recipients, conserving mass.

Why this exists
---------------
Runoff products place discharge at river MOUTHS resolved on their own grid.
JRA55-do's ``friver`` is on a 0.25 deg river grid; an ocean model at 1 deg
swallows those coastlines, so a large share of the discharge lands on cells
the model calls land.  A model that simply masks dry cells then **discards**
that water instead of delivering it — measured on JRA55-do v1.4.0, ~71 % of
the global total, and 68 % of the total rides on ~100 source cells, so this
is not a roundoff (see ``docs/ocean/fidelity/fesom2_gap_analysis.md``).

FESOM2 solves it with ``use_runoff_mapper`` / ``runoff_radius`` in
``namelist.forcing``; NEMO and MOM6 have equivalents.  legoESM had nothing,
which left the freshwater budget of every JRA55-do-forced run open.

What this does
--------------
The mapping is a pure function of the land mask and the grid, so it is built
ONCE on the host (:func:`build_runoff_map`) and applied per step on device
(:func:`apply_runoff_map`) as a single scatter-add — jit-safe, fixed shapes,
and differentiable in the runoff field (the map itself is static index data).

Two schemes, both mass-conserving:

``"nearest"`` (default)
    Each dry donor's discharge goes to its single nearest wet cell within
    ``radius_m``.  Exactly conservative, and the cheapest thing that is
    correct.
``"spread"``
    Each dry donor's discharge is divided among ALL wet cells within
    ``radius_m``, weighted by their area, which is closer to what FESOM's
    mapper does and avoids a salinity crater where a large river lands in one
    coarse cell.

Conservation convention
-----------------------
Runoff is a flux DENSITY [kg m-2 s-1].  Moving the contents of a donor cell
of area ``A_src`` into a recipient of area ``A_dst`` therefore scales by
``A_src / A_dst`` so that ``sum(F * A)`` is unchanged.  :func:`build_runoff_map`
reports the share it could not place (no wet cell within ``radius_m``) rather
than silently dropping it.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants

#: Routing schemes :func:`build_runoff_map` accepts.  A typo must raise, not
#: silently select a different physical treatment (CLAUDE.md dispatch rule).
VALID_RUNOFF_SCHEMES: tuple[str, ...] = ("nearest", "spread")


class RunoffMap(NamedTuple):
    """Static routing plan from dry donor cells to wet recipients.

    Index arrays are FLAT into the grid's ravelled spatial shape, so the same
    structure serves lat-lon ``(n_lat, n_lon)`` and unstructured ``(nCells,)``
    grids.

    Attributes
    ----------
    src_idx, dst_idx : int32 (n_pairs,)
        Donor and recipient flat cell indices, one entry per (donor,
        recipient) pair.  ``"nearest"`` yields one pair per donor;
        ``"spread"`` yields one per donor-recipient pair.
    weight : float64 (n_pairs,)
        ``fraction_of_donor * A_src / A_dst`` — the factor converting the
        donor's flux density into the recipient's.  Fractions over a donor
        sum to 1.
    wet_mask_flat : bool (n_cells,)
        True where the cell is ocean.  Applied on device so a donor's own
        cell is zeroed after its water is moved.
    shape : tuple[int, ...]
        Spatial shape the flat indices refer to.
    unrouted_fraction : float
        Share of the field's global discharge that found NO wet cell within
        ``radius_m``, evaluated on the reference field passed to the builder
        (0.0 when none was passed).  Diagnostic; it is NOT silently fixed.
    """
    src_idx: jnp.ndarray
    dst_idx: jnp.ndarray
    weight: jnp.ndarray
    wet_mask_flat: jnp.ndarray
    shape: tuple
    unrouted_fraction: float


def _unit_sphere_xyz(lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """Unit-sphere Cartesian coordinates — wrap- and pole-safe.

    Chord distance on the unit sphere is monotonic in great-circle distance,
    so a Euclidean KD-tree search over these gives the correct neighbour
    ordering without any dateline or polar special-casing.
    """
    cl = np.cos(lat_rad)
    return np.stack([cl * np.cos(lon_rad), cl * np.sin(lon_rad),
                     np.sin(lat_rad)], axis=-1)


def _chord_from_arc(radius_m: float) -> float:
    """Unit-sphere search bound for a great-circle arc of ``radius_m``.

    Without a clamp, a radius beyond ``pi * R_earth`` runs past the peak of
    ``sin`` and the chord SHRINKS again — so asking for a bigger search radius
    would silently find FEWER neighbours, and a radius near ``2 * pi * R``
    would find none at all.

    At or beyond the antipode the answer is ``inf``, not the chord ``2.0``.
    A cell exactly opposite the donor sits AT chord 2.0, and whether it is
    inside a bound of exactly 2.0 then depends on the last bit of the
    coordinates — float32 grid metrics put it outside, float64 inside.  An
    unbounded search says "the whole sphere" without that knife edge.
    """
    arc = float(radius_m) / constants.R_earth
    if arc >= np.pi:
        return np.inf
    return 2.0 * np.sin(0.5 * arc)


def build_runoff_map(
    ocean_mask,
    cell_area,
    lat_rad,
    lon_rad,
    *,
    radius_m: float = 500.0e3,
    scheme: str = "nearest",
    reference_runoff=None,
) -> RunoffMap:
    """Build the static dry-to-wet routing plan.

    Parameters
    ----------
    ocean_mask : array, grid spatial shape
        True/1 where ocean.  Cells that are not ocean are donors.
    cell_area : array, same shape
        Cell area [m^2].  Needed because runoff is a flux density: the
        density changes when water moves between cells of different size.
    lat_rad, lon_rad : arrays broadcastable to the grid spatial shape
        Cell-centre coordinates in RADIANS.
    radius_m : float
        Great-circle search radius [m].  Default 500 km, matching FESOM2's
        ``runoff_radius``.
    scheme : {"nearest", "spread"}
        See the module docstring.  Unknown values raise.
    reference_runoff : array or None
        A representative runoff field used ONLY to report
        ``unrouted_fraction``.  Pass one (e.g. the first forcing record) to
        learn how much discharge the radius fails to place.

    Returns
    -------
    RunoffMap
    """
    if scheme not in VALID_RUNOFF_SCHEMES:
        raise ValueError(
            f"Unknown runoff routing scheme {scheme!r}; expected one of "
            f"{VALID_RUNOFF_SCHEMES}. 'nearest' sends each dry cell's "
            "discharge to its single closest wet cell; 'spread' divides it "
            "over every wet cell within radius_m."
        )
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError(
            f"radius_m must be finite and > 0; got {radius_m}.")

    from scipy.spatial import cKDTree

    wet = np.asarray(ocean_mask) > 0.5
    shape = wet.shape
    area = np.broadcast_to(np.asarray(cell_area, dtype=np.float64),
                           shape).ravel()
    latf = np.broadcast_to(np.asarray(lat_rad, dtype=np.float64),
                           shape).ravel()
    lonf = np.broadcast_to(np.asarray(lon_rad, dtype=np.float64),
                           shape).ravel()
    wet_flat = wet.ravel()
    if not wet_flat.any():
        raise ValueError(
            "build_runoff_map: the ocean mask has no wet cells; there is "
            "nowhere to route runoff to."
        )

    dry_idx = np.flatnonzero(~wet_flat)
    wet_idx = np.flatnonzero(wet_flat)
    xyz_wet = _unit_sphere_xyz(latf[wet_idx], lonf[wet_idx])
    xyz_dry = _unit_sphere_xyz(latf[dry_idx], lonf[dry_idx])
    tree = cKDTree(xyz_wet)
    cutoff = _chord_from_arc(float(radius_m))

    src_list: list[np.ndarray] = []
    dst_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    unrouted = np.zeros(dry_idx.size, dtype=bool)

    if scheme == "nearest":
        dist, nn = tree.query(xyz_dry, k=1,
                              distance_upper_bound=cutoff)
        # cKDTree marks "no neighbour within the bound" with index == n_wet.
        found = nn < wet_idx.size
        unrouted = ~found
        src = dry_idx[found]
        dst = wet_idx[nn[found]]
        src_list.append(src)
        dst_list.append(dst)
        w_list.append(area[src] / area[dst])
    else:  # "spread"
        neighbours = tree.query_ball_point(xyz_dry, r=cutoff)
        for i, nb in enumerate(neighbours):
            if not nb:
                unrouted[i] = True
                continue
            nb = np.asarray(nb, dtype=np.int64)
            recip = wet_idx[nb]
            a_recip = area[recip]
            frac = a_recip / a_recip.sum()      # area-weighted share
            s = dry_idx[i]
            src_list.append(np.full(recip.size, s, dtype=np.int64))
            dst_list.append(recip)
            w_list.append(frac * area[s] / a_recip)

    src_idx = (np.concatenate(src_list) if src_list
               else np.zeros(0, dtype=np.int64))
    dst_idx = (np.concatenate(dst_list) if dst_list
               else np.zeros(0, dtype=np.int64))
    weight = (np.concatenate(w_list) if w_list
              else np.zeros(0, dtype=np.float64))

    unrouted_fraction = 0.0
    if reference_runoff is not None:
        ref = np.broadcast_to(np.asarray(reference_runoff, dtype=np.float64),
                              shape).ravel()
        disc = ref * area
        total = float(disc.sum())
        if total > 0.0:
            unrouted_fraction = float(
                disc[dry_idx[unrouted]].sum() / total)

    return RunoffMap(
        src_idx=jnp.asarray(src_idx, dtype=jnp.int32),
        dst_idx=jnp.asarray(dst_idx, dtype=jnp.int32),
        weight=jnp.asarray(weight),
        wet_mask_flat=jnp.asarray(wet_flat),
        shape=tuple(shape),
        unrouted_fraction=unrouted_fraction,
    )


def apply_runoff_map(runoff, rmap: RunoffMap):
    """Route ``runoff`` [kg m-2 s-1] onto wet cells.

    Pure and jit-safe: a gather, a scatter-add and a mask, all at fixed
    shapes.  Differentiable in ``runoff`` (the map is static index data), so
    it composes with the training paths.

    A wet cell keeps its own runoff and gains every donor routed to it; a dry
    cell is left at zero.  ``sum(runoff * area)`` is unchanged except for the
    share the builder reported as ``unrouted_fraction``.
    """
    flat = jnp.reshape(runoff, (-1,))
    if flat.shape[0] != rmap.wet_mask_flat.shape[0]:
        raise ValueError(
            f"apply_runoff_map: runoff has {flat.shape[0]} cells but the map "
            f"was built for {rmap.wet_mask_flat.shape[0]} "
            f"(shape {rmap.shape}). Rebuild the map for this grid."
        )
    kept = jnp.where(rmap.wet_mask_flat, flat, 0.0)
    moved = kept.at[rmap.dst_idx].add(
        flat[rmap.src_idx] * rmap.weight.astype(flat.dtype)
    )
    return jnp.reshape(moved, runoff.shape)
