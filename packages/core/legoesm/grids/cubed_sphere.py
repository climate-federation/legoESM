"""Cubed-sphere grid for legoESM.

Implements a gnomonic EQUIANGULAR cubed-sphere grid with 6 faces (cell-edge
coordinates equally spaced in ANGLE alpha in [-pi/4, pi/4], projected via
tan(alpha) — FV3's gnomonic_angl / grid_type=2).  NOTE: GFDL FV3's *operational*
grid is gnomonic_ed (grid_type=0), a different (more cell-uniform) gnomonic
variant — see fv_grid_utils.F90:gnomonic_ed.  Same family + same spherical-excess
cell areas, but the cell-corner distribution differs from FV3's operational grid.
Each face is an N x N grid of cells. The grid uses an A-grid (collocated)
staggering for the shallow-water milestone, with all variables at cell centers.

The cubed-sphere maps 6 faces of a cube onto the sphere via gnomonic
(central) projection. This gives quasi-uniform resolution with no polar
singularity, and regular 2D arrays on each face — ideal for JAX.

References
----------
- Ronchi, Iacono, Paolucci (1996): The "Cubed Sphere"
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- Nair, Thomas, Loft (2005): A Discontinuous Galerkin Transport Scheme on the Cubed Sphere
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.halo import (
    compute_padded_angle,
    compute_padded_half_metrics,
    compute_halo_interp_offsets,
    compute_halo_interp_offsets_h2,
    compute_halo_interp_offsets_h3,
)


class CubedSphereGrid(NamedTuple):
    """Cubed-sphere grid data structure.

    All arrays have shape (6, n, n) where 6 = number of faces,
    except padded arrays which are (6, n+2, n+2).
    Registered as a JAX pytree via NamedTuple.

    Attributes
    ----------
    n : int
        Number of cells per face edge. Total cells = 6 * n * n.
    radius : float
        Sphere radius [m].
    lon : jax.Array
        Longitude at cell centers [rad], shape (6, n, n).
    lat : jax.Array
        Latitude at cell centers [rad], shape (6, n, n).
    area : jax.Array
        Cell areas [m^2], shape (6, n, n).
    dx : jax.Array
        Cell width in x-direction [m], shape (6, n, n).
    dy : jax.Array
        Cell width in y-direction [m], shape (6, n, n).
    f : jax.Array
        Coriolis parameter at cell centers [1/s], shape (6, n, n).
    cos_lat : jax.Array
        Cosine of latitude, shape (6, n, n).
    sin_lat : jax.Array
        Sine of latitude, shape (6, n, n).
    angle : jax.Array
        Grid rotation angle relative to east [rad], shape (6, n, n).
    x_cart : jax.Array
        Cartesian x-coordinate on unit sphere, shape (6, n, n).
    y_cart : jax.Array
        Cartesian y-coordinate on unit sphere, shape (6, n, n).
    z_cart : jax.Array
        Cartesian z-coordinate on unit sphere, shape (6, n, n).
    angle_padded : jax.Array
        Grid rotation angle on extended grid [rad], shape (6, n+2, n+2).
        Used by vector halo exchange to correctly rotate velocity
        components at face boundaries.
    cos_angle : jax.Array
        Cosine of grid angle, shape (6, n, n). Precomputed for
        vector halo exchange.
    sin_angle : jax.Array
        Sine of grid angle, shape (6, n, n). Precomputed for
        vector halo exchange.
    cos_angle_padded : jax.Array
        Cosine of padded grid angle, shape (6, n+2, n+2).
    sin_angle_padded : jax.Array
        Sine of padded grid angle, shape (6, n+2, n+2).
    hx_ext : jax.Array
        Half dx extrapolated to halo, shape (6, n+2, n+2).
        Used in divergence computation.
    hy_ext : jax.Array
        Half dy extrapolated to halo, shape (6, n+2, n+2).
        Used in divergence computation.
    cos_angle_padded_h2 : jax.Array
        Cosine of padded grid angle for halo=2, shape (6, n+4, n+4).
    sin_angle_padded_h2 : jax.Array
        Sine of padded grid angle for halo=2, shape (6, n+4, n+4).
    hx_ext_h2 : jax.Array
        Half dx on halo=2 extended grid, shape (6, n+4, n+4).
    hy_ext_h2 : jax.Array
        Half dy on halo=2 extended grid, shape (6, n+4, n+4).
    halo_interp_offsets_h2 : jax.Array
        Interpolation offsets for halo=2 exchange, shape (6, 4, 2, n).
    halo_interp_offsets_h3 : jax.Array
        Interpolation offsets for halo=3 exchange, shape (6, 4, 3, n).
        Iter-532: precomputed for the iter-496..501 ng=3 halo
        extension that supports the FB-chain stability work
        (review-doc item #2).
    cos_angle_padded_h3 : jax.Array
        Cosine of padded grid angle for halo=3, shape (6, n+6, n+6).
        Iter-595: added for the ng=3 vector halo round-trip.
    sin_angle_padded_h3 : jax.Array
        Sine of padded grid angle for halo=3, shape (6, n+6, n+6).
    hx_ext_h3 : jax.Array
        Half dx on halo=3 extended grid, shape (6, n+6, n+6).
    hy_ext_h3 : jax.Array
        Half dy on halo=3 extended grid, shape (6, n+6, n+6).
    duogrid : DuoGridData or None
        Duo-Grid kinked-to-extended remapping data. When not None,
        pad_halo applies the Duo-Grid remap instead of interp_offsets.
    gnomonic_form : str
        Static grid provenance: which gnomonic node distribution built this
        grid — ``"equiangular"`` (legoESM's historical default, the FV3
        gnomonic_angl family) or ``"ed"`` (FV3's operational gnomonic_ed).
        Stored as pytree aux_data (never a traced leaf, never inferred from
        dx/dy aspect ratios).  Set by ``create_cubed_sphere``; immutable.
    fv3_grid_type : int
        Static grid provenance: the FV3 ``grid_type`` this distribution
        corresponds to (0 = gnomonic_ed, the FV3 default; 2 = the
        equiangular gnomonic_angl family).  Aux_data like
        ``gnomonic_form``.
    """
    n: int
    radius: float
    lon: jax.Array
    lat: jax.Array
    area: jax.Array
    dx: jax.Array
    dy: jax.Array
    f: jax.Array
    cos_lat: jax.Array
    sin_lat: jax.Array
    angle: jax.Array
    x_cart: jax.Array
    y_cart: jax.Array
    z_cart: jax.Array
    angle_padded: jax.Array
    cos_angle: jax.Array
    sin_angle: jax.Array
    cos_angle_padded: jax.Array
    sin_angle_padded: jax.Array
    hx_ext: jax.Array
    hy_ext: jax.Array
    halo_interp_offsets: jax.Array
    cos_angle_padded_h2: jax.Array
    sin_angle_padded_h2: jax.Array
    hx_ext_h2: jax.Array
    hy_ext_h2: jax.Array
    halo_interp_offsets_h2: jax.Array
    halo_interp_offsets_h3: jax.Array
    cos_angle_padded_h3: jax.Array
    sin_angle_padded_h3: jax.Array
    hx_ext_h3: jax.Array
    hy_ext_h3: jax.Array
    duogrid: object  # DuoGridData | None — use object to avoid circular import
    gnomonic_form: str = "equiangular"  # static provenance (pytree aux_data)
    fv3_grid_type: int = 2              # static provenance (pytree aux_data)

    @property
    def n_cells(self) -> int:
        return 6 * self.n * self.n

    @property
    def total_area(self) -> jax.Array:
        return jnp.sum(self.area)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (6, self.n, self.n)

    @property
    def resolution_km(self) -> float:
        """Approximate resolution in kilometers.

        Each cube face spans π/2 radians, so the nominal grid spacing
        is (π/2)*R/n.
        """
        return (jnp.pi / 2) * self.radius / (self.n * 1000.0)

    @property
    def bounded_domain(self) -> bool:
        """Iter-865b: Fortran-faithful ``bounded_domain`` flag per
        ``fv_arrays.F90:1512``: ``bounded_domain = (regional .or.
        nested .or. duogrid)``.  In legoESM:
        - duogrid: ``self.duogrid is not None``.
        - regional / nested: a single-face panel (``self.lat.shape[0]
          == 1``; see ``create_cubed_sphere_panel`` and the
          ``data.shape[0] == 1`` branch of ``pad_halo`` which applies
          Neumann wall BCs instead of inter-face halo exchange).

        Operators with legacy edge-handling fallbacks should bypass
        them when ``bounded_domain`` is True so the duogrid /
        regional Fortran-faithful path is used uniformly across the
        codebase.  Iter-865 originally hardcoded the gate to
        ``self.duogrid is None``; iter-865b exposes the proper
        bounded-domain abstraction so future regional/nested support
        gates the same way.

        **Cubed-sphere MPI face-scatter:** a rank that owns a SINGLE global
        cube face (``--cs-mpi-scatter`` with 6 ranks) also has
        ``lat.shape[0] == 1``, but it is NOT a regional/nested panel — its
        cross-face seams are filled by MPI halo exchange (the ``pad_halo``
        wall-BC branch is itself skipped when ``_halo_backend == 'mpi'``), so
        ``bounded_domain`` must be False there or operators would wrongly skip
        global-cube edge/corner handling.  Detect this the same way ``pad_halo``
        does: the MPI backend is active and the active topology is the
        face-only cube topology (carries ``local_face_ids``).
        """
        if self.duogrid is not None:
            return True
        if self.lat.shape[0] == 1:
            from legoesm.grids.halo import get_halo_backend, get_mpi_topology

            if get_halo_backend() == "mpi":
                topo = get_mpi_topology()
                if topo is not None and hasattr(topo, "local_face_ids"):
                    # Rank-local scattered global-cube face, not a regional panel.
                    return False
            return True
        return False

    # ------------------------------------------------------------------
    # GridProtocol properties
    # ------------------------------------------------------------------

    @property
    def grid_lat(self) -> jax.Array:
        return self.lat

    @property
    def grid_lon(self) -> jax.Array:
        return self.lon

    @property
    def grid_area(self) -> jax.Array:
        return self.area

    @property
    def grid_total_area(self):
        return jnp.sum(self.area)

    @property
    def grid_coriolis(self) -> jax.Array:
        return self.f

    @property
    def grid_radius(self) -> float:
        return self.radius

    @property
    def grid_n_columns(self) -> int:
        return 6 * self.n * self.n

    @property
    def grid_shape_2d(self) -> tuple[int, ...]:
        return (6, self.n, self.n)

    def to_columns(self, field):
        extra = field.shape[3:]
        return field.reshape(6 * self.n * self.n, *extra)

    def from_columns(self, cols):
        extra = cols.shape[1:]
        return cols.reshape(6, self.n, self.n, *extra)


# Number of trailing provenance fields excluded from the pytree leaves.
_GRID_PROVENANCE_FIELDS = ("gnomonic_form", "fv3_grid_type")


def _cubed_sphere_grid_flatten(g: "CubedSphereGrid"):
    n_prov = len(_GRID_PROVENANCE_FIELDS)
    return tuple(g[:-n_prov]), tuple(g[-n_prov:])


def _cubed_sphere_grid_unflatten(aux, children) -> "CubedSphereGrid":
    return CubedSphereGrid(*children, *aux)


# Explicit registration overrides JAX's built-in NamedTuple handling so the
# string/int grid provenance travels as STATIC aux_data instead of pytree
# leaves: a str leaf breaks jit/grad tracing (the iter72 finding that forced
# the old dx/dy aspect-ratio inference), while aux_data participates in the
# treedef, so grids of different gnomonic form can never be silently
# substituted for one another under jit.  All pre-existing fields remain
# leaves — flatten order and content are unchanged for them.
jax.tree_util.register_pytree_node(
    CubedSphereGrid, _cubed_sphere_grid_flatten, _cubed_sphere_grid_unflatten,
)


def create_cubed_sphere(
    n: int,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
    use_duogrid: bool = False,
    k2e_nord: int = 2,
    duogrid_ng: int | None = None,
    stretch_fac: float = 1.0,
    target_lon: float = 0.0,
    target_lat: float = -0.5 * 3.141592653589793,  # -π/2 = no rotation
    do_cube_transform: bool = False,
    shift_fac: float = 0.0,
    gnomonic: str = "equiangular",
) -> CubedSphereGrid:
    """Create a cubed-sphere grid.

    Parameters
    ----------
    n : int
        Number of cells per face edge. Common values:
        C48 (~200km), C96 (~100km), C192 (~50km), C384 (~25km).
    radius : float
        Sphere radius in meters. Default: Earth radius.
    omega : float
        Planetary rotation rate [rad/s]. Default: Earth rotation rate.
        Scale for small-Earth experiments.
    gnomonic : str, default "equiangular"
        Grid-line distribution.  ``"equiangular"`` = legoESM's historical
        gnomonic_angl (FV3 grid_type=2); ``"ed"`` = FV3's OPERATIONAL
        gnomonic_ed (grid_type=0, equal-great-circle edges, near-uniform cells,
        max aspect 1.06).  The ``"ed"`` path swaps in the ``*_ed`` metric/halo
        builders (centres, padded angle/half-metrics, areas, halo-interp
        offsets) and is incompatible with the Schmidt/shift transforms.

    Returns
    -------
    CubedSphereGrid
        The grid with all metric terms computed.
    """
    if gnomonic == "equiangular":
        _lonlat, _angle_b, _half_b, _area_b = (
            _compute_gnomonic_lonlat, compute_padded_angle,
            compute_padded_half_metrics, _compute_exact_cell_areas)
        _off_b, _off_h2_b, _off_h3_b = (
            compute_halo_interp_offsets, compute_halo_interp_offsets_h2,
            compute_halo_interp_offsets_h3)
    elif gnomonic == "ed":
        from legoesm.grids.halo import (
            compute_halo_interp_offsets_ed, compute_halo_interp_offsets_ed_h2,
            compute_halo_interp_offsets_ed_h3)
        _lonlat, _angle_b, _half_b, _area_b = (
            _compute_gnomonic_ed_lonlat, compute_padded_angle_ed,
            compute_padded_half_metrics_ed, _compute_exact_cell_areas_ed)
        _off_b, _off_h2_b, _off_h3_b = (
            compute_halo_interp_offsets_ed, compute_halo_interp_offsets_ed_h2,
            compute_halo_interp_offsets_ed_h3)
    else:
        raise ValueError(
            f"gnomonic must be 'equiangular' or 'ed', got {gnomonic!r}")

    # Compute gnomonic coordinates on each face
    lon, lat = _lonlat(n)

    # FV3_3D iter 586/589: optional Schmidt stretching.
    apply_schmidt = (
        abs(stretch_fac - 1.0) > 1e-5
        or target_lat > -0.5 * jnp.pi + 1e-5
    )
    if gnomonic == "ed" and (apply_schmidt or shift_fac > 1e-4):
        raise ValueError(
            "gnomonic='ed' is incompatible with Schmidt/shift transforms "
            "(the *_ed padded-metric builders do not apply them).")
    if apply_schmidt:
        if do_cube_transform:
            # FV3 cube_transform (fv_grid_utils.F90:920-980)
            lon, lat = cube_transform(
                lon, lat, stretch_fac, target_lon, target_lat,
            )
        else:
            # FV3 direct_transform / do_schmidt (fv_grid_utils.F90:870-917)
            lon, lat = schmidt_transform(
                lon, lat, stretch_fac, target_lon, target_lat,
            )

    # FV3_3D iter 591: shift_fac longitude shift (FV3 fv_grid_tools.F90:662-663).
    # Only applied when NOT using Schmidt/cube_transform (gated in FV3).
    # FV3 default shift_fac=18 → west-shift by π/18 = 10° (away from Japan).
    if shift_fac > 1e-4 and not apply_schmidt:
        lon = lon - jnp.pi / shift_fac
        lon = jnp.where(lon < 0.0, lon + 2.0 * jnp.pi, lon)

    # Cartesian coordinates on unit sphere
    cos_lat = jnp.cos(lat)
    sin_lat = jnp.sin(lat)
    cos_lon = jnp.cos(lon)
    sin_lon = jnp.sin(lon)

    x_cart = cos_lat * cos_lon
    y_cart = cos_lat * sin_lon
    z_cart = sin_lat

    # Coriolis parameter
    f = 2.0 * omega * sin_lat

    # Compute padded quantities from gnomonic extension (smooth across
    # the interior-halo boundary).  All metrics and angles derive from a
    # single source — the extended gnomonic grid — so there is no seam
    # discontinuity at cube-face edges.  The O(Δα⁴) accuracy difference
    # vs pad_halo-based interior values is well below the O(Δα²)
    # truncation error of the 2nd-order stencils.
    angle_padded = _angle_b(n)
    hx_ext, hy_ext = _half_b(n, radius)

    # Extract interior from padded arrays (no override — single source)
    angle = angle_padded[:, 1:-1, 1:-1]
    dx = 2.0 * hx_ext[:, 1:-1, 1:-1]
    dy = 2.0 * hy_ext[:, 1:-1, 1:-1]
    area = _area_b(n, radius)

    # Precompute trig of grid angle for vector halo exchange
    cos_angle_val = jnp.cos(angle)
    sin_angle_val = jnp.sin(angle)
    cos_angle_padded_val = jnp.cos(angle_padded)
    sin_angle_padded_val = jnp.sin(angle_padded)

    # Halo interpolation offsets for corrected cross-face exchange
    halo_offsets = _off_b(n)

    # halo=2 quantities for higher-order reconstruction (PPM, WENO5)
    angle_padded_h2 = _angle_b(n, halo=2)
    hx_ext_h2, hy_ext_h2 = _half_b(n, radius, halo=2)
    cos_angle_padded_h2_val = jnp.cos(angle_padded_h2)
    sin_angle_padded_h2_val = jnp.sin(angle_padded_h2)
    halo_offsets_h2 = _off_h2_b(n)

    # halo=3 quantities for the iter-496..501 ng=3 halo extension
    # (FB-chain stability prerequisite, review-doc item #2).
    halo_offsets_h3 = _off_h3_b(n)
    # Iter-595: add grid-angle + half-metrics at halo=3 so the vector
    # halo round-trip has the padded-angle reference needed to enable
    # `pad_halo_vector(halo=3)` on the non-MPI backend.
    angle_padded_h3 = _angle_b(n, halo=3)
    hx_ext_h3, hy_ext_h3 = _half_b(n, radius, halo=3)
    cos_angle_padded_h3_val = jnp.cos(angle_padded_h3)
    sin_angle_padded_h3_val = jnp.sin(angle_padded_h3)

    # Optional: Duo-Grid kinked-to-extended remapping data.
    # Duo-Grid requires ng >= 2 so both halo depths used by the FV3
    # d2a2c_vect and PPM transport paths are remapped. Since ng <= n//2,
    # this means n >= 4. For n < 4, silently skip — these toy grids are
    # too small for face-boundary artifacts to be meaningful.
    duogrid = None
    if use_duogrid and n >= 4:
        # Default ng: min(3, n//2) — FV3 uses 3 at production resolutions,
        # but small test grids need a smaller halo to fit the stencil.
        ng = duogrid_ng if duogrid_ng is not None else min(3, n // 2)
        if gnomonic == "ed":
            # Phase-3b: ED grids pair with the ED-NATIVE duo-grid tables
            # (oracle-pinned k2e + ED gnomonic extension).  The legacy
            # builder's equiangular extension lines are the wrong
            # interpolants on ED (~4e-2 coefficient error) — this is a
            # correctness pairing, not a tuned-solver discretization choice,
            # so it is not gated.  Requires the FV3 upstream k2e_nord=4.
            from legoesm.grids.fv3_native_halos import (
                create_fv3_native_duogrid_data,
            )
            # k2e_nord is pinned to the upstream duo-grid default (4); the
            # 2-point option is a legacy-equiangular knob with no FV3
            # counterpart, so a lower request is coerced up, never down.
            duogrid = create_fv3_native_duogrid_data(n, ng=ng, k2e_nord=4)
        else:
            from legoesm.grids.duogrid import create_duogrid_data
            duogrid = create_duogrid_data(n, radius=radius, ng=ng,
                                          k2e_nord=k2e_nord)

    # Grid arrays use the storage dtype from the precision policy.
    # Defaults to float32 for backward compatibility.
    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            _dt = get_policy().storage
        except Exception:
            _dt = jnp.float32
    else:
        _dt = dtype
    grid = CubedSphereGrid(
        n=n,
        radius=radius,
        lon=lon.astype(_dt),
        lat=lat.astype(_dt),
        area=area.astype(_dt),
        dx=dx.astype(_dt),
        dy=dy.astype(_dt),
        f=f.astype(_dt),
        cos_lat=cos_lat.astype(_dt),
        sin_lat=sin_lat.astype(_dt),
        angle=angle.astype(_dt),
        x_cart=x_cart.astype(_dt),
        y_cart=y_cart.astype(_dt),
        z_cart=z_cart.astype(_dt),
        angle_padded=angle_padded.astype(_dt),
        cos_angle=cos_angle_val.astype(_dt),
        sin_angle=sin_angle_val.astype(_dt),
        cos_angle_padded=cos_angle_padded_val.astype(_dt),
        sin_angle_padded=sin_angle_padded_val.astype(_dt),
        hx_ext=hx_ext.astype(_dt),
        hy_ext=hy_ext.astype(_dt),
        halo_interp_offsets=halo_offsets.astype(_dt),
        cos_angle_padded_h2=cos_angle_padded_h2_val.astype(_dt),
        sin_angle_padded_h2=sin_angle_padded_h2_val.astype(_dt),
        hx_ext_h2=hx_ext_h2.astype(_dt),
        hy_ext_h2=hy_ext_h2.astype(_dt),
        halo_interp_offsets_h2=halo_offsets_h2.astype(_dt),
        halo_interp_offsets_h3=halo_offsets_h3.astype(_dt),
        cos_angle_padded_h3=cos_angle_padded_h3_val.astype(_dt),
        sin_angle_padded_h3=sin_angle_padded_h3_val.astype(_dt),
        hx_ext_h3=hx_ext_h3.astype(_dt),
        hy_ext_h3=hy_ext_h3.astype(_dt),
        duogrid=duogrid,
        gnomonic_form=gnomonic,
        fv3_grid_type=0 if gnomonic == "ed" else 2,
    )

    # Eagerly populate the vectorized halo index cache so that the
    # first call to pad_halo (which may happen inside jax.lax.scan)
    # does not trigger a cache write during JAX tracing.
    from legoesm.grids.halo import precompute_halo_tables
    precompute_halo_tables(n)

    return grid


def create_fv3_native_cubed_sphere(
    n: int,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
    use_duogrid: bool = False,
    k2e_nord: int = 2,
    duogrid_ng: int | None = None,
) -> CubedSphereGrid:
    """Create the FV3-native cubed-sphere grid (gnomonic_ed, FV3 grid_type=0).

    Phase-1 FV3-native entry point: the grid FV3 actually runs on
    (``fv_arrays.F90`` default ``grid_type = 0`` → ``gnomonic_ed`` in
    ``fv_grid_utils.F90``), with static provenance stamped on the returned
    grid (``gnomonic_form="ed"``, ``fv3_grid_type=0``).

    This selects only the FV3-native NODE DISTRIBUTION.  Metric construction
    (phase 2), cross-face duogrid halos (phase 3), and the forward-backward
    solver core (phase 4) are separate fidelity phases — a grid built here
    does NOT by itself make a model run FV3-faithful.

    The historic legoESM default remains ``create_cubed_sphere(n)``
    (equiangular); it is unchanged by this factory.  Schmidt/stretch/shift
    transforms are not supported on the ED path and are deliberately not
    accepted here.
    """
    return create_cubed_sphere(
        n,
        radius=radius,
        omega=omega,
        dtype=dtype,
        use_duogrid=use_duogrid,
        k2e_nord=k2e_nord,
        duogrid_ng=duogrid_ng,
        gnomonic="ed",
    )


def schmidt_transform(
    lon: jax.Array,
    lat: jax.Array,
    stretch_fac: float,
    target_lon: float,
    target_lat: float,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 586: Schmidt transformation for stretched cubed-sphere.

    Faithful port of FV3 ``direct_transform`` (fv_grid_utils.F90:870-917).
    Applies a conformal stretching that locally enhances resolution at
    ``(target_lon, target_lat)`` by factor ``stretch_fac``.

    Algorithm:
    1. Latitude stretching:
           lat_t = asin( (c²-1 + (c²+1)·sin_lat) / (c²+1 + (c²-1)·sin_lat) )
       where c = stretch_fac.  c > 1 → stretching (high-res near target).
    2. Pole rotation: rotate the stretched-pole frame so the new pole
       lies at (target_lon, target_lat).

    Parameters
    ----------
    lon, lat : jax.Array
        Input gnomonic coordinates (any shape).  Lat in [-π/2, π/2],
        lon in [0, 2π].
    stretch_fac : float
        Stretching factor c.  1.0 = no stretch.  Typical 2-5 for regional
        focus.  When |c-1| < 1e-5 stretching is skipped (only rotation).
    target_lon, target_lat : float
        Center of high-res face in radians.  When target_lat = -π/2
        (equivalent of FV3 default -90°), no rotation applied.

    Returns
    -------
    lon_new, lat_new : jax.Array
        Transformed coordinates.

    Notes
    -----
    Faithful to FV3.  Adds stretched-grid support to ``create_cubed_sphere``
    via the ``stretch_fac``/``target_*`` kwargs.  Closes user audit item
    #1 (stretched grid) partial — nested grids (2-way refinement) deferred.
    """
    c = stretch_fac
    c2p1 = 1.0 + c * c
    c2m1 = 1.0 - c * c

    # Step 1: latitude stretching.
    sin_lat = jnp.sin(lat)
    # When |c²-1| < 1e-7, no stretching → lat_t = lat.
    do_stretch = abs(c2m1) > 1e-7
    if do_stretch:
        lat_t = jnp.arcsin(
            (c2m1 + c2p1 * sin_lat) / (c2p1 + c2m1 * sin_lat)
        )
    else:
        lat_t = lat

    # Step 2: pole rotation to (target_lon, target_lat).
    sin_p = jnp.sin(target_lat)
    cos_p = jnp.cos(target_lat)
    sin_lat_t = jnp.sin(lat_t)
    cos_lat_t = jnp.cos(lat_t)
    cos_lon_old = jnp.cos(lon)
    sin_lon_old = jnp.sin(lon)

    sin_o = -(sin_p * sin_lat_t + cos_p * cos_lat_t * cos_lon_old)
    sin_o = jnp.clip(sin_o, -1.0, 1.0)  # numerical safety for asin

    is_pole = (1.0 - jnp.abs(sin_o)) < 1e-7
    p2 = 0.5 * jnp.pi
    two_pi = 2.0 * jnp.pi

    # Non-pole branch
    lat_rot = jnp.arcsin(sin_o)
    lon_rot = target_lon + jnp.arctan2(
        -cos_lat_t * sin_lon_old,
        -sin_lat_t * cos_p + cos_lat_t * sin_p * cos_lon_old,
    )
    lon_rot = jnp.where(lon_rot < 0.0, lon_rot + two_pi, lon_rot)
    lon_rot = jnp.where(lon_rot >= two_pi, lon_rot - two_pi, lon_rot)

    # Pole branch
    lat_pole = jnp.sign(sin_o) * p2
    lon_pole = jnp.zeros_like(lon_rot)

    lon_new = jnp.where(is_pole, lon_pole, lon_rot)
    lat_new = jnp.where(is_pole, lat_pole, lat_rot)
    return lon_new, lat_new


def cube_transform(
    lon: jax.Array,
    lat: jax.Array,
    stretch_fac: float,
    target_lon: float,
    target_lat: float,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 589: cube_transform (revised Schmidt at north pole).

    Faithful port of FV3 ``cube_transform`` (fv_grid_utils.F90:920-980).
    Same algorithm as ``schmidt_transform`` (iter 586) but with a
    ``lon += π`` shift before the pole rotation to get the final
    orientation correct.  Selected by FV3 namelist via
    ``do_cube_transform=.true.`` (alternative to ``do_schmidt``).

    Algorithm:
    1. Latitude stretching (identical to direct_transform):
       ``lat_t = asin((c²-1 + (c²+1)·sin_lat) / (c²+1 + (c²-1)·sin_lat))``
    2. **Add π to lon** (the only difference from direct_transform).
    3. Pole rotation to (target_lon, target_lat).

    Parameters
    ----------
    lon, lat : jax.Array
        Input gnomonic coordinates.  Lat ∈ [-π/2, π/2], lon ∈ [0, 2π].
    stretch_fac : float
        Stretching factor c.  1.0 = no stretch.
    target_lon, target_lat : float
        Center of high-res face in radians.

    Returns
    -------
    lon_new, lat_new : jax.Array
        Transformed coordinates.

    See Also
    --------
    schmidt_transform : iter 586, ``do_schmidt`` variant (no π shift).
    """
    c = stretch_fac
    c2p1 = 1.0 + c * c
    c2m1 = 1.0 - c * c

    sin_lat = jnp.sin(lat)
    do_stretch = abs(c2m1) > 1e-7
    if do_stretch:
        lat_t = jnp.arcsin(
            (c2m1 + c2p1 * sin_lat) / (c2p1 + c2m1 * sin_lat)
        )
    else:
        lat_t = lat

    sin_p = jnp.sin(target_lat)
    cos_p = jnp.cos(target_lat)
    sin_lat_t = jnp.sin(lat_t)
    cos_lat_t = jnp.cos(lat_t)

    # iter-589: the only difference from schmidt_transform — lon += π
    lon_pi = lon + jnp.pi
    cos_lon_pi = jnp.cos(lon_pi)
    sin_lon_pi = jnp.sin(lon_pi)

    sin_o = -(sin_p * sin_lat_t + cos_p * cos_lat_t * cos_lon_pi)
    sin_o = jnp.clip(sin_o, -1.0, 1.0)

    is_pole = (1.0 - jnp.abs(sin_o)) < 1e-7
    p2 = 0.5 * jnp.pi
    two_pi = 2.0 * jnp.pi

    lat_rot = jnp.arcsin(sin_o)
    lon_rot = target_lon + jnp.arctan2(
        -cos_lat_t * sin_lon_pi,
        -sin_lat_t * cos_p + cos_lat_t * sin_p * cos_lon_pi,
    )
    lon_rot = jnp.where(lon_rot < 0.0, lon_rot + two_pi, lon_rot)
    lon_rot = jnp.where(lon_rot >= two_pi, lon_rot - two_pi, lon_rot)

    lat_pole = jnp.sign(sin_o) * p2
    lon_pole = jnp.zeros_like(lon_rot)

    lon_new = jnp.where(is_pole, lon_pole, lon_rot)
    lat_new = jnp.where(is_pole, lat_pole, lat_rot)
    return lon_new, lat_new


def _compute_gnomonic_lonlat(n: int) -> tuple[jax.Array, jax.Array]:
    """Compute longitude and latitude on the gnomonic cubed-sphere.

    Uses the EQUIANGULAR gnomonic projection (alpha equally spaced in angle,
    x = tan(alpha)) — i.e. FV3 gnomonic_angl / grid_type=2, NOT equidistant
    (grid_type=1) nor FV3's operational gnomonic_ed (grid_type=0). Each face of
    the cube is mapped to the sphere via central (gnomonic) projection.

    Parameters
    ----------
    n : int
        Number of cells per face edge.

    Returns
    -------
    lon, lat : arrays of shape (6, n, n) in radians.
    """
    # Local coordinates on each face: [-pi/4, pi/4]
    # Cell centers at uniform spacing
    alpha = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, n, endpoint=False)
    alpha = alpha + (jnp.pi / 4) / n  # Shift to cell centers
    alpha_x, alpha_y = jnp.meshgrid(alpha, alpha, indexing='ij')

    # Gnomonic projection: (alpha_x, alpha_y) -> (x, y, z) on unit sphere
    # For each face, we define local Cartesian coordinates, then
    # project onto the sphere and convert to lon/lat.

    all_lon = []
    all_lat = []

    for face in range(6):
        x, y, z = _face_to_cartesian(face, alpha_x, alpha_y)
        r = jnp.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / r, y / r, z / r

        face_lon = jnp.mod(jnp.arctan2(y, x), 2.0 * jnp.pi)  # [0, 2π)
        face_lat = jnp.arcsin(jnp.clip(z, -1.0, 1.0))

        all_lon.append(face_lon)
        all_lat.append(face_lat)

    lon = jnp.stack(all_lon, axis=0)  # (6, n, n)
    lat = jnp.stack(all_lat, axis=0)  # (6, n, n)

    return lon, lat


def _gnomonic_ed_6face_from_theta(
    theta: jax.Array, alpha: float,
) -> tuple[jax.Array, jax.Array]:
    """gnomonic_ed 6-face grid in create's numbering from a 1D W-edge angle
    array ``theta`` (grid POINTS — cell centers or corners as supplied).

    Shared pipeline construct → ``-π`` FV3 orientation shift → ``mirror_grid_
    faces`` → ``gnomonic_ed_remap_to_create``.  Both symmetrization passes are
    machine-zero no-ops on gnomonic_ed.  This is the SINGLE source of gnomonic_ed
    cell-center positions so grid.lon/lat and the padded metrics (angle, hx/hy)
    refer to the SAME centers — mirroring the equiangular convention (cell-
    centre parametric coordinate for both), unlike a cell_center2-of-corners
    centre which differs by ~½ cell (iter67 consistency fix).
    """
    lon1, lat1 = _gnomonic_ed_construct(theta, alpha=alpha)
    lon1 = lon1 - jnp.pi
    lon6, lat6 = mirror_grid_faces(lon1, lat1)
    return gnomonic_ed_remap_to_create(lon6, lat6)


def _compute_gnomonic_ed_lonlat(n: int) -> tuple[jax.Array, jax.Array]:
    """Cell-center lon/lat for the FV3 OPERATIONAL gnomonic_ed grid (grid_type=0).

    Building block for the gated ``create_cubed_sphere(gnomonic="ed")`` path.
    Drop-in ``(6, n, n)`` replacement for :func:`_compute_gnomonic_lonlat`
    (equiangular).  Uses the cell-centre great-circle-edge angle distribution
    through the shared :func:`_gnomonic_ed_6face_from_theta` pipeline, so these
    centres are CONSISTENT with the padded metric builders (same definition;
    iter67 fixed an earlier cell_center2-of-corners centre that differed by ~½
    cell from the padded-metric centres).

    gnomonic_ed gives near-uniform cells (max aspect 1.06 vs equiangular's 1.40
    at corners) — why FV3 uses it operationally and the candidate fix for the
    C96 high-res cube-edge eigenmode.  Tested in
    ``tests/grids/test_gnomonic_ed_centers_iter62.py``.

    Centres = ``cell_center2`` of the gnomonic_ed CORNERS (the FV3 agrid
    definition).  NOTE (iter67): constructing centres via ``construct`` at a
    cell-centre θ distribution is WRONG for gnomonic_ed — its great-circle
    construction is range-dependent (a θ sub-range yields a different cell
    distribution, 1.80 vs the √2 1.31 corner-derived ratio), unlike
    equiangular's parametric ``tan(α)``.  So corner→cell_center2 is the
    consistent centre source; the padded metric builders must match it
    (corner-derived), not construct-at-cell-centre-θ.
    """
    lon_c, lat_c = make_fv3_native_grid(n, grid_type=0)  # (6, n+1, n+1) corners
    lon_c, lat_c = gnomonic_ed_remap_to_create(lon_c, lat_c)  # → create numbering
    return cell_center2(
        lon_c[:, :-1, :-1], lat_c[:, :-1, :-1],   # SW
        lon_c[:, 1:, :-1], lat_c[:, 1:, :-1],     # SE
        lon_c[:, 1:, 1:], lat_c[:, 1:, 1:],       # NE
        lon_c[:, :-1, 1:], lat_c[:, :-1, 1:],     # NW
    )


# Face permutation + D4 rotation mapping make_fv3_native_grid's FV3 face
# numbering/orientation onto create_cubed_sphere's _face_to_cartesian numbering.
# Originally derived iter66 against the pre-oracle mirror; re-derived after
# the phase-1 mirror_grid fix (polar tiles were hemisphere-swapped vs FV3 —
# scripts/tmp/_derive_remap_table.py match ≤1e-11, unique over rot90×flips)
# such that the FINAL create-layout grid is UNCHANGED.  Now canonical:
# FV3 face 3 (north tile, index 2) lands on create's +z slot 4 and FV3
# face 6 (south, index 5) on create's -z slot 5.  Seam continuity through
# create's halo tables is unchanged (byte-identical composite).
_GNOMONIC_ED_FACE_PERM = (0, 1, 3, 4, 2, 5)
_GNOMONIC_ED_FACE_ROT = (0, 0, 3, 3, 1, 0)  # k for jnp.rot90 (counter-clockwise)


def gnomonic_ed_remap_to_create(
    lon: jax.Array, lat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Remap a 6-face grid from make_fv3_native_grid's FV3 face
    numbering/orientation to create_cubed_sphere's convention.

    Required because make_fv3_native_grid builds faces by mirroring face-1
    (FV3 orientation) — seam-continuous but PERMUTED/ROTATED relative to
    create's per-face ``_face_to_cartesian`` layout.  Applying this keeps the
    seam topology consistent with create's halo tables.  Works on corner
    ``(6, m, m)`` or centre ``(6, n, n)`` arrays.
    """
    lo = jnp.stack([
        jnp.rot90(lon[_GNOMONIC_ED_FACE_PERM[F]], _GNOMONIC_ED_FACE_ROT[F])
        for F in range(6)
    ])
    la = jnp.stack([
        jnp.rot90(lat[_GNOMONIC_ED_FACE_PERM[F]], _GNOMONIC_ED_FACE_ROT[F])
        for F in range(6)
    ])
    return lo, la


def _face_to_cartesian(
    face: int, alpha_x: jax.Array, alpha_y: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map local gnomonic coordinates to 3D Cartesian coordinates.

    Face numbering:
        0: +x (front)   - equatorial, centered at 0 lon
        1: +y (right)    - equatorial, centered at 90E
        2: -x (back)     - equatorial, centered at 180E
        3: -y (left)     - equatorial, centered at 90W
        4: +z (top)      - north pole
        5: -z (bottom)   - south pole
    """
    tan_x = jnp.tan(alpha_x)
    tan_y = jnp.tan(alpha_y)

    if face == 0:    # +x face
        x = jnp.ones_like(tan_x)
        y = tan_x
        z = tan_y
    elif face == 1:  # +y face
        x = -tan_x
        y = jnp.ones_like(tan_x)
        z = tan_y
    elif face == 2:  # -x face
        x = -jnp.ones_like(tan_x)
        y = -tan_x
        z = tan_y
    elif face == 3:  # -y face
        x = tan_x
        y = -jnp.ones_like(tan_x)
        z = tan_y
    elif face == 4:  # +z face (north pole)
        x = -tan_y
        y = tan_x
        z = jnp.ones_like(tan_x)
    elif face == 5:  # -z face (south pole)
        x = tan_y
        y = tan_x
        z = -jnp.ones_like(tan_x)
    else:
        raise ValueError(f"Invalid face index: {face}")

    return x, y, z


def _compute_exact_cell_areas(n: int, radius: float) -> jax.Array:
    """Compute exact spherical cell areas using l'Huilier's theorem.

    Each cell is a spherical quadrilateral defined by its 4 corners on
    the gnomonic grid.  We split each quad into 2 spherical triangles
    and sum their spherical excess (= area on unit sphere).

    Parameters
    ----------
    n : int
        Number of cells per face edge.
    radius : float
        Sphere radius [m].

    Returns
    -------
    area : jax.Array, shape (6, n, n)
        Exact cell areas [m^2].
    """
    # Cell corners: n+1 points along each edge
    alpha_edges = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, n + 1)
    ax, ay = jnp.meshgrid(alpha_edges, alpha_edges, indexing='ij')

    all_areas = []
    for face in range(6):
        # Corner Cartesian coordinates on unit sphere
        x, y, z = _face_to_cartesian(face, ax, ay)
        r = jnp.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / r, y / r, z / r  # (n+1, n+1)

        # For each cell (i,j), corners at (i,j), (i+1,j), (i+1,j+1), (i,j+1)
        # SW, SE, NE, NW
        sw_x, sw_y, sw_z = x[:-1, :-1], y[:-1, :-1], z[:-1, :-1]
        se_x, se_y, se_z = x[1:, :-1], y[1:, :-1], z[1:, :-1]
        ne_x, ne_y, ne_z = x[1:, 1:], y[1:, 1:], z[1:, 1:]
        nw_x, nw_y, nw_z = x[:-1, 1:], y[:-1, 1:], z[:-1, 1:]

        def _triangle_excess(x1, y1, z1, x2, y2, z2, x3, y3, z3):
            """Spherical excess of triangle on unit sphere via l'Huilier."""
            # Arc lengths between vertices
            dot12 = jnp.clip(x1*x2 + y1*y2 + z1*z2, -1.0, 1.0)
            dot23 = jnp.clip(x2*x3 + y2*y3 + z2*z3, -1.0, 1.0)
            dot31 = jnp.clip(x3*x1 + y3*y1 + z3*z1, -1.0, 1.0)
            a = jnp.arccos(dot12)
            b = jnp.arccos(dot23)
            c = jnp.arccos(dot31)
            s = 0.5 * (a + b + c)
            # l'Huilier's theorem
            tan_E4_sq = jnp.clip(
                jnp.tan(s / 2) * jnp.tan((s - a) / 2)
                * jnp.tan((s - b) / 2) * jnp.tan((s - c) / 2),
                0.0, None,
            )
            return 4.0 * jnp.arctan(jnp.sqrt(tan_E4_sq))

        # Split quad into 2 triangles: (SW,SE,NE) + (SW,NE,NW)
        e1 = _triangle_excess(sw_x, sw_y, sw_z, se_x, se_y, se_z,
                              ne_x, ne_y, ne_z)
        e2 = _triangle_excess(sw_x, sw_y, sw_z, ne_x, ne_y, ne_z,
                              nw_x, nw_y, nw_z)
        all_areas.append(radius**2 * (e1 + e2))

    return jnp.stack(all_areas, axis=0)


def lonlat_to_cartesian(
    lon: jax.Array, lat: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert lon/lat (radians) to unit sphere Cartesian coordinates."""
    cos_lat = jnp.cos(lat)
    return cos_lat * jnp.cos(lon), cos_lat * jnp.sin(lon), jnp.sin(lat)


def great_circle_distance(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    radius: float = constants.R_earth,
) -> jax.Array:
    """Compute great-circle distance using the Haversine formula."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = jnp.sin(dlat / 2)**2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon / 2)**2
    return 2.0 * radius * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0.0, 1.0)))


def mid_pt_sphere(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 608: great-circle midpoint of two (lon, lat) points.

    Faithful port of FV3 ``mid_pt_sphere`` (fv_grid_utils.F90:1981-1992).
    Algorithm:
        1. (lon, lat) → 3D Cartesian unit vector e
        2. e_mid = (e1 + e2) / 2 (Cartesian midpoint)
        3. Normalize e_mid → unit sphere
        4. Cartesian → (lon, lat)

    The result is the point on the great circle through (p1, p2)
    equidistant from both endpoints.  NOT the same as the (lon, lat)
    average — that gives wrong results across the dateline or poles.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : jax.Array (any shape, broadcastable)
        Two points on the sphere in radians.

    Returns
    -------
    lon_mid, lat_mid : jax.Array
        Midpoint on the great circle (radians).
    """
    # latlon → Cartesian
    cl1, sl1 = jnp.cos(lat1), jnp.sin(lat1)
    cl2, sl2 = jnp.cos(lat2), jnp.sin(lat2)
    x1 = cl1 * jnp.cos(lon1)
    y1 = cl1 * jnp.sin(lon1)
    z1 = sl1
    x2 = cl2 * jnp.cos(lon2)
    y2 = cl2 * jnp.sin(lon2)
    z2 = sl2
    # Cartesian midpoint
    xm = 0.5 * (x1 + x2)
    ym = 0.5 * (y1 + y2)
    zm = 0.5 * (z1 + z2)
    # Normalize to unit sphere
    norm = jnp.sqrt(xm * xm + ym * ym + zm * zm)
    norm = jnp.where(norm > 1e-30, norm, 1.0)
    xm = xm / norm
    ym = ym / norm
    zm = zm / norm
    # Back to (lon, lat)
    lat_mid = jnp.arcsin(jnp.clip(zm, -1.0, 1.0))
    lon_mid = jnp.arctan2(ym, xm)
    # Wrap lon to [0, 2π)
    lon_mid = jnp.where(lon_mid < 0.0, lon_mid + 2.0 * jnp.pi, lon_mid)
    return lon_mid, lat_mid


def latlon2xyz(
    lon: jax.Array, lat: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 611: FV3-named alias for ``lonlat_to_cartesian``.

    Faithful port of FV3 ``latlon2xyz`` (fv_grid_utils.F90:1639-1665).
    Convert (lon, lat) in radians to 3D Cartesian unit-sphere
    coordinates::

        x = cos(lat) cos(lon)
        y = cos(lat) sin(lon)
        z = sin(lat)
    """
    return lonlat_to_cartesian(lon, lat)


def xyz2latlon(
    x: jax.Array, y: jax.Array, z: jax.Array,
    eps: float = 1e-10,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 611: Cartesian → (lon, lat) inverse of ``latlon2xyz``.

    Faithful port of FV3 ``cart_to_latlon`` (fv_grid_utils.F90:1739-1777).
    Normalizes (x, y, z) to the unit sphere first; returns ``lon`` in
    ``[0, 2π)`` and ``lat`` in ``[-π/2, π/2]``.

    Matches FV3's ``esl=1.d-10`` guard near the poles (where
    ``|x|+|y| < esl``, longitude is set to 0).
    """
    dist = jnp.sqrt(x * x + y * y + z * z)
    safe = jnp.where(dist > 0.0, dist, 1.0)
    x_n = x / safe
    y_n = y / safe
    z_n = z / safe
    lat = jnp.arcsin(jnp.clip(z_n, -1.0, 1.0))
    near_pole = (jnp.abs(x_n) + jnp.abs(y_n)) < eps
    lon = jnp.where(near_pole, 0.0, jnp.arctan2(y_n, x_n))
    lon = jnp.where(lon < 0.0, lon + 2.0 * jnp.pi, lon)
    return lon, lat


def inner_prod(
    v1: jax.Array, v2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 611: Cartesian dot product.

    Faithful port of FV3 ``inner_prod`` (fv_grid_utils.F90:984-998).
    Takes the last axis as the 3-vector component; broadcasts over
    leading axes.  Each ``v1`` and ``v2`` is shape ``(..., 3)``.
    """
    return jnp.sum(v1 * v2, axis=-1)


def vect_cross(
    p1: jax.Array, p2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 611: Cartesian cross product ``e = p1 × p2``.

    Faithful port of FV3 ``vect_cross`` (fv_grid_utils.F90:1781-1791).
    Takes the last axis as the 3-vector component; broadcasts over
    leading axes.
    """
    return jnp.cross(p1, p2, axis=-1)


def normalize_vect(
    e: jax.Array, eps: float = 1e-30,
) -> jax.Array:
    """FV3_3D iter 611: normalize Cartesian vector to unit length.

    Faithful port of FV3 ``normalize_vect`` (fv_grid_utils.F90:
    1880-1893).  Takes the last axis as the 3-vector component;
    broadcasts over leading axes.  Zero-vector input returns the
    input unchanged (avoiding NaN).
    """
    pdot = jnp.sqrt(jnp.sum(e * e, axis=-1, keepdims=True))
    safe = jnp.where(pdot > eps, pdot, 1.0)
    return e / safe


def mid_pt3_cart(
    p1: jax.Array, p2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 611: Cartesian-input great-circle midpoint.

    Faithful port of FV3 ``mid_pt3_cart`` (fv_grid_utils.F90:
    1996-2022).  Returns the normalized sum (p1 + p2) / |p1 + p2|.

    Takes the last axis as the 3-vector component; broadcasts over
    leading axes.  Each of ``p1``, ``p2`` should already be on the
    unit sphere (no extra normalization beyond the post-sum step).
    """
    return normalize_vect(p1 + p2)


def mid_pt_cart(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 611: (lon, lat)-input → Cartesian midpoint vector.

    Faithful port of FV3 ``mid_pt_cart`` (fv_grid_utils.F90:
    2026-2036).  Convenience for code that takes (lon, lat) inputs
    but wants the Cartesian midpoint (e.g., FV3 grid generation).
    Returns shape ``(..., 3)``.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    p1 = jnp.stack([x1, y1, z1], axis=-1)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    return mid_pt3_cart(p1, p2)


def get_unit_vect2(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 611: unit tangent vector at the GC midpoint.

    Faithful port of FV3 ``get_unit_vect2`` (fv_grid_utils.F90:
    1848-1863).  Returns the unit tangent vector to the great
    circle through (e1, e2), evaluated at the midpoint and pointing
    from e1 toward e2.  Used in FV3 ``edge_factors`` /
    ``efactor_a2c_v`` for metric construction.

    Algorithm:
        p1 = latlon2xyz(e1)
        p2 = latlon2xyz(e2)
        pc = mid_pt3_cart(p1, p2)
        p3 = p2 × p1           (great-circle pole)
        uc = pc × p3           (tangent at pc)
        uc / |uc|

    Returns shape ``(..., 3)``.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    p1 = jnp.stack([x1, y1, z1], axis=-1)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    pc = mid_pt3_cart(p1, p2)
    p3 = vect_cross(p2, p1)
    uc = vect_cross(pc, p3)
    return normalize_vect(uc)


def mirror_xyz(
    p1: jax.Array, p2: jax.Array, p0: jax.Array,
) -> jax.Array:
    """FV3_3D iter 612: reflect ``p0`` across great-circle plane (p1, p2).

    Faithful port of FV3 ``mirror_xyz`` (fv_grid_utils.F90:1668-1702).
    The mirror plane is the great circle through ``p1`` and ``p2``;
    the plane normal is ``nb = (p1 × p2) / |p1 × p2|``.  Mirror image
    of ``p0`` is::

        p = p0 - 2·(p0·nb)·nb

    Used in FV3 cubed-sphere grid generation (panel reflections
    across face symmetry planes).

    Takes the last axis as the 3-vector component; broadcasts on
    leading axes.  Each of ``p1``, ``p2``, ``p0`` is shape
    ``(..., 3)``; result is ``(..., 3)``.
    """
    nb_raw = vect_cross(p1, p2)
    nb = normalize_vect(nb_raw)
    pdot = jnp.sum(p0 * nb, axis=-1, keepdims=True)
    return p0 - 2.0 * pdot * nb


def mirror_latlon(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon0: jax.Array, lat0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 612: (lon, lat) reflection across great-circle (p1,p2).

    Faithful port of FV3 ``mirror_latlon`` (fv_grid_utils.F90:
    1705-1736).  Converts inputs to Cartesian, calls ``mirror_xyz``,
    converts back.  Returns ``(lon3, lat3)`` of the mirror image.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    x0, y0, z0 = latlon2xyz(lon0, lat0)
    p1 = jnp.stack([x1, y1, z1], axis=-1)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    p0 = jnp.stack([x0, y0, z0], axis=-1)
    p3 = mirror_xyz(p1, p2, p0)
    return xyz2latlon(p3[..., 0], p3[..., 1], p3[..., 2])


def intp_great_circle(
    beta: jax.Array,
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 612: linear-in-Cartesian great-circle interpolation.

    Faithful port of FV3 ``intp_great_circle`` (fv_grid_utils.F90:
    1896-1925).  At ``beta ∈ [0, 1]`` interpolates from ``p1``
    (β=0) to ``p2`` (β=1) along the great circle::

        s = (1-β)·e1 + β·e2;   e_out = s / |s|

    NOTE: this is the SECANT linear interpolant projected to the
    sphere — NOT slerp.  For β=0.5 it matches ``mid_pt_sphere``.
    For an arc-length-uniform variant use ``slerp``.
    """
    alpha = 1.0 - beta
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    s1 = alpha * x1 + beta * x2
    s2 = alpha * y1 + beta * y2
    s3 = alpha * z1 + beta * z2
    dd = jnp.sqrt(s1 * s1 + s2 * s2 + s3 * s3)
    safe = jnp.where(dd > 0.0, dd, 1.0)
    return xyz2latlon(s1 / safe, s2 / safe, s3 / safe)


def slerp(
    beta: jax.Array,
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    eps_omg: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 612: spherical linear interpolation (Shoemake slerp).

    Faithful port of FV3 ``spherical_linear_interpolation``
    (fv_grid_utils.F90:1927-1979).  Arc-length-uniform
    interpolation along the great circle::

        ω = acos(e1·e2)
        e_b = (sin((1-β)ω)·e1 + sin(βω)·e2) / sin(ω)

    Returns ``(lon_b, lat_b)`` at parameter ``β ∈ [0, 1]``.

    Antipodal-point safety: FV3 raises a fatal error for
    ``|ω| < 1e-5``; here we silently return the secant interpolant
    (well-defined for ω=0 colocated points; near-antipodal points
    still have ambiguous slerp direction so caller should avoid).
    """
    alpha = 1.0 - beta
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    dot = jnp.clip(x1 * x2 + y1 * y2 + z1 * z2, -1.0, 1.0)
    omg = jnp.arccos(dot)
    sin_omg = jnp.sin(omg)
    safe_sin = jnp.where(jnp.abs(sin_omg) > eps_omg, sin_omg, 1.0)
    w1 = jnp.sin(alpha * omg) / safe_sin
    w2 = jnp.sin(beta * omg) / safe_sin
    # Fallback to secant for tiny ω (well-defined colocated case)
    secant = jnp.abs(omg) <= eps_omg
    w1 = jnp.where(secant, alpha, w1)
    w2 = jnp.where(secant, beta, w2)
    xb = w1 * x1 + w2 * x2
    yb = w1 * y1 + w2 * y2
    zb = w1 * z1 + w2 * z2
    return xyz2latlon(xb, yb, zb)


def spherical_angle(
    p1: jax.Array, p2: jax.Array, p3: jax.Array,
) -> jax.Array:
    """FV3_3D iter 613: angle at vertex ``p1`` of spherical triangle (p1, p2, p3).

    Faithful port of FV3 ``spherical_angle`` (fv_grid_utils.F90:
    2838-2895).  Computes::

        P = p1 × p2
        Q = p1 × p3
        cos(angle) = (P·Q) / (|P|·|Q|)

    With FV3's degenerate-input fixups:
        - ``ddd <= 0`` (colinear or coincident points) → angle = 0
        - ``|cos| > 1`` (numerical) → angle = π or 0 by sign

    Takes the last axis as the 3-vector component; broadcasts over
    leading axes.
    """
    p_vec = vect_cross(p1, p2)
    q_vec = vect_cross(p1, p3)
    p_sq = jnp.sum(p_vec * p_vec, axis=-1)
    q_sq = jnp.sum(q_vec * q_vec, axis=-1)
    pq = jnp.sum(p_vec * q_vec, axis=-1)
    ddd = p_sq * q_sq
    safe = jnp.where(ddd > 0.0, ddd, 1.0)
    cos_a = pq / jnp.sqrt(safe)
    cos_a = jnp.clip(cos_a, -1.0, 1.0)
    angle = jnp.arccos(cos_a)
    # Degenerate ddd <= 0 → 0
    return jnp.where(ddd > 0.0, angle, 0.0)


def cell_center3(
    p1: jax.Array, p2: jax.Array, p3: jax.Array, p4: jax.Array,
) -> jax.Array:
    """FV3_3D iter 613: Cartesian cell center from 4 corner points.

    Faithful port of FV3 ``cell_center3`` (fv_grid_utils.F90:
    2728-2745).  Returns normalized sum ``(p1+p2+p3+p4)/|sum|``.

    Each ``pi`` is shape ``(..., 3)`` on the unit sphere; result is
    ``(..., 3)`` on the unit sphere.
    """
    return normalize_vect(p1 + p2 + p3 + p4)


def cell_center2(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon3: jax.Array, lat3: jax.Array,
    lon4: jax.Array, lat4: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 613: (lon, lat) cell center from 4 corner (lon, lat).

    Faithful port of FV3 ``cell_center2`` (fv_grid_utils.F90:
    2700-2725).  Latlon wrapper for ``cell_center3``.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    x3, y3, z3 = latlon2xyz(lon3, lat3)
    x4, y4, z4 = latlon2xyz(lon4, lat4)
    p1 = jnp.stack([x1, y1, z1], axis=-1)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    p3 = jnp.stack([x3, y3, z3], axis=-1)
    p4 = jnp.stack([x4, y4, z4], axis=-1)
    ec = cell_center3(p1, p2, p3, p4)
    return xyz2latlon(ec[..., 0], ec[..., 1], ec[..., 2])


def dist2side_latlon(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon_p: jax.Array, lat_p: jax.Array,
) -> jax.Array:
    """FV3_3D iter 613: angular distance from point to great-circle arc.

    Faithful port of FV3 ``dist2side_latlon`` (fv_grid_utils.F90:
    2812-2834).  Returns the normalized (angular) distance on the
    unit sphere from point ``p`` to the great-circle arc through
    ``(v1, v2)``::

        dist = asin( sin(side) · sin(angle) )

    where ``side`` is the angular distance v1 → p and ``angle`` is
    the spherical angle ∠(v1 v2; v1 p).

    Returns a scalar (or broadcast result) in radians.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    xp, yp, zp = latlon2xyz(lon_p, lat_p)
    c1 = jnp.stack([x1, y1, z1], axis=-1)
    c2 = jnp.stack([x2, y2, z2], axis=-1)
    cp = jnp.stack([xp, yp, zp], axis=-1)
    angle = spherical_angle(c1, c2, cp)
    # side = great-circle distance v1 → p on UNIT sphere (radius=1)
    side = great_circle_distance(lon1, lat1, lon_p, lat_p, radius=1.0)
    return jnp.arcsin(jnp.clip(jnp.sin(side) * jnp.sin(angle), -1.0, 1.0))


def get_center_vect(
    pp: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 618: cell-center tangent unit vectors (u1, u2).

    Faithful port of FV3 ``get_center_vect`` (fv_grid_utils.F90:
    1795-1845), non-``OLD_VECT`` branch (FV3 default).  Given an
    array of cell corner positions ``pp`` of shape
    ``(..., n+1, n+1, 3)``, returns the two unit tangent vectors
    at each cell center::

        pc = cell_center3(SW, SE, NW, NE)
        # u1 (along i / x-direction):
        p1_w = mid_pt3_cart(SW, NW)   # west edge midpoint
        p2_e = mid_pt3_cart(SE, NE)   # east edge midpoint
        p3   = p2_e × p1_w
        u1   = normalize(pc × p3)
        # u2 (along j / y-direction):
        p1_s = mid_pt3_cart(SW, SE)   # south edge midpoint
        p2_n = mid_pt3_cart(NW, NE)   # north edge midpoint
        p3   = p2_n × p1_s
        u2   = normalize(pc × p3)

    Returns ``(u1, u2)`` each of shape ``(..., n, n, 3)``.

    Used by FV3 vector-halo rotation: edges between faces project
    vector components onto these per-cell tangent vectors.
    """
    sw = pp[..., :-1, :-1, :]
    se = pp[..., 1:, :-1, :]
    nw = pp[..., :-1, 1:, :]
    ne = pp[..., 1:, 1:, :]
    pc = cell_center3(sw, se, nw, ne)
    # u1 along i-direction (east-west edges)
    p1_w = mid_pt3_cart(sw, nw)
    p2_e = mid_pt3_cart(se, ne)
    p3_1 = vect_cross(p2_e, p1_w)
    u1 = normalize_vect(vect_cross(pc, p3_1))
    # u2 along j-direction (north-south edges)
    p1_s = mid_pt3_cart(sw, se)
    p2_n = mid_pt3_cart(nw, ne)
    p3_2 = vect_cross(p2_n, p1_s)
    u2 = normalize_vect(vect_cross(pc, p3_2))
    return u1, u2


def init_cubed_to_latlon(
    agrid_lon: jax.Array, agrid_lat: jax.Array,
    ec1: jax.Array, ec2: jax.Array,
    sin_sg5: jax.Array,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array,
    jax.Array, jax.Array, jax.Array, jax.Array,
    jax.Array, jax.Array,
]:
    """FV3_3D iter 626: D-grid → latlon wind rotation matrices.

    Faithful port of FV3 ``init_cubed_to_latlon``
    (fv_grid_utils.F90:2321-2384), grid_type<4 branch.  Computes
    the 8 rotation-matrix entries used by FV3 ``c2l_ord4`` to
    rotate D-grid winds to (east, north) at cell centers.

    Algorithm:
        1. vlon, vlat = unit_vect_latlon(agrid)  — local frame
        2. z11 = ec1 · vlon                       — inner products
           z12 = ec1 · vlat
           z21 = ec2 · vlon
           z22 = ec2 · vlat
        3. a11 =  0.5·z22 / sin_sg5
           a12 = -0.5·z12 / sin_sg5
           a21 = -0.5·z21 / sin_sg5
           a22 =  0.5·z11 / sin_sg5

    The (a11, a12, a21, a22) matrix gives the D-grid → (u_east,
    v_north) projection at each cell center.

    Parameters
    ----------
    agrid_lon, agrid_lat : jax.Array, shape ``(..., n, n)``
        Cell-center positions (radians).
    ec1, ec2 : jax.Array, shape ``(..., n, n, 3)``
        Cell-edge unit tangent vectors (FV3 ``ec1``, ``ec2``).
    sin_sg5 : jax.Array, shape ``(..., n, n)``
        sin of the cell-area diagonal (FV3 ``sin_sg(:,:,5)``).

    Returns
    -------
    (a11, a12, a21, a22, z11, z12, z21, z22, vlon, vlat) : tuple
        Rotation matrix entries + intermediate z and unit vectors.
    """
    vlon, vlat = unit_vect_latlon(agrid_lon, agrid_lat)
    z11 = inner_prod(ec1, vlon)
    z12 = inner_prod(ec1, vlat)
    z21 = inner_prod(ec2, vlon)
    z22 = inner_prod(ec2, vlat)
    safe_sin = jnp.where(jnp.abs(sin_sg5) > 1e-30, sin_sg5, 1.0)
    a11 = 0.5 * z22 / safe_sin
    a12 = -0.5 * z12 / safe_sin
    a21 = -0.5 * z21 / safe_sin
    a22 = 0.5 * z11 / safe_sin
    return a11, a12, a21, a22, z11, z12, z21, z22, vlon, vlat


def mirror_grid_face1_symmetrize(
    face1_lon: jax.Array, face1_lat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 625: face-1 SIGN-averaging symmetrization.

    Faithful port of FV3 ``mirror_grid`` first loop
    (fv_grid_tools.F90:2774-2807).  Symmetrizes face 1 about both
    the lon=0 meridian (i-axis) and the equator (j-axis) by:

        1. For each symmetric 4-tuple of grid points
           ``(i, j), (npx-i+1, j), (i, npy-j+1), (npx-i+1, npy-j+1)``:
        2. Compute the average of the absolute values, then assign
           ``SIGN(avg, original_value)`` to each of the 4 corners.

    Result: ``|lon|`` and ``|lat|`` are pairwise-equal across the
    mirror, preserving the sign-pattern of the original grid.

    For odd ``npx``, the central column ``i = (npx+1)/2`` is
    forced to ``lon = 0`` (FV3 lines 2799-2804).

    Parameters
    ----------
    face1_lon, face1_lat : jax.Array, shape ``(npx, npy)``
        Face-1 corner positions in radians.

    Returns
    -------
    lon_sym, lat_sym : jax.Array, shape ``(npx, npy)``
        Symmetrized face-1 grid.
    """
    npx = face1_lon.shape[0]
    npy = face1_lon.shape[1]

    # Build mirrors via reverse-indexing
    lon = face1_lon
    lat = face1_lat
    # 4-tuple of absolute lons
    avg_abs_lon = 0.25 * (
        jnp.abs(lon)
        + jnp.abs(lon[::-1, :])
        + jnp.abs(lon[:, ::-1])
        + jnp.abs(lon[::-1, ::-1])
    )
    avg_abs_lat = 0.25 * (
        jnp.abs(lat)
        + jnp.abs(lat[::-1, :])
        + jnp.abs(lat[:, ::-1])
        + jnp.abs(lat[::-1, ::-1])
    )
    # Apply SIGN(avg, original_value)
    lon_sym = jnp.copysign(avg_abs_lon, lon)
    lat_sym = jnp.copysign(avg_abs_lat, lat)

    # Odd-npx central column: lon = 0
    if npx % 2 == 1:
        center_i = (npx - 1) // 2
        lon_sym = lon_sym.at[center_i, :].set(0.0)
    if npy % 2 == 1:
        # FV3 doesn't have a corresponding odd-npy clause for lat=0,
        # but if the grid is symmetric about the equator, lat=0
        # naturally at j-center; SIGN-averaging already enforces this.
        center_j = (npy - 1) // 2
        # lat at center row is already 0 by symmetry; force exactly 0
        lat_sym = lat_sym.at[:, center_j].set(
            jnp.where(jnp.abs(lat_sym[:, center_j]) < 1e-12, 0.0,
                      lat_sym[:, center_j])
        )

    return lon_sym, lat_sym


def mirror_grid_faces(
    face1_lon: jax.Array, face1_lat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Build the 6-face cubed-sphere from face 1 — exact FV3 ``mirror_grid``.

    Transcription of FV3 ``mirror_grid`` faces 2-6 (fv_grid_tools.F90:2666-
    2754 @ 6f658bd0) including two details the original iter-624 port got
    wrong (caught by the phase-1 Fortran oracle,
    tests/grids/test_fv3_native_grid_phase1.py):

    1. FV3 rotates via ``rot_3d(..., convert=1)``, whose spherical<->cartesian
       converters use the DEFAULT build convention (``RIGHT_HAND`` undefined):
       ``z = -sin(lat)`` and ``lat = acos(z) - pi/2``.  Rotating right-handed
       cartesian vectors instead (the old port) flips the polar tiles: face 3
       must be the NORTH tile and face 6 the SOUTH tile — the convention the
       mirror's own pole-forcing lines hard-code.
    2. The odd-npx pole/dateline-consistency forcing rows (faces 3, 4, 6)
       were missing entirely.

    Rotation sequences (FV3 comments; angles in degrees):

        face 2: rot_z(-90°)
        face 3: rot_z(-90°) → rot_x(+90°)   (north-pole tile)
        face 4: rot_z(-180°) → rot_x(+90°)
        face 5: rot_z(+90°) → rot_y(+90°)
        face 6: rot_y(+90°) → rot_z(0°)     (south-pole tile)

    Parameters
    ----------
    face1_lon, face1_lat : jax.Array, shape ``(n+1, n+1)``
        Face-1 corner positions in radians (e.g., output of
        ``gnomonic_grids``).

    Returns
    -------
    lons, lats : jax.Array, shape ``(6, n+1, n+1)``
        6-face cubed-sphere corner positions in radians, FV3 face
        numbering/orientation.  Face 1 is the input.  Longitudes of faces
        2-6 are in atan2 range [-pi, pi] as in FV3 (face 1 keeps its input
        range).

    Note: this covers the rotation sequence for faces 2-6.  FV3's first
    loop (nreg=1 sign-symmetrization) is ``mirror_grid_face1_symmetrize`` —
    input here is assumed already symmetrized (post-``symm_ed``).
    """
    npx, npy = face1_lon.shape

    # FV3 default-build spherical<->cartesian (fv_grid_tools.F90
    # spherical_to_cartesian / cartesian_to_spherical with RIGHT_HAND
    # undefined): z = -sin(lat), lat = acos(z) - pi/2.  The rot_3d cartesian
    # cores are the shared ones (rot_3d below).
    def _sph2cart(lon, lat):
        return (jnp.cos(lon) * jnp.cos(lat),
                jnp.sin(lon) * jnp.cos(lat),
                -jnp.sin(lat))

    def _cart2sph(x, y, z):
        r = jnp.sqrt(x * x + y * y + z * z)
        lon = jnp.where(jnp.abs(x) + jnp.abs(y) < 1e-10,
                        0.0, jnp.arctan2(y, x))
        lat = jnp.arccos(jnp.clip(z / r, -1.0, 1.0)) - 0.5 * jnp.pi
        return lon, lat

    x1, y1, z1 = _sph2cart(face1_lon, face1_lat)

    def _rot_seq(*ops):
        x, y, z = x1, y1, z1
        for axis, ang in ops:
            x, y, z = rot_3d(axis, x, y, z, jnp.asarray(ang), degrees=True)
        return _cart2sph(x, y, z)

    lon2, lat2 = _rot_seq((3, -90.0))
    lon3, lat3 = _rot_seq((3, -90.0), (1, 90.0))
    lon4, lat4 = _rot_seq((3, -180.0), (1, 90.0))
    lon5, lat5 = _rot_seq((3, 90.0), (2, 90.0))
    lon6, lat6 = _rot_seq((2, 90.0), (3, 0.0))

    # Odd-npx pole / dateline-Greenwich consistency forcing
    # (fv_grid_tools.F90:2686-2698, 2709-2714, 2733-2745).  npx/npy are
    # static Python ints — this is init-time control flow, not traced.
    if npx % 2 != 0:
        ci = (npx - 1) // 2  # 0-based centre index (Fortran 1+(npx-1)/2)
        cj = (npy - 1) // 2
        ii = jnp.arange(npx)[:, None]
        jj = jnp.arange(npy)[None, :]

        # face 3 (north tile): centre point -> exact pole; centre row ->
        # Greenwich west of centre, dateline east of centre.
        lat3 = jnp.where((ii == ci) & (jj == ci), 0.5 * jnp.pi, lat3)
        lon3 = jnp.where((ii == ci) & (jj == ci), 0.0, lon3)
        lon3 = jnp.where((jj == cj) & (ii < ci), 0.0, lon3)
        lon3 = jnp.where((jj == cj) & (ii > ci), jnp.pi, lon3)

        # face 4: centre row on the dateline.
        lon4 = jnp.where(jj == cj, jnp.pi, lon4)

        # face 6 (south tile): centre point -> exact pole; centre column ->
        # Greenwich north of centre, dateline south of centre.
        lat6 = jnp.where((ii == ci) & (jj == ci), -0.5 * jnp.pi, lat6)
        lon6 = jnp.where((ii == ci) & (jj == ci), 0.0, lon6)
        lon6 = jnp.where((ii == ci) & (jj > cj), 0.0, lon6)
        lon6 = jnp.where((ii == ci) & (jj < cj), jnp.pi, lon6)

    lons = jnp.stack([face1_lon, lon2, lon3, lon4, lon5, lon6], axis=0)
    lats = jnp.stack([face1_lat, lat2, lat3, lat4, lat5, lat6], axis=0)
    return lons, lats


def rot_3d(
    axis: int,
    x1: jax.Array, y1: jax.Array, z1: jax.Array,
    angle: jax.Array,
    degrees: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 623: 3D rotation about coordinate axis.

    Faithful port of FV3 ``rot_3d`` (fv_grid_tools.F90:2410-2467).
    Rotates Cartesian (x1, y1, z1) by ``angle`` about axis::

        axis = 1: x-axis (y, z rotated)
        axis = 2: y-axis (x, z rotated)
        axis = 3: z-axis (x, y rotated)

    FV3 sign convention (left-handed about each axis as the code
    is written):
        axis 1: y' = c·y + s·z,    z' = -s·y + c·z
        axis 2: x' = c·x - s·z,    z' =  s·x + c·z
        axis 3: x' = c·x + s·y,    y' = -s·x + c·y

    Parameters
    ----------
    axis : int
        Rotation axis (1, 2, or 3).
    x1, y1, z1 : jax.Array
        Input Cartesian coordinates (any shape).
    angle : jax.Array
        Rotation angle (radians unless ``degrees=True``).
    degrees : bool, default False
        If True, ``angle`` is in degrees.
    """
    a = jnp.deg2rad(angle) if degrees else angle
    c = jnp.cos(a)
    s = jnp.sin(a)
    if axis == 1:
        return x1, c * y1 + s * z1, -s * y1 + c * z1
    if axis == 2:
        return c * x1 - s * z1, y1, s * x1 + c * z1
    if axis == 3:
        return c * x1 + s * y1, -s * x1 + c * y1, z1
    raise ValueError(f"Invalid axis: {axis} (must be 1, 2, or 3)")


def fill_ghost(
    q: jax.Array, ng: int, value: float,
) -> jax.Array:
    """FV3_3D iter 633: fill 4 corner-ghost regions with constant.

    Faithful JAX port of FV3 ``fill_ghost_r4`` / ``fill_ghost_r8``
    (fv_grid_utils.F90:3070-3147).  Fills the 4 corner-ghost
    rectangular regions OUTSIDE the face corners with ``value``.
    Used to mask FV3's cube-vertex singularity (no well-defined
    neighbor at the 8 cube corners, propagated to 4 corner-ghost
    blocks per face).

    Input ``q`` has shape ``(..., npx-1+2·ng, npy-1+2·ng)`` where:
        - npx-1 = number of interior cells in x (cell-centered)
        - ng = number of halo cells on each side

    The 4 corner-ghost regions are the rectangles in the halo
    where BOTH i and j are outside the interior range:
        - SW corner ghost: i ∈ [0, ng-1], j ∈ [0, ng-1]
        - SE corner ghost: i ∈ [-ng:], j ∈ [0, ng-1]
        - NE corner ghost: i ∈ [-ng:], j ∈ [-ng:]
        - NW corner ghost: i ∈ [0, ng-1], j ∈ [-ng:]

    Parameters
    ----------
    q : jax.Array, shape ``(..., n_x_halo, n_y_halo)``
        Field with halo.  Two trailing axes interpreted as (i, j).
    ng : int
        Number of halo cells on each side.
    value : float
        Fill value for corner ghost cells.

    Returns
    -------
    q_filled : jax.Array
        Copy of ``q`` with the 4 corner-ghost regions set to ``value``.
    """
    n_x = q.shape[-2]
    n_y = q.shape[-1]
    i_idx = jnp.arange(n_x)[:, None]
    j_idx = jnp.arange(n_y)[None, :]
    # Interior: ng <= i < n_x - ng, ng <= j < n_y - ng
    # Corner ghost: (i < ng AND j < ng) OR (i >= n_x-ng AND j < ng) OR
    #               (i >= n_x-ng AND j >= n_y-ng) OR (i < ng AND j >= n_y-ng)
    i_lo = i_idx < ng
    i_hi = i_idx >= (n_x - ng)
    j_lo = j_idx < ng
    j_hi = j_idx >= (n_y - ng)
    corner_mask = (
        (i_lo & j_lo) | (i_hi & j_lo) | (i_hi & j_hi) | (i_lo & j_hi)
    )
    # Broadcast mask over leading axes
    return jnp.where(corner_mask, value, q)


def global_qsum(p: jax.Array) -> jax.Array:
    """FV3_3D iter 632: quick global sum without area weighting.

    Faithful JAX port of FV3 ``global_qsum`` (fv_grid_utils.F90:
    2999-3018).  Serial (non-MPI) implementation; for distributed
    runs use ``legoesm.distributed.global_sum_mpi``.

    Returns the scalar sum of all elements in ``p`` (no area
    weighting; unlike iter-623 ``g_sum``).
    """
    return jnp.sum(p)


def global_mx(q: jax.Array) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 632: global min / max reduction.

    Faithful JAX port of FV3 ``global_mx`` (fv_grid_utils.F90:
    3020-3046).  Serial (non-MPI) implementation; for distributed
    runs use ``legoesm.distributed.global_min_mpi`` /
    ``global_max_mpi``.

    Returns ``(qmin, qmax)`` over all elements of ``q``.
    """
    return jnp.min(q), jnp.max(q)


def global_mx_c(q: jax.Array) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 632: global min / max at cell corners.

    Faithful JAX port of FV3 ``global_mx_c`` (fv_grid_utils.F90:
    3048-3067).  Identical to ``global_mx`` but FV3 distinguishes
    cell-center vs corner indexing in the signature; for legoESM
    they're identical operations on the input array.
    """
    return jnp.min(q), jnp.max(q)


def g_sum(
    p: jax.Array, area: jax.Array, mode: int = 0,
) -> jax.Array:
    """FV3_3D iter 623: area-weighted global sum.

    Faithful JAX port of FV3 ``g_sum`` (fv_grid_utils.F90:2946-2996),
    serial branch.  Computes::

        gsum = Σ_ij  p(i,j) · area(i,j)

    If ``mode == 1``, returns ``gsum / global_area`` (area-weighted
    global mean).  Otherwise returns ``gsum`` (area-weighted total).

    Parameters
    ----------
    p : jax.Array
        Field to be summed (any shape; must match ``area`` shape).
    area : jax.Array
        Cell areas (matching shape).
    mode : int, default 0
        If 1, divide by global area (returns area-weighted mean).

    Returns
    -------
    g : jax.Array
        Scalar global sum (or area-weighted mean if mode=1).

    Note: serial (non-MPI) implementation.  MPI reduction is the
    caller's responsibility (legoESM uses ``global_sum_mpi`` for
    distributed runs).
    """
    weighted = p * area
    gsum = jnp.sum(weighted)
    if mode == 1:
        global_area = jnp.sum(area)
        return gsum / global_area
    return gsum


def edge_factor_along_axis_nonortho(
    agrid_outside_lon: jax.Array, agrid_outside_lat: jax.Array,
    agrid_inside_lon: jax.Array, agrid_inside_lat: jax.Array,
    grid_corner_lon: jax.Array, grid_corner_lat: jax.Array,
) -> jax.Array:
    """FV3_3D iter 631: A→B grid interpolation weights at face boundary.

    Faithful port of FV3 ``edge_factors`` non-ortho branch
    (fv_grid_utils.F90:1212-1289).  Single-axis 1D variant: for
    one face boundary, computes per-corner interpolation weights
    ``edge_factor[j] = d2 / (d1 + d2)`` where:

        py[j]     = mid_pt_sphere(agrid_outside[j], agrid_inside[j])
        d1[j+1]   = great_circle_dist(py[j],   grid_corner[j+1])
        d2[j+1]   = great_circle_dist(py[j+1], grid_corner[j+1])

    This is used by FV3's A-grid → B-grid (corner-located)
    interpolation at non-orthogonal cubed-sphere face boundaries::

        q_corner[j+1] = (1 - edge[j+1]) · q_A[j+1] + edge[j+1] · q_A[j]

    Parameters
    ----------
    agrid_outside_lon, agrid_outside_lat : jax.Array, shape ``(n,)``
        A-grid cell-center positions just OUTSIDE the boundary
        (the halo cells across the face edge).
    agrid_inside_lon, agrid_inside_lat : jax.Array, shape ``(n,)``
        A-grid cell-center positions just INSIDE the boundary.
    grid_corner_lon, grid_corner_lat : jax.Array, shape ``(n+1,)``
        B-grid (corner) positions along the boundary.

    Returns
    -------
    edge_factor : jax.Array, shape ``(n+1,)``
        Per-corner interpolation weights.  Corner j+1 (interior)
        gets ``d2/(d1+d2)``; corners 0 and n (the face corners
        themselves) get NaN — FV3 also leaves them as ``big_number``
        (lines 1213-1216) since the edge factor formula degenerates
        there.
    """
    # Midpoints between outside and inside cells at each row
    py_lon, py_lat = mid_pt_sphere(
        agrid_outside_lon, agrid_outside_lat,
        agrid_inside_lon, agrid_inside_lat,
    )  # shape (n,)
    # For each interior corner j ∈ [1, n-1]: d1 = dist(py[j-1], grid[j]);
    # d2 = dist(py[j], grid[j])
    # py[j-1] = py[:-1], py[j] = py[1:]
    # grid corners interior: grid[1:-1] (shape (n-1,))
    d1 = great_circle_distance(
        py_lon[:-1], py_lat[:-1],
        grid_corner_lon[1:-1], grid_corner_lat[1:-1],
        radius=1.0,
    )
    d2 = great_circle_distance(
        py_lon[1:], py_lat[1:],
        grid_corner_lon[1:-1], grid_corner_lat[1:-1],
        radius=1.0,
    )
    safe_sum = jnp.where(d1 + d2 > 0.0, d1 + d2, 1.0)
    interior = d2 / safe_sum
    # Build full (n+1,) array with NaN at endpoints (FV3 big_number)
    edge_factor = jnp.full(grid_corner_lon.shape[0], jnp.nan)
    edge_factor = edge_factor.at[1:-1].set(interior)
    return edge_factor


def make_fv3_native_grid(
    im: int,
    grid_type: int = 0,
    symmetrize_face1: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 629: end-to-end FV3 native cubed-sphere grid builder.

    Integration wrapper for iters 622, 625, 624:

        1. ``gnomonic_grids(im, grid_type)``       — face-1 (iter 622)
        2. ``mirror_grid_face1_symmetrize``        — face-1 sym (iter 625)
        3. ``mirror_grid_faces``                   — faces 2-6 (iter 624)

    Reproduces FV3's full cubed-sphere construction pipeline as
    a single public API.  Returns the 6-face (lon, lat) arrays
    matching the FV3 face numbering convention (1..6 → indices 0..5).

    Parameters
    ----------
    im : int
        Number of cells per face edge.
    grid_type : int, default 0
        Grid type forwarded to ``gnomonic_grids``:
            0 → ``gnomonic_ed``   (FV3 canonical)
            1 → ``gnomonic_dist``
            2 → ``gnomonic_angl``
    symmetrize_face1 : bool, default True
        If True, apply ``mirror_grid_face1_symmetrize`` (FV3
        first-loop SIGN-averaging) before mirroring to 6 faces.

    Returns
    -------
    lons, lats : jax.Array, shape ``(6, im+1, im+1)``
        Cubed-sphere corner positions for all 6 faces in radians.
    """
    lon1, lat1 = gnomonic_grids(im, grid_type=grid_type)
    if symmetrize_face1:
        lon1, lat1 = mirror_grid_face1_symmetrize(lon1, lat1)
    return mirror_grid_faces(lon1, lat1)


def gnomonic_grids(
    im: int, grid_type: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 622: dispatcher for FV3 ``gnomonic_grids``.

    Faithful port of FV3 ``gnomonic_grids`` (fv_grid_utils.F90:
    1290-1311).  Dispatches to one of three grid generators by
    ``grid_type``:

        grid_type = 0 → ``gnomonic_ed``   (canonical, equal-distance edges; FV3 default)
        grid_type = 1 → ``gnomonic_dist`` (linear equi-distance gnomonic)
        grid_type = 2 → ``gnomonic_angl`` (equi-angular gnomonic)

    Post-processing (FV3 lines 1301-1308) for all grid_type < 3:
        1. ``symm_ed`` symmetrizes about i/j midplanes.
        2. Longitude shift by -π to bring grid into FV3's standard
           orientation (face 2 center → 0, not π).

    Parameters
    ----------
    im : int
        Number of cells per face edge.  Grid has shape ``(im+1, im+1)``.
    grid_type : int, default 0
        Grid construction algorithm (0, 1, or 2).

    Returns
    -------
    lon, lat : jax.Array, shape ``(im+1, im+1)``
        Cubed-sphere face corner positions in radians (FV3
        orientation after the -π shift).
    """
    if grid_type == 0:
        lon, lat = gnomonic_ed(im)
    elif grid_type == 1:
        lon, lat = gnomonic_dist(im)
    elif grid_type == 2:
        lon, lat = gnomonic_angl(im)
    else:
        raise ValueError(
            f"Unsupported grid_type: {grid_type} (must be 0, 1, or 2)"
        )
    # grid_type < 3 post-processing (FV3 lines 1301-1308)
    lon, lat = symm_ed(lon, lat)
    lon = lon - jnp.pi
    return lon, lat


def gnomonic_ed(im: int) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 621: equal-distance-edge cubed-sphere grid for face 2.

    Faithful port of FV3 ``gnomonic_ed`` (fv_grid_utils.F90:1313-1407).
    This is FV3's grid of choice for global cloud-resolving runs.

    Properties (FV3 docstring):
        - Defined by intersections of great circles
        - max(dx,dy) / min(dx,dy) = √2 ≈ 1.4142
        - Max aspect ratio = 1.06089
        - N-S coordinate curves are const longitude on the 4 faces
          with the equator

    Algorithm:
        1. East/West edges at constant longitude (0.75π, 1.25π).
        2. North/South edges obtained by ``mirror_latlon`` of W
           edge across the (NW, SE) diagonal.
        3. Interior Cartesian coordinates obtained by projecting
           edge values onto the constant-x = -1/√3 face cube.
        4. Convert back to (lon, lat).

    Parameters
    ----------
    im : int
        Number of cells per face edge.  Grid has shape ``(im+1, im+1)``.

    Returns
    -------
    lon, lat : jax.Array, shape ``(im+1, im+1)``
        Cubed-sphere face-2 corner positions in radians.
    """
    rsq3 = 1.0 / jnp.sqrt(3.0)
    alpha = jnp.arcsin(rsq3)
    pi = jnp.pi
    dely = 2.0 * alpha / im

    n = im + 1

    # Step 1: W and E edges (FV3 lines 1345-1350)
    j_idx = jnp.arange(n, dtype=jnp.float64)
    lon = jnp.zeros((n, n), dtype=jnp.float64)
    lat = jnp.zeros((n, n), dtype=jnp.float64)
    west_theta = -alpha + dely * j_idx
    lon = lon.at[0, :].set(0.75 * pi)
    lon = lon.at[im, :].set(1.25 * pi)
    lat = lat.at[0, :].set(west_theta)
    lat = lat.at[im, :].set(west_theta)

    # Step 2: S and N edges by mirror_latlon of W edge column (FV3 lines 1354-1359)
    # FV3 loop: for i in 2..im:
    #   mirror_latlon( (lon[0,0], lat[0,0]),  (lon[im,im], lat[im,im]),
    #                  (lon[0,i-1], lat[0,i-1]), (lon[i-1, 0], lat[i-1, 0]) )
    # Vectorize over i ∈ [1, im-1] (0-indexed)
    jnp.arange(1, im, dtype=jnp.float64)
    # Reference: SW corner (already at lon[0,0], lat[0,0]) and NE corner
    # (already at lon[im,im], lat[im,im]).  But these are not yet set
    # — lon[im,im] = lat[im,im] are from the W/E edge assignments.
    # W/E edges already set lon[0,0]=0.75π, lat[0,0]=-α; lon[im,im]=1.25π,
    # lat[im,im]=alpha.
    lon_sw, lat_sw = lon[0, 0], lat[0, 0]
    lon_ne, lat_ne = lon[im, im], lat[im, im]
    # Source points (W edge column at j=i): vary i in [1, im-1]
    i_int = jnp.arange(1, im)
    lon_src = lon[0, i_int]
    lat_src = lat[0, i_int]
    lon_s_row, lat_s_row = mirror_latlon(
        lon_sw, lat_sw,
        lon_ne, lat_ne,
        lon_src, lat_src,
    )
    # South edge (j=0): row i, S edge → (lamda(i,1), theta(i,1))
    lon = lon.at[i_int, 0].set(lon_s_row)
    lat = lat.at[i_int, 0].set(lat_s_row)
    # North edge (j=im): same lon, theta flipped
    lon = lon.at[i_int, im].set(lon_s_row)
    lat = lat.at[i_int, im].set(-lat_s_row)

    # Step 3: Convert edges to Cartesian, project onto constant-x face
    # (FV3 lines 1370-1382)
    # i=0 column (W edge), j ∈ [1, im-1]
    x_w_full, y_w_full, z_w_full = latlon2xyz(lon[0, :], lat[0, :])
    safe_x_w = jnp.where(jnp.abs(x_w_full) > 1e-30, x_w_full, 1.0)
    pp2_i0 = -y_w_full * rsq3 / safe_x_w  # y' = -y * rsq3 / x
    pp3_i0 = -z_w_full * rsq3 / safe_x_w
    # j=0 row (S edge), i ∈ [1, im-1]
    x_s_full, y_s_full, z_s_full = latlon2xyz(lon[:, 0], lat[:, 0])
    safe_x_s = jnp.where(jnp.abs(x_s_full) > 1e-30, x_s_full, 1.0)
    pp2_j0 = -y_s_full * rsq3 / safe_x_s
    pp3_j0 = -z_s_full * rsq3 / safe_x_s
    # 4 corners: latlon2xyz directly
    # FV3 uses raw latlon2xyz for corners but the same projection is needed
    # for j=0 and j=im endpoints too.  For interior points, we use the
    # projection.  For the corners, latlon2xyz gives the position on the
    # unit sphere.  But the FV3 algorithm explicitly sets pp(i, 1) and
    # pp(1, j) from the projection then sets corner positions from raw
    # latlon2xyz2.  Final step is pp(2,i,j) = pp(2,i,1) and
    # pp(3,i,j) = pp(3,1,j) for interior (i>1, j>1).
    # This means the interior i=0, j ∈ [1, im-1] uses the projected
    # values; for i=0 and j=0 corners use direct latlon2xyz.
    # FV3 line 1386: pp(1,i,j) = -rsq3 for ALL i, j → constant x face.

    # Step 4: Build full (pp2, pp3) by taking pp2 from j=0 row (i-varying)
    # and pp3 from i=0 column (j-varying).  This gives the cube-face
    # coordinates on the constant-x face.
    pp1 = jnp.full((n, n), -rsq3)
    # pp2[i, j] = pp2_j0[i] (varies with i, constant in j)
    pp2 = jnp.broadcast_to(pp2_j0[:, None], (n, n))
    # pp3[i, j] = pp3_i0[j] (varies with j, constant in i)
    pp3 = jnp.broadcast_to(pp3_i0[None, :], (n, n))
    # At the 4 corners use direct latlon2xyz (FV3 lines 1362-1365)
    # Corner SW (i=0, j=0): use lon[0,0]/lat[0,0] → (x, y, z) directly
    # We need to override the broadcast values at the 4 corners and
    # the i=0/j=0 edges with the exact values from the projection above.
    # i=0 column: pp2[0, j] should be from latlon2xyz directly (W edge);
    # but the projection above already gives the right answer when
    # pp2_j0[0] = pp2_i0[0] = 0 (W-edge has lon=0.75π, so y/x ratio is
    # known).  Verify by ensuring pp2[0, j] doesn't break and pp3[i, 0]
    # likewise.

    # j=0 row: pp3 should be pp3_j0 (S edge i-vary), NOT pp3_i0[0]
    pp3 = pp3.at[:, 0].set(pp3_j0)
    # i=0 col: pp2 should be pp2_i0 (W edge j-vary), NOT pp2_j0[0]
    pp2 = pp2.at[0, :].set(pp2_i0)
    # j=im row: similar, use S edge mirrored to N
    pp3 = pp3.at[:, im].set(-pp3_j0)  # N edge: theta flipped → z flipped
    # i=im col: E edge mirror of W edge: lon=1.25π so y/x ratio flipped
    pp2 = pp2.at[im, :].set(-pp2_i0)  # E edge: y flipped relative to W

    # Step 5: Convert pp back to (lon, lat) (FV3 line 1399)
    lon_out, lat_out = xyz2latlon(pp1, pp2, pp3)
    return lon_out, lat_out


def _gnomonic_ed_construct(
    theta_w: jax.Array, alpha: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Generalized gnomonic_ed face-2 construction for an arbitrary W-edge
    latitude array ``theta_w`` (the equal-great-circle-angle edge distribution).

    Identical to :func:`gnomonic_ed` (FV3 fv_grid_utils.F90:1313) but with the
    W/E-edge latitudes supplied directly instead of derived from ``im`` — so the
    same construction builds the in-domain grid (``theta_w = -α + (2α/im)·[0..im]``,
    α=arcsin(1/√3)) AND the HALO-extended grid (``theta_w`` spanning beyond ±α).
    This is the reusable core for the gated gnomonic_ed padded-metric builders.

    ``alpha`` fixes the SW/NE corners that define the mirror diagonal
    (default arcsin(1/√3) = the in-domain face half-extent).  This MUST stay
    pinned to the in-domain face even when ``theta_w`` extends into the halo —
    otherwise the diagonal moves and the interior no longer matches the
    in-domain grid.  The extended W-edge points are reflected across this FIXED
    diagonal to give the (continued) S-edge great circle.

    Returns the ``(m, m)`` face-2 (lon, lat), ``m = len(theta_w)``.
    """
    rsq3 = 1.0 / jnp.sqrt(3.0)
    pi = jnp.pi
    if alpha is None:
        alpha = float(jnp.arcsin(rsq3))
    theta_w = jnp.asarray(theta_w)
    m = theta_w.shape[0]
    im = m - 1

    lon = jnp.zeros((m, m), dtype=jnp.float64)
    lat = jnp.zeros((m, m), dtype=jnp.float64)
    # W (i=0) and E (i=im) edges: constant lon, lat = theta_w
    lon = lon.at[0, :].set(0.75 * pi)
    lon = lon.at[im, :].set(1.25 * pi)
    lat = lat.at[0, :].set(theta_w)
    lat = lat.at[im, :].set(theta_w)

    # S/N edges by mirror_latlon of the W-edge column across the SW–NE diagonal.
    # The diagonal is pinned to the IN-DOMAIN corners (lat=±alpha) so the
    # interior is invariant to halo extension of theta_w.
    i_int = jnp.arange(1, im)
    lon_s_row, lat_s_row = mirror_latlon(
        0.75 * pi, -alpha, 1.25 * pi, alpha,
        lon[0, i_int], lat[0, i_int],
    )
    lon = lon.at[i_int, 0].set(lon_s_row)
    lat = lat.at[i_int, 0].set(lat_s_row)
    lon = lon.at[i_int, im].set(lon_s_row)
    lat = lat.at[i_int, im].set(-lat_s_row)

    # Project edges onto the constant-x = -1/√3 cube face
    x_w, y_w, z_w = latlon2xyz(lon[0, :], lat[0, :])
    safe_x_w = jnp.where(jnp.abs(x_w) > 1e-30, x_w, 1.0)
    pp2_i0 = -y_w * rsq3 / safe_x_w
    pp3_i0 = -z_w * rsq3 / safe_x_w
    x_s, y_s, z_s = latlon2xyz(lon[:, 0], lat[:, 0])
    safe_x_s = jnp.where(jnp.abs(x_s) > 1e-30, x_s, 1.0)
    pp2_j0 = -y_s * rsq3 / safe_x_s
    pp3_j0 = -z_s * rsq3 / safe_x_s

    pp1 = jnp.full((m, m), -rsq3)
    pp2 = jnp.broadcast_to(pp2_j0[:, None], (m, m))
    pp3 = jnp.broadcast_to(pp3_i0[None, :], (m, m))
    pp3 = pp3.at[:, 0].set(pp3_j0)
    pp2 = pp2.at[0, :].set(pp2_i0)
    pp3 = pp3.at[:, im].set(-pp3_j0)
    pp2 = pp2.at[im, :].set(-pp2_i0)

    return xyz2latlon(pp1, pp2, pp3)


def gnomonic_ed_padded_centers(n: int, halo: int) -> tuple[jax.Array, jax.Array]:
    """gnomonic_ed cell CENTRES on the padded grid, ``(6, n_big, n_big)`` create-
    numbered, ``n_big = n+2*halo+2`` (the equiangular padded-metric convention).

    Definition-A centres (cell_center2 of the extended CORNER grid) — CONSISTENT
    with :func:`_compute_gnomonic_ed_lonlat` (in-domain block matches to ~1e-15).
    Corner θ are uniform in great-circle edge angle (gnomonic_ed's defining
    property); the construct's interior is invariant to halo extension (the
    mirror diagonal is pinned to ±α), so the in-domain centres reproduce the FV3
    grid and the surrounding cells are the same-face halo extension.  This is
    the single source the padded metric builders use, so grid.lon/lat and the
    metrics share one centre definition (iter67 consistency fix; the earlier
    construct-at-cell-centre-θ approach was range-dependent and inconsistent).
    """
    rsq3 = 1.0 / jnp.sqrt(3.0)
    alpha = float(jnp.arcsin(rsq3))
    dely = 2.0 * alpha / n
    ext = halo + 1
    n_big = n + 2 * halo + 2
    kc = jnp.arange(n_big + 1, dtype=jnp.float64)  # corner nodes
    theta_c = -alpha - ext * dely + kc * dely
    lo, la = _gnomonic_ed_6face_from_theta(theta_c, alpha)  # (6, M, M) corners
    return cell_center2(
        lo[:, :-1, :-1], la[:, :-1, :-1], lo[:, 1:, :-1], la[:, 1:, :-1],
        lo[:, 1:, 1:], la[:, 1:, 1:], lo[:, :-1, 1:], la[:, :-1, 1:],
    )  # (6, n_big, n_big)


def _gnomonic_ed_angle_1d(ncells: int) -> jax.Array:
    """1D gnomonic-ANGLE distribution of the FV3 gnomonic_ed grid: ``ncells+1``
    nodes, symmetric about 0, spanning ``[-π/4, π/4]``, non-uniform (cells wider
    at the face centre, ~√2 narrower at the edges).

    The gnomonic_ed grid is EXACTLY separable per face in gnomonic angle and all
    6 faces are congruent, so the whole grid is
    ``face_gnomonic_to_lonlat(f, meshgrid(this, this))`` — verified to reproduce
    the FV3 native ``make_fv3_native_grid(grid_type=0)`` corners to ~3e-15 over
    all 6 faces, with per-face separability ~1e-16 (iter73).  This is the ed
    analog of equiangular's uniform ``linspace(-π/4, π/4)``: the SAME builder,
    only the 1D node distribution differs.  Extracting it (vs the
    construct→mirror→remap pipeline) lets the cdgrid extended grids extend the
    edge-PERPENDICULAR halo by simple 1D extrapolation in gnomonic angle — which
    the construct cannot do (it pins the W/E edges to the boundary meridians,
    collapsing the perpendicular halo at the cube corners; iter73 W2 bug).
    """
    lon0, lat0 = gnomonic_ed_remap_to_create(
        *make_fv3_native_grid(ncells, grid_type=0))
    lon0 = jnp.asarray(lon0)[0]
    lat0 = jnp.asarray(lat0)[0]
    # face 0 (+x): tan(alpha_x) = y/x varies along i only (separable to ~1e-16)
    x = jnp.cos(lat0) * jnp.cos(lon0)
    y = jnp.cos(lat0) * jnp.sin(lon0)
    return jnp.mean(jnp.arctan2(y, x), axis=1)  # (ncells+1,)


def _gnomonic_ed_extrap1d(a: jax.Array) -> jax.Array:
    """Linear ±1-node extrapolation of a 1D gnomonic-angle array — the smooth
    SAME-FACE ghost node for the cdgrid centred-difference stencils.  Reduces
    EXACTLY to equiangular's ``linspace`` extension (``±dα``) for a uniform
    array, so this is the faithful ed analog (the halo error vs a θ-uniform
    extension is O(cell²), and unlike the construct it never collapses)."""
    return jnp.concatenate([2.0 * a[:1] - a[1:2], a, 2.0 * a[-1:] - a[-2:-1]])


def _gnomonic_ed_faces_from_angle_1d(
    ax_1d: jax.Array, ay_1d: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Build all 6 gnomonic_ed faces from the separable 1D angle array(s) via the
    tested forward map :func:`legoesm.grids.halo.face_gnomonic_to_lonlat` — the
    same builder the equiangular cdgrid uses, only the 1D distribution differs."""
    from legoesm.grids.halo import face_gnomonic_to_lonlat
    if ay_1d is None:
        ay_1d = ax_1d
    ax_mesh, ay_mesh = jnp.meshgrid(ax_1d, ay_1d, indexing="ij")
    los, las = [], []
    for f in range(6):
        lo, la = face_gnomonic_to_lonlat(f, ax_mesh, ay_mesh)
        los.append(lo)
        las.append(la)
    return jnp.stack(los), jnp.stack(las)


def gnomonic_ed_supergrid_lonlat(n: int) -> tuple[jax.Array, jax.Array]:
    """gnomonic_ed 2×-refined SUPERGRID nodes, ``(6, 2n+1, 2n+1)`` create-numbered
    (area_c/dxc/dyc source).  REFINEMENT-CONSISTENT: the even nodes ``[::2, ::2]``
    reproduce the gnomonic_ed corners (the 2n angle array's even entries ARE the
    n-corner angles).  iter73: built from the separable 1D ed angle array via
    :func:`_gnomonic_ed_faces_from_angle_1d` (matches the construct→mirror→remap
    pipeline to ~3e-15) — the ed analog of equiangular's ``linspace`` supergrid.
    """
    return _gnomonic_ed_faces_from_angle_1d(_gnomonic_ed_angle_1d(2 * n))


def gnomonic_ed_corner_ext_lonlat(n: int) -> tuple[jax.Array, jax.Array]:
    """gnomonic_ed corner grid + 1 halo, ``(6, n+3, n+3)`` create-numbered — the
    ed source for the cdgrid corner/edge grid-angle extended grid (the ed analog
    of equiangular's ``linspace(-π/4-dα, π/4+dα, n+3)``).  Interior ``[1:-1,1:-1]``
    = the gnomonic_ed corners; the ±1 halo is the SAME-FACE smooth continuation
    (1D linear extrapolation in gnomonic angle), so the cube-corner centred-
    difference stencils stay non-degenerate.

    iter73: REPLACES the construct→extended-θ halo, which collapsed the edge-
    perpendicular halo onto the boundary meridians at the 4 cube corners (zero-
    width cells → degenerate corner metrics → 8× W2 error vs equiangular).  See
    :func:`_gnomonic_ed_angle_1d`.
    """
    return _gnomonic_ed_faces_from_angle_1d(
        _gnomonic_ed_extrap1d(_gnomonic_ed_angle_1d(n)))


def gnomonic_ed_padded_supergrid_lonlat(n: int) -> tuple[jax.Array, jax.Array]:
    """gnomonic_ed padded supergrid, ``(6, 2n+3, 2n+3)`` create-numbered — the ed
    source for the cdgrid sin_sg/cos_sg (the ed analog of equiangular's
    ``linspace(-π/4-dα/2, π/4+dα/2, 2n+3)``).  Interior ``[1:-1,1:-1]`` = the 2×
    supergrid; the ±1 (half-cell) halo is the SAME-FACE 1D extrapolation.

    iter73: REPLACES the construct halo (which collapsed at the cube corners,
    corrupting the corner sin_sg).  See :func:`_gnomonic_ed_angle_1d`.
    """
    return _gnomonic_ed_faces_from_angle_1d(
        _gnomonic_ed_extrap1d(_gnomonic_ed_angle_1d(2 * n)))


def compute_padded_half_metrics_ed(
    n: int, radius: float, halo: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """gnomonic_ed counterpart of :func:`legoesm.grids.halo.compute_padded_half_metrics`.

    ``hx_ext, hy_ext`` of shape ``(6, n+2*halo, n+2*halo)``, each = half the
    single-cell edge length (``dx/2``) — built on the FV3 gnomonic_ed grid via
    the 2-cell great-circle chord of the definition-A padded CENTRES
    (:func:`gnomonic_ed_padded_centers`), per face (CONSISTENT with grid.lon/lat;
    the chord convention matches the equiangular builder).
    """
    lon, lat = gnomonic_ed_padded_centers(n, halo)  # (6, n_big, n_big)
    cos_lat = jnp.cos(lat)
    x = cos_lat * jnp.cos(lon)
    y = cos_lat * jnp.sin(lon)
    z = jnp.sin(lat)
    dx_chord = jnp.sqrt(
        (x[:, 2:, 1:-1] - x[:, :-2, 1:-1]) ** 2
        + (y[:, 2:, 1:-1] - y[:, :-2, 1:-1]) ** 2
        + (z[:, 2:, 1:-1] - z[:, :-2, 1:-1]) ** 2
    )
    hx = radius * 2.0 * jnp.arcsin(jnp.clip(dx_chord / 2.0, 0.0, 1.0)) * 0.5
    dy_chord = jnp.sqrt(
        (x[:, 1:-1, 2:] - x[:, 1:-1, :-2]) ** 2
        + (y[:, 1:-1, 2:] - y[:, 1:-1, :-2]) ** 2
        + (z[:, 1:-1, 2:] - z[:, 1:-1, :-2]) ** 2
    )
    hy = radius * 2.0 * jnp.arcsin(jnp.clip(dy_chord / 2.0, 0.0, 1.0)) * 0.5
    return hx, hy  # (6, n+2*halo, n+2*halo)


def compute_padded_angle_ed(n: int, halo: int = 1) -> jax.Array:
    """gnomonic_ed counterpart of :func:`legoesm.grids.halo.compute_padded_angle`.

    Grid angle on the padded grid, ``(6, n+2*halo, n+2*halo)`` — per-face
    i-direction centered-difference angle (identical formula to the equiangular
    builder) of the definition-A padded CENTRES (CONSISTENT with grid.lon/lat).
    Angle is face-dependent (equatorial faces 0-3 share one value; polar 4-5
    differ by π — a real N/S orientation flip, present in the equiangular grid).
    """
    lon6, lat6 = gnomonic_ed_padded_centers(n, halo)  # (6, n_big, n_big)
    all_angle = []
    for f in range(6):
        lon, lat = lon6[f], lat6[f]
        dlon_dx = lon[2:, 1:-1] - lon[:-2, 1:-1]
        dlat_dx = lat[2:, 1:-1] - lat[:-2, 1:-1]
        dlon_dx = jnp.where(dlon_dx > jnp.pi, dlon_dx - 2 * jnp.pi, dlon_dx)
        dlon_dx = jnp.where(dlon_dx < -jnp.pi, dlon_dx + 2 * jnp.pi, dlon_dx)
        cos_lat_ext = jnp.cos(lat[1:-1, 1:-1])
        all_angle.append(jnp.arctan2(dlat_dx, dlon_dx * cos_lat_ext))
    return jnp.stack(all_angle, axis=0)


def _compute_exact_cell_areas_ed(n: int, radius: float) -> jax.Array:
    """gnomonic_ed counterpart of :func:`_compute_exact_cell_areas`.

    Spherical-excess cell areas on the FV3 gnomonic_ed grid, ``(6, n, n)``,
    via the faithful FV3 :func:`get_area` applied to the remapped (create-
    numbered) gnomonic_ed corners.
    """
    lon_c, lat_c = make_fv3_native_grid(n, grid_type=0)
    lon_c, lat_c = gnomonic_ed_remap_to_create(lon_c, lat_c)  # (6, n+1, n+1)
    return get_area(
        lon_c[:, :-1, :-1], lat_c[:, :-1, :-1],   # SW
        lon_c[:, 1:, :-1], lat_c[:, 1:, :-1],     # SE
        lon_c[:, 1:, 1:], lat_c[:, 1:, 1:],       # NE
        lon_c[:, :-1, 1:], lat_c[:, :-1, 1:],     # NW
        radius=radius,
    )


def symm_ed(
    lamda: jax.Array, theta: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 619: enforce ED-grid symmetry about i/j midplanes.

    Faithful port of FV3 ``symm_ed`` (fv_grid_utils.F90:1587-1626).
    Operates on a face-2 ED grid of shape ``(im+1, im+1)`` and
    enforces symmetry in both axes via three passes:

    1. Copy lamda's first column into all interior columns
       (FV3 lines 1595-1599).
    2. Symmetrize about i=im/2+1: pair (i, im+2-i) gets avg/π
       reflection (lines 1601-1611).
    3. Symmetrize about j=im/2+1: pair (j, im+2-j) gets avg in
       theta (with sign flip) and avg in lamda (lines 1614-1624).

    Assumes input is the FV3 face-2 orientation produced by
    ``gnomonic_dist`` — symmetries use the FV3 ``+π/-π``
    convention that's specific to that face orientation.

    Parameters
    ----------
    lamda, theta : jax.Array, shape ``(im+1, im+1)``
        Longitude, latitude in radians (FV3 face-2 layout).

    Returns
    -------
    lamda_sym, theta_sym : jax.Array, shape ``(im+1, im+1)``
        Symmetrized grid.
    """
    n = lamda.shape[0]
    im = n - 1
    pi = jnp.pi

    # Step 1: lamda[i, 1:im+1] = lamda[i, 0] for i ∈ [1, im-1]
    # FV3 lines 1595-1599: only interior columns (j>0) updated;
    # row i=0 and i=im untouched.
    lamda = lamda.at[1:im, 1:im + 1].set(lamda[1:im, 0:1])

    # Step 2: symmetrize about i=im/2+1 (FV3 1601-1611)
    i_half = im // 2
    i_idx = jnp.arange(i_half)
    ip_idx = im - i_idx
    avg_lon = 0.5 * (lamda[i_idx, :] - lamda[ip_idx, :])
    lamda = lamda.at[i_idx, :].set(avg_lon + pi)
    lamda = lamda.at[ip_idx, :].set(pi - avg_lon)
    avg_lat = 0.5 * (theta[i_idx, :] + theta[ip_idx, :])
    theta = theta.at[i_idx, :].set(avg_lat)
    theta = theta.at[ip_idx, :].set(avg_lat)

    # Step 3: symmetrize about j=im/2+1 (FV3 1614-1624)
    # Only i ∈ [1, im-1] (interior columns) are updated
    j_half = im // 2
    j_idx = jnp.arange(j_half)
    jp_idx = im - j_idx
    int_i = slice(1, im)
    avg_lon_j = 0.5 * (lamda[int_i, :][:, j_idx] + lamda[int_i, :][:, jp_idx])
    lamda = lamda.at[int_i, j_idx].set(avg_lon_j)
    lamda = lamda.at[int_i, jp_idx].set(avg_lon_j)
    avg_lat_j = 0.5 * (theta[int_i, :][:, j_idx] - theta[int_i, :][:, jp_idx])
    theta = theta.at[int_i, j_idx].set(avg_lat_j)
    theta = theta.at[int_i, jp_idx].set(-avg_lat_j)

    return lamda, theta


def gnomonic_angl(im: int) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 617: equi-angular gnomonic grid for FV3 face 2.

    Faithful port of FV3 ``gnomonic_angl`` (fv_grid_utils.F90:
    1531-1556).  Builds the canonical FV3 equi-angular cubed-
    sphere grid for face 2 (-x face)::

        dp = π/(2·im)
        p1 = -1/√3                                (constant)
        p2 = -1/√3 · tan(-π/4 + (j-1)·dp)
        p3 =  1/√3 · tan(-π/4 + (k-1)·dp)

    Then ``cart_to_latlon`` to (lon, lat).

    Parameters
    ----------
    im : int
        Number of cells per face edge.  Grid has shape ``(im+1, im+1)``.

    Returns
    -------
    lon, lat : jax.Array, shape ``(im+1, im+1)``
        Cubed-sphere corner positions in radians.
    """
    dp = 0.5 * jnp.pi / im
    rsq3 = 1.0 / jnp.sqrt(3.0)
    idx = jnp.arange(im + 1, dtype=jnp.float64)
    # Match FV3 (j, k) layout: j varies axis 0, k varies axis 1
    j_grid, k_grid = jnp.meshgrid(idx, idx, indexing="ij")
    p1 = jnp.full_like(j_grid, -rsq3)
    p2 = -rsq3 * jnp.tan(-0.25 * jnp.pi + j_grid * dp)
    p3 = rsq3 * jnp.tan(-0.25 * jnp.pi + k_grid * dp)
    return xyz2latlon(p1, p2, p3)


def gnomonic_dist(im: int) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 617: equi-distance gnomonic grid for FV3 face 2.

    Faithful port of FV3 ``gnomonic_dist`` (fv_grid_utils.F90:
    1558-1585).  Builds the equi-distance cubed-sphere grid for
    face 2 (-x face)::

        p1 = -1/√3                                (constant)
        p2 =  1/√3 - (j-1)·2/(im·√3)
        p3 = -1/√3 + (k-1)·2/(im·√3)

    Then ``cart_to_latlon`` to (lon, lat).

    Same return convention as ``gnomonic_angl``.
    """
    rsq3 = 1.0 / jnp.sqrt(3.0)
    xf = -rsq3
    y0 = rsq3
    dy = -2.0 * rsq3 / im
    z0 = -rsq3
    dz = 2.0 * rsq3 / im
    idx = jnp.arange(im + 1, dtype=jnp.float64)
    j_grid, k_grid = jnp.meshgrid(idx, idx, indexing="ij")
    p1 = jnp.full_like(j_grid, xf)
    p2 = y0 + j_grid * dy
    p3 = z0 + k_grid * dz
    return xyz2latlon(p1, p2, p3)


def rotate_winds_sphere_cube(
    u: jax.Array, v: jax.Array,
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon3: jax.Array, lat3: jax.Array,
    lon4: jax.Array, lat4: jax.Array,
    lon_t: jax.Array, lat_t: jax.Array,
    direction: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 661: rotate winds between sphere and cube frames at point.

    Faithful JAX port of FV3 ``rotate_winds``
    (tools/test_cases.F90:8183-8226).

    Geometry: at central point ``t1=(lon_t, lat_t)``, the i-axis
    of the cube goes from p3 → p1 (projected to tangent plane);
    j-axis goes from p4 → p2.  FV3 lon-shift by π convention is
    applied to (e_lon, e_lat) of the geographic frame at t1.

    Algorithm:

        ee1 = get_unit_vector_fv3(p3, t1, p1)        # cube i-axis
        ee2 = get_unit_vector_fv3(p4, t1, p2)        # cube j-axis
        elon = (-sin(λ-π), cos(λ-π), 0)              # geo east at t1
        elat = (-sin(φ)·cos(λ-π), -sin(φ)·sin(λ-π), cos(φ))
        g_ij = ee_i · e_lonlat_j
        if dir=1 (sphere → cube):
            newu = u·g11 + v·g12
            newv = u·g21 + v·g22
        else (cube → sphere):
            det = g11·g22 - g21·g12
            newu = (u·g22 - v·g12) / det
            newv = (-u·g21 + v·g11) / det

    Parameters
    ----------
    u, v : jax.Array
        Wind components (broadcastable to scalar or matching p1..t1).
    lon1, lat1, ..., lon4, lat4 : jax.Array
        4 neighboring points (p1, p2, p3, p4) in lat/lon.
    lon_t, lat_t : jax.Array
        Central point t1.
    direction : int, default 1
        1 = sphere-to-cube; 2 = cube-to-sphere.

    Returns
    -------
    newu, newv : jax.Array
    """
    ee1 = get_unit_vector_fv3(lon3, lat3, lon_t, lat_t, lon1, lat1)
    ee2 = get_unit_vector_fv3(lon4, lat4, lon_t, lat_t, lon2, lat2)
    # FV3 lon-shift by π convention
    lon_shifted = lon_t - jnp.pi
    sin_lon = jnp.sin(lon_shifted)
    cos_lon = jnp.cos(lon_shifted)
    sin_lat = jnp.sin(lat_t)
    cos_lat = jnp.cos(lat_t)
    elon = jnp.stack([-sin_lon, cos_lon, jnp.zeros_like(sin_lon)], axis=-1)
    elat = jnp.stack(
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=-1,
    )
    g11 = inner_prod(ee1, elon)
    g12 = inner_prod(ee1, elat)
    g21 = inner_prod(ee2, elon)
    g22 = inner_prod(ee2, elat)
    if direction == 1:
        new_u = u * g11 + v * g12
        new_v = u * g21 + v * g22
    elif direction == 2:
        det = g11 * g22 - g21 * g12
        safe_det = jnp.where(jnp.abs(det) > 1e-30, det, 1.0)
        new_u = (u * g22 - v * g12) / safe_det
        new_v = (-u * g21 + v * g11) / safe_det
    else:
        raise ValueError(f"direction must be 1 or 2, got {direction}")
    return new_u, new_v


def dcmip16_bc_uwind_pert(
    z: jax.Array, lat: jax.Array, lon: jax.Array,
    up: float = 1.0,
    zp: float = 1.5e4,
    Rp: float | None = None,
    center_lon: float | None = None,
    center_lat: float | None = None,
) -> jax.Array:
    """FV3_3D iter 671: DCMIP16 BC localized wind perturbation.

    Faithful JAX port of FV3 ``DCMIP16_BC_uwind_pert``
    (tools/test_cases.F90:6823-6838).  Localized Gaussian-in-x,
    Hermite-cubic-in-z wind perturbation for triggering the
    baroclinic instability in DCMIP16 Test 410.

    Algorithm:
        zrat = z / zp
        ZZ   = max(1 - 3·zrat² + 2·zrat³, 0)        (Hermite vertical taper)
        dst  = great_circle_distance(point, center)
        pert = max(0, up · ZZ · exp(-(dst/Rp)²))

    Default FV3 constants:
        up=1 m/s (peak amplitude)
        zp=15000 m (vertical scale)
        Rp=R_earth/10 (horizontal scale)
        center = (π/9, 2π/9) (FV3 perturbation focal point)

    Parameters
    ----------
    z : jax.Array
        Height (m).
    lat, lon : jax.Array
        Cell-center positions (radians).
    up, zp, Rp, center_lon, center_lat : float, optional
        DCMIP16 BC perturbation parameters.

    Returns
    -------
    pert : jax.Array
        Wind perturbation (m/s).
    """
    pi = jnp.pi
    if Rp is None:
        Rp = constants.R_earth / 10.0
    if center_lon is None:
        center_lon = pi / 9.0
    if center_lat is None:
        center_lat = 2.0 * pi / 9.0
    zrat = z / zp
    ZZ = jnp.maximum(1.0 - 3.0 * zrat * zrat + 2.0 * zrat * zrat * zrat, 0.0)
    dst = great_circle_distance(
        lon, lat,
        jnp.asarray(center_lon), jnp.asarray(center_lat),
        radius=constants.R_earth,
    )
    return jnp.maximum(0.0, up * ZZ * jnp.exp(-((dst / Rp) ** 2)))


def dcmip16_bc_uwind(
    z: jax.Array, T: jax.Array, lat: jax.Array,
    KK: float = 3.0,
    Te: float = 310.0,
    Tp: float = 240.0,
    b: float = 2.0,
) -> jax.Array:
    """FV3_3D iter 668: DCMIP16 BC zonal wind profile.

    Faithful JAX port of FV3 ``DCMIP16_BC_uwind``
    (tools/test_cases.F90:6807-6821).  Baroclinic-wind profile
    derived from T via geostrophic balance + centripetal::

        Tir = z·exp(-(z·g/(b·R_d·T0))²)
        Ti2 = 0.5·(K+2)·(Te-Tp)/(Te·Tp)·Tir
        UU  = g·K/R · Ti2 · (cos(lat)^(K-1) - cos(lat)^(K+1)) · T
        u   = -Ω·R·cos(lat) + sqrt((Ω·R·cos(lat))² + R·cos(lat)·UU)

    Used with iter-667 ``dcmip16_bc_temperature`` to build the
    DCMIP16 Test 410 IC.

    Parameters
    ----------
    z : jax.Array
        Height (m).
    T : jax.Array
        Temperature (K), from ``dcmip16_bc_temperature``.
    lat : jax.Array
        Latitude (radians).
    """
    g = constants.g
    Rdgas = constants.R_d
    radius = constants.R_earth
    omega = constants.Omega
    T0 = 0.5 * (Te + Tp)
    zsc = z * g / (b * Rdgas * T0)
    Tir = z * jnp.exp(-zsc * zsc)
    Ti2 = 0.5 * (KK + 2.0) * (Te - Tp) / (Te * Tp) * Tir
    cos_lat = jnp.cos(lat)
    K_int = int(KK)
    UU = (
        g * KK / radius * Ti2
        * (cos_lat ** (K_int - 1) - cos_lat ** (K_int + 1)) * T
    )
    discriminant = (omega * radius * cos_lat) ** 2 + radius * cos_lat * UU
    safe_disc = jnp.maximum(discriminant, 0.0)
    return -omega * radius * cos_lat + jnp.sqrt(safe_disc)


def dcmip16_bc_sphum(
    p: jax.Array, ps: jax.Array, lat: jax.Array,
    q0: float = 0.018,
    qt: float = 1.0e-12,
    phiW: float | None = None,
    pw: float = 34000.0,
    p0: float = 1.0e5,
    ptrop: float = 1.0e4,
) -> jax.Array:
    """FV3_3D iter 668: DCMIP16 BC specific humidity profile.

    Faithful JAX port of FV3 ``DCMIP16_BC_sphum``
    (tools/test_cases.F90:6840-6852).

    Algorithm:

        eta = p / ps
        if p > ptrop:
            q = q0·exp(-(lat/phiW)⁴)·exp(-((eta-1)·p0/pw)²)
        else:
            q = qt

    Default DCMIP16 BC constants (FV3 lines 6499-6503):
        q0=0.018, qt=1e-12, phiW=2π/9, pw=34000, p0=1e5, ptrop=1e4

    Used in FV3 DCMIP16 Test 410 BC moist IC.
    """
    if phiW is None:
        phiW = 2.0 * jnp.pi / 9.0
    eta = p / ps
    q_moist = (
        q0
        * jnp.exp(-((lat / phiW) ** 4))
        * jnp.exp(-((eta - 1.0) * p0 / pw) ** 2)
    )
    return jnp.where(p > ptrop, q_moist, qt)


def dcmip16_bc_temperature(
    z: jax.Array, lat: jax.Array,
    KK: float = 3.0,
    Te: float = 310.0,
    Tp: float = 240.0,
    b: float = 2.0,
    lapse: float = 0.005,
) -> jax.Array:
    """FV3_3D iter 667: DCMIP16 baroclinic-instability temperature profile.

    Faithful JAX port of FV3 ``DCMIP16_BC_temperature``
    (tools/test_cases.F90:6774-6789).  Jablonowski-Williamson
    BC test temperature::

        IT = cos(lat)^K - K/(K+2) · cos(lat)^(K+2)
        zsc = z·g/(b·R_d·T0)
        Tr = (1 - 2·zsc²) · exp(-zsc²)
        T1 = (1/T0)·exp(lapse·z/T0) + (T0-Tp)/(T0·Tp)·Tr
        T2 = 0.5·(K+2)·(Te-Tp)/(Te·Tp)·Tr
        T  = 1 / (T1 - T2·IT)

    Default DCMIP16 BC constants:
        KK = 3 (zonal wave number)
        Te = 310 K, Tp = 240 K, T0 = (Te+Tp)/2 = 275 K
        b = 2, lapse = 0.005 K/m
    """
    g = constants.g
    Rdgas = constants.R_d
    T0 = 0.5 * (Te + Tp)  # FV3 note: WRONG in document, here = 275
    IT = (
        jnp.cos(lat) ** KK
        - KK / (KK + 2.0) * jnp.cos(lat) ** (KK + 2.0)
    )
    zsc = z * g / (b * Rdgas * T0)
    Tr = (1.0 - 2.0 * zsc * zsc) * jnp.exp(-zsc * zsc)
    T1 = (1.0 / T0) * jnp.exp(lapse * z / T0) + (T0 - Tp) / (T0 * Tp) * Tr
    T2 = 0.5 * (KK + 2.0) * (Te - Tp) / (Te * Tp) * Tr
    return 1.0 / (T1 - T2 * IT)


def dcmip16_bc_pressure(
    z: jax.Array, lat: jax.Array,
    KK: float = 3.0,
    Te: float = 310.0,
    Tp: float = 240.0,
    b: float = 2.0,
    lapse: float = 0.005,
    p0: float = 1.0e5,
) -> jax.Array:
    """FV3_3D iter 667: DCMIP16 BC pressure profile (companion to T).

    Faithful JAX port of FV3 ``DCMIP16_BC_pressure``
    (tools/test_cases.F90:6791-6805):

        IT  = cos(lat)^K - K/(K+2) · cos(lat)^(K+2)
        Tir = z · exp(-(z·g/(b·R_d·T0))²)
        Ti1 = (1/lapse)·(exp(lapse·z/T0) - 1) + Tir·(T0-Tp)/(T0·Tp)
        Ti2 = 0.5·(K+2)·(Te-Tp)/(Te·Tp)·Tir
        p   = p0·exp(-g/R_d · (Ti1 - Ti2·IT))

    Used in FV3 DCMIP16 baroclinic-instability test (Test 410).
    """
    g = constants.g
    Rdgas = constants.R_d
    T0 = 0.5 * (Te + Tp)
    IT = (
        jnp.cos(lat) ** KK
        - KK / (KK + 2.0) * jnp.cos(lat) ** (KK + 2.0)
    )
    zsc = z * g / (b * Rdgas * T0)
    Tir = z * jnp.exp(-zsc * zsc)
    Ti1 = (
        (1.0 / lapse) * (jnp.exp(lapse * z / T0) - 1.0)
        + Tir * (T0 - Tp) / (T0 * Tp)
    )
    Ti2 = 0.5 * (KK + 2.0) * (Te - Tp) / (Te * Tp) * Tir
    return p0 * jnp.exp(-g / Rdgas * (Ti1 - Ti2 * IT))


def bilinear_interp_apply(
    src_field: jax.Array,
    id1: jax.Array, id2: jax.Array, jc: jax.Array,
    s2c: jax.Array,
) -> jax.Array:
    """FV3_3D iter 675: apply bilinear remap weights to source field.

    Faithful JAX port of FV3 ``apply_inc_on_3d_scalar`` core
    (tools/fv_treat_da_inc.F90:339-360, inner bilinear loop).

    Algorithm:

        target[..., i, j] = s2c[..., i, j, 0] · src[id1[i, j], jc[i, j]    ]
                          + s2c[..., i, j, 1] · src[id2[i, j], jc[i, j]    ]
                          + s2c[..., i, j, 2] · src[id2[i, j], jc[i, j]+1  ]
                          + s2c[..., i, j, 3] · src[id1[i, j], jc[i, j]+1  ]

    Pairs with iter-673 ``remap_coef_fv3`` (produces id1, id2, jc, s2c)
    to provide full lat-lon → cubed-sphere bilinear interpolation.

    Parameters
    ----------
    src_field : jax.Array, shape (im, jm) or (im, jm, km)
        Source field on regular lat-lon grid.  Trailing axes
        broadcast.
    id1, id2 : jax.Array (int), shape (...,)
        Source longitude indices.
    jc : jax.Array (int), shape (...,)
        Source latitude index (jc and jc+1 are used for bilinear).
    s2c : jax.Array, shape (..., 4)
        Bilinear weights (SW, SE, NE, NW).

    Returns
    -------
    target : jax.Array, shape matches id1 (+ trailing dims of src_field)
    """
    # Gather source values at the 4 corners
    f_sw = src_field[id1, jc]                    # (..., [km])
    f_se = src_field[id2, jc]
    f_ne = src_field[id2, jc + 1]
    f_nw = src_field[id1, jc + 1]
    # Combine
    w_sw = s2c[..., 0]
    w_se = s2c[..., 1]
    w_ne = s2c[..., 2]
    w_nw = s2c[..., 3]
    # Add level-axis broadcast if needed
    if f_sw.ndim > id1.ndim:
        extra = f_sw.ndim - id1.ndim
        for _ in range(extra):
            w_sw = w_sw[..., None]
            w_se = w_se[..., None]
            w_ne = w_ne[..., None]
            w_nw = w_nw[..., None]
    return w_sw * f_sw + w_se * f_se + w_ne * f_ne + w_nw * f_nw


def dcmip16_tc_uwind_pert(
    z: jax.Array, r: jax.Array,
    lon: jax.Array, lat: jax.Array,
    Tv0: float | None = None,
    lapse: float = 7.0e-3,
    zt: float = 15000.0,
    rp: float = 282000.0,
    zp: float = 7000.0,
    pb: float = 101500.0,
    dp: float = 1115.0,
    q0: float = 0.021,
    lamp: float = None,
    phip: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 672: DCMIP16 TC vortex wind perturbation.

    Faithful JAX port of FV3 ``DCMIP16_TC_uwind_pert``
    (tools/test_cases.F90:7168-7197).

    Algorithm (z ≤ zt):
        rfac = (r/rp)^1.5
        fr5  = 0.5·fc·r              # fc = 2·Ω·sin(phip)
        Tvrd = (Tv0 - lapse·z)·R_d
        vt = -fr5 + sqrt(fr5² - 1.5·rfac·Tvrd /
                          (1 + 2·Tvrd·z/(g·zp²) - (pb/dp)·exp(rfac + (z/zp)²)))
        d1  = sin(phip)·cos(lat) - cos(phip)·sin(lat)·cos(lon - lamp)
        d2  = cos(phip)·sin(lon - lamp)
        d   = max(1e-25, sqrt(d1² + d2²))
        uu = vt · d1 / d
        vv = vt · d2 / d
    z > zt: uu = vv = 0

    Default FV3 constants:
        lamp = π (TC center longitude)
        phip = π/18 (TC center latitude, ~10°N)
        Tv0  = 302.15·(1+0.608·q0)
        fc   = 2·Ω·sin(phip)

    Used in FV3 DCMIP16 Test 411 (TC).  With iter-666/669/671:
    full TC IC stack available.

    Returns (uu, vv) wind perturbation components.
    """
    pi = jnp.pi
    if Tv0 is None:
        Tv0 = 302.15 * (1.0 + 0.608 * q0)
    if lamp is None:
        lamp = pi
    if phip is None:
        phip = pi / 18.0
    g = constants.g
    Rdgas = constants.R_d
    # iter-778: delegate Coriolis to coriolis_parameter_fv3
    fc = coriolis_parameter_fv3(jnp.asarray(phip))
    rfac = jnp.sqrt(r / rp) ** 3
    fr5 = 0.5 * fc * r
    Tv = Tv0 - lapse * z
    Tvrd = Tv * Rdgas
    denom = (
        1.0
        + 2.0 * Tvrd * z / (g * zp * zp)
        - (pb / dp) * jnp.exp(rfac + (z / zp) ** 2)
    )
    safe_denom = jnp.where(jnp.abs(denom) > 1e-30, denom, 1.0)
    radicand = fr5 ** 2 - (1.5 * rfac * Tvrd) / safe_denom
    vt = -fr5 + jnp.sqrt(jnp.maximum(radicand, 0.0))
    d1 = (
        jnp.sin(phip) * jnp.cos(lat)
        - jnp.cos(phip) * jnp.sin(lat) * jnp.cos(lon - lamp)
    )
    d2 = jnp.cos(phip) * jnp.sin(lon - lamp)
    d = jnp.maximum(1.0e-25, jnp.sqrt(d1 * d1 + d2 * d2))
    uu_below = vt * d1 / d
    vv_below = vt * d2 / d
    uu = jnp.where(z > zt, 0.0, uu_below)
    vv = jnp.where(z > zt, 0.0, vv_below)
    return uu, vv


def dcmip16_tc_temperature(
    z: jax.Array, r: jax.Array,
    Tv0: float | None = None,
    lapse: float = 7.0e-3,
    zt: float = 15000.0,
    rp: float = 282000.0,
    zp: float = 7000.0,
    pb: float = 101500.0,
    dp: float = 1115.0,
    q0: float = 0.021,
) -> jax.Array:
    """FV3_3D iter 669: DCMIP16 TC temperature profile.

    Faithful JAX port of FV3 ``DCMIP16_TC_temperature``
    (tools/test_cases.F90:7137-7152).

    Algorithm:
        z > zt:  T = Tvt = Tv0 - lapse·zt
        else:
          Tv    = Tv0 - lapse·z
          term1 = g·zp²·(1 - (pb/dp)·exp((r/rp)^1.5 + (z/zp)²))
          term2 = 2·R_d·Tv·z
          T     = Tv·(1 + (1/(1 + term2/term1) - 1))

    Used in FV3 DCMIP16 Test 411 (TC).

    Defaults from FV3 lines 6880-6897: Tv0 = 302.15·(1+0.608·q0).

    Parameters
    ----------
    z : jax.Array
        Height (m).
    r : jax.Array
        Great-circle distance from TC center (m).
    Tv0 : float, optional
        Sea-level virtual temperature; default 302.15·(1+0.608·q0).
    lapse, zt, rp, zp, pb, dp, q0 : float
        DCMIP16 TC parameters (see defaults).
    """
    g = constants.g
    Rdgas = constants.R_d
    if Tv0 is None:
        Tv0 = 302.15 * (1.0 + 0.608 * q0)
    Tvt = Tv0 - lapse * zt
    Tv = Tv0 - lapse * z
    rfac = jnp.sqrt(r / rp) ** 3
    term1 = g * zp * zp * (1.0 - (pb / dp) * jnp.exp(rfac + (z / zp) ** 2))
    term2 = 2.0 * Rdgas * Tv * z
    # Safe-divide for term2/term1
    safe_term1 = jnp.where(jnp.abs(term1) > 1e-30, term1, 1.0)
    T_below = Tv + Tv * (1.0 / (1.0 + term2 / safe_term1) - 1.0)
    return jnp.where(z > zt, Tvt, T_below)


def dcmip16_tc_pressure(
    z: jax.Array, r: jax.Array,
    Tv0: float | None = None,
    lapse: float = 7.0e-3,
    zt: float = 15000.0,
    rp: float = 282000.0,
    zp: float = 7000.0,
    pb: float = 101500.0,
    dp: float = 1115.0,
    q0: float = 0.021,
) -> jax.Array:
    """FV3_3D iter 669: DCMIP16 TC pressure profile.

    Faithful JAX port of FV3 ``DCMIP16_TC_pressure``
    (tools/test_cases.F90:7155-7167).

    Algorithm:
        z <= zt:
          p = pb·exp(g/(R_d·lapse)·ln((Tv0-lapse·z)/Tv0))
              - dp·exp(-(r/rp)^1.5 - (z/zp)²)·exp(g/(R_d·lapse)·ln(...))
        z > zt:
          p = ptt·exp(g·(zt-z)/(R_d·Tvt))
          where ptt = pb·(Tvt/Tv0)^(g/R_d/lapse)
    """
    g = constants.g
    Rdgas = constants.R_d
    if Tv0 is None:
        Tv0 = 302.15 * (1.0 + 0.608 * q0)
    Tvt = Tv0 - lapse * zt
    ptt = pb * (Tvt / Tv0) ** (g / (Rdgas * lapse))
    # z <= zt branch
    Tv = Tv0 - lapse * z
    ratio = jnp.maximum(Tv / Tv0, 1e-30)
    p_base = pb * jnp.exp(g / (Rdgas * lapse) * jnp.log(ratio))
    rfac = jnp.sqrt(r / rp) ** 3
    p_below = p_base - dp * jnp.exp(-rfac - (z / zp) ** 2) * jnp.exp(
        g / (Rdgas * lapse) * jnp.log(ratio)
    )
    # z > zt branch
    p_above = ptt * jnp.exp(g * (zt - z) / (Rdgas * Tvt))
    return jnp.where(z <= zt, p_below, p_above)


def dcmip16_tc_sphum(
    z: jax.Array,
    q0: float = 0.021,
    qt: float = 1.0e-11,
    zq1: float = 3000.0,
    zq2: float = 8000.0,
    zt: float = 15000.0,
) -> jax.Array:
    """FV3_3D iter 666: DCMIP16 Reed-Jablonowski TC humidity profile.

    Faithful JAX port of FV3 ``DCMIP16_TC_sphum`` (tools/test_cases.F90:
    7198-7208, DCMIP16 TC test).  Specific humidity (kg/kg) as
    function of height:

        if z >= zt: q = qt   (stratospheric background)
        else:       q = q0 · exp(-z/zq1) · exp(-(z/zq2)²)

    Default DCMIP16 TC constants (FV3 lines 6880-6886):
        q0  = 0.021 kg/kg  (surface peak)
        qt  = 1e-11 kg/kg  (stratospheric)
        zq1 = 3000 m       (exponential decay)
        zq2 = 8000 m       (Gaussian truncation)
        zt  = 15000 m      (tropopause)

    Used in FV3 DCMIP16 idealized tropical-cyclone test.

    Parameters
    ----------
    z : jax.Array
        Height(s) in meters.
    q0, qt, zq1, zq2, zt : float
        TC profile constants (see defaults).

    Returns
    -------
    q : jax.Array
        Specific humidity (kg/kg).
    """
    q_below = q0 * jnp.exp(-z / zq1) * jnp.exp(-(z / zq2) ** 2)
    return jnp.where(z < zt, q_below, qt)


def case9_B(
    lon: jax.Array, lat: jax.Array, gh0: float | None = None,
) -> jax.Array:
    """FV3_3D iter 664: Williamson case 9 spatial forcing pattern B(λ, φ).

    Faithful JAX port of FV3 ``get_case9_B`` (tools/test_cases.F90:
    4361-4389).  Returns::

        if sin(φ) > 0:
            yy = (cos(φ) / sin(φ))² = cot²(φ)
            B  = gh0 · yy · exp(1 - yy) · sin(λ)
        else:
            B = 0

    Default gh0 = 720·g (FV3 calibrated for the SW orographic
    forcing test).  The forcing peaks where yy=1 (i.e., lat=π/4)
    with magnitude gh0·sin(λ).

    Parameters
    ----------
    lon, lat : jax.Array
        Cell-center positions (radians).
    gh0 : float, optional
        Forcing peak amplitude (default 720·g).

    Returns
    -------
    B : jax.Array
        Spatial forcing field.
    """
    if gh0 is None:
        gh0 = 720.0 * constants.g
    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)
    # Safe cot²: avoid divide-by-zero at equator and poles
    safe_sin = jnp.where(jnp.abs(sin_lat) > 1e-30, sin_lat, 1.0)
    yy = (cos_lat / safe_sin) ** 2
    myB = gh0 * yy * jnp.exp(1.0 - yy)
    B = myB * jnp.sin(lon)
    # Zero in southern hemisphere (and equator)
    return jnp.where(sin_lat > 0.0, B, 0.0)


def case9_AofT(
    tday: jax.Array,
) -> jax.Array:
    """FV3_3D iter 664: Williamson case 9 amplitude modulation AofT(t).

    Faithful JAX port of FV3 ``case9_forcing1`` amplitude logic
    (tools/test_cases.F90:4391-4424).  Time-varying amplitude:

        tday ≤ 4:           A = 0.5·(1 - cos(π·tday/4))   [ramp up]
        4 < tday ≤ 16:      A = 1                          [peak]
        16 < tday ≤ 20:     A = 0.5·(1 + cos(π·(tday-16)/4)) [ramp down]
        tday > 20:          A = 0.5·(1 - cos(π·(tday-20)/4)) [new cycle]

    Parameters
    ----------
    tday : jax.Array
        Time in days.

    Returns
    -------
    A : jax.Array
        Amplitude ∈ [0, 1].
    """
    pi = jnp.pi
    ramp_up = 0.5 * (1.0 - jnp.cos(0.25 * pi * tday))
    peak = jnp.ones_like(ramp_up)
    ramp_down = 0.5 * (1.0 + jnp.cos(0.25 * pi * (tday - 16.0)))
    new_cycle = 0.5 * (1.0 - jnp.cos(0.25 * pi * (tday - 20.0)))
    A = jnp.where(
        tday <= 4.0, ramp_up,
        jnp.where(
            tday <= 16.0, peak,
            jnp.where(tday <= 20.0, ramp_down, new_cycle),
        ),
    )
    return A


def add_rankine_vortex(
    u: jax.Array, v: jax.Array,
    grid_lon: jax.Array, grid_lat: jax.Array,
    ubar: float, r0: float,
    center_lon: float, center_lat: float,
    radius: float = constants.R_earth,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 662: add Rankine vortex to D-grid winds.

    Faithful JAX port of FV3 ``rankine_vortex``
    (tools/test_cases.F90:4207-4292).  Adds a Rankine-vortex
    tangential wind onto the D-grid u, v fields at face corners,
    projected onto the local cube-grid tangent vectors.

    Tangential wind profile:
        vr = ubar · r/r0    if r < r0  (solid-body core)
        vr = ubar · r0/r    if r ≥ r0  (1/r decay outside)

    where r is great-circle distance from cell-edge midpoint to
    vortex center (radius·acos(cos_p)).

    Parameters
    ----------
    u : jax.Array, shape (..., n_x, n_y+1)
        D-grid u-wind (north/south edges); updated in-place style.
    v : jax.Array, shape (..., n_x+1, n_y)
        D-grid v-wind (east/west edges).
    grid_lon : jax.Array, shape (..., n_x+1, n_y+1)
        Cubed-sphere corner longitudes.
    grid_lat : jax.Array, shape (..., n_x+1, n_y+1)
        Cubed-sphere corner latitudes.
    ubar : float
        Maximum tangential wind (m/s).
    r0 : float
        Radius of maximum wind (m).
    center_lon, center_lat : float
        Vortex center (radians).
    radius : float, default constants.R_earth
        Sphere radius (m).

    Returns
    -------
    u_new, v_new : jax.Array
        D-grid winds with vortex added.
    """

    def _tangential_wind_at(p2_lon, p2_lat):
        """Compute vortex contributions (utmp, vtmp) at point p2."""
        # Shift p2_lon by -center_lon
        p2_lon_s = p2_lon - center_lon
        cos_p = (
            jnp.sin(p2_lat) * jnp.sin(center_lat)
            + jnp.cos(p2_lat) * jnp.cos(center_lat) * jnp.cos(p2_lon_s)
        )
        cos_p = jnp.clip(cos_p, -1.0, 1.0)
        r = radius * jnp.arccos(cos_p)
        # Tangential wind magnitude
        vr_inside = ubar * r / r0
        vr_outside = ubar * r0 / jnp.maximum(r, 1e-30)
        vr = jnp.where(r < r0, vr_inside, vr_outside)
        # Direction of vortex motion (in shifted frame)
        x1 = jnp.cos(p2_lat) * jnp.sin(p2_lon_s)
        y1 = (
            jnp.sin(p2_lat) * jnp.cos(center_lat)
            - jnp.cos(p2_lat) * jnp.sin(center_lat) * jnp.cos(p2_lon_s)
        )
        d2 = jnp.maximum(jnp.sqrt(x1 * x1 + y1 * y1), 1.0e-25)
        utmp = -vr * y1 / d2
        vtmp = vr * x1 / d2
        # Return utmp, vtmp + shifted p2 for elon/elat
        return utmp, vtmp, p2_lon_s, p2_lat

    # ---- u-wind on j-edges: average grid[i, j] and grid[i+1, j] in lon
    # u shape (..., n_x, n_y+1); grid shape (..., n_x+1, n_y+1)
    # j-edge midpoint p2[i, j] = mid_pt_sphere(grid[i, j], grid[i+1, j])
    sw_lon = grid_lon[..., :-1, :]   # (..., n_x, n_y+1)
    sw_lat = grid_lat[..., :-1, :]
    se_lon = grid_lon[..., 1:, :]
    se_lat = grid_lat[..., 1:, :]
    p2_lon_u, p2_lat_u = mid_pt_sphere(sw_lon, sw_lat, se_lon, se_lat)
    utmp_u, vtmp_u, p2_lon_s, p2_lat_s = _tangential_wind_at(p2_lon_u, p2_lat_u)
    # Cube tangent e1 at p2 from p3=(grid[i,j]-center, grid[i,j].lat)
    # to p4=(grid[i+1,j]-center, grid[i+1,j].lat)
    e1 = get_unit_vect2(
        sw_lon - center_lon, sw_lat,
        se_lon - center_lon, se_lat,
    )
    elon_u, elat_u = unit_vect_latlon(p2_lon_s, p2_lat_s)
    u_add = utmp_u * inner_prod(e1, elon_u) + vtmp_u * inner_prod(e1, elat_u)
    u_new = u + u_add

    # ---- v-wind on i-edges: average grid[i, j] and grid[i, j+1] in lat
    # v shape (..., n_x+1, n_y); grid shape (..., n_x+1, n_y+1)
    s_lon = grid_lon[..., :, :-1]
    s_lat = grid_lat[..., :, :-1]
    n_lon = grid_lon[..., :, 1:]
    n_lat = grid_lat[..., :, 1:]
    p2_lon_v, p2_lat_v = mid_pt_sphere(s_lon, s_lat, n_lon, n_lat)
    utmp_v, vtmp_v, p2_lon_s2, p2_lat_s2 = _tangential_wind_at(p2_lon_v, p2_lat_v)
    e2 = get_unit_vect2(
        s_lon - center_lon, s_lat,
        n_lon - center_lon, n_lat,
    )
    elon_v, elat_v = unit_vect_latlon(p2_lon_s2, p2_lat_s2)
    v_add = utmp_v * inner_prod(e2, elon_v) + vtmp_v * inner_prod(e2, elat_v)
    v_new = v + v_add
    return u_new, v_new


def project_sphere_v(
    f: jax.Array, e: jax.Array,
) -> jax.Array:
    """FV3_3D iter 659: project vector onto sphere-tangent plane.

    Faithful JAX port of FV3 ``project_sphere_v``
    (fv_grid_utils.F90:3345-3361).  Given a unit-sphere position
    ``e`` and a 3-vector ``f``, returns ``f`` projected onto the
    tangent plane at ``e``::

        ap = f · e
        f_tangent = f - ap·e

    Takes the last axis as the 3-vector component; broadcasts on
    leading axes.
    """
    ap = jnp.sum(f * e, axis=-1, keepdims=True)
    return f - ap * e


def get_unit_vector_fv3(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon3: jax.Array, lat3: jax.Array,
) -> jax.Array:
    """FV3_3D iter 659 (restored iter-1069): unit tangent vector at p2
    from p1 → p3.

    Faithful JAX port of FV3 ``get_unit_vector``
    (``tools/test_cases.F90:8366-8385``).  Used by FV3-faithful
    test-case wind initialization paths (``rotate_winds_fv3``,
    ``dcmip16_tc_rotate_winds``) for the DCMIP-16 baroclinic-wave and
    tropical-cyclone test cases.  Currently called only from those
    test-init paths; placed here (rather than in test utilities)
    because it is a faithful FV3 oracle port that may be reused by
    future production code paths that perform sphere-to-cube wind
    rotation outside the standard ``rotate_winds_geo_to_grid`` flow.

    Algorithm:

        xyz1, xyz2, xyz3 = latlon2xyz(...)
        uvect = xyz3 - xyz1                     # chord
        uvect = project_sphere_v(uvect, xyz2)   # tangent at p2
        uvect = normalize(uvect)

    Returns the unit tangent vector at ``p2`` pointing in the
    direction from ``p1`` toward ``p3`` (projected onto the local
    tangent plane).

    Note: iter 905 (commit c1c0e42b) removed this definition during
    drift cleanup but left the callers in
    ``rotate_winds_fv3`` / ``dcmip16_tc_rotate_winds`` referencing
    it, which caused ``NameError`` at test time
    (test_fv3_rotate_winds_iter661.py / test_fv3_tc_uwind_pert_iter672.py).
    iter-1069 restores it.

    Returns shape ``(..., 3)``; broadcasts on leading axes.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    x3, y3, z3 = latlon2xyz(lon3, lat3)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    uvect_raw = jnp.stack([x3 - x1, y3 - y1, z3 - z1], axis=-1)
    uvect_tangent = project_sphere_v(uvect_raw, p2)
    return normalize_vect(uvect_tangent)


def coriolis_parameter_fv3(
    lat: jax.Array,
    units: str = "rad",
) -> jax.Array:
    """FV3_3D iter 778 (restored iter-1069): Coriolis parameter
    ``f = 2·Ω·sin(lat)``.

    Vertical component of the planetary vorticity vector
    (``2·Ω·sin(lat)``) acting on horizontal flow.  Centred on Earth:
    ``Ω = constants.Omega`` = 7.292·10⁻⁵ rad/s.

    Currently called only by ``dcmip16_tc_uwind_pert`` (the DCMIP-16
    tropical-cyclone wind initialization).  Placed here (rather than
    in test utilities) because it is a faithful FV3 oracle port of
    a fundamental geophysical quantity that may be reused by future
    production code paths (geostrophic balance, Rossby-wave
    dispersion, Ekman pumping, etc.).

    Note: iter 905 (commit c1c0e42b) removed this definition during
    drift cleanup but left the caller in ``dcmip16_tc_uwind_pert``
    referencing it, which caused ``NameError`` at test time.
    iter-1069 restores it.

    Parameters
    ----------
    lat : jax.Array
        Latitude (radians by default; pass ``units='deg'`` for
        degrees input).
    units : {'rad', 'deg'}

    Returns
    -------
    f : jax.Array
        Coriolis parameter (s⁻¹).
    """
    if units not in ("rad", "deg"):
        raise ValueError(f"units must be 'rad' or 'deg', got {units!r}")
    lat_rad = jnp.radians(lat) if units == "deg" else lat
    return 2.0 * constants.Omega * jnp.sin(lat_rad)


def terminator_tracers(
    lon: jax.Array, lat: jax.Array,
    km: int,
    qcly: float = 4.0e-6,
    k2: float = 1.0,
    lc: float | None = None,
    thc: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 658: DCMIP 2016 terminator chemistry tracer IC.

    Faithful JAX port of FV3 ``terminator_tracers``
    (tools/test_cases.F90:4136-4205).  DCMIP 2016 idealized
    chemistry test (Lauritzen et al.); paired Cl / Cl2 tracers
    that exchange under photolysis at a localized "sun":

        k1   = max(0, sin(lat)·sin(thc) + cos(lat)·cos(thc)·cos(lon - lc))
        r    = k1/k2 · 0.25
        D    = sqrt(r² + 2·r·qcly)
        Cl   = D - r
        Cl2  = 0.5·(qcly - Cl)

    Same pattern at every vertical level.  Assumes DRY mixing
    ratio (FV3 docstring note).

    Default sun position lc=5π/3, thc=π/9 matches FV3.

    Parameters
    ----------
    lon, lat : jax.Array, shape (..., n_x, n_y)
        Cell-center positions in radians.
    km : int
        Number of vertical levels.
    qcly : float, default 4e-6
        Total chlorine family mixing ratio (kg/kg, DRY).
    k2 : float, default 1.0
        Recombination rate constant.
    lc, thc : float, optional
        Sun position (radians); default FV3 values.

    Returns
    -------
    Cl, Cl2 : jax.Array, shape (..., n_x, n_y, km)
        Chemical species mixing ratios.
    """
    if lc is None:
        lc = 5.0 * jnp.pi / 3.0
    if thc is None:
        thc = jnp.pi / 9.0
    sinthc = jnp.sin(thc)
    costhc = jnp.cos(thc)
    cos_phot = (
        jnp.sin(lat) * sinthc
        + jnp.cos(lat) * costhc * jnp.cos(lon - lc)
    )
    k1 = jnp.maximum(0.0, cos_phot)
    r = k1 / k2 * 0.25
    D = jnp.sqrt(r * r + 2.0 * r * qcly)
    Cl_2d = D - r
    Cl2_2d = 0.5 * (qcly - Cl_2d)
    # Broadcast over km
    Cl = jnp.broadcast_to(Cl_2d[..., None], Cl_2d.shape + (km,))
    Cl2 = jnp.broadcast_to(Cl2_2d[..., None], Cl2_2d.shape + (km,))
    return Cl, Cl2


def checker_tracers(
    lon: jax.Array, lat: jax.Array,
    nq: int, km: int,
    nx: float = 9.0, ny: float = 9.0,
    rn: float | None = None,
    rng_key: jax.Array | None = None,
) -> jax.Array:
    """FV3_3D iter 657: checkerboard tracer pattern with optional noise.

    Faithful JAX port of FV3 ``checker_tracers`` (tools/test_cases.F90:
    4067-4135).  Builds a checkerboard tracer pattern based on::

        qt[i, j] = 0.01  if sin(nx·lon)·sin(ny·lat) > 0
                   0     otherwise

    Defaults nx=ny=9 give 20°×20° checker boxes (per FV3 docstring).
    Optional ``rn`` adds uniform random perturbation rn·U(0,1).
    Broadcast across vertical levels (km) and tracer count (nq).

    Coded for the HIWPP benchmark by S.-J. Lin (2014).

    Parameters
    ----------
    lon, lat : jax.Array, shape (..., n_x, n_y)
        Cell-center positions in radians.
    nq : int
        Number of tracers.
    km : int
        Number of vertical levels.
    nx, ny : float, default 9.0
        East-west / North-south wave numbers.
    rn : float, optional
        Magnitude of random perturbation (FV3 suggests 0.1).
    rng_key : jax.Array, optional
        JAX PRNG key required if ``rn is not None``.

    Returns
    -------
    q : jax.Array, shape (..., n_x, n_y, km, nq)
        Tracer field.
    """
    qt = jnp.where(
        jnp.sin(nx * lon) * jnp.sin(ny * lat) > 0.0,
        0.01,
        0.0,
    )
    # Broadcast to (..., n_x, n_y, km, nq)
    q = jnp.broadcast_to(qt[..., None, None], qt.shape + (km, nq))
    if rn is not None:
        if rng_key is None:
            raise ValueError("rng_key required when rn is not None")
        noise = rn * jax.random.uniform(rng_key, q.shape)
        q = q + noise
    return q


def atod_vort_on(
    uin: jax.Array, vin: jax.Array,
    dxa: jax.Array, dya: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 655: A-grid → D-grid winds (circulation-conserving).

    Analog of FV3 ``atod`` (tools/test_cases.F90:7833-7892) using
    the circulation-conserving formula consistent with iter-652
    ``dtoa_vort_on`` (the inverse mapping).  FV3's source uses
    ``interp_left_edge_1d`` (interpOrder-dependent); the
    circulation-conserving variant is::

        uout[i, j] = (uin[i, j-1]·dya[i, j-1] + uin[i, j]·dya[i, j])
                    / (dya[i, j-1] + dya[i, j])
        vout[i, j] = (vin[i-1, j]·dxa[i-1, j] + vin[i, j]·dxa[i, j])
                    / (dxa[i-1, j] + dxa[i, j])

    D-grid u lives on north/south edges (n_x, n_y+1); D-grid v
    on east/west edges (n_x+1, n_y).  Interior edges only;
    boundary edges (uout[:, 0], uout[:, -1], vout[0, :], vout[-1, :])
    zero-initialized (FV3 fills via halo).

    Pairs with iter-652 ``dtoa_vort_on`` (D→A) for round-trip
    A-grid ↔ D-grid via circulation-conserving averages.

    Parameters
    ----------
    uin, vin : jax.Array, shape (..., n_x, n_y)
        A-grid wind components.
    dxa, dya : jax.Array, shape (..., n_x, n_y)
        A-grid (cell-center) edge lengths.

    Returns
    -------
    uout : jax.Array, shape (..., n_x, n_y+1)
        D-grid u (north/south edges).
    vout : jax.Array, shape (..., n_x+1, n_y)
        D-grid v (east/west edges).
    """
    n_x = uin.shape[-2]
    n_y = uin.shape[-1]
    leading_shape = uin.shape[:-2]

    # uout: average A-grid uin along j (n_y → n_y-1 interior edges)
    interior_u = (
        (uin[..., :, :-1] * dya[..., :, :-1]
         + uin[..., :, 1:] * dya[..., :, 1:])
        / (dya[..., :, :-1] + dya[..., :, 1:])
    )  # shape (..., n_x, n_y-1)
    uout = jnp.zeros(leading_shape + (n_x, n_y + 1))
    uout = uout.at[..., :, 1:n_y].set(interior_u)

    # vout: average A-grid vin along i (n_x → n_x-1 interior edges)
    interior_v = (
        (vin[..., :-1, :] * dxa[..., :-1, :]
         + vin[..., 1:, :] * dxa[..., 1:, :])
        / (dxa[..., :-1, :] + dxa[..., 1:, :])
    )  # shape (..., n_x-1, n_y)
    vout = jnp.zeros(leading_shape + (n_x + 1, n_y))
    vout = vout.at[..., 1:n_x, :].set(interior_v)
    return uout, vout


def atoc_vort_on(
    uin: jax.Array, vin: jax.Array,
    dxa: jax.Array, dya: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 654: A-grid → C-grid winds (circulation-conserving).

    Faithful JAX port of FV3 ``atoc`` (tools/test_cases.F90:7965-
    8112, ``VORT_ON`` branch, no ``ALT_INTERP``).

    Algorithm (interior C-grid edges only):

        uout[i, j] = (uin[i, j]·dxa[i, j] + uin[i-1, j]·dxa[i-1, j])
                    / (dxa[i, j] + dxa[i-1, j])
        vout[i, j] = (vin[i, j]·dya[i, j] + vin[i, j-1]·dya[i, j-1])
                    / (dya[i, j] + dya[i, j-1])

    Interior edges only (FV3 ``i ∈ [isd+1, ied]``, ``j ∈ [jsd+1, jed]``).
    Boundary edges (uout[0, :] / uout[-1, :] / vout[:, 0] / vout[:, -1])
    are zero-initialized; FV3 sets them via halo communication or
    fill_corners afterward.

    Parameters
    ----------
    uin : jax.Array, shape (..., n_x, n_y)
        A-grid u (cell-center).
    vin : jax.Array, shape (..., n_x, n_y)
        A-grid v (cell-center).
    dxa, dya : jax.Array, shape (..., n_x, n_y)
        A-grid (cell-center) edge lengths.

    Returns
    -------
    uout : jax.Array, shape (..., n_x+1, n_y)
        C-grid u (east/west edges).  Boundary edges = 0.
    vout : jax.Array, shape (..., n_x, n_y+1)
        C-grid v (north/south edges).  Boundary edges = 0.
    """
    # Build C-grid uout via vectorized average of adjacent A-grid columns.
    # uout[i, j] for i ∈ [1, n_x-1] uses uin[i-1, j] and uin[i, j].
    interior_u = (
        (uin[..., 1:, :] * dxa[..., 1:, :]
         + uin[..., :-1, :] * dxa[..., :-1, :])
        / (dxa[..., 1:, :] + dxa[..., :-1, :])
    )  # shape (..., n_x-1, n_y)
    n_x = uin.shape[-2]
    n_y = uin.shape[-1]
    # Allocate full uout (..., n_x+1, n_y) with zeros and fill interior
    leading_shape = uin.shape[:-2]
    uout = jnp.zeros(leading_shape + (n_x + 1, n_y))
    uout = uout.at[..., 1:n_x, :].set(interior_u)

    interior_v = (
        (vin[..., :, 1:] * dya[..., :, 1:]
         + vin[..., :, :-1] * dya[..., :, :-1])
        / (dya[..., :, 1:] + dya[..., :, :-1])
    )  # shape (..., n_x, n_y-1)
    vout = jnp.zeros(leading_shape + (n_x, n_y + 1))
    vout = vout.at[..., :, 1:n_y].set(interior_v)
    return uout, vout


def ctoa_vort_on(
    uin: jax.Array, vin: jax.Array,
    dx: jax.Array, dy: jax.Array,
    dxa: jax.Array, dya: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 653: C-grid → A-grid winds (circulation-conserving).

    Faithful JAX port of FV3 ``ctoa`` (tools/test_cases.F90:8114-
    8174, simple/circulation-conserving branch — the
    commented-out FV3 lines 8147-8157).  Note C-grid u is on
    east/west edges (shape (n_x+1, n_y), opposite of D-grid u);
    C-grid v is on north/south edges (shape (n_x, n_y+1)).

    Algorithm:

        uout[i, j] = 0.5·(uin[i, j]·dy[i, j] + uin[i+1, j]·dy[i+1, j])
                       / dya[i, j]
        vout[i, j] = 0.5·(vin[i, j]·dx[i, j] + vin[i, j+1]·dx[i, j+1])
                       / dxa[i, j]

    Mirror of iter-652 ``dtoa_vort_on`` with input axes swapped
    (C-grid puts u on east/west edges; D-grid puts u on north/
    south edges).

    Parameters
    ----------
    uin : jax.Array, shape (..., n_x+1, n_y)
        C-grid u (east/west edges).
    vin : jax.Array, shape (..., n_x, n_y+1)
        C-grid v (north/south edges).
    dx, dy : jax.Array, shape (..., n_x, n_y+1) and (..., n_x+1, n_y)
        Edge lengths.
    dxa, dya : jax.Array, shape (..., n_x, n_y)
        A-grid (cell-center) edge lengths.

    Returns
    -------
    uout, vout : jax.Array, shape (..., n_x, n_y)
        A-grid cell-center wind components (covariant).
    """
    # uout: average uin·dy along i (axis -2 of uin)
    uout = 0.5 * (
        uin[..., :-1, :] * dy[..., :-1, :]
        + uin[..., 1:, :] * dy[..., 1:, :]
    ) / dya
    # vout: average vin·dx along j (axis -1 of vin)
    vout = 0.5 * (
        vin[..., :, :-1] * dx[..., :, :-1]
        + vin[..., :, 1:] * dx[..., :, 1:]
    ) / dxa
    return uout, vout


def dtoa_vort_on(
    uin: jax.Array, vin: jax.Array,
    dx: jax.Array, dy: jax.Array,
    dxa: jax.Array, dya: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 652: D-grid → A-grid winds (circulation-conserving).

    Faithful JAX port of FV3 ``dtoa`` (tools/test_cases.F90:
    7896-7955, ``VORT_ON`` branch).  Circulation- (vorticity-)
    conserving interpolation from D-grid covariant winds to
    A-grid cell-center winds::

        uout[i, j] = 0.5·(uin[i, j]·dx[i, j] + uin[i, j+1]·dx[i, j+1])
                       / dxa[i, j]
        vout[i, j] = 0.5·(vin[i, j]·dy[i, j] + vin[i+1, j]·dy[i+1, j])
                       / dya[i, j]

    Used by FV3 test-case diagnostics + visualizations.  Differs
    from iter-627 ``c2l_ord2_fv3`` (which applies the a-matrix
    rotation); this is the raw covariant→cell-center step.

    Parameters
    ----------
    uin : jax.Array, shape (..., n_x, n_y+1)
        D-grid u (north/south edges, covariant).
    vin : jax.Array, shape (..., n_x+1, n_y)
        D-grid v (east/west edges, covariant).
    dx, dy : jax.Array, shape (..., n_x, n_y+1) and (..., n_x+1, n_y)
        Edge lengths (matching uin, vin shapes).
    dxa, dya : jax.Array, shape (..., n_x, n_y)
        A-grid (cell-center) edge lengths.

    Returns
    -------
    uout, vout : jax.Array, shape (..., n_x, n_y)
        A-grid cell-center wind components (covariant).
    """
    # uout: average uin·dx along j (axis -1 of uin)
    uout = 0.5 * (
        uin[..., :, :-1] * dx[..., :, :-1]
        + uin[..., :, 1:] * dx[..., :, 1:]
    ) / dxa
    # vout: average vin·dy along i (axis -2 of vin)
    vout = 0.5 * (
        vin[..., :-1, :] * dy[..., :-1, :]
        + vin[..., 1:, :] * dy[..., 1:, :]
    ) / dya
    return uout, vout


def get_pt_on_great_circle(
    lon1: jax.Array, lat1: jax.Array,
    dist: jax.Array, heading: jax.Array,
    radius: float = constants.R_earth,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 651: point on great circle at given distance + heading.

    Faithful JAX port of FV3 ``get_pt_on_great_circle``
    (tools/test_cases.F90:4805-4826).  Given a start point
    (lon1, lat1), great-circle distance ``dist``, and initial
    heading (radians clockwise from north), returns the target
    point (lon3, lat3) on the same great circle.

    Algorithm:
        pha = dist / radius                            # angular dist
        lat3 = asin(cos(heading)·cos(lat1)·sin(pha)
                    + sin(lat1)·cos(pha))
        dp   = atan2(sin(heading)·sin(pha)·cos(lat1),
                     cos(pha) - sin(lat1)·sin(lat3))
        lon3 = ((lon1 - π) - dp + π) mod 2π            # FV3 0-2π
                                                       # wrap

    Used in FV3 for tropical-cyclone test cases (placing vortex
    along a path) and spherical-trajectory computations.

    Parameters
    ----------
    lon1, lat1 : jax.Array
        Start point (radians).  Broadcasting on leading axes
        supported.
    dist : jax.Array
        Great-circle distance from start (meters; same units as
        ``radius``).
    heading : jax.Array
        Initial heading at start (radians; 0 = north, π/2 = east).
    radius : float, default constants.R_earth

    Returns
    -------
    lon3, lat3 : jax.Array
        Target point on great circle (radians).  lon3 wrapped to
        [0, 2π).
    """
    pha = dist / radius
    sin_pha = jnp.sin(pha)
    cos_pha = jnp.cos(pha)
    sin_lat1 = jnp.sin(lat1)
    cos_lat1 = jnp.cos(lat1)
    sin_heading = jnp.sin(heading)
    cos_heading = jnp.cos(heading)
    lat3 = jnp.arcsin(jnp.clip(
        cos_heading * cos_lat1 * sin_pha + sin_lat1 * cos_pha,
        -1.0, 1.0,
    ))
    dp = jnp.arctan2(
        sin_heading * sin_pha * cos_lat1,
        cos_pha - sin_lat1 * jnp.sin(lat3),
    )
    two_pi = 2.0 * jnp.pi
    lon3 = jnp.mod((lon1 - jnp.pi) - dp + jnp.pi, two_pi)
    return lon3, lat3


def intersect_great_circles(
    a1: jax.Array, a2: jax.Array,
    b1: jax.Array, b2: jax.Array,
    radius: float = 1.0,
    eps: float = 1e-30,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 616: intersection of two great circles (Cartesian inputs).

    Faithful port of FV3 ``intersect`` (fv_grid_utils.F90:2096-2194).
    Two great circles are defined by:
        - Circle A: arc through ``a1`` and ``a2``
        - Circle B: arc through ``b1`` and ``b2``

    Returns:
        - ``x_inter``: the intersection point on sphere closest to
          the centroid of (a1, a2, b1, b2) (FV3's ``get_nearest``
          branch); scaled to ``radius``.
        - ``local_a``: ``True`` if ``x_inter`` lies between ``a1``
          and ``a2`` on circle A (chord distance check, F90:2186).
        - ``local_b``: same for circle B.

    Each ``ai``, ``bi`` shape ``(..., 3)``; broadcasts on leading
    axes.  ``local_a`` / ``local_b`` are boolean arrays.

    Matches FV3's exact determinant formulation (lines 2128-2147)
    for bit-equivalence; handles the FV3 degenerate branches
    (``b1_xyz=0`` → x_inter=b1; ``b2_xyz=0`` → x_inter=b2) via
    ``jnp.where``.
    """
    a1x = a1[..., 0]; a1y = a1[..., 1]; a1z = a1[..., 2]
    a2x = a2[..., 0]; a2y = a2[..., 1]; a2z = a2[..., 2]
    b1x = b1[..., 0]; b1y = b1[..., 1]; b1z = b1[..., 2]
    b2x = b2[..., 0]; b2y = b2[..., 1]; b2z = b2[..., 2]

    a2_xy = a2x * a1y - a2y * a1x
    b1_xy = b1x * a1y - b1y * a1x
    b2_xy = b2x * a1y - b2y * a1x

    a2_xz = a2x * a1z - a2z * a1x
    b1_xz = b1x * a1z - b1z * a1x
    b2_xz = b2x * a1z - b2z * a1x

    b1_xyz = b1_xy * a2_xz - b1_xz * a2_xy
    b2_xyz = b2_xy * a2_xz - b2_xz * a2_xy

    # General branch: x_raw = b2 - b1 * (b2_xyz / b1_xyz), normalized to radius
    safe_b1_xyz = jnp.where(jnp.abs(b1_xyz) > eps, b1_xyz, 1.0)
    ratio = b2_xyz / safe_b1_xyz
    x_general = b2 - b1 * ratio[..., None]
    length = jnp.sqrt(jnp.sum(x_general * x_general, axis=-1, keepdims=True))
    safe_len = jnp.where(length > eps, length, 1.0)
    x_general = radius * x_general / safe_len

    # FV3 degenerate branches (F90:2139-2142)
    b1_zero = jnp.abs(b1_xyz) <= eps
    b2_zero = (~b1_zero) & (jnp.abs(b2_xyz) <= eps)
    x_inter = jnp.where(
        b1_zero[..., None], b1,
        jnp.where(b2_zero[..., None], b2, x_general),
    )

    # get_nearest: pick ±x_inter closer to centroid (F90:2157-2169)
    center = 0.25 * (a1 + a2 + b1 + b2)
    dx_pos = x_inter - center
    dx_neg = -x_inter - center
    d_pos = jnp.sum(dx_pos * dx_pos, axis=-1)
    d_neg = jnp.sum(dx_neg * dx_neg, axis=-1)
    x_inter = jnp.where(
        (d_neg < d_pos)[..., None], -x_inter, x_inter,
    )

    # check_local for A and B (F90:2171-2192): chord-distance test
    def _check_local(x1: jax.Array, x2: jax.Array) -> jax.Array:
        dx = x1 - x2
        dist = jnp.sum(dx * dx, axis=-1)
        dx1 = x1 - x_inter
        dx2 = x2 - x_inter
        d1 = jnp.sum(dx1 * dx1, axis=-1)
        d2 = jnp.sum(dx2 * dx2, axis=-1)
        return (d1 <= dist) & (d2 <= dist)

    local_a = _check_local(a1, a2)
    local_b = _check_local(b1, b2)
    return x_inter, local_a, local_b


def unit_vect_latlon(
    lon: jax.Array, lat: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 615: east/north unit tangent vectors at (lon, lat).

    Faithful port of FV3 ``unit_vect_latlon`` (fv_grid_utils.F90:
    2286-2309).  Returns the two Cartesian unit vectors of the
    local geographic frame at the input (lon, lat) point::

        elon = (-sin λ,    cos λ,    0    )
        elat = (-sin φ cos λ, -sin φ sin λ, cos φ)

    where ``λ`` = lon, ``φ`` = lat.  Used by FV3 ``c2l_ord4`` to
    rotate D-grid winds to geographic (east, north) frame.

    Returns two arrays of shape ``(..., 3)``; broadcasts on
    leading axes.
    """
    sin_lon = jnp.sin(lon)
    cos_lon = jnp.cos(lon)
    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)
    zero = jnp.zeros_like(sin_lon)
    elon = jnp.stack([-sin_lon, cos_lon, zero], axis=-1)
    elat = jnp.stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=-1)
    return elon, elat


def get_unit_vect3(
    p1: jax.Array, p2: jax.Array,
) -> jax.Array:
    """FV3_3D iter 615: unit tangent vector at GC midpoint (Cartesian variant).

    Faithful port of FV3 ``get_unit_vect3`` (fv_grid_utils.F90:
    1865-1876).  Cartesian-input version of iter-611
    ``get_unit_vect2`` — takes ``p1``, ``p2`` already in Cartesian
    (last axis = 3-vector) and returns the unit tangent vector at
    the great-circle midpoint pointing from p1 → p2.

    Algorithm (FV3 exact):
        pc = mid_pt3_cart(p1, p2)
        p3 = p2 × p1                   (great-circle pole)
        uc = pc × p3                   (tangent at pc)
        uc / |uc|
    """
    pc = mid_pt3_cart(p1, p2)
    p3 = vect_cross(p2, p1)
    uc = vect_cross(pc, p3)
    return normalize_vect(uc)


def great_circle_distance_cart(
    v1: jax.Array, v2: jax.Array,
    radius: float = constants.R_earth,
) -> jax.Array:
    """FV3_3D iter 614: great-circle distance from Cartesian inputs.

    Faithful port of FV3 ``great_circle_dist_cart`` (fv_grid_utils.F90:
    2065-2092)::

        cos(d/R) = (v1·v2) / (|v1|·|v2|)
        d = R · acos(clip(cos, -1, 1))

    Each ``v1``, ``v2`` shape ``(..., 3)`` on (or near) the unit
    sphere; result broadcasts on leading axes.  Result has same
    units as ``radius`` (default: legoESM R_earth in metres).

    Differentiable; safe near antipodal points via clip.
    """
    norm = jnp.sum(v1 * v1, axis=-1) * jnp.sum(v2 * v2, axis=-1)
    safe_norm = jnp.where(norm > 0.0, norm, 1.0)
    dot = jnp.sum(v1 * v2, axis=-1) / jnp.sqrt(safe_norm)
    dot = jnp.clip(dot, -1.0, 1.0)
    return radius * jnp.arccos(dot)


def get_area(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon3: jax.Array, lat3: jax.Array,
    lon4: jax.Array, lat4: jax.Array,
    radius: float = constants.R_earth,
) -> jax.Array:
    """FV3_3D iter 614: spherical-excess cell area for quadrilateral cell.

    Faithful port of FV3 ``get_area`` (fv_grid_utils.F90:2749-2790).
    Computes the four spherical angles at the cell corners and uses
    the spherical-excess formula::

        Area = (α1 + α2 + α3 + α4 - 2π) · R²

    The corner-order convention matches FV3's signature exactly
    (note the FV3 call uses ``p1, p4, p2, p3``):

        4 ----- 3
        |       |
        |       |
        1 ----- 2

    and the four corner angles are taken at vertices 1, 2, 3, 4 in
    counterclockwise order.

    Result has units of ``radius²`` (default: legoESM R_earth in m²).
    """
    # Build Cartesian corner vectors
    e1 = jnp.stack(list(latlon2xyz(lon1, lat1)), axis=-1)
    e2 = jnp.stack(list(latlon2xyz(lon2, lat2)), axis=-1)
    e3 = jnp.stack(list(latlon2xyz(lon3, lat3)), axis=-1)
    e4 = jnp.stack(list(latlon2xyz(lon4, lat4)), axis=-1)
    # FV3 fv_grid_utils.F90:2757-2782 corner-angle convention:
    #   ang1 = ∠(at p1; p2 → p4)
    #   ang2 = ∠(at p2; p3 → p1)
    #   ang3 = ∠(at p3; p4 → p2)
    #   ang4 = ∠(at p4; p3 → p1)
    ang1 = spherical_angle(e1, e2, e4)
    ang2 = spherical_angle(e2, e3, e1)
    ang3 = spherical_angle(e3, e4, e2)
    ang4 = spherical_angle(e4, e3, e1)
    excess = ang1 + ang2 + ang3 + ang4 - 2.0 * jnp.pi
    return excess * (radius * radius)


def expand_cell(
    lon1: jax.Array, lat1: jax.Array,
    lon2: jax.Array, lat2: jax.Array,
    lon3: jax.Array, lat3: jax.Array,
    lon4: jax.Array, lat4: jax.Array,
    fac: float,
) -> tuple[
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
]:
    """FV3_3D iter 613: expand 4-corner cell about its center by factor ``fac``.

    Faithful port of FV3 ``expand_cell`` (fv_grid_utils.F90:
    2631-2697).  Returns 4 new (lon, lat) corners with the cell
    extrapolated (fac > 1) or shrunk (fac < 1) about the
    spherical center.

        fac = 1: returns the input corners unchanged
        fac = 0: all 4 corners collapse to the cell center
        fac > 1: expansion outward (cell grows)

    All output corners are forced to lie on the unit sphere via
    re-normalization, matching FV3 lines 2675-2686.
    """
    x1, y1, z1 = latlon2xyz(lon1, lat1)
    x2, y2, z2 = latlon2xyz(lon2, lat2)
    x3, y3, z3 = latlon2xyz(lon3, lat3)
    x4, y4, z4 = latlon2xyz(lon4, lat4)
    p1 = jnp.stack([x1, y1, z1], axis=-1)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    p3 = jnp.stack([x3, y3, z3], axis=-1)
    p4 = jnp.stack([x4, y4, z4], axis=-1)
    ec = cell_center3(p1, p2, p3, p4)
    qq1 = normalize_vect(ec + fac * (p1 - ec))
    qq2 = normalize_vect(ec + fac * (p2 - ec))
    qq3 = normalize_vect(ec + fac * (p3 - ec))
    qq4 = normalize_vect(ec + fac * (p4 - ec))
    return (
        xyz2latlon(qq1[..., 0], qq1[..., 1], qq1[..., 2]),
        xyz2latlon(qq2[..., 0], qq2[..., 1], qq2[..., 2]),
        xyz2latlon(qq3[..., 0], qq3[..., 1], qq3[..., 2]),
        xyz2latlon(qq4[..., 0], qq4[..., 1], qq4[..., 2]),
    )


def rotate_winds_geo_to_grid(
    u_east: jax.Array, v_north: jax.Array, angle: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Rotate winds from geographic (east, north) to grid-aligned (x, y)."""
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_grid = cos_a * u_east + sin_a * v_north
    v_grid = -sin_a * u_east + cos_a * v_north
    return u_grid, v_grid


def apply_small_earth_scaling(
    grid: CubedSphereGrid,
    factor: float,
) -> CubedSphereGrid:
    """Create a small-Earth grid by scaling radius and rotation rate.

    On a small Earth of radius R/X:
    - dx, dy scale as 1/X
    - Cell areas scale as 1/X^2
    - Omega scales as X (to keep Rossby number constant)
    - Coriolis f scales as X

    Parameters
    ----------
    grid : CubedSphereGrid
        Original grid at Earth radius.
    factor : float
        Reduction factor X. Earth radius becomes R_earth/X.

    Returns
    -------
    CubedSphereGrid
        New grid with scaled metrics.
    """
    if factor == 1.0:
        return grid

    from legoesm import constants
    # Forward the static provenance: rebuilding without it silently turned an
    # ED grid back into equiangular (codex P1, phase-1 FV3-native work).
    # NOTE: dtype/duogrid are NOT forwarded — pre-existing behavior kept.
    return create_cubed_sphere(
        grid.n,
        radius=constants.R_earth / factor,
        omega=constants.Omega * factor,
        gnomonic=grid.gnomonic_form,
    )


def rotate_winds_grid_to_geo(
    u_grid: jax.Array, v_grid: jax.Array, angle: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Rotate winds from grid-aligned (x, y) to geographic (east, north)."""
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_east = cos_a * u_grid - sin_a * v_grid
    v_north = sin_a * u_grid + cos_a * v_grid
    return u_east, v_north


def create_cubed_sphere_panel(
    n: int,
    face_id: int = 0,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
    return_cdgrid: bool = False,
    gnomonic: str = "equiangular",
) -> "CubedSphereGrid | tuple[CubedSphereGrid, ...]":
    """Create a single-face cubed-sphere panel for regional experiments.

    Extracts one face from a full cubed-sphere grid and returns a
    ``CubedSphereGrid`` with leading dimension 1 instead of 6.

    All existing operators (gradient, divergence, Laplacian) work
    automatically because :func:`~legoesm.grids.halo.pad_halo` detects
    ``data.shape[0] == 1`` and applies Neumann (zero-gradient) wall
    boundary conditions instead of inter-face halo exchange.

    Parameters
    ----------
    n : int
        Grid resolution (cells per face edge).
    face_id : int
        Which cube face to extract (0-5, default 0 = equatorial).
    radius : float
        Sphere radius [m].
    omega : float
        Rotation rate [rad/s].
    gnomonic : str, default "equiangular"
        Node distribution of the underlying full grid — forwarded to
        :func:`create_cubed_sphere` and stamped as static provenance on the
        returned panel (``"equiangular"`` legacy or ``"ed"`` FV3-native).

    Returns
    -------
    CubedSphereGrid or (CubedSphereGrid, CubedSphereCDGrid)
        Grid with all arrays shaped ``(1, n, n)`` or ``(1, n+2, n+2)``
        for the selected face.  Use with a wall land mask (0 on
        boundary, 1 in interior) for closed-basin experiments.

        When ``return_cdgrid=True``, also returns the single-face
        C-D grid needed by the ocean baroclinic solver.
    """
    full = create_cubed_sphere(
        n, radius=radius, omega=omega, dtype=dtype, gnomonic=gnomonic)

    # Extract single face, keeping leading dimension
    f = face_id
    _s = lambda arr: arr[f:f+1]  # (6,...) → (1,...)
    _p = lambda arr: arr[f:f+1]  # same for padded arrays

    # Padded angle/metric arrays: extract face then re-pad with edge BC
    # (the full grid's padded arrays have inter-face halo data that doesn't
    # apply to a single-face panel).
    def _repad(arr_full_face, halo=1):
        """Re-pad a single face with Neumann BC."""
        interior = arr_full_face  # (1, n, n) or (1, n+2h, n+2h)
        if interior.shape[1] > n:
            interior = interior[:, halo:-halo, halo:-halo]
        return jnp.pad(interior, ((0, 0), (halo, halo), (halo, halo)),
                        mode="edge")

    angle_p = _repad(_s(full.angle_padded), halo=1)

    panel = CubedSphereGrid(
        n=n,
        radius=float(radius),
        lon=_s(full.lon),
        lat=_s(full.lat),
        area=_s(full.area),
        dx=_s(full.dx),
        dy=_s(full.dy),
        f=_s(full.f),
        cos_lat=_s(full.cos_lat),
        sin_lat=_s(full.sin_lat),
        angle=_s(full.angle),
        x_cart=_s(full.x_cart),
        y_cart=_s(full.y_cart),
        z_cart=_s(full.z_cart),
        angle_padded=angle_p,
        cos_angle=_s(full.cos_angle),
        sin_angle=_s(full.sin_angle),
        cos_angle_padded=jnp.cos(angle_p),
        sin_angle_padded=jnp.sin(angle_p),
        hx_ext=_repad(_s(full.hx_ext), halo=1),
        hy_ext=_repad(_s(full.hy_ext), halo=1),
        halo_interp_offsets=None,  # not needed — wall BC
        cos_angle_padded_h2=jnp.cos(_repad(_s(full.angle), halo=2)),
        sin_angle_padded_h2=jnp.sin(_repad(_s(full.angle), halo=2)),
        hx_ext_h2=_repad(_s(full.hx_ext_h2), halo=2),
        hy_ext_h2=_repad(_s(full.hy_ext_h2), halo=2),
        halo_interp_offsets_h2=None,  # not needed — wall BC
        halo_interp_offsets_h3=None,  # not needed — wall BC
        cos_angle_padded_h3=jnp.cos(_repad(_s(full.angle), halo=3)),
        sin_angle_padded_h3=jnp.sin(_repad(_s(full.angle), halo=3)),
        hx_ext_h3=_repad(_s(full.hx_ext_h3), halo=3),
        hy_ext_h3=_repad(_s(full.hy_ext_h3), halo=3),
        duogrid=None,  # regional panel: no cross-face duogrid data
        gnomonic_form=full.gnomonic_form,
        fv3_grid_type=full.fv3_grid_type,
    )

    if not return_cdgrid:
        return panel

    if gnomonic == "ed":
        # Phase-2 codex finding: slicing the GLOBAL ED C/D metrics into a
        # walled single-face panel would carry the FV3 ×2 cube-seam edge
        # conventions onto what are physically WALLS — the FV3 bounded
        # (regional) metric construction differs and is not implemented.
        # The exact-metrics claim of the FV3-native path must not silently
        # degrade here, so fail closed.  (The A-grid ED panel above is fine:
        # its metrics are cell-local.)
        raise NotImplementedError(
            "create_cubed_sphere_panel(gnomonic='ed', return_cdgrid=True): "
            "the FV3-native C/D metrics are the GLOBAL-cube construction; "
            "slicing them into a walled panel would mislabel cube-seam "
            "edge conventions as wall metrics. A bounded-domain FV3 metric "
            "builder is not implemented — use the equiangular panel or "
            "request the A-grid ED panel (return_cdgrid=False).")

    # Also build single-face C-D grid from the full cdgrid.
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    full_cdgrid = create_cubed_sphere_cdgrid(full, omega=omega)

    # Extract face f from every array leaf in the cdgrid, replacing the
    # base grid reference with the panel.
    def _extract_face(leaf):
        if hasattr(leaf, 'shape') and hasattr(leaf, 'ndim'):
            if leaf.ndim >= 3 and leaf.shape[0] == 6:
                return leaf[f:f+1]
        return leaf

    cdgrid_panel = jax.tree.map(_extract_face, full_cdgrid)
    # Replace the base grid with the panel (base is index 0 of the NamedTuple)
    cdgrid_panel = cdgrid_panel._replace(base=panel)

    # Restore the FV3 bounded_domain rsin_u/rsin_v convention for the panel.
    # The full-grid build applies the 1/sin panel-edge override only when
    # bounded_domain is False (fv_arrays.F90:1512).  A single-face panel is
    # a bounded_domain case (regional) and must use 1/sin² everywhere —
    # matching fv_grid_utils.F90:509.  Undo the override that was inherited
    # from the 6-face build.
    import jax.numpy as _jnp
    _EPS = float(_jnp.finfo(_jnp.float32).eps)
    # cosa_u_panel was already extracted; recompute rsin_u = 1/sin² there.
    sina_u_sq_panel = _jnp.maximum(
        1.0 - cdgrid_panel.cosa_u**2, _EPS)
    rsin_u_panel = 1.0 / _jnp.maximum(sina_u_sq_panel, _EPS)
    sina_v_sq_panel = _jnp.maximum(
        1.0 - cdgrid_panel.cosa_v**2, _EPS)
    rsin_v_panel = 1.0 / _jnp.maximum(sina_v_sq_panel, _EPS)
    cdgrid_panel = cdgrid_panel._replace(
        rsin_u=rsin_u_panel.astype(cdgrid_panel.rsin_u.dtype),
        rsin_v=rsin_v_panel.astype(cdgrid_panel.rsin_v.dtype),
    )
    return panel, cdgrid_panel
