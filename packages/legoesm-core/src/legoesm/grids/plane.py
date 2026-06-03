"""Doubly-periodic Cartesian plane grid for cloud-resolving simulations.

Staged-not-integrated: this module ships the grid + operator data
structure used by the future plane non-hydrostatic dycore (PR2 of the
CRM rollout). It has no producing dispatch path yet — callers
construct it directly via :func:`create_plane_grid` for unit tests
and operator validation. The plane NH dycore that consumes it lands
in a follow-up PR.

Grid conventions
----------------
- Doubly periodic in both horizontal axes. No walls.
- Uniform spacing ``dx``, ``dy`` (metres). Cells centred at
  ``xc[i] = (i + 0.5) * dx`` for i in ``0 .. nx-1`` (likewise ``yc``).
- Arakawa-C staggering: scalars at cell centres, ``u`` at x-faces
  (between cells i-1 and i), ``v`` at y-faces. Vertical velocity ``w``
  on interfaces. See ``plane_operators`` for index formulas and
  energy-consistent inner-product weights.
- All ``u``, ``v``, scalar arrays have horizontal shape ``(ny, nx)``
  with no duplicated periodic endpoint. ``w`` has shape ``(nlev+1, ny,
  nx)``. Periodic neighbours come from ``jnp.roll`` or ``jnp.pad(...,
  mode='wrap')``.
- Coriolis is selectable: ``'none'`` (f = 0), ``'f_plane'`` (f = f0
  everywhere), or ``'beta_plane'`` (f = f0 + beta * (y - Ly/2)).
- The plane is a tangent-plane approximation of Earth at the central
  latitude ``lat0``. ``grid_lat`` and ``grid_lon`` are stored as
  constant arrays equal to ``lat0``, ``lon0`` so that downstream
  physics (radiation insolation) sees a single latitude for the whole
  domain — appropriate for RCEMIP-style runs. ``grid_radius`` returns
  ``constants.R_earth`` only as metadata for global-grid compatibility;
  the plane operators themselves use no radius.

Surface mask
------------
``surface_mask`` is all-ones (no land). The plane is intended for
RCE-style runs where the surface type (ocean / fixed-SST / prescribed
flux) is decided at the experiment-config level, not by the grid.

References
----------
Klemp & Wilhelmson (1978) "The simulation of three-dimensional
convective storm dynamics", J. Atmos. Sci.
Wing et al. (2018) "Radiative-Convective Equilibrium Model
Intercomparison Project", Geosci. Model Dev.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants


CoriolisMode = str  # Literal['none', 'f_plane', 'beta_plane']


class PlaneGrid(NamedTuple):
    """Doubly-periodic Cartesian plane horizontal grid.

    All arrays are JAX arrays. As a NamedTuple this is a valid JAX
    pytree: integer / string / float fields are static (treedef);
    array fields are dynamic leaves.

    Attributes
    ----------
    nx, ny : int
        Number of cells along x and y (zonal, meridional in tangent
        plane sense).
    nlev : int
        Number of vertical levels (cell-centre count).
    dx, dy : float
        Cell spacing in metres.
    Lx, Ly : float
        Domain extent in metres. Derived: ``Lx = nx * dx``.
    lat0, lon0 : float
        Central latitude / longitude in degrees for radiation. The
        whole plane is treated as if at ``(lat0, lon0)``.
    f0 : float
        Constant Coriolis parameter [s^-1] used for f-plane and as the
        reference value for beta-plane. Zero in non-rotating ('none') mode.
    beta : float
        Beta-plane meridional gradient of f [s^-1 m^-1]. Zero outside
        beta-plane mode. ``f_y`` is fully materialised at construction
        time.
    coriolis_mode_code : int
        Integer encoding of the Coriolis mode (0=none, 1=f_plane,
        2=beta_plane). Stored as an int because JAX cannot trace
        Python strings inside a NamedTuple. Use the
        :attr:`coriolis_mode` property for the decoded string.
    xc, yc : jax.Array
        Cell-centre coordinates, shape ``(nx,)``, ``(ny,)``, metres.
    xu : jax.Array
        x-face coordinates, shape ``(nx,)``. ``xu[i]`` is the face
        between cell ``i-1`` and ``i`` (mod nx), positioned at
        ``i * dx``.
    yv : jax.Array
        y-face coordinates, shape ``(ny,)``.
    f_y : jax.Array
        Coriolis parameter as 2D field at cell centres, shape
        ``(ny, nx)``.
    area_T : jax.Array
        Per-cell area at cell centres, shape ``(ny, nx)``. Uniform
        ``dx * dy`` on the plane but stored 2D for protocol
        compatibility.
    total_area : jax.Array
        Scalar sum equal to ``Lx * Ly``.
    surface_mask : jax.Array
        Shape ``(ny, nx)``. Convention: ``1.0`` = active surface cell;
        ``0.0`` = blocked. Uniformly ``1.0`` on the plane (no land).
        The complementary ``land_mask`` property returns
        ``1 - surface_mask`` for ocean-convention consumers.
    """

    nx: int
    ny: int
    nlev: int
    dx: float
    dy: float
    Lx: float
    Ly: float
    lat0: float
    lon0: float
    f0: float
    beta: float
    coriolis_mode_code: int
    xc: jax.Array
    yc: jax.Array
    xu: jax.Array
    yv: jax.Array
    f_y: jax.Array
    area_T: jax.Array
    total_area: jax.Array
    surface_mask: jax.Array

    # ------------------------------------------------------------------
    # GridProtocol
    # ------------------------------------------------------------------

    @property
    def grid_lat(self) -> jax.Array:
        """Per-cell latitude [rad] — constant ``lat0`` over the plane."""
        return jnp.deg2rad(self.lat0) * jnp.ones(
            (self.ny, self.nx), dtype=self.area_T.dtype
        )

    @property
    def grid_lon(self) -> jax.Array:
        """Per-cell longitude [rad] — constant ``lon0`` over the plane."""
        return jnp.deg2rad(self.lon0) * jnp.ones(
            (self.ny, self.nx), dtype=self.area_T.dtype
        )

    @property
    def grid_area(self) -> jax.Array:
        return self.area_T

    @property
    def grid_total_area(self) -> jax.Array:
        return self.total_area

    @property
    def grid_coriolis(self) -> jax.Array:
        return self.f_y

    @property
    def grid_radius(self) -> float:
        """Tangent-plane radius — metadata only.

        Plane operators do not consume any radius; the plane grid is a
        local Cartesian patch with metric ``dx, dy``. ``grid_radius``
        exists exclusively to satisfy ``GridProtocol`` for generic
        plotting / diagnostic code that iterates over heterogeneous
        grids. Downstream callers must not apply spherical assumptions
        (Coriolis from latitude, metric divisions by cos(lat),
        spherical-area corrections) to a plane grid.
        """
        return float(constants.R_earth)

    @property
    def n(self) -> int:
        """Resolution proxy for code expecting ``grid.n``."""
        return self.ny

    @property
    def grid_n_columns(self) -> int:
        return self.nx * self.ny

    @property
    def grid_shape_2d(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    # ------------------------------------------------------------------
    # Coriolis-mode decoding (string form kept outside the pytree)
    # ------------------------------------------------------------------

    @property
    def coriolis_mode(self) -> str:
        """Decoded Coriolis-mode string from ``coriolis_mode_code``.

        **Host-only.** Indexes a Python tuple with the integer code, so
        must NOT be called from inside a JIT-traced function — the
        tracer cannot index a tuple. Use this property for diagnostics,
        config round-tripping, logging, and Python-side branching.
        Inside JIT-traced code branch on ``self.f0`` and ``self.beta``
        instead, or pre-compute mode-dependent arrays at construction
        (``f_y`` is already fully materialised so no per-step branch is
        needed).
        """
        return _CORIOLIS_MODE_NAMES[self.coriolis_mode_code]

    # ------------------------------------------------------------------
    # Compatibility aliases for downstream grid consumers
    # ------------------------------------------------------------------

    @property
    def area(self) -> jax.Array:
        """Alias for ``area_T`` matching the ``LatLonGrid.area`` name."""
        return self.area_T

    @property
    def dx_face(self) -> jax.Array:
        """Per-face zonal spacing [m], shape ``(ny, nx)``.

        Uniform ``dx`` on the plane, broadcast to a 2D array so callers
        that expect a per-cell metric (matching curvilinear-grid usage)
        work without special-casing the plane.
        """
        return jnp.full((self.ny, self.nx), float(self.dx),
                        dtype=self.area_T.dtype)

    @property
    def dy_face(self) -> jax.Array:
        """Per-face meridional spacing [m], shape ``(ny, nx)``."""
        return jnp.full((self.ny, self.nx), float(self.dy),
                        dtype=self.area_T.dtype)

    @property
    def land_mask(self) -> jax.Array:
        """Land-blocking mask following the ocean convention.

        Convention used throughout this module:

        - ``surface_mask[j, i] == 1`` means the cell is an **active
          surface cell** (the plane simulation can compute fluxes
          through it). On the plane it is uniformly ``1``.
        - ``land_mask[j, i] == 1`` means the cell is **blocked land**
          (a generic ocean/coupler convention where land cells are
          masked out). Computed as ``1 - surface_mask``, so it is
          uniformly ``0`` on the plane.

        These two conventions are deliberately complementary, not
        synonymous. The property exists so that downstream code written
        for ocean grids (which expect ``land_mask == 1`` to block) runs
        on the plane without changes.
        """
        return 1.0 - self.surface_mask

    # ------------------------------------------------------------------
    # Column reshaping helpers
    # ------------------------------------------------------------------

    def to_columns(self, field: jax.Array) -> jax.Array:
        """Reshape a 2D-horizontal field to columns.

        Accepts shape ``(ny, nx, *extra)`` and returns
        ``(ny * nx, *extra)`` — matching ``LatLonGrid.to_columns``.

        Note
        ----
        Plane dycore operators in ``plane_operators`` consume fields
        shaped ``(*extra, ny, nx)`` (e.g. ``(nlev, ny, nx)``). To pass
        such a field through ``to_columns`` first transpose the
        horizontal axes to the front:

            field_zyx = ...                # shape (nlev, ny, nx)
            field_yxz = jnp.moveaxis(field_zyx, 0, -1)  # (ny, nx, nlev)
            cols = grid.to_columns(field_yxz)            # (ny*nx, nlev)
        """
        if field.ndim < 2:
            raise ValueError(
                f"to_columns expects ndim >= 2 with leading (ny, nx); "
                f"got shape {field.shape}"
            )
        if field.shape[0] != self.ny or field.shape[1] != self.nx:
            raise ValueError(
                f"to_columns expects shape (ny={self.ny}, nx={self.nx}, "
                f"...); got {field.shape}"
            )
        extra = field.shape[2:]
        return field.reshape(self.ny * self.nx, *extra)

    def from_columns(self, cols: jax.Array) -> jax.Array:
        """Inverse of ``to_columns``: ``(ny*nx, *extra) -> (ny, nx, *extra)``."""
        if cols.ndim < 1:
            raise ValueError(
                f"from_columns expects ndim >= 1; got shape {cols.shape}"
            )
        if cols.shape[0] != self.ny * self.nx:
            raise ValueError(
                f"from_columns expects leading axis ny*nx="
                f"{self.ny * self.nx}; got {cols.shape}"
            )
        extra = cols.shape[1:]
        return cols.reshape(self.ny, self.nx, *extra)


_CORIOLIS_MODE_NAMES = ("none", "f_plane", "beta_plane")
_CORIOLIS_MODE_CODES = {name: i for i, name in enumerate(_CORIOLIS_MODE_NAMES)}
_VALID_CORIOLIS_MODES = _CORIOLIS_MODE_NAMES


def create_plane_grid(
    nx: int,
    ny: int,
    nlev: int,
    dx: float,
    dy: float,
    lat0: float = 0.0,
    lon0: float = 0.0,
    coriolis_mode: CoriolisMode = "none",
    f0: float = 0.0,
    beta: float = 0.0,
    dtype=None,
) -> PlaneGrid:
    """Construct a doubly-periodic plane grid.

    Parameters
    ----------
    nx, ny : int
        Horizontal cell counts. Must be >= 4 (centred fourth-order
        stencils later).
    nlev : int
        Vertical level count. Must be >= 2.
    dx, dy : float
        Horizontal cell spacing in metres. Must be positive.
    lat0, lon0 : float, optional
        Central latitude / longitude in degrees for radiation. Default
        0, 0 (equator, Greenwich) — typical RCEMIP setup.
    coriolis_mode : str, optional
        One of ``'none'``, ``'f_plane'``, ``'beta_plane'``. Default
        ``'none'`` (no rotation, classic RCE).
    f0 : float, optional
        Constant Coriolis parameter [s^-1] for f-plane and reference
        value for beta-plane. Default 0.
    beta : float, optional
        Meridional gradient of f [s^-1 m^-1]. Required nonzero when
        ``coriolis_mode='beta_plane'``.
    dtype : optional
        Storage dtype for JAX arrays. Defaults to the current precision
        policy's storage type, falling back to float32.

    Returns
    -------
    PlaneGrid

    Raises
    ------
    ValueError
        For ``nx < 4``, ``ny < 4``, ``nlev < 2``, non-positive ``dx``
        or ``dy``, ``lat0`` outside ``[-90, 90]``, unrecognised
        ``coriolis_mode``, or ``coriolis_mode='beta_plane'`` with
        ``beta == 0``.
    """
    if nx < 4:
        raise ValueError(f"nx must be >= 4, got {nx}")
    if ny < 4:
        raise ValueError(f"ny must be >= 4, got {ny}")
    if nlev < 2:
        raise ValueError(f"nlev must be >= 2, got {nlev}")
    if dx <= 0:
        raise ValueError(f"dx must be positive, got {dx}")
    if dy <= 0:
        raise ValueError(f"dy must be positive, got {dy}")
    if not -90.0 <= lat0 <= 90.0:
        raise ValueError(f"lat0 must lie in [-90, 90], got {lat0}")
    if coriolis_mode not in _VALID_CORIOLIS_MODES:
        raise ValueError(
            f"coriolis_mode must be one of {_VALID_CORIOLIS_MODES}, "
            f"got {coriolis_mode!r}"
        )
    if coriolis_mode == "beta_plane" and beta == 0.0:
        raise ValueError(
            "coriolis_mode='beta_plane' requires nonzero beta"
        )

    coriolis_mode_code = _CORIOLIS_MODE_CODES[coriolis_mode]

    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().storage
        except Exception:
            dtype = jnp.float32

    Lx = float(nx * dx)
    Ly = float(ny * dy)

    xc = (jnp.arange(nx, dtype=dtype) + 0.5) * dx
    yc = (jnp.arange(ny, dtype=dtype) + 0.5) * dy
    xu = jnp.arange(nx, dtype=dtype) * dx
    yv = jnp.arange(ny, dtype=dtype) * dy

    if coriolis_mode == "none":
        f_y_1d = jnp.zeros((ny,), dtype=dtype)
    elif coriolis_mode == "f_plane":
        f_y_1d = jnp.full((ny,), float(f0), dtype=dtype)
    else:  # 'beta_plane'
        f_y_1d = jnp.asarray(
            f0 + beta * (yc - 0.5 * Ly), dtype=dtype
        )
    f_y = f_y_1d[:, None] * jnp.ones((1, nx), dtype=dtype)

    area_T = jnp.full((ny, nx), float(dx * dy), dtype=dtype)
    total_area = jnp.asarray(Lx * Ly, dtype=dtype)
    surface_mask = jnp.ones((ny, nx), dtype=dtype)

    return PlaneGrid(
        nx=int(nx),
        ny=int(ny),
        nlev=int(nlev),
        dx=float(dx),
        dy=float(dy),
        Lx=Lx,
        Ly=Ly,
        lat0=float(lat0),
        lon0=float(lon0),
        f0=float(f0),
        beta=float(beta),
        coriolis_mode_code=int(coriolis_mode_code),
        xc=xc,
        yc=yc,
        xu=xu,
        yv=yv,
        f_y=f_y,
        area_T=area_T,
        total_area=total_area,
        surface_mask=surface_mask,
    )
