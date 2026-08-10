"""legoESM adapter: FESOM2-JAX dycore as a selectable ocean grid/dycore.

This module makes FESOM2 (https://github.com/koldunovn/fesom_jax) a *selectable*
ocean grid/dycore inside legoESM — not an additional model.  The legoESM
timeloop and its generic diagnostics (RPE, transports, overturning, …)
work on any state that exposes the ``Field``-based interface described in
``legoesm.core.field`` and any grid that exposes ``.area`` as a 1-D array.

The adaptation happens at TWO seams — :class:`FesomOceanState` and
:class:`FesomOceanGrid` — which keep the ``grid_type ==`` branch count
SMALL and confined.  Most sites that dispatch on ``grid_type`` fall
through to their generic ``else`` because
``FesomOceanGrid.grid_type == "fesom"`` matches no existing branch.
A small number of sites (``lock_exchange``, ``restart``, ``cfl``, and
the matrix's own ``_compute_rpe``) did need a new FESOM branch; the two
seams keep that set minimal and prevent it from growing.  The RPE
diagnostic ``legoesm.ocean.rpe.compute_rpe`` genuinely needs no change
— there is a passing test for that.

Deferred-import policy (review F3)
---------------------------------
Every ``fesom_jax`` import is at **function scope**.  Importing this module
must NOT require ``fesom_jax`` — legoESM ships without it.  Only *using*
the adapter (constructing the model, building a mesh, creating an IC) requires
the package.  A clear :class:`ImportError` with the install command is raised
at the call site via :func:`_require_fesom_jax`.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, NamedTuple, Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field

if TYPE_CHECKING:  # never evaluated at runtime — safe without fesom_jax
    from fesom_jax.mesh import Mesh
    from fesom_jax.state import State


__all__ = [
    "FesomOceanConfig",
    "FesomOceanGrid",
    "FesomOceanState",
    "FesomOceanModel",
    "build_flat_bottom_mesh",
    "create_lock_exchange_state",
]


# =============================================================================
# Helper — deferred-import gate (F3)
# =============================================================================

def _require_fesom_jax() -> None:
    """Raise a helpful ImportError if fesom_jax is not installed."""
    try:
        import fesom_jax  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "fesom_jax is required to use the FESOM ocean adapter. "
            "Install it with:\n"
            "    pip install fesom_jax\n"
            "or:\n"
            "    pip install git+https://github.com/koldunovn/fesom_jax.git"
        ) from exc


# =============================================================================
# Config
# =============================================================================

class FesomOceanConfig(NamedTuple):
    """Configuration for the FESOM ocean model adapter.

    All three fields default to ``None`` ("use fesom_jax's own default").
    Resolution to concrete values happens in
    :meth:`FesomOceanModel.__init__`, which imports ``fesom_jax.config``
    lazily.  Constructing a :class:`FesomOceanConfig` therefore never
    requires ``fesom_jax``.
    """
    dt: Optional[float] = None
    k_ver: Optional[float] = None
    a_ver: Optional[float] = None
    # Vertical coordinate / free-surface mode. "linfs" (default, and
    # fesom_jax's own default) keeps layer thicknesses FIXED and carries
    # the free-surface volume change as a surface concentration/dilution
    # term -- tracer CONTENT is then conserved only to that
    # approximation (measured: volume-weighted heat drifts ~1e-6 relative
    # over 2 days of geostrophic adjustment, against ~1e-15 for the
    # z-star arms). "zstar" selects fesom_jax's ALE z-star coordinate, in
    # which thicknesses move with eta and content is conserved.
    # Static Python string (selects a compile-time branch), NOT a leaf.
    vertical_coordinate: str = "linfs"


# =============================================================================
# FesomOceanGrid — the grid seam
# =============================================================================

@dataclasses.dataclass(frozen=True)
class FesomOceanGrid:
    """legoESM grid facade over a FESOM :class:`fesom_jax.mesh.Mesh`.

    legoESM's generic ocean diagnostics ask a grid for ``.area``, ``.lon``,
    ``.lat``.  By presenting those here, every ``grid_type ==`` dispatch
    site in legoESM falls through to its generic ``else`` branch — no
    per-site FESOM branch is needed.

    WARNING — ``.area`` must be 1-D AND must be the volume-area
    ----------------------------------------------------------------
    ``mesh.area`` is ``(nod2D, nl)`` — per-level **upper-face** surface
    areas.  ``mesh.areasvol`` is ``(nod2D, nl)`` — per-level **volume**
    areas (the horizontal cross-section that multiplies ``dz`` to give a
    volume element).  RPE is a volume integral of ``z * rho``; its per-layer
    volume element is ``areasvol * dz``, NOT ``area * dz``.  Returning the
    wrong one would make ``compute_rpe`` return a plausible but **wrong**
    number — silently, because both arrays are 2-D and would broadcast.
    """

    mesh: Any  # fesom_jax.mesh.Mesh

    @property
    def grid_type(self) -> str:
        return "fesom"

    @property
    def area(self):
        return jnp.asarray(self.mesh.areasvol[:, 0], dtype=jnp.float64)

    @property
    def area_upper(self):
        return jnp.asarray(self.mesh.area[:, 0], dtype=jnp.float64)

    @property
    def lon(self):
        return jnp.asarray(self.mesh.geo_coord_nod2D[:, 0], dtype=jnp.float64)

    @property
    def lat(self):
        return jnp.asarray(self.mesh.geo_coord_nod2D[:, 1], dtype=jnp.float64)


# =============================================================================
# FesomOceanState — the state seam
# =============================================================================

@dataclasses.dataclass(frozen=True)
class FesomOceanState:
    """legoESM ocean-state facade over a native ``fesom_jax.state.State``.

    Pytree design (revised — mesh is NOT a meta field)
    --------------------------------------------------
    Registered via :func:`jax.tree_util.register_dataclass`.

    * **Leaves** (pytree children): ``inner``, ``uv_node``, ``H_bathy``,
      ``land_mask``.
    * **Aux data** (static, part of the treedef): ``nlev``,
      ``is_first_step``.

    The ``mesh`` is NOT stored on the state.  Storing a pytree-registered
    object as aux_data is invalid JAX: two value-identical meshes are not
    the same object, and treedef-equality attempts ``mesh == mesh``, which
    raises "truth value of an array is ambiguous".  It also forced the
    compiled function to be handed the *exact* object identity — fragile.

    Instead, the per-step node velocity is materialised as a real data leaf,
    :attr:`uv_node`, computed once per step inside
    :meth:`FesomOceanModel.step` via ``fesom_jax.pp.compute_vel_nodes``.
    The :attr:`u` / :attr:`v` properties are cheap slices of that leaf, so
    reading both (as the restart writer does) does the interpolation only
    once per step — previously it recomputed on every property access.

    ``T``, ``S``, ``eta``, ``u``, ``v`` and ``uv_elem`` are **properties**.
    ``u`` / ``v`` read the stored :attr:`uv_node` leaf; the others read
    ``inner`` directly.  ``with_inner`` was REMOVED: silently carrying a
    stale ``uv_node`` across an inner swap is exactly the class of bug
    this revision eliminates.  Mutators must use ``dataclasses.replace``
    and pass a fresh ``uv_node`` explicitly.

    ``is_first_step`` lives HERE, on the state, not on the model: a model
    shared across JIT boundaries cannot carry mutable Python state.  It is
    a **compile-time constant** whose two values produce two compiled
    variants of ``step_jit``; flipping it is a legitimate recompile, not a
    runtime state change.  Because it sits in the treedef as a meta field,
    it MUST NOT be carried through a ``lax.scan`` body — the carry's
    treedef would change between the first iteration (flag True) and the
    second (flag False).
    """

    inner: Any            # fesom_jax.state.State — live pytree leaf
    uv_node: jax.Array    # (nod2D, nl, 2) — current-step node velocity, live leaf
    H_bathy: Field        # live pytree leaf (immutable, mesh-derived)
    land_mask: Field      # live pytree leaf (immutable, mesh-derived)
    nlev: int             # static aux_data — mesh.nl - 1
    is_first_step: bool = True  # static aux_data — compile-time constant

    # ------------------------------------------------------------------
    # legoESM Field views (properties — always reflect current inner)
    # ------------------------------------------------------------------

    @property
    def T(self) -> Field:
        """Potential temperature, ``(n_cells, nlev)``, degC."""
        return Field(
            data=jnp.asarray(self.inner.T[:, :self.nlev], dtype=jnp.float64),
            name="T",
            dims=("cell", "nlev"),
            units="degC",
            long_name="potential temperature",
        )

    @property
    def S(self) -> Field:
        """Practical salinity, ``(n_cells, nlev)``, psu."""
        return Field(
            data=jnp.asarray(self.inner.S[:, :self.nlev], dtype=jnp.float64),
            name="S",
            dims=("cell", "nlev"),
            units="psu",
            long_name="practical salinity",
        )

    @property
    def eta(self) -> Field:
        """Sea-surface height, ``(n_cells,)``, m."""
        return Field(
            data=jnp.asarray(self.inner.eta_n, dtype=jnp.float64),
            name="eta",
            dims=("cell",),
            units="m",
            long_name="sea-surface height",
        )

    @property
    def u(self) -> Field:
        """Zonal velocity at NODES, ``(n_cells, nlev)``, m/s.

        Cheap slice of the stored :attr:`uv_node` data leaf, which
        :meth:`FesomOceanModel.step` computes ONCE per step via FESOM's
        own ``compute_vel_nodes(mesh, inner.uv)`` — the area-weighted
        element→node interpolation of the CURRENT element velocity.

        This is NOT the lagged ``inner.uvnode`` (FESOM stores that before
        ``uv`` is updated).  Reading both :attr:`u` and :attr:`v` in the
        same step no longer recomputes the interpolation twice.

        .. note::

            A budget built on ``u`` / ``v`` is still an APPROXIMATION:
            the node values are an *interpolation* of the prognostic
            element velocity.  For an exact budget use :attr:`uv_elem`.
        """
        return Field(
            data=jnp.asarray(self.uv_node[:, :self.nlev, 0], dtype=jnp.float64),
            name="u",
            dims=("cell", "nlev"),
            units="m/s",
            long_name="zonal velocity (current node interpolation of element velocity)",
        )

    @property
    def v(self) -> Field:
        """Meridional velocity at NODES, ``(n_cells, nlev)``, m/s.

        Cheap slice of the stored :attr:`uv_node` leaf — see :attr:`u`.
        """
        return Field(
            data=jnp.asarray(self.uv_node[:, :self.nlev, 1], dtype=jnp.float64),
            name="v",
            dims=("cell", "nlev"),
            units="m/s",
            long_name="meridional velocity (current node interpolation of element velocity)",
        )

    @property
    def uv_elem(self) -> Field:
        """Prognostic element velocity, ``(n_elem, nlev, 2)``, m/s."""
        return Field(
            data=jnp.asarray(self.inner.uv[:, :self.nlev, :], dtype=jnp.float64),
            name="uv_elem",
            dims=("elem", "nlev", "comp"),
            units="m/s",
            long_name="prognostic element velocity",
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_fesom(cls, inner: "State", mesh: "Mesh") -> "FesomOceanState":
        """Build the legoESM facade from a native state and mesh.

        ``uv_node`` is materialised here from ``inner.uv`` via FESOM's own
        ``compute_vel_nodes``.  For a rest state this is zeros.

        ``land_mask`` is DERIVED from ``mesh.node_layer_mask``: a node is
        ocean (1.0) iff it has at least one wet layer, land (0.0)
        otherwise.
        """
        _require_fesom_jax()
        from fesom_jax.pp import compute_vel_nodes

        nlev = int(mesh.nl) - 1
        uv_node = jnp.asarray(
            compute_vel_nodes(mesh, inner.uv), dtype=jnp.float64
        )

        # SIGN GATE: FESOM mesh.depth is NEGATIVE downward; legoESM H_bathy
        # is POSITIVE depth.  Negate once here.
        H_bathy = Field(
            data=jnp.asarray(-mesh.depth, dtype=jnp.float64),
            name="H_bathy",
            dims=("cell",),
            units="m",
            long_name="basin depth (positive downward)",
        )

        land_mask = Field(
            data=jnp.asarray(
                mesh.node_layer_mask.any(axis=1), dtype=jnp.float64
            ),
            name="land_mask",
            dims=("cell",),
            units="",
            long_name="ocean mask (1=ocean, 0=land)",
        )

        return cls(
            inner=inner,
            uv_node=uv_node,
            H_bathy=H_bathy,
            land_mask=land_mask,
            nlev=nlev,
        )


# Register as a JAX pytree.
# Leaves: inner, uv_node, H_bathy, land_mask.
# Aux (static, part of treedef): nlev, is_first_step.
#
# IMPORTANT: ``mesh`` is NOT a meta field.  A pytree-registered object
# with array contents cannot be hashed/equated as static aux_data; two
# value-identical meshes make treedef equality raise "truth value of an
# array is ambiguous" and force a recompile per object identity.  The
# per-step node velocity lives in ``uv_node`` as a real data leaf instead.
jax.tree_util.register_dataclass(
    FesomOceanState,
    data_fields=["inner", "uv_node", "H_bathy", "land_mask"],
    meta_fields=["nlev", "is_first_step"],
)


# =============================================================================
# Flat-bottom mesh derivation
# =============================================================================

def build_flat_bottom_mesh(
    mesh: "Mesh",
    H_max: float,
    nlev: int,
    *,
    land_lat_threshold: float = 90.0,
    zbar=None,
) -> "Mesh":
    """Return a flat-bottom copy of *mesh* with a uniform vertical grid.

    All horizontal geometry is preserved unchanged.  Only the vertical
    structure is replaced.  See the in-source docstring history for full
    detail; this revision changes ONLY the ``nlevels_nod2D_min``
    computation (see "nlevels_nod2D_min correction" below).
    """
    if H_max <= 0:
        raise ValueError(
            f"H_max must be a POSITIVE depth in metres; got H_max={H_max}."
        )
    if nlev < 1:
        raise ValueError(
            f"nlev must be >= 1; got nlev={nlev}."
        )
    if land_lat_threshold < 0.0 or land_lat_threshold > 90.0:
        raise ValueError(
            f"land_lat_threshold must be in [0, 90] degrees; "
            f"got {land_lat_threshold}."
        )

    _require_fesom_jax()
    from fesom_jax.mesh import level_masks

    nl = int(nlev) + 1
    nod2D = int(mesh.nod2D)
    elem2D = int(mesh.elem2D)

    # ``zbar=None`` keeps the uniform linspace (legacy behaviour). Passing the
    # host model's interface depths lets FESOM sit on the SAME vertical grid as
    # the other dycores -- without that, a comparison weights FESOM's column
    # with another arm's dz. Both conventions are NEGATIVE downward, so this is
    # a direct copy with NO sign flip; negating it would invert the column.
    if zbar is None:
        zbar = jnp.linspace(0.0, -float(H_max), nl, dtype=jnp.float64)
    else:
        zbar = jnp.asarray(zbar, dtype=jnp.float64)
        if zbar.shape != (nl,):
            raise ValueError(
                f"zbar has shape {tuple(zbar.shape)}; expected ({nl},) = "
                f"(nlev+1,) for nlev={nlev}."
            )
        if not bool(jnp.isclose(zbar[0], 0.0)):
            raise ValueError(
                f"zbar[0] = {float(zbar[0])} must be 0.0 (surface interface)."
            )
        if not bool(jnp.all(jnp.diff(zbar) < 0.0)):
            raise ValueError(
                "zbar must be strictly decreasing (FESOM depths are NEGATIVE "
                "downward)."
            )
        if not bool(jnp.isclose(zbar[-1], -float(H_max))):
            raise ValueError(
                f"zbar[-1] = {float(zbar[-1])} must equal -H_max = "
                f"{-float(H_max)}."
            )
    Z = 0.5 * (zbar[:-1] + zbar[1:])
    depth = jnp.full((nod2D,), -float(H_max), dtype=jnp.float64)
    zbar_3d_n = jnp.broadcast_to(zbar, (nod2D, nl))

    ulevels_nod2D = jnp.ones((nod2D,), dtype=jnp.int32)
    ulevels_nod2D_max = jnp.ones((nod2D,), dtype=jnp.int32)
    ulevels = jnp.ones((elem2D,), dtype=jnp.int32)

    nlevels_nod2D = jnp.full((nod2D,), nl, dtype=jnp.int32)
    nlevels = jnp.full((elem2D,), nl, dtype=jnp.int32)

    area = jnp.broadcast_to(mesh.area[:, 0:1], (nod2D, nl))
    areasvol = jnp.broadcast_to(mesh.areasvol[:, 0:1], (nod2D, nl))

    # -- DRY-COLUMN MECHANISM ---------------------------------------------
    geo_lat_rad = np.asarray(mesh.geo_coord_nod2D[:, 1])
    dry = np.abs(np.degrees(geo_lat_rad)) > float(land_lat_threshold)

    elem_nodes_np = np.asarray(mesh.elem_nodes)
    elem_dry = dry[elem_nodes_np].any(axis=1)

    # COLLAPSED node / element level bounds (numpy host-side).
    nlev_n_np = np.where(
        dry,
        np.asarray(ulevels_nod2D),
        np.asarray(nlevels_nod2D),
    ).astype(np.int32)
    nlev_e_np = np.where(
        elem_dry,
        np.asarray(ulevels),
        np.asarray(nlevels),
    ).astype(np.int32)

    # -- nlevels_nod2D_min CORRECTION (F1) --------------------------------
    # FESOM defines nlevels_nod2D_min as the MIN over all INCIDENT
    # ELEMENTS' nlevels, then min'd with the node's own nlevels_nod2D.
    # Verbatim from fesom_jax/scripts/prepare_mesh.py:
    #     pair_node = elem_nodes.reshape(-1)
    #     pair_elem = np.repeat(np.arange(E), 3)
    #     np.minimum.at(buf, pair_node, nlevels[pair_elem])
    #     nlevels_nod2D_min = np.minimum(buf, nlevels_nod2D)
    # The PREVIOUS code here set it to ``nlev_n_np`` (the node's own
    # bound), which left a WET node adjacent to a collapsed dry element
    # at 21 instead of 1.  We use the collapsed element bound
    # ``nlev_e_np`` so dry elements correctly pull their incident wet
    # nodes down to 1.
    pair_node = elem_nodes_np.reshape(-1)
    pair_elem = np.repeat(np.arange(elem2D, dtype=np.int64), 3)
    nl_min_buf = np.full(nod2D, np.iinfo(np.int32).max, dtype=np.int32)
    np.minimum.at(nl_min_buf, pair_node, nlev_e_np[pair_elem])
    nlev_n_min_np = np.minimum(nl_min_buf, nlev_n_np)
    # ---------------------------------------------------------------------

    node_layer_mask, node_iface_mask = level_masks(
        ulevels_nod2D, jnp.asarray(nlev_n_np, dtype=jnp.int32), nl
    )
    elem_layer_mask, elem_iface_mask = level_masks(
        ulevels, jnp.asarray(nlev_e_np, dtype=jnp.int32), nl
    )

    depth = jnp.where(jnp.asarray(dry), 0.0, depth)

    return dataclasses.replace(
        mesh,
        nl=nl,
        zbar=zbar,
        Z=Z,
        depth=depth,
        zbar_3d_n=zbar_3d_n,
        ulevels_nod2D=ulevels_nod2D,
        ulevels_nod2D_max=ulevels_nod2D_max,
        ulevels=ulevels,
        nlevels_nod2D=jnp.asarray(nlev_n_np, dtype=jnp.int32),
        # Correct FESOM formula: min over incident elements, then with
        # the node's own value.  See F1 correction block above.
        nlevels_nod2D_min=jnp.asarray(nlev_n_min_np, dtype=jnp.int32),
        nlevels=jnp.asarray(nlev_e_np, dtype=jnp.int32),
        area=area,
        areasvol=areasvol,
        node_layer_mask=node_layer_mask,
        node_iface_mask=node_iface_mask,
        elem_layer_mask=elem_layer_mask,
        elem_iface_mask=elem_iface_mask,
    )


# =============================================================================
# Initial condition
# =============================================================================

def resolve_ale_cfg(vertical_coordinate: str):
    """``AleConfig`` for *vertical_coordinate*, or ``None`` for linfs.

    RAISES on an unknown mode -- a silent fall-through to linfs would run
    a different vertical coordinate than the caller asked for (the repo's
    dispatch-hardening rule).
    """
    if vertical_coordinate == "linfs":
        return None
    if vertical_coordinate == "zstar":
        _require_fesom_jax()
        from fesom_jax.ale import AleConfig
        return AleConfig()
    raise ValueError(
        f"FesomOceanConfig.vertical_coordinate={vertical_coordinate!r} is "
        f"not supported; expected 'linfs' (fixed thicknesses) or 'zstar' "
        f"(ALE moving thicknesses).")


def element_centroid_lat_lon(mesh: "Mesh") -> tuple[jax.Array, jax.Array]:
    """GEOGRAPHIC (lat, lon) of every element centroid, radians.

    FESOM stores velocity at element centres, so any analytic velocity IC
    needs element coordinates. ``mesh.elem_center_x/y`` are ROTATED
    radians (the pi mesh's rotation is not guaranteed to be the identity),
    so the centroid is built from the three nodes' GEOGRAPHIC coordinates
    via a unit-vector mean -- rotation-independent and dateline-safe (a
    plain longitude average puts an element straddling 180 deg at 0 deg).
    """
    _require_fesom_jax()
    geo = jnp.asarray(mesh.geo_coord_nod2D, dtype=jnp.float64)
    lon_n, lat_n = geo[:, 0], geo[:, 1]
    xyz = jnp.stack([jnp.cos(lat_n) * jnp.cos(lon_n),
                     jnp.cos(lat_n) * jnp.sin(lon_n),
                     jnp.sin(lat_n)], axis=-1)          # (nod2D, 3)
    v = jnp.mean(xyz[mesh.elem_nodes], axis=1)          # (elem2D, 3)
    lat_e = jnp.arctan2(v[:, 2], jnp.hypot(v[:, 0], v[:, 1]))
    lon_e = jnp.mod(jnp.arctan2(v[:, 1], v[:, 0]), 2.0 * jnp.pi)
    return lat_e, lon_e


def geographic_to_rotated_vector(mesh: "Mesh", u_geo, v_geo):
    """Rotate an (east, north) GEOGRAPHIC vector into the mesh's ROTATED
    frame, per element. Returns ``(u_rot, v_rot)``.

    FESOM integrates velocity in the mesh's rotated frame
    (``coord_nod2D``), which on the packaged meshes is NOT the geographic
    frame (``geo_coord_nod2D``) -- they differ by up to 500 deg of
    longitude on the pi mesh. Writing east/north components straight into
    ``uv`` would therefore point an analytic IC in the wrong direction, by
    an amount that varies across the mesh (codex 2026-08-10).

    The local frame angle is measured FROM THE MESH: take the direction
    from the element's first node to its second, express it as an azimuth
    in each frame, and difference. That needs no Euler matrix and no
    private import from fesom_jax, and it degenerates to the identity on
    an unrotated mesh.
    """
    _require_fesom_jax()

    def _azimuth(coord):
        c = jnp.asarray(coord, dtype=jnp.float64)
        lon, lat = c[:, 0], c[:, 1]
        a = mesh.elem_nodes[:, 0]
        b = mesh.elem_nodes[:, 1]
        dlon = (lon[b] - lon[a] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        # Local tangent-plane components of the a->b direction.
        east = dlon * jnp.cos(0.5 * (lat[a] + lat[b]))
        north = lat[b] - lat[a]
        return jnp.arctan2(east, north)

    alpha = _azimuth(mesh.coord_nod2D) - _azimuth(mesh.geo_coord_nod2D)
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    u_geo = jnp.asarray(u_geo, dtype=jnp.float64)
    v_geo = jnp.asarray(v_geo, dtype=jnp.float64)
    # Rotating the FRAME by alpha rotates the components by -alpha.
    return (ca * u_geo + sa * v_geo, -sa * u_geo + ca * v_geo)


def with_fields(
    state: "FesomOceanState",
    mesh: "Mesh",
    *,
    T: jnp.ndarray | None = None,
    S: jnp.ndarray | None = None,
    eta: jnp.ndarray | None = None,
    uv_elem: jnp.ndarray | None = None,
) -> "FesomOceanState":
    """Return *state* with the given fields replaced (analytic-IC setter).

    ``T``/``S`` are node fields ``(nod2D, nlev)`` or ``(nod2D,)`` (a single
    column value broadcast down); ``eta`` is ``(nod2D,)``; ``uv_elem`` is
    ``(elem2D, nlev, 2)`` or ``(elem2D, 2)`` (broadcast down the column).
    Shapes are checked against the mesh rather than broadcast blindly -- a
    silently transposed IC is the failure mode this guards.

    The pad column (fesom_jax carries ``nl`` = ``nlev + 1`` slots) is
    filled by repeating the deepest value, and ``T_old``/``uv`` time levels
    are set together so the IC is self-consistent at step 0. ``uv_node`` is
    recomputed from the new element velocity, never carried stale.
    """
    _require_fesom_jax()
    from fesom_jax.pp import compute_vel_nodes

    inner = state.inner
    nlev = state.nlev
    n_node = int(mesh.nod2D)
    n_elem = int(mesh.elem2D)
    repl: dict[str, Any] = {}

    def _to_column(arr, name, n_expected, n_pad_slots):
        a = jnp.asarray(arr, dtype=jnp.float64)
        if a.ndim not in (1, 2):
            raise ValueError(
                f"with_fields: {name} must be rank 1 ({n_expected},) or rank "
                f"2 ({n_expected}, nlev); got shape {a.shape}.")
        if a.ndim == 1:
            if a.shape[0] != n_expected:
                raise ValueError(
                    f"with_fields: {name} has {a.shape[0]} entries but the "
                    f"mesh has {n_expected}.")
            a = jnp.broadcast_to(a[:, None], (n_expected, n_pad_slots))
            return a
        if a.shape[0] != n_expected:
            raise ValueError(
                f"with_fields: {name} leading dim {a.shape[0]} != mesh "
                f"{n_expected} (transposed IC?).")
        if a.shape[1] == n_pad_slots:
            return a
        if a.shape[1] != nlev:
            raise ValueError(
                f"with_fields: {name} has {a.shape[1]} levels; expected "
                f"{nlev} (or {n_pad_slots} including the pad slot).")
        return jnp.concatenate([a, a[:, -1:]], axis=1)

    n_slots = inner.T.shape[1]
    if T is not None:
        Tf = _to_column(T, "T", n_node, n_slots)
        repl.update(T=Tf, T_old=Tf)
    if S is not None:
        Sf = _to_column(S, "S", n_node, n_slots)
        repl.update(S=Sf, S_old=Sf)
    if eta is not None:
        e = jnp.asarray(eta, dtype=jnp.float64)
        if e.shape != (n_node,):
            raise ValueError(
                f"with_fields: eta shape {e.shape} != ({n_node},).")
        repl.update(eta_n=e)
    if uv_elem is not None:
        uv = jnp.asarray(uv_elem, dtype=jnp.float64)
        if uv.ndim not in (2, 3):
            raise ValueError(
                f"with_fields: uv_elem must be rank 2 ({n_elem}, 2) or rank "
                f"3 ({n_elem}, nlev, 2); got shape {uv.shape}.")
        if uv.ndim == 3 and uv.shape[1] not in (nlev, inner.uv.shape[1]):
            raise ValueError(
                f"with_fields: uv_elem has {uv.shape[1]} levels; expected "
                f"{nlev} (or {inner.uv.shape[1]} including the pad slot).")
        if uv.ndim == 2:
            if uv.shape != (n_elem, 2):
                raise ValueError(
                    f"with_fields: uv_elem shape {uv.shape} != "
                    f"({n_elem}, 2).")
            uv = jnp.broadcast_to(uv[:, None, :],
                                  (n_elem, inner.uv.shape[1], 2))
        elif uv.shape[0] != n_elem or uv.shape[-1] != 2:
            raise ValueError(
                f"with_fields: uv_elem shape {uv.shape} != "
                f"({n_elem}, nlev, 2).")
        elif uv.shape[1] == nlev:
            uv = jnp.concatenate([uv, uv[:, -1:, :]], axis=1)
        repl.update(uv=uv)

    if not repl:
        return state
    new_inner = dataclasses.replace(inner, **repl)
    facade = FesomOceanState.from_fesom(new_inner, mesh)
    return dataclasses.replace(
        facade,
        uv_node=jnp.asarray(compute_vel_nodes(mesh, new_inner.uv),
                            dtype=jnp.float64),
        is_first_step=state.is_first_step,
    )


def create_rest_state(
    mesh: "Mesh",
    z_coord,
    *,
    T_water_init_C: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
    stratified: bool = True,
    vertical_coordinate: str = "linfs",
) -> FesomOceanState:
    """Rest state on *mesh*: zero velocity, flat free surface, uniform S.

    ``stratified=True`` applies the SAME exponential profile the structured
    and MPAS rest states use --
    ``T(z) = T_deep + (T_water_init_C - T_deep) * exp(z / scale_depth)``
    evaluated at ``z_coord.z_full_ref`` (``legoesm.ocean.eos.scale_depth``) --
    so a cross-grid rest-state comparison differs only in the grid.
    ``stratified=False`` gives the uniform-T control (``T_water_init_C``
    everywhere), the barotropic-PGF isolation case.

    FESOM has no land in these idealized meshes (the flat-bottom mesh is
    all-wet), so there is no land-mask argument: the ``*_no_land`` and
    ``*_with_land`` matrix variants collapse to the same FESOM state.

    ``mesh`` MUST be the flattened flat-bottom mesh from
    :func:`build_flat_bottom_mesh`.
    """
    _require_fesom_jax()
    from fesom_jax.state import State
    from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH

    state = State.rest(mesh, T0=float(T_water_init_C), S0=float(S_uniform),
                       ale_cfg=resolve_ale_cfg(vertical_coordinate))
    if stratified:
        z_full = jnp.asarray(z_coord.z_full_ref, dtype=jnp.float64)
        T_profile = T_deep + (T_water_init_C - T_deep) * jnp.exp(
            z_full / _SCALE_DEPTH)
        # fesom_jax carries ONE padding level below the last wet layer
        # (FesomOceanState.T drops it), so the inner array is nlev+1 deep.
        # Repeat the bottom value into the pad rather than silently
        # broadcasting a wrong-length profile.
        n_inner = state.T.shape[1]
        if n_inner not in (T_profile.shape[0], T_profile.shape[0] + 1):
            raise ValueError(
                f"z_coord has {T_profile.shape[0]} levels but the FESOM "
                f"state column is {n_inner} deep (expected that or +1 for "
                f"the pad); the rest state would be built on a different "
                f"column than the model integrates."
            )
        if n_inner == T_profile.shape[0] + 1:
            T_profile = jnp.concatenate([T_profile, T_profile[-1:]])
        T_field = jnp.broadcast_to(T_profile[None, :], state.T.shape)
        state = dataclasses.replace(state, T=T_field, T_old=T_field)

    facade = FesomOceanState.from_fesom(state, mesh)
    return dataclasses.replace(facade, uv_node=jnp.zeros_like(facade.uv_node))


def create_lock_exchange_state(mesh: "Mesh", config: Any) -> FesomOceanState:
    """Build the lock-exchange IC on *mesh* and wrap it as a
    :class:`FesomOceanState`.

    Temperature is ``config.T_cold_C`` for nodes **west** of the front and
    ``config.T_warm_C`` for nodes **east** (legoESM convention).  Salinity
    is uniform at ``config.S_uniform``.

    ``uv_node`` is forced to ZEROS here (rest): the IC has zero velocity,
    so the element→node interpolation is exactly zero, but setting it
    explicitly makes the IC self-consistent regardless of any future change
    to ``compute_vel_nodes`` (and means the property :attr:`u` / :attr:`v`
    read zero on step 0 without paying for the interpolation at IC build
    time).

    .. warning::

        ``mesh`` MUST be the already-flattened flat-bottom mesh produced by
        :func:`build_flat_bottom_mesh`.

    Longitude / dateline convention (F2)
    ------------------------------------
    The front uses **geographic** longitude and normalises the DIFFERENCE
    (not the longitude itself) to remain correct across the dateline.
    """
    _require_fesom_jax()
    from fesom_jax.state import State

    state = State.rest(mesh, T0=config.T_reference_C, S0=config.S_uniform)

    geo_lon_deg = jnp.degrees(mesh.geo_coord_nod2D[:, 0])
    front_lon_deg = float(config.front_longitude)

    west_of_front = (
        (geo_lon_deg - front_lon_deg + 180.0) % 360.0 - 180.0
    ) < 0.0
    is_warm = ~west_of_front

    T_node = jnp.where(
        is_warm,
        jnp.asarray(config.T_warm_C, dtype=jnp.float64),
        jnp.asarray(config.T_cold_C, dtype=jnp.float64),
    )
    T_field = jnp.broadcast_to(T_node[:, None], state.T.shape)

    inner = dataclasses.replace(state, T=T_field, T_old=T_field)

    # Build the facade (which materialises uv_node from inner.uv via
    # compute_vel_nodes), then OVERRIDE uv_node to zeros — the IC is at
    # rest, and an explicit zero makes the contract obvious.
    facade = FesomOceanState.from_fesom(inner, mesh)
    return dataclasses.replace(
        facade,
        uv_node=jnp.zeros_like(facade.uv_node),
    )


# =============================================================================
# Model adapter
# =============================================================================

class FesomOceanModel:
    """legoESM ocean model adapter that runs the FESOM2-JAX dycore.

    The model holds ``self.mesh`` (a Python-side reference, NOT a pytree
    field on the state — see :class:`FesomOceanState`).  ``step`` uses it
    once per step to materialise :attr:`FesomOceanState.uv_node` from the
    freshly-stepped element velocity via ``compute_vel_nodes``.

    The model is STATELESS: ``is_first_step`` lives on the state (as a
    pytree meta field), not on the model.  This is required for
    JIT-safety.
    """

    def __init__(self, mesh: "Mesh", z_coord: Any, config: FesomOceanConfig):
        _require_fesom_jax()
        from fesom_jax import config as fconfig
        from fesom_jax import ssh as fssh
        from fesom_jax.params import Params

        self.mesh = mesh
        self.z_coord = z_coord
        self.config = config

        self._dt = (
            float(config.dt) if config.dt is not None
            else float(fconfig.DT_DEFAULT)
        )
        k_ver = (
            float(config.k_ver) if config.k_ver is not None
            else float(fconfig.K_VER)
        )
        a_ver = (
            float(config.a_ver) if config.a_ver is not None
            else float(fconfig.A_VER)
        )

        # None => linfs (fesom_jax's default); an AleConfig => z-star.
        # Static: it selects a compile-time branch inside step_jit.
        self._ale_cfg = resolve_ale_cfg(
            getattr(config, "vertical_coordinate", "linfs")
            if config is not None else "linfs")
        self._ssh_op = fssh.build_ssh_operator(mesh, dt=self._dt)
        self._stress_surf = jnp.zeros((int(mesh.elem2D), 2), dtype=jnp.float64)
        self._params = Params(
            k_ver=jnp.asarray(k_ver, dtype=jnp.float64),
            a_ver=jnp.asarray(a_ver, dtype=jnp.float64),
        )

    def step(self, state: FesomOceanState, dt: float) -> FesomOceanState:
        """Advance *state* by one baroclinic step of ``dt`` seconds.

        ``is_first_step`` is read from ``state.is_first_step`` (a STATIC
        meta field — a compile-time constant).  The returned state has
        ``is_first_step=False`` AND a fresh ``uv_node`` materialised from
        the new element velocity via ``compute_vel_nodes``.

        Because the new ``uv_node`` is computed here ONCE per step, the
        :attr:`FesomOceanState.u` and :attr:`FesomOceanState.v` properties
        are cheap slices — reading both (e.g. by the restart writer) does
        NOT recompute the interpolation twice.

        .. warning::

            Do NOT carry a :class:`FesomOceanState` whose
            ``is_first_step`` flips (True → False) through a
            ``lax.scan`` body — the carry's treedef changes.
        """
        dt = float(dt)
        if not math.isclose(dt, self._dt, rel_tol=1e-12):
            raise ValueError(
                f"dt mismatch: step called with dt={dt}, but the SSH "
                f"operator was built for dt={self._dt}. The SSH stiffness "
                f"matrix is dt-specific; refusing to silently reuse a "
                f"stale operator."
            )

        from fesom_jax import step as fstep
        from fesom_jax.pp import compute_vel_nodes

        new_inner = fstep.step_jit(
            state.inner,
            self.mesh,
            self._ssh_op,
            self._stress_surf,
            self._params,
            dt=self._dt,
            is_first_step=state.is_first_step,
            ale_cfg=self._ale_cfg,
        )

        # Materialise the per-step node velocity ONCE.  ``u`` / ``v``
        # become cheap slices of this leaf.
        new_uv_node = jnp.asarray(
            compute_vel_nodes(self.mesh, new_inner.uv), dtype=jnp.float64
        )

        # Explicit dataclasses.replace — there is no ``with_inner`` helper
        # any more (it would have silently carried a STALE uv_node).
        return dataclasses.replace(
            state,
            inner=new_inner,
            uv_node=new_uv_node,
            is_first_step=False,
        )

    def run(
        self, state: FesomOceanState, dt: float, n_steps: int
    ) -> FesomOceanState:
        """Advance *state* by ``n_steps`` baroclinic steps of ``dt`` seconds.

        This is an EAGER Python loop over :meth:`step`.  It correctly
        threads the ``is_first_step`` flag: step 1 reads it from the
        incoming state (True for a fresh IC, which selects the AB2
        first-step branch inside ``step_jit``), and steps 2..N run with
        ``is_first_step=False`` baked in (the outgoing state of step 1
        already carries ``False`` in its treedef).

        This method does NOT implement FESOM's checkpointed
        ``lax.scan``-based integration path
        (``fesom_jax.integrate.integrate``); that path is not wired through
        this adapter.  The eager loop here exists to make the
        flag-threading contract explicit and testable; legoESM's matrix
        already advances steps by repeated ``.step`` calls and would
        produce identical results.
        """
        if n_steps < 1:
            raise ValueError(
                f"n_steps must be >= 1; got n_steps={n_steps}."
            )
        s = self.step(state, dt)
        for _ in range(n_steps - 1):
            s = self.step(s, dt)
        return s
