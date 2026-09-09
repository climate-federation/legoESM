r"""Multi-device SPMD step for the MPAS/Voronoi OCEAN (route-B, pure JAX).

The GPU twin of :mod:`legoesm.parallel.voronoi_mpi`'s route-A (mpi4jax) lane
and the Voronoi analogue of :mod:`legoesm.ocean.dynamics.sharded_ocean_step`
(lat-lon bands).  Nothing here re-derives partitioning, halo transport or
numerics: the mesh is reordered + padded by
:func:`~legoesm.parallel.voronoi_partition.reorder_voronoi_for_sharding`, the
per-device local meshes and the coloured ``ppermute`` halo schedule come from
:mod:`legoesm.parallel.sharded_dynamics` (the atmosphere's MPAS SPMD lane), and
the step itself is ``MPASOceanModel._step_impl`` run on each device's padded
local mesh with a ``ppermute``-backed :class:`~legoesm.parallel.voronoi_mpi.
MPASOceanHaloRefresh` — the SAME in-step stage-frontier refresh sites the MPI
lane audited (``docs/performance/scaling/mpas_ocean_distributed_stage_audit.md``).

Layout
------
The GLOBAL state lives OUTSIDE ``shard_map`` as ``jax.Array``s over the
(reordered + padded) global mesh, sharded in contiguous OWNED blocks along the
cell / edge axis (``P("device")``).  Per-step forcing (bulk fluxes, sea-ice
thermodynamics, sponge ramp, SSS restoring, freeze cap) is elementwise and runs
on those sharded arrays exactly as in the lat-lon lane.  One ``shard_map`` per
step:

1. packs the owned cell prognostics (T, S, eta, w, land_mask, H_bathy, tke, the
   per-cell forcing) into one buffer and the owned edge field (u) into another,
   fills the ``(owned + halo)`` local buffers with the coloured ``ppermute``
   rounds (``ppermute_halo_fill``),
2. runs ``_step_impl`` on the device's padded local mesh with
   ``halo_refresh`` re-arming the halo at each dependency frontier (cells /
   edges / both / vertices, all ``ppermute``), owned-cell masks for every
   global reduction and ``psum`` over the ``"device"`` axis as the reducer,
3. returns the owned rows.

Reductions inside the step (barotropic PCG dots, eta-floor redistribution,
conservation fixer) route through the ``"spmd"`` halo backend armed on the
device mesh (:func:`legoesm.parallel.reductions.spmd_reduce_axis`).

Refusals (loud, at build time — never a silent serial fallback): a mesh not
prepared for ``n_devices``; ``use_baroclinic_rho_ref`` (a device-local
horizontal mean); ``runoff_depth_spread_map`` (a per-cell static the model
bakes in at construction).  ``normalize_freshwater`` IS supported: its eta and
virtual-salt means take the owned-masked psum through the refresh context.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from legoesm.core.field import Field
from legoesm.core.state import MPASOceanState
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.parallel.mesh import (
    DeviceConfig,
    create_voronoi_device_mesh,
    multiprocess_safe_device_put,
)
from legoesm.parallel.sharded_dynamics import (
    build_ppermute_schedule,
    build_voronoi_partition_infra,
    greedy_edge_coloring,
    ppermute_halo_fill,
)
from legoesm.parallel.voronoi_mpi import MPASOceanHaloRefresh

logger = logging.getLogger(__name__)

SPMD_AXIS = "device"
_DEFAULT_HALO_DEPTH = 2


class MPASOceanSPMDLayout(NamedTuple):
    """Everything the sharded step needs, built once per (mesh, n_devices)."""

    dev_config: DeviceConfig
    n_devices: int
    n_cells: int            # padded global cell count (divisible by n_devices)
    n_edges: int
    n_vertices: int
    n_cells_real: int       # cells of the ORIGINAL mesh (before padding)
    cells_per: int
    edges_per: int
    max_lc: int
    max_le: int
    max_lv: int
    stacked_meshes: Any     # VoronoiMesh pytree, leaves (n_dev, max_l*), device-put
    halo_args: tuple        # per-round (send_c, recv_c, send_e, recv_e), device-put
    ppermute_perms: list
    vhalo_args: tuple       # per-round (send_v, recv_v, se0, re0), device-put
    vppermute_perms: list
    gather_cells: np.ndarray    # (n_dev, max_lc) global cell index per local slot
    gather_edges: np.ndarray    # (n_dev, max_le)
    local_vertices: list        # per device: global vertex ids of its local slots
    vertex_owner: np.ndarray    # (nVertices,) owning device (smallest incident cell)
    upup: tuple | None      # stacked (upup_pos, upup_neg) per device, or None
    mesh_statics: dict      # Python ints/floats re-applied to the local mesh in-body

    @property
    def cell_sharding(self) -> NamedSharding:
        return self.dev_config.face_sharding

    @property
    def replicated_sharding(self) -> NamedSharding:
        return self.dev_config.replicated_sharding


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _build_vertex_ppermute_schedule(partitions, n_dev, vertex_owner, max_lv):
    """Coloured ppermute schedule for the vertex lane (the K_zeta_bih del4
    mid-refresh is the only vertex consumer).

    ``vertex_owner[g]`` is the device owning the SMALLEST incident cell of
    ``g`` — the mesh's own vertex-owner rule (``reorder_voronoi_for_sharding``)
    — so the source device owns a cell touching ``g`` and therefore holds the
    whole 2-ring the vertex Laplacian needs inside its depth-2 halo.  The
    partition infra's equal vertex blocks are NOT used here: they are an
    artefact of the atmosphere lane, which never exchanges vertex fields.
    Every local vertex of every device that it does not own is refreshed
    (owned = ``vertex_owner[g] == d``); sends index the source's FULL local
    vertex array, receives target the local slot, padding targets the garbage
    slot ``max_lv``.
    """
    send_map: dict[tuple[int, int], list[int]] = {}
    recv_map: dict[tuple[int, int], list[int]] = {}
    pairs: set[tuple[int, int]] = set()
    for d, part in enumerate(partitions):
        for h in range(part.n_local_vertices):
            g = int(part.local_vertices[h])
            owner = int(vertex_owner[g])
            if owner == d:
                continue
            src_local = int(partitions[owner].vertex_g2l[g])
            if src_local < 0:
                raise RuntimeError(
                    f"vertex {g} needed by device {d} is not local to its "
                    f"owner {owner} (owner of its smallest incident cell); "
                    "cannot build the vertex ppermute schedule")
            send_map.setdefault((owner, d), []).append(src_local)
            recv_map.setdefault((d, owner), []).append(h)
            pairs.add((min(d, owner), max(d, owner)))
    if not pairs:
        return [], []
    colors = greedy_edge_coloring(pairs)
    n_rounds = max(colors.values()) + 1
    rounds: dict[int, list[tuple[int, int]]] = {}
    for pair, c in colors.items():
        rounds.setdefault(c, []).append(pair)
    perms_out, args_out = [], []
    for r in range(n_rounds):
        partner: dict[int, int] = {}
        perm: list[tuple[int, int]] = []
        max_n = 1
        for u, v in rounds[r]:
            perm += [(u, v), (v, u)]
            partner[u] = v
            partner[v] = u
            max_n = max(max_n, len(send_map.get((u, v), [])),
                        len(send_map.get((v, u), [])))
        sv = np.zeros((n_dev, max_n), dtype=np.int64)
        rv = np.full((n_dev, max_n), max_lv, dtype=np.int64)
        for d in range(n_dev):
            if d not in partner:
                continue
            dp = partner[d]
            for j, idx in enumerate(send_map.get((d, dp), [])):
                sv[d, j] = idx
            for j, pos in enumerate(recv_map.get((d, dp), [])):
                rv[d, j] = pos
        perms_out.append(perm)
        args_out.append((sv, rv,
                         np.zeros((n_dev, max_n), dtype=np.int64),   # edge dummy
                         np.zeros((n_dev, max_n), dtype=np.int64)))
    return perms_out, args_out


def build_mpas_ocean_spmd_layout(
    mesh: VoronoiMesh,
    n_devices: int,
    *,
    halo_depth: int = _DEFAULT_HALO_DEPTH,
    n_cells_real: int | None = None,
    tracer_advection: str = "upwind",
    nlev: int = 1,
    devices=None,
) -> MPASOceanSPMDLayout:
    """Build the SPMD layout for ``mesh`` (ALREADY reordered + padded for
    ``n_devices`` by :func:`reorder_voronoi_for_sharding`) and ARM the
    ``"spmd"`` halo backend on the device mesh so in-step reductions psum.

    ``n_cells_real`` is the pre-padding cell count (defaults to ``mesh.nCells``
    when no padding happened); padded tail cells must be land in the state
    (see :func:`mask_padded_cells`).
    """
    if mesh.nCells % n_devices or mesh.nEdges % n_devices:
        raise ValueError(
            f"mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) is not "
            f"padded for {n_devices} devices; prepare it with "
            "reorder_voronoi_for_sharding(mesh, n_devices) first.")
    dev = create_voronoi_device_mesh(
        mesh.nCells, mesh.nEdges, mesh.nVertices,
        n_devices=n_devices, devices=devices)
    if dev.n_devices != n_devices or dev.mesh is None:
        raise ValueError(
            f"requested {n_devices} devices but only {dev.n_devices} usable "
            f"({len(jax.devices())} visible)")
    n_dev = n_devices
    cells_per = mesh.nCells // n_dev
    edges_per = mesh.nEdges // n_dev
    (stacked, gather_cells, gather_edges, _noc, _noe, max_lc, max_le,
     partitions, cell_owner) = build_voronoi_partition_infra(
        mesh, n_dev, halo_depth=halo_depth)
    max_lv = int(stacked.latVertex.shape[1])
    sharding = dev.face_sharding

    def _put(x):
        return multiprocess_safe_device_put(jnp.asarray(x), sharding)

    stacked_dev = jax.tree.map(_put, stacked)

    sched = build_ppermute_schedule(
        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
        cell_width=3 * nlev + 8, edge_width=nlev)
    halo_args = tuple(
        (_put(sched["send_cell_idx"][r]), _put(sched["recv_cell_pos"][r]),
         _put(sched["send_edge_idx"][r]), _put(sched["recv_edge_pos"][r]))
        for r in range(sched["n_rounds"]))
    # vertex owner = owner (cell block) of the smallest incident real cell;
    # padded ghost cells never touch a vertex (connectivity -1 / 0-filled),
    # but a ghost's ``cellsOnVertex`` entries are ignored via the >= 0 guard.
    cov = np.asarray(mesh.cellsOnVertex)
    cov_safe = np.where(cov >= 0, cov, mesh.nCells)
    min_cell_v = np.min(cov_safe, axis=0)
    if np.any(min_cell_v >= mesh.nCells):
        raise ValueError("mesh has a vertex with no incident cell")
    vertex_owner = np.minimum(min_cell_v // cells_per, n_dev - 1)
    vperms, vargs = _build_vertex_ppermute_schedule(
        partitions, n_dev, vertex_owner, max_lv)
    vhalo_args = tuple(tuple(_put(a) for a in arg) for arg in vargs)

    upup = None
    if tracer_advection in ("tvd", "superbee"):
        from legoesm.ocean.dynamics.advection_mpas import compute_upup_cells
        pos, neg = [], []
        for d in range(n_dev):
            lm = jax.tree.map(lambda x, _d=d: x[_d], stacked)
            lm = lm._replace(**_mesh_statics(mesh, max_lc, max_le, max_lv))
            p_, n_ = compute_upup_cells(lm)
            pos.append(np.asarray(p_))
            neg.append(np.asarray(n_))
        upup = (_put(np.stack(pos)), _put(np.stack(neg)))

    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    set_halo_backend("spmd")
    set_spmd_mesh(dev.mesh)

    logger.info(
        "MPAS ocean SPMD layout: %d devices, cells_per=%d (max local %d), "
        "edges_per=%d (max local %d), %d ppermute rounds (+%d vertex rounds)",
        n_dev, cells_per, max_lc, edges_per, max_le, sched["n_rounds"],
        len(vperms))
    return MPASOceanSPMDLayout(
        dev_config=dev, n_devices=n_dev,
        n_cells=mesh.nCells, n_edges=mesh.nEdges, n_vertices=mesh.nVertices,
        n_cells_real=(mesh.nCells if n_cells_real is None else int(n_cells_real)),
        cells_per=cells_per, edges_per=edges_per,
        max_lc=max_lc, max_le=max_le, max_lv=max_lv,
        stacked_meshes=stacked_dev, halo_args=halo_args,
        ppermute_perms=sched["ppermute_perms"],
        vhalo_args=vhalo_args, vppermute_perms=vperms,
        gather_cells=np.asarray(gather_cells), gather_edges=np.asarray(gather_edges),
        local_vertices=[np.asarray(pt.local_vertices) for pt in partitions],
        vertex_owner=np.asarray(vertex_owner),
        upup=upup,
        mesh_statics=_mesh_statics(mesh, max_lc, max_le, max_lv),
    )


def _mesh_statics(mesh: VoronoiMesh, max_lc: int, max_le: int, max_lv: int) -> dict:
    """The VoronoiMesh scalar leaves, as Python values for the LOCAL padded
    mesh — stacking turned them into arrays, and consumers use them as shapes."""
    return dict(nCells=max_lc, nEdges=max_le, nVertices=max_lv,
                maxEdges=int(mesh.maxEdges), vertexDegree=int(mesh.vertexDegree),
                radius=float(mesh.radius))


def disarm_mpas_ocean_spmd() -> None:
    """Restore the halo backend to ``"local"`` and clear the SPMD mesh armed by
    :func:`build_mpas_ocean_spmd_layout` — call when the sharded run is over so
    a later serial step in the same interpreter does not pick the ``"device"``
    reduction axis and ``psum`` outside a ``shard_map``."""
    from legoesm.grids.halo import set_halo_backend, set_spmd_mesh
    set_halo_backend("local")
    set_spmd_mesh(None)


def refuse_unsupported_spmd_config(config) -> None:
    """Loud refusals for MPASOceanConfig features whose distributed form does
    not exist on this lane (never a silent single-device fallback)."""
    if bool(getattr(config, "use_baroclinic_rho_ref", False)):
        raise NotImplementedError(
            "MPAS ocean SPMD: use_baroclinic_rho_ref=True is unsupported — its "
            "dynamic rho_ref(z) is a device-local unmasked horizontal mean; use "
            "the static rho_ref_z (use_static_baroclinic_rho_ref) or rho_0.")
    if getattr(config, "runoff_depth_spread_map", None) is not None:
        raise NotImplementedError(
            "MPAS ocean SPMD: runoff_depth_spread_map is unsupported — the "
            "model bakes this per-cell static in at construction, so it is "
            "not re-sliced per device.")


# ---------------------------------------------------------------------------
# Scatter / gather
# ---------------------------------------------------------------------------

def n_real_cells(mesh: VoronoiMesh) -> int:
    """Cells of the ORIGINAL mesh inside a padded one: padded ghosts are the
    only cells with ``nEdgesOnCell == 0`` (real SCVT cells have 5 or 6)."""
    return int(np.sum(np.asarray(mesh.nEdgesOnCell) > 0))


def mask_padded_cells(state: MPASOceanState, n_cells_real: int) -> MPASOceanState:
    """Force the padded tail cells (``>= n_cells_real``) to land: zero
    land_mask and bathymetry.  Padded ghost cells sit at lat/lon 0, so the
    rest-state builders classify them as OCEAN otherwise."""
    if n_cells_real >= state.land_mask.data.shape[0]:
        return state
    lm = state.land_mask.data.at[n_cells_real:].set(0)
    hb = state.H_bathy.data.at[n_cells_real:].set(0)
    return state._replace(land_mask=state.land_mask.replace(data=lm),
                          H_bathy=state.H_bathy.replace(data=hb))


def _leaf_spec(layout: MPASOceanSPMDLayout, arr) -> P:
    if arr is None:
        return P()
    n0 = int(arr.shape[0]) if getattr(arr, "ndim", 0) else -1
    if n0 in (layout.n_cells, layout.n_edges):
        return P(SPMD_AXIS)
    return P()


def shard_state_mpas_ocean_spmd(state: MPASOceanState, layout: MPASOceanSPMDLayout):
    """Place a GLOBAL (reordered + padded) state on the device mesh: cell /
    edge leaves in contiguous owned blocks, ``rho_ref_z`` replicated."""
    from legoesm.parallel.latlon_spmd import shard_leaf
    multi = jax.process_count() > 1

    def _f(fld):
        if fld is None:
            return None
        spec = _leaf_spec(layout, fld.data)
        sh = NamedSharding(layout.dev_config.mesh, spec)
        return fld.replace(data=shard_leaf(jnp.asarray(fld.data), sh, multiprocess=multi))

    return MPASOceanState(**{k: _f(getattr(state, k)) for k in state._fields})


def shard_cell_array_spmd(arr, layout: MPASOceanSPMDLayout, axis: int = -1):
    """Shard an array whose ``axis`` runs over the padded global cells (a
    forcing stack ``(n_steps, nCells)`` -> ``P(None, "device")``)."""
    from legoesm.parallel.latlon_spmd import shard_leaf
    arr = jnp.asarray(arr)
    ax = axis % arr.ndim
    if int(arr.shape[ax]) != layout.n_cells:
        raise ValueError(
            f"shard_cell_array_spmd: axis {ax} has length {arr.shape[ax]}, "
            f"expected the padded cell count {layout.n_cells}")
    spec = P(*([None] * ax + [SPMD_AXIS] + [None] * (arr.ndim - ax - 1)))
    return shard_leaf(arr, NamedSharding(layout.dev_config.mesh, spec),
                      multiprocess=jax.process_count() > 1)


def shard_cell_stack_spmd(stack, layout: MPASOceanSPMDLayout):
    """Pytree twin of :func:`shard_cell_array_spmd` for ``(n, nCells)``
    forcing stacks (dict / NamedTuple leaves); non-cell leaves pass through."""
    def _f(x):
        if hasattr(x, "shape") and x.ndim >= 2 and int(x.shape[1]) == layout.n_cells:
            return shard_cell_array_spmd(x, layout, axis=1)
        if hasattr(x, "shape") and x.ndim >= 1 and int(x.shape[0]) == layout.n_cells:
            return shard_cell_array_spmd(x, layout, axis=0)
        return x
    return jax.tree.map(_f, stack)


def gather_state_mpas_ocean_spmd(state, layout: MPASOceanSPMDLayout, *, to_host: bool = False):
    """Fully-replicated (every process addressable) GLOBAL state in the
    reordered + padded cell order — the layout every driver consumer (grid,
    snapshots, restarts) uses on this lane."""
    from legoesm.parallel.latlon_spmd import replicate_leaf
    rep = layout.dev_config.replicated_sharding
    multi = jax.process_count() > 1

    def _g(x):
        if not isinstance(x, jax.Array):
            return x
        out = replicate_leaf(x, rep, multiprocess=multi)
        return np.asarray(out) if to_host else out
    return jax.tree.map(_g, state)


# ---------------------------------------------------------------------------
# Packing helpers (run INSIDE shard_map)
# ---------------------------------------------------------------------------

def _pack(fields, dtype):
    """(n, ...) fields -> (n, W) buffer + per-field (shape_tail, dtype, width)."""
    cols, meta = [], []
    for f in fields:
        tail = tuple(f.shape[1:])
        w = int(np.prod(tail)) if tail else 1
        cols.append(f.reshape(f.shape[0], w).astype(dtype))
        meta.append((tail, f.dtype, w))
    if not cols:
        return None, meta
    return jnp.concatenate(cols, axis=1), meta


def _unpack(buf, meta):
    out, off = [], 0
    for tail, dt, w in meta:
        sl = buf[:, off:off + w]
        out.append(sl.reshape((buf.shape[0],) + tail).astype(dt))
        off += w
    return out


def spmd_vertex_refresh_probe(layout: MPASOceanSPMDLayout, stacked_vertex_field):
    """Run ONLY the vertex lane on a stacked ``(n_dev, max_lv[, k])`` local
    vertex field and return the refreshed stacked field (host numpy).

    A direct semantic gate for the vertex ppermute schedule: fill each device's
    OWNED vertices (owner = smallest incident cell's block) with their global
    ids and everything else with a sentinel; after the refresh every LOCAL
    vertex of every device must carry its global id.  The step's parity test
    cannot see this lane on a depth-2 halo (the local vertex Laplacian is
    already complete there), so this probe is what keeps it honest.
    """
    from legoesm.parallel.shard_map_compat import shard_map
    if not layout.vppermute_perms:
        return np.asarray(stacked_vertex_field)
    max_lv, vperms = layout.max_lv, layout.vppermute_perms

    def _body(v_sl, vhalo_sl):
        v = v_sl[0]
        dt_ = v.dtype
        buf, meta = _pack([v], dt_)
        loc, _ = ppermute_halo_fill(buf, jnp.zeros((1, 0), dtype=dt_), vhalo_sl,
                                    vperms, max_lv, 1)
        return _unpack(loc, meta)[0][None]

    fn = shard_map(_body, mesh=layout.dev_config.mesh,
                   in_specs=(P(SPMD_AXIS), jax.tree.map(lambda _: P(SPMD_AXIS), layout.vhalo_args)),
                   out_specs=P(SPMD_AXIS), check_vma=False)
    x = multiprocess_safe_device_put(jnp.asarray(stacked_vertex_field), layout.cell_sharding)
    return np.asarray(fn(x, layout.vhalo_args))


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

def make_sharded_mpas_ocean_step(model, layout: MPASOceanSPMDLayout) -> Callable:
    """Build ``spmd_step(state, dt, aux=None, *, freshwater=None,
    surface_forcing=None, sponge=None) -> state`` for a model whose ``mesh`` is
    the reordered + padded GLOBAL mesh of ``layout``.

    Call-compatible with the lat-lon lane's ``spmd_step`` (``aux`` accepted and
    ignored).  Forcing leaves: traced per-cell / per-edge arrays ride the entry
    ``ppermute`` pack; CONCRETE per-cell arrays (closure constants such as the
    sponge's ``T_ref`` / ``S_ref``) are localised once per array identity and
    passed as stacked device operands; scalars replicate.
    """
    from legoesm.parallel.reductions import batch_psum_spmd
    from legoesm.parallel.shard_map_compat import shard_map

    refuse_unsupported_spmd_config(model.config)
    if int(model.mesh.nCells) != layout.n_cells or int(model.mesh.nEdges) != layout.n_edges:
        raise ValueError(
            "make_sharded_mpas_ocean_step: model.mesh must be the layout's "
            f"prepared global mesh (nCells {model.mesh.nCells} vs {layout.n_cells})")
    if getattr(model, "_upup_pos", None) is not None and layout.upup is None:
        raise ValueError(
            "model uses the tvd/superbee upup stencil but the layout was built "
            "without tracer_advection=...; rebuild the layout with "
            f"tracer_advection={model.config.tracer_advection!r}")

    n_dev = layout.n_devices
    cells_per, edges_per = layout.cells_per, layout.edges_per
    max_lc, max_le, max_lv = layout.max_lc, layout.max_le, layout.max_lv
    perms, vperms = layout.ppermute_perms, layout.vppermute_perms
    jax_mesh = layout.dev_config.mesh
    statics = layout.mesh_statics
    owned_c = jnp.arange(max_lc) < cells_per
    owned_e = jnp.arange(max_le) < edges_per
    n_cells, n_edges = layout.n_cells, layout.n_edges
    _local_cache: dict[int, Any] = {}

    def _entity_kind(x):
        if not hasattr(x, "shape") or x.ndim == 0:
            return "rep"
        if int(x.shape[0]) == n_cells:
            return "cell"
        if int(x.shape[0]) == n_edges:
            return "edge"
        return "rep"

    def _localize_by_index(x, idx):
        loc = np.asarray(x)[idx]                              # (n_dev, max_l*, ...)
        return multiprocess_safe_device_put(jnp.asarray(loc), layout.cell_sharding)

    # The vertical coordinate may carry PER-CELL statics (partial cells:
    # h_partial, bottom_level, is_active, t_depth_ref; NEMO EEN operands).
    # Localise them per device once; replicated leaves stay closure constants.
    # (First MPAS SPMD smoke, job 27326448: the global (nCells, nlev) h_partial
    # met the local (max_lc, 1) water column in compute_layer_thickness.)
    z_leaves, z_tree = jax.tree.flatten(model.z_coord)
    z_kinds = [_entity_kind(x) for x in z_leaves]
    z_static = tuple(
        _localize_by_index(x, layout.gather_cells if k == "cell" else layout.gather_edges)
        for x, k in zip(z_leaves, z_kinds) if k != "rep")
    # A per-cell / per-edge array inside the CONFIG would reach the local step
    # at global size too; none is localised here, so refuse loudly.
    for x in jax.tree.leaves(model.config):
        if _entity_kind(x) != "rep":
            raise NotImplementedError(
                "MPAS ocean SPMD: model.config carries a per-cell/per-edge array "
                f"leaf of shape {tuple(x.shape)} that is not localised per device")

    def _fill_cells_edges(cell_owned, edge_owned, halo_sl):
        """Owned buffers -> (owned+halo) local buffers via the coloured rounds.
        Zero-width dummies stand in for an absent entity class."""
        if cell_owned is None:
            cell_owned = jnp.zeros((cells_per, 0), dtype=edge_owned.dtype)
        if edge_owned is None:
            edge_owned = jnp.zeros((edges_per, 0), dtype=cell_owned.dtype)
        return ppermute_halo_fill(cell_owned, edge_owned, halo_sl, perms, max_lc, max_le)

    def _make_refresh(halo_sl, vhalo_sl):
        def _cells(*fields):
            dt_ = jnp.result_type(*[f.dtype for f in fields])
            buf, meta = _pack([f[:cells_per] for f in fields], dt_)
            loc, _ = _fill_cells_edges(buf, None, halo_sl)
            return tuple(_unpack(loc, meta))

        def _edges(*fields):
            dt_ = jnp.result_type(*[f.dtype for f in fields])
            buf, meta = _pack([f[:edges_per] for f in fields], dt_)
            _, loc = _fill_cells_edges(None, buf, halo_sl)
            return tuple(_unpack(loc, meta))

        def _both(edge_fields, cell_fields):
            edge_fields, cell_fields = tuple(edge_fields), tuple(cell_fields)
            dt_ = jnp.result_type(*[f.dtype for f in edge_fields + cell_fields])
            ebuf, emeta = _pack([f[:edges_per] for f in edge_fields], dt_)
            cbuf, cmeta = _pack([f[:cells_per] for f in cell_fields], dt_)
            cloc, eloc = _fill_cells_edges(cbuf, ebuf, halo_sl)
            return (tuple(_unpack(eloc, emeta)) if emeta else (),
                    tuple(_unpack(cloc, cmeta)) if cmeta else ())

        def _vertices(*fields):
            if not vperms:
                return tuple(fields)
            dt_ = jnp.result_type(*[f.dtype for f in fields])
            buf, meta = _pack(list(fields), dt_)     # FULL local vertex arrays
            dummy_e = jnp.zeros((1, 0), dtype=dt_)
            loc, _ = ppermute_halo_fill(buf, dummy_e, vhalo_sl, vperms, max_lv, 1)
            return tuple(_unpack(loc, meta))

        return MPASOceanHaloRefresh(
            edges=_edges, cells=_cells, both=_both, vertices=_vertices,
            owned_mask_cells=owned_c, owned_mask_edges=owned_e,
            global_sum=lambda vals: batch_psum_spmd(list(vals), SPMD_AXIS),
        )

    def _classify(leaves):
        """Split forcing leaves into (traced cell, traced edge, concrete cell,
        replicated) with positional bookkeeping."""
        kinds = []
        for x in leaves:
            if x is None or not hasattr(x, "shape") or x.ndim == 0:
                kinds.append("rep")
            elif int(x.shape[0]) == n_cells:
                # Route-B multicontroller: a localised static would be a
                # process-spanning sharded array captured by closure in the
                # caller's jit (refused by JAX); ride the traced pack instead.
                kinds.append("cell" if (isinstance(x, jax.core.Tracer)
                                        or jax.process_count() > 1) else "cell_static")
            elif int(x.shape[0]) == n_edges:
                kinds.append("edge")
            else:
                raise ValueError(
                    "MPAS ocean SPMD: forcing leaf of shape "
                    f"{tuple(x.shape)} is neither per-cell ({n_cells}), per-edge "
                    f"({n_edges}) nor scalar")
        return kinds

    def _localize_static(x):
        # One slot per (shape, dtype): a rebuilt reference array of the same
        # shape replaces the previous entry, so the cache is bounded by the
        # number of distinct static forcing shapes, not by call count.
        key = (tuple(x.shape), str(getattr(x, "dtype", None)))
        hit = _local_cache.get(key)
        if hit is not None and hit[0] is x:
            return hit[1]
        host = np.asarray(x)
        loc = host[layout.gather_cells]                       # (n_dev, max_lc, ...)
        # Concrete even when called from inside a jit trace (the block scan):
        # the localised static is a per-device CONSTANT, not a traced operand.
        with jax.ensure_compile_time_eval():
            loc = multiprocess_safe_device_put(jnp.asarray(loc), layout.cell_sharding)
        _local_cache[key] = (x, loc)
        return loc

    def _local_step(u_o, cpack_o, rho_ref, rep_leaves, static_cells, z_sl, dt,
                    cmeta, ckinds, forcing_tree, n_state_cells,
                    mesh_sl, halo_sl, vhalo_sl, upup_sl):
        """INSIDE shard_map: fill halos, step on the local mesh, return owned."""
        cloc, uloc = _fill_cells_edges(cpack_o, u_o, halo_sl)
        cell_fields = _unpack(cloc, cmeta)
        T_l, S_l, eta_l, w_l, hb_l, lm_l = cell_fields[:6]
        rest = cell_fields[6:]
        tke_l = None
        if n_state_cells == 7:
            tke_l = rest[0]
            rest = rest[1:]
        my_mesh = jax.tree.map(lambda x: x[0], mesh_sl)._replace(**statics)
        model_l = copy.copy(model)
        model_l.mesh = my_mesh
        it_z = iter(z_sl)
        model_l.z_coord = jax.tree.unflatten(
            z_tree, [next(it_z)[0] if k != "rep" else x
                     for x, k in zip(z_leaves, z_kinds)])
        if upup_sl is not None:
            model_l._upup_pos = upup_sl[0][0]
            model_l._upup_neg = upup_sl[1][0]
        # Loud gate: no array of GLOBAL cell/edge size may reach the local step
        # (a closure constant the localisation missed would broadcast wrongly).
        if n_cells != max_lc or n_edges != max_le:
            for name, tree in (("mesh", model_l.mesh), ("z_coord", model_l.z_coord)):
                for leaf in jax.tree.leaves(tree):
                    if hasattr(leaf, "shape") and leaf.ndim and int(leaf.shape[0]) in (n_cells, n_edges):
                        raise RuntimeError(
                            f"MPAS ocean SPMD: global-sized array {tuple(leaf.shape)} in "
                            f"model.{name} reached the local step (not localised)")
        mk = model_state_template
        local_state = MPASOceanState(
            u=mk.u.replace(data=uloc),
            T=mk.T.replace(data=T_l), S=mk.S.replace(data=S_l),
            eta=mk.eta.replace(data=eta_l), w=mk.w.replace(data=w_l),
            H_bathy=mk.H_bathy.replace(data=hb_l),
            land_mask=mk.land_mask.replace(data=lm_l),
            rho_ref_z=(None if mk.rho_ref_z is None else mk.rho_ref_z.replace(data=rho_ref)),
            tke=(None if tke_l is None else mk.tke.replace(data=tke_l)),
        )
        # Re-assemble the forcing pytrees from their classified leaves.
        it_cell, it_rep, it_static = iter(rest), iter(rep_leaves), iter(static_cells)
        leaves = []
        for k in ckinds:
            if k == "cell":
                leaves.append(next(it_cell))
            elif k == "cell_static":
                leaves.append(next(it_static)[0])
            else:
                leaves.append(next(it_rep))
        fw, sf, sp = jax.tree.unflatten(forcing_tree, leaves)
        hr = _make_refresh(halo_sl, vhalo_sl)
        new = model_l._step_impl(local_state, dt, freshwater=fw,
                                 surface_forcing=sf, sponge=sp, halo_refresh=hr)
        out = (new.u.data[:edges_per], new.T.data[:cells_per], new.S.data[:cells_per],
               new.eta.data[:cells_per], new.w.data[:cells_per])
        if tke_l is not None:
            out = out + (new.tke.data[:cells_per],)
        return out

    model_state_template = None
    _cache: dict = {}

    # The sharded per-device operands (local meshes, halo schedules, per-cell
    # z_coord statics, upup stencil).  A jitted CALLER (the JRA55 block scan)
    # must receive them as ARGUMENTS under route-B multicontroller — a
    # process-spanning jax.Array cannot be a closure constant — so they are
    # exposed as ``spmd_step.aux`` (the lat-lon lane's convention; the loop
    # passes ``aux=spmd_step.aux`` into the block function).
    default_aux = (layout.stacked_meshes, layout.halo_args, layout.vhalo_args,
                   layout.upup or (), z_static)

    def spmd_step(state: MPASOceanState, dt, aux=None, *, freshwater=None,
                  surface_forcing=None, sponge=None):
        nonlocal model_state_template
        if aux is None:
            aux = default_aux
        aux_meshes, aux_halo, aux_vhalo, aux_upup, aux_z = aux
        model_state_template = state
        has_tke = state.tke is not None
        cells = [state.T.data, state.S.data, state.eta.data, state.w.data,
                 state.H_bathy.data, state.land_mask.data]
        if has_tke:
            cells.append(state.tke.data)
        n_state_cells = len(cells)
        leaves, forcing_tree = jax.tree.flatten((freshwater, surface_forcing, sponge))
        kinds = _classify(leaves)
        for x, k in zip(leaves, kinds):
            if k == "edge":
                raise NotImplementedError(
                    "MPAS ocean SPMD: per-edge forcing leaves (sponge u_ref) "
                    "are not supported on this lane")
        cell_forcing = [x for x, k in zip(leaves, kinds) if k == "cell"]
        rep_leaves = tuple(jnp.asarray(x) for x, k in zip(leaves, kinds) if k == "rep")
        static_cells = tuple(_localize_static(x) for x, k in zip(leaves, kinds)
                             if k == "cell_static")
        dt_ = jnp.result_type(*[c.dtype for c in cells])
        cpack, cmeta = _pack(cells + [jnp.asarray(x) for x in cell_forcing], dt_)
        # shard_map wants array args: an absent rho_ref_z rides as an empty array.
        rho = (jnp.zeros((0,), dtype=dt_) if state.rho_ref_z is None
               else state.rho_ref_z.data)
        key = (tuple(kinds), n_state_cells, tuple(m[0] for m in cmeta),
               len(static_cells), state.rho_ref_z is None, float(dt),
               forcing_tree)
        fn = _cache.get(key)
        if fn is None:
            n_rep = len(rep_leaves)
            n_stat = len(static_cells)
            n_z = len(aux_z)

            def _body(u_o, cpack_o, rho_ref, *rest):
                rep = rest[:n_rep]
                stat = rest[n_rep:n_rep + n_stat]
                z_sl = rest[n_rep + n_stat:n_rep + n_stat + n_z]
                mesh_sl, halo_sl, vhalo_sl, upup_sl = rest[n_rep + n_stat + n_z:]
                return _local_step(u_o, cpack_o, rho_ref, rep, stat, z_sl, dt,
                                   cmeta, kinds, forcing_tree, n_state_cells,
                                   mesh_sl, halo_sl, vhalo_sl,
                                   None if len(upup_sl) == 0 else upup_sl)

            n_out = 5 + (1 if has_tke else 0)
            fn = shard_map(
                _body, mesh=jax_mesh,
                in_specs=(P(SPMD_AXIS), P(SPMD_AXIS), P(),
                          *([P()] * n_rep), *([P(SPMD_AXIS)] * n_stat),
                          *([P(SPMD_AXIS)] * n_z),
                          jax.tree.map(lambda _: P(SPMD_AXIS), layout.stacked_meshes),
                          jax.tree.map(lambda _: P(SPMD_AXIS), layout.halo_args),
                          jax.tree.map(lambda _: P(SPMD_AXIS), layout.vhalo_args),
                          jax.tree.map(lambda _: P(SPMD_AXIS), layout.upup or ())),
                out_specs=tuple([P(SPMD_AXIS)] * n_out),
                check_vma=False,
            )
            # jit so an eager caller (the restoring lane's per-step loop)
            # runs ONE compiled program per step, not op-by-op dispatch.
            fn = jax.jit(fn)
            _cache[key] = fn
        out = fn(state.u.data, cpack, rho, *rep_leaves, *static_cells, *aux_z,
                 aux_meshes, aux_halo, aux_vhalo, aux_upup)
        u_n, T_n, S_n, eta_n, w_n = out[:5]
        new = state._replace(
            u=state.u.replace(data=u_n), T=state.T.replace(data=T_n),
            S=state.S.replace(data=S_n), eta=state.eta.replace(data=eta_n),
            w=state.w.replace(data=w_n))
        if has_tke:
            new = new._replace(tke=state.tke.replace(data=out[5]))
        return new

    spmd_step.layout = layout
    spmd_step.aux = default_aux
    return spmd_step
