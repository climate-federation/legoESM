"""MPI 2D periodic pencil decomposition for the doubly-periodic plane.

PR5 of the CRM rollout. Layout + halo exchange + scatter/gather for
``PlaneGrid`` + ``PlaneNonHydrostaticState`` across an MPI mesh.

Differences from :mod:`legoesm.parallel.latlon_mpi`:

* No poles → no folding, no vector sign flip; doubly periodic boundary
  in both axes.
* 2D pencil decomposition (both ``ny`` and ``nx`` split) instead of
  the lat-lon 1D band.
* Halo exchange uses 4 face neighbours (N/S/E/W) plus 4 corners.

AD safety
---------
Every ``sendrecv`` call goes through ``get_sendrecv_vjp`` from
:mod:`legoesm.parallel.halo_exchange` (existing ``@jax.custom_vjp``
wrapper that swaps source / dest in the backward pass per
``CLAUDE.md`` "MPI AD compat" rule). ``allreduce(SUM)`` reductions
inherit AD via ``global_sum_mpi`` (same module).

Single-rank fallback
--------------------
``n_ranks == 1`` short-circuits to a pure-JAX path that uses
``jnp.roll`` for periodic halos and skips every MPI call. This lets
the plane MPI code run inside CI without an ``mpirun`` launcher and
keeps single-rank vs multi-rank correctness easy to verify on a
laptop. The single-rank path is exercised by
``tests/unit/test_plane_mpi_pencil.py``.

Multi-rank coverage
-------------------
``layout.n_ranks > 1`` runs a two-stage sendrecv: N/S along the
``y`` axis first, then E/W of the *already-NS-padded* array so the
four corner halos arrive via two-axis composition (mirrors the
cubed-sphere ``_pad_halo_mpi`` pattern). All sendrecv calls flow
through ``get_sendrecv_vjp`` so :func:`jax.grad` works through
the exchange. Validated by ``tests/distributed/test_plane_pencil_mpi.py``
under the OpenMPI ``mpi-distributed.yml`` CI job (np = 2, 4).

Public-API maturity
-------------------
* :func:`make_plane_pencil_layout` — stable.
* :func:`exchange_halo_plane_yxz` — single-rank + multi-rank stable.
* :func:`scatter_plane_field` — single-rank stable (pure slice).
* :func:`gather_plane_field` — single-rank identity; multi-rank
  uses ``comm.gather`` to rank 0 with 2D reassembly via layout.
* :func:`make_plane_pencil_grid` — stable; recomputes ``f_y``,
  ``Lx``, ``Ly``, ``total_area`` against the global plane so
  beta-plane Coriolis is correct on every rank (Codex PR5 iter-1
  finding M1).
"""

from __future__ import annotations

from typing import NamedTuple

import math

import jax
import jax.numpy as jnp

from legoesm.grids.plane import PlaneGrid, create_plane_grid


class PlanePencilLayout(NamedTuple):
    """2D periodic pencil decomposition metadata for the plane grid.

    Decomposes the global ``(ny, nx)`` plane across an
    ``n_ranks_y × n_ranks_x`` MPI mesh. Each rank owns a contiguous
    sub-block plus 1-cell periodic halos in all 4 directions (and
    via two-axis composition, the 4 corners).

    Attributes
    ----------
    rank : int
        Flat MPI rank ``ry * n_ranks_x + rx``.
    n_ranks : int
        Total number of MPI processes (must equal
        ``n_ranks_y * n_ranks_x``).
    n_ranks_y, n_ranks_x : int
        Cartesian mesh dimensions.
    ry, rx : int
        2D coordinates of this rank in the mesh.
    ny_global, nx_global : int
        Global horizontal cell counts.
    ny_local, nx_local : int
        Local owned cell counts (interior, no halos).
    iy_start, iy_end, ix_start, ix_end : int
        Global indices of the owned sub-block.
    north_rank, south_rank, east_rank, west_rank : int
        Flat ranks of the four periodic neighbours.
    halo : int
        Halo width in cells (default 1). Matches the operator radius
        in :mod:`legoesm.atmosphere.dynamics.les.plane_operators`.
    """
    rank: int
    n_ranks: int
    n_ranks_y: int
    n_ranks_x: int
    ry: int
    rx: int
    ny_global: int
    nx_global: int
    ny_local: int
    nx_local: int
    iy_start: int
    iy_end: int
    ix_start: int
    ix_end: int
    north_rank: int
    south_rank: int
    east_rank: int
    west_rank: int
    halo: int


def make_plane_pencil_layout(
    rank: int,
    n_ranks: int,
    n_ranks_y: int,
    n_ranks_x: int,
    ny_global: int,
    nx_global: int,
    halo: int = 1,
) -> PlanePencilLayout:
    """Construct a 2D periodic pencil layout.

    Splits ``(ny_global, nx_global)`` as evenly as possible across
    the ``n_ranks_y × n_ranks_x`` mesh; remainder cells distributed
    to the first few ranks per axis.

    Parameters
    ----------
    rank : int
        Flat MPI rank (must equal ``ry * n_ranks_x + rx``).
    n_ranks : int
        Total ranks (validated against ``n_ranks_y * n_ranks_x``).
    n_ranks_y, n_ranks_x : int
        Mesh dimensions.
    ny_global, nx_global : int
        Global horizontal counts.
    halo : int
        Halo width (default 1). Must be < min(ny_local, nx_local).

    Returns
    -------
    PlanePencilLayout
    """
    if n_ranks != n_ranks_y * n_ranks_x:
        raise ValueError(
            f"n_ranks={n_ranks} must equal n_ranks_y * n_ranks_x = "
            f"{n_ranks_y * n_ranks_x}."
        )
    if not 0 <= rank < n_ranks:
        raise ValueError(f"rank={rank} out of range [0, {n_ranks}).")
    if n_ranks_y > ny_global:
        raise ValueError(
            f"n_ranks_y={n_ranks_y} exceeds ny_global={ny_global}; "
            "over-decomposition would produce empty rank sub-blocks."
        )
    if n_ranks_x > nx_global:
        raise ValueError(
            f"n_ranks_x={n_ranks_x} exceeds nx_global={nx_global}; "
            "over-decomposition would produce empty rank sub-blocks."
        )
    if halo <= 0:
        raise ValueError(
            f"halo={halo} must be > 0; pass 1 for the standard PR1 "
            "operator stencil radius."
        )
    ry, rx = divmod(rank, n_ranks_x)

    def _slice(n_global, n_ranks_axis, idx):
        base = n_global // n_ranks_axis
        rem = n_global % n_ranks_axis
        if idx < rem:
            n_local = base + 1
            start = idx * (base + 1)
        else:
            n_local = base
            start = rem * (base + 1) + (idx - rem) * base
        return n_local, start, start + n_local

    ny_local, iy_start, iy_end = _slice(ny_global, n_ranks_y, ry)
    nx_local, ix_start, ix_end = _slice(nx_global, n_ranks_x, rx)
    if halo >= min(ny_local, nx_local):
        raise ValueError(
            f"halo={halo} must be < min(ny_local={ny_local}, "
            f"nx_local={nx_local}); decomposition too fine for this halo."
        )

    # Periodic neighbours via Cartesian wrap.
    def _wrap(idx, n):
        return idx % n

    north_ry, south_ry = _wrap(ry + 1, n_ranks_y), _wrap(ry - 1, n_ranks_y)
    east_rx, west_rx = _wrap(rx + 1, n_ranks_x), _wrap(rx - 1, n_ranks_x)

    return PlanePencilLayout(
        rank=rank, n_ranks=n_ranks,
        n_ranks_y=n_ranks_y, n_ranks_x=n_ranks_x,
        ry=ry, rx=rx,
        ny_global=ny_global, nx_global=nx_global,
        ny_local=ny_local, nx_local=nx_local,
        iy_start=iy_start, iy_end=iy_end,
        ix_start=ix_start, ix_end=ix_end,
        north_rank=north_ry * n_ranks_x + rx,
        south_rank=south_ry * n_ranks_x + rx,
        east_rank=ry * n_ranks_x + east_rx,
        west_rank=ry * n_ranks_x + west_rx,
        halo=halo,
    )


def exchange_halo_plane_yxz(
    field_yxz: jax.Array,
    layout: PlanePencilLayout,
) -> jax.Array:
    """Exchange periodic halos for a ``(ny_local, nx_local, ...)`` field.

    **Axis convention**: the FIRST two axes are ``(ny_local,
    nx_local)`` — matches the
    :class:`legoesm.core.state.PlaneNonHydrostaticState` layout used
    everywhere in the plane dycore. This differs from
    :func:`legoesm.atmosphere.dynamics.les.plane_operators.pad_halo_plane_4d`
    (PR1) which expects ``(batch, nlev, ny, nx)`` with the
    horizontal axes LAST; PR1's helper is the single-rank,
    last-two-axis convention used inside cubed-sphere-shaped
    operator code, while this PR5 helper is for the plane state
    layout. Mixing the two will give wrong padding — the caller is
    responsible for matching the function name to its layout.

    Output shape ``(ny_local + 2h, nx_local + 2h, ...)``. The four
    interior edges are filled from the matching neighbour rank via
    ``_sendrecv_vjp`` (AD-safe); the four corners come from
    composing the two-axis wrap. Single-rank ``n_ranks == 1`` path
    uses ``jnp.pad(mode='wrap')`` and skips MPI entirely.

    Parameters
    ----------
    field_yxz : jax.Array
        Local interior data, shape ``(ny_local, nx_local, ...)``.
        Must satisfy ``field_yxz.shape[:2] == (layout.ny_local,
        layout.nx_local)``.
    layout : PlanePencilLayout

    Returns
    -------
    jax.Array
        Padded field with halos filled.
    """
    if field_yxz.ndim < 2:
        raise ValueError(
            f"exchange_halo_plane_yxz expects ndim >= 2 with leading "
            f"(ny_local, nx_local); got shape {field_yxz.shape}"
        )
    if field_yxz.shape[:2] != (layout.ny_local, layout.nx_local):
        raise ValueError(
            f"field_yxz leading axes {field_yxz.shape[:2]} must match "
            f"(layout.ny_local={layout.ny_local}, "
            f"layout.nx_local={layout.nx_local})."
        )
    h = layout.halo
    pad_widths = [(h, h), (h, h)] + [(0, 0)] * (field_yxz.ndim - 2)

    # Single-rank fast path — pure JAX, matches the PR1
    # ``pad_halo_plane_4d`` convention.
    if layout.n_ranks == 1:
        return jnp.pad(field_yxz, pad_widths, mode="wrap")

    # Multi-rank path. Each axis (y, x) handled INDEPENDENTLY:
    #   * n_ranks_axis == 1 → local periodic wrap on that axis (no
    #     MPI). Codex review 2026-05-24: previously the multi-rank
    #     path always issued sendrecv even when the periodic
    #     neighbour was the current rank. That triggered two
    #     self-sendrecv on the same tag base with identical
    #     `(source, dest)` for both directions — MPI matched the
    #     messages arbitrarily and filled the east halo with the
    #     east boundary (instead of the west) or vice versa.
    #   * n_ranks_axis >  1 → mpi4jax sendrecv with the periodic
    #     neighbour rank.
    # NS stage always runs first; the EW stage operates on the
    # NS-padded array so the four corners arrive via two-axis
    # composition regardless of which axis used MPI vs local wrap.
    h = layout.halo

    if layout.n_ranks_y > 1 or layout.n_ranks_x > 1:
        try:
            import mpi4jax
            from mpi4py import MPI
        except ImportError as exc:
            raise ImportError(
                "Multi-rank plane halo exchange requires mpi4jax + "
                "mpi4py (install with `pip install -e \".[mpi]\"` and "
                "have OpenMPI available)."
            ) from exc
        from legoesm.parallel.halo_exchange import get_sendrecv_vjp
        comm = MPI.COMM_WORLD
        sendrecv = get_sendrecv_vjp(mpi4jax)
    else:
        sendrecv = None  # never used; single-rank shortcut above

    trailing = field_yxz.shape[2:]
    # iter-57 Codex LOW#2 doc: MPI tag namespace. N/S exchanges use
    # tag 1000+, E/W use 2000+; mpi4jax adds a per-field index so the
    # two ranges never collide as long as no individual call site
    # uses >999 distinct tags. Two-axis pencil layout only emits 2
    # tag bases total per packed halo round, well below the limit.
    _TAG_NS = 1_000
    _TAG_EW = 2_000

    # ----- Stage 1: N/S along the y axis -----
    if layout.n_ranks_y == 1:
        # Local periodic wrap on y; MPI would self-deadlock or
        # silently misfill halos for the single-rank-on-y case.
        ns_padded = jnp.pad(
            field_yxz,
            [(h, h), (0, 0)] + [(0, 0)] * (field_yxz.ndim - 2),
            mode="wrap",
        )
    else:
        flat_shape_ns = (h * layout.nx_local, *trailing)
        flat_size_ns = math.prod(flat_shape_ns)  # static shape -> jit-safe Python int
        send_to_north = field_yxz[-h:].reshape(flat_size_ns)
        send_to_south = field_yxz[:h].reshape(flat_size_ns)
        recv_template = jnp.zeros_like(send_to_north)
        # Directional ring-shift exchange — correct AND deadlock-free even
        # when an axis has exactly 2 ranks (north_rank == south_rank).  The
        # old ``send-to-X & recv-from-X`` pattern paired both sendrecvs to the
        # same neighbour under a SHARED tag, so at n_ranks==2 MPI matched the
        # WRONG message and delivered the neighbour's opposite edge (silent
        # halo corruption — du_dt off by ~0.5 vs single-process).
        # Fill SOUTH halo: send my NORTH edge -> north_rank, recv <- south_rank
        # (its north edge).  Fill NORTH halo: send my SOUTH edge -> south_rank,
        # recv <- north_rank.  One tag per shift direction (a consistent ring
        # shift, so send/recv pair within the SAME call across the ring).
        from_south = sendrecv(
            send_to_north, recv_template,
            layout.south_rank, layout.north_rank,
            _TAG_NS, _TAG_NS, comm,
        )
        from_north = sendrecv(
            send_to_south, recv_template,
            layout.north_rank, layout.south_rank,
            _TAG_NS + 1, _TAG_NS + 1, comm,
        )
        south_halo = from_south.reshape((h, layout.nx_local, *trailing))
        north_halo = from_north.reshape((h, layout.nx_local, *trailing))
        ns_padded = jnp.concatenate(
            [south_halo, field_yxz, north_halo], axis=0,
        )

    # ----- Stage 2: E/W of the NS-padded array -----
    # ns_padded already carries the NS halo rows, so E/W slabs of
    # ns_padded contain the four corner cells of the periodic halo.
    if layout.n_ranks_x == 1:
        padded = jnp.pad(
            ns_padded,
            [(0, 0), (h, h)] + [(0, 0)] * (ns_padded.ndim - 2),
            mode="wrap",
        )
    else:
        ny_padded = ns_padded.shape[0]
        flat_shape_ew = (ny_padded * h, *trailing)
        flat_size_ew = math.prod(flat_shape_ew)  # static shape -> jit-safe Python int
        send_to_east = ns_padded[:, -h:].reshape(flat_size_ew)
        send_to_west = ns_padded[:, :h].reshape(flat_size_ew)
        recv_template_ew = jnp.zeros_like(send_to_east)
        # Same directional ring-shift as the N/S stage (correct + deadlock-free
        # at n_ranks_x == 2 where west_rank == east_rank).  Fill WEST halo:
        # send my EAST edge -> east_rank, recv <- west_rank (its east edge).
        # Fill EAST halo: send my WEST edge -> west_rank, recv <- east_rank.
        from_west = sendrecv(
            send_to_east, recv_template_ew,
            layout.west_rank, layout.east_rank,
            _TAG_EW, _TAG_EW, comm,
        )
        from_east = sendrecv(
            send_to_west, recv_template_ew,
            layout.east_rank, layout.west_rank,
            _TAG_EW + 1, _TAG_EW + 1, comm,
        )
        west_halo = from_west.reshape((ny_padded, h, *trailing))
        east_halo = from_east.reshape((ny_padded, h, *trailing))
        padded = jnp.concatenate(
            [west_halo, ns_padded, east_halo], axis=1,
        )

    return padded


# --------------------------------------------------------------------- #
# Scatter / gather helpers                                              #
# --------------------------------------------------------------------- #


def packed_exchange_halo_plane_yxz(
    *fields: jax.Array,
    layout: PlanePencilLayout,
) -> list[jax.Array]:
    """Exchange halos for multiple ``(ny_local, nx_local, ...)`` fields
    in a single MPI round.

    Stacks fields along a NEW trailing axis, performs ONE halo
    exchange (with proportionally larger MPI messages), then splits
    + reshapes. Reduces MPI message count from ``len(fields)``
    exchanges to 1 — critical on macOS shared-memory MPI where the
    per-call mpi4jax dispatch overhead dominates many-small-message
    workloads (~2-5 ms per sendrecv vs ~50 μs on cluster IB).

    All fields must share the same ``(ny_local, nx_local)`` prefix +
    same trailing-axis dimensions (typically same ``nlev``). If
    trailing-axis shapes differ per field, use the unpacked
    :func:`exchange_halo_plane_yxz` instead.

    Parameters
    ----------
    *fields : jax.Array
        Local interior arrays, each ``(ny_local, nx_local, ...)``
        with the same trailing axes.
    layout : PlanePencilLayout

    Returns
    -------
    list[jax.Array]
        Padded arrays, each shaped like the original input plus the
        added halo (``ny_local + 2h, nx_local + 2h, ...``).
    """
    if not fields:
        return []
    ref = fields[0]
    for f in fields[1:]:
        if f.shape != ref.shape:
            raise ValueError(
                f"packed_exchange_halo_plane_yxz requires all fields "
                f"share the same shape; got {ref.shape} vs {f.shape}."
            )
    # Stack along NEW trailing axis → single (ny, nx, ..., n_fields).
    stacked = jnp.stack(fields, axis=-1)
    padded = exchange_halo_plane_yxz(stacked, layout)
    # Split back along the trailing axis.
    return [padded[..., i] for i in range(len(fields))]


def scatter_plane_field(
    global_field_yxz: jax.Array,
    layout: PlanePencilLayout,
) -> jax.Array:
    """Extract the rank's owned sub-block from a global field.

    Parameters
    ----------
    global_field_yxz : jax.Array
        Shape ``(ny_global, nx_global, ...)``.
    layout : PlanePencilLayout

    Returns
    -------
    jax.Array
        Shape ``(ny_local, nx_local, ...)``.
    """
    return global_field_yxz[
        layout.iy_start:layout.iy_end,
        layout.ix_start:layout.ix_end,
    ]


def gather_plane_field(
    local_field_yxz: jax.Array,
    layout: PlanePencilLayout,
) -> jax.Array | None:
    """Gather rank-local sub-blocks back to a global field on rank 0.

    Uses ``MPI.COMM_WORLD.gather`` (host-side); each rank also sends
    its ``(ry, rx, iy_start, ix_start, ny_local, nx_local)`` layout
    metadata so rank 0 can place each sub-block at its global
    position. The implementation works for uneven decompositions
    (remainder cells assigned to low ranks) because every rank's
    contribution carries its actual start/extent rather than assuming
    a uniform block size.

    Parameters
    ----------
    local_field_yxz : jax.Array
        Shape ``(ny_local, nx_local, ...)``.
    layout : PlanePencilLayout

    Returns
    -------
    jax.Array on rank 0 with shape ``(ny_global, nx_global, ...)``;
    ``None`` on other ranks. On single-rank ``n_ranks == 1`` returns
    the input unchanged on every rank.

    **Return contract (asymmetric):** non-root ranks receive
    ``None``; calling code must check ``layout.rank == 0`` (or
    ``result is not None``) before treating the return as an array.
    Codex review 2026-05-24 flagged this as easy-to-misuse; rename
    to ``gather_plane_field_to_root`` if the asymmetry needs to be
    surfaced more prominently in your call sites.

    Not differentiable: this is an I/O / diagnostic gather and is
    not intended to participate in gradient flow (mirrors
    ``gather_state_latlon`` semantics).
    """
    if layout.n_ranks == 1:
        return local_field_yxz
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(
            "Multi-rank gather requires mpi4py (install with "
            "`pip install -e \".[mpi]\"`)."
        ) from exc

    comm = MPI.COMM_WORLD
    # Force host-side numpy: comm.gather is a CPU collective; if we
    # ship a device array, mpi4py copies via DLPack which is slower
    # and ties up device memory.
    import numpy as np
    local_np = np.asarray(local_field_yxz)
    payload = (
        layout.iy_start, layout.iy_end,
        layout.ix_start, layout.ix_end,
        local_np,
    )
    gathered = comm.gather(payload, root=0)
    if layout.rank != 0:
        return None

    trailing = local_field_yxz.shape[2:]
    global_shape = (layout.ny_global, layout.nx_global, *trailing)
    out = np.empty(global_shape, dtype=local_np.dtype)
    for (iy0, iy1, ix0, ix1, sub) in gathered:
        out[iy0:iy1, ix0:ix1] = sub
    return jnp.asarray(out)


def make_plane_pencil_grid(
    layout: PlanePencilLayout,
    dx: float,
    dy: float,
    nlev: int,
    dtype=None,
    coriolis_mode: str = "none",
    f0: float = 0.0,
    beta: float = 0.0,
    lat0: float = 0.0,
    lon0: float = 0.0,
) -> PlaneGrid:
    """Build a ``PlaneGrid`` covering the rank-local sub-domain.

    The returned grid carries ``ny=layout.ny_local``,
    ``nx=layout.nx_local`` (interior only; halo cells are added by
    :func:`exchange_halo_plane_4d`). Coordinates ``xc``, ``yc``,
    ``Lx``, ``Ly``, ``total_area``, and ``f_y`` are computed against
    the **global** plane, not the local sub-block:

    * ``xc``, ``yc`` are offset to global cell-centre positions so
      diagnostics + radiation latitude reference the right place.
    * ``Lx = nx_global * dx``, ``Ly = ny_global * dy`` so any
      length-scale division (e.g. RCEMIP domain area) matches the
      single-rank reference.
    * ``f_y`` uses the global ``y`` coordinate so beta-plane
      ``f(y) = f0 + beta * (y - Ly/2)`` honours the global meridional
      position of the rank's sub-domain (Codex iter-1 finding M1).
    * ``total_area`` carries the **global** total ``Lx * Ly``; if a
      caller needs the rank-local area it is ``area_T.sum()``.

    Parameters
    ----------
    layout : PlanePencilLayout
    dx, dy : float
        Cell spacings (must match those used to build ``layout``).
    nlev : int
    dtype : optional
    coriolis_mode : {'none', 'f_plane', 'beta_plane'}
    f0, beta : float
    lat0, lon0 : float
    """
    # Build local PlaneGrid for arrays (xc, yc still LOCAL here).
    local = create_plane_grid(
        nx=layout.nx_local, ny=layout.ny_local, nlev=nlev,
        dx=dx, dy=dy,
        # Coriolis must be recomputed below against the GLOBAL y;
        # build with no Coriolis here to avoid double work.
        coriolis_mode="none", f0=0.0, beta=0.0,
        lat0=lat0, lon0=lon0, dtype=dtype,
    )
    # Offset xc/yc to global coordinates.
    xc_global = local.xc + layout.ix_start * dx
    yc_global = local.yc + layout.iy_start * dy
    Lx_global = float(layout.nx_global * dx)
    Ly_global = float(layout.ny_global * dy)
    # Recompute f_y against the **global** y so beta-plane gradients
    # are physically correct across ranks.
    if coriolis_mode == "none":
        f_y_1d = jnp.zeros((layout.ny_local,), dtype=local.area_T.dtype)
    elif coriolis_mode == "f_plane":
        f_y_1d = jnp.full(
            (layout.ny_local,), float(f0), dtype=local.area_T.dtype,
        )
    elif coriolis_mode == "beta_plane":
        if beta == 0.0:
            raise ValueError(
                "coriolis_mode='beta_plane' requires nonzero beta."
            )
        f_y_1d = jnp.asarray(
            f0 + beta * (yc_global - 0.5 * Ly_global),
            dtype=local.area_T.dtype,
        )
    else:
        raise ValueError(
            f"coriolis_mode must be 'none', 'f_plane', or 'beta_plane'; "
            f"got {coriolis_mode!r}"
        )
    f_y_global = f_y_1d[:, None] * jnp.ones(
        (1, layout.nx_local), dtype=local.area_T.dtype,
    )
    total_area_global = jnp.asarray(
        Lx_global * Ly_global, dtype=local.area_T.dtype,
    )
    return local._replace(
        xc=xc_global,
        yc=yc_global,
        Lx=Lx_global,
        Ly=Ly_global,
        f_y=f_y_global,
        total_area=total_area_global,
        f0=float(f0),
        beta=float(beta),
        coriolis_mode_code={
            "none": 0, "f_plane": 1, "beta_plane": 2,
        }[coriolis_mode],
    )
