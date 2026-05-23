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
Every ``sendrecv`` call goes through ``_get_sendrecv_vjp`` from
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

Multi-rank coverage (staged-not-validated yet)
----------------------------------------------
The multi-rank halo exchange path (``layout.n_ranks > 1``) IS
implemented but is NOT yet exercised by an automated test because
this dev environment lacks ``libmpi.dylib`` / OpenMPI; the
``tests/distributed/`` conftest imports ``mpi4jax`` eagerly and
fails at collection time. A multi-rank smoke test
(``tests/distributed/test_plane_pencil_mpi.py``) lands in the
follow-up PR that ships under the OpenMPI-installed CI job, and
will verify (a) the padded local field equals the slice of a
globally wrapped reference, (b) AD through ``exchange_halo_plane_yxz``
returns finite gradients via ``_sendrecv_vjp``, (c) uneven
decompositions + halo widths > 1.

Public-API maturity
-------------------
* :func:`make_plane_pencil_layout` — stable.
* :func:`exchange_halo_plane_yxz` — single-rank stable; multi-rank
  implemented but pending CI validation under ``mpirun``.
* :func:`scatter_plane_field` — single-rank stable (pure slice).
* :func:`gather_plane_field` — single-rank identity; multi-rank
  raises ``NotImplementedError`` (intentional, lands with the
  distributed test suite).
* :func:`make_plane_pencil_grid` — stable; recomputes ``f_y``,
  ``Lx``, ``Ly``, ``total_area`` against the global plane so
  beta-plane Coriolis is correct on every rank (Codex PR5 iter-1
  finding M1).
"""

from __future__ import annotations

from typing import NamedTuple

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
        in :mod:`legoesm.atmosphere.dynamics.plane_operators`.
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
    :func:`legoesm.atmosphere.dynamics.plane_operators.pad_halo_plane_4d`
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

    # Multi-rank path is gated until the OpenMPI-installed CI job
    # lands the distributed validation test
    # (``tests/distributed/test_plane_pencil_mpi.py``). Codex PR5
    # iter-2 finding M2: MPI argument order / tags / corner
    # exchange / ``_sendrecv_vjp`` backward correctness are not
    # safe to commit implemented-but-untested — they would
    # deadlock or silently exchange wrong halos on the first
    # multi-rank run. The reference algorithm (E/W first, then
    # N/S using already-filled E/W halos so corners arrive via
    # two-axis composition) is documented in
    # :mod:`legoesm.parallel.latlon_mpi`'s 1D-band variant and
    # will be ported here together with the test.
    raise NotImplementedError(
        f"Multi-rank plane halo exchange (n_ranks={layout.n_ranks}) "
        "is staged-not-implemented in PR5. The implementation lands "
        "together with the distributed validation test in the "
        "follow-up PR that has OpenMPI + mpi4jax runtime available; "
        "single-rank ``layout.n_ranks == 1`` is fully supported via "
        "the JAX fast-path above."
    )


# --------------------------------------------------------------------- #
# Scatter / gather helpers                                              #
# --------------------------------------------------------------------- #


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
) -> jax.Array:
    """Gather rank-local sub-blocks back to a global field.

    **Staged-not-implemented for multi-rank.** Single-rank
    short-circuits to identity; ``n_ranks > 1`` raises
    ``NotImplementedError``. The multi-rank assembly lands together
    with the distributed test suite in the follow-up PR so the
    gather logic is validated against a multi-rank ``mpirun`` run
    rather than committed untested.

    On rank 0 the multi-rank path will return the assembled
    ``(ny_global, nx_global, ...)`` array; on other ranks the local
    sub-block unchanged (caller responsible for ignoring the
    non-root return).
    """
    if layout.n_ranks == 1:
        return local_field_yxz
    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Multi-rank gather requires mpi4jax + mpi4py."
        ) from exc
    # mpi4jax.allgather concatenates along axis 0; reshape + transpose
    # to reassemble the 2D pencil.
    raise NotImplementedError(
        "Multi-rank gather lands with the dedicated tests/distributed/ "
        "test suite (PR5 follow-up); single-rank gather is identity."
    )


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
