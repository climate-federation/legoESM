# Distributed Architecture: MPI, SPMD, and the Path Forward

Resolves discussion in issue #359. Captures the design principles and concrete,
enforceable rules that prevent the class of bug exposed by #356 (and fixed in
#357 / #358), where MPI code paths diverged from the serial/SPMD paths and
produced `MPI_ERR_TRUNCATE` or wrong boundary values.

## Current architecture

legoESM supports three parallel backends, dispatched at runtime via the
mutable singleton in `packages/core/legoesm/grids/halo.py`
(`set_halo_backend` / `get_halo_backend`):

| Backend | Scope | Mechanism | AD-safe? |
|---|---|---|---|
| `local` | Single device | Array indexing / `jnp.pad` | Yes (trivially) |
| `spmd` | Single-node multi-GPU | `shard_map` + `ppermute` / `psum` | Yes (via `shard_map` VJP) |
| `mpi` | Multi-node | `mpi4jax.sendrecv` / `allreduce` | Partial — `SUM` only; `sendrecv` via `@jax.custom_vjp` |

The lat-lon band decomposition (`LatLonBandLayout`) splits the latitude axis
into contiguous bands, one per rank. Longitude stays periodic and is never
partitioned. Cubed-sphere / MPAS use their own topologies.

## Root cause of the #356 bug class

Both #356 bugs (`MPI_ERR_TRUNCATE`; NaN blow-up at `np >= 4`) had the same
root cause: **rank-dependent code branching** — different ranks executing a
different sequence of JAX/MPI operations. Under the tripolar fold, the
pre-#357 code set `fold.is_active = False` on non-northernmost ranks, so
`is_tripolar()` disagreed across ranks and the `pad_ns_*` operators took
different code paths → mismatched MPI call counts → truncate; and v-face
quantities were formed from rank-local data → missing neighbour rows at
partition cuts → wrong values.

## Design principles

### 1. Same code on all ranks; data-dependent branching only

Every traced JAX function must execute the **same sequence of JAX/MPI
operations on all ranks**. The only permitted branching is on *data values*
(`jnp.where`, `lax.cond`) — never on rank identity, and never on a grid
attribute (`fold.is_active`) in a way that changes the **count** of MPI
collectives.

The canonical realization (post-#357): make `is_tripolar()` a pure grid-type
query that returns the same value on every rank of a tripolar run, and call
the halo exchange (`pad_ns_zero`) **unconditionally** so the MPI call count is
identical across ranks. Apply the fold seam as a *local post-processing
overwrite* guarded by `_fold_is_local(grid)` (true on exactly one rank —
the seam owner), which performs **no** collective.

```python
def _fold_is_local(grid) -> bool:
    fold = getattr(grid, "fold", None)
    return fold is not None and bool(fold.is_active) and fold.fold_j >= 0
```

### 2. Pre-pad-then-operate, not operate-then-pad

When an operator needs neighbour data (gradient, interpolation, divergence,
min-rule face thickness), **pad the input CELL field first** via the
backend-dispatched halo (`pad_ns_zero`, `pad_halo_latlon`), then run the
compact stencil on the padded field, then restore physical boundary
conditions with the backend-aware `zero_polar_lat_ends`, then overwrite the
fold seam on the owning rank.

Anti-pattern (misses the neighbour row at a partition cut):

```python
f_v_interior = 0.5 * (f[:-1] + f[1:])   # rank-local only — WRONG at cuts
f_v = pad_ns_zero(f_v_interior)         # pads the already-formed face array
```

Correct pattern:

```python
f_padded = pad_ns_zero(f)               # MPI halo fills the neighbour cell row
f_v = 0.5 * (f_padded[:-1] + f_padded[1:])
f_v = zero_polar_lat_ends(f_v)          # zeros only the physical poles
if _fold_is_local(grid):
    f_v = jnp.concatenate([f_v[:-1], fold_partner_row(f, grid)], axis=0)
```

Why: halo-exchanging an already-formed *face* array gives the neighbour's
top *interior* face (off by one cell), not the cross-partition face — that
face is owned by neither rank's interior array and must be reconstructed from
both adjacent cells.

### 3. `is_tripolar()` is a grid-type query, not a rank-dependent gate

`is_tripolar(grid)` must return the same value on all ranks. Use it for
metric selection (2D tripolar metrics vs 1D regular). Use `_fold_is_local(grid)`
for anything that touches the fold seam itself.

### 4. AD-safety of collectives

`allreduce(SUM)` and the custom-VJP `sendrecv` are differentiable. `MAX` /
`MIN` / `allgather` / `bcast` are **diagnostics only** — keep them out of
loss-bearing paths. Floor any denominator that a metric can legitimately
zero (e.g. a tripole built with `min_dx_m=0.0` has an exact-zero south
`dy_v`): divide by `jnp.maximum(metric, 1e-30)` before the polar overwrite,
or a `nonzero / 0` intermediate will poison reverse-mode AD and trip
`jax_debug_nans` even when the primal row is later set to zero (#358).

## Enforcement

`tests/distributed/test_mpi_architecture_invariants.py` is a source-audit
guard against regressions of the #356 class. It asserts that the lat-lon
C-grid dynamics modules use `fold.is_active` only inside the `is_tripolar` /
`_fold_is_local` definitions (or comments/docstrings) — every *functional*
fold gate must go through `_fold_is_local` (i.e. carry the `fold_j >= 0`
locality check). Runtime correctness is covered by the rank-vs-serial value
tests in `tests/distributed/test_latlon_mpi_tripole.py`
(`TestIssue356GradientYPartitionCut`, `TestPgfYMeridionalFoldMPI`,
`TestCurlVertexFoldMPI`, `TestMinCellToVfaceFoldMPI`) at `np=2` and `np=4`.

## Path forward

- **Keep `mpi4jax` for multi-node** until an FFI-based release lands (mpi4jax
  0.8.x's custom-call API is deprecated in JAX 0.9 and removed in 0.10; pin
  JAX `< 0.10` for MPI workloads — see `runtime/distributed.py` version guard).
- **Prefer `shard_map` (SPMD)** for single-node multi-GPU; its VJP is fully
  AD-safe and avoids the rank-branching foot-gun by construction.
- **New v-face / vertex operators** on the lat-lon C-grid must follow
  principles 1–2 and ship a rank-vs-serial MPI value test alongside the
  serial unit test.
