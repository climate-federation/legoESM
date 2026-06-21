# Distributed fixed-iteration PCG for the implicit barotropic solve

The ocean weak-scaling fix for `barotropic_solver="implicit_cn"`.

## Problem

`explicit_substep` does `n_substeps ∝ resolution` (the barotropic
gravity-wave CFL), so per-rank cost grows with the *global* problem size
→ weak scaling is impossible. `implicit_cn` removes the CFL with a
single-step Crank–Nicolson free surface + an elliptic Helmholtz solve,
but the stock `jax.scipy.sparse.linalg.cg` is **not** MPI-safe:

- its dot products are rank-local `jnp` reductions (wrong global α/β); and
- its `while_loop` trip count is residual-dependent, so different ranks
  iterate a different number of times → mismatched `sendrecv` call
  counts inside `A_op` → **deadlock**.

## Design (locked)

- **Solver.** A hand-rolled Jacobi-preconditioned CG run for **exactly**
  `M = barotropic_implicit_pcg_fixed_iters` iterations via a static
  `lax.fori_loop`. No residual-dependent `while_loop` → every rank runs
  the identical collective schedule → no deadlock, static JIT trace.
- **Reductions.** Standard CG has two sequentially-dependent inner
  products per iteration — `p·Ap` (for α) and `r·z` (for β, which needs
  the post-α `r`) — so the loop issues **two** batched
  `allreduce(SUM)`/iter. The residual monitor `r·r` is folded into the
  second reduction (free). Two reductions/iter is still a **fixed**
  count independent of global resolution (the weak-scaling property);
  one-reduction pipelined CG (Chronopoulos–Gear) is possible but changes
  the rounding and would break the tight single-rank equivalence pin, so
  it is not used. `allreduce(SUM)` is the only AD-safe collective. Under
  a single process the reduction is a no-op (local sum == global sum),
  so the *algorithm* is identical with/without MPI.
- **Dispatch.** `barotropic_solver` stays `"implicit_cn"`; the solver
  body dispatches *internally* on the static `is_distributed()`:
  single-rank keeps stock `jax.scipy` CG (numerics preserved — the
  single-rank/multi-rank match is *numerical equivalence ≤1e-10 f64*,
  not bit-identity), multi-rank runs the distributed fixed-`M` PCG. No
  new scheme name.
- **AD.** The distributed solve is **UNROLLED** and differentiated
  straight through: reverse-mode AD flows through the fixed-length
  `fori_loop`/`scan`, the halo `_sendrecv_vjp` (a `jax.custom_vjp` with a
  full backward rule), and the `allreduce(SUM)` dot products (full VJP).
  These are exactly the two collectives the repo MPI-AD doctrine blesses,
  so `jax.grad` works under MPI. The gradient is the derivative of "`M`
  PCG iterations", which equals the implicit-solve gradient `η = A⁻¹·rhs`
  to within the converged residual (M chosen so residual ≤ tol; verified
  to match a finite difference to ≤1e-6 in f64). The **single-rank** path
  keeps stock `jax.scipy.cg`, whose own internal `custom_linear_solve`
  gives the exact implicit-function adjoint (no MPI halo there). MPAS
  stays single-rank stock CG (see *MPAS scope*).

#### Why NOT `jax.lax.custom_linear_solve` (the original plan)

The original design wrapped the solve in `custom_linear_solve` (for the
implicit-function adjoint, avoiding storing the `M` primal iterations).
**That is incompatible with the MPI halo and was empirically verified to
crash** (np=2/4/8 fail even forward-only; np=1 fine): `custom_linear_solve`
forms the operator-parameter cotangent by **linear-transposing** `matvec`
(`A_op`), and under MPI `A_op` contains the halo `sendrecv` via
`_sendrecv_vjp` — a `custom_vjp` with **no transpose rule**
(`NotImplementedError: Transpose rule for 'custom_vjp_call' not
implemented`). Unrolling needs only the `custom_vjp` *backward* (which
exists), never its transpose, so it sidesteps the incompatibility.

The cost is reverse-mode memory ∝ `M` (scan stores loop intermediates) —
acceptable for the cheap 2D barotropic solve at `M≈60`; remat only if
production MPI AD shows memory pressure.

#### Helmholtz area-weighted self-adjointness (background)

`A = I − coeff·∇·(H·∇)` is built from the FV gradient and FV divergence,
discrete adjoints **only in the area-weighted inner product**
`⟨x,y⟩_W = Σ x·y·A`: in matrix form `A = I − coeff·W⁻¹·L` (`W=diag(area)`,
`L` the FV operator before the diagonal `1/area`; verified lat-lon +
MPAS), so `W·A` is symmetric and `Aᵀ = W·A·W⁻¹`. This is the reason a
*naive* `custom_linear_solve` Euclidean `transpose_solve == solve` would
have been wrong (it would need `y = area·PCG(A, c/area)`). The shipped
unrolled path doesn't form the transpose at all, so it needs no area
weighting — but the self-adjointness is what makes the operator
well-posed for CG. **Verified numerically**
(`scripts/tmp/_diag_helmholtz_adjoint.py`, since deleted): `W·A` is
symmetric to round-off (`rel ~1e-18`) for flat `H`, the zonal/meridional
sub-operators, AND production **min-rule cell-derived** face depths
(`min(H_left, H_right)`); it is *not* symmetric for *independent random*
face values (`rel ~1e-3`) because the FV adjoint identity needs the same
shared-face depth on both neighbours. The unit test therefore builds
`H_u/H_v` from a cell field via the min-rule.

## `M` selection and the residual contract

`M` is a config field (`barotropic_implicit_pcg_fixed_iters`, default
60), **not** a magic literal. A fixed `M` is a *hypothesis*, not a
guarantee — tripole-fold / coastal conditioning can require more. So the
solve **always runs exactly `M` iterations** and then computes the
**global** relative residual `sqrt(global(r·r)/global(rhs·rhs))` as a
returned **diagnostic** (the solver functions take `return_residual=True`
and append the scalar). The unit test asserts it ≤
`barotropic_implicit_pcg_residual_tol` **outside** the JIT; a driver/bench
can do the same via that API. It is *never* loop control (that would
reintroduce data-dependent collectives → deadlock). (Threading the
residual out through `model.step` — which today returns only state — into
production logging is a small follow-up; the contract is reachable now
via the solver-level `return_residual` argument.)

The post-solve global-mean mass projection only fixes the area-mean
bias, **not** spatial under-convergence — hence the residual contract.

## Mass conservation (pre-existing MPI bug, fixed here)

The post-solve mean-drift projection previously used rank-local
`jnp.sum` over ocean-area / target-mass / actual-mass → a **per-rank**
mass correction under MPI (distorts the spatial η field per-rank; the
outer `fix_eta_drift` global projection can only fix the global mean and
may be disabled). Now the three sums go through one MPI-aware batched
`SUM` (the same `is_multi_process()`-gated `batch_allreduce_mpi` pattern
as the adjacent `eta_floor.clamp_and_redistribute`).

## Shared helper (no duplicated numerics)

The distributed solver lives in
`ocean/dynamics/barotropic_common.solve_helmholtz_implicit`; the lat-lon
C-grid solver (`barotropic_implicit_latlon_cgrid.py`) calls it. The
`A_op`/`M_inv` operators are opaque callables (the halo exchange and
metrics live inside them), so the helper stays grid-agnostic and touches
no operator package — it is written to be reusable by MPAS too once the
Voronoi halo / owned-mask infrastructure lands (see *MPAS scope*). MPAS
currently shares only the grid-agnostic CG numerics conceptually; it
runs stock `jax.scipy` CG. The shared
`barotropic_common.compute_filter_weights` / `bebt_blend` / `maxvel_clip`
helpers (and the new solver) keep barotropic numerics single-home;
`tests/ocean/unit/test_no_scheme_duplication.py` enforces it.

## MPAS scope (DEFERRED — single-rank stock CG only)

The Voronoi Helmholtz has the *identical* `I − coeff·div(H·grad)`
structure with the same area-weighted symmetry, so in principle the
shared helper reuses cleanly. **But MPAS is NOT distributed here**, for
two infrastructure reasons the adversarial review surfaced:

1. **Halo-in-matvec.** The MPAS barotropic `A_op` → `gradient_edge`
   indexes `phi_cell[cellsOnEdge]` *locally* and does **no** ghost-cell
   exchange (neither does the explicit substep loop). A fixed-iteration
   PCG changes `p`/`eta` every iteration, so ghost cells would go stale
   after the first matvec. A correct distributed MPAS PCG must call a
   Voronoi cell halo-exchange **inside** `A_op`.
2. **Owned-cell reductions.** Voronoi local meshes hold owned **+ halo**
   cells (`voronoi_mpi.make_voronoi_partition_layout` exposes
   `owned_mask_cells`). PCG dot products / residual / mass projection
   would double-count ghost cells unless every global SUM is masked to
   owned cells.

The lat-lon band decomposition has neither problem (its `A_op` pre-pads
through the backend-dispatched halo, and cell rows partition without
overlap → no ghost cells in the reduction). So **MPAS keeps stock CG**
(single-rank only — its MPI path is forward-only for AD per
`voronoi_mpi.py`). The only MPAS change shipped here is the
single-rank-safe accumulation-precision mass projection; the
distributed PCG is `TODO(distributed-mpas-pcg)` (see the Step-4 note in
`barotropic_implicit_mpas.py`). The config fields exist on
`MPASOceanConfig` so the schema matches the lat-lon path when it lands.

## Acceptance

1. Single-rank PCG vs stock CG: numerical equivalence ≤1e-10 (f64).
2. np=2 parity: gathered η/u/v after solve == single-rank decomposition
   ≤1e-10 (f64), no deadlock under JIT and grad.
3. Residual: final global relative residual ≤
   `barotropic_implicit_pcg_residual_tol` at 1° and ¼°.
4. Mass: global `Σ η·area·mask` after solve/projection matches target at
   roundoff (global SUM, no per-rank mean correction).
5. AD: `jax.grad` through one barotropic solve finite and matches finite
   difference on a tiny grid, single-rank and structurally under np=2.
6. Weak scaling: `bench_ocean_mpi_scaling.py --baro-solver implicit_cn`
   weak np 1/2/4/8 shows **flat** step time (~`2*M` reductions/step,
   resolution-independent) vs `explicit_substep`'s growth.
7. Solver choice stays config-driven: at coarse 1° the explicit substep
   count is small, so explicit may still win — pick via `--baro-solver`.
