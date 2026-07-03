# MPAS Ocean Performance Optimization Plan

*Created: 2026-04-29*
*Goal: bring MPAS per-sim-year wall time from ~2.5× lat-lon down to ~1.5× lat-lon at comparable resolution (`ico4` vs `36×72`), without changing scientific behavior.*

## Current state (baseline)

At `ico4` (~2562 cells, ~4° spacing) with `barotropic_solver="implicit_cn"`,
20 vertical levels, on Apple Silicon CPU:

| Run | wall / sim-yr | Notes |
|---|---|---|
| lat-lon `36×72` implicit | ~5 min/yr | Reference |
| MPAS `ico4` implicit | ~13 min/yr | 2.6× slower |

For Eady-channel runs at ~100 km, the gap widens to 5–8× because per-step
overhead dominates short integrations.

**Bit-equivalent constraint**: optimizations must preserve scientific
output to within `1e-12` relative on a 100-step regression test.  Pure
algorithmic improvements (warm-start, sparser matvec) and pure
restructuring (sparse matrix replacing gather loop) qualify; precision
changes (float64→float32) do not.

---

## Phase 0 — Profile and baseline (2 hours, mandatory)

Without a flame graph we are guessing.

1. **Set up profile capture** in `scripts/profile_mpas_step.py`:
   - Build `MPASOceanModel` at `ico4`, 20 levels, `implicit_cn`.
   - Warm up JIT (5 untimed steps).
   - `with jax.profiler.trace("/tmp/mpas_trace")`: 100 steps.
   - Open in `tensorboard --logdir=/tmp/mpas_trace --bind_all`.
   - Save flame graph PNG + per-op time table to
     `docs/dev-notes/research/mpas_profile_baseline.md`.

2. **Identify the top-5 hot ops** by cumulative time.  Expected
   candidates (to be confirmed):
   - TRiSK PV / tangential reconstruction at edges
   - `gradient_edge_3d`, `cell_to_edge_avg_3d`, `divergence_cell_3d`
   - Neumann fill (called from GM/Redi and EOS iteration)
   - Implicit-solver PCG matvec
   - Wright EOS polynomial evaluation

3. **Acceptance**: numbers + flame graph committed.  Decide which of
   Phases 1–6 are worth doing based on what actually dominates.  This
   document orders them by *expected* payoff; profile may reorder.

---

## Phase 1 — Precomputed Neumann-fill operator (4 hours, ~10–20% gain)

**Hypothesis**: `_voronoi_neumann_fill` runs three passes of
`gather → mask → average → where` on every call.  Three callers per
step (slope, q-fill, ρ-fill) × 3 passes = 9 gather rounds per step.
The fill *topology* is constant, only field values change.

**Approach**:
1. At mesh-build time (in `VoronoiMesh.__post_init__` or a new
   `precompute_neumann_fill_operator` helper), construct a single sparse
   matrix `F: nCells → nCells` such that `F @ f` reproduces the 3-pass
   fill in one shot.
   - Compose `F = F_3 · F_2 · F_1`, multiply once.
   - Materialise as `jax.experimental.sparse.BCOO`.
2. Replace `_voronoi_neumann_fill` body with a single
   `bcoo_dot_general(F, f)`; keep the existing function name and
   signature for back-compat.
3. Mask multiplication still applied at the end (cheap).

**Validation**:
- Bit-compare current fill output vs new at 1e-12 on `ico4` random
  density field with realistic land mask.
- Re-run `tests/ocean/unit/test_gm_redi_mpas.py` — must pass.

**Risk**: building `F` for n_passes=3 with up to 12-edge polygons gives
a sparse matrix with up to ~80 nz/row.  Still tiny compared to nCells.

---

## Phase 2 — Sparse-matmul replacement for hot edge operators (1 day, ~1.5× on operators)

**Hypothesis**: the 5 most-used Voronoi primitives all reduce to
"gather an indexed array, weight by precomputed scalars, sum into the
output array."  XLA does not vectorize this cleanly when written as
`arr[idx]` — the indirection blocks vectorization.  As sparse matmul,
it lowers to optimized BLAS-style kernels.

**Targets** (in order):
1. `gradient_edge_3d`: `grad_n = (q[c1] − q[c2]) / dcEdge` → sparse
   `grad = G @ q` with `G` having two non-zeros per row.
2. `divergence_cell_3d`: `div_c = Σ_e sign · F_e · dvEdge / areaCell` →
   sparse `div = D @ F`.
3. `cell_to_edge_avg_3d`: 0.5-weighted average of `q[c1] + q[c2]` →
   sparse `M @ q` with two `0.5`-entries per row.
4. `_perot_inner_product_cell`: per-cell weighted sum of edge products
   → bilinear sparse contraction.  Two sparse matrices, one per arg.
5. (If profile shows hot) TRiSK tangential reconstruction: sparse
   matrix with up to ~10 nz/row from `edgesOnEdge`.

**Approach**:
1. Add `precompute_sparse_operators(mesh) → MeshSparseOperators` (a
   NamedTuple of BCOO matrices) in `core/operators_voronoi.py`.
2. Add `_3d_sparse` variants that take the mesh + the sparse op and do
   `bcoo_dot_general` over the leading cell/edge axis, broadcasting
   over `nlev`.
3. Switch call sites incrementally, gated by a config flag
   `MPASOceanConfig.use_sparse_operators: bool = False`.  Default off
   until validated.
4. Once 100-step bit-comparison passes for all callers, flip the
   default to `True` and remove the flag in a follow-up PR.

**Validation**:
- Per-op unit test: `|sparse(x) − dense(x)| / |dense(x)| < 1e-13`.
- Full regression: 100-step bit-compare on `ico4` Eady channel.
- Differentiability: `jax.grad` of a tracer integral through the
  sparse operators must produce gradients matching the dense path
  to 1e-10 (sparse matmul has slightly different summation order,
  hence not bit-identical).

**Risk**: BCOO matvec on Apple Silicon CPU is well-supported; on GPU
it's even faster.  Differentiability is guaranteed (BCOO supports
`jax.grad`).  The risk is in correctness of the precomputed sparse
matrices — hence the per-op unit tests.

---

## Phase 3 — PCG warm-start + diagonal preconditioner (3 hours, 2–3× on barotropic solver)

**Hypothesis**: the implicit Crank–Nicolson solver runs PCG with
Jacobi preconditioning from a zero initial guess every step.  The
solution between consecutive timesteps changes only by O(dt) — warm-
starting from the previous step's η will cut iterations from ~10 to ~3.

**Approach**:
1. Add `eta_pcg_guess: jnp.ndarray | None = None` to
   `BarotropicState` (or carry through the model state, alongside
   `eta` and `u_bar`).
2. In `barotropic_implicit_mpas`, pass `eta_pcg_guess` as
   `x0` to the CG call.  After solve, store the new `eta` as the next
   guess (it's the same field, so this is essentially free).
3. Optional: explore a slightly smarter preconditioner — compute the
   diagonal of the Helmholtz matrix `A(η) = η − coeff · div(H · grad η)`
   exactly (currently Jacobi uses 1.0 or rough diag).  Should converge
   in ~5 iterations cold, ~2 warm.

**Validation**:
- Iteration-count timeseries from a 100-step run: confirm warm-start
  cuts mean iterations from ~10 to ~3.
- 1-yr Eady channel run: bit-compare endpoint.
  Warm-start is mathematically equivalent to converged-PCG output
  with the same tolerance, so should bit-match within solver tolerance
  (~1e-10).

**Risk**: low.  Warm-starting CG is a textbook trick; only failure
mode is divergence if `eta_pcg_guess` somehow drifts wildly between
steps (it doesn't — η is band-limited by gravity-wave physics).

---

## Phase 4 — Single-pass EOS for linear case (2 hours, ~5–10% gain when linear EOS is used)

**Hypothesis**: `iterate_eos_and_pressure_anomaly` runs 2 EOS calls
per dycore call to converge the implicit pressure-anomaly iteration.
For `eos="linear"`, ρ does not depend on pressure → second pass is
guaranteed to do nothing.  We waste 50% of the EOS work on every
linear-EOS experiment (which is most of the matrix and all the
GM/Redi validation cases).

**Approach**:
1. In `ocean_tendency_common.iterate_eos_and_pressure_anomaly`, accept
   `n_iter` argument (already does — default 2).
2. At call sites in `ocean_model_mpas.py` and the GM/Redi top-level,
   detect linear EOS and pass `n_iter=1`:
   ```python
   n_iter = 1 if config.eos == "linear" else 2
   ```
3. Add explicit unit test confirming `n_iter=1` and `n_iter=2` give
   bit-identical ρ for linear EOS.

**Validation**: per-step regression on `ico4` Eady GM/Redi —
linear-EOS bit-match required, Wright unchanged.

**Risk**: zero, this is a math identity for linear EOS.

---

## Phase 5 — JIT scope audit (4 hours, 5–15% gain if hot)

**Hypothesis**: if `gm_redi_tracer_tendency_mpas`, the implicit
solver, and the dycore step are independently JIT-compiled and called
as black boxes from `step()`, XLA cannot fuse across operator
boundaries.  Common gathers (e.g., the same edge-mask multiply, the
same `cellsOnEdge` lookup) get re-executed.

**Approach**:
1. Inspect with `jax.make_jaxpr(model.step)(state, dt)` — count gather
   operations and look for redundant ones.
2. If multiple JIT regions exist, fuse via a single
   `@jax.jit(static_argnums=...)` decorator on the public `step()`.
   Internal helpers should not have their own `@jax.jit` if they're
   only called from inside `step`.
3. Audit all `@jax.jit` decorations in
   `src/legoesm/ocean/dynamics/*_mpas*.py` — keep only the outer one.

**Validation**:
- 100-step bit-compare on `ico4`.
- Compile-time should *increase* (one big trace), step-time should
  *decrease* (more fusion).  Net wall for 1-yr run is the metric.

**Risk**: medium.  Fusing a too-large region can blow XLA's
optimization budget and produce slower code.  Measure both directions.

---

## Phase 6 — Better PCG preconditioner (1 day, ~1.5× on implicit solve, *only if Phase 3 underperforms*)

**Hypothesis**: Jacobi preconditioning is weak for the Helmholtz
operator near polar caps where `H` and grid spacing vary rapidly.

**Approach**:
1. Compute the **exact diagonal** of the Helmholtz operator
   `A(η) = η − dt²·θ²·g · div(H · grad η)` once at mesh-build time.
   Diagonal is a function of `dcEdge`, `dvEdge`, `areaCell`, `H_e`.
2. Replace Jacobi with `M^{-1} = diag(diag(A))^{-1}`.
3. Optional bonus: try one V-cycle of geometric multigrid as an
   M-step.  Voronoi multigrid coarsening is non-trivial — this is
   a bigger chunk of work, defer unless Phase 3+5 don't get us to
   target.

**Validation**: convergence-iteration count must monotonically
decrease vs Jacobi on a battery of test cases (Eady, Drake, global
overturning).

**Risk**: medium — getting the diagonal right requires care; an
incorrect preconditioner doesn't break correctness (CG still
converges) but degrades performance.

---

## Out of scope

- **GPU port (CUDA / Metal)**: would benefit MPAS more than lat-lon,
  but requires jax-mps / MLX maturity or moving compute off Apple
  Silicon.  Track separately.
- **Custom kernels** (Triton, pallas) for hot ops: way more
  engineering than legoESM's research scope justifies before mesh
  refinement is unlocked.
- **Float32**: not worth conservation drift at long integration
  lengths.
- **Algorithmic changes** (e.g., switching from TRiSK to a different
  Voronoi reconstruction): not a performance fix, a re-design.

---

## Phasing and dependencies

```
Phase 0 (profile)  ─┬─→  Phase 1 (Neumann)
                    ├─→  Phase 2 (sparse ops)   ┐
                    ├─→  Phase 3 (PCG warm)     ├─→  Phase 6 (precond, if needed)
                    ├─→  Phase 4 (linear EOS)
                    └─→  Phase 5 (JIT scope)
```

Phases 1–5 are independent; they can be done in any order or in
parallel by separate worktrees.  Phase 6 is conditional on Phase 3
not hitting the target.

## Headline target

| Metric | Now | After Phases 1–5 | Stretch (with Phase 6) |
|---|---|---|---|
| MPAS / lat-lon ratio at `ico4` | 2.6× | 1.5× | 1.3× |
| 50-yr `ico4` wall | 11 h | 6.5 h | 5.6 h |

If we hit 1.5×, the long MPAS overturning runs become as routine as
the lat-lon ones.  If we hit 1.3×, MPAS becomes preferable for any
experiment where polar resolution matters.

---

## Validation harness

Add `tests/performance/test_mpas_step_speed.py`:

- Fixture: `ico4`, 20 levels, `implicit_cn`, ocean-only init.
- Warm up 5 steps.
- Time 100 steps.
- Assert per-step wall < threshold (initially permissive; tighten as
  phases land).
- Skip-marker if not on Apple Silicon CPU (other backends will have
  different absolute numbers).

This makes regressions visible in CI rather than discovered the next
time someone runs a 50-yr experiment.

---

## Risks and how we mitigate

1. **A "speedup" that breaks AD silently.** Mitigation: every phase
   includes a `jax.grad` check on a small loss through the modified
   path.

2. **A speedup that breaks bit-reproducibility of long runs.**
   Mitigation: each phase has a 100-step regression with `1e-12`
   relative tolerance.  Sparse matmul (Phase 2) is the only phase
   where we accept *non-bit-identical* output (different summation
   order); that one holds the higher 1e-10 bar plus a passing 5-yr
   ico4 spinup with statistically equivalent diagnostics.

3. **Time spent on optimization that's a dead-end before mesh
   refinement.** Mitigation: Phase 0 gives us data; if profile shows
   we're already memory-bandwidth-bound everywhere, Phases 2 and 6 are
   re-evaluated.

## Reference docs (where to look first)

- Profiling JAX: https://docs.jax.dev/en/latest/profiling.html
- BCOO sparse matmul: `jax.experimental.sparse` docs.
- TRiSK weights: Thuburn 2008 (`docs/references/Thuburn2008.pdf`),
  Ringler+ 2010 (`docs/references/Ringler2010.pdf`).
- Existing implicit solver: `barotropic_implicit_mpas.py` and the
  lat-lon analog (cleaner reference design).
