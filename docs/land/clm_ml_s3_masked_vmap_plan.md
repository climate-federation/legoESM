# CLM-ML S3 — masked-vmap for AMIP-scale compile

**Status:** design (not implemented). Prereq for production-resolution coupled CLM-ML.
**Owner note:** the kernel changes are in the EXTERNAL repo `clm-ml-jax`
(github.com/AyaLahlou/clm-ml-jax) → need an upstream PR + release + a `[canopy]` pin
bump. The legoESM interface changes are local.

## Problem

The ncol>1 traceable canopy (S2) is an interface-level Python loop over columns
(`compute_clm_ml_canopy_fluxes` calls `MLCanopyFluxes` once per column). The loop
UNROLLS in the trace, so the HLO — and the compile time — is **O(ncol)**:

| ncol | land-tile compile (measured / extrapolated) |
|---|---|
| 1   | ~seconds |
| 24  | ~11 min (measured; `test_clm_ml_pipeline_step_jits`) |
| 1000+ | hours / OOM |

This is the only wall left for coupled CLM-ML at AMIP resolution; the physics,
differentiability, warm-start, coupler threading and faithful solar zenith all work
(S1/S2/coupler/cos_zenith, 10 commits on `feat/clmml-dehost-global`).

## Why there is no shortcut

The backend physics reads a **concrete Python int** `ncan` (per column) to drive
`for ic in range(1, ncan+1)` loops and `arr[p, 1:ncan+1]` slices. Under `jax.vmap`
or `jax.lax.map`/`scan` over columns, `ncan` would become a batched/traced value —
Python-range and static-slice both break. So columns cannot be vectorised until
every concrete-`ncan` site is converted to a **fixed-`nlevmlcan` masked op**
(`ncan` becomes DATA — a per-column mask — not control flow). Partial conversion
buys nothing: any remaining concrete-`ncan` site still forces per-column tracing.

## Scope (measured, clm-ml-jax @ f36ba27 + feat/device-solar-zenith)

Concrete-`ncan`/`ntop`/`nbot` sites that must become masked:

| module | `range(1,ncan+1)` loops | `int()` casts | notes |
|---|---|---|---|
| MLCanopyTurbulenceMod   | 10 | 6 | `_HF2008_diff`: wind/mflx/conductance layer loops |
| MLinitVerticalMod       | 12 | 1 | cold-start ONLY (runs eager at warm-start; can stay concrete — see below) |
| MLSolarRadiationMod     | 0  | 10 | Norman/TwoStream: `_ncan/_ntop/_nbot` per fp |
| MLCanopyFluxesMod       | 2  | 8  | PAI loop, `ncan_vals`, o2ref |
| MLFluxProfileSolutionMod| 0  | 7  | WellMixed / implicit solve |
| MLLeafPhotosynthesisMod | 0  | 2  | per-patch `_ncan_p` |
| MLLongwaveRadiationMod  | 0  | 2  | `ntop`/`nbot` bounds |
| MLRungeKuttaMod         | 0  | 1  | `n = grid.ncan` |

`MLinitVerticalMod` is EXEMPT from the vmap path: vertical structure is built once
at the eager warm-start (setup), never inside the jitted step — leave it concrete.
So the vmap target is the **per-sub-step physics** (turbulence + radiation + RK +
flux-profile + photosynthesis), ~14 loops + ~30 int-casts.

## Approach

1. **Fixed width.** All per-layer arrays are already `[np, nlevmlcan+2]` (padded).
   Replace every `range(1, ncan+1)` / `[1:ncan+1]` with a full `1:nlevmlcan+1`
   op multiplied by a per-column layer mask `m = jnp.arange(1, nlevmlcan+1) <= ncan`
   (and analogous `ntop`/`nbot` masks). Reductions become `jnp.sum(x * m)`; scatters
   become `jnp.where(m, new, old)`. `ncan`/`ntop`/`nbot` move from Python ints to
   `(ncol,)` DATA arrays in `GridInfo` (already per-column from S2's tuple — collapse
   the tuple into batched arrays).
2. **Batch the patch dim.** Drop the `for p in filter` / `for fp` loops; operate on
   the whole `[ncol, nlevmlcan, ...]` block. `grid.p` → the column axis.
3. **vmap or lax.map over columns** at the `MLCanopyFluxes` seam so the canopy is
   traced ONCE (compile O(1) in ncol). `lax.map` if memory-bound, `vmap` if not.
4. **Interface.** `compute_clm_ml_canopy_fluxes`: replace the S2 per-column Python
   loop with the single batched call; `extract_clm_ml_grid_info` returns batched
   `(ncol,)` `ncan/ntop/nbot` arrays instead of a tuple of scalars.

## Validation (non-negotiable)

- **Per-column parity:** the masked-vmap result MUST equal the S2 loop result
  column-for-column (atol 1e-6), reusing `test_multicolumn_matches_independent_single_columns`
  as the oracle. The S2 loop is the trusted reference.
- **Compile-scaling:** assert ncol=2 and ncol=64 have ~equal compile time (O(1)),
  vs the S2 loop's O(ncol).
- **Masking self-test:** a column with ncan < nlevmlcan must give bit-identical
  fluxes to the concrete-slice path (the mask zeroes exactly the padding layers).
- **Gradients:** `jax.grad` through the vmapped canopy stays finite (the masks are
  smooth `where`, no `int(tracer)`).
- Fresh process per test (CLM-ML process globals contaminate; reset
  `ifc._last_topology_key`).

## Risk

HIGH — per-layer numerics rewrite across 5 physics modules in an externally-owned
repo. Every masked reduction is a chance for an off-by-one at the `ntop`/`nbot`
boundary. Mitigate with the column-parity oracle (S2 loop) run BEFORE trusting any
speedup. Multi-session; do one module at a time behind a `masked_vmap` flag,
keeping the S2 loop as the default until all modules + parity pass.

## Complementary: MPI/SPMD (partial scale without S3)

Distributing columns across R ranks makes each rank's loop **O(ncol/R)** — so MPI is
a partial scale path even before S3 (100 ranks × 40 cols/rank ≈ the ncol=40 compile).
Blocker: the 1-based `mlcanopy` `[ncol+1]` arrays don't match the per-column scatter
(`model_driver._map_flat_column_leaves`, assumes `[ncol]`), and the warm-start +
grid_info are built on GLOBAL columns before the scatter. Needs: (a) a 1-based-aware
scatter for `canopy_state` (scatter `[1:ncol+1]`, keep index 0), (b) per-rank
grid_info (slice the tuple / re-extract after scatter), (c) rank-local topology
(the interface re-installs per trace using the local ncol — already handles it).
Currently refused at `model_driver.py` (the MPI-scatter guard).
