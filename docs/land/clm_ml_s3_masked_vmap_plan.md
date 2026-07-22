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

## RE-SCOPING (2026-07-22) — uniform ncan/ntop makes S3 far smaller

Two facts discovered after vendoring the backend in-repo change the S3 cost estimate:

1. **The backend is now vendored** (`packages/land/legoesm/land/canopy/clm_ml_backend/`,
   PR #1269). S3 lands IN legoESM — no external clm-ml-jax PR/release needed.
2. **`ncan`/`ntop` are UNIFORM across columns** in the DEFAULT (explicit-count)
   layering mode (`nlayer_within>0 and nlayer_above>0`, set by #1268's
   `_apply_canopy_layering`). `MLinitVerticalMod` lines 146-151:
   `_ntop = nlayer_within`, `_ncan = _ntop + nabove` — **independent of the column's
   htop**. Only `nbot` (beta-distribution + `dpai_min` zeroing) varies per column/PFT.

So S3 does NOT need to mask every per-layer loop. The tractable path:

- **`lax.map` (or `vmap`) the single-column kernel over columns** at the interface
  (replaces the S2 Python loop → O(1) compile). Under the map, the patch index
  `grid.p = c` is a TRACED int — jax handles traced-index gather (`X[p]`) and scatter
  (`.at[p].set`), so the per-patch data access needs no restructure.
- **`ncan`/`ntop` stay SHARED CONCRETE** (uniform, from the config) → every
  `range(1, ncan+1)` / `range(1, ntop+1)` loop stays STATIC (unrolled once, shared
  across the map). Turbulence, RungeKutta, FluxProfile, Photosynthesis use only
  `ncan`/`ntop` → **NO masking** (they were most of the 14 loops).
- **`nbot` is the only per-column-varying int** → make `grid.nbot` a TRACED scalar
  and MASK just the `nbot`-dependent loops. Scope: `MLSolarRadiationMod` (34 refs,
  `range(ntop, nbot-1, -1)`) + `MLLongwaveRadiationMod` (14 refs, `range(nbot, ntop+1)`).
  Convert each to a static `range(1, ntop+1)` with a `(ic >= nbot)` / `(ic <= ntop)`
  mask (nbot traced). ~2 modules, not 5.
- **`MLinitVerticalMod` stays exempt** (eager warm-start only).

Gate the map path on uniform structure (explicit-count mode); fall back to the S2
loop for height-increment mode (varying ncan) — so it is never wrong, only slower
there.

VALIDATION unchanged: whole-canopy column-parity vs the S2 loop
(`test_multicolumn_matches_independent_single_columns`) is the oracle; assert equal
compile time at ncol=2 vs ncol=64 (O(1)); gradients stay finite.

RISK: still a radiation-kernel edit (solar/longwave nbot masking) with slow (5-16 min)
per-parity validation, but ~2 modules + the interface map — a focused single-session
effort, not the 5-module rewrite the original estimate implied.

## FINAL IMPLEMENTATION SCOPE (2026-07-22) — lax.scan over columns, traced p

The cleanest O(1)-compile path is **`jax.lax.scan` over columns** (NOT vmap): the scan
carry IS the shared `mlcanopy` (scattered at a traced patch index `p=c` each step), so
NO per-column-pure kernel restructure is needed — the existing scatter-into-shared flow
threads through the carry. Requirements (all in the vendored backend now, no external PR):

1. **Traced `p` support in the `grid=` path.** The kernel still reads per-column CLM
   PROCESS GLOBALS by `p` with numpy + `int()`, which break under a traced `p`. Convert
   these to `jnp` dynamic gather (device, traced-index-safe):
   - `MLSolarRadiationMod.py:164` `pft = int(_itype_np[p])` — the per-column PFT. Thread
     it as a per-column jnp value (from `grid` / a `(ncol,)` array) and make the MLpftcon
     table lookups (`table[pft]`, spread across Solar + `MLLeafPhotosynthesisMod`) DYNAMIC
     gathers on traced `pft`. This is the biggest sub-task.
   - `MLCanopyFluxesMod.py:489,1050-1056,1137-1142` `int(_patch_col_np[p])` /
     `_patch_gridcell_np[p]` / `_snl_np[c]` / `grc.latdeg[g]` — patch→column→gridcell
     topology maps. Single-site they are identity-ish; under scan, jnp-gather at traced p.
   - `col.z/zi` (soil grid) are UNIFORM across columns → fine as globals (no per-p read).
2. **`ncan`/`ntop` stay CONCRETE static** (uniform explicit-count) → all their
   `range(1, ncan+1)` loops stay static across the scan body. No change.
3. **`nbot` becomes a traced per-column value** (`grid.nbot` from a `(ncol,)` array) →
   MASK the `range(nbot, ntop+1)` / `range(ntop, nbot-1, -1)` loops in
   `MLSolarRadiationMod` (34 refs) + `MLLongwaveRadiationMod` (14 refs): iterate the
   static `1..ntop` and gate each layer op with `(ic >= nbot)`. DELICATE — the LW/SW
   two-stream solves are layer-coupled; the mask must zero exactly the below-`nbot`
   layers' contributions without changing the linear-system structure. This is the
   radiation-numerics risk; validate as a NO-OP first (nbot still concrete → masked ==
   ranged), THEN make nbot traced.
4. **Interface:** replace the S2 per-column Python loop with `lax.scan` (carry=mlcanopy,
   xs=arange(ncol) + per-column nbot/pft arrays). `extract_clm_ml_grid_info` returns
   `(ncol,)` nbot/pft arrays alongside the shared static ncan/ntop.

INCREMENTAL, VALIDATED ORDER (S2 loop stays default behind a `scan_columns` flag):
(a) nbot-mask Solar+Longwave as a NO-OP (concrete nbot) → device==host parity UNCHANGED.
(b) de-host PFT + topology reads to jnp-gather (concrete p) → parity UNCHANGED.
(c) flip to lax.scan (traced p, traced nbot/pft) → whole-canopy column-parity vs S2 loop.
(d) assert O(1) compile (ncol=2 vs ncol=64).

Each of (a)-(d) is independently parity-checkable against the current per-column path —
so the radiation-boundary risk is caught before the scan flip. Focused single-session
effort; do NOT rush (an nbot-boundary bug "would corrupt all global runs").

## SHIPPED (2026-07-22) — Phase 1: uniform-structure lax.scan achieves the S3 goal

**S3's compile goal is DONE for the current interface via a uniform-structure scan —
the delicate nbot-masking radiation rewrite of (a)/(c) turned out to be UNNECESSARY
for anything the interface can currently produce.** Two measured facts collapsed the
scope:

1. **The interface is single-PFT.** `compute_clm_ml_canopy_fluxes`'s topology setup
   sets `itype_arr[p] = pft_clm` for EVERY column (one `CLMMLCanopyConfig.pft_clm`),
   so all columns share a PFT. Per-column PFT is not plumbed.
2. **`nbot` is invariant to LAI for a fixed PFT.** Warm-starting ncol=3 with
   `lai_override` spanning `[0.05, 0.15, 12.0]` (desert→rainforest) gave the SAME
   `(ncan,ntop,nbot,pft) = (9,8,2,7)` for every column — the dpai_min zeroing +
   redistribution leaves the active-layer bottom fixed across the whole realistic LAI
   range.

⇒ Every current multi-column CLM-ML run has UNIFORM `(ncan,ntop,nbot,pft)`, so the
**single lax.scan over columns** (Phase 1) — carry = shared mlcanopy, only the patch
index `p` traced, structure closed-over CONCRETE — traces ONCE and gives **O(1)
compile in ncol** with NO per-layer masking and NO dynamic pft-gather. Validated
(`tests/land/integration/test_clm_ml_jit_forward.py`): device==host single-column
no-op, ncol=2 column-parity vs independent single-column runs, and an ncol=4
O(1)-compile + scan==loop no-op test. Codex-clean (3 rounds). Behind
`CLMMLCanopyConfig.scan_columns` (default True); heterogeneous structure (should it
ever arise) falls back to the proven S2 per-column loop — correct, only O(ncol) there.

### Remaining ONLY when per-column PFT is plumbed (not today)
Heterogeneous structure becomes reachable only if a future interface assigns
per-column PFT (different biomes) — then `nbot`/`pft` vary across columns and the
single concrete-structure scan would be wrong (it falls back to the O(ncol) loop).
The PREFERRED extension is **NOT** the nbot-masking radiation rewrite of (a)/(c)
above (HIGH risk — coupled two-stream tridiagonal solve) but **GROUP-BY-STRUCTURE**:
partition columns into groups sharing `(ncan,ntop,nbot,pft)`, run ONE Phase-1 uniform
scan per group (body closed over that group's concrete structure, `jax.lax.scan` over
the group's patch-index array). Compile is O(#distinct structures) — bounded by
~(PFTs × nbot values), independent of ncol — reusing the VALIDATED uniform-scan
machinery with ZERO radiation risk. (Prototyped 2026-07-22 and reverted UNSHIPPED:
its multi-group path is currently unreachable/untestable with the single-PFT interface,
so shipping it would be untested speculative code — implement + test it in the same PR
that plumbs per-column PFT.) The nbot-masking approach (a)/(c) remains a theoretical
alternative only if a single O(1) scan over ALL heterogeneous columns is ever needed
(vs O(#structures) groups), and only with the mixed-structure parity oracle that
per-column PFT would make constructible.
