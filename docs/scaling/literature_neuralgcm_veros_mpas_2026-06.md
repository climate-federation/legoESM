# NeuralGCM · Veros(JAX) · MPAS — borrowable parallelization strategies

Companion to `literature_parallelization_2026-06.md` and
`distance_to_limit_2026-06-13.md`. Three primary-source deep reads
(2026-06-13) of the systems closest to legoESM: NeuralGCM/dinosaur
(JAX spectral atm), Veros (JAX ocean + mpi4jax), MPAS-A/MPAS-O
(unstructured MPI). Goal: concrete levers for our measured walls, with
honest "does it transfer to Ginsburg (≤32 CPU-MPI ranks, PCIe GPU pairs,
NO NVLink/IB)" verdicts.

## THE convergent finding (highest ROI, act on this)

**Veros and MPAS-O INDEPENDENTLY diagnose our exact ocean-barotropic
wall (Wall A/B) and give the SAME cure: stop doing per-substep /
per-iteration latency-bound barotropic communication; do ONE implicit
elliptic solve per baroclinic step whose iteration count is
~resolution-independent (multigrid-class preconditioner).**

Our wall today: implicit Crank–Nicolson barotropic via fixed-M Jacobi-PCG
weak-scaling-capped by **M≈60 iterations × ~111 µs allreduce LATENCY per
step**. The single-reduce (Chronopoulos–Gear) lever attacked the *per-
iteration reduction count* (3→1) and gave only 3–7% — we marked it
"neutralized." **The literature says the real lever is the ITERATION
COUNT, not the per-iteration reduction count:**

- **MPAS-O** (Kang et al. 2021, JAMES): replaced 30–60 explicit
  barotropic subcycles (each a latency-bound halo) with a **semi-implicit
  Helmholtz SSH solve** using **pipelined PBiCGStab (2 overlapped
  allreduces/iter) + Restricted Additive Schwarz (RAS) preconditioner →
  ~6 FIXED iterations** independent of core count. Result: **2.9×
  barotropic, 1.9× total at 16,320 cores.** RAS = block-Jacobi (local,
  zero-comm) extended by ONE halo layer → 1 halo exchange/precond-apply;
  matrix is time-independent so it's assembled+factored once.
- **Veros** (Häfner et al. 2021, JAMES): never built a distributed JAX
  Krylov at all — uses **PETSc BiCGStab + GAMG algebraic multigrid**
  (iteration count ~resolution-independent, O(5–15) V-cycles) for the
  distributed case, or **gathers the 2-D field to rank 0 and solves
  serially** otherwise. Their explicit verdict: the barotropic solve "is
  the bottleneck at high processor counts."

**Why this is huge for us specifically:** CG/BiCGStab/MG use ONLY
sum-reductions for their dot products → **differentiable through our
`global_sum_mpi` allreduce(SUM) VJP**. We can build the multigrid
preconditioner in pure JAX (V-cycle = weighted-Jacobi smooth +
strided-average restriction + injection prolongation + tiny coarse
solve — all halo-exchange + local stencils, all `lax.scan`/`vmap`-able,
all AD-safe), which is BETTER than Veros's PETSc (AD-opaque) and MPAS-O's
Fortran. **60→~6-10 iters = 6-10× fewer latency events** — far beyond the
3–7% the reduction-count levers gave. This is the next major increment.

## Ranked levers (all three systems merged)

### 1. JAX geometric/algebraic-multigrid (or RAS) preconditioner for the implicit barotropic solve — Wall A/B. HIGHEST ROI.
Cuts barotropic Krylov iterations ~60 → ~6-10 (resolution-independent).
AD-safe (sum-reductions only). Port: (a) assemble the SSH Helmholtz
matvec reusing our divergence/gradient kernels; (b) PCG/BiCGStab as a
fixed-max-iter `lax.while_loop` (static shapes, no recompile);
(c) start with Jacobi precond (MPAS Fig 11: Jacobi suffices below
thousands of ranks — at ≤32 ranks we may not even need RAS), add a
V-cycle or RAS-one-halo-overlap precond only if iteration count climbs.
Evidence: MPAS-O 2.9×/1.9× @16k cores; Veros GAMG keeps JAX+MPI within
1.4× of Fortran to 256 cores. **Differentiable MG beats both prior arts.**

### 2. Field-coalesced halo exchange — one `ppermute` over STACKED fields — Walls A & C. CHEAP, HIGH ROI on PCIe/Gloo.
MPAS `multihalo_exchange_list` packs multiple fields into one exchange.
We already fuse vertical levels (`pad_halo_4d`); extend the doctrine
ACROSS FIELDS: `jnp.stack` the same-halo-depth fields of a stage → ONE
`ppermute` → unstack. On PCIe/no-IB where per-message LATENCY dominates,
K small halos → 1 cuts K latency events to 1. Directly feeds task #11
(cube-path halo fusion, cut 54 ppermutes/step).

### 3. Sub-communicator (axis-restricted) reductions — Wall B / 2-D decomp polar filter. MEDIUM.
Veros `comm.Split` builds per-row/col subcomms so a zonal (along-lon) sum
reduces over `num_proc_lon` ranks, not all P. mpi4jax `allreduce` takes a
`comm=` arg → composes with our AD-safe sum. Apply to pole-fold /
polar-filter zonal sums in the lat-lon 2-D decomposition (the transpose
increment). Cuts reduction width P → proc_lon.

### 4. Recursive nested gradient-checkpoint scan — training memory, hardware-agnostic. CLEAN BORROW.
NeuralGCM `nested_checkpoint_scan`: O(max(nested_lengths)) live memory vs
O(prod), re-running forward `len-1` times. Pure JAX, identical on CPU/
GPU/MPI. Port: factor the `build_segment_fn(...).raw` `lax.scan` length
(e.g. 360=(6,6,10)); composes with `eqx.filter_value_and_grad` + the
non-donating `.raw`. Lets rollout length grow without OOM.

### 5. Offline graph-partition + contiguous renumber for wet-cell balance — Wall C (Voronoi/MPAS). MEDIUM, conditional.
MPAS METIS k-way on the Delaunay dual (balance cells, min edge-cut);
Kang's geometric reorder makes each rank's cells contiguous → a plain
1-D shard == the graph partition (the bridge to `shard_map`). For
wet/dry: **cull dead cells from the partition graph** (don't weight
them) — MPAS-O uses ocean-only meshes. At ≤32 ranks try a **Hilbert SFC
ordering first** (trivially contiguous, captures most of METIS); reach
for weighted METIS only if straggler imbalance is MEASURED.

**IMPLEMENTED 2026-06-21** (branch `perf/voronoi-graph-partition-sfc`): the
METIS-k-way and Hilbert-SFC recommendations above are now in
`parallel/voronoi_partition.py`. `method="auto"` (default) uses METIS when
`pymetis` is present else RCB; `method="sfc"` gives the dependency-free Hilbert
contiguous partition; `reorder_voronoi_for_sharding()` Hilbert-orders cells
within each shard so a plain 1-D `NamedSharding` chunk is spatially compact.
Remaining open from this item: wet/dry dead-cell culling from the partition
graph (ocean-only meshes) and weighted/straggler-driven METIS — still
conditional on MEASURED imbalance.

### 6. Transform/precision levers (spectral atm path only) — Wall C. LOW-MEDIUM.
NeuralGCM: spherical-harmonic transform as DENSE einsum (not FFT) on the
matmul unit, at **TF32/bf16 (3-pass) not f32 (6-pass)**. The precision
lever transfers to GPU tensor cores (shrinks transform bytes, attacks the
small-kernel/HBM-roofline floor); dense-vs-FFT is resolution-dependent —
measure. Spectral path ONLY; irrelevant to cubed-sphere FV3.

## Design validation (no action, confidence boosters)
- **Veros's 2-D pencil == our lat-lon 2-D decomposition exactly**: x(lon)
  scattered first then y(lat); `cyclic` flag = E/W periodic RING vs
  `None`-terminated N/S LINE — identical to our `exchange_halo_lon` ring +
  pole-terminated line. Even-division constraint matches. **Our increment
  1-3 design is the production Veros shape.**
- **Corner handling**: Veros exchanges 4 diagonal corners explicitly in an
  8-direction clockwise `sendrecv` sweep; we do N/S-then-E/W so corners
  ride the lat-padded edge columns into the E/W ring — both correct; ours
  is one fewer message pattern.
- **Our halo-AD is at/ahead of the mpi4jax frontier**: Veros core has ZERO
  end-to-end AD (grep-confirmed); mpi4jax ships VJPs for few collectives.
  Our `_sendrecv_vjp` custom_vjp + allreduce(SUM)-only discipline is
  novel relative to both prior arts.

## Do NOT bother (with reasons)
- **Wider halos to batch barotropic substeps ("halo every K substeps").**
  MPAS-O EXPLICITLY rejected this — explicit scheme is only conditionally
  stable so a wider halo buys only a few stale-data substeps before
  instability, while N-layer halo memory+redundant-compute grows fast.
  Do the implicit solve (#1) instead. **This redirects us off a tempting
  dead-end.**
- **PETSc/AMG or any C-library solver** if we want gradients — AD-opaque.
  Build the multigrid in JAX (#1).
- **MPI persistent requests + hand-placed `MPI_WAIT` interior/exterior
  overlap.** XLA owns collective scheduling; `shard_map` bodies are
  SPMD-uniform. The reuse benefit comes free from XLA caching the
  compiled `ppermute` schedule across `lax.scan`. Borrow only the
  *principle* (emit halo-independent interior compute between a `ppermute`
  and its consumer so XLA's latency-hiding scheduler can overlap it).
- **Model-parallel spectral transform (NeuralGCM sharded collective
  matmul) on our hardware.** Real and impressive on TPU/NVLink (overlaps
  `ppermute` with MXU matmul), but the win NEEDS a fabric that overlaps
  comm with compute. On CPU-Gloo (no overlap) and PCIe-no-NVLink it will
  anti-scale like our same-node multi-process CPU wall. Treat as a
  FUTURE-fabric / 2-GPU-only lever, NOT a CPU-MPI lever. (It does refute
  "transform must stay on one device" as a theoretical matter.)
- **Veros's headline GPU strong-scaling as a target** — every good number
  is on 9.6 TB/s NVLink and even there strong scaling died (3.4× on 16
  GPUs). On PCIe-no-IB: weak-scale + saturate each device, don't
  strong-scale the barotropic solve.
- **MPAS contiguity flags / hash-table bootstrap tuning** — solve problems
  at 10⁵-10⁶ ranks; noise at ≤32.

## Sources
NeuralGCM: Kochkov et al. Nature 2024 (arXiv 2311.07222, App B.3/E.2/G);
github.com/neuralgcm/dinosaur (`spherical_harmonic.py`,
`jax_numpy_utils.py`, `time_integration.py`). Veros: Häfner et al. JAMES
2021 (10.1029/2021MS002717); mpi4jax JOSS 2021 (10.21105/joss.03419);
github.com/team-ocean/veros (`distributed.py`, `core/external/solvers/`).
MPAS: Kang et al. JAMES 2021 (10.1029/2020MS002238, OSTI 1782059);
Heinzeller et al. GMD 2016 (9:77); Jacobsen & Duda non-blocking-halo
design doc; Ringler et al. JCP 2010 (TRiSK operators).
