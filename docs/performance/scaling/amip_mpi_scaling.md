# AMIP-like MPI/GPU scaling campaign — atmosphere then ocean

Goal: push weak + strong scaling of AMIP-like atmosphere (then ocean) runs as
close to the theoretical (roofline / linear-speedup) limit as possible, on MPI
and GPU, for **all grid types**. Testbed physics tier ladder:
Held-Suarez → gray atmosphere → RRTMGP. Held-Suarez is the cheapest tested
forcing and the campaign's strong/weak-scaling tester.

Hardware (this host): RTX 5090 Laptop GPU (live, `jax.default_backend()==gpu`),
24-core CPU, user-space MPICH 4.2.3 at `~/.local/mpich`, `.venv-mpi` (JAX 0.9.2
+ mpi4jax). Run MPI with:

```bash
export PATH="$HOME/.local/mpich/bin:$PATH" \
       LD_LIBRARY_PATH="$HOME/.local/mpich/lib:$LD_LIBRARY_PATH" \
       PYTHONPATH="$PWD"
JAX_PLATFORMS=cpu MPI4JAX_NO_WARN_JAX_VERSION=1 OMP_NUM_THREADS=2 \
  mpirun -np N .venv-mpi/bin/python scripts/bench/run_cpu_mpi_scaling.py \
    --grid icosahedral --physics held_suarez --mode single --resolution 5 \
    --n-levels 26 --n-warmup 3 --n-timing 30
```

## State of MPI decomposition at campaign start (entry audit)

`scripts/bench/run_cpu_mpi_scaling.py` + `_valid_rank_counts`:

| grid          | MPI multi-rank?                          | physics on multi-rank? |
|---------------|------------------------------------------|------------------------|
| icosahedral   | YES — real RCB domain decomposition      | **was refused (iter 1 fix)** |
| latlon        | NO — `make_latlon_mpi_step` raises `NotImplementedError` (#115, C-grid ops not localized) | n/a |
| cubed-sphere  | NO — every rank holds full `(6,n,n,..)` state (replicated, not decomposed) | n/a |
| spectral      | NO — rank-1 only                         | n/a |

So at start, the ONLY grid with genuine multi-rank atmosphere MPI scaling was
icosahedral, and even that was **dycore-only** — `make_voronoi_mpi_step` built a
physics-free step, and the harness hard-refused `--physics held_suarez` on
>1 rank. Net: no grid did real multi-rank MPI scaling *with* the requested
Held-Suarez tester.

## Iteration 1 (2026-06-08): enable multi-rank Held-Suarez on icosahedral

Change (`packages/core/legoesm/parallel/voronoi_mpi.py`,
`scripts/bench/run_cpu_mpi_scaling.py`):

- Added `physics_fn=` to `make_voronoi_mpi_step`. Operator-split, mirroring the
  serial `MPASPrimitiveEquationModel._step_jit` exactly: dynamics RK-integrate →
  fresh halo exchange → evaluate `physics_fn` once on post-dynamics state →
  `state += dt*tendency` → T floor → fix_mass. Held-Suarez is column-local
  (Newtonian relaxation), so it adds no horizontal halo coupling beyond the one
  pre-physics exchange (needed so boundary-owned edges see valid neighbour-cell
  pressure for `cellsOnEdge` edge-sigma). `physics_fn` passed in as a callable —
  core `parallel` keeps zero dependency on the atmosphere component.
- Removed the harness rank-1 refusal + sweep-generator guard for
  icosahedral+physics.
- Limitation (documented in-code): stateful operator-split `phys_state` carry is
  NOT threaded through the MPI step yet — stateless/additive physics only
  (Held-Suarez qualifies). Future item for gray/RRTMGP tiers with prognostic
  physics state.

Correctness: 1-rank and 2-rank gathered trajectories vs serial
`model.step(physics_fn=held_suarez_forcing_mpas)`, 5 steps, I4/L8:
T rel ~1e-10, u rel ~4e-10, p_s rel ~1e-8 (fix_mass reduction order), no NaN.
Codex adversarial review: clean on all 5 axes (ordering, halo, NamedTuple
`type() is tuple` guard, base-state choice, AD/JIT/pytree).

### Baseline measurements (I5 = 10242 cells × 26 lev, dt=390 s, float64)

Strong scaling (fixed I5, vary ranks):

| ranks | HS ms/step | HS speedup | dycore-only ms/step |
|------:|-----------:|-----------:|--------------------:|
| 1     | 29.23      | 1.00×      | 28.29 |
| 2     | 23.58      | 1.24× (62%)| 22.35 |
| 4     | 24.22      | 1.21× (30%)| 22.41 |

Weak (const ~66k cells/rank, apples-to-apples): rank1 I4 = 6.51 ms vs
rank4 I5 = 23.70 ms → 3.6× slower → ~27% weak efficiency.

### Key diagnosis

HS overhead is ~1 ms/step (29.23 vs 28.29 at 1 rank) — cheap and correct.
The strong-scaling plateau is **identical with and without physics**, so it is
**pre-existing in the dynamics MPI halo path**, not from the HS addition:
2→4 ranks yields **zero** improvement. Suspected causes (next iteration):
per-step MPI collective latency (fix_mass allreduce + 3 RK-stage halo sendrecv)
dominating at ~66k cells/rank; surface/volume halo growth; or per-rank XLA
thread contention. Profile before optimizing.

## Iteration 2 (2026-06-08): root-cause the strong-scaling plateau

Profiled the icosahedral dynamics MPI step. Ruled out, in order:

- **Thread oversubscription**: single-thread-per-rank (`XLA_FLAGS=
  --xla_cpu_multi_thread_eigen=false`, `OMP_NUM_THREADS=1`) gives the *same*
  plateau (I5 HS 28.97→23.59→24.08→25.73 ms for 1/2/4/8 ranks). NOT the cause.
  Corollary: the MPAS CPU step barely multithreads (many small gather/scatter
  ops), so more-threads doesn't help either.
- **Raw MPI latency**: allreduce 15–78 µs, sendrecv-8KB 6–16 µs (np 2/4/8).
  ~170 µs total/step — negligible.
- **mpi4jax-in-JIT and -in-scan cost**: 0.77–2.0 ms/step (python-loop),
  2.0–3.5 ms/step (in `lax.scan`). Cheap.
- **mpi4jax as scan optimization barrier**: 1-rank MPI step in scan (34.6 ms) ≈
  plain `model.step` in scan (33.9 ms). Collectives do NOT defeat scan
  amortization. NOT the cause.

**Decomposition (the answer)** — under mpirun, real MPI step vs pure local
compute (collectives removed), I6, single-thread, in scan:

| np | real MPI step | pure local compute | comm overhead |
|---:|--------------:|-------------------:|--------------:|
| 1  | 131.4 ms      | 131.7 ms           | ~0            |
| 4  | 103.6 ms      |  81.3 ms           | **22.3 ms**   |
| 8  | 109.1 ms      |  79.1 ms           | **30.0 ms**   |

Two independent strong-scaling killers, both quantified:

1. **Halo-exchange comm = 22–30 ms/step** — two orders of magnitude above raw
   MPI latency. Root cause located in `_exchange_voronoi_mpi`
   (`packages/core/legoesm/parallel/halo_exchange_voronoi.py`): it loops over
   neighbor ranks doing a **separate blocking `mpi4jax.sendrecv` per neighbor**,
   and mutates `field` *inside* the loop (`field.at[recv_idx].set(recv_data)`).
   But `send_idx` are OWNED cells — send buffers do NOT depend on prior recvs
   (which write HALO cells). That false read-after-write dependency **serializes
   all neighbor exchanges**. np=4 has 3 cell + 3 edge neighbors/rank → 3 RK
   stages × 6 sendrecv = **18 serialized blocking sendrecv/step** ≈ 1.2 ms each
   = the 22 ms. np=8 has 6 cell + 6 edge neighbors/rank → 3 × 12 = **36
   serialized sendrecv/step → 30 ms**. Comm scales with neighbor count, exactly
   tracking the worsening plateau (18→36 sendrecv = 22→30 ms), confirming
   serialization as the mechanism.
2. **Pure local compute floor** — I6 local barely shrinks (131.7 → 81.3 → 79.1
   ms for 4×/8× fewer cells); a large fixed per-step XLA:CPU dispatch cost
   (many un-fusable indirect-indexing kernels) dominates at small per-rank size.

Note: `lax.scan` over the timing window amortizes a large per-step dispatch
overhead (global I6 naive python-loop 200 ms → in-scan 131 ms), so always
measure in-scan to match the production integrator. Cross-resolution fits of the
"fixed floor + per-edge" model are noisy on CPU (thread/cache effects); treat the
floor as a regime, not a precise constant.

## Iteration 3 (2026-06-08): halo-exchange refactor + comm root-cause refined

Refactored `_exchange_mpi` (`halo_exchange_voronoi.py`): gather all neighbor
send buffers from the ORIGINAL field, issue all sendrecvs, then **one**
concatenated scatter at the end (was: a whole-field `field.at[recv_idx].set`
*per neighbor* with a false owned-read-after-halo-write dependency). Result is
**bit-identical** (2-rank HS T rel 1.1e-10, same digits as before; matches
serial). Codex review: clean on all 5 axes (ordering verified against
`_assemble_schedule`, empty-neighbor guard, disjoint indices, dtype/shape, AD/JIT).

**But it is perf-neutral on CPU**: comm overhead 22.1 ms @ np=4 (was 22.3),
28.4 ms @ np=8 (was 30.0) — within noise. So the earlier "N whole-field copies +
false dependency" was NOT the cost (XLA already reused buffers). Refined
root-cause via two microbenches:

- **sendrecv call count is NOT the cost**: 1/3/6/12 blocking mpi4jax.sendrecv per
  step (fixed total payload, synchronized ranks) = 0.10/0.11/0.10/0.18 ms.
  Dispatch + transfer are cheap.
- Therefore the 22–30 ms is **synchronization jitter** accumulated across the
  18 (np=4) / 36 (np=8) *blocking* sendrecv barriers per step. `pure_local`
  compute has zero sync and is perfectly balanced (81 ms on every rank), but each
  real blocking sendrecv forces both ranks to rendezvous; on a shared 24-core
  node, per-barrier OS/runtime jitter × 18–36 barriers/step = tens of ms. This is
  why comm grows with neighbor count (more barriers), not with payload.

Refactor kept regardless: it is correct, cleaner, removes a real anti-pattern,
and **will** help on GPU/NCCL where the N full-field copies and the serialization
do cost (async collectives, no per-message host barrier). The true CPU fix is a
**non-blocking exchange** (post all Irecv/Isend, one Waitall) to collapse the
per-message barriers into one — intricate under mpi4jax's blocking-sendrecv-only
JIT API; deferred.

**Strategic pivot**: CPU-MPI comm is sync-barrier/jitter-bound — a shared-node
artifact, not the production target. The campaign's untouched high-value half is
**GPU**, where the dispatch floor is hidden by async exec and "theoretical limit"
(roofline) is most meaningful. Iter 4 moves to GPU single-device throughput.

## Iteration 4 (2026-06-08): GPU single-device throughput, icosahedral HS

RTX 5090 Laptop, `.venv` backend resolves to `gpu`. Icosahedral Held-Suarez,
L26, float32, in `lax.scan` (200 steps), full end-to-end model.step incl HS:

| res | nCells  | ms/step | Mcells/s |
|-----|--------:|--------:|---------:|
| I4  |   2562  |  0.168  |   397    |
| I5  |  10242  |  0.559  | **477**  |
| I6  |  40962  |  2.863  |   372    |
| I7  | 163842  | 14.965  |   285    |

Throughput **peaks at I5 then DECLINES** — the opposite of a healthy
throughput-vs-size saturation curve — and sits at only ~5–10 % of effective HBM
(~730 GB/s sustained). The MPAS/TRiSK dycore is **gather-bound on GPU**:
unstructured indirect indexing over `cellsOnEdge` / `edgesOnCell` gives
uncoalesced memory access and low arithmetic intensity (many small kernels), so
cache-miss cost grows with mesh size → throughput falls at I6/I7.

Lever test — reorder the mesh for locality (`reorder_voronoi_for_sharding`,
single-GPU cache, NOT for sharding here):

| res | default        | reorder/256    | gain |
|-----|---------------:|---------------:|-----:|
| I6  | 2.842 ms / 375 | 2.795 ms / 383 | +2 % |
| I7  | 14.97 ms / 285 | 13.74 ms / 311 | **+9 %** |

Real but modest — confirms uncoalesced gather as *a* contributor, not the
dominant one. fp64 not characterized (consumer Blackwell fp64 ≈ 1/64 fp32; fp32
is the GPU production precision per CLAUDE.md "finite-volume can be float32").

Takeaways: (a) the unstructured MPAS dycore will not approach GPU roofline
without reducing/fusing the indirect gathers (large refactor); (b) **structured
grids (cubed-sphere, spectral, lat-lon) should hit far higher roofline %**
(coalesced, no indirect indexing) — measuring them is the next high-value GPU
step and gives the "all grid types" comparison the campaign wants; (c) mesh
reorder (+9 % at scale, low-risk preprocessing) is a candidate to wire into the
production GPU path, but it must thread through init + I/O gather consistently
(multi-file, equivalence-tested) — defer until the cross-grid picture says
icosahedral GPU is worth it.

## Iteration 5 (2026-06-08): GPU cross-grid throughput (all 4 grids), HS

Single-GPU RTX 5090, Held-Suarez, L26, in `lax.scan`. float32 for the grid-point
dycores; spectral is float64-only (transforms require it):

| grid          | sizes        | peak Mcells/s | curve shape                  |
|---------------|--------------|--------------:|------------------------------|
| spectral (f64)| T21          |       **8**   | GPU-hostile (fp64+global FFT)|
| icosahedral   | I4→I7        |      477 (I5) | peaks then **declines** (gather-bound) |
| cubed-sphere  | C24→C96      |      666 (C96)| **saturates** (healthy)      |
| lat-lon       | LL64→LL256   |     1100 (LL128) | peaks then declines (pole cost @LL256) |

Per-size detail:
- cubed-sphere f32: C24 473 / C48 656 / C96 666 Mc/s (0.190 / 0.548 / 2.158 ms).
- lat-lon f32: LL64 968 / LL128 1100 / LL256 645 Mc/s (0.220 / 0.774 / 5.287 ms).
- spectral f64: T21 7.89 ms/step, 8 Mc/s (60 k cells) — two orders below the rest.

Conclusions (the cross-grid "theoretical limit" picture):
1. **Structured grid-point dycores (cubed-sphere, lat-lon) are 1.4–2.3× faster**
   than unstructured icosahedral and ~80× faster than spectral on GPU — coalesced
   memory access, no indirect gathers, no global transforms.
2. **Cubed-sphere has the healthiest saturation curve** (throughput rises then
   plateaus at ~666 Mc/s) → best choice for large-scale single-GPU AMIP. Lat-lon
   peaks higher (1100) but degrades at LL256 (polar-filter / very small pole dt).
3. **Spectral is GPU-hostile** (fp64 + global Legendre/FFT all-to-all) — keep it
   CPU-side (consistent with CLAUDE.md "Apple Silicon: spectral on CPU").
4. None reaches the ~730 GB/s HBM roofline; cubed-sphere at 666 Mc/s is closest.
   Headroom remains via kernel fusion / higher arithmetic intensity (large refactor).

Caveat: `run_levante_gpu_scaling.py` returned EMPTY results — ROOT-CAUSED in
iter 6 as the fatal XLA-flag abort (now fixed). The iter-5 numbers above are from
a direct probe reusing `_build_amip_step` (`/tmp/gpu_struct.py` pattern); the
harness itself now works post-fix (still needs `LD_LIBRARY_PATH=MPICH` for its
single-GPU MPI load).

## Iteration 6 (2026-06-08): production GPU XLA-flag fix + gray atmosphere

**Critical production fix** (`packages/core/legoesm/runtime/backend.py`): the
`_NVIDIA_GPU_XLA_FLAGS` dict set `xla_gpu_enable_async_all_reduce=true` and
`xla_gpu_enable_async_collectives=true`. These flag NAMES were removed in current
XLA (JAX 0.10), so XLA emits an **uncatchable `LOG(FATAL) Unknown flags in
XLA_FLAGS`** the instant any GPU backend initialises — silently aborting *every*
production GPU run via `ParallelRuntime.create()` and the GPU scaling harness
(this is the real cause of iter-4/5's "empty results"). Removed both; kept
`latency_hiding_scheduler`, `highest_priority_async_stream`,
`cudnn_gemm_fusion_level=3`, `command_buffer=FUSION,CUSTOM_CALL,COLLECTIVES`
(all accepted by current XLA — verified: gray_sbm GPU run now compiles + times).
[2026-08-11 update: `COLLECTIVES` later measured +25% step time on the MPAS
shard_map lane (job 26873637) and dropped from the default; opt back in with
`LEGOESM_XLA_CMDBUF_COLLECTIVES=1`. See `backend.py`.]
Codex-reviewed: removal is correct (current XLA makes collectives async by
default; LHS handles compute/comm overlap → no serialization on multi-GPU). Repo
grep: no other references to the removed flag strings. Direct probes (iters 4–5)
worked only because they bypass backend.py.

Harness caveat: `run_levante_gpu_scaling.py` also force-loads MPI (`libmpi.so`)
even for single-GPU — needs `LD_LIBRARY_PATH=$HOME/.local/mpich/lib` on the
GPU `.venv` (no MPI installed). Should skip MPI for `n_gpus==1` (future cleanup).

Gray atmosphere (`gray_sbm` = gray radiation + SBM convection + moist tracers,
operator-split segment), single-GPU float32 HS-vs-gray:

| grid / res     | gray Mc/s | HS Mc/s | gray slowdown |
|----------------|----------:|--------:|--------------:|
| cubed C48      |    196    |   656   | 3.3×          |
| cubed C96      |    203    |   666   | 3.3×          |
| cubed C192     |    167    |    —    |               |
| lat-lon LL64   |    325    |   968   | 3.0×          |
| lat-lon LL128  |    473    |  1100   | 2.3×          |
| lat-lon LL256  |    417    |   645   |               |

Gray adds ~2.3–3.3× step cost over Held-Suarez (the radiation+convection+tracer
column), and saturates healthily (~200 Mc/s cubed-sphere, ~470 Mc/s lat-lon).
Lat-lon again faster than cubed-sphere. Gray tier **confirmed** → RRTMGP next.

## Iteration 7 (2026-06-08): RRTMGP tier — catastrophic kernel cost found

`rrtmg_full` = RRTMGP radiation + SBM convection + Kessler microphysics (segment).
Single-GPU float32, cubed-sphere weak (C24, 89 856 cells):

- **JIT compile: 23.0 s** (fast — RRTMGP is NOT compile-bound; my first C48 run
  looked "hung" only because the harness silently auto-scales `--n-timing` up to
  600 and at ~11 s/step that is ~2 h of timing).
- **Timing: 10 901 ms/step**, SYPD 0.113. The SYPD math (dt=450 s) confirms this
  is per-dt-step, not per-segment.

Comparison at comparable size: gray ~1.8 ms/step, Held-Suarez ~0.2 ms/step ⇒
RRTMGP is **~6 000× slower than gray, ~50 000× slower than HS**. This is FAR
beyond the physical ~10–50× radiation cost — a **performance pathology in the
RRTMGP radiative-transfer kernel**, not normal expense. Back-of-envelope: 90 k
cols × 26 lev × ~256 g-points ≈ 6e8 cell-g-point ops should be ms-range on GPU;
10.9 s ⇒ ~5e10 elem/s effective = catastrophic under-utilization. Likely an
unvectorized Python/scan loop over g-points/bands/columns, a per-step
full-spectrum recompute that ignores `RAD_UPDATE_STEPS`, or a host sync per call.

Implication: **parallel scaling of the RRTMGP tier is moot until this kernel is
fixed** — radiation dwarfs everything else. This is the dominant atmosphere
bottleneck for full-physics AMIP. Next: profile
`packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/` to localize
the pathology (g-point/band/column iteration in `kernel_ops.py` /
`interpolation.py` / `optics/`, and whether radiation respects sub-stepping in
the segment path).

## Iteration 8 (2026-06-08): RRTMGP pathology LOCATED + fix VALIDATED

Root cause — `rte/two_stream.py` `solve_lw` (line ~404) and `solve_sw` (~618):
the radiative transfer **scans SEQUENTIALLY over all ~256 g-points**:

```python
_scan_step_ckpt = jax.checkpoint(_scan_step, prevent_cse=True,
                                 policy=jax.checkpoint_policies.nothing_saveable)
fluxes, _ = jax.lax.scan(_scan_step_ckpt, init_val, jnp.arange(optics_lib.n_gpt_lw))
```

The `jax.checkpoint(nothing_saveable, prevent_cse=True)` is a **backward-pass
memory** optimization (comment: drops ~21→~1 MiB/col so T127 fits a 24 GiB GPU
under `eqx.filter_value_and_grad`). But it (a) serializes the embarrassingly
parallel g-point axis — one tiny kernel per g-point — starving the GPU, and (b)
`prevent_cse=True` blocks XLA optimization. Critically it is applied
**unconditionally**, even though the forward benchmark builds the segment with
`gradient_checkpoint=False` — the kernel ignores that flag.

Per-g-point work is a clean **map-reduce**: `step_fn(igpt, cum)` computes that
g-point's flux via `_compute_local_properties_lw(...,igpt,...)` →
`lw_transport(...)` and adds it to a cumulative sum. The only genuinely
sequential axis is the 26 vertical levels (adding-doubling up/down the column),
which stays a scan *inside* each g-point.

### Iteration 9 (2026-06-08): fix IMPLEMENTED — 6.1× RRTMGP radiation

Added `_accumulate_over_gpoints(step_fn, n_gpt, init_val, gpoint_batch_size)` in
`rte/two_stream.py`, used by both `solve_lw` and `solve_sw`:
- `gpoint_batch_size == 0` (default): the original checkpointed scan, byte-for-byte
  unchanged — REQUIRED for reverse-mode AD / training memory.
- `> 0`: process g-points in parallel blocks of that size via `jax.vmap`
  (`flux_fn(ig) = step_fn(ig, zeros)` since step_fn is purely additive into the
  carry), out-of-range padding indices clamped to 0 and masked to zero, scanning
  over `ceil(n_gpt/bs)` blocks to bound peak memory.

Plumbed `RRTMGPConfig.gpoint_batch_size` (default 0) → `rrtmgp.py` → both solve
calls. Validation (`solve_columns`, 64 cols, fp64): batch=0 vs 32 LW+SW fluxes
match to rel ~1e-15 (summation reassociation only); non-divisible bs=48 (divides
neither n_gpt_lw=256 nor n_gpt_sw=224) matches to ~4e-13; no NaN.

GPU radiation kernel (`solve_columns`, 3456 cols × 26 lev, fp32):

| gpoint_batch_size | ms     | speedup |
|-------------------|-------:|--------:|
| 0 (scan)          | 10 259 | 1.0×    |
| 16                |  1 832 | 5.6×    |
| 32                |  1 679 | **6.1×**|
| 64                | OOM (9.97 GiB) | — (chunking necessary; validates design) |

**6.1× faster radiation** (10.3 s → 1.68 s), bit-identical forward, training path
(batch=0) untouched, memory-bounded. Codex-reviewed: no CRITICAL/HIGH; two MEDIUM
verification gaps (SW carry purely additive; no Python control flow on `igpt`)
both closed by code audit.

**float64 also improves** (same 3456-col GPU bench): scan 19 279 ms → batch=16
7 696 ms = **2.5×**. Smaller than fp32's 6.1× — consumer-GPU fp64 (~1/64 rate)
is arithmetic-bound, so parallelising g-points hides less launch overhead — and
the memory ceiling is lower: fp64 `batch=32` OOMs (10 GiB), so **batch≈16 is the
fp64 max** on 24 GiB vs ≈32 for fp32. Both precisions benefit; pick block size by
precision.

The ~10.9 s/step RRTMGP segment (iter 7) was
radiation-dominated, so end-to-end RRTMGP should drop ~6× once the forward driver
sets `gpoint_batch_size>0` (iter 10 plumbing). Remaining radiation cost is the
genuine 224+256 g-points × 26-level recurrence + optics.

### Iteration 10 (2026-06-08): plumbed to production + end-to-end confirmed

Added `ExperimentConfig.rrtmgp_gpoint_batch_size` (default 0) →
`physics_pipeline.py` + `model_driver.py` (both RRTMGPConfig sites) →
`RRTMGPConfig.gpoint_batch_size`. GPU benchmark sets it to 32 (forward-only).
End-to-end (cubed-sphere weak C24, full segment = dynamics+RRTMGP+SBM+Kessler):
**10 901 → 1 830 ms/step = 5.96×** (compile 23.0 → 15.5 s; SYPD 0.113 → 0.673).
Codex-reviewed: OK — NamedTuple field-insertion safe (all callers keyword-only),
no AD/training contamination (training builds its own config, default 0; the
benchmark's `build_segment_fn(gradient_checkpoint=False)` forward path).

**RRTMGP arc complete (iters 7→10): found 6000× slow → root-caused (sequential
g-points) → fixed kernel (6.1×) → shipped to production (5.96× end-to-end).**

---

Original validation (synthetic, GPU, 3456 col × 26 lev × 256 gpt, each g-point a
26-level sequential scan): **scan-over-gpoints 33.9 ms vs vmap-over-gpoints
1.29 ms = 26× faster**. Real RRTMGP per-g-point work is heavier ⇒ the true gain
is larger (toward closing the ~6000× gap; also removes `prevent_cse` + checkpoint
overhead).

## Iteration 13 (2026-06-08): OCEAN single-GPU throughput baseline

Atmosphere multi-rank gap reassessed: cubed-sphere already has a face-sharded
SPMD path (`parallel/sharded_dynamics.py`, `make_sharded_step`, shard_map, ≤6
faces + sub-face tiling, tested in `test_cubed_sphere_spmd_step.py`) — it targets
≥2 GPUs/TPU, so its multi-device scaling can't be wall-clock-benchmarked on this
1-GPU host (simulated CPU devices share cores). Not a code gap; hardware-blocked
here. Atmosphere has its major wins; pivot to the ocean half.

Ocean GPU throughput (`bench_ocean_gpu_scaling.py`, float32, single RTX 5090,
explicit_substep barotropic):

| grid    | size  | n_cells   | ms/step | Mcells/s |
|---------|-------|----------:|--------:|---------:|
| latlon  | LL32  |    40 960 |  1.36   |   30.1   |
| latlon  | LL64  |   163 840 |  1.76   |   93.3   |
| latlon  | LL96  |   368 640 |  1.97   |  187.0   |
| latlon  | LL128 |   655 360 |  2.56   |  256.3   |
| latlon  | LL192 | 1 474 560 |  4.67   | **315.6**|
| MPAS    | I4    |    51 240 |  2.50   |   20.5   |
| MPAS    | I5    |   204 840 |  3.43   |   59.7   |
| MPAS    | I6    |   819 240 |  7.97   | **102.8**|

Healthy curves (throughput rises with size, no decline) — both grids still below
saturation at these sizes (small sizes are launch/dispatch-bound, like the
atmosphere). Same structured>unstructured pattern: **lat-lon ~3× MPAS**
(coalesced C-grid vs gather-bound Voronoi). Ocean throughput is lower than the
atmosphere dycore (lat-lon atm HS ~1100 vs ocean ~316 Mc/s) — more work per cell
(baroclinic + barotropic substepping + EOS).

NOTE: those iter-13 numbers used the bench's `explicit_substep` default, which is
NOT the production solver — corrected in iter 14.

## Iteration 14 (2026-06-08): ocean barotropic solver — production lever found

Hunted the ocean bottleneck (RRTMGP lesson). The barotropic substeps are
GENUINELY sequential (temporal integration — unlike RRTMGP g-points, can't
parallelise). But the algorithm choice matters: `explicit_substep` iterates
`n_barotropic_substeps = 30` sequential small steps (+ a per-substep allreduce,
`eta_floor.py`); `implicit_cn` solves one Crank-Nicolson free-surface system per
baroclinic step. GPU float32:

| grid         | explicit_substep | implicit_cn   | speedup |
|--------------|-----------------:|--------------:|--------:|
| latlon LL128 | 2.56 ms / 256    | 1.79 ms / 366 | 1.41×   |
| latlon LL192 | 4.67 ms / 316    | 3.69 ms / 400 | 1.27×   |
| MPAS I5      | 3.43 ms / 60     | 1.39 ms / 147 | **2.47×** |
| MPAS I6      | 7.97 ms / 103    | 3.25 ms / 252 | **2.45×** |

`implicit_cn` is 1.3–2.5× faster (MPAS most — its 30-substep + allreduce path is
costliest) AND is the VALIDATED PRODUCTION solver: `run_omip.py` selects
`barotropic_solver="implicit_cn"` in all three production paths. So the bench's
`explicit_substep` default was *misrepresenting* production ocean performance.

Fix: `bench_ocean_gpu_scaling.py` now defaults both `--mpas-baro-solver` and
`--ll-baro-solver` to `implicit_cn`, matching `run_omip.py`. No production physics
changed (implicit_cn already exists + is the production default) → no fidelity
risk. Production-representative single-GPU ocean throughput: **lat-lon ~400 Mc/s
(LL192), MPAS ~252 Mc/s (I6)** — both healthy/rising.

## Iteration 15 (2026-06-08): ocean fp64 + MPI survey + hardware ceiling

Ocean fp64 (implicit_cn, single GPU):

| grid         | fp32 Mc/s | fp64 Mc/s | fp64 penalty |
|--------------|----------:|----------:|-------------:|
| latlon LL128 |    366    |    132    | 2.8×         |
| latlon LL192 |    400    |    152    | 2.6×         |
| MPAS I5      |    147    |     99    | 1.5×         |
| MPAS I6      |    252    |    200    | 1.26×        |

fp64 penalty is larger for lat-lon (arithmetic-heavy implicit CN solve hits the
consumer 1/64 fp64 rate harder) than MPAS (memory/gather-bound). Net reversal:
MPAS fp64 (200) > lat-lon fp64 (152), opposite of fp32. Pick the grid by precision.

Ocean MPI: uses the generic cubed-sphere distributed layout
(`parallel.distributed` scatter/gather/`make_layout`, tested in
`tests/ocean/distributed/test_ocean_mpi_conservation.py`) — the SAME multi-device
path as the atmosphere, so its multi-device scaling is hardware-blocked on this
1-GPU host (and CPU MPI is sync-barrier-bound, iters 2–3).

### Campaign status / hardware ceiling (this host: 1× RTX 5090 + 24-core CPU)

Major algorithmic wins found + shipped to main:
- **Atmosphere**: RRTMGP g-point vmap 6.1× (5.96× end-to-end), GPU XLA fatal-flag
  fix (unblocked all GPU runs), multi-rank Held-Suarez + halo refactor (MPI),
  full cross-grid single-GPU throughput map.
- **Ocean**: barotropic `implicit_cn` 1.3–2.5× lever (bench corrected to
  production), full cross-grid single-GPU map (fp32 + fp64).

Remaining gains are **hardware-bound**, not code-bound on this host:
- True multi-device strong/weak scaling (cubed-sphere face-sharding, ocean
  cubed-sphere layout) needs ≥2 GPUs — infra exists + is tested, just can't be
  wall-clock-benchmarked here.
- CPU MPI strong scaling is sync-barrier/jitter-bound on a shared node (iters
  2–3) — a non-blocking halo exchange is the only remaining CPU lever, intricate
  under mpi4jax's blocking-only JIT API.
- Closing the remaining single-GPU roofline gap (~30% HBM for structured grids)
  needs kernel fusion / higher arithmetic intensity — large, diminishing returns.

## Iteration 19 (2026-06-08): CPU-MPI strong-scaling plateau — root cause SETTLED

Tested rank-to-core binding (`mpirun --bind-to core --map-by core`) as the last
CPU strong-scaling lever. Clean A/B, same system state, icosahedral I5 HS,
single-thread/rank:

| ranks | unbound ms | --bind-to core ms |
|------:|-----------:|------------------:|
| 1     | 31.86      | 72.81             |
| 8     | 30.22      | 34.65             |

Binding is WORSE at both ends (pins a rank to one core → starves its memory
bandwidth). Unbound 1→8 speedup is **1.05×** (flat).

**Definitive cause of the plateau: single-socket memory-bandwidth saturation** —
NOT OS jitter, NOT collective latency, NOT a code bug. The icosahedral step is
memory-bandwidth-bound; at N ranks each does 1/N the work but all N contend for
the same DRAM bus on one socket, so aggregate bandwidth is fixed → no speedup.
Real strong scaling needs **multiple sockets / nodes** (more aggregate bandwidth)
— exactly what HPC clusters provide and this single node lacks. This is the
hardware-topology gap vs SOTA, confirmed empirically.

## Backlog (campaign)

1. **DONE (iter 3)**: `_exchange_mpi` one-scatter refactor — bit-identical,
   codex-clean, perf-neutral on CPU (comm is sync-barrier-bound, not copy-bound),
   kept for GPU/NCCL benefit.
2. **DONE (iter 4)**: GPU icosahedral HS throughput characterized — gather-bound,
   ~5–10 % roofline, peaks at I5; reorder gives +9 % at I7.
3. **DONE (iter 5)**: GPU cross-grid HS throughput — cubed-sphere best-behaved
   (666 Mc/s), lat-lon fastest peak (1100), icosahedral gather-bound (477),
   spectral GPU-hostile (8).
4. **DONE (iter 6)**: production GPU XLA-flag fatal-abort fixed (backend.py);
   gray atmosphere GPU throughput on cubed-sphere + lat-lon (~2.3–3.3× slower
   than HS). Gray tier confirmed.
5. **DONE (iter 7)**: RRTMGP tier — RUNS but ~6000× slower than gray
   (10.9 s/step at C24); radiative-transfer kernel is pathologically slow.
6. **DONE (iter 8)**: pathology located (`two_stream.py` sequential g-point scan
   + `prevent_cse`/checkpoint, applied even forward-only) and fix validated
   (vmap-over-g-points = 26× on GPU synthetic).
7. **DONE (iter 9)**: forward g-point fast-path implemented (chunked vmap,
   `RRTMGPConfig.gpoint_batch_size`) — 6.1× radiation on GPU, bit-identical,
   codex-clean, training path untouched.
8. **DONE (iter 10)**: plumbed `rrtmgp_gpoint_batch_size` through
   ExperimentConfig→driver→RRTMGPConfig; benchmark uses 32. End-to-end RRTMGP C24
   5.96× (10901→1830 ms/step). Codex-clean.
9. **NEXT (iter 11)**: regression-validate the accumulated working-tree changes
   (8 files, iters 1/3/6/9/10) — run radiation tests
   (`tests/atmosphere/hydrostatic/unit/test_rrtmgp_stratosphere.py`,
   `test_radiation.py`), distributed/voronoi tests
   (`tests/distributed/test_voronoi_mpi.py`, `test_voronoi_halo.py`,
   `tests/parallel/test_voronoi_sharded_equivalence.py`), and a conservation
   smoke. Confirm green before more features.
10. CPU-MPI (if revisited): non-blocking halo exchange (Irecv/Isend + Waitall) to
   collapse the 18–36 per-step blocking barriers into one; reduce local-compute
   dispatch floor (fuse MPAS tendency gather/scatter — `jax.make_jaxpr` first).
7. Harness cleanups: `run_levante_gpu_scaling.py` skip MPI load for `n_gpus==1`.
8. Domain-decompose cubed-sphere (currently replicated); localize lat-lon C-grid
   operators (#115) for multi-rank.
9. Ocean: repeat for all grids.
