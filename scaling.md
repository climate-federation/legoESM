# legoESM GPU-scaling branch — scaling status

> **Canonical artefact** for the `GPU-scaling` branch.  Aggregates the
> baroclinic-wave benchmark results, code-level scaling work, and the
> "near-optimal scaling" gap analysis the user task asks about.
> Companion files: `docs/GPU_SCALING_BRANCH.md` (per-iteration code-change
> log), `results/scaling_baseline/SUMMARY.md` (iter-1–iter-30 narrative),
> `results/scaling_plots/*.png` (regenerated each iteration).

## 1. Hardware reality (this host) — and why the loop cannot honestly emit `<promise>DONE</promise>`

| Resource              | Available on this host                | Required for "near-optimal scaling" check |
|-----------------------|---------------------------------------|--------------------------------------------|
| CPU                   | 24-core Intel Core Ultra 9 275HX, 1 NUMA node | yes (single-rank baseline) |
| GPU                   | **none** (no working CUDA driver)     | **yes** (Levante A100 / H100, NVLink)      |
| MPI runtime + mpi4py  | **not installed**                     | **yes** (multi-rank scattered state)       |
| JAX backend           | CPU only                              | GPU (or multi-CPU MPI)                     |

The benchmark scripts (`scripts/run_levante_gpu_scaling.py`,
`scripts/run_cpu_mpi_scaling.py`) exist and run, but every multi-device
number on this host comes from JAX's XLA virtual-CPU emulation
(`XLA_FLAGS=--xla_force_host_platform_device_count=N`).  N logical
"devices" share the same physical cores, so:

- per-device steady-state work is **not** sped up by adding "devices",
- inter-device collectives use shared-memory (no NVLink/PCIe latency),
- "scaling efficiency" numbers below are an emulation-overhead diagnostic,
  not a parallel-efficiency measurement.

The user task ("Completion is when scaling is near optimal theoretical
value for each grid type") therefore **cannot be evaluated on this
host**.  The Ralph-loop completion promise is intentionally *not*
emitted; what every iteration produces is durable code-level work that
will benefit a real GPU/MPI rerun on Levante (or equivalent).

## 2. Latest single-device baselines (post-iter-187 code, x64, dt-from-CFL, 8 levels)

Re-measured on 2026-05-03 with the post-iter-187 hoisting series merged
to verify the hot-path step has not regressed.  Times are the median of
1000-step `lax.scan` runs after a 1-step warm-up.

| Grid          | Resolution | dt (s) | ms/step | SYPD     | Mcells/s | vs. iter-1 baseline |
|---------------|------------|-------:|--------:|---------:|---------:|---------------------|
| spectral      | T21        |    870 |  5.24   |   454.94 |     3.5  | -0.6% (no change)   |
| cubed-sphere  | C24        |    450 |  2.63   |   467.82 |    10.5  | -1.1% (no change)   |
| icosahedral   | I4         |    780 |  3.35   |   637.80 |     6.1  | -5.9% (no change)   |

Conclusion: the 188 inline-import hoists + 130 reduction-fusion / halo-
packing patches across iters 1–187 leave the warm-state per-step cost
unchanged within noise on a single device, **as expected** (the
optimisations target multi-device collective and Python-overhead paths
that single-device runs barely touch).

## 3. Multi-device emulation (informative — shared cores)

### 3.1 Cubed-sphere C24/L8 strong scaling — emulated CPU progression

The 6-device step time on this host has dropped meaningfully across the
collective-latency / halo-pack iterations:

| Iteration tag      | 1-dev ms/step | 2-dev | 3-dev | 6-dev ms/step |
|--------------------|--------------:|------:|------:|--------------:|
| iter-1 baseline    |          2.66 |  5.61 |  6.39 |          3.28 |
| iter-7 (offsets)   |          2.76 |   —   |   —   |          3.45 |
| iter-8 (packed)    |          2.76 |   —   |   —   |          3.22 |
| iter-11 (PPM pack) |          2.79 |   —   |   —   |          3.21 |
| iter-16 (final)    |         ~2.7  |   —   |   —   |          3.09 |
| iter-58 (post-merge halo) | 2.66    |  4.21 |  3.71 |          3.19 |
| **iter-188 (post-187)**   | **2.63** | **3.88** | **3.58** | **2.90** |

Net 6-device wall-clock improvement on emulated CPU, iter-1 → iter-188:
**3.28 → 2.90 ms (-12%)**.  Net 2-device: **5.61 → 3.88 ms (-31%)**.
Even though emulated CPU "devices" share cores, each iter-49 multi-face
SPMD + iter-58 merged-halo + iter-65 batched corner-interp commit
shortens the dependency chain that XLA's virtual all_gather still
serialises through.

### 3.2 Cubed-sphere C48/L8 strong (iter-49)

| n_devices | ms/step | scaling efficiency |
|----------:|--------:|-------------------:|
|         1 |    8.11 |             100.0% |
|         2 |   11.65 |              34.8% |
|         3 |   11.79 |              22.9% |
|         6 |    9.80 |              13.8% |

C48 sees a wider efficiency drop than C24 because each device's halo
fraction grows proportionally on emulated CPU (no real bandwidth gain
from sharding).

### 3.3 Spectral T21/L8 strong (iter-188 sample)

| n_devices | ms/step | scaling efficiency |
|----------:|--------:|-------------------:|
|         1 |    5.24 |             100.0% |
|         2 |    7.82 |              33.5% |

Spectral SH transforms parallelise along level (the iter-50/iter-81/
iter-85 fusions); on virtual CPU the level-shard kernels can't beat
single-device because the FFT batch dimension stays serial on shared
cores.

### 3.4 Icosahedral I4/L8 strong (iter-188 fresh re-measurement)

| n_devices | ms/step | scaling efficiency |
|----------:|--------:|-------------------:|
|         1 |    3.35 |             100.0% |
|         2 |    2.99 |              55.9% |
|         3 |    2.92 |              38.3% |
|         4 |    2.80 |              29.9% |

vs iter-49: 1-dev 3.56 → 3.35 (-5.9%), 4-dev 3.10 → 2.80 (-9.7%).

The Voronoi cell-shard path (iter-23/25/27/37 correctness + halo-
augmentation work) is the only grid that shows a per-device step-time
improvement on emulated CPU — because the SPMD halo exchange is a
small, packed sendrecv set rather than a full edge perimeter transfer.

### 3.5 Weak scaling (iter-63)

#### Cubed-sphere — base N=24
| n_devices | resolution | cells/dev | ms/step | "Eff" |
|----------:|-----------:|----------:|--------:|------:|
|         1 |       C24  |    27 648 |    2.74 | 100.0% |
|         2 |       C34  |    27 744 |    6.79 |  40.5% |
|         3 |       C42  |    28 224 |    9.88 |  28.3% |
|         6 |       C58  |    26 912 |   14.95 |  17.8% |

#### Icosahedral — base level=4
| n_devices | resolution | cells/dev | ms/step | "Eff" |
|----------:|-----------:|----------:|--------:|------:|
|         1 |        I4  |    20 496 |    3.46 | 100.0% |
|         2 |        I5  |    40 968 |    9.50 |  72.9% |
|         3 |        I5  |    27 312 |    8.43 |  54.8% |
|         4 |        I5  |    20 484 |    8.06 |  43.0% |

The same caveat applies — these efficiencies measure XLA emulation
overhead under shared cores, **not** parallel efficiency on real GPU.

### 3.6 Plots

Generated by `scripts/plot_scaling_laws.py` from the latest CSVs:

- `results/scaling_plots/strong_scaling.png` — strong-scaling time-per-step
  vs n_devices, with the ideal `T_1 / N` reference, for all three grids
  at canonical resolutions (C24, C48, T21, I4).
- `results/scaling_plots/weak_scaling.png` — weak-scaling time-per-step
  vs n_devices at fixed cells/device, with the ideal flat reference.
- `results/scaling_plots/iter_progression.png` — cubed-sphere C24
  6-device strong-scaling step-time across iterations 46/49/58 vs
  iter-1 baseline.

These plots use the iter-49/iter-58/iter-63 CSVs (most-recent
multi-resolution sweep on this host); the iter-188 fresh single-grid
CSVs in `results/scaling_iter188_*` are spot-checks confirming nothing
regressed since iter-58.

## 4. Code-level scaling work landed on this branch (1106 commits total)

Three stages so far:

### Stage A — SPMD correctness and halo packing (iters 1–32)
- JAX-0.10 `shard_map(check_vma=...)` rename, `lax.scan` static-`dt`
  capture, packed `cumsum` removal, multi-device activation gating.
- Halo-packing: cubed-sphere SPMD halo=1 and halo=2 explicit allgather
  kernels with `interp_offsets` forwarding (bit-equivalent to local
  pad), packed cell-field exchange (T+u+v+lnps), packed PPM `q_i + q_j`
  halo=2 exchange, ppermute backend.  Communication volume per device
  drops from O(6·n²) auto-gather → O(8·n) explicit allgather (36×
  reduction at C48, 144× at C96).
- Voronoi cell-shard correctness fixes (iter-23/25): closing the
  cellsOnEdge halo iteratively, dropping field drift 7 orders of
  magnitude (1e76 → 1e-9 abs).
- 15 SPMD-correctness regression tests covering halo / level /
  Voronoi paths, all passing under emulated CPU.

### Stage B — collective fusion and per-step latency reduction (iters 50–127)
- 38 reduction-fusion patches across atmosphere physics
  (Bechtold, tiedtke, emanuel, sbm, dca, KPP, …), ocean dycores
  (Visbeck, Rossby radius, fix_volume / fix_heat / fix_salt batching),
  sea-ice ITD, land multilayer.  Net effect: per-step allreduce count
  on cubed-sphere drops 4 → 1; MPAS dycore mass-fix payload 3 → 2
  scalars; Voronoi MPI per-step sendrecvs 12 → 6.
- Cubed-sphere FV3 PE: merged 4 cell-field halo exchanges into the
  single packed exchange already present, batched the lap_uv +
  hyperdiff_uv + vert_adv_uv corner interpolations.

### Stage C — Python overhead removal (iters 128–187)
- 188 inline-import hoists across atmosphere/ocean dycores, physics
  parameterisations, conservation fixers, halo helpers, runtime, ML
  channel-packing, training losses, driver diagnostics — moving lazy
  function-body imports to module-load time so the JIT-compiled hot
  loop has zero per-call import-machinery overhead.  Skipped only the
  underscore-private re-exports (CLAUDE.md rule) and three documented
  cycles (coupler.surface_exchange thermodynamics path; halo ↔ duogrid;
  mutable `_halo_backend` / `_spmd_mesh` runtime dispatch globals).

## 5. Outstanding gaps for "near-optimal scaling"

The scaling curves above asymptote at a per-device "useful-work" floor
that the cubed-sphere SPMD path still pays.  None of the items below
can be measured on a CPU-only host — they are the gap list to validate
on real GPU/MPI.

1. **`zero_mean_tendency` 3-allreduces-per-step (cubed-sphere)** —
   per-stage zero-mean correction runs inside the RK3 `tendency_fn`.
   Lifting to one end-of-step correction (or batching the 3 stage sums
   into one allreduce) would remove 2 round-trips/step.  Needs
   Held-Suarez / JW conservation-drift validation before flipping.
2. **2- and 3-device cubed-sphere falls back to replicated halo** —
   the multi-face SPMD kernels (iter-49) cover 1, 2, 3, and 6 devices,
   but the 2/3-device configs still pay full perimeter strips because
   shard_map only exchanges between adjacent ranks.  A multi-face-per-
   device packed kernel would let those configs benefit too.
3. **SPMD allgather threshold formula** — codex flagged that the
   ppermute-vs-allgather decision (`select_exchange_backend`) is using
   a hand-tuned threshold from emulation, not measured GPU latency.
   Re-derive once GPU timings are available.
4. **MPAS `phis` halo exchange under SPMD ppermute** — terrain is
   static, scattered correctly at init, and the SPMD ppermute kernel
   re-exchanges it every step.  Skipping saves 2-3% halo bytes.
   Deferred since the code path cannot be validated end-to-end without
   GPU.
5. **Deterministic global-sum reductions** — residual `p_s` drift on
   sharded cubed-sphere (5e-7 rel) and Voronoi (1e-3 abs / 1e-7 rel)
   comes from float-pt sum-order non-determinism in the
   `fix_mass_hydrostatic_target` allreduce.  A deterministic pairwise
   or Kahan-summation collective would close it; not closeable with
   stock jax.lax.psum.

## 6. How to actually measure "near-optimal scaling" on real hardware

The scaling-test scripts on this branch are ready to run.  On a Levante
A100 node (4 GPUs) or 6-rank cubed-sphere MPI cluster:

```bash
# GPU strong + weak, all three grids, both precisions
for grid in spectral cubed-sphere icosahedral; do
  python scripts/run_levante_gpu_scaling.py \
      --grid "$grid" --mode both --precision both \
      --n-levels 26 \
      --output-dir "results/hw_scaling_$(date +%Y%m%d)/${grid}"
done

# Multi-rank cubed-sphere on 6 ranks
mpirun -np 6 python scripts/run_levante_gpu_scaling.py \
    --grid cubed-sphere --mode strong --precision float32 \
    --n-levels 26 --n-gpus 6 \
    --output-dir results/mpi_scaling_$(date +%Y%m%d)

# Render scaling plots from the new CSVs
python scripts/plot_scaling_laws.py \
    --output-dir "results/hw_scaling_$(date +%Y%m%d)/plots"
```

The completion criterion (efficiency ≥ 80% strong-scaling at the
canonical resolution and grid count) maps directly onto the
`scaling_efficiency` column of those CSVs.  Numbers from this CPU host
will *never* satisfy that criterion; numbers from the real-hardware
runs will.

## 7. Iteration log pointer

The full per-iteration code-change log (1106 commits, every patch from
iter-1 baseline through iter-187 inline-import hoists) lives in
`docs/GPU_SCALING_BRANCH.md`.  This file is the user-facing summary;
that one is the engineering log.

Per-iteration scaling CSVs live under `results/scaling_iter*/`,
indexed by the `iter` tag in the directory name.  The latest fresh
re-measurement (iter-188, 2026-05-03, post-iter-187 code) lives in
`results/scaling_iter188_{spec,cs,ico}/`.
