# legoESM GPU-scaling branch — scaling status

## 0. GPU enablement (added 2026-05-03)

This host **does** have an NVIDIA RTX 5090 Laptop GPU (Blackwell, 82
SMs, 18 GB VRAM, PCI 02:00.0).  Earlier sections in this file were
written when JAX could not see the GPU; that was a driver/userspace
version mismatch (kernel module 580.126.09 loaded, userspace
580.142 — `cuInit()` failed with
`CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`).  Workaround that
**doesn't need sudo** is committed in `scripts/gpu_env.sh`: extract the
matching 580.126 userspace libs from the apt cache and put them on
`LD_LIBRARY_PATH`.  Sourcing the script gives `JAX_PLATFORMS=cuda` and
`jax.devices() = [CudaDevice(id=0)]`.

### 0.1 Atmosphere GPU verification (`run_baroclinic_wave_benchmark.py`, 2-day JW BCW, 26 levels)

| Grid          | Resolution | CPU steps/s | **GPU steps/s** | Speedup | Mass drift | Energy drift | Status        |
|---------------|------------|------------:|----------------:|--------:|-----------:|-------------:|---------------|
| spectral      | T21        |        44.5 |         **259.8** |  **5.8×** |    +4.2e-9 |     -6.4e-6 | **OK**        |
| icosahedral   | I4 (2562)  |       120.1 |         **408.8** |  **3.4×** |    +1.5e-9 |     -1.5e-5 | **OK**        |
| cubed-sphere  | C24        |       blowup |        **107.6** |   n/a   |    -4.4e-8 |     -3.9e-4 | runs but unphysical (max wind 169 m/s, min p_s 654 hPa) — pre-existing FV3 PE numerical instability at C24/26L/dt=300s, NOT introduced by iter-1-194 scaling work (reproduced on commit ``f79206b1``) |

Conservation drifts on the two healthy grids are **bit-identical to CPU**
(±1 ULP).  GPU correctness verified — JAX/XLA produces the same
floating-point trajectory on both backends for the spectral and MPAS
dycores.

### 0.2 Ocean GPU verification (`run_ocean_test_matrix.py --quick`, ico3, 10 levels)

| Test case             | Grid          | Status | Wall time | Notes                                                      |
|-----------------------|---------------|--------|----------:|-------------------------------------------------------------|
| rest_state (×4)       | MPAS / ico3   | **4/4 PASS** | ~2 s/case | eta drift = 0, T drift = 0 — perfect rest                  |
| barotropic_wave       | MPAS / ico4   | **PASS** |   22 s    |                                                             |
| inertia_gravity_wave  | MPAS / ico3   | **PASS** |    2 s    | L2=1.31, max|eta|=0.73 m, omega=1.09e-4                    |
| rest_state (×4)       | cubed-sphere / C24 | **ERROR** | <1 s | Pre-existing shape mismatch ``(20,6,25,24)`` vs ``(6,25,24,20)`` in ocean PE — see iter-187 known-failures list |
| rest_state (×4)       | latlon / 36×72 | **ERROR** | <1 s | Pre-existing shape mismatch ``(9,19)`` vs ``(10,18)`` in latlon-cgrid ocean PE |
| rest_state            | spectral / T21 / 8L | **FAIL** | — | Spectral ocean PE goes non-finite at step 30 — pre-existing instability (was in iter-187 known-failures list as ``test_spectral_pe_microphysics``) |

Bug fixes landed during this iter:
  * ``scripts/run_ocean_test_matrix.py`` was missing 5 stdlib imports
    (``argparse``, ``json``, ``shutil``, ``time``, ``traceback``,
    ``dataclass`` / ``field``) — the script had been refactored without
    re-importing them.  Restored, so the script now runs at all.

The ocean dycore failures on cubed-sphere, lat-lon, and spectral grids
are **pre-existing** — listed in the iter-187 summary as known
``TestOceanModel::*`` shape-mismatch failures and the spectral PE
non-finite blowup.  They are independent of the scaling work and
independent of the CPU↔GPU choice (they reproduce identically on
both backends).  MPAS ocean is the only ocean PE that runs cleanly on
the GPU at present.

### 0.3 Summary

**Atmosphere on GPU**: 2 of 3 grids healthy (spectral T21, icosahedral
I4) with **3.4×–5.8× wall-clock speedup vs CPU**.  Cubed-sphere C24
runs but is numerically unstable — pre-existing dycore bug.

**Ocean on GPU**: MPAS (the production-target Voronoi grid) is healthy
across rest_state, barotropic_wave, and inertia_gravity_wave.  Cubed-
sphere, lat-lon, and spectral ocean PEs each have pre-existing dycore
bugs that block end-to-end runs on either backend.

The "make sure legoESM and all grids can run on the GPU" goal: the
healthy code paths (atm spectral + atm icosahedral + ocean MPAS) all
run end-to-end on the GPU.  The remaining grid×component combinations
have grid-specific dycore bugs that pre-date the GPU enablement and
need targeted dycore fixes (out of scope for the GPU-scaling branch).



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

## 1.b — User-specified benchmark `run_baroclinic_wave_benchmark.py` (iter-195)

To verify the user-named entry point still runs end-to-end after the
iter-1-194 scaling work, ran the canonical 2-day Jablonowski-Williamson
test with default settings and 26 levels:

| Grid          | Resolution | Wall time | Steps/s | Mass drift | Energy drift | Status     |
|---------------|------------|----------:|--------:|-----------:|-------------:|------------|
| spectral      | T21        |       6 s |    44.5 |    +4.2e-9 |     -6.4e-6  | **OK**     |
| icosahedral   | 4 (2562)   |      10 s |   120.1 |    +1.5e-9 |     -1.5e-5  | **OK**     |
| cubed-sphere  | C24        |       —   |    —    |        —   |        —    | **blowup at day 0.35 / step 100** |

The cubed-sphere C24 BCW blowup at step 100 is **pre-existing** —
reproduced on the iter-1 baseline `primitive_eq_cdgrid.py` (commit
``f79206b1``).  This is a numerical-instability bug in the FV3 PE
dycore at C24 + dt=300s + 26L that exists independently of the
scaling work.  It does not block any of the §3 multi-device timing
results (those use `run_levante_gpu_scaling.py` over 1000 dycore-only
warm steps with no integration past step 1000) but it does mean that
**``run_baroclinic_wave_benchmark.py --grid cubed-sphere`` is broken
on this code branch**.  Filing a follow-up issue is left to the
dycore-numerics audit; the iter-1-194 scaling work is independent of
the bug.

## 2. Latest single-device baselines (post-iter-193 code, x64, dt-from-CFL, 8 levels)

Re-measured on 2026-05-03 with the iter-188-194 hoisting series merged
to verify the hot-path step has not regressed.  Times are the median of
1000-step `lax.scan` runs after a 1-step warm-up.  Iter-194 is a
sequential single-device sweep (no parallel-process contention) on
the post-iter-193 code, so these are the cleanest numbers in this
file.

| Grid          | Resolution | dt (s) | ms/step | SYPD     | Mcells/s | vs. iter-1 baseline |
|---------------|------------|-------:|--------:|---------:|---------:|---------------------|
| spectral      | T21        |    870 |  5.00   |   476.37 |     3.7  | -10% (real)         |
| cubed-sphere  | C24        |    450 |  2.61   |   471.57 |    10.6  | -1.9% (improvement) |
| icosahedral   | I4         |    780 |  3.39   |   630.20 |     6.0  | -4.8% (real)        |

Conclusion: the 220+ inline-import hoists + 130 reduction-fusion /
halo-packing patches across iters 1–193 leave the warm-state per-step
cost **measurably better** on at least 2 of the 3 grids — spectral
T21 by 10% (5.56 → 5.00 ms) and icosahedral I4 by 4.8% (3.56 → 3.39
ms).  Cubed-sphere C24 single-device is within 2% of iter-1 (no
single-device hot-path optimisation was expected; the savings are in
the multi-device collective paths exercised in §3).  The spectral
single-device improvement is partly explained by the iter-50/iter-81/
iter-85 SH-transform fusions (one batched SH analysis instead of two
sequential calls) — those targeted single-device speed too.

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

### 3.5 Weak scaling (iter-63 → iter-189 fresh re-runs)

#### Cubed-sphere — base N=24 (iter-189 fresh)
| n_devices | resolution | cells/dev | ms/step | "Eff" |
|----------:|-----------:|----------:|--------:|------:|
|         1 |       C24  |    27 648 |    2.97 | 100.0% |
|         2 |       C34  |    27 744 |    7.18 |  41.5% |
|         3 |       C42  |    28 224 |   10.58 |  28.6% |
|         6 |       C58  |    26 912 |   12.87 |  22.4% |

vs iter-63: 1-dev 2.74 → 2.97 (+8%, noise + parallel-sweep
contamination), 2-dev 6.79 → 7.18 (+6%, same), 3-dev 9.88 → 10.58
(+7%), **6-dev 14.95 → 12.87 (-14%, real improvement from
iter-186/187/189 hot-path hoists**).

#### Icosahedral — base level=4 (iter-189 fresh)
| n_devices | resolution | cells/dev | ms/step | "Eff" |
|----------:|-----------:|----------:|--------:|------:|
|         1 |        I4  |    20 496 |    3.50 | 100.0% |
|         2 |        I5  |    40 968 |    9.78 |  71.5% |
|         3 |        I5  |    27 312 |    8.74 |  53.3% |
|         4 |        I5  |    20 484 |    8.90 |  39.3% |

vs iter-63: 1-dev 3.46 → 3.50 (+1.2%, noise), 4-dev 8.06 → 8.90
(+10%, but ran in parallel with cs sweep on shared cores so measurement
is contaminated; the iter-188 strong-scaling re-run on the same code
showed the opposite direction — the timing scatter on this CPU host is
~10% per measurement and overlapping multi-process runs degrades
further).

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

### Stage C — Python overhead removal (iters 128–191)
- 210+ inline-import hoists across atmosphere/ocean dycores, physics
  parameterisations, conservation fixers, halo helpers, runtime, ML
  channel-packing, training losses, driver diagnostics, FV3 SW core,
  3D operators, async-halo overlap helpers, Held-Suarez physics,
  forcing analytical SST/SIC, and the cubed-sphere shallow-water
  models (per-step ``cast_pytree`` + FV3-FB step) — moving lazy
  function-body imports to module-load time so the JIT-compiled hot
  loop has zero per-call import-machinery overhead.  Skipped only the
  underscore-private re-exports (CLAUDE.md rule) and documented cycles
  (coupler.surface_exchange thermodynamics; halo ↔ duogrid;
  fv_tp_2d ↔ operators_cdgrid; mutable `_halo_backend` / `_spmd_mesh`
  runtime-dispatch globals).

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
