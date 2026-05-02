# GPU-scaling branch overview

This document captures the work on the `GPU-scaling` branch — a 25-commit
sequence focused on multi-device numerical correctness and per-step
collective-count reductions in the SPMD execution paths of the three
atmospheric dycores (cubed-sphere FV3 PE, spectral PE, MPAS / Voronoi PE).

## Why the branch exists

The branch was started to evaluate `run_baroclinic_wave_benchmark.py`
weak / strong scaling across all three grid types on multi-CPU and GPU
hardware.  The host on which the iterations ran has no GPU and no MPI,
so end-to-end "near-optimal" scaling cannot be measured here.  Instead
the branch focuses on the code-level work that benefits any future
GPU/MPI run:

1. Unblocking the multi-device path under JAX 0.10 (4 trace-time
   bugs that prevented any sharded run from compiling).
2. Reducing per-step collective count and bandwidth on the
   cubed-sphere SPMD path.
3. Closing the SPMD-vs-single-device numerical gap on all three
   grids (cubed-sphere, spectral, Voronoi).
4. Surfacing and fixing two pre-existing correctness bugs in the
   Voronoi sharded path that had been silently producing wrong
   numerics (the previous "successful" multi-device benchmark output
   was actually garbage — measured u → 1e76 after one step).
5. Codifying the work in 15 SPMD-correctness unit tests so future
   changes are caught.

Real-hardware verification still requires running on a GPU / MPI
cluster per `docs/REAL_HARDWARE_SCALING.md`.

## Per-RK3-step collective and bandwidth reductions

| Path                                               | Pre-branch | Post-branch |
|----------------------------------------------------|-----------:|------------:|
| Cubed-sphere allreduces                            |          4 |           1 |
| Cubed-sphere extra SPMD halo from `fv3_to_hydrostatic` |     1 |           0 |
| Cubed-sphere SPMD halo=1 cell-field collectives    |          4 |           1 |
| Cubed-sphere SPMD halo=2 bandwidth                 |   ~6n²·#h2 |    ~8n·#h2 (36×–144× less) |
| Cubed-sphere SPMD halo=1 with offsets at high-res  |  all_gather|    ppermute (~50× less interconnect) |
| Cubed-sphere SPMD halo at 2/3 devices              | replicated-fallback | multi-face all_gather (31–44% lower per-step time) |
| Spectral PE σ-coord cross-level collectives        |          3 |           1 |
| Hybrid mass-flux + dp_s_dt cross-level collectives (CS, spectral) | 4 (2 in fn + 2 in callers) | 2 |
| compute_sigma_dot internal cross-level collectives  |          2 |           1 |
| Cubed-sphere FV3 σ-coord σ̇ + dp_s_dt cross-level collectives | 3 | 1 |
| MPAS σ-coord σ̇ + dp_s_dt cross-cell-shard reductions |       2 |           1 |
| latlon C-grid PE σ̇/mass-flux + dp_s_dt cross-shard reductions (per branch) | 2 | 1 |
| Cubed-sphere PPM halo=2 calls (SPMD or MPI)        |          3 |           2 |
| MPAS dycore mass-fix payload                       |          3 |           2 |
| Voronoi MPI sendrecvs                              |         12 |           6 |
| Voronoi MPI mass-fix payload                       |          3 |           2 |
| Voronoi SPMD mass-fix HLOs                         |          2 |           1 |

## SPMD-vs-single-device numerical equivalence

| Grid          | Sharding axis        | Equivalence achieved (1 SSP-RK3 step) |
|---------------|----------------------|----------------------------------------|
| Cubed-sphere  | face (6 devices)     | u/v/T at FMA precision (1e-14); p_s 5e-7 rel |
| Spectral      | level (2 or 4 dev)   | all fields bit-equivalent at 1e-12 |
| Voronoi/MPAS  | cell (2, 3, or 4 dev)| u/T 1e-8 abs; p_s 1e-3 abs (1e-7 rel) |

The residual `p_s` drifts on cubed-sphere and Voronoi are the
post-step `fix_mass_hydrostatic_target` allreduce float-point sum-order
drift — fundamental to sharded reductions, not closeable without a
deterministic-reduction collective.

## Pre-existing bugs surfaced and fixed

- **JAX 0.10 trace-time bugs** (commit `32cfeb40`): four scattered
  bugs blocked the multi-device cubed-sphere / Voronoi / spectral
  paths from compiling under JAX 0.10:
    * `run_levante_gpu_scaling.py:_make_scan_runner` passed `dt` as a
      `lax.scan` tracer; `SpectralPrimitiveEquationModel.step` does
      Python `==` comparisons on `dt`, requiring a concrete value.
    * `shard_map(check_rep=...)` was renamed `check_vma=...` in JAX 0.10.
    * `cubesphere_exchange.packed_pad_halo_4d` passed
      `jnp.cumsum(jnp.array(...))` (traced) to `jnp.split` despite a
      docstring warning about that exact mistake.
    * `make_sharded_step` activated SPMD halo for `n_devices ≤ 6` but
      the kernels assumed exactly 1 face per device; 2- or 3-device
      configs silently dropped faces.

- **Voronoi SPMD `cellsOnEdge` -1 remap bug** (commit `4d4a439a`):
  `_build_voronoi_partition_infra` used `|` on the halo-edge filter,
  including any edge with at least one neighbour cell in
  `local_cells`.  The other neighbour was remapped to `-1` by
  `cell_g2l`, and `gradient_edge(phi, mesh)` did `phi[c1=-1]` —
  Python's last-element indexing — producing meaningless gradients
  → `u → 5.94e+76` after one step.  Fix: AND-edge-halo + augment
  cell halo with owned-edge other-cells + bump `halo_depth` 2→3.

- **Voronoi SPMD halo not closed under cellsOnEdge** (commit
  `632a2bdf`): residual ~1% drift after iter-23 because halo cells
  lacked some of their adjacent edges.  Iter-25 added an iterative
  augmentation that pulls in the OTHER cell of every halo edge until
  the halo is closed; drift drops 7 orders of magnitude (u: 0.97 →
  1.4e-08).

- **Cubed-sphere SPMD `pad_halo_vector_4d` dropped offsets** (commit
  `9d1b84e0`): the SPMD vector halo entry point silently dropped
  `interp_offsets`.  `divergence_3d` (called by `hyperdiffusion_3d`)
  consequently produced gradients without the per-edge Lagrange
  correction.  Forwarding offsets through the SPMD vector kernel
  closes the cubed-sphere bit-equivalence gap (u/v/T at FMA precision).

## Test coverage (15 SPMD-correctness tests)

| File                                                    | # tests | What |
|---------------------------------------------------------|--------:|------|
| `tests/parallel/test_cubesphere_exchange.py::TestSPMDWithOffsets` | 8 | halo bit-equivalence |
| `tests/parallel/test_cubed_sphere_spmd_step.py`          |       6 | end-to-end cubed-sphere SPMD step (1 + 10 RK3 steps; with-physics; C48; 2- and 3-device multi-face) |
| `tests/parallel/test_spectral_level_shard.py`            |       2 | level-shard equivalence (2 and 4 devices) |
| `tests/parallel/test_voronoi_sharded_equivalence.py`     |       3 | cell-shard equivalence (2, 3, 4 devices) |

Run with the appropriate `XLA_FLAGS=--xla_force_host_platform_device_count=N`.

## Outstanding follow-ups (not addressed in this branch)

- **Multi-face ppermute**: the all_gather-based SPMD halo kernels
  (halo=1 and halo=2) now support multi-face shards
  (`n_faces_per_shard ∈ {1, 2, 3, 6}`, iter-49); the ppermute kernel
  still assumes exactly one face per shard, so 2- and 3-device
  configurations route through the all_gather kernel.  At small face
  counts that's negligible — the all_gather payload is a constant
  multiple of the ppermute payload up to the 6-device limit — so this
  is low priority.

- **MPI-side `interp_offsets` plumbing**: `pad_halo_mpi_4d` raises
  `NotImplementedError` on `interp_offsets`.  Single-face-panel and
  testing paths still go through the local `_pad_halo_local_h2`
  fallback under MPI; the production MPI path uses `duogrid` (which
  already handles its own correction post-exchange) so this is low
  priority.

- **Deterministic-reduction allreduce**: the `p_s` 5e-7 rel residual
  drift in cubed-sphere SPMD and the `p_s` 1e-7 rel residual in
  Voronoi SPMD come from XLA's allreduce sum-order; a
  reproducible-reduction primitive would close the last gap but is a
  JAX-level change.

## Commit log

```
90c5cd27 latlon C-grid PE: reuse cumsum across dp_s_dt + sigma_dot/mass_flux
68d3f8be docs: record iter-53 MPAS σ-coord cumsum reuse
6f2dd430 mpas dycore: reuse cumsum across dp_s_dt + sigma_dot
8331acbc docs: record iter-52 compute_sigma_dot collective reductions
181e65de vertical: add compute_sigma_dot_and_total; cubed-sphere FV3 PE reuses both
5b69b9f6 vertical: drop redundant column-sum in compute_sigma_dot
13a94241 docs: record iter-51 hybrid mass-flux collective reduction
bb3ebd9c hybrid mass flux: return D_total_p; reuse in cubed-sphere + spectral PE
715045c5 docs: record iter-50 spectral PE column-sum collective reduction
da3b87f8 spectral PE: drop redundant column-sum collectives in non-hybrid path
9cb136f1 docs: record iter-49 multi-face SPMD halo scaling impact
51abec8a cubesphere_exchange: force all_gather backend at n_devices != 6
54b09ccd cubesphere_exchange: multi-face SPMD halo kernels (allgather, allgather_h2)
eb7ba319 tests: cubed-sphere SPMD bit-equiv at C48
522dffe8 tests: cubed-sphere SPMD with Held-Suarez physics bit-equiv
570c51aa tests: extend spectral level-shard equivalence to T42
92c177f9 refactor: extract _close_halo_under_cellsOnEdge helper
63e75704 tests: extend voronoi sharded equivalence to subdivision_level=5
54902899 voronoi sharded: tighten iter-25 augmentation cap from 4 to 2 passes
dc53e6a8 docs: GPU-scaling branch overview
863fb639 tests: parametrise cubed-sphere SPMD bit-equiv test over n_steps
9d1b84e0 fix cubed-sphere SPMD: forward interp_offsets through pad_halo_vector_4d
459e1599 tests: cubed-sphere SPMD step end-to-end physical-envelope test
8c6f2d11 tests: extend spectral level-shard equivalence to 4-device coverage
2f27388a tests: extend voronoi sharded equivalence to 2/3/4-device coverage
56d34257 voronoi sharded: vectorise iter-25 halo-augmentation loop
632a2bdf voronoi sharded: tighten halo to bit-equivalence with single-device
4d4a439a fix Voronoi sharded path: stop edge-cell remap from producing -1 indices
180b90b3 tests + docs: surface pre-existing Voronoi sharded correctness issue
85aaa115 tests: spectral PE level-shard equivalence (single vs 2-device)
a7f4de67 cubesphere_exchange: ppermute kernel applies interp_offsets too
df2659e1 tests: pad_halo_pair_h2 backend dispatch coverage
c9a5a09d pad_halo_pair_h2: extend MPI fallback to packed_pad_halo_mpi_4d
246b96a0 PPM transport + halo: pack q_i / q_j halo=2 exchange under SPMD
3f044e62 cubesphere_exchange: generalize packed_pad_halo_4d to halo=2
0d0100fe tests: SPMD halo exchange with interp_offsets bit-equivalence
b3b0bef4 cd-grid SPMD: pack the halo=1 cell-field exchange like the MPI path
2ba9f280 cubed-sphere SPMD: forward interp_offsets through halo=1 / halo=2 / packed
6254a55a cubesphere_exchange: implement SPMD halo=2 via 2-strip all_gather
b35142ab cubesphere_exchange: remove dead halo=2 SPMD allgather build
04b20276 mpas dycore: drop constant total_area from per-step mass-fix reduction
d0cde112 cd-grid PE: skip fv3_to_hydrostatic in conservation fixer
0960c28b scaling-bench: drop redundant per-stage zero_mean_ps_tendency
83fe8ddc scaling: apply codex adversarial-review fixes (round 1)
32cfeb40 fix scaling-script and SPMD halo bugs blocking multi-device runs
```
