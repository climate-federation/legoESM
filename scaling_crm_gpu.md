# CRM GPU scaling — legoESM

Track single-GPU scaling for the plane CRM (compressible Euler, f-plane).
Mirror of `scaling_gpu.md` but for the LES/CRM regime (typical dx=2 km,
nlev≈30, dt~2 s).

## Hardware

- Mobile RTX 5090, 24 GB GDDR7, ~730 GB/s sustained HBM (iter-8 of prior study)
- CUDA 13.0

## Reused infrastructure

- `scripts/run_levante_gpu_scaling.py` — `TimingResult`, `write_csv`, `write_json`, `_configure_jax`
- `bench_plane_crm_dd_scaling.py` MPI bench — same f-plane dycore path (`step` on single-rank uses identical kernels minus halo exchange)
- `src/legoesm/atmosphere/dynamics/compressible_euler_plane.PlaneCompressibleEulerModel`
- `make_flat_plane_terrain_metric`, `make_rest_state`, `create_plane_grid`, `create_stretched_height_coordinate`

## Iteration log

### Iter 2 — 2026-05-27 — profile + acoustic-substep sweep

**Profile (N=96, fp64) via `scripts/profile_crm_step.py`:**

| stage                       | ms/step | % full step |
|-----------------------------|---------|-------------|
| full_step                   | 17.86   | 100         |
| acoustic_substeps × 12 (1 RK3 stage) | 6.96    | 39          |
| acoustic_substeps × 12 × 3  | 20.88   | ~94         |
| slow_tend (1 RK3 stage)     | 0.31    | 1.8         |
| slow_tend × 3 (full RK3)    | 0.94    | 5           |
| hyperdiff_biharmonic        | 0.019   | 0.1         |
| sponge_taper                | 0.012   | 0.1         |

**~94% of step time in semi-implicit acoustic substep loop** (36 column-Thomas
solves per step). Slow tendency 5%, diffusion negligible. Only useful lever:
reduce `n_acoustic_substeps`.

**Substep sweep (N=96, 30 bench steps, fp64):**

| nsub | ms/step | Mc/s | 30-step finite? |
|------|---------|------|-----------------|
|  2   |  3.36   |  82  | NaN             |
|  4   |  6.37   |  43  | True            |
|  6   |  9.16   |  30  | True            |
|  8   | 12.07   |  23  | True            |
| 12   | 17.15   |  16  | True (baseline) |

nsub=4 → **2.7× speedup**, stable over 30 steps. nsub=2 violates acoustic
CFL — gravity-wave / sound-speed limit.

**Caveat: 200-step CFL is fragile** — even nsub=12 default NaN'd on a
random rest-state kick (0.001 K) because hyperdiff can't damp the
broadband noise spectrum at this dx. Production CRM init (smooth
perturbation + active sponge) needed for long-run stability claims;
the substep speedup is bench-grade only until validated against a
real CRM IC (Wing RCEMIP, BOMEX, etc.).

`bench_crm_gpu_scaling.py` now exposes `--n-acoustic-substeps` flag.

**Full nsub=6 sweep on bench (30 steps fp64):**

| res  | ms/step (nsub=12) | ms/step (nsub=6) | Mc/s (nsub=6) | speedup |
|------|-------------------|------------------|---------------|---------|
| N=48 | 11.40             |  5.79            |  11.9         | 1.95×   |
| N=96 | 17.48             |  9.04            |  30.6         | 1.94×   |
| N=192| 42.95             | 25.60            |  **43.2**     | 1.68×   |

CRM peak so far: **43.2 Mc/s @ N=192 fp64 nsub=6**. Still ~3× below
atm CS peak (141 Mc/s fp64) but in the same memory-bound regime now
that substep count is reasonable.

### Iter 1 — 2026-05-27 — scaffold + first sweep

- New branch `crm_gpu` off main `21098286`
- New `scripts/bench_crm_gpu_scaling.py` (~180 LOC, reuses atm-bench helpers)
- Config: f-plane, semi-implicit acoustic, 12 substeps, no physics, fp64
- Sweep: N=48, 96, 192, 384 horizontal cells, nlev=30, dx=2000 m, dt=2 s
- Next: run first sweep, capture baseline, identify bottlenecks
