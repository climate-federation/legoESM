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

### Iter 1 — 2026-05-27 — scaffold + first sweep

- New branch `crm_gpu` off main `21098286`
- New `scripts/bench_crm_gpu_scaling.py` (~180 LOC, reuses atm-bench helpers)
- Config: f-plane, semi-implicit acoustic, 12 substeps, no physics, fp64
- Sweep: N=48, 96, 192, 384 horizontal cells, nlev=30, dx=2000 m, dt=2 s
- Next: run first sweep, capture baseline, identify bottlenecks
