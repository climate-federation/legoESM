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

### Iter 3 — 2026-05-27 — push to N=384 + fp32 sweep

**fp64 N=384 nsub=6:** 164.98 ms / 26.8 Mc/s — falls 38% past N=192 peak.
**Plateau-then-fall confirmed** — state at N=384 (~400 MB fp64) far past
L2 (48 MB) so HBM dominates with no reuse.

**fp32 nsub=6 sweep (keeps climbing past N=192):**

| res    | total cell-lev | ms/step | Mc/s |
|--------|----------------|---------|------|
| N=96   |   276,480      |  7.28   |  38  |
| N=192  | 1,105,920      | 23.43   |  47  |
| N=384  | 4,423,680      | 78.02   | **57** |

fp32/fp64 ratio at N=384 = 2.1 → fully bandwidth-bound. Smaller state
size (200 MB) closer to L2 fit. **CRM peak overall: 57 Mc/s @ N=384
fp32 nsub=6.**

Beats prior fp64 peak (43 Mc/s) by 32%. Still 4× below ocean LL
impl_cn peak (546 Mc/s) because:
1. Acoustic substep loop is fundamentally serial (12-iter Thomas chain)
2. Plane operators (gradient, divergence) on full 3D state per substep
3. Slow tendency RK3 outer still doubles base work

### Iter 4 — 2026-05-27 — fill curve N=24/256, plot

Filled small + mid-N for full saturation curve. nsub=6, both precisions:

| res    | total cell-lev | fp32 ms/Mc/s   | fp64 ms/Mc/s   |
|--------|----------------|----------------|----------------|
| N=24   |    17,280      | 11.43 /  1.5   |  8.10 /  2.1   |
| N=48   |    69,120      |  5.79 / 11.9   |  5.79 / 11.9   |
| N=96   |   276,480      |  7.28 / 38     |  9.04 / 30.6   |
| N=192  | 1,105,920      | 23.43 / 47     | 25.60 / 43.2   |
| N=256  | 1,966,080      | 34.74 / 57     | 45.17 / 43.5   |
| N=384  | 4,423,680      | 78.02 / 57     |164.98 / 26.8   |

**fp32 plateau at 57 Mc/s spanning N=256-384** — true CRM ceiling on
this GPU. fp64 plateaus at ~43 Mc/s and falls past N=256.

Dispatch floor: ~10 ms at N=24 regardless of precision (acoustic
substep launch × 36 + scan overhead). Same shape as atm CS, ocean
LL — fixed launch cost dominates below 50k cell-lev.

Plot reuse: `scripts/plot_gpu_scaling.py` (from PR #319, already in
main) → `results/scaling_crm_gpu/scaling_gpu_*.png` (4 figures).

### Iter 5 — 2026-05-27 — bottleneck root-cause + N=512 plateau confirm

**Per-stage breakdown N=384 fp64 nsub=6:**
- full step = 158 ms
- nsub=1 step = 36 ms ⇒ slow_tend = 4 ms/RK3 stage × 3 = 12 ms (8%)
- per-substep = 8.1 ms × 18 substep calls = 146 ms (92%)

Each substep solves column-Thomas tridiagonal (vertical semi-implicit
acoustic). 384² = 147k columns × 30 levels = ~44M ops/substep,
producing only 5.4 GFLOPs/substep on fp64 — **0.5% of fp64 ALU peak,
3% of HBM peak**. Column-Thomas is serial along the vertical axis
(forward+backward sweep), so JAX/XLA cannot vectorize beyond the
horizontal column dimension. The substep loop is already
`jax.lax.fori_loop`-fused — no extra kernel launch overhead.

⇒ **Root cause: column-Thomas vertical recursion is the fundamental
serial bottleneck.** Cannot beat this without batched-tridiagonal
GPU primitive (cuSPARSE / cuSolverDn). Out of scope for "minimum
code production."

**fp32 N=512 nsub=6:** 167.7 ms / 46.9 Mc/s — falls 18% past peak.
Final fp32 plateau: N=256-384 at **57 Mc/s**.

### Iter 3 codex review applied:
- [HIGH] post-warmup + post-timing `jnp.isfinite` assert added —
  NaN/Inf now raises RuntimeError instead of silently fast
- [HIGH] horizontal acoustic CFL (`c_sound·dt/nsub/dx`) computed +
  warned when >0.7; printed in run header at startup
- Sub-warning thresholds preserved (default config CFL=0.057
  comfortable, nsub=2 would be 0.85 → blocked)

### Iter 1 — 2026-05-27 — scaffold + first sweep

- New branch `crm_gpu` off main `21098286`
- New `scripts/bench_crm_gpu_scaling.py` (~180 LOC, reuses atm-bench helpers)
- Config: f-plane, semi-implicit acoustic, 12 substeps, no physics, fp64
- Sweep: N=48, 96, 192, 384 horizontal cells, nlev=30, dx=2000 m, dt=2 s
- Next: run first sweep, capture baseline, identify bottlenecks
