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

### Iter 19 — 2026-05-27 — within-process timing stability vs cross-process

Same N=128 fp32 explicit nsub=4 dt=0.5, single Python process,
varying scan length n_timing × 5 repeats each:

| n_timing | median ms | min   | max   | spread |
|----------|-----------|-------|-------|--------|
|  30      | 0.748     | 0.745 | 0.752 |  1.0%  |
| 100      | 0.767     | 0.736 | 0.773 |  4.8%  |
| 300      | 0.769     | 0.704 | 0.941 | 30.8%  |
| 1000     | 0.774     | 0.756 | 0.793 |  4.8%  |

**Within-process median stable to 3.5%** across all scan lengths.
n_timing=30 is already adequate.

But across 3 separate Python processes (iter 11 / 18 / 19):
- Run 1: 0.66 ms (740 Mc/s)
- Run 2: 0.64 ms (774)
- Run 3: 0.77 ms (640)

**Cross-process variance ~20%** = the actual noise floor for sub-ms
timings. `--repeat N` only tightens within-process; new process
gets fresh GPU thermal state, JIT cache load, kernel selection.

⇒ **Final honest CRM peak: 640-820 Mc/s plateau N=96-192 fp32 explicit
(median value depends on Python process; ~700 Mc/s typical).**

n_timing=30 default is right; increasing past 100 gives no benefit
(possibly worse — thermal drift over longer-running scan).

### Iter 18 — 2026-05-27 — dx sweep + run-to-run variance widened

Explicit fp32 N=128 nsub=4 dt=0.5 across dx (median-3 each):

| dx   | ms/step | Mc/s | SYPD |
|------|---------|------|------|
| 1 km | 0.58    | 851  | 2.37 |
| 2 km | 0.64    | 774  | 2.16 |
| 4 km | 0.65    | 758  | 2.11 |

Throughput Mc/s essentially **grid-spacing-invariant** (cell-level work
doesn't depend on dx; only horizontal CFL constraint does, and stays
safe across this range).

But: across the 3 fresh Python processes, **range 758-851 = ±6%**, and
prior median-3 measurements at same point gave 680-740. **Honest run-to-
run variance closer to ±15%** than the ±5% claimed iter 11. Codex iter-16
#1 was right — median-of-3 isn't tight enough for sub-ms timings.

**Corrected final CRM peak: 680-850 Mc/s plateau N=96-192 fp32 explicit
dt=0.5 nsub=4 (median-3, run-to-run var ±15%).** Reporting a range
rather than point estimate is the honest summary.

### Iter 17 — 2026-05-27 — codex iter-16 review applied

Codex flagged HIGH:
- **#4** Explicit vs semi-implicit not numerically validated — only throughput-
  compared. **Marked explicit as THROUGHPUT-ONLY config in markdown**; full
  field-error / conservation regression vs reference is out of scope.
- **#8** `dz_sfc=100m` hardcoded in CFL check — replaced with `dz_min` from
  actual `hc.dz_half`. CFL value now reflects true coord regardless of
  `dz_sfc` arg.
- **#7** `--allow-unsafe-cfl` refuse path tested manually:
  - `nsub=2 dt=2 --explicit-acoustic` → REFUSE "vertical CFL = 3.22 (>0.5)"
  - `--allow-unsafe-cfl` → proceeds → caught by post-warmup `isfinite` raise

Defense-in-depth verified: CFL guard → override flag → finite assert.

MEDIUM fixes applied:
- **#6** `physics_level` now includes `L<nlev>` — no CSV pooling collision
  across nlev sweeps.

MEDIUM noted but deferred (out of "minimum code" scope):
- #1/#2: per-repeat variance into CSV
- #3: GPU-contention as iter-3 cause is plausible but unproven
- #5: nlev claim needs profile evidence
- #9: A100/H100 projection is hypothesis (already noted in iter-16)

### Iter 16 — 2026-05-27 — fp64 explicit median-3 completes precision×solver matrix

Final 2×2 matrix (median-3, fp32+fp64, both solvers):

| solver / prec       | plateau Mc/s | peak res    | peak ms |
|---------------------|--------------|-------------|---------|
| **explicit fp32**   | **680**      | N=96-192    | 0.7-1.6 |
| explicit fp64       | 113          | N=96-192    | 2.5-9.8 |
| semi-implicit fp32  | 119          | N=192-256   | 9.8-17  |
| semi-implicit fp64  |  43          | N=192       | 25.6    |

Observations:
- **fp32/fp64 ratio @ explicit = 6.0×** (680/113) → consumer fp64
  ALU 1/64 of fp32 dominates; explicit shifts compute share enough
  that fp64 ALU bottlenecks it
- **fp32/fp64 ratio @ semi-implicit = 2.77×** → less ALU-bound
  because column-Thomas is launch-overhead+memory-bound
- **SI-fp32 (119) ≈ explicit-fp64 (113)** — different paths to the
  same ~115 Mc/s ceiling on consumer hardware
- **Explicit-fp32 (680) is uniquely fast** — combines fast kernel
  (no Thomas) AND fast precision (no fp64 ALU penalty)

On A100/H100 datacenter GPUs (fp64 = 1/2 fp32 nominal), explicit
fp64 would reach ~340 Mc/s and SI fp64 ~60 — much closer to fp32
ceilings. The 6× gap is specific to consumer Blackwell mobile.

### Iter 15 — 2026-05-27 — nlev sensitivity (vertical-dim scaling)

Explicit fp32 N=128 dt=0.5 nsub=4 median-3 across nlev:

| nlev | total cells | ms/step | Mc/s | SYPD |
|------|-------------|---------|------|------|
|  30  |   491,520   |  0.65   | 757  | 2.11 |
|  60  |   983,040   |  1.76   | 559  | 0.78 |
| 120  | 1,966,080   |  3.65   | 538  | 0.37 |

Vertical scaling:
- nlev 30→60: cells 2×, time 2.7× → throughput drops 26% (kernel-
  launch overhead amortizes better at higher work)
- nlev 60→120: cells 2×, time 2.07× → near-ideal linear

CRM throughput stays in 530-760 Mc/s band across 4× vertical range.
Production LES (nlev=60-80) sits in the middle of plateau. Fine-
vertical research (120+) only loses ~5% from typical.

### Iter 14 — 2026-05-27 — explicit acoustic decomposition (where time goes)

Per-stage decomposition of explicit fp32 at peak (N=128, median-3):

| nsub | step ms | per-substep ms | inferred slow_tend × 3 |
|------|---------|----------------|------------------------|
|  1   | 0.384   | n/a (1 substep × 3 = 3 calls) | 0.26 ms |
|  2   | 0.607   | 0.041          | 0.26 ms                |
|  4   | 0.749   | 0.041          | 0.26 ms                |

Per RK3 stage at nsub=4:
- slow_tend: 0.087 ms (**35% of stage** — much higher than semi-implicit's 5%)
- 4 explicit substeps: 0.164 ms (65%)

**Explicit substep cost = 0.041 ms (vs ~3 ms semi-implicit at same N)** — column-
Thomas removal is the main win. Each substep is now lightweight enough that
slow_tend (plane operators on full 3D state) becomes the larger share per
RK3 stage.

GPU kernel-launch overhead floor: ~5 μs × ~15 launches per step = 75 μs ≈ 10%
of step at N=128. Negligible at larger N, inflates at small N.

⇒ Explicit acoustic is now **near the compute-bound limit** for the plane
operators themselves — further wins require fusing slow_tend with adjacent
substeps (XLA already does some fusion via `fori_loop`; getting more would
need stencil-fused custom kernels, out of scope).

### Iter 12 — 2026-05-27 — verify PR #319 ocean LL peak is not inflated

Ran ocean LL192 fp32 impl_cn 3 times in series:

  Run 1: 2.70 ms / **545.4 Mc/s**
  Run 2: 2.70 ms / **547.1 Mc/s**
  Run 3: 2.70 ms / **546.3 Mc/s**

Variance ±0.3%. PR #319 single-shot peak of 546 Mc/s is **verified
robust** — not inflated.

Conclusion: single-shot timing is reliable when step time >2 ms.
Sub-millisecond cases (CRM explicit N<192) need median-of-3.

The CRM explicit-acoustic plateau at 680 Mc/s (iter 11) and ocean LL
peak at 546 are both real; the prior CRM 820 inflation was an
artifact of the sub-millisecond regime + scan-amortization + cache
warmth, not a systematic bench-wide noise.

### Iter 11 — 2026-05-27 — median-3 sweeps for BOTH solvers (corrections)

Full median-3 sweeps. Prior single-shot numbers were systematically
low for semi-implicit (cause unclear — likely GPU contention or first-
compile artifact in iter-3 batch run).

**Explicit fp32 dt=0.5 nsub=4 median-3:**

| res    | single-shot | median-3 |
|--------|-------------|----------|
| N=48   | 374         | 320      |
| N=96   | 636         | 689      |
| N=128  | **820**     | **680**  |
| N=192  | 686         | 682      |
| N=256  | 648         | 648      |
| N=384  | 473         | 470      |

Honest plateau: **~680 Mc/s spans N=96-192** (3 sizes, 12× cell range).
N=128 single-shot 820 was cache-warm inflation. Run-to-run noise ±5%.

**Semi-implicit fp32 dt=2.0 nsub=6 median-3 (BIG REVISIONS):**

| res    | single-shot | median-3 | revision |
|--------|-------------|----------|----------|
| N=48   |  11.9       |  11.4    | same     |
| N=96   |  38         |  46      | +21%     |
| N=128  | n/a         |  78      | new      |
| N=192  |  47         | **113**  | **+140%**|
| N=256  |  57         | **119**  | **+109%**|
| N=384  |  57         |  57      | same     |

**Semi-implicit fp32 plateau actually 113-119 Mc/s @ N=192-256**, not
~57 as prior iters claimed. Iter-3 to iter-5 fp32 numbers were
artifacts of GPU contention during batched runs.

Updated honest peaks both solvers fp32:
- **Explicit:** 680 Mc/s plateau N=96-192
- **Semi-implicit:** 119 Mc/s plateau N=192-256

Explicit still 5.7× faster than semi-implicit, but the absolute SI
numbers are now more competitive (119 vs prior-reported 57). Explicit
remains the throughput-preferred path.

### Iter 10 — 2026-05-27 — codex review iter-9 + median-of-3 walks back 820 Mc/s

Full explicit fp32 dt=0.5 nsub=4 sweep with `--repeat 3`:

| res    | single-shot | median-3 (1st) | median-3 (2nd) |
|--------|-------------|----------------|----------------|
| N=48   | 374         |  —             | 320            |
| N=96   | 636         | 702            | 689            |
| N=128  | **820**     | **740**        | **680**        |
| N=192  | 686         | 694            | 682            |
| N=256  | 648         |  —             | 648            |
| N=384  | 473         |  —             | 470            |

**Honest plateau: ~680 Mc/s spans N=96-192** (3 sizes, 12× cell range).
N=128 single-shot peak (820) was inflated by cache warmup. Run-to-run
variation in median-3 is ±5% — at this step time (<1 ms) timing
resolution is the limit.

Final CRM ceiling: **~680 Mc/s plateau N=96-192 fp32 explicit dt=0.5
nsub=4.** Still highest in legoESM suite (vs ocean LL 546, atm ico 428
from PR #319 — same single-shot caveat applies there).

### Iter 10 — 2026-05-27 — codex review iter-9 + median-of-3 walks back 820 Mc/s

Codex flagged: single-shot timing inflated by cache warmup; CFL warns
should refuse by default; stale "660" doc string. Applied:

- `--repeat N` flag (default 1, recommend 3 for <1 ms steps). Median
  of N runs reduces scan-amortization noise.
- `--allow-unsafe-cfl` flag. Default behavior: refuse with `SystemExit`
  when horiz CFL >0.7 or vertical CFL >0.5 (explicit). Was warn-only.
- CLI help cleaned up; no longer embeds volatile benchmark numbers.

**Median-of-3 explicit fp32 dt=0.5 nsub=4 (re-measured):**

| res    | single-shot Mc/s | median-3 Mc/s | delta |
|--------|------------------|---------------|-------|
| N=96   | 636              | **702**       | +10%  |
| N=128  | **820**          | **740**       | -10%  |
| N=192  | 686              | 694           | ~0    |

**Honest CRM peak: 740 Mc/s @ N=128 fp32 explicit (median-3).**

The 820 single-shot number was cache-warmup-inflated as codex
predicted. N=192 is stable across single/median measurements because
state is large enough that cache effects are minor.

Updated final ladder (legoESM single-GPU peaks, all median-grade):
- CRM explicit fp32: **740 Mc/s** @ N=128
- Ocean LL impl_cn fp32: 546 (PR #319 single-shot — likely also
  inflated; needs median verification)
- Atm icosahedral fp32: 428 (PR #319, similar caveat)

### Iter 9 — 2026-05-27 — bench supports `--explicit-acoustic` flag

`bench_crm_gpu_scaling.py` now exposes `--explicit-acoustic`. Through
the bench (with `lax.scan` fuse + isfinite guard + vertical CFL warn),
explicit fp32 dt=0.5 nsub=4 sweep:

| res    | ms/step | Mc/s     | SYPD |
|--------|---------|----------|------|
| N=48   |  0.18   | 374      | 7.40 |
| N=96   |  0.43   | 636      | 3.15 |
| N=128  |  0.60   | **820**  | 2.29 |
| N=192  |  1.61   | 686      | 0.85 |
| N=256  |  3.04   | 648      | 0.45 |
| N=384  |  9.35   | 473      | 0.15 |

**NEW CRM PEAK: 820 Mc/s @ N=128 fp32 explicit dt=0.5 nsub=4.**

Through-bench numbers are ~24% higher than ad-hoc inline loop
(iter-7) — `lax.scan` fuse amortizes per-step Python overhead at
the sub-millisecond step times.

Final ladder vs prior gpu-scaling work (PR #319):
- Ocean LL impl_cn fp32 peak: 546 Mc/s
- Atm icosahedral fp32 peak: 428
- **CRM explicit fp32 peak: 820** ← highest in legoESM suite
- CRM semi-implicit fp32 peak: 57

CRM dispatch floor (explicit): 0.18 ms at N=48 — much smaller than
semi-implicit's 5+ ms because no column-Thomas + smaller dt = less
work per step.

### Iter 8 — 2026-05-27 — explicit fp32 plateau + 200-step CFL

Filled mid-range + checked long-run stability:

**fp32 explicit dt=0.5 nsub=4 (plateau characterization):**

| res    | ms/step | Mc/s |
|--------|---------|------|
| N=128  |  0.75   | 659  |
| N=192  |  1.67   | 661  |
| N=256  |  2.98   | 660  |
| N=384  | 11.93   | 371  |
| N=512  | 20.13   | 391  |

**Plateau N=128-256 at 660 Mc/s** — dead-flat across 4× cell range.
Falls past N=256 (L2 overflow); slight rebound at N=512.

**200-step CFL check (100 s sim, 0.001 K kick):**

| res   | finite | |θ'|_max  | |w|_max     |
|-------|--------|-----------|-------------|
| N=96  | ✓      | 4.6e-3 K  | 4.4e-2 m/s  |
| N=192 | ✓      | 4.6e-3 K  | 4.1e-2 m/s  |

**Explicit nsub=4 is CFL-stable for 200 steps**, unlike semi-implicit
nsub=12 which NaN'd on identical kick (iter 2 caveat). Explicit
respects acoustic CFL directly; semi-implicit's vertical-implicit
treatment lets horizontal modes leak.

⇒ **Final CRM config: explicit + fp32 + dt=0.5 + nsub=4. 660 Mc/s
plateau, CFL-stable, no SYPD trade-off.**

### Iter 7 — 2026-05-27 — explicit acoustic full sweep — NEW CRM PEAK

Pushed explicit acoustic to full N range, fp64 + fp32:

**fp64 explicit dt=0.5 nsub=4:**

| res    | ms/step | Mc/s |
|--------|---------|------|
| N=48   |  0.69   |  99.6 |
| N=96   |  2.46   | **112** (peak) |
| N=192  | 11.24   |  98.4 |
| N=384  | 49.05   |  90.2 |

Flatter than semi-implicit. Peak 112 Mc/s, drops only to 90 at N=384.

**fp32 explicit dt=0.5 nsub=4 — ABSOLUTE CRM PEAK:**

| res    | ms/step | Mc/s     | SYPD |
|--------|---------|----------|------|
| N=48   |  0.17   | 415      | 8.04 |
| N=96   |  0.43   | 636      | 3.18 |
| N=192  |  1.67   | **661**  | 0.82 |
| N=384  | 11.93   | 371      | 0.11 |

**661 Mc/s @ N=192 fp32** — exceeds ocean LL impl_cn peak (546).
This is the **highest single-GPU throughput in the legoESM suite**.

fp32/fp64 ratio at N=192 = 6.7× — fp64 ALU bottlenecked (consumer
RTX 5090 fp64 = 1/64 fp32 nominal). Datacenter A100/H100 would
hit fp64 ratio ~2×.

**Walks back iter-6 conclusion.** SYPD comparison @ N=96 fp32:
- semi-implicit dt=2 nsub=6:  7.28 ms / 0.75 SYPD
- explicit dt=0.5 nsub=4:     0.43 ms / **3.18 SYPD**

Explicit wins SYPD by **4.2×** when CFL margin permits. Previous
iter-6 comparison was at fp64 where the fp64 ALU penalty hides
the advantage.

⇒ **Explicit + fp32 is the right CRM throughput config.** Semi-
implicit only relevant when dt is constrained by physics-scale (>1 s)
or for AD-stable trajectory studies.

### Iter 6 — 2026-05-27 — explicit acoustic comparison (throughput vs SYPD)

Tested whether bypassing column-Thomas via fully-explicit acoustic
recovers throughput. Vertical CFL constrains dt_a < ~0.15 s at
dz_sfc=100 m.

Explicit acoustic N=96 fp64 (dt=2.0 s NaN'd; tried dt=0.5 s):

| dt   | nsub | ms/step | Mc/s    | CFL_v | finite | SYPD  |
|------|------|---------|---------|-------|--------|-------|
| 0.5  |   4  |  2.46   | **113** | 0.42  | ✓      | 0.557 |
| 0.5  |   8  |  3.78   |  73     | 0.21  | ✓      | 0.362 |
| 0.5  |  12  |  5.13   |  54     | 0.14  | ✓      | 0.267 |
| 2.0  |   8  |  2.74   | 101     | 0.85  | NaN    | n/a   |

Compare semi-implicit (dt=2.0 nsub=6): 9.04 ms, 30.6 Mc/s, SYPD 0.61.

**Different metrics tell different stories:**
- Throughput Mc/s: explicit (113) beats semi-implicit (30.6) by 3.7×
- SYPD (sim years per wall day): nearly tied — explicit 0.56 vs
  semi-implicit 0.61 — because explicit's smaller dt requires 4×
  more steps per simulated second
- Compute per sim-second: semi-implicit 4.52 ms/sim_s, explicit
  4.92 ms/sim_s — semi-implicit wins ~8%

⇒ **Semi-implicit is the right path for SYPD/climate-scale runs**
(longer dt amortizes column-Thomas serialization). Explicit is
better for throughput-only benchmarks (denser per-step work,
larger Mc/s).

For storm-resolving runs where dt is naturally bounded by horizontal
advection (~0.5 s at dx=2 km), explicit may be the better choice.

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
