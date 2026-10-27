# Certified SWG reverse-mode gradient gate — round 2 receipt

Date: 2026-10-12

Status: **LANDED; DIRECT 60-DAY CPU MEMORY FOOTPRINT CONFIRMED; WALL-TIME
RETRACTED; GPU OPERATOR ROWS OPEN**.
The round-1 GPU resource result is retracted in the executable and receipt,
the replacement dual-sampler GPU instrument is fail-closed, and the sixty-day
CPU agreement and three schedule-resource rows were measured directly. No
physics, card choice, loss, bar, stabilizer, clipping, guard, configuration, or
`stop_gradient` changed.

## Provenance

The incoming tip was `27603d95a`. The frozen round-2 protocol and repaired
instrument are commit `93c294b47edbace6c9256d61380f0d13f9d4e3a0`. All new
CPU and retraction JSON files are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swggg_rounds/round2/` and stamp
that clean commit. CPU rows label fp64/libm, float64 state and geometry, the
admitted kt=2 stage-1 boundary, and the production
`LatLonCGridOceanModel._step_jitted` entry.

## Round-1 GPU disposition

The operator's native-policy GPU agreement rows remain valid. The six resource
rows are **RETRACTED AS VACUOUS**, not reinterpreted: their preallocated-pool
increments were `none=0`, `uniform=6`, and `binomial=4` MiB at both 24 and 720
steps. Running the current executable on each old three-row suite produces
VACUOUS because no JAX allocator increment exists. The retraction artifacts
are `round1_gpu_resources_24step_retraction.json` and
`round1_gpu_resources_720step_retraction.json`.

| GPU steps | reverse/forward relative | reverse/FD relative | agreement verdict | resource verdict |
|---:|---:|---:|---|---|
| 1 | `2.27064171926424e-16` | `3.57798841572499e-07` | **CONFIRMED** | not measured |
| 24 | `9.30692734647596e-16` | `2.1980290555407674e-06` | **CONFIRMED** | **VACUOUS, RETRACTED** |
| 720 | `2.3070259307585205e-16` | `9.041792746430404e-07` | **CONFIRMED** | **VACUOUS, RETRACTED** |

## Repaired GPU memory instrument — allocator requirement retracted

**ROUND-3 RETRACTION:** the round-2 requirement below that forced
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` is wrong. On the operator GPU that
allocator makes `jax.devices()[0].memory_stats()` return `None`, so every
resource row was correctly refused and no round-2 GPU memory result exists.
Round 3 retains `XLA_PYTHON_CLIENT_PREALLOCATE=false`, refuses the platform
allocator, and uses the default JAX allocator whose memory statistics were
demonstrated by the operator. The sampler-thread restart defect in the
round-2 executable is also fixed in round 3 by creating one sampler per pass.

The retracted round-2 protocol would have run every resource row in a fresh
process with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` set before JAX import. The process
boundary would reset the process-lifetime peak. The first reverse pass was
intended to record JAX
`bytes_in_use` and `peak_bytes_in_use` before and after, plus concurrently
sampled physical-GPU-1 `nvidia-smi memory.used`. The JSON retains both sets of
baselines, peaks, increments, and sample counts.

An individual row is VACUOUS unless the JAX process peak and `nvidia-smi` both
advance. Its three-schedule suite is CONFIRMED only when both samplers resolve
positive increments ordered `none > binomial > uniform`. No ratio threshold
was added. The planted `none` measurement under the `uniform` label makes both
ordering checks false and returns REFUTED in the direct test.

The committed operator entry point is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`.
It pins physical GPU 1, uses fp64/native with the non-certified label, runs
agreement at 1/24/720/1440 steps, runs all three schedules at 24/720/1440
steps in separate processes, then runs the ordering gate and plant for each
window. This sandbox cannot acquire CUDA rows.

## Directional agreement through production

The scalar objective, perturbation, direction, FD step, and bars are unchanged:
mean variance-normalized wet-cell SSH/U/V MSE; keys 0 and 42; perturbation norm
`1e-3`; central FD step `1e-3`; reverse/forward relative error at most `1e-9`;
reverse/FD relative error at most `0.05`.

| steps | reverse | forward | central FD | rev/fwd relative | rev/FD relative | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `-3.055917560677806e-02` | `-3.0559175606777962e-02` | `-3.0559186540819565e-02` | `3.1788984069699327e-15` | `3.577988403236465e-07` | **CONFIRMED** |
| 24 | `-2.1842644706797177e-05` | `-2.1842644706797268e-05` | `-2.1842692717673664e-05` | `4.188117305913815e-15` | `2.198029203985421e-06` | **CONFIRMED** |
| 720 | `-3.5854846776002614e-09` | `-3.5854846769147900e-09` | `-3.5854814359568625e-09` | `1.9117954763975566e-10` | `9.041018691902848e-07` | **CONFIRMED** |
| 1440 | `-7.514160251955201e-10` | `-7.514160250379746e-10` | `-7.514108760061941e-10` | `2.0966487001510505e-10` | `6.852647738821974e-06` | **CONFIRMED** |

All derivatives are finite and nonzero; no NaN appeared. The 1440-step JSON
SHA-256 is
`bc6ad0ce85cb92559fcc0e081f95e4878ac1988f0eb007cf828bd3c985498f33`.
The round-1 broken-VJP plant remains binding and unchanged.

## Direct sixty-day resource rows — wall-time retraction

**ROUND-3 RETRACTION:** both wall-time columns below include retracing and
compilation of an un-jitted `value_and_grad` transformation and are not reverse
execution costs. The absolute peak RSS values remain valid total process
footprints. The increments are retained only as diagnostics because their
post-cold baselines already include retained compiler and allocator memory.
Round 3 directly replaces the execution timings with a pre-jitted timer.

Each 1440-step row ran in its own process. CPU host RSS was sampled during one
blocked warm reverse after a cold compile and reverse. All losses and gradients
are finite and nonzero. The schedule primal is bitwise for binomial and within
the frozen contract for none/uniform; their maximum difference from the
per-step production state is `1.2789769243681803e-13`.

| schedule | cold whole-call (s; retracted as execution cost) | second whole-call (s; retracted) | baseline (MiB) | peak RSS footprint (MiB) | increment (MiB; diagnostic only) | samples | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| none | `46.8561` | `51.5346` | `25597.848` | `33868.055` | `8270.207` | 917 | **CONFIRMED** |
| uniform | `26.7938` | `25.2765` | `2133.180` | `2422.047` | `288.867` | 496 | **CONFIRMED** |
| binomial | `63.7616` | `57.9683` | `2072.852` | `2558.863` | `486.012` | 1142 | **CONFIRMED** |

The resource JSON SHA-256 values are `08a38f338fbed5d5b6b21dc44baa00133fc4b71a784c9d224d6e2abfe04465b6`
(none), `a8a7bd6edf793a93e3867c282b0bdbb76298593d4729317a7aa9176b59290b4c`
(uniform), and
`1677eccc4cf51556f9131be67873db807433db34994db1e1c95c4735b382ec06`
(binomial). These direct values replace the round-1 estimates; they do not
alter or retroactively validate the extrapolation formulas.

Relative to the round-1 peak-RSS extrapolations, the direct 1440-step values
moved by `+4062.594 MiB` (`+13.63%`) for none, `+215.313 MiB` (`+9.76%`)
for uniform, and `-82.098 MiB` (`-3.11%`) for binomial. These deltas compare
total peak-RSS predictions with direct total peak-RSS observations; they do
not rehabilitate the retracted increment interpretation.

## Training composition

Training on the certified SWG card will use the per-step Python loop measured
by this gate: each step crosses the production jitted `swe_rk3` boundary and is
rematerialized as one step. It will not use the retained uniform-scan
composition. That scan's 24-step zero-perturbation primal differed from
production by `8.881784197001252e-16` in SSH,
`1.3877787807814457e-17` in U, and `6.071532165918825e-18` in V. The bitwise
prediction remains **REFUTED**; the last-bit gap is quoted rather than waived.

## Tests and controls

- Shared gate direct tests: **22 passed**.
- Isolated SWG production gradient-matrix row: **1 passed**.
- Python compilation, Ruff, GPU shell syntax, and `git diff --check`: PASS.
- Old 24/720 GPU resource suites: executable VACUOUS with exit 1.
- Citation gate on this exact receipt and shifted-citation plant: recorded in
  the final round commit.

## OPEN

1. The operator acquired the 1440-step GPU agreement row at the round-2 tip;
   it is valid and is carried into the round-3 receipt. The repaired
   24/720/1440 GPU resource rows remain unacquired at this round's tip.
2. The inherited stale digest pins in
   `tests/ocean/unit/test_nemo_vortex_card.py` remain GYRE-lane debt and are not
   re-pinned here. Expected GYRE/LOCK/OVERFLOW values are
   `5850bbcfcd351c29` / `4ec82201f0fae7bf` / `7f1ec89b3e6578f5`; actual
   round-6 values are `f02d2f42a570f413` / `d5dfdf83b3a1e0ab` /
   `83de0010e1170032`. The operator reproduced the same three failures on the
   pre-round SWG tip and round-1 tip.

## Source citations

The certified compiled SWG dispatcher is the three-stage program at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:122-321`; its executed stage-3 momentum
block contains vector advection, ENS vorticity, constant Laplacian viscosity,
centered wind, and linear drag at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:260-273`. The total-ENS operator and its
live F-thickness divisions are in
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675`.
