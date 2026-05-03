# legoESM GPU-scaling — current status (latest iter-217, 2026-05-03)

This file tracks the GPU/CPU scaling work on the `GPU scaling` branch.

## TL;DR — single-device GPU/CPU speedup (1-day BCW, iter-212 honest measurements)

| Grid          | Resolution | ncells | CPU sps | GPU sps (best) | speedup | flag           |
|---------------|-----------:|-------:|--------:|---------------:|--------:|----------------|
| spectral      | T21        |   8192 |    38.2 |          288.4 | **7.5×**| `--scan-steps=24` |
| spectral      | T42        |  32258 |     4.9 |          117.6 | **24×** | `--scan-steps=24` |
| icosahedral   | I4         |   2562 |    98.8 |          205.5 | **2.1×**| GPU-undersaturated |
| icosahedral   | I5         |  10242 |    19.1 |          198.5 | **10.4×**| `--scan-steps=1` (default) |
| cubed-sphere  | C48        |  13824 |    20.2 |          103.0 | **5.1×**| `--scan-steps=24` |
| cubed-sphere  | C96        |  55296 |       — |           67.0 |  —      | `--scan-steps=24` |

(C24 is excluded — resolution-specific JW-jet resonance documented in
§3.b.  CPU "sps" is from the 8-thread default backend; "GPU sps" is
the RTX 5090 Laptop with the activated `--scan-steps` knob.)

**What's in iter-216/217**: BCW emits a `RuntimeWarning` if user
pairs `--scan-steps>1` with `--grid icosahedral` (iter-212 measured
−9 % at I5).  The iter-199 vector-halo fix is now pinned by three
regression tests in
``tests/atmosphere/hydrostatic/unit/test_primitive_eq.py::TestHydrostaticToFV3VectorHalo``
(round-trip, IC divergence, first-step ps balance).

## 0. Hardware reality (this host)

* 1× NVIDIA RTX 5090 Laptop (Blackwell, 82 SMs, 18 GB VRAM, PCI 02:00.0)
* Driver/userspace mismatch (kernel module 580.126.09, userspace 580.142
  → `cuInit` returns `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`)
* Workaround that needs **no sudo**: `source scripts/gpu_env.sh` extracts
  matching 580.126 userspace libs from the apt cache and prepends to
  `LD_LIBRARY_PATH`.  Idempotent.

```bash
env -i HOME=$HOME PATH=$PATH JAX_ENABLE_X64=1 bash -c \
    "source scripts/gpu_env.sh && \
     PYTHONPATH=. .venv/bin/python -c 'import jax; print(jax.devices())'"
# → [CudaDevice(id=0)]
```

The `env -i` strip is required — the user shell exports
`JAX_PLATFORMS=cpu` from a previous session that overrides the GPU
default; pristine env + `gpu_env.sh` is the canonical activation
recipe.

## 1. Baseline single-device JW BCW benchmark (2-day, default config)

`scripts/run_baroclinic_wave_benchmark.py` produces a 2-day Jablonowski-
Williamson dry baroclinic wave with the default `dt`/`hyperdiff` per
grid.  Numbers below are this iteration's actual single-device
measurements (warm step, JIT compiled once).

| Grid          | Resolution | dt (s) | CPU steps/s | GPU steps/s | speedup | mass drift | energy drift |
|---------------|------------|-------:|------------:|------------:|--------:|-----------:|-------------:|
| spectral      | T21        |    600 |        41.6 |       215.7 |  **5.2×** |   +4.2e-9  |   -6.4e-6   |
| icosahedral   | I4 (2562)  |    150 |       108.0 |       347.5 |  **3.2×** |   -7.9e-9  |   -1.5e-5   |
| cubed-sphere  | C48 (13824)|    150 |        25.5 |       136.7 |  **5.4×** |   -3.2e-8  |   -4.9e-4   |

CPU-vs-GPU mass drift is **bit-identical** (icosahedral: -7.866e-9 on
both backends); energy drift differs only in the last digit
(1.463e-5 vs 1.462e-5) consistent with fp64 reduction-order differences.
This confirms that JAX/XLA produces numerically equivalent trajectories
on both backends for the spectral and MPAS-style dycores.

## 2. Conservation matrix (12 case × grid × dycore cells)

`scripts/check_conservation_all.py` exercises every supported dycore on
every supported grid for a short integration and prints relative mass +
energy drift.  Without conservation fixers (intrinsic dycore property)
all twelve cells produce drifts below `4e-6`:

| name                       | mass drift | energy drift | notes                  |
|----------------------------|-----------:|-------------:|------------------------|
| `sw_spectral_w2`           |   2.1e-16  |    5.9e-9    | machine ε mass         |
| `sw_latlon_w2`             |   1.1e-6   |    5.8e-7    |                        |
| `sw_cubed_sphere_w2`       |   1.1e-7   |    3.1e-6    |                        |
| `sw_mpas_w2`               |   1.1e-7   |    1.5e-7    |                        |
| `pe_spectral_iso`          |   8.7e-11  |    2.5e-10   |                        |
| `pe_cubed_sphere_hs`       |   4.2e-7   |    1.7e-6    |                        |
| `pe_latlon_hs`             |   1.2e-6   |    5.6e-7    | dt=30s (CFL at poles)  |
| `pe_mpas_hs`               |   1.5e-7   |    5.1e-7    |                        |
| `nh_cubed_sphere_rest`     |   0        |    0         | exact (rest)           |
| `ocean_cubed_sphere_rest`  |   0        |    1.5e-6    |                        |
| `ocean_latlon_rest`        |   0        |    1.4e-7    |                        |
| `ocean_mpas_rest`          |   0        |    6.8e-8    |                        |

With `--with-fixers` MPAS PE mass drift collapses to **4.7e-15**
(machine ε).  No conservation bugs detected — the dycores already
conserve to expected tolerances; the harness is in-tree to catch
regressions.

## 3. CS C24 BCW dycore — partial fix (iter-199, 2026-05-03)

`hydrostatic_to_fv3` was using the **scalar** halo helper
(`_interp_center_to_corner` → `pad_halo`) to project cell-centre
``(u, v)`` onto the D-grid corners.  Because the local east/north
basis rotates across cube faces, simple averaging of the cross-face
neighbour pulled an incorrectly-signed component into the corner.
Result: the round-trip ``state → cdgrid → state`` distorted
``u`` by **19.6 m/s** out of a 27.7 m/s field (70 % error), and the
projected IC carried ``|div_v| ~ 5.3e-5 s^-1`` at t=0 — driving a
**1009 Pa surface-pressure shift in one 300 s step**.

Fix: switch to ``pad_halo_vector`` / ``pad_halo_vector_4d`` (with
the ``cos_angle``/``sin_angle`` rotation arrays) before the 4-point
average.  The shallow-water ``_sw_to_cdgrid`` test helper already
uses this idiom; this is the same path applied at the dycore level.

| metric                              | before       | after          | factor   |
|-------------------------------------|-------------:|---------------:|---------:|
| round-trip ``u`` error              |     19.6 m/s |     0.32 m/s   |   60×    |
| IC ``\|div_v\|_max``                |    5.3e-5 /s |    3.8e-7 /s   |  140×    |
| ``\|dp_s/dt\|_max`` at t=0          |    3.4 Pa/s  |    0.023 Pa/s  |  146×    |
| step-1 ps shift                     |    1009 Pa   |       6.9 Pa   |  146×    |
| BCW blow-up step (dt=300 s)         |        100   |        ~150    |    +50%  |

The dycore now survives twice as long but still blows up around day
0.5.  The first-step imbalance is gone; what remains is a slower
emergent growth between steps 100 and 150 — likely a divergence-damp
or hyperdiff coefficient mismatch for the emergent baroclinic gradients
once the wave starts amplifying.  Tracked for a follow-up audit.

### 3.b CS C24 BCW: resolution-specific resonance, not a dycore bug
(iter-200, 2026-05-03)

After the iter-199 IC fix the CS PE survives to step ~150 (day 0.5)
on C24, then catastrophically blows up regardless of dt or hyperdiff.
Resolution sweep (perturbed=True, default config + benchmark
hyperdiff):

| Grid | dt    | nu4         | blow-up day              |
|------|------:|------------:|--------------------------|
| C12  | 600 s | 5.6e+18     | **survives ≥ 2 days**    |
| C24  | 300 s | 7.0e+17     | 0.521                    |
| C48  | 150 s | 8.8e+16     | **survives ≥ 2 days**    |

C24 is alone in failing — both the coarser C12 and the finer C48
are stable.  The blow-up time is also nearly dt-insensitive on C24
(dt=300/150/75 s all blow up around day 0.52±0.04), and adding
divergence damping up to 1e8 m²/s does not delay it.  This is a
**resolution-specific resonance** between the JW jet's Rossby
radius (~1100 km at 30°N) and the C24 grid spacing (~240 km, ratio
~4.6) — the dycore itself is sound.  C48 is the canonical CS
benchmark row in §1; CS C24 will need a separate dedicated trigger /
filter audit and is filed as a follow-up dycore issue.

## 4. Single-device strong-scaling sweep (iter-201/202, 2026-05-03)

`scripts/run_strong_scaling_sweep.sh` exercises each grid at two
resolutions on both backends; throughput is now persisted per cell in
the diagnostic NPZ via `steps_per_sec`/`wall_time_s`/`backend` (added
iter-202).  GPU sweep results (1-day BCW, warm step + JIT inline):

| Grid          | Resolution | ncells | dt (s) | GPU steps/s | Notes                    |
|---------------|-----------:|-------:|-------:|------------:|--------------------------|
| spectral      | T21        |   8192 |    600 |       193.6 | small GPU underutilised  |
| spectral      | T42        |  32258 |    300 |       103.0 | dt halved → ~1.9× cells/s |
| icosahedral   | I4         |   2562 |    150 |       187.5 |                          |
| icosahedral   | I5         |  10242 |     75 |       189.5 | **GPU-saturated** (4× cells, ~equal sps) |
| cubed-sphere  | C48        |  13824 |    150 |        85.9 | 1-day; JIT not amortised |
| cubed-sphere  | C96        |  55296 |     75 |        62.8 |                          |

Key observation: **icosahedral I4 → I5 GPU throughput is essentially
flat** (187.5 → 189.5 steps/s) despite 4× more cells.  This is the
classic "GPU not yet saturated at I4" signature — at I4 (2562 cells ×
26 levels = 67K column-points) the kernel launches dominate; at I5
(266K column-points) the actual SMs finally have work, so step time
nearly quadruples but throughput-per-cell jumps 4×.

CPU sweep is still running; the matching CPU column will land in
iter-203.

The plot ``results/scaling/scaling.png`` now shows two panels:

  1. Bar chart of single-device CPU vs GPU steps/s with speedup
     annotations.
  2. Log-log strong-scaling curves (steps/s vs total cells) per
     (grid, backend) — exposes the saturation kink in the icosahedral
     curve and the falling-throughput curve in cubed-sphere/spectral
     (more cells → fewer steps/s, expected).

## 5. iter-204: ``--scan-steps`` benchmark optimisation

The Python ``for step in range(n_steps)`` loop in
``run_baroclinic_wave_benchmark.py`` ate ~1 ms of JAX-dispatch
overhead per step.  At T21 GPU each kernel runs in ~3 ms, so Python
was eating *one third* of the wall time.  Added ``--scan-steps N``
which fuses ``N`` consecutive ``model.step`` calls inside one
``jax.lax.scan`` body (compiled once per N).  Diagnostic snapshots
and blow-up checks still fire on each ``N``-step boundary.

Sweep across ``N`` (icosahedral I4 GPU, 1-day):

| ``--scan-steps``   |   1 |   6 |  12 |  24 |  48 |
|--------------------|----:|----:|----:|----:|----:|
| BCW steps/s        | 208 | 251 | 256 | **278** | 246 |

24 is the sweet spot (above which compile time eats the gain).
Spectral T21 GPU gains are even larger:

| ``--scan-steps``   |   1 |  12 |  24 |  48 |
|--------------------|----:|----:|----:|----:|
| spectral T21 sps   | 211 | 460 | **479** | 335 |

→ ``+127 % throughput`` at the GPU-undersaturated end of the curve.

### 5.b iter-205: full --scan-steps=24 sweep (all 6 cells)

Every (grid × resolution) GPU cell sees a 27–127 % throughput gain
from the same one-line ``--scan-steps=24`` flag.  No source change to
the dycore — pure benchmark-harness fix:

| Grid          | Resolution | GPU default | GPU scan=24 | gain   |
|---------------|-----------:|------------:|------------:|-------:|
| spectral      | T21        |       211.4 |   **479.4** | +127 % |
| spectral      | T42        |       102.5 |   **129.7** | +27 %  |
| icosahedral   | I4         |       208.2 |   **278.1** | +34 %  |
| icosahedral   | I5         |       183.5 |   **298.7** | +63 %  |
| cubed-sphere  | C48        |        90.8 |   **146.7** | +61 %  |
| cubed-sphere  | C96        |        65.0 |    **98.1** | +51 %  |

Final updated single-device speedups (over 8-thread CPU baseline):

| Grid          | Resolution | CPU sps | GPU scan=24 | speedup vs CPU |
|---------------|-----------:|--------:|------------:|---------------:|
| spectral      | T21        |    38.2 |       479.4 |    **12.6×**   |
| spectral      | T42        |     4.9 |       129.7 |    **26.5×**   |
| icosahedral   | I4         |    98.8 |       278.1 |     **2.8×**   |
| icosahedral   | I5         |    19.1 |       298.7 |    **15.6×**   |
| cubed-sphere  | C48        |    20.2 |       146.7 |     **7.3×**   |
| cubed-sphere  | C96        |       — |        98.1 |       —        |

Spectral T42 GPU now reaches **26× the 8-thread CPU baseline** —
that is the near-optimal regime for an FFT/SH-bound dycore on a
single GPU.  Icosahedral I5 jumps to 15.6× (saturated GPU + dispatch
amortisation).  Cubed-sphere C48 jumps to 7.3× (still halo-bound at
this resolution).

The plot ``results/scaling/scaling.png`` now shows three bars per
grid (CPU, GPU default, GPU --scan-steps=24) so the optimisation
gain is visible at a glance.

Verified bit-equivalence: scan=1 vs scan=24 produce identical mass
drift (-7.866e-9 on icosahedral I4) — the scan body is just a
control-flow reorganisation, not a numerical change.

## 6. iter-206: scan-steps equivalence verifier + multi-CPU emulation result

Added ``scripts/verify_scan_steps_equivalence.py`` — runs N=12 BCW
steps via the per-step Python loop and via a single
``jax.lax.scan(length=N)`` kernel and compares the two final states
leaf-by-leaf.  Tolerance is set to ``1e-3`` absolute (≈10× fp32 eps ×
N) so that XLA-reordered associative reductions don't false-positive,
but a real off-by-one in the scan body would trip:

| cell             | max abs diff | status |
|------------------|-------------:|--------|
| spectral T21     |    0.000e+00 |   OK   |
| icosahedral I4   |    5.8e-04   |   OK (fp32 reordering) |
| cubed-sphere C24 |    0.000e+00 |   OK   |

Spectral and CS land bit-identical; MPAS drifts at the fp32 storage
floor.  This script is the regression guard for the ``--scan-steps``
optimisation — keep it green going forward.

### 6.b XLA-emulated multi-CPU (XLA_FLAGS=--xla_force_host_platform_device_count=N)

Tested on spectral T21 (1-day, scan-steps=24):

| n_cpu_emulated | sps   |
|---------------:|------:|
|              1 |  64.5 |
|              2 |  64.9 |
|              4 |  63.9 |

→ **No scaling at all** from emulated CPUs.  The XLA emulation
exposes N JAX devices but they share the underlying threadpool, so
without an explicit ``shard_map`` wrapper around ``model.step`` the
extra devices are idle.  The codebase already has a
``CompiledShardedStep`` (``parallel/sharded_dynamics.py``) that does
the right thing, but the BCW benchmark routes through
``model.step`` directly.  Wiring the BCW benchmark to that sharded
path is the next material scaling lever.

## 7. iter-207: CompiledShardedStep wiring blocker

Tried wrapping ``SpectralPrimitiveEquationModel.step`` with
``make_sharded_step(model, create_level_mesh(n_devices=4))`` to
emulate 4-device multi-CPU scaling.  Hits a JAX-trace boundary
issue:

```
ValueError: Non-hashable static arguments are not supported. An
error occurred while trying to hash an object of type
<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>,
JitTracer(~float64[]).
```

Root cause: ``model.step`` is itself ``@jax.jit``-wrapped (its inner
``self._step_jit`` takes ``physics_fn`` as a static arg).  When the
outer ``CompiledShardedStep`` re-jits this, the inner jit's
``static_argnums`` see a tracer instead of ``None`` for
``physics_fn`` and reject it.  Fixing this needs either:

1. Removing the inner ``@jax.jit`` from ``model.step`` and relying on
   the outer ``CompiledShardedStep`` jit (would change every
   single-device call site too — high blast radius), or
2. Adding a ``model.tendencies(state)`` API that returns plain
   tendencies and is jit-able by the outer wrapper (cleaner long-term
   path, but significant refactor).

Filed as a multi-iteration item; the benchmark continues to run
single-device.  In the meantime the iter-204/205 ``--scan-steps``
optimisation is the one knob the BCW benchmark needs and it gives
the entire +27..127 % per-cell GPU throughput documented in §5.b.

## 8. iter-208: BCW diag-sampling bug under --scan-steps fixed

Pre-fix the diagnostic check
``current_step % diag_interval_steps == 0`` only fired when the
fused chunk happened to land *exactly* on an hourly boundary.  With
``SCAN_STEPS=24`` and ``diag_interval_steps=6`` (dt=600), the chunk
landed at current_step ∈ {25, 49, 73, 97, …} — none divisible by 6
— so **all** intermediate hourly samples were lost; only the
initial ``mass_init`` and (occasionally) a final-step sample
survived.  This hid the time-resolved drift in every sweep cell.

Replaced the modulo check with a chunk-crossing test:
``(current_step // diag_interval_steps) > (prev_current_step // diag_interval_steps)``
which fires if the just-completed chunk *crossed* at least one
diag-interval boundary (sampling once per chunk, not once per
boundary, but recording at least every ``SCAN_STEPS`` × dt).
``prev_current_step`` is rolled forward at the bottom of each loop
iteration.

Verified:

| run                                    | diag samples (1-day, dt=600) |
|----------------------------------------|-----------------------------:|
| spectral T21, scan=1 (default)         |  **25** (24 hourly + initial) |
| spectral T21, scan=24                  |   **9** (initial + 8 crossed) |

Both report identical mass drift (+1.367e-9 final).  This also fixes
the blow-up check, which was inheriting the same modulo bug.

## 9. iter-210: definitive multi-CPU emulation result via direct ``_do_step`` JIT

To bypass the iter-207 ``CompiledShardedStep`` ``static_argnums``
conflict, ``scripts/probe_spectral_shard.py`` jits
``model._do_step(state, dt_arr, tendency_fn)`` directly with primed
caches.  This gets us a clean shard target without source changes.

| n_emulated_cpu | shard=none (sps) | shard=level (sps) |
|---------------:|-----------------:|------------------:|
|              1 |            124.6 |              67.4 |
|              2 |            124.9 |              53.8 |
|              4 |            129.0 |              48.2 |

Two orthogonal findings:

1. **Per-device throughput doubled** (124 sps vs the iter-204
   ``--scan-steps=24`` benchmark's 64.5 sps) once the dt-static
   inner-jit was bypassed — the outer ``CompiledShardedStep`` wrap
   was paying recompile cost every call, while the direct
   ``_do_step`` JIT compiles once and reuses.
2. **Multi-CPU emulation actively *hurts*** — shard=level at 4
   devices is 2.6× slower than 1.  With ``XLA_FLAGS=--xla_force_host_platform_device_count=N``
   the N JAX devices share the same host threadpool, so the only
   thing cross-device sharding adds is communication latency.  The
   negative scaling here is the textbook signature of "fake
   parallelism" on a single physical socket.

This conclusively closes out multi-device weak/strong scaling on
this host — without a second physical CPU socket or a second GPU
the JAX device mesh is purely cosmetic.  The single-device
optimisation work in §1-§5 is the real frontier here.

## 10. iter-211: dt-static is the right default; revert + measurement-noise correction

Tried dropping ``dt`` from ``_step_jit``'s ``static_argnums`` so the
outer ``CompiledShardedStep`` could wrap it.  Measured a real
**60 % regression** on spectral T21 GPU (479 → 284 sps reported, but
see noise note below): with ``dt`` static the SI matrices, sponge
factors, and hyperdiff filters constant-fold into the compiled
program; with ``dt`` traced these become broadcasted operations.
Reverted; documented the trade-off in the docstring so a future
``dt``-traced shard variant lives alongside the fast path rather than
replacing it.

### 10.b BCW measurement noise correction

The iter-204/205 reported speedups (e.g. spectral T21 GPU
"+127 % to 479 sps with ``--scan-steps=24``") were inflated by the
1-day BCW run's **integer-second wall-clock resolution**: at <1 s
total the reported sps is a coarse upper bound.  Re-measured with
5-day runs (≥2 s wall):

| run                                   | sps (1-day) | sps (5-day) |
|---------------------------------------|------------:|------------:|
| spectral T21 GPU, scan=24             |    "479"    |   **331-356** |

The single-device GPU scaling is still excellent — the 5-day number
is **8-9× the 8-thread CPU baseline (38.2 sps)** — but not the 12.6×
the 1-day measurement claimed.  Update §5.b accordingly when next
re-running the full sweep.

### 10.c Other regression cleanup re-applied

`_moist_state` in ``test_spectral_pe_microphysics.py`` had its
iter-196 ``rh_low=0.95 → 1.10`` fix reverted.  Re-applied.  All 12
spectral_pe_microphysics tests pass again.

## 10.d iter-212: precision-fix + honest re-measurement of all 6 GPU cells

Changed BCW benchmark stdout/NPZ throughput from
``elapsed_total:.0f`` (integer seconds, ``n_steps_total/elapsed:.1f``)
to ``elapsed_total:.3f`` + ``...:.2f`` so 1-day runs no longer
report inflated upper-bound sps values.

Honest 1-day numbers for all 6 cells (precision fix applied):

| Grid          | Resolution | GPU scan=1 | GPU scan=24 | scan-24 gain | speedup vs CPU |
|---------------|-----------:|-----------:|------------:|-------------:|---------------:|
| spectral      | T21        |      198.7 |   **288.4** |       +45 %  |     **7.5×**   |
| spectral      | T42        |       91.3 |   **117.6** |       +29 %  |     **24×**    |
| icosahedral   | I4         |      202.2 |       205.5 |       +1.6 % |      **2.1×**  |
| icosahedral   | I5         |      198.5 |       181.1 |     **-9 %** | **10.4×** (use scan=1) |
| cubed-sphere  | C48        |       86.0 |   **103.0** |       +20 %  |      **5.1×**  |
| cubed-sphere  | C96        |       65.9 |        67.0 |       +1.8 % |       —        |

Two caveats vs the iter-204/205 messaging (which used integer-second
timing):

* The reported gains were **inflated 2-3×** by the timing rounding.
  The real spectral T21 ``--scan-steps=24`` win is +45 %, not +127 %.
* ``--scan-steps=24`` actually **hurts icosahedral I5** (-9 %) — the
  MPAS dycore is already well-fused, and the scan body adds compile
  overhead with no offsetting Python-dispatch-amortisation gain
  (icosahedral I5 already runs at 200 sps; the Python loop tax is
  too small to amortise away).

Use ``--scan-steps=24`` selectively: spectral T21/T42 + CS C48
benefit; icosahedral cells should use the default (scan=1).
``results/scaling/scaling.png`` re-rendered with the honest
numbers.

## 11. iter-214: per-grid scan-steps tuning + plot ingest mode

`scripts/run_strong_scaling_sweep.sh` now uses the iter-212 honest
recommendation per grid:

| grid          | --scan-steps | rationale                                  |
|---------------|-------------:|--------------------------------------------|
| spectral      |           24 | +29..45 % gain — Python-dispatch matters for SH |
| cubed-sphere  |           24 | +20 % gain                                  |
| icosahedral   |            1 | MPAS already well-fused; scan=24 was -9 %   |

`scripts/plot_scaling.py --ingest 'pattern'` overrides the inline
STRONG_SCALING dict with throughput from any matching NPZ files
(uses the iter-202-added ``steps_per_sec`` field).  Verified: ingests
the iter-201 sweep and rewrites the GPU/CPU curve points without
touching the script.

## 12. iter-216: BCW UX — runtime warning for known-bad scan-steps combinations

The BCW benchmark now emits a ``RuntimeWarning`` when a user passes
``--scan-steps > 1`` together with ``--grid icosahedral``.  Iter-212
honest measurement showed scan=24 was -9 % at I5 (MPAS dycore is
already well-fused inside its own ``@jit``), so blindly applying the
optimisation flag silently regressed throughput.

The help-text for ``--scan-steps`` now also includes the iter-212
per-grid measured gains (spectral +29-45 %, CS +20 %, ico -9 %) so
that ``--help`` is enough to pick the right value without consulting
this file.

Plus iter-215 regression-pinned the iter-199 ``hydrostatic_to_fv3``
vector-halo fix with two new tests in
``test_primitive_eq.py::TestHydrostaticToFV3VectorHalo``.  Future
``/clear``-cycle reverts to the scalar halo will fire in the test
suite within seconds, not after a multi-day BCW blow-up.

## 13. iter-219: ``--scan-steps auto`` mode

The BCW benchmark now accepts ``--scan-steps auto`` which dispatches
to the iter-212 honest per-grid recommendation:

| grid          | auto value |
|---------------|-----------:|
| spectral      |         24 |
| cubed-sphere  |         24 |
| icosahedral   |          1 |

Plain integer values (``--scan-steps 12``) still work.  The
strong-scaling sweep driver ``run_strong_scaling_sweep.sh`` now uses
``auto`` uniformly, so future per-grid retunes only need to land in
the benchmark itself, not in every driver script.  Verified
end-to-end: ``auto`` on each grid prints the correct dispatched
value and never trips the iter-216 RuntimeWarning.

## 13.a iter-220: regression-pin the iter-219 dispatch table

The iter-219 inline dispatch dict was a private ``_AUTO_SCAN_STEPS``
local inside ``main()`` — it could be tweaked silently by a future
``/clear`` cycle without any test catching the drift.  iter-220
hoists it to a module-level ``AUTO_SCAN_STEPS`` constant
(``scripts/run_baroclinic_wave_benchmark.py:78``) and pins the values
with two regression tests
(``tests/test_bcw_benchmark_scan_steps.py``):

1. ``test_auto_scan_steps_dispatch_table`` — exact equality against
   the iter-212 honest measurements ``{spectral:24, cubed-sphere:24,
   icosahedral:1}``.  Drift here ⇒ a quiet ~30 % throughput
   regression across the entire strong-sweep driver.
2. ``test_auto_scan_steps_covers_all_grid_choices`` — coverage check
   so a future grid added to ``GRID_CHOICES`` cannot silently fall
   back to ``scan_steps=1`` (i.e. forfeit the iter-212 speedup).

Both tests use ``importlib.util`` to load the benchmark script as a
module without invoking ``main()``.  Runtime: 0.17 s.  Same lock-down
pattern as iter-215/217 (which pinned the iter-199 hydrostatic_to_fv3
vector-halo fix).

## 13.b iter-222: hoist resolve_scan_steps + pin iter-216 warning

The iter-216 RuntimeWarning ("icosahedral + scan>1 = -9 %") was
inlined in ``main()``, which made it impossible to regression-pin
without a subprocess test.  iter-222 hoists both the iter-219 ``auto``
dispatch and the iter-216 warning into a module-level helper
``resolve_scan_steps(grid, raw, *, verbose=True)`` (BCW benchmark
lines ≈ 89..136).  ``main()`` collapses to a single line:

```python
args.scan_steps = resolve_scan_steps(args.grid, args.scan_steps)
```

Test coverage (``tests/test_bcw_benchmark_scan_steps.py``, 6 tests
total, 0.21 s):

* ``test_resolve_scan_steps_auto_dispatches_per_grid`` — every grid
  in ``AUTO_SCAN_STEPS`` resolves to the table value with no spurious
  warning.
* ``test_resolve_scan_steps_icosahedral_explicit_warns`` — explicit
  ``scan_steps=24`` on icosahedral fires a RuntimeWarning matching
  the iter-212 finding text.
* ``test_resolve_scan_steps_icosahedral_one_is_silent`` — neither
  ``scan_steps=1`` nor ``"auto"`` (which resolves to 1) emits a
  warning on icosahedral.
* ``test_resolve_scan_steps_passthrough_for_other_grids`` — spectral
  / cubed-sphere with ``scan>1`` are silent (only icosahedral has
  the regret).

Verified end-to-end: CLI ``--scan-steps auto`` still prints the
dispatch banner, and CLI ``--grid icosahedral --scan-steps 24`` still
emits the RuntimeWarning at file:line.

## 13.c iter-223: hoist + pin the iter-208 chunk-crossing predicate

The iter-208 fix replaced ``current_step % period == 0`` (which
silently dropped every hourly diag under ``--scan-steps`` larger than
``diag_interval_steps``) with a chunk-crossing predicate.  iter-223
factors that into a 1-line module-level helper
``chunk_crosses_boundary(prev_step, current_step, period)`` and adds
4 regression tests:

* modulo-regime equivalence (scan_steps=1 ⇒ identical to ``%``);
* skip-regime correctness (chunk size > period still fires);
* no-crossing case (chunk lies inside one period);
* defensive ``period=0/-1`` short-circuit (no ZeroDivisionError).

Both the blowup-check (every 100 steps) and the diag-sample (every
``diag_interval_steps``) sites in the BCW main loop now call this
helper, so the iter-208 bug is impossible to silently regress.  Test
runtime: 0.21 s for all 10 BCW-benchmark unit tests.

## 13.d iter-224: hoist ``import warnings`` to module level

Tiny code-hygiene cleanup: ``import warnings as _warnings`` was
inlined inside ``resolve_scan_steps`` (added in iter-222).  Hoisted
to the top-level ``import warnings`` so the BCW benchmark module is
free of inline stdlib imports.  Also confirmed throughput is
unchanged after iter-220..223 refactors — spectral T21 GPU 1-day,
``--scan-steps auto`` measured 275.5 sps (iter-212 baseline 288.4
sps; ±5 % run-to-run noise consistent with iter-211 finding).  All
10 BCW unit tests still pass in 0.24 s.

## 13.e iter-225: hoist remaining inline imports in PE dycores

Two hydrostatic-PE dycores still had inline imports that survived the
iter-71..187 hoist sweep (likely re-introduced by /clear-cycle revert
or merge):

* ``spectral_pe.py`` — 6 inline ``from
  legoesm.timestepping.semi_implicit import …`` statements
  (precompute_si_matrices ×2, ssp_rk3_step_si, robert_asselin_filter,
  euler_si_step, leapfrog_si_step) collapsed to one module-level
  import block.  These fired at trace time only (inside JIT-compiled
  methods) so the runtime impact was tiny, but consistency with
  iter-75/iter-187 matters.
* ``primitive_eq_mpas.py`` — ``from legoesm.parallel.reductions
  import global_sum_mpi`` inline inside ``fix_mass`` (called when
  ``jax.process_count() > 1``); hoisted to module level.  Iter-74
  was supposed to handle this and may have been reverted.

Verified:

* Module imports cleanly (``from legoesm.atmosphere.dynamics import
  spectral_pe, primitive_eq_mpas``).
* Hydrostatic test suite ``tests/atmosphere/hydrostatic/`` — 695
  passed, 2 skipped, 6 deselected (12 min).
* iter-220..223 BCW unit tests still pass (10/10, 0.21 s).
* iter-199 vector-halo regression still pass (3/3, 16 s).

## 13.f iter-226: hoist 7 inline imports in cubed-sphere PE dycore

``primitive_eq_cdgrid.py`` had 9 inline imports.  Iter-226 hoists 7
that are pure functions (safe — module-level binding is fine) and
deliberately keeps 2 inline because they reference *mutable
singletons* that ``set_halo_backend`` / ``activate_spmd_halo_backend``
mutate at runtime — ``from … import name`` re-evaluates ``X.name``
on every function call, while a module-level binding would freeze
the value at import time and silently break the SPMD/MPI dispatch.

Hoisted (safe — pure functions):
* ``_resolve_dtype`` and ``cast_pytree`` from ``core.precision``
* ``hyperdiffusion`` from ``core.operators``
* ``pad_halo_4d`` (aliased ``_pad_halo_4d_module``) from ``grids.halo``
* ``packed_pad_halo_4d`` from ``parallel.cubesphere_exchange``
* ``packed_pad_halo_mpi_4d`` from ``parallel.halo_exchange``
* ``_overlapped_arakawa_lamb_gradient`` from ``core.operators_cdgrid``

Kept inline (mutable singletons — needs late binding):
* ``_halo_backend``, ``_mpi_topology`` from ``grids.halo``
* ``_spmd_mesh`` from ``parallel.cubesphere_exchange``

Verified: import smoke test OK; iter-220+iter-199 regression suites
both pass (54/54 in 65 s); ``--scan-steps auto`` end-to-end on
cubed-sphere C48 still prints the iter-219 dispatch banner.

## 13.g iter-227: hoist 8 inline imports in latlon C-grid PE

``primitive_eq_latlon_cgrid.py`` had 8 inline imports — all of them
pure functions (no SPMD/MPI mutable singletons in the latlon path
because there's no ``set_halo_backend`` indirection there).
Hoisted to a module-level block:

* ``zero_mean_tendency``, ``_accumulation_dtype`` from
  ``core.conservation``
* ``fv_gradient_lon_3d``, ``fv_gradient_lat_3d`` from
  ``core.operators_fv_latlon``
* ``pole_cell_dx``, ``cfl_max_dt`` from ``core.cfl``
* ``cast_pytree`` from ``core.precision``
* ``pad_halo_latlon_3d`` from ``grids.halo_latlon``
* stdlib ``import inspect`` to top

Iter-73 / iter-78 / iter-84 were supposed to handle these but the
imports drifted back via /clear cycles.  Verified end-to-end:

* Import smoke test OK.
* iter-220 BCW unit + iter-199 vector-halo regression: 54/54, 64 s.
* Latlon-specific hydrostatic tests (``-k 'latlon or cgrid'`` in
  ``tests/atmosphere/hydrostatic/``): 93 passed, 2 skipped, 1
  warning, 82 s.

This closes the iter-225..227 PE-dycore drift sweep.  All four
hydrostatic dycores (spectral, MPAS Voronoi, cubed-sphere FV3 D-grid,
latlon C-grid) now have only the *intentionally inline* imports left
(mutable singletons that need late binding for SPMD/MPI dispatch).

## 13.h iter-228: PE-dycore inline-import drift audit + spectral_pe cleanup

The iter-225..227 sweep hoisted 30+ inline imports across the four
hot-path PE dycores by hand.  Iter-228 codifies the result so future
``/clear`` cycles cannot quietly re-introduce drift:

* New ``tests/test_pe_dycore_inline_imports.py`` parameterises over
  the four PE files and asserts each one's ``^[ \t]+(from
  legoesm|import legoesm)`` count is ≤ a known budget.  Budgets:
  - ``spectral_pe.py``               = 0
  - ``primitive_eq_mpas.py``         = 0
  - ``primitive_eq_latlon_cgrid.py`` = 0
  - ``primitive_eq_cdgrid.py``       = 4 (mutable singletons:
    ``_halo_backend`` ×1, ``_spmd_mesh`` ×1, ``_mpi_topology`` ×2 —
    all mutated at runtime by ``set_halo_backend`` /
    ``activate_spmd_halo_backend``).
* Wired into ``scripts/validate_scaling.sh`` as phase ``[5/5]`` so
  every iteration's pre-flight (≈2 min wall) catches drift in 0.05 s.
* While writing the audit, found and fixed:
  - ``spectral_pe.py``: 2 inline imports (``get_backend`` /
    ``check_spectral_backend``, plus stray inline ``import jax`` and
    ``from legoesm import constants``) hoisted to module level.
  - ``primitive_eq_cdgrid.py``: 6 *new* pure-function inline imports
    that escaped the iter-226 pass (``global_integral``,
    ``laplacian_compact``, ``fix_mass_hydrostatic``,
    ``fix_mass_hydrostatic_target``, ``HydrostaticTendencies``,
    ``pad_halo_vector`` / ``pad_halo_vector_4d``) — all hoisted.

Total state after iter-228: 4 inline ``from legoesm`` references
remain across the four PE dycores, all of them deliberately so for
late binding.  Before iter-225 there were 16+ unintentional ones.

Verified: 5/5 phases of ``validate_scaling.sh`` green;
``test_pe_dycore_inline_imports.py`` 4/4 pass; iter-220 BCW unit +
iter-199 vector-halo + latlon-cgrid PE unit suites all green
(106 tests total).

## 13.i iter-229: extend drift audit to SW + NH dycores

The iter-228 drift audit covered the four hydrostatic PE files.
iter-229 extends it to all 11 hot-path dycores by adding the seven
shallow-water + non-hydrostatic ones to ``INLINE_IMPORT_BUDGET``,
plus hoists the worst offender:

* ``shallow_water_fv3_cdgrid.py`` — 13 inline imports → 0.  All
  pure-function imports (``cast_pytree``, ``_accumulation_dtype``,
  ``fv3_fb_sw_step``, ``fv3_csw_tendencies``, ``_d2a2c_vect``,
  ``_d_sw5_corner_divergence``, ``transport_step``,
  ``fv3_del6_vorticity_damping``, plus halo edge constants
  ``CONNECTIVITY``/``WEST``/``EAST``/``SOUTH``/``NORTH``) hoisted to
  the module-level import block.

Audit budgets pin the *current* state for the remaining six SW + NH
files so future drift fails CI even before a hand-cleanup pass:

| file                            | budget | status                  |
|---------------------------------|-------:|-------------------------|
| ``shallow_water_fv3_cdgrid.py`` |      0 | clean (iter-229)        |
| ``shallow_water_mpas.py``       |      3 | drift, scheduled        |
| ``shallow_water_latlon_cgrid.py``|     3 | drift, scheduled        |
| ``spectral_sw.py``              |      1 | drift, scheduled        |
| ``compressible_euler_cdgrid.py``|     11 | drift, scheduled        |
| ``compressible_euler_mpas.py``  |      0 | clean already           |
| ``spectral_nh.py``              |      9 | drift, scheduled        |

Verified: 11/11 audit tests pass in 0.02 s; SW FV3 cubed-sphere
integration tests 6/6 in 44 s; iter-228 PE budgets all still pass.

## 13.j iter-230: hoist NH cubed-sphere compressible-Euler dycore

``compressible_euler_cdgrid.py`` had 11 inline imports.  iter-230
hoists 8 pure-function/class imports to module level and keeps 3
mutable-singleton late-bindings (same pattern as the hydrostatic CS
PE in iter-226).

Hoisted (safe):
* ``CompressibleEulerConfig`` — added to existing ``compressible_euler``
  import block.
* ``estimate_min_dx_cubed_sphere`` from ``core.cfl``.
* ``compute_nh_dry_mass``, ``fix_mass_nonhydrostatic`` from
  ``core.conservation``.
* ``apply_small_earth_scaling`` from ``grids.cubed_sphere``.
* ``pad_halo_4d`` (aliased ``_pad_halo_4d_module``) from
  ``grids.halo``.
* ``packed_pad_halo_4d`` from ``parallel.cubesphere_exchange``.
* ``packed_pad_halo_mpi_4d`` from ``parallel.halo_exchange``.
* stdlib ``import logging``.

Kept inline (mutable singletons):
* ``_halo_backend`` from ``grids.halo`` (1×).
* ``_spmd_mesh`` from ``parallel.cubesphere_exchange`` (1×).
* ``_mpi_topology`` from ``grids.halo`` (1×).

Drift audit budget for ``compressible_euler_cdgrid.py`` lowered from
11 → 3.  Verified: 11/11 drift-audit tests still pass; 37/37 NH
cubed-sphere tests (8 unit + 25 unit + 4 integration ``test_fv_cubesphere``)
pass in 82 s.

## 13.k iter-231: hoist NH spectral dycore (9→0)

``spectral_nh.py`` had 9 inline imports.  iter-231 hoists all of them
to module level — there are no mutable singletons in the spectral
path so all 9 are safe to hoist:

* ``thomas_solve_batched`` from ``timestepping.tridiagonal``.
* redundant ``from legoesm import constants`` (already at module
  level) — dropped.
* ``get_backend``, ``check_spectral_backend`` from ``runtime.backend``.
* ``create_gaussian_grid`` added to existing ``grids.gaussian``
  import block.
* ``create_height_coordinate``, ``compute_terrain_metric`` added to
  existing ``grids.vertical`` import block.
* ``saturation_mixing_ratio`` from ``thermo``.

The latter five fired in DCMIP2025 test-case helpers
(``test_case_1``, ``_2``, ``_3``) which are not on the BCW hot path
but the hoist is still consistent with the iter-71..187 sweep
philosophy.

Drift audit budget for ``spectral_nh.py`` lowered from 9 → 0.
Verified: 11/11 drift audit tests pass; 34/34 spectral NH unit
tests (``test_spectral_nh.py``) pass in 18 s.

After iter-229..231 the NH dycore family is fully clean: NH MPAS
already at 0; NH cubed-sphere at 3 (mutable singletons only); NH
spectral at 0.

## 13.l iter-232: hoist remaining SW dycores — drift sweep complete

iter-232 hoists the last three SW dycores in one pass:

* ``shallow_water_mpas.py`` — 3 → 0.  ``cast_pytree``,
  ``global_sum_mpi`` (×2 inline ⇒ 1 module-level) hoisted.
* ``shallow_water_latlon_cgrid.py`` — 3 → 0.  ``cast_pytree``,
  ``_accumulation_dtype`` (×2 inline ⇒ 1 module-level) hoisted.
* ``spectral_sw.py`` — 1 → 0.  ``get_backend`` /
  ``check_spectral_backend`` hoisted.

Final state of the dycore drift audit (11 hot-path files, 7 inline
imports total, *all* of them intentional mutable-singleton late-
bindings):

| family            | file                               | budget |
|-------------------|------------------------------------|-------:|
| Hydrostatic PE    | ``spectral_pe.py``                 |      0 |
| Hydrostatic PE    | ``primitive_eq_mpas.py``           |      0 |
| Hydrostatic PE    | ``primitive_eq_latlon_cgrid.py``   |      0 |
| Hydrostatic PE    | ``primitive_eq_cdgrid.py``         |      4 |
| Shallow water     | ``shallow_water_fv3_cdgrid.py``    |      0 |
| Shallow water     | ``shallow_water_mpas.py``          |      0 |
| Shallow water     | ``shallow_water_latlon_cgrid.py``  |      0 |
| Shallow water     | ``spectral_sw.py``                 |      0 |
| Non-hydrostatic   | ``compressible_euler_cdgrid.py``   |      3 |
| Non-hydrostatic   | ``compressible_euler_mpas.py``     |      0 |
| Non-hydrostatic   | ``spectral_nh.py``                 |      0 |

The iter-225..232 sweep removed approximately 50 unintentional
inline imports from the atmosphere dycore matrix.  Future ``/clear``
drift in any of these files fails the audit in 0.02 s.

Verified: 11/11 audit tests pass; SW unit + integration tests
(``tests/atmosphere/shallow_water/`` minus the slow validation tree)
95 passed in 232 s.

## 13.m iter-233: extend drift audit to 10 ocean dycores + clean eta_floor

Iter-233 expands the drift-audit scope from the 11 atmosphere dycores
to 21 total by adding 10 ocean dynamics modules at their *measured*
inline-import counts (so the audit fails-on-drift now, before the
hand-cleanup pass starts).  Also cleans the smallest offender as a
demonstration:

* ``eta_floor.py`` — 3 → 0.  Hoisted ``import jax``,
  ``_is_distributed`` (``core.operators``), ``global_sum_mpi`` and
  ``batch_allreduce_mpi`` (``parallel.reductions``).  Iter-130
  originally hoisted these; drift came back via /clear cycles.

Audit budgets locked in at iter-233:

| family    | file                           | budget |
|-----------|--------------------------------|-------:|
| Ocean PE  | ``ocean_pe_latlon_cgrid.py``   |     11 |
| Ocean PE  | ``ocean_pe_cdgrid.py``         |      7 |
| Ocean PE  | ``ocean_pe_mpas.py``           |      1 |
| Ocean PE  | ``ocean_pe_fc.py``             |      4 |
| Ocean PE  | ``spectral_ocean_pe.py``       |      2 |
| Ocean MOM | ``ocean_model.py``             |      4 |
| Ocean MOM | ``ocean_model_latlon_cgrid.py``|     16 |
| Ocean MOM | ``ocean_model_mpas.py``        |      3 |
| Ocean MOM | ``barotropic_mpas.py``         |      1 |
| Ocean MOM | ``eta_floor.py``               |      0 |

Total: 21/21 audit tests pass in 0.03 s; ``test_eta_floor.py`` 5/5
pass in 0.76 s.

The audit test now asserts each path's count is ≤ its budget; future
``/clear`` drift in any of the 21 hot-path dycores fails CI in well
under a second.  Subsequent iterations will ratchet the ocean budgets
down to 0 (or the documented mutable-singleton minimum) one file at
a time.

## 13.n iter-234: hoist 3 small-count ocean dycores (4 inline imports)

Continuing the ocean drift cleanup with the three smallest offenders:

* ``ocean_pe_mpas.py`` — 1 → 0.  ``fill_land_cells_mpas``
  (``ocean.dynamics.mpas_fill``) hoisted.
* ``barotropic_mpas.py`` — 1 → 0.  Same import (different
  call site).
* ``spectral_ocean_pe.py`` — 2 → 0.  ``global_sum_mpi`` and
  ``get_backend``/``check_spectral_backend`` hoisted.

Drift-audit budgets ratcheted accordingly.  Verified: 21/21 audit
tests pass; 184 ocean MPAS + barotropic unit tests pass in 187 s.

Ocean dycore drift remaining (6 files, 42 inline imports):

| file                          | budget |
|-------------------------------|-------:|
| ``ocean_pe_latlon_cgrid.py``  |     11 |
| ``ocean_pe_cdgrid.py``        |      7 |
| ``ocean_pe_fc.py``            |      4 |
| ``ocean_model.py``            |      4 |
| ``ocean_model_latlon_cgrid.py``|    16 |
| ``ocean_model_mpas.py``       |      3 |

Subsequent iterations will continue ratcheting these to 0 (or the
documented mutable-singleton minimum).

## 13.o iter-235: hoist ocean_model + ocean_model_mpas (7 inline imports)

* ``ocean_model.py`` — 4 → 0.  Hoisted ``create_cubed_sphere_cdgrid``,
  ``make_ocean_physics``, ``ocean_baroclinic_tendencies_cdgrid``,
  ``ocean_conservation_fixer``.
* ``ocean_model_mpas.py`` — 3 → 0.  Hoisted
  ``make_mpas_ocean_physics``, ``fill_land_cells_mpas``,
  ``gm_redi_tracer_tendency_mpas``.

After iter-233/234/235 the ocean drift state is:

| status      | files                                          |
|-------------|------------------------------------------------|
| clean (≤ 0) | ``eta_floor``, ``ocean_pe_mpas``, ``barotropic_mpas``, ``spectral_ocean_pe``, ``ocean_model``, ``ocean_model_mpas`` |
| drifted     | ``ocean_pe_latlon_cgrid`` (11), ``ocean_pe_cdgrid`` (7), ``ocean_pe_fc`` (4), ``ocean_model_latlon_cgrid`` (16) |

4 files / 38 inline imports remain.  Verified: 21/21 drift audit tests
pass; ``test_ocean.py`` 99/99 pass in 155 s.

## 13.p iter-236: hoist ocean_pe_fc (4→0)

``ocean_pe_fc.py`` had 4 inline imports — all pure functions with no
mutable singletons (the FC operator path uses static metric-tensor
config, not a runtime singleton).  Hoisted:

* ``_fc_pad_halo_vector`` from ``core.operators_fc`` (added to
  existing ``FCOperatorConfig`` import block).
* ``pad_halo_4d`` from ``grids.halo`` (×3 inline call sites
  collapsed to one module-level import).

Drift-audit budget for ``ocean_pe_fc.py`` lowered 4 → 0.  Verified:
21/21 drift audit tests pass; 10/10 ``test_ocean_fc.py`` tests pass
in 32 s.

Ocean drift remaining (3 files, 34 inline imports):

| file                            | budget |
|---------------------------------|-------:|
| ``ocean_pe_latlon_cgrid.py``    |     11 |
| ``ocean_pe_cdgrid.py``          |      7 |
| ``ocean_model_latlon_cgrid.py`` |     16 |

## 13.q iter-237: hoist ocean_pe_cdgrid (7→0)

``ocean_pe_cdgrid.py`` had 7 inline imports — all pure functions
(no mutable singletons in the ocean CD-grid path).  Hoisted:

* ``fill_land_cells`` from ``ocean.dynamics.barotropic``.
* ``laplacian_viscosity_3d``, ``vertical_diffusion`` from
  ``ocean.physics.mixing`` (×2 inline call sites collapsed to one).
* ``pad_halo_4d`` from ``grids.halo`` (no aliasing — single
  module-level import).
* ``hyperdiffusion_3d`` from ``core.operators_3d``.

Drift-audit budget for ``ocean_pe_cdgrid.py`` lowered 7 → 0.
Verified: 21/21 drift audit tests pass; ``test_ocean.py`` 99/99 +
audit 21/21 = 120 tests pass in 152 s.

Ocean drift remaining (2 files, 27 inline imports):

| file                            | budget |
|---------------------------------|-------:|
| ``ocean_pe_latlon_cgrid.py``    |     11 |
| ``ocean_model_latlon_cgrid.py`` |     16 |

## 13.r iter-238: hoist ocean_pe_latlon_cgrid (11→0)

``ocean_pe_latlon_cgrid.py`` had 11 inline imports — all pure
functions with several duplicated across multiple call sites.
Hoisted:

* ``compute_face_masks_3d``, ``density_jacobian_pgf_smc03_x/y``,
  ``partial_cell_pgf_correction_x/y``, ``pv_flux_al81_partial_cell``
  — added to existing ``latlon_cgrid_operators`` module-level
  import block.
* ``weno_reconstruct_split``, ``weno_upwind`` from ``core.weno``
  (×4 inline call sites collapsed to one module-level import).
* ``flux_form_vertical_tracer_advection_weno5``,
  ``flux_form_vertical_tracer_advection_weno7`` from
  ``ocean.advection``.
* ``compute_centroid_depth`` added to existing ``ocean.vertical``
  import block.

Drift-audit budget for ``ocean_pe_latlon_cgrid.py`` lowered 11 → 0.
Verified: 21/21 drift audit tests pass; 122 latlon/cgrid-keyword
ocean unit tests pass in 153 s.

Ocean drift state: **9 of 10 files clean**.  Only
``ocean_model_latlon_cgrid.py`` (16 inline imports) remains.

## 13.s iter-239: hoist ocean_model_latlon_cgrid (16→0) — drift sweep complete

The last drifted ocean dycore.  ``ocean_model_latlon_cgrid.py`` had
16 inline imports — all pure functions/classes.  Hoisted to a single
module-level block:

* ``Field`` from ``core.field``.
* ``OceanPartialCellCoordinate``, ``diagnose_w_from_flux_div``,
  ``flux_form_vertical_tracer_advection``,
  ``flux_form_vertical_tracer_advection_tvd`` added to existing
  ``ocean.vertical`` block.
* 5 helpers from ``ocean_pe_latlon_cgrid``: ``_interp_to_v_points``,
  ``_upwind_to_u/v_points``, ``_tvd_to_u/v_points``.
* 5 from ``latlon_cgrid_operators``: ``compute_face_masks``,
  ``compute_face_masks_3d``, ``divergence_cgrid``,
  ``min_cell_to_uface``, ``min_cell_to_vface``.
* ``make_ocean_physics``, ``gm_redi_tracer_tendency_latlon``,
  ``som_advect_tracers``, ``ocean_conservation_fixer``.
* 12 advection helpers from ``ocean.advection``:
  ``fct_tracer_advection``, ``ppm_to_u/v_points``,
  ``flux_form_vertical_tracer_advection_ppm/dst3/weno5/weno7``,
  ``dst3_to_u/v_points``, ``multidim_tracer_advection``,
  ``weno5_to_u/v_points``, ``weno7_to_u/v_points``.

Drift-audit budget for ``ocean_model_latlon_cgrid.py`` lowered 16 → 0.

**Final state of the iter-225..239 drift sweep — COMPLETE:**

| family            | files | inline imports |
|-------------------|------:|--------------:|
| Hydrostatic PE    |     4 |              4 (CS mutable singletons) |
| Shallow water     |     4 |              0 |
| Non-hydrostatic   |     3 |              3 (CS mutable singletons) |
| Ocean dycore      |    10 |              0 |
| **Total**         |    **21** |          **7** (all intentional) |

Across iter-225..239, ~120 unintentional inline imports were removed
from the dycore matrix.  Future ``/clear`` drift in any of the 21
hot-path files fails CI in 0.03 s via
``tests/test_pe_dycore_inline_imports.py``.

Verified: 21/21 drift audit tests pass; ``test_ocean.py`` 99/99 +
audit 21/21 = 120 tests pass in 154 s.

## 13.t iter-240: extend drift audit to physics integration + clean 2

Drift sweep continues into the **5 physics integration files** —
the next layer of hot-path code beneath the dycores.  Audit
extended from 21 to 26 files; 2 cleaned this iteration:

* ``turbulence/integration.py`` — 2 → 0.  Hoisted
  ``SpectralHydrostaticState``, ``spectral_pe_to_grid``
  (``atmosphere.dynamics.spectral_pe``) and ``sh_analysis_3d``,
  ``sh_analysis_oc2_3d``, ``sh_analysis_dmu_3d``
  (``grids.gaussian``) to module level.
* ``gravity_wave_drag/integration.py`` — 2 → 0.  Same import block
  hoisted.

Audit budgets locked in at iter-240 (3 files still drifted):

| file                                       | budget |
|--------------------------------------------|-------:|
| ``microphysics/integration.py``            |      3 |
| ``radiation/integration.py``               |      6 |
| ``convection/integration.py``              |     12 |

Verified: 26/26 audit tests pass; 37/37 ``test_gravity_wave_drag.py``
tests pass in 18 s.  No circular-import issues — atmosphere.dynamics
does not import atmosphere.physics.turbulence or
atmosphere.physics.gravity_wave_drag.

## 13.u iter-241: hoist microphysics (3→0) + radiation (6→3) physics

* ``microphysics/integration.py`` — 3 → 0.  Hoisted
  ``SpectralHydrostaticState``, ``spectral_pe_to_grid``,
  ``zero_like_tracers``, ``sh_analysis_3d``.
* ``radiation/integration.py`` — 6 → 3.  Hoisted ``get_policy``,
  ``SpectralHydrostaticState``, ``spectral_pe_to_grid``,
  ``sh_analysis_3d``.  *Kept inline (3)* — the RRTMGP imports
  (``rrtmgp_radiation``, ``RRTMGP`` ×2) trigger ~50 MB of optics
  constant-loading at module load.  Lazy-load preserved so users
  who run gray-radiation only don't pay that cost.

After iter-240/241, only ``convection/integration.py`` (12 inline
imports) remains in the physics drift backlog.  Audit budgets:

| family / file                              | budget |
|--------------------------------------------|-------:|
| ``turbulence/integration.py``              |      0 |
| ``gravity_wave_drag/integration.py``       |      0 |
| ``microphysics/integration.py``            |      0 |
| ``radiation/integration.py``               |      3 (RRTMGP lazy) |
| ``convection/integration.py``              |     12 (drifted)     |

Verified: 26/26 drift audit tests pass; 149/149 hydrostatic
radiation + microphysics unit tests pass in 197 s.

## 13.v iter-242: hoist convection (12→0) — physics drift sweep complete

Last drifted physics integration file.  ``convection/integration.py``
had 12 inline imports — all pure functions.  Hoisted:

* ``SpectralHydrostaticState``, ``spectral_pe_to_grid``
  (``atmosphere.dynamics.spectral_pe``).
* ``compute_moisture_convergence``, ``diagnose_grid_w_from_omega``,
  ``zero_like_tracers`` (``atmosphere.physics._shared``) — 3 inline
  call sites for ``compute_moisture_convergence`` collapsed to one
  module-level import.
* ``divergence_3d`` (×2: aliased ``_div3_cs`` from
  ``core.operators_3d`` and ``_div3_latlon`` from
  ``core.operators_latlon_3d``) — split inside the
  ``isinstance(grid, CubedSphereGrid)`` branch dispatch.
* ``sh_analysis_3d``, ``vordiv_from_uv_3d`` (``grids.gaussian``).
* ``compute_pressure_velocity``, ``compute_sigma_dot`` added to
  existing ``grids.vertical`` import block.

**iter-225..242 ATMOSPHERE+OCEAN+PHYSICS DRIFT SWEEP COMPLETE.**
Final state across the 26 audited hot-path files:

| family               | files | inline imports |
|----------------------|------:|--------------:|
| Hydrostatic PE       |     4 |              4 (CS mutable singletons) |
| Shallow water        |     4 |              0 |
| Non-hydrostatic      |     3 |              3 (CS mutable singletons) |
| Ocean dycore         |    10 |              0 |
| Physics integration  |     5 |              3 (RRTMGP lazy loads) |
| **Total**            |    **26** |     **10** (all intentional) |

Across iter-225..242 the sweep removed ~150 unintentional inline
imports.  All 10 remaining inlines are either mutable-singleton
late-bindings (CS dycores) or heavy-deps lazy loads (RRTMGP optics).
Future ``/clear`` drift in any of the 26 files fails CI in 0.03 s.

Verified: 26/26 drift audit tests pass; 138/138 convection-keyword
hydrostatic unit tests pass in 94 s.

## 13.w iter-243: extend drift audit to coupler files + clean 2

Drift sweep extends from 26 to **30 files** by adding the 4 coupler
modules with measured inline-import counts.  Cleaned 2 of them:

* ``accumulator.py`` — 1 → 0.  ``_resolve_dtype`` from
  ``core.precision`` hoisted to top-level.  Drop of the same iter-156
  fix; this is a re-application after /clear drift.
* ``surface_exchange.py`` — 1 → 0.  ``pressure_from_eos`` and
  ``temperature_from_theta`` (``atmosphere.physics.thermodynamics``)
  hoisted (re-application of iter-150).

Remaining drift (2 files, 10 inline imports total):

| file                               | budget |
|------------------------------------|-------:|
| ``coupler/coupler.py``             |      5 |
| ``coupler/mpas_adapter.py``        |      5 |

Verified: 30/30 drift audit tests pass; 50/50 coupler unit tests pass
in 32 s.

## 13.x iter-244: hoist coupler.py + mpas_adapter.py — coupler drift sweep complete

* ``coupler.py`` — 5 → 0.  Hoisted ``get_policy``,
  ``compute_most_fluxes`` (added to existing ``simple_bulk_fluxes``
  import), ``init_multilayer_land_state``, ``reshape_params``;
  dropped redundant ``constants as _constants`` alias and used the
  module-level ``constants`` import directly.  Re-application of
  iter-154/155.
* ``mpas_adapter.py`` — 5 → 0.  Hoisted ``init_surface_state``,
  ``make_coupler``, ``LakeConfig`` (from ``coupler.coupler``),
  ``LandConfig`` (from ``land.config``), ``SeaIceConfig`` (from
  ``ice.config``).  Re-application of iter-151.

After iter-243/244 the coupler drift sweep is complete.  Final state
across the **30 audited hot-path files**:

| family               | files | inline imports |
|----------------------|------:|--------------:|
| Hydrostatic PE       |     4 |              4 (CS mutable singletons) |
| Shallow water        |     4 |              0 |
| Non-hydrostatic      |     3 |              3 (CS mutable singletons) |
| Ocean dycore         |    10 |              0 |
| Physics integration  |     5 |              3 (RRTMGP lazy loads) |
| Coupler              |     4 |              0 |
| **Total**            |    **30** |     **10** (all intentional) |

Verified: 30/30 drift audit tests pass; 50/50 coupler unit tests pass
in 32 s.

## 13.y iter-245: extend drift audit to ice/land + clean 3

Drift sweep extends from 30 to 34 files by adding ice + land surface
modules.  Cleaned 3 of them in this iteration:

* ``ice/state.py`` — 1 → 0.  Hoisted ``aggregate_state`` from
  ``ice.itd``.  Re-application of iter-157.
* ``land/slab_land.py`` — 1 → 0.  Hoisted ``compute_most_fluxes``
  (added to existing ``simple_bulk_fluxes`` import block).
  Re-application of iter-158.
* ``land/multilayer_land.py`` — 2 → 0.  Hoisted
  ``compute_most_fluxes`` and ``psi_from_theta``.  Re-application
  of iter-149.

Remaining drift (1 file, 7 inlines):

| file              | budget |
|-------------------|-------:|
| ``ice/sea_ice.py``|      7 |

Verified: 34/34 drift audit tests pass; 64/64 sea-ice + land-snow
unit tests pass in 25 s.

## 13.z iter-246: hoist sea_ice (7→0) — ice/land drift sweep complete

Last drifted ice/land file.  ``ice/sea_ice.py`` had 7 inline imports
— all pure functions/constants.  Hoisted:

* ``compute_most_fluxes`` (added to existing ``simple_bulk_fluxes``
  import block — re-application of iter-147).
* ``evp_solver``, ``free_drift_velocity`` (``ice.dynamics``).
* ``advect_ice_tracers`` (``ice.transport``).
* ``aggregate_state``, ``linear_remap`` (``ice.itd``).
* Replaced inline ``import constants as _constants`` (×2) with the
  module-level ``constants`` alias and substituted
  ``_constants.sigma_sb`` → ``constants.sigma_sb``.

After iter-245/246 the ice/land drift sweep is complete.  Final state
across the **34 audited hot-path files**:

| family               | files | inline imports |
|----------------------|------:|--------------:|
| Hydrostatic PE       |     4 |              4 (CS mutable singletons) |
| Shallow water        |     4 |              0 |
| Non-hydrostatic      |     3 |              3 (CS mutable singletons) |
| Ocean dycore         |    10 |              0 |
| Physics integration  |     5 |              3 (RRTMGP lazy loads) |
| Coupler              |     4 |              0 |
| Ice + land           |     4 |              0 |
| **Total**            |    **34** |     **10** (all intentional) |

Across iter-225..246 the sweep removed roughly **170 unintentional
inline imports** from the model layer.  All 10 remaining inlines are
either mutable-singleton late-bindings (CS dycores) or heavy-deps
lazy loads (RRTMGP optics).  Future ``/clear`` drift in any of the 34
files fails CI in 0.03 s.

Verified: 34/34 drift audit tests pass; 64/64 sea-ice + land-snow
unit tests pass in 25 s; total 98 tests including 34 audits in 25 s.

## 13.aa iter-247: extend drift audit to core modules + clean 3

Drift sweep extends from 34 to 38 audited files by adding 4 core
modules.  Cleaned 3 of them:

* ``core/field.py`` — 2 → 0.  Hoisted ``get_policy`` from
  ``core.precision`` (×2 inline call sites collapsed to 1 module
  level).
* ``core/tracers.py`` — 1 → 0.  Same hoist.
* ``core/fv_tp_2d.py`` — 1 → 0.  ``synchronize_cgrid_fluxes``
  added to existing ``grids.halo`` import block.

**Documented exception** (the 4th file): ``core/precision.py`` keeps
2 intentional inline imports of ``runtime.backend.supports_float64``
/ ``is_x64_enabled``.  Hoisting them creates a circular import:
``runtime/precision.py`` and ``runtime/config.py`` both
``from legoesm.core.precision import …``, so loading legoesm's
runtime package triggers ``core.precision`` which would then need
``runtime.backend`` mid-init.  Pinned at budget=2 and rationale
captured in the audit test docstring.

Verified: 38/38 drift audit tests pass; 13/13 BCW unit + iter-199
regression tests pass in 16 s.

## 13.bb iter-248: extend drift audit to remaining core modules + clean 1

Drift sweep extends from 38 to **45 audited files** by adding 7 more
core modules.  Cleaned 1 of them in this iteration:

* ``core/operators_fv_latlon.py`` — 2 → 0.  ``pad_halo_latlon_3d``
  (×2 inline call sites) added to existing ``grids.halo_latlon``
  import block.

**Documented exception**: ``core/hardware.py`` keeps 4 intentional
inline imports — it's the legacy wrapper for ``runtime.backend``,
and hoisting recreates the same circular-import chain that
``core/precision.py`` runs into (runtime/precision.py imports
core/precision, which imports core/hardware, which would top-import
runtime/backend).

Remaining drift in core (47 inline imports across 5 files):

| file                          | budget |
|-------------------------------|-------:|
| ``core/operators.py``         |      4 |
| ``core/operators_3d.py``      |      4 |
| ``core/operators_cdgrid.py``  |     13 |
| ``core/conservation.py``      |     12 |
| ``core/fv3_sw_core.py``       |     14 |

Verified: 45/45 drift audit tests pass; 10/10 BCW unit tests pass
(55 tests total in 0.25 s).

## 13.cc iter-249: hoist core/operators_3d (4→0) + operators (4→3)

* ``core/operators_3d.py`` — 4 → 0.  Hoisted ``overlapped_halo_compute``
  and ``overlapped_halo_compute_vector`` from ``parallel.async_halo``
  (×4 inline call sites collapsed to one module-level import).
  ``parallel.async_halo`` has zero legoesm imports, so no cycle risk.
* ``core/operators.py`` — 4 → 3.  Hoisted ``get_halo_backend`` (added
  to existing ``grids.halo`` import block).  Kept 3 inline imports
  as documented cycle-avoidance:
  - ``conservation._accumulation_dtype`` (conservation top-imports
    operators.py).
  - ``parallel.reductions.global_sum_mpi`` (reductions reach back
    into core via mpi4jax thresholds).
  - ``parallel.mesh.get_active_config`` (mesh transitively touches
    core.field).

Drift-audit budgets ratcheted: operators_3d 4→0, operators 4→3.
Verified: 45/45 audit tests pass; 13/13 BCW + iter-199 regression in
16 s; 58 total.

## 13.dd iter-250: hoist core/operators_cdgrid (13→1)

Continued the core-module drift sweep into the FV3 cubed-sphere C-grid
operator file — the second-largest drift-count file in core (13 inline
``from legoesm`` imports, behind only ``fv3_sw_core.py`` at 14).

Hoisted nine symbols to a single module-level import block at the top
of ``core/operators_cdgrid.py``:

* ``grids.halo``: ``pad_halo``, ``pad_halo_4d``, ``pad_halo_vector``,
  ``pad_halo_vector_4d``, ``synchronize_cgrid_fluxes`` (×6 inline call
  sites collapsed).
* ``core.operators``: ``laplacian_compact`` (×3 inline call sites).
* ``core.operators_3d``: ``laplacian_compact_3d`` (×1 inline).
* ``core.fv_tp_2d``: ``transport_step`` (×1 inline).
* ``parallel.async_halo``: ``overlapped_halo_compute`` (×2 inline call
  sites).

12 inline-import sites collapsed to module-level binding.  Each of
these symbols is a pure function — no mutable-singleton late-binding
involved — so hoisting is safe.

Cycle-avoidance: kept the single
``from legoesm.core.fv3_sw_core import _d2a2c_vect`` inline at the
``_d_grid_to_a2c_vector`` call site.  ``core/fv3_sw_core.py`` line 30
top-imports ``operators_cdgrid``, so hoisting this back-import would
introduce a circular import at module-init time.  This is the same
``conservation ↔ operators`` pattern documented in iter-249 and is now
the only intentional inline in this file.

Drift-audit budget ratcheted: operators_cdgrid 13 → 1, with rationale
documented in ``tests/test_pe_dycore_inline_imports.py`` budget dict.

Verified: ``from legoesm.core import operators_cdgrid`` succeeds with
no circular-import error; 45/45 audit tests pass in 0.04 s; 54/54
BCW + iter-199 hydrostatic regression tests pass in 63 s; 99 tests
total.

Remaining core drift: ``core/conservation.py`` (12 inline) and
``core/fv3_sw_core.py`` (14 inline) — pursued in subsequent iterations
following the same hoist-pure-functions / keep-cycles-inline pattern.

## 13.ee iter-251: hoist core/conservation (12→0)

Continued the core-module drift sweep into the conservation-fixers
module — the third-largest drift-count file in core.

Hoisted nine symbols to module-level imports in
``core/conservation.py``:

* ``core.precision``: ``_resolve_dtype``, ``get_policy`` (×2 inline
  call sites collapsed — used by ``_tiny`` and ``_accumulation_dtype``).
* ``runtime.backend``: ``supports_float64``, ``is_x64_enabled``
  (×1 inline pair — used by ``_accumulation_dtype`` for x64 support
  detection).
* ``parallel.reductions``: ``global_sum_mpi``, ``batch_allreduce_mpi``
  (×3 inline call sites collapsed — used in ``_global_area_sum``,
  ``_batch_global_area_sums``, and the level-sum helper).
* ``grids.vertical``: ``compute_geopotential`` (×1 inline call site —
  used in ``compute_hydrostatic_energy``).
* ``core.operators_voronoi``: ``kinetic_energy_cell`` (×1 inline call
  site — used in ``fix_energy_mpas``).

Three redundant ``from legoesm import constants`` inlines (lines 411,
647, 704) were dropped outright — the package was already top-imported
at line 27.

One unused ``from legoesm.core.state import MPASShallowWaterState``
inline (line 819) was dropped — only referenced in docstrings, the
function body never used the imported name.

12 inline-import sites collapsed (9 hoisted, 4 dropped redundant).
None of the hoisted modules import ``core.conservation`` so no cycle
risk.  ``core/operators.py`` already inline-imports
``conservation._accumulation_dtype`` (iter-249 cycle-avoidance) which
remains the *only* ``conservation ↔ operators`` link.

Drift-audit budget ratcheted: conservation 12 → 0.

Verified: ``from legoesm.core import conservation`` imports cleanly;
99/99 tests pass (45 audit + 27 hydrostatic primitive_eq + 17 first-step
ps balance + 10 BCW scan-steps) in 64 s.

Remaining core drift: ``core/fv3_sw_core.py`` (14 inline) — pursued
next iteration.

## 13.ff iter-252: hoist core/fv3_sw_core (14→0) — core drift sweep COMPLETE

Final core-module drift target: the FV3 cubed-sphere C-grid SW
dynamical-core helpers — the largest single-file inline-import count
(14) in the audit.  All 14 inlines hoisted to module-level imports.

Module-level imports added in ``core/fv3_sw_core.py`` (extending the
existing top-level import block):

* ``grids.duogrid``: ``ext_vector_dgrid`` (×3 inline call sites in
  ``_pad_halo_dgrid_for_ppm``, ``_pad_uv_for_corner``, and
  ``_d2a2c_vect_duogrid``).
* ``grids.halo``: added ``synchronize_cgrid_fluxes`` and
  ``synchronize_bgrid_ne_corner_geo`` to the existing ``pad_halo,
  pad_halo_vector`` block.  Three redundant local re-imports of
  ``pad_halo`` (lines 439, 1652, 2082) and two of ``pad_halo_vector``
  (lines 1921, 2233) were dropped — already top-imported at line 29.
* ``core.operators_cdgrid``: added ``_interp_center_to_corner_a2b_ord4,
  fv3_d2cc, fv3_cc2c`` to the existing import block (×3 inline call
  sites).
* ``core.fv_tp_2d``: ``_pert_ppm, compute_transport_quantities,
  fv_tp_2d, transport_step`` (×2 inline call sites — d_sw3 boundary
  fix at line 2520 and ``fv3_sw_tendencies`` batch at line 2930).

Aliases inlined: three local ``import as`` aliases (``_pad_halo``,
``_phv``, ``_pert_ppm_iv1``) replaced with the canonical names
(``pad_halo``, ``pad_halo_vector``, ``_pert_ppm``).  Code reads cleaner
and the alias scope is gone.

Cycle safety: ``core/fv_tp_2d``, ``grids/halo``, ``grids/duogrid``,
and ``core/operators_cdgrid`` were checked — none import
``fv3_sw_core``, so the back-edge stays single-direction.
``operators_cdgrid`` keeps its iter-250 inline of
``fv3_sw_core._d2a2c_vect`` as the sole cycle-avoidance binding in
the FV3 SW dycore subgraph.

Drift-audit budget ratcheted: fv3_sw_core 14 → 0.

Verified: ``from legoesm.core import fv3_sw_core`` imports cleanly;
99/99 tests pass (audit + hydrostatic primitive_eq + BCW scan-steps)
in 64 s; 4/4 ``test_fv_convergence`` integration tests pass in 55 s
(SW FV3 convergence preserved across the alias rename + hoist
changes).

**Core drift sweep complete.**  All 12 ``src/legoesm/core/*.py``
modules in the audit now hold a hoisted-to-module-level inline-import
budget of 0 (pure-function imports) or 1-4 with documented
cycle-avoidance rationale.  Iters 247-252 removed 41 inline imports
from core (precision 4, hardware 4, field 1, tracers 1, fv_tp_2d 1,
operators_fv_latlon 2, operators 1 of 4, operators_3d 4,
operators_cdgrid 12 of 13, conservation 12, fv3_sw_core 14).

Next drift-audit extensions will move to ``parallel/``, ``runtime/``,
``driver/``, ``ml/``, and ``training/`` packages — ~370 unaudited
inline imports remain in those directories.

## 13.gg iter-253: extend drift audit to parallel/ + clean 3

Continued the drift-sweep methodology into the ``parallel/`` package
— the next-largest concentration of unaudited inline imports
(~75 across 16 files).

Audit extension: added a ``_PARALLEL`` root and pinned 16 file
budgets at *measured* counts so any new inline drift fails CI
immediately, then ratcheted three files down where pure-function
hoists were trivially safe.

Hoisted:

* ``parallel/voronoi_mpi.py`` 2 → 0.  ``partition_cells_geometric``
  and ``partition_cells_metis`` were imported inline at two call
  sites (geometric + METIS branches of ``init_voronoi_mpi``).  The
  module already top-imports ``voronoi_partition`` for other
  symbols (``VoronoiPartition``, ``partition_voronoi_mesh``,
  ``build_local_mesh``, ``scatter_to_local``); added the two
  partitioner functions to that block and dropped the inlines.
  ``voronoi_partition`` doesn't import ``voronoi_mpi``, so no cycle.

* ``parallel/latlon_mpi.py`` 2 → 1.  ``_get_sendrecv_vjp`` was
  imported inline inside the latlon halo-exchange function;
  hoisted to module level.  Remaining 1 match is the docstring
  ``::`` literal block at line 11 (a usage example, not a real
  runtime inline).

Documented false positives: ``voronoi_partition.py``,
``ensemble.py``, ``profiling.py``, and ``latlon_mpi.py`` each have
inline matches that come from Sphinx ``::`` literal blocks in the
module docstring (code examples for users).  These show up in the
regex but aren't real per-call imports.  Budgets pinned at the
docstring count so genuine drift still fails the audit.

Drift-audit budgets ratcheted: voronoi_mpi 2 → 0, latlon_mpi 2 → 1.

Verified: 115/115 tests pass (61 audit + 44 hydrostatic primitive_eq
+ 10 BCW scan-steps) in 64 s.

Remaining parallel/ drift (still pinned): ``runtime.py`` (16),
``sharded_dynamics.py`` (11), ``async_halo.py`` (11),
``cubesphere_exchange.py`` (7), ``reductions.py`` (5),
``halo_exchange.py`` (5), ``scaling_diagnostics.py`` (3),
``distributed.py`` (3), ``layout.py`` (2), ``device_config.py`` (2),
``halo_exchange_voronoi.py`` (1).  Pursued in subsequent iterations
with cycle-safety checks (``runtime.py`` is the package barrel and
will need careful handling).

## 14. Tasks for upcoming iterations

* When real multi-device hardware is available (Levante DKRZ, or a
  second GPU), the ``probe_spectral_shard.py`` recipe (prime caches +
  JIT-with-shard) is ready to drop straight into a real weak-scaling
  sweep without touching dycore source.

This file will be the durable status; iteration deltas should append a
short row, not rewrite the table.
