# ORCA2 Round 349 Preregistration — Per-Core CPU Benchmark and Step Profile

- **Decision**: 219
- **Round**: `orca2_r349`
- **Branch**: `orca2_r349_cpu_benchmark`
- **Base commit**: `cc4b886a5829b2f94de2ea7fe537f52c3e3c4545`
- **Date**: 2026-11-05
- **Status**: registered before any round-349 model timing or profile run

## 1. Question and hypothesis

Measure the fair per-core CPU speed of the standalone ORCA2 one-year K0 setup,
as simulated years per wall day (SYPD), for NEMO 5.0.2 and legoESM on the same
physical CPU cores. Profile one steady legoESM exact-mode step to locate its
cost.

The registered hypothesis is:

> **legoESM's CPU time is dominated by XLA CPU kernels that use about one core
> regardless of n.**

- **CONFIRM** if legoESM exact-mode steady median seconds/step improves by less
  than 1.5x from `n=1` to `n=40` **and** its steady-window mean busy cores at
  `n=40` is below 2.0. Busy cores for each step is total process CPU seconds
  across all threads divided by that step's wall time, with no per-thread
  activity cutoff.
- **REFUTE** otherwise.
- No intermediate verdict is defined. Missing `n=1` or `n=40`, fewer than 16
  steady steps, a failed affinity/thread control, or a busy-core measurement
  outside the timed step makes the verdict **UNMEASURED**.

The profile also reports the fraction of the profiled step attributable to the
eager surface/bulk-flux path and the fraction with no compiled XLA CPU kernel
active. These are descriptive measurements and have no acceptance threshold.

## 2. Reused implementation and fixed model configuration

The round-349 entry point extends
`nemo_testcase_l4_orca2_round343_speed_benchmark.py`; it does not implement a
second ORCA2 benchmark. It reuses:

- the round-343 exact/normal benchmark's K0 card construction, forcing
  preprocessing, two-trace compile split, step loop, timing statistics, and
  fast-literal card verifier;
- round 344's `fast_literal` mode, which changes only
  `ocean.transcendentals: libm -> native` and retains the exact recipe;
- round 348's compiled exact `libm` implementation;
- `scripts/bench/sample_cpu_cores.py` for the corrected `/proc` CPU-tick
  parser and busy-core arithmetic;
- `scripts/validate/fv3_native/summarize_perfetto_trace.py` for Perfetto trace
  loading and event accounting.

No model default, physical selector, timestep, grid, forcing, or K0 setting is
changed. Both legoESM modes use fp64, the CPU backend, `dt=10800 s`, and the
round-343/342 standalone K0 inputs.

## 3. CPU topology, binding, and thread-pool control

The node is two Xeon Gold 6242R sockets with 40 physical cores and 80 hardware
threads. Linux topology is read from
`/sys/devices/system/cpu/cpu*/topology/{core_id,physical_package_id,thread_siblings_list}`.
For each physical `(socket, core)` pair, the lowest online logical CPU is kept;
the sorted first `n` representatives are used for `n in (1, 2, 4, 8, 16, 40)`.
The selected set must contain `n` distinct physical cores and no two selected
CPUs may be thread siblings.

The identical CPU-ID set is passed to legoESM through `taskset -c` and to NEMO
through Open MPI `--cpu-set <ids> --bind-to core -np n`.

Deviation (2026-11-06, user decision): installed Open MPI 5.0.10 binds every
rank to the whole `--cpu-set` under `--bind-to core` and refuses to map
`np=40` across both sockets, so NEMO now runs `--bind-to none` and rank `r`
pins itself with `taskset` to the `r`-th CPU of the same list (rank wrapper),
one hardware thread per rank exactly as legoESM is pinned.

Installed JAX/JAXLIB is 0.10.0. The installed
`/home/dbalwada/legoESM/.venv/lib/python3.13/site-packages/jaxlib/xla_client.py:64-87`
defines `make_cpu_client` without a thread-count argument and delegates directly
to `get_tfrt_cpu_client`; this build exposes no real per-process XLA CPU-pool
size control. The rejected/ignored `intra_op_parallelism_threads` spelling is
not used. The launcher uses only the real
`--xla_cpu_multi_thread_eigen=true` XLA flag. The XLA worker count is measured,
not assumed: under each `taskset` mask it must follow the process affinity. The
launcher also sets
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
and `NUMEXPR_NUM_THREADS` to `n` for those libraries.

Every legoESM timed step records process and per-thread `utime+stime` deltas
from `/proc/self/task/*/stat`. `cores_busy` is total process CPU seconds divided
by step wall seconds; there is no busy-thread threshold. The artifact records
every thread's CPU seconds, name, and `os.sched_getaffinity(thread_id)`, plus
the `tf_XLAEigen`, other-worker, and total thread counts. Every live thread's
affinity must equal the selected CPU-ID set exactly or the arm is refused. The
reported utilization and hypothesis input is the arithmetic mean `cores_busy`
over the steady window; the median is retained as telemetry.

## 4. legoESM timing protocol

For each core count, run two isolated processes:

1. `exact`: the movie's fp64 compiled-`libm` mode;
2. `fast_literal`: the exact recipe with only native XLA transcendentals.

Each full arm runs 18 steps. Steps 1 and 2 are the separate cold-start and
carried-state compile/warm-up traces. The steady window is derived from the
actual requested step count as `3..steps`; an empty or odd-length steady window
is refused so the two `nn_fsbc=2` parities have equal samples. At 18 steps this
is steps 3–18, containing eight odd and eight even parities. Record individual
warm-up times, estimated compile overhead, steady median/mean/IQR/min/max,
odd/even medians, actual busy cores, active threads, affinity, host memory, and
background-load preflight.

For each result:

- `SYPD = 86400 / (2920 * seconds_per_step)`;
- `core_seconds_per_simulated_step = n * seconds_per_step`.

The requested four-step smoke is a non-citable control: two warm-up steps and
two steady steps, fast-literal only, at `n=1` and `n=2`. It receives no campaign
verdict.

## 5. NEMO operator protocol

The certified source is the observer-free K0 year directory
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/orca2_k0_year1_a_np2`
with binary SHA-256 from its producer manifest. The operator script never runs
or builds NEMO in agent smoke work.

For each `n`, the operator script creates a fresh directory under the declared
round-349 scratch root, copies the executable, namelists, and XML/config files,
and recreates the input links. It changes only run-control output fields:

- `nn_itend: 2920 -> 96`;
- `nn_istate: 1 -> 0`;
- `nn_stock: 8 -> 0`.

Thus the short benchmark writes neither initial-state output nor periodic/final
restarts; `nn_write=0` and the binary has no `key_xios`, so history output is
also off. Physics, forcing, cadence, and timestep remain those of the certified
year run. The historical `1767.523521 s / 2920 steps / 2 MPI ranks` result is
reported separately because it includes the year run's eight-step restart
cadence and is not substituted for a round-349 arm.

NEMO seconds/step is parsed from the exact registered steady `timing.output`
block at steps 7–92 (86 steps). A rank wrapper records each rank's inherited
`Cpus_allowed_list` before launch; all ranks must each occupy exactly one of the
requested Linux CPU IDs. Summed rank user+system CPU time divided by the outer
elapsed wall time supplies actual busy cores. Allocation core-seconds/step and
SYPD use the formulas above.

Before any NEMO arm, the script refuses when:

- a `nemo` or `nemo.exe` process exists, or its immediate parent is named
  `mpirun`, `orterun`, or `prterun`;
- available RAM is below 150 GiB;
- any selected logical CPU or its hyperthread sibling exceeds 10% utilization
  over a 10-second sample;
- any unrelated process averages at least one full core over that sample;
- Open MPI, GNU `time`, the certified binary, or required inputs are absent.

Dry-run performs topology/config/command generation only and does not copy
files, sample for ten seconds, or launch NEMO.

## Operator choices confirmed by the user (DECISION 221, 2026-11-05)

- Keep the selected physical cores and their hyperthread siblings at or below
  10% utilization over the unchanged 10-second preflight sample.
- Keep refusal when any unrelated process averages at least 1.0 core over that
  sample.
- Keep `nn_istate=0` and `nn_stock=0` for the short NEMO benchmark.

## 6. legoESM profile protocol

Profile exact/libm step `kt=3` after the two compile/warm-up steps at `n=1` and
`n=40`. `jax.profiler.trace(..., create_perfetto_trace=True)` captures exactly
that synchronized odd-parity step. The whole returned pytree is blocked before
the timer and annotation close. An explicit `orca2_round349_surface_eager`
annotation brackets the S0-F/surface-card dispatch path; absence of that event
is fatal, and the annotated surface result is synchronized before the region
closes so its duration is a serialized surface-path wall cost. The trace parser clips complete events to
the registered step annotation and reports:

- top cost centres by inclusive CPU time, grouped from event name and source
  metadata into tracer advection, TKE/vertical mixing, EOS/density,
  barotropic/free surface, momentum/vorticity, lateral mixing/GM-Redi, surface
  bulk flux, shortwave, freshwater/runoff, compiled libm callback,
  optimization-barrier/copy, other compiled kernels, and Python dispatch;
- each centre's event count, CPU seconds, and fraction of classified CPU time;
- compiled-kernel wall coverage (union of XLA-worker event intervals);
- `eager_or_dispatch_fraction = 1 - compiled_kernel_wall_coverage / step_wall`;
- eager surface-path wall fraction from the explicit surface annotation.

Inclusive CPU-time shares rank centres but are not added as wall time when
worker events overlap. Unclassified events remain explicit; they are never
dropped or reassigned post hoc. Thread-name/event-text attribution and the
compiled-worker union are heuristic trace classifications; the synchronized
surface fraction is non-exclusive with compiled-worker coverage.

## 7. Report and provenance requirements

The combined JSON and Markdown reports contain one row per model/mode/core
count with: seconds/step, SYPD, allocated core-seconds/step, actual busy cores,
per-thread CPU seconds, worker counts and exact affinities, steps and timing
window, CPU IDs, mode, dtype/backend, background-load sample, git/worktree
provenance, input/binary identity, and artifact path. Profiles
carry the same provenance plus the trace path and registered step.

The report validator refuses duplicate rows, unregistered core counts, wrong
mode/core bindings, invalid SYPD arithmetic, fewer than 16 steady legoESM steps
in a full report, missing provenance/load fields, or a profile without cost
centres and dispatch/kernel fractions.

No number from a dry run or four-step smoke is eligible for the final table.
