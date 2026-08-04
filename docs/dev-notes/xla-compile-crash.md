# "Fatal Python error: Segmentation fault" during long pytest runs

Two *different* crashes on the Levante login node both surface as
`Fatal Python error: Aborted/Segmentation fault` mid-run and both look like a
test failure. They have different causes and different fixes. Diagnosed
2026-08-01/02.

## Crash 1 — concurrent processes exhaust the thread rlimit

**Signature** (this one names itself, if you catch stderr):

```
F env.cc:93] Check failed: ret == 0 (11 vs. 0)
Thread tf_XLAPjRtCpuClient creation via pthread_create() failed.
  @ Eigen::ThreadPoolTempl<>::ThreadPoolTempl()
```

`11` is `EAGAIN`. XLA sizes its Eigen thread pool from the **CPU affinity
mask**, and the login node has 256 CPUs against `ulimit -u` = 2048. Measured
threads for one JAX CPU client compiling a trivial jitted matmul:

| CPUs    | 4  | 8  | 16 | 32  | 64  | 256 |
|---------|----|----|----|-----|-----|-----|
| threads | 29 | 45 | 77 | 141 | 237 | 621 |

Uncapped that is 621 threads each, so **four** concurrent clients (2484) are the
first count over the 2048 ceiling — three (1863) still fit. Reproduced
deterministically: 5 concurrent clients → 5/5 abort; with affinity capped to 32
CPUs → 0/5, and 0/12.

Knobs that do **not** work (all measured, thread count in brackets):
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` [621],
`--xla_cpu_parallel_codegen_split_count=8` [621],
`--xla_force_host_platform_device_count=1` [621],
`TF_NUM_INTRAOP_THREADS=8` [620], `OMP_NUM_THREADS=8` [565].
`NPROC=8` helps partially [125]. Only the affinity mask really controls it.

**Fix:** `tests/conftest.py` caps this process's CPU affinity before importing
JAX. The cap is *derived*, not fixed: it budgets half the thread ceiling across
the concurrent xdist workers and inverts the threads-per-CPU fit above, so a
host with a tighter `ulimit -u` (or more workers) gets a smaller cap; 32 is only
the ceiling, reached on the measured host. `LEGOESM_TEST_CPU_CAP` overrides
outright; `<= 0` disables. Under xdist each worker takes a **disjoint** slice —
otherwise all four inherit the same low-numbered CPUs and pile their XLA pools
onto one core set. MPI ranks are exempt: the launcher owns placement, and
clamping every rank to the same cores would serialise the job (detection is
best-effort over the common launcher rank vars; for an unlisted launcher set
`LEGOESM_TEST_CPU_CAP=0`). Capping is free here: these tests are XLA-compile-
bound, not intra-op-parallelism-bound (`tests/test_corner_div_damp_nh.py`:
166.9 s on 256 CPUs, 166.3 s on 8, 152.7 s on 32).

## Crash 2 — XLA compile failure, message destroyed by a stale unwinder

**Signature:** abort/segfault inside `backend_compile_and_load`, with
faulthandler printing `Current thread's C stack trace` and then dying before a
single C frame, and **no** glog `Check failed` line.

Sequence: XLA's `compile_and_load` returns a **failed status** → JAX throws a
C++ exception to report it → **`libgcc` faults in `_Unwind_Find_FDE` while
unwinding**, destroying the message. faulthandler cannot print a C stack because
it is itself stuck in the broken unwinder.

The host `libgcc_s` is GCC 8 (RHEL 8, max symbol `GCC_7.0.0`) against a jaxlib
built with a far newer toolchain. `LD_PRELOAD` of gcc-13.3's `libgcc_s` changes
the failure **segfault → abort** — same test, so the stale unwinder is in the
path but is not the root cause.

Hypotheses **killed by measurement** — do not re-propose without new evidence:

| hypothesis         | measurement                                      |
|--------------------|--------------------------------------------------|
| thread exhaustion  | threads FLAT at 264 during the crash, limit 2048 |
| memory / cgroup OOM| slice peak 16.9 GB of 50.2 GB, `oom_kill 0`      |
| compiler stack     | still crashes with `ulimit -s 65536` (64 MB)     |
| mapping exhaustion | `/proc/<pid>/maps` FLAT at 413, limit 65530      |
| JAX version        | JAX 0.9 fails identically; pinning is no remedy  |

It is **not** a defect in this repo's AD code as far as the evidence goes: no
compiled dycore code has begun executing when it fails, the same gradient passes
standalone under JIT, and it passes with `JAX_DISABLE_JIT=1`.

**Reproducer** (~5 min, deterministic, crashes on test #7 of the last file):

```bash
# from the project environment (`uv sync --extra all`); `.venv/bin/python` works too
JAX_ENABLE_X64=1 python -m pytest \
  tests/test_a2b_zeta_corner_nh.py tests/test_async_halo_nh.py \
  tests/test_atmosphere_cross_grid_plots.py tests/test_bcw_benchmark_scan_steps.py \
  tests/test_bcw_latlon_smoke.py tests/test_citation_metadata.py \
  tests/test_corner_div_damp_nh.py -q -p no:randomly
```

Bisection: `test_atmosphere_cross_grid_plots.py` is necessary but not
sufficient — `plots + corner` alone passes (208), all-seven-minus-plots passes
(31), the full seven segfault. So it is cumulative process state, not one bad
file.

**Mitigation:** run the top-level files across worker processes,
`-n 4 --dist loadfile` (wired into the CI ratchet job). The seven-file
reproducer then gives 231 passed in 2:45, and the full 47-file set completes
(8336 passed, 6:21) instead of dying partway.

**This is a mitigation, not a guarantee.** `loadfile` hands a worker a whole
file but keeps that worker alive for the ~11 files it receives, so state still
accumulates within a worker; it broke up the particular accumulating sequence.
If the crash returns, the deterministic fallback is one pytest process per file.

**The real fix is upstream.** Worth filing against JAX/jaxlib with the native
trace and an `--xla_dump_to` HLO dump from the failing compilation.

## Provenance of the measurements

Everything above was measured on:

| | |
|---|---|
| host | Levante login node (`levante4`), 256 CPUs, RHEL 8 |
| `ulimit -u` / `-s` | 2048 / 8192 KB (hard: unlimited) |
| cgroup `memory.max` | 50.2 GB (per-user slice) |
| `libgcc_s` | GCC 8 (`libgcc_s-8-20210514.so.1`, max symbol `GCC_7.0.0`) |
| python / jax / jaxlib | 3.14.6 / 0.10.0 / 0.10.0 |
| repo commit | 8ef11312d (origin/main) |
| date | 2026-08-01/02 |

Thread counts came from `len(os.listdir('/proc/<pid>/task'))` after one jitted
`(x @ x.T).sum()` on a 64x64 array, single sample per CPU count (the numbers are
stable to a few threads, and the effect being measured is a 14x span). The
"JAX 0.9 fails too" result was obtained by codex in a separate environment and
is second-hand -- treat it as indicative, not as a controlled bisect.
