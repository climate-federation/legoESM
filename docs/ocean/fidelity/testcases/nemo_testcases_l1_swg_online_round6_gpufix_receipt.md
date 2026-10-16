# NEMO testcase L1 SWG online learning — Round 6 GPU-path receipt (gpufix)

Date: 2026-10-14

Base tip: `a0b5535b2e9750e11c51dd819cc1240814f07764` (round 5c), work copy
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/claude-swgonl-gpufix`,
changes left uncommitted for the operator.  All GPU work ran on physical GPU 1
with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.18`, shared with the 20-year truth job
until it finished at 05:36 UTC; nothing was killed and GPU 0 was not used.  Scratch scripts and logs:
`/home/dbalwada/.claude/jobs/b3690d5d/tmp/claude_r6/gpufix/impl/` (logs in
`out/`).

## Verdict

**FIXED.** The hourly dual-operator sampler and the closure network both run
on the V100S.  The only failing component was the round-5c filtered
restriction: XLA compiled one of its 97-tap shifted-slice passes into a single
fused kernel that faults with `CUDA_ERROR_ILLEGAL_ADDRESS`.  The filter is now
evaluated as two dense matrix products per extensive (`G_y X G_x^T`, wall
parity folded into the matrices), so the GPU runs library GEMMs instead.  The
definition is unchanged, CPU and GPU agree to rounding, the GPU smoke workers
pass, and a two-day dual chunk keeps bit-for-bit next-day restart parity.

Two things the overnight generation needs decided first (see Decisions): the
data contract's exact chunk-adjacency check (review DC-02) is confirmed to
refuse GPU chunks, and the filtered stream is reproducible across reruns only
to rounding unless GEMM autotuning is turned off.

## Reproduction and cause (round-5c code, GPU 1)

| Run (smallest first) | Result |
|---|---|
| Filtered restriction alone, outside any scan, real restart day 731 | compiles (9.31 s, temp 16,896,016 B), then `CUDA_ERROR_ILLEGAL_ADDRESS` (`out/r01_filtered_gpu.log`) |
| Same, `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (empty: no command buffers / CUDA graphs) | same fault (`out/r01_filtered_nocb_gpu.log`); the flag is recognised, since an unknown flag aborts XLA (`Unknown flag in XLA_FLAGS`, `out/r05_sigma8_unroll1.log`) |
| Same, plus `CUDA_LAUNCH_BLOCKING=1` | `Failed to launch CUDA kernel: loop_add_fusion_4; block dims: 128x1x1; grid dims: 800x1x1 ... CUDA_ERROR_ILLEGAL_ADDRESS` (`out/r01_filtered_blocking_gpu.log`) |
| What that kernel is (optimized HLO, `out/filtered_gpu_hlo.txt`, `%fused_add.4`) | the axis-0 97-tap pass of the cell-volume filter fused with the `[2:642]` slice of the 644x644 input and the reflection reverse/concatenate |
| The same 97-tap filter on 640x640 parameter inputs (no input slice to fuse) | runs, 0.066 s (`out/r01_filter_array_gpu.log`) |
| Same restrictor with sigma 1 (13 taps) / sigma 4 (49 taps) | run; sigma 4 equals the CPU to 1.49e-12 m, 1.3e-15 and 1.8e-15 m/s (`out/r05_sigma4.log`) |
| Box restriction alone | runs (`out/r01_box_gpu.log`) |
| Real 7,696-parameter network, 24 h forward and reverse, both objectives (start state restricted on the CPU) | run: symmetric 2.3088e-09, peak 291 MB; template 1.1019e-08, peak 290 MB (`out/r04_*.log`) |
| The operator's chunk, one day from 731 through `run_long`, with only the coarse-graining module swapped back to round 5c (`r21_chunk_with_module.py`) | `Segmentation fault (core dumped)`, exit 139, only `manifest.pending.json` written: the operator's symptom.  The same harness with the round-6 module completes with restart parity (below) |

Cause, as far as measured: an XLA GPU code-generation defect for one large
loop fusion (many shifted slices of a reflected, sliced input), not memory
(16.9 MB temporary in a 5.9 GB pool), not command buffers, not cuDNN or cuBLAS
(the faulting kernel is XLA-generated).  Whether it is specific to compute
capability 7.0 cannot be separated here (only V100S cards exist).  Because
XLA's fusion decisions depend on the surrounding program, a shifted-slice
formulation that runs in one context can fault in another, which is why a flag
or a smaller kernel was not taken as the fix.

## Fix

- `swg_coarse_graining.py:447` `closed_wall_filter_matrix`: the dense matrix of
  one closed-wall pass, built by applying the module's own NumPy reference
  (`_pad_even_cell_centres` / `_pad_odd_boundary_faces` and the valid
  convolution) to the identity (`swg_coarse_graining.py:475`).  For the odd
  (wall-normal) parity the two wall-face basis vectors are dropped
  (`swg_coarse_graining.py:471`); callers already require those faces to be
  exactly zero.
- `swg_coarse_graining.py:524` `filter_closed_cgrid_extensives_array`: each
  extensive is `G_y X G_x^T`, axis 0 first (`swg_coarse_graining.py:542`),
  float64 at `Precision.HIGHEST` (`:543`); exact zero normal walls are
  re-imposed as before (`swg_coarse_graining.py:552`).  It is no longer
  `jax.jit`-decorated because the matrices are built from the concrete kernel
  when the caller is traced; the in-scan restrictor keeps the kernel in NumPy
  (`swg_coarse_graining.py:857`).
- `swg_coarse_graining.py:486`: `shifted_weighted_sum` stays for the
  symmetric objective's 9-tap Gaussian (owned elsewhere) with a note that
  long stencils go through the matrix path.
- GPU smoke: the sampler worker now also executes the production nested-scan
  sampler program (two fine steps per hour) and requires its last sample to
  equal the stand-alone sampler on the stepped state
  (`nemo_testcase_l1_swg_online_gpu_smoke.py:143`, called at
  `nemo_testcase_l1_swg_online_gpu_smoke.py:239`); round 5c only lowered that
  program.  Output schema bumped to v2 (`nemo_testcase_l1_swg_online_gpu_smoke.py:372`).
- No change to the driver or the chunk script was needed.

Pre-implementation search: `grep -rln "filter_matrix|toeplitz|banded_matrix|convolution_matrix|smoothing_matrix"`
over `packages/` and `src/` found nothing relevant (one unrelated hit in
`fast_sbm/collision.py`); the matrix builder reuses the module's existing
NumPy padding and convolution helpers rather than re-deriving the parities.

## Measurements after the fix

| Quantity | Value (log under `out/`) |
|---|---|
| CPU, NumPy entry, round 5c vs round 6, real days 731 and 1095 | SSH <= 2.84e-13 m, U <= 3.33e-16, V <= 4.44e-16 m/s (`r06_old_vs_new_cpu.log`) |
| CPU, matrix vs shifted-slice definition, random closed-basin fields | <= 1.67e-16 at amplitudes 0.05-0.26 (`r12_count_ops.log`) |
| CPU, vs the round-5 dense 97x97 result, day 1095 | 1.1937117960769683e-12 m, 2.6645352591003757e-15, 1.7763568394002505e-15 m/s, same with one Eigen thread (`r07_dense_pin.log`, `r07_dense_pin_one_thread.log`) |
| CPU, in-scan JAX vs NumPy | day 731: 4.37e-13 m, 5.55e-16, 4.44e-16 m/s; test synthetic state: 4.884e-13 m (`r13_synthetic_jax_vs_numpy.log`; round 5c: 2.0473e-12 m) |
| GPU vs CPU, stand-alone dual sample, day 731 | box 2.24e-13 m, 2.2e-16, 2.2e-16 m/s; filtered 4.14e-13 m, 3.33e-16, 4.44e-16 m/s (`smoke_sampler_v2.json`, `smoke_sampler_final.json`) |
| GPU in-scan vs stand-alone (smoke) | SSH 0.0; U, V <= 4.44e-16 m/s (same file) |
| Continuity, wall-normal, volume, transfer-function tests | pass (`24 passed` below) |
| Lowered restrictor, StableHLO ops | round 5c: 1185 `slice`, 0 `dot_general`; round 6: 13 `slice`, 6 `dot_general` (`r12_count_ops.log`) |
| GPU compiled memory, stand-alone dual sampler | temp 20,594,448 B (args 9,953,664 B, out 338,736 B) |
| GPU compiled memory, production day programs | hourly dual day temp 99,654,560 B vs flat day 89,683,144 B (`r16_program_memory_gpu.log`) |
| GPU process peak | two-day chunk 524,360,704 B (`r08_chunk_gpu_731_2d.log`); smoke sampler worker 550,299,904 B |

GPU smoke, the three workers run directly with the parent's exact arguments
(the parent refuses an uncommitted tree, `nemo_testcase_l1_swg_online_gpu_smoke.py:348`),
final code, GPU 1 alone (`out/smoke_*_final.log`; while shared they took
34.1, 61.8 and 69.8 s):

```text
GPU_SMOKE_TASK_PASS task=dual_hourly_sampler wall_seconds=23.723 peak_bytes=550308352
GPU_SMOKE_TASK_PASS task=reverse_24h_symmetric wall_seconds=45.594 peak_bytes=317566720
GPU_SMOKE_TASK_PASS task=reverse_24h_template wall_seconds=46.032 peak_bytes=316139264
```

Two-day dual chunk 731 -> 733 on GPU 1, through the driver's own `run_long`
with the chunk script's arguments (`gpu-native`, maddison, hourly targets,
`--verify-next` truth day 732; the wrapper refuses an uncommitted tree):

```text
CHUNK_STATUS complete backend gpu
RESTART_NEXT_VERIFICATION {'all_equal': True, 'ssh_m_unequal': 0, 'ssh_m_max_abs': 0.0, 'u_m_s_unequal': 0, 'u_m_s_max_abs': 0.0, 'v_m_s_unequal': 0, 'v_m_s_max_abs': 0.0}
WALL_S total 64.44 per-day (day1 includes compile) [47.76, 16.56]
```

The stored stream equals the NumPy reference restrictions of the truth
restarts at t0, hour 24 and hour 48 for both operators to 2.274e-13; controls
(other operator 0.629 m, previous hour 1.25e-2 / 1.14e-2) do not
(`r15_validate_gpu_stream.log`).

Next-day parity check (the operator's question): it compares bit for bit
(`run_swg_double_gyre.py:173`, uint64 views of SSH, U, V), on day 1 of each
chunk only (`run_swg_double_gyre.py:893`); a single unequal word writes status
`refused` (`run_swg_double_gyre.py:899`) and raises
(`run_swg_double_gyre.py:902`), so the chunk script stops (`run_swg_online_hourly_chunks.sh:2`)
and a rerun archives the chunk and retries.  On the GPU the nested hourly
program reproduces the flat truth program word for word (0 unequal words,
both streams on, chunks from days 731 and 732); the 6% / 1e-14 nested-vs-flat
difference the review measured did not occur on the GPU.  A CPU acquisition
with `--verify-next` against these GPU truth restarts would refuse (not run
here; the review's CPU regeneration matched the GPU restart only to 1.2e-14).

## Review findings in this area

| Finding | Disposition | Evidence (command -> decisive line) | Reversion that makes its test fail |
|---|---|---|---|
| Operator 2026-10-14 03:53: smoke sampler worker exit 2 (`CUDA_ERROR_ILLEGAL_ADDRESS`), two-day dual chunk segfault | **fixed** (`swg_coarse_graining.py:524`) | smoke workers -> three `GPU_SMOKE_TASK_PASS`; `r08_chunk_harness.py gpu-native 731 2` -> `all_equal: True`, 0 unequal words | round-5c module in place of round 6 (`r14_run_with_module.py ... old_swg_coarse_graining.py`) -> `test_filtered_restriction_lowers_to_matrix_products_not_tap_slices` `assert 0 == 6` (`test_swg_coarse_graining.py:290`) and the day-1095 pin `V 1.887379141862766e-15` vs `1.7763568394002505e-15` (`test_swg_coarse_graining.py:631`); `5 failed, 19 passed` (`out/reversion_round5c_module.log`); on the GPU the round-5c restrictor faults (table above) |
| NEW R6-1: the smoke never executed the in-scan sampler program | **fixed** (`nemo_testcase_l1_swg_online_gpu_smoke.py:143`) | smoke sampler JSON -> six `in-scan/stand-alone` rows, max 4.44e-16 m/s | round-5c smoke script -> `AttributeError ... _run_in_scan_sampler`; in-scan check comparing hour 23 -> `1 failed, 2 passed` (`out/smoke_reversion.log`); left-endpoint plant inside the test (`test_swg_online_gpu_smoke.py:98`) |
| D1-04: no test runs the schema-v2 writer of `run_long` | **fixed**: `test_run_long_dual_hourly_writer_operators_endpoints_and_t0` (`test_run_swg_double_gyre.py:267`) runs it on the real 6.25 km grid with a dyadic drift model and checks t0, hours 1 and 24 for both streams against NumPy, plus `absolute_hour` (`test_run_swg_double_gyre.py:321`) | `-k dual_hourly_writer` -> `1 passed` (20.7 s) | `r18_driver_mutants.py` (`out/driver_mutants_d104.log`): sigma one fine cell, streams swapped on disk, t0 after day one, one fine step per sample, left endpoint -> each `1 failed`, max difference 5.47e-05 to 1.77e-03; absolute hour one day late -> `assert [25, 26, ...] == [1, 2, ...]`; unmutated `1 passed` |
| D1-05: sampler-on/off parity shown only on a toy model; `--verify-next` checks day 1 only | **refuted on the GPU** for parity: real fine model, both streams on, restart words equal (0 of 3 x 414,736) for chunks from 731 and 732; **deferred**: day-1-only verification (on the GPU one deterministic program per chunk makes day 1 representative; a last-day check needs a new driver flag and was not asked) | chunk harness lines above | n/a (measurement) |
| D1-06: the in-test even-reflection plant is not a parity discriminator | **fixed**: two-arm plant through the full commutator, odd must commute and even must miss by more than 1e-3 (`test_swg_coarse_graining.py:191`) | `24 passed` | even-normal-parity mutant of `closed_wall_filter_matrix` (`out/mutant_even_normal.log`) -> 7 tests fail, commutator 0.0686 vs 2e-15 |
| DC-02 (driver side): chunk k's last in-scan record vs chunk k+1's eager t0 | **confirmed on the GPU, deferred** to the data-contract owner (decision 1) | `r09_adjacency_words.log` -> box SSH `2035 of 7056` words differ, U `2847`, V `2360`; filtered `2036`, `2908`, `1611`; max 5.684e-14 m, 4.441e-16 m/s | n/a |
| NEW R6-2: the filtered stream is not bitwise reproducible across processes | **deferred** (decision 2) | same model hours from two runs (`r09_adjacency_words.log`): box 0 words, filtered U `7310 of 169344`; default vs autotuning off (`r17_default_vs_at0.log`): box 0 words, filtered up to 2.274e-13 m, 3.331e-16 m/s; the GEMM algorithms XLA's autotuner picks vary by process (three saved runs `['0','1','1','0']`, two earlier session runs `['1','1','0','0']`, `['0','1','0','0']`); with `--xla_gpu_autotune_level=0` two processes give equal digests (`r10_autotune0_run*.log`) and two 1-day chunks 0 unequal words in all 12 arrays (`r17_at0_a_vs_b.log`), restart parity still bitwise | n/a |
| D1-01, D1-02, D1-03, D1-N1, D1-N2, D1-N3 | not in this task's files (analyzer, objective, loss mask) | — | — |

## Overnight generation

Use GPU 1, now that the truth job has finished (05:36 UTC).  Warm time per
fine day:

| Device | Flat day (no sampler) | Hourly dual day | First chunk day incl. compile |
|---|---|---|---|
| GPU 1 alone (`r11_day_timing_unshared.log`, `r08_chunk_gpu_731_2d_unshared.log`) | 7.16 s | 7.28 s (7.33 s through `run_long`) | 26.16 s |
| GPU 1 shared with the truth job (`r11_day_timing_shared.log`) | 16.01 s | 16.26 s | 46-48 s |
| Same, autotuning off (`r11_day_timing_shared_autotune0.log`) | 16.11 s | 16.36 s | 37 s |
| CPU, `cpu-scanned`, 80 cores at load 78-118 (`r08_chunk_cpu_731_2d.log`) | — | 95.77 s | 137.52 s |

The sampler costs about 0.1-0.25 s per day.  The unshared two-day chunk again
kept bitwise restart parity.  For the round-5c ACQUISITION_NEEDED ranges
(4,380 days 731 -> 5111 in 146 thirty-day chunks, plus the 365-day test
year in 13) that is about 9.6 h of stepping plus roughly 1.2 h of per-chunk
compile and start-up on GPU 1 alone, about 23 h if the GPU is shared, and
about 125 h on the CPU.  The 0.18 pool is ample (peak 0.52-0.55 GB).  The operator's
round-5c command lines run unchanged after commit; they name GPU 0, which is
reserved, so substitute 1.

## Decisions needed (before the 12-year generation)

1. Chunk adjacency (DC-02).  The driver cannot make chunk k+1's stored t0
   bitwise equal to chunk k's last in-scan record (different compiled
   contexts; box differs too).  (a) The contract checks adjacency to a
   tolerance of about 1e-12 m and 1e-15 m/s and uses one of the two records
   — my pick; (b) the driver computes every sample with one stand-alone
   executable per process and autotuning off.
2. Rerun reproducibility of the filtered stream.  (a) Accept differences up
   to 2.3e-13 m and 3.3e-16 m/s between reruns of a chunk; (b) run
   acquisitions with `XLA_FLAGS=--xla_gpu_autotune_level=0` (measured
   bit-identical across processes, no slowdown) — my pick.

## Tests run (CPU, `JAX_ENABLE_X64=1`)

```text
tests/ocean/fidelity/test_swg_coarse_graining.py                 24 passed in 24.91s
tests/ocean/fidelity/test_run_swg_double_gyre.py
tests/ocean/fidelity/test_swg_online_gpu_smoke.py                23 passed in 97.82s (0:01:37)
tests/ml/test_swg_online.py                                      18 passed in 135.37s (0:02:15)
ruff check (five changed Python files)                           All checks passed!
```

Citation check.  The lane gate cannot verify this receipt: its `run()`
refuses an uncommitted tree before reading anything, and its `CITATION_MAP`
holds only oracle anchors, so all 23 citations its extractor finds here are
unmapped (it does not read `.sh` citations).  The gate's test file gives
`2 failed, 15 passed`; both failures are that same uncommitted-tree refusal
(the file passed at the base tip in the round-5c battery).  Instead,
`r20_check_citations.py` checks all 24 citations against an expected symbol
at the cited line and requires the symbol to be absent two lines later:
`own check: 24 citations checked; failures: []` (`out/citation_check.log`);
the same check on a copy with one citation shifted by two exits 1
(`out/citation_check_plant.log`).

The lane battery in `autopilot.sh` does not collect
`tests/ocean/fidelity/test_swg_coarse_graining.py`,
`tests/ocean/fidelity/test_run_swg_double_gyre.py` or `tests/ml/test_swg_online.py`
(its glob looks for `tests/ocean/unit/test_swg_coarse*.py`); the round-5c
"full battery" therefore never ran them.

## UNVERIFIED

- The XLA-internal reason the fused kernel faults; only the kernel, its
  fused operations and the tap-count dependence were measured (no
  compute-sanitizer on this machine).  Whether a newer XLA or a non-Volta
  card would run the round-5c program is unknown.
- That the GPU smoke parent prints `GPU_SMOKE_PASS` and that
  `run_swg_online_hourly_chunks.sh` itself passes: both refuse an
  uncommitted tree, so the workers and `run_long` were run directly with the
  same arguments.  The operator should rerun both after committing.
- The 12-year wall time is extrapolated from one- and two-day runs; the
  per-chunk compile and start-up (about 25-30 s) is estimated, not timed
  end to end.
- cuBLAS bit-for-bit determinism for a fixed algorithm is assumed from
  within-process repeats and the two autotuning-off processes, not from
  documentation read here.
- `Compiled.memory_analysis`, `Device.memory_stats` and `Lowered.as_text` were
  used as the round-5c code already uses them; `jnp.matmul(..., precision=)`
  was read in the installed `jax/_src/numpy/tensor_contractions.py`.
- No adversarial review (codex or GLM) was run on this diff by me; the
  operator's review step applies.
