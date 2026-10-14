# SWG online learning, round 8: driver and scorer integration receipt

Date: 2026-10-14. Base `2a227dbc9` (round 7 merged). Work copy
`claude-swgonl-r8`; every change is uncommitted (the operator commits). CPU
only, fp64, `JAX_PLATFORMS=cpu`. Scope: the five items the round-7 evaluation
verifier left open (closure key names, no daily means, comparability with the
controls, checkpoint never checked against a file, the merged driver test),
one end-to-end test, and a pilot configuration. **Nothing was launched**: no
training and no free run; the only model integrations were measurements of
at most 12 model days, written to scratch.

Commands run from the work copy with
`PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`,
`pytest -q -p no:cacheprovider`, one file and one heavy process at a time.
Scratch (scripts, logs, JSON): `/home/dbalwada/.claude/jobs/b3690d5d/tmp/claude_r8/impl/`
(`m01_*` to `m09_*` measurements, `mutants.py`/`mutrun.py`/`reversions.py`
and `reversions.jsonl` for the reversions, `check_citations.py`, `logs/`).

## Verdict

- **A free run written by the training driver is now admitted by the
  scorer.** The manifest schema is defined once, in
  `legoesm.training.swg_online`: the driver writes the closure block through
  `free_run_closure` and the scorer reads it through `admit_free_run_closure`.
  Corrupted manifests are refused.
- **Free runs write, every model day, the end-of-day snapshot beside the mean
  of every post-step state**, through the double-gyre driver's own daily-mean
  stepper, under the scorer's dual per-file labels. They run on the CPU with
  libm because the configuration says so (two new required keys; nothing else
  is admitted).
- **They are a separate command, `free-run`.** The libm path runs only on the
  CPU, so a GPU training process cannot run them; GPU training no longer waits
  for them.
- **Seed and controls now differ only by the closure, measured on days 1-2:**
  the new free-run path with no closure reproduces the nu=10 control bit for
  bit (0 differing words in 6 of 6 arrays).
- **The scorer hashes the checkpoint file the manifest names**, and refuses a
  missing or changed file.
- **Found here: no configuration could pin its test split.** The test-year
  chunks are `dual_test_day*`; every configuration labelled them `dual`
  (0 chunks pinned), and the driver refused `_` in a label. Fixed.
- **Pilot:** about 1.7 h of GPU training from the operator's measured GPU
  profile (11.4 h with the stated 4.6x assumption), then two 20-year CPU free
  runs of about 10-12 h each, side by side.

## Findings, dispositions, evidence, reversions

Each reversion undoes one fix in a scratch copy of its file, never in the work
copy (round 7's mechanism, `mutrun.py`), and runs the tests named for it.
The unmutated work copy passes the union of those tests:
`28 passed in 187.94s`. Each entry below quotes the decisive failure.

| Finding | Disposition | Evidence | Reversion: the test that fails |
|---|---|---|---|
| (a) The driver wrote `closure.checkpoint = {path, sha256}`, the scorer required `closure.checkpoint_sha256`: every driver manifest refused | One schema in the shared module: manifest label `FREE_RUN_MANIFEST_SCHEMA` (`swg_online.py:1560`), exactly `FREE_RUN_CLOSURE_KEYS`, validated by `free_run_closure` (`swg_online.py:1572`) for the writer (`run_swg_online_training.py:887`, again before writing at `run_swg_online_training.py:531`) and by `admit_free_run_closure` (`swg_online.py:1621`) for the scorer (`nemo_testcase_l1_swg_online_evaluation.py:1087`, inside the new single-run `admit_trained_run`, `nemo_testcase_l1_swg_online_evaluation.py:1073`, which `bind_trained_runs` now calls for each seed) | `test_run_swg_online_training.py:621`: the driver's manifest is admitted, with the trained checkpoint's path and digest; three corrupted copies refused (the round-7 schema label, another digest, the round-7 `checkpoint` block).  `test_swg_online_evaluation.py:435`: 29 plants refused before scoring (18 named, plus each of the 11 schema keys missing in turn) | `r8_driver_round7_closure_block` (driver spells its own round-7 block): end-to-end test fails at the scorer's admission of the driver's own manifest (`free-run closure must hold exactly the keys [...], found ['checkpoint', ...]`).  `r8_scorer_round7_binding` (scorer back to round 7's reading): end-to-end test `DID NOT RAISE` on the round-7 schema label; binding test `Failed: checkpoint file missing` (round 7 admits it) |
| (b) Free runs wrote snapshots only, under their own per-file label, which the scorer refuses | `free_run_day` (`run_swg_online_training.py:494`) wraps `swg_online_step` (closure at every RK3 stage) in the double-gyre driver's `build_scanned_day_mean_stepper` (`run_swg_online_training.py:515`, `run_swg_double_gyre.py:211`); every day file is a dual file under the scorer's labels (`run_swg_online_training.py:584`), now defined once beside the schema (`swg_online.py:1555`) and imported by the scorer | `test_run_swg_online_training.py:621`: the scorer's loader admits days 1-3 of a four-day free run (each snapshot shown from the data against its own and the next day's mean), and day 1's stored mean equals the mean of its 24 post-step states recomputed one step at a time from the checkpoint (to 1e-12 of the field).  Real card, below: day-1 mean equal to the step-by-step mean within 4.9e-17 m | `r8_free_run_snapshot_only` (round-7 day file): end-to-end and merged tests fail with the base's own error (`no known per-file sampling label (schema 'legoesm-swg-online-free-run-v1')`).  `r8_free_run_mean_is_end_state`: end-to-end test: `w01/day0001.npz: ssh is not an end-of-day snapshot`.  `r8_free_run_without_closure` (the day stepped without the network): end-to-end test: day 1's mean differs from the recomputed one beyond `atol=8.1e-13` (400 of 576 values) |
| (c) The controls ran on the CPU with libm; daily means existed only in the double-gyre driver's GPU mode, so a trained run would differ from the controls in device and math library too | Two required configuration keys, `free_run.device` and `free_run.transcendentals`, admitted only as the controls ran (`CONTROL_EXECUTION`, `run_swg_online_training.py:106`, checked at `run_swg_online_training.py:248`); the `free-run` command has no device flag and selects the configured device (`run_swg_online_training.py:1065`); `run_free_run` refuses any other execution (`run_swg_online_training.py:865`). The double-gyre driver's GPU-only rule (`run_swg_double_gyre.py:566`) is untouched: the stepper function is called directly | Real card, below: the no-closure free run through the new path reproduces the nu=10 control bit for bit on days 1-2.  `test_run_swg_online_training.py:621`: the free run refuses a cuda/native and a cpu/native execution.  `test_run_swg_online_training.py:735`: `free-run --device` is refused, and so is a process set up for cuda.  `test_run_swg_online_training.py:158`: a configuration with `cuda` or `native` is refused | `r8_free_run_execution_unchecked`: end-to-end test `DID NOT RAISE <class 'RuntimeError'>`.  `r8_free_run_device_flag`: device test `DID NOT RAISE <class 'SystemExit'>`.  `r8_config_free_run_execution_any`: configuration test, 2 failed (`device=cuda`, `transcendentals=native`: `DID NOT RAISE`) |
| (d) The checkpoint hash in a manifest was never checked against a file | `free_run_closure` requires an absolute `checkpoint_path` that exists (`swg_online.py:1613`) and whose SHA-256 is `checkpoint_sha256` (`swg_online.py:1615`); the scorer applies it to every run it admits | `test_run_swg_online_training.py:621`: the trained run's checkpoint moved away, then changed by one bit: both refused.  `test_swg_online_evaluation.py:435`: plants "checkpoint file missing" and "checkpoint file changed" | `r8_checkpoint_file_unchecked`: end-to-end test `DID NOT RAISE` (another digest admitted); binding test `Failed: checkpoint file missing`.  `r8_checkpoint_digest_unchecked` (file must exist, digest not compared): end-to-end test `DID NOT RAISE`; binding test `Failed: checkpoint file changed` |
| (e) After the merge, `test_free_runs_are_scored_by_the_harness_and_identify_the_closure` failed (at the base: `ValueError: day0001.npz: no known per-file sampling label (schema 'legoesm-swg-online-free-run-v1')`, 1 failed in 127 s) | Rewritten for the `free-run` command and the schema (`test_run_swg_online_training.py:578`): both trained window lengths are free-run, each names its own last checkpoint, both are admitted and day 1 is shown from the data; a complete free run is never overwritten; an untrained window length is refused | Passes (final run below) | `r8_free_run_snapshot_only` fails it with the base's own error (above) |
| Found here: the test-year chunks are `dual_test_dayAAAA_to_dayBBBB` (13 complete, days 5141-5506); every configuration labelled them `dual`, which pins 0 of them, and the driver refused any `_` in a label, so the test split could never be covered | Label check admits `_` (`run_swg_online_training.py:210`); the three experiment-one configurations and the pilot use `dual_test` for the test-year root (a correction of fact, not a choice) | `m06_chunk_label.py`: `dual` pins 0 chunks, `dual_test` pins 13 (days 5141-5506).  `test_run_swg_online_training.py:799`: every committed record label pins complete chunks on disk.  The driver's own `check-data` on the pilot: `DATA_CHECKED sources=67 batches=6` (36 s) | `r8_chunk_label_no_underscore`: records test: `chunk_label must be a lowercase [a-z0-9_] word`.  The configuration's old label is shown by `m06_chunk_label.py` (0 chunks), not by a mutant |
| Round-7 verifier, minor 1: day-file digests given as a list crashed the binding (`AttributeError`, exit 1 = FAIL) | `_day_file_pins`, the one reading shared by loader and binding, refuses a non-mapping (`nemo_testcase_l1_swg_online_evaluation.py:403`) | `test_swg_online_evaluation.py:435`: plant "day-file digests listed" refused | `r8_day_file_digests_unchecked`: binding test: `AttributeError: 'list' object has no attribute 'values'` |
| Task 5: a one-seed pilot; window-length and training-year subsets only as configuration keys | `config/swg_online/pilot_template_ring_excluded.json`. Both subsets are configuration keys the driver already validated (`curriculum.window_days`, an increasing subset of 1/2/4/8; `dataset.splits_days.train`); neither has a flag or a default. One code change: a configuration may list one seed (`run_swg_online_training.py:189`); the gate still scores only three or more | `test_run_swg_online_training.py:115`: the pilot equals candidate template+excluded except one seed, windows [1, 2] and the train split [731, 1461] (the first two training years) | `r8_three_seeds_required`: pilot test: `ValueError: seeds must list one or more distinct seeds` |

## What a free run is now

- `free-run --config <file> --seed <s> --run-root <dir> --window-days <w>`
  (`run_free_run`, `run_swg_online_training.py:852`) reads the run directory
  that `train` made: the configuration file's digest and the seed must be the
  run's; window length `w` must have its last epoch logged; that epoch's
  checkpoint must match the logged digest before it is loaded.
- It integrates 7,300 days from rest on the configured device and math
  policy, one compiled day per model day, and writes into
  `<run>/free_runs/wNN/` the dual day files and the manifest. An interrupted
  free run is archived and restarted from rest; a complete one is never
  overwritten.
- Day file: `schema = legoesm-swg-double-gyre-daily-v2` (the double-gyre
  driver's dual format, which the scorer reads as "snapshot and daily mean"),
  the two sampling labels, `daily_mean_sample_count = 216`, the three
  snapshot and three daily-mean fields. The daily means of transport are not
  written: no reader of a coarse free run uses them.
- Manifest: `schema = legoesm-swg-online-free-run-v2`; `backend` and
  `transcendentals` of the process; the resolved dg50 namelist; the whole-run
  health record; the day-file digests; and the closure block
  `{run_id, experiment, config_sha256, dataset_contract_sha256, network_seed,
  window_days, objective, wall_ring, network_spec, checkpoint_path,
  checkpoint_sha256}`.
- `train` no longer runs free runs, and its log no longer lists them (log
  schema v2). `profile` times a free-run day only on the free runs'
  configured device.

## Measurements

**No-closure free run against the nu=10 control** (`m03_dg50_control_and_mean.py`,
real dg50 card, CPU, libm): the new free-run day with no network, from rest,
days 1 and 2, against the frozen control's day files: 0 differing 64-bit
words in all six arrays (SSH, U, V on each day; largest control values
0.12 m, 0.017 and 0.032 m/s). Beyond day 2 this is not measured; the program
is the same every day.

**Daily mean on the real card** (same script; closure = seed 20261013's
initial network): the stored day-1 mean against the mean of the 216
post-step states recomputed one jitted step at a time differs by at most
4.9e-17 m (SSH; largest mean 0.036 m), 1.2e-17 and 1.3e-17 m/s (U, V). The
end-of-day states of the two programs differ in 5,555 to 5,969 of 7,056
words per field (84 x 84), by at most 8.0e-17: round-off of two different
compiled programs. The stored mean differs from the snapshot.

**The test's shortened window** (`m02_tiny_midpoint.py`, the tests' 1,000 km
card): the scorer's midpoint ratio stays below 1 on days 1-3 (largest 0.89,
U on day 3) and exceeds it on day 5 (V 1.05) and day 7 (U 1.28), in the
spin-up from rest (cause not measured). The end-to-end test therefore admits
days 1-3 of a four-day run. The scored dg50 days start at 366; round 7
measured ratios of 0.24-0.42 on dg50 days 3651-3654.

**Cost of one 20-year CPU free run** (`m04_free_run_cost.py`, run alone and
by `m08_*`/`m09_*`; results `m04_*.json`, `m08_*.json`, `m09_*.json`; real
dg50 card, seed 20261013's initial network, daily means, 8-12 days each, one
process at a time; day 1 carries the compilation and is excluded). Other
users' jobs set the load (1-minute average on 80 logical CPUs, 40 physical
cores), so every row carries it:

| CPUs the process may use | Seconds per model day (range) | CPU seconds per day | 20 years | Load during the run |
|---|---:|---:|---:|---|
| all 80, unpinned | 4.55 (4.43-4.72) | 43.8 | 9.2 h | 29 |
| all 80, unpinned | 4.76 (4.08-6.15) | 33.8 | 9.6 h | 64-68 |
| all 80, unpinned | 7.14 (5.07-11.15) | 41.1 | 14.5 h | 24 to 89 |
| 4 physical cores (8 logical) | 4.80 (4.69-5.08) | 17.4 | 9.7 h | 55 to 33 |
| 4 physical cores | 6.18 (5.08-9.40) | 20.0 | 12.5 h | 75 to 65 |
| 2 physical cores | 9.41 (7.73-13.41) | 20.8 | 19.1 h | 18 to 56 |
| 2 physical cores | 9.58 (7.19-10.58) | 22.6 | 19.4 h | 111 to 61 |
| 1 physical core (2 logical) | 9.82 (9.13-11.18) | 19.5 | 19.9 h | 27 to 18 |
| 1 physical core | 17.65 (16.95-18.61) | 33.3 | 35.8 h | 136 to 124 |
| 1 logical CPU | 25.2 (20.3-28.9) | 25.0 | 51.1 h | 93 to 135 |

Round 7 measured 4.2-4.5 s per day unpinned at load 11-26. Resident memory
is 0.94-1.04 GiB per run. For comparison, the operator's GPU profile timed a
(round-7, snapshot-only) free-run day at 0.33 s, about 40 minutes per 20
years; the configuration admits only the controls' CPU and libm (choice 3).

- **One 20-year CPU free run takes about 9-10 h** on a lightly loaded
  machine (4.5-4.8 s per day, unpinned or on 4 physical cores), and about
  14.5 h at load 90.
- **How many at once.** Unpinned, one run keeps 6-10 logical CPUs busy
  (34-44 CPU seconds per day), so unpinned runs would start slowing each
  other beyond about eight. Pin each run to its own physical cores instead
  (`taskset -c`, both hyperthreads): at 4 cores each, 10 runs fit on the 40
  cores at about 10-12.5 h; at 1 core each, 40 fit at about 20 h, the most
  runs per hour. Subtract the cores other users' jobs hold at launch (the
  load average ranged from 18 to about 230 today): a pinned run whose core
  other jobs shared slowed from 9.8 to 17.6 s per day. PLAUSIBLE: no two free
  runs were run at once (one heavy process at a time); memory is not binding
  (1 GiB each, 1 TB).
- **Pilot:** two runs pinned to 4 physical cores each (for example
  `taskset -c 0-3,40-43` and `taskset -c 4-7,44-47`), about 10-12.5 h each,
  side by side.

**Pilot GPU training time** (`m05_pilot_estimate.py`; batches per epoch by
the profile's rule `floor(days / w) // 10` on the pilot's splits: 73 training
and 84 validation batches at 1 day, 36 and 42 at 2 days; five epochs each):

| Basis | 1-day windows | 2-day windows | Training |
|---|---:|---:|---:|
| CPU, measured (round-7 profile: 188 and 390 s per step, 59 and 119 s per validation batch) | 25.9 h | 26.5 h | 52.4 h |
| GPU, CPU cost / 4.6 (the stated assumption: the fine model's compiled day, 20.73 against 4.49 ms per step) | | | 11.4 h |
| GPU, the operator's measured profile (GPU 0, commit `2a227dbc9`, experiment-one configuration, 10 windows per call; 6.23 and 12.43 s per step, 1.11 and 2.22 s per validation batch) | 0.76 h | 0.75 h | 1.51 h |

The operator's profile is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round7/profile/gpu0_profile_wpc10.json`
(written 08:04 today, read only); its per-step costs do not depend on the
split lengths, so they apply to the pilot's batch counts.

- **Expected: about 1.7 h on one GPU**: 1.51 h of steps and validation,
  0.16 h of loading and start rebalancing (1,175 batches at the 0.5 s
  measured on the CPU host in round 7) and 0.02 h of compilation. This round
  changed none of the training programs, so the profile measures them.
- The measured training step is 30-31 times faster on the GPU than on the
  CPU, so the 4.6x assumption overstated the GPU time by a factor of 6.6-6.8.
  Peak device memory at 10 windows per call: 1.10 GiB (1 day), 1.59 GiB
  (2 days).
- **What the operator must measure first:** the GPU profile the estimate
  needs already exists (above); the record is complete (146 chunks of the
  12-year record and 13 of the test year, all `complete`,
  `m01_record_status.py`), and `check-data` on the pilot passed here
  (`DATA_CHECKED sources=67 batches=6`), to be repeated on the committed
  tree; the one cost still to confirm is the free-run day on the cores the
  free runs will use, at that day's load (the table above).
- **Wall time of the pilot:** about 1.7 h of GPU training, then the
  window-2 free run (the window-1 run can start after about 0.85 h): about
  12-14 h in all.

## Running the pilot (not done here)

From a committed tree, with the `PYTHONPATH` above and `JAX_ENABLE_X64=1`:

```text
env JAX_PLATFORMS=cpu python scripts/run/run_swg_online_training.py check-data --config config/swg_online/pilot_template_ring_excluded.json --device cpu --output <dir>/pilot_check_data.json
env JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=<k> python scripts/run/run_swg_online_training.py train --config config/swg_online/pilot_template_ring_excluded.json --device cuda --physical-gpu <k> --windows-per-call 10 --seed 20261013 --run-root <root>
# after training_log.json lists w01_e05, and again with --window-days 2 after w02_e05:
env JAX_PLATFORMS=cpu taskset -c 0-3,40-43 python scripts/run/run_swg_online_training.py free-run --config config/swg_online/pilot_template_ring_excluded.json --seed 20261013 --run-root <root> --window-days 1
```

One seed is not a gate score: the held-out gate needs three or more seeds.
Each free run can be admitted on its own (`admit_trained_run`) and loaded
window by window.

## Choices made

ASKED (the task): one manifest schema in one place for writer and reader,
with an admission test and a refusal test; daily means from every step
through the double-gyre driver's machinery, with the scorer's per-file
labels; the CPU and libm by an explicit configuration key with no default in
code; the scorer checking the checkpoint file; the merged test fixed; one
end-to-end CPU test with a shortened window; the pilot's objective, wall
ring, seed, window lengths, epochs, batch, optimiser, training years,
validation and 20-year CPU free runs; window-length and training-year subsets
only as configuration keys; the cost statements.

UNASKED, each offered for revert:

1. Free runs moved out of `train` into the separate `free-run` command
   (forced by GPU training with CPU/libm free runs: libm is CPU-only); the
   training log drops its free-run list (log schema v2).
2. The names: flat `checkpoint_path` and `checkpoint_sha256` (round-7
   evaluation decision 4, option 1: the scorer's name kept), the round-7
   driver's other identity keys kept; manifest schema `...-free-run-v2`;
   configuration schema `...-training-v2`; sampling value
   `end_of_day_snapshot_and_daily_mean`.
3. The free-run execution admits only the controls' (cpu, libm); no GPU free
   runs.
4. Day files carry the double-gyre driver's dual label, not a new free-run
   label; no transport means.
5. The dual per-file labels moved from the scorer into the shared module.
6. A configuration may list one seed (needed by the one-seed pilot).
7. The label check admits `_`; the test-year label corrected to `dual_test`
   in all four configurations (a fact, recorded here because it changes
   committed files).
8. `_day_file_pins` refuses non-mapping digests (round-7 verifier minor 1).
9. `profile` times the free-run day only on the configured device.
10. `free-run` binds the run directory by the configuration digest and seed,
    refuses an untrained window length, never overwrites a complete free run.
11. The design note's driver section, cost paragraph, requirement row and
    open decision 5 updated to the new behaviour (round-6 review paragraph
    left as it was).
12. The free run's inert key stays `PRNGKey(network_seed)` (round 7).

## Tests

Final runs on the final code, one file at a time:

- `tests/ml/test_run_swg_online_training.py`: `47 passed in 361.27s`
  (end-to-end test 47 s; merged test 7 s after its shared fixture).
- `tests/ocean/fidelity/test_swg_online_evaluation.py`: `36 passed in 48.81s`.
- `tests/ml/test_swg_online.py` (the shared module's tests; file unchanged):
  `54 passed in 326.19s`.
- `tests/test_scripts_layout.py`: `4 passed in 0.03s`.
- `tests/test_no_private_cross_imports.py`: `1 failed, 5 passed`; the failure
  is the base's (`swg_online.py` imports `_entry_from_state` at
  `2a227dbc9`, line 29).
- `tests/test_no_hardcoded_constants.py`: `1 failed, 5416 passed, 2 skipped`;
  the failure is the base's (`273.15` in `tests/unit/test_jra55_do.py`).
- `ruff check` on the five changed Python files: `All checks passed!`.
- Reversions: 13 of 13 fail their tests (`reversions.jsonl`, logs in
  `mut/`); the unmutated tree gives `28 passed in 187.94s`. A first pass
  (`reversions_pass1.jsonl`) ran before two edits and is superseded: its
  scorer reversion restored only round 7's key check, and the end-to-end test
  then listed its corrupted manifests in another order.
- Citations: `check_citations.py` on this receipt: `CITATIONS 30 checked,
  0 failing`; a copy with one citation moved by two lines gives `1 failing`.

## Review

UNREVIEWED. By instruction no external review was run; the operator runs one
review of the round.

## Open

- The scorer's midpoint test was measured on no trained closure; a closure
  with strong variability faster than a day could be refused loudly (round 7,
  unchanged). On the test card the spin-up from rest already exceeds 1 on
  days 5 and 7.
- Seed and controls are shown identical (no closure) on days 1-2 only.
- An interrupted free run restarts from rest: up to 10-12 h lost.
- The checkpoint path is absolute: moving a run directory makes its free runs
  refuse until the manifest is rewritten.
- A stray busy loop of another session (`until false; do :; done`, running
  71 days) holds one core of this machine; not touched.

## UNVERIFIED

- Nothing ran on a GPU. The GPU numbers are the operator's profile, read from
  its file; the pilot's own GPU profile was not run.
- The concurrency statement is not measured with two runs at once.
- The CPU timings ran beside other users' jobs at 1-minute loads of 18 to
  136.
- The free run on the real card ran 12 days at most; a 20-year run was not
  made.
- The tiny-card end-to-end test trains two steps; a trained network's free
  run on the real card was not made.
