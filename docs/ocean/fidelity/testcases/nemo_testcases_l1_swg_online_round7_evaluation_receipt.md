# NEMO testcase L1 SWG online learning — Round 7 evaluation receipt

Date: 2026-10-14.  Scope: codex round-6 defects 2, 3 and 4 (the held-out
evaluation) and the operator's adequacy measurement (the rotational-velocity
early-path value was an array), then the four items the round-7 verifier
left open (section "Verifier follow-up").  Work copy `claude-swgonl-r7-eval`,
base `d63e2b9f4`, changes uncommitted.  CPU only, fp64 (`JAX_ENABLE_X64=1`).
Files changed: the held-out evaluation and the reference-adequacy tool under
`scripts/validate/ocean_fidelity/testcases/`, their two test files under
`tests/ocean/fidelity/`, and this receipt (the follow-up touched only the
evaluation, its test file and this receipt).  The shared climate instrument,
the restriction module, the driver and the training module are untouched.
Scratch scripts, logs, JSON and reversion trees:
`/home/dbalwada/.claude/jobs/b3690d5d/tmp/claude_r7/eval/impl/`; follow-up:
`/home/dbalwada/.claude/jobs/b3690d5d/tmp/claude_r7b/impl/`.

## Round 7 evaluation — verdict

- **Adequacy: PASS** on the registered inputs once the rotational-velocity
  value is one number per day (round-6 code: FAIL from the array alone).
  Days 1-8 normalised RMS 0.44-0.47 % (rotational velocity), 0.36-0.42 %
  (sea level), 0.44-0.50 % (vorticity); pattern correlation >= 0.999987.
- **Every gate tolerance and allowance is now the truth's spread.**  The
  noise allowance was the last one taken from a control; it is now the
  truth's half-range.  The perfect model passes and both no-closure controls
  fail, for both objectives (margins below).
- **The `template` frozen record did not verify at the merged base**
  (`freeze --objective template --verify` exit 1): the round-6 GPU fix
  changed the filtered restriction's arithmetic after the record was frozen.
  Re-measured by the same rule (32 of 189 numbers move, at most 5.6e-15
  relative); a 2-second test now pins both operators' arithmetic.
- **A trained-closure score needs three or more distinct runs**, each
  manifest naming its checkpoint SHA-256, network seed, objective and
  training window, matching the request.  Runs must differ in their day files
  (read from the manifest exactly as the loader reads them, a non-empty set)
  and in their scored data (one digest per loaded window).  The reviewer's
  three-labels-one-run plant and 17 others are refused before anything is
  scored; one run's data re-saved under new bytes is refused once its windows
  are loaded, before the gate.
- **Sampling is decided per file and shown from the data.**  A missing or
  unknown per-file label refuses; nothing falls back to the manifest.  A
  day's OWN daily mean under its snapshot names is detected from the data in
  every dual archive, including all 5840 scored days of the gate truth; the
  NEXT day's daily mean under those names is not (it passes the test).  A
  finite snapshot beside a non-finite daily mean is refused.  A
  single-sampling archive is admitted only under a manifest frozen in code.
  The reviewer's renamed payload is refused with and without a forged label.

## Findings, dispositions, evidence, reversions

Every reversion ran in its own scratch tree (`mut/<name>/`, exact text
replacements in `mutations.py`, decisive lines in `mutations.log`); the work
copy was never touched.  The unmutated round-7 tree gave `41 passed`.  Line
anchors below are the current code's (after the follow-up).

| Finding | Disposition | Evidence | Reversion: the test that fails |
|---|---|---|---|
| Operator: the rotational-velocity early value is an array (eight lists of 6400 values; "first day above 0.1" = 1562) | Fixed: the RMS runs over every axis but the day, pooling both components and every wet point (`nemo_testcase_l1_swg_online_reference_adequacy.py:105`), as the pattern correlation already did | `test_swg_online_reference_adequacy.py:66`: one value per day, equal to the known answer `delta_k / sqrt(sum_wet(u^2+v^2))` to 1e-12; real rerun PASS | `ad_rotational_axis1` (axis 1 again): `:66` fails, the per-day values are lists |
| Codex 2: spectral-health allowance from the worse control's spread | Fixed: the truth's half-range (`nemo_testcase_l1_swg_online_evaluation.py:904`) | `test_swg_online_evaluation.py:150`: per metric, limit = worse control median + truth half-range; half the allowance passes, 1.5 times fails.  `:176`: widening every control's window-to-window spread tenfold moves no threshold | `ev_noise_control_spread`: `:150` and `:176` fail for both objectives (all three noise limits move) |
| Found here: the `template` record frozen at `a0b5535b2` is not reproduced at `d63e2b9f4` (verify exit 1: `truth_loo`, `controls`, `bars` differ; windows, pins, manifests equal) | Cause CONFIRMED: the round-6 GPU fix (`83588a379`) evaluates the filtered restriction as matrix products; on fine day 366 it differs from the freeze-time module by up to 1.5e-13 relative in sea level and 4e-16 in velocity, and is bitwise reproducible across processes and Eigen threading.  Re-measured under the same rule (`nemo_testcase_l1_swg_online_evaluation.py:504`); a digest test pins each operator's arithmetic (`test_swg_online_evaluation.py:646`) | template verify after re-freezing: `PASS freeze`, exit 0, all six keys equal; symmetric verify passes unchanged | `ev_operator_freeze_time_arithmetic` (the freeze-time module restored): `:646[template]` fails (digest `5dd998ac...` instead of `0c1e9367...`); `[symmetric]` passes, the box arithmetic being unchanged.  The old numbers themselves are caught only by `freeze --verify` (exit 1 above) |
| Codex 3: three labels can bind one run; a manifest need not identify checkpoint or seed | Fixed: `bind_trained_runs` (`nemo_testcase_l1_swg_online_evaluation.py:1067`) runs before anything is scored (`:1151`); the request states `--window-days` (`:1221`) | `test_swg_online_evaluation.py:412`: 16 plants refused (one run under three labels; symlinked root; copied manifest; duplicate network seed; duplicate checkpoint; copied day files under a relabelled manifest; malformed checkpoint; boolean seed; objective differs; window differs; two runs; each of the four keys missing; an inadmissible window), none reaches the scorer; three distinct runs are then scored.  The follow-up adds two plants (below) | `ev_no_binding`: `Failed: reviewer: one run, three labels`.  `ev_no_day_file_check`: `Failed: copied day files, relabelled`.  `ev_no_request_match`: `Failed: objective differs`.  `ev_no_identity_keys`: `KeyError: 'checkpoint_sha256'`.  `ev_no_directory_check`: the reviewer's plant is still refused, as a duplicate manifest; `:412` fails on which check refuses it |
| Codex 4: a manifest label decides the sampling, a missing one defaults to snapshot, and a renamed daily mean is read | Fixed: per-file admission with no fallback (`nemo_testcase_l1_swg_online_evaluation.py:302`); the data test (`:330`) in both readers (`:385`, `:436`); single-sampling archives only under frozen manifests (`:426`, set at `:51`) | `test_swg_online_evaluation.py:541`: the reviewer's plant, without and with a forged v1 label.  `:562`: the daily-mean product.  `:568`: dual archives of real fields; all three fields or any one renamed is refused, four label plants are refused, genuine days are admitted bit-equal.  `:624`/`:656`: real gate truth admitted; its day 367 renamed is refused.  Both `freeze --verify` runs read every scored fine day through the test | `ev_single_sampling_admitted`: `:541[v1]` DID NOT RAISE.  `ev_no_data_check_native`: `:568` DID NOT RAISE.  `ev_no_data_check_fine`: `:656` DID NOT RAISE.  `ev_dual_labels_unchecked`: `:568` DID NOT RAISE.  `ev_label_fallback`: `:541[None]` and `:568` fail; the plants are still refused, by the single-sampling rule, so these tests pin which layer refuses.  `ev_single_with_mean_fields`: `:562` fails; refused anyway (no snapshot fields) |
| Consequence: the last snapshot of a window needs the next day | `available_windows` requires it (`nemo_testcase_l1_swg_online_evaluation.py:981`) | `test_swg_online_evaluation.py:680` | `ev_available_windows_old`: `:680`, 1095 completed days give `[(366, 1095)]`, not `[]` |

## Verifier follow-up (round 7b)

The round-7 verifier built four plants against the code above
(`claude_r7/eval/verify/plants.py`); a copy re-run on the current code
(`claude_r7b/impl/plants.py`, output `plants.log`) gives the second
column.  Reversions ran as above (`claude_r7b/impl/mutate.py`, logs in
`mut/<name>.log`); the unmutated tree gives `36 passed`.

| Verifier item | Plant now | Fix | Test (the verifier's plant) | Reversion: the test that fails |
|---|---|---|---|---|
| 1. A renamed daily mean passes the data test when a daily-mean field holds one NaN (the test skipped any non-finite input) | REFUSED (`the ssh snapshot is finite but its own day's daily mean is not`) | Only a non-finite SNAPSHOT field skips the test (`nemo_testcase_l1_swg_online_evaluation.py:346`); a finite snapshot beside a non-finite daily mean of its own or the next day refuses (`:348`).  A skipped snapshot is in the returned arrays, so the window's score row is `physical_health: {all_finite: false, passes: false}`, the gate reports `physical_health_pass: false`, `pass: false`, and `score` exits 1 (FAIL); the gate truth refuses instead (`window_statistics` admits only finite arrays) | `test_swg_online_evaluation.py:610`: the plant (daily means under the snapshot names, one NaN in every day's daily-mean fields); a genuine snapshot beside a NaN in its own day's mean (day 400) and in the next day's (day 406 for day 405).  The skip-and-fail-health path: `:360` (NaN in day 5's u snapshot; that window's row fails health) | `r7b_nan_skip_any` (round-7 skip restored): all three cases `DID NOT RAISE` |
| 2. One run's day files bind under three labels when the digests sit under the loader's second key (`output_daily_sha256`) | REFUSED (`s0 and s1 share day files`) | The binding takes the day files from the loader's own function (`nemo_testcase_l1_swg_online_evaluation.py:395`, used at `:421` and `:1111`) and refuses an empty set (`:1112`).  Chosen over refusing the second key: either is one changed line plus the empty-set check; the shared function (4 more lines) makes binding and loader one reading of one fact, while a refusal of the second key would stay correct only as long as the loader reads `daily_sha256` first; no manifest contract narrows | `test_swg_online_evaluation.py:412`, plants "copied day files, second digest key" (the verifier's: three relabelled manifests, one set of day files, digests under the second key) and "no day files pinned" | `r7b_binding_round7` (round-7 binding): `Failed: copied day files, second digest key`.  `r7b_binding_empty_allowed`: `Failed: no day files pinned`.  `r7b_binding_first_key_only`: the plant is still refused, by the empty-set rule, so `:412` fails on which check refuses it (`Regex pattern did not match`) |
| 3. Distinctness held on file bytes only: one run re-saved three times with an extra unused entry was admitted | `bind_trained_runs` alone still ADMITS it (it compares manifests and bytes, by design); scoring REFUSES it (the verifier's archives carry no deck or health record, so the same plant is rebuilt with them in the test) | Each score row carries `scored_data_sha256`, one SHA-256 over the window's loaded (ssh, u, v) arrays (`nemo_testcase_l1_swg_online_evaluation.py:1062`); before the gate, two seeds holding identical data in any scored window (any window of either) refuse (`:1162`) | `test_swg_online_evaluation.py:498`: the plant through the real loader and scorer (one window, 400-402, truth climatology from the same days): the binding admits, `score_trained_closure` refuses "hold identical data"; three runs whose fields are scaled by 1 + k*1e-9 are scored | `r7b_no_data_distinctness`: `DID NOT RAISE` |
| 4. Receipt wording: "a daily mean under snapshot names is detected" | P2 (the NEXT day's daily mean under snapshot names): ADMITTED, as the arithmetic says it must be | Wording corrected (verdict, and the section below); launch prerequisites listed (section before the decisions) | none (wording) | none |

## What tells a daily mean from a snapshot in this model

Chosen: a property of the data, not a digest (the driver is not in this
piece).  In a dual archive each day file holds the end-of-day snapshot `s_d`
and the daily mean `m_d` of the same model day.  `s_d` is half a day after the
centre of `m_d` and half a day before the centre of `m_{d+1}`, so it lies
nearer their average than its own day's mean:
`|s_d - (m_d + m_{d+1})/2| < |s_d - m_d|` (second order in the day's change on
the left, first order on the right).  A day's OWN daily mean under its
snapshot names makes the right side exactly zero, so the test fails.  The
NEXT day's daily mean under those names passes (left side half the
day-to-day change, right side all of it): the test does not catch that
one-day shift (round-7 verifier plant P2, admitted).  It is checked per field
(sea level, U, V), with no parameter, on every scored day
(`nemo_testcase_l1_swg_online_evaluation.py:330`).  Only a non-finite
snapshot field skips it; a finite snapshot beside a non-finite daily mean
of its own or the next day refuses.

Ratio left/right on real data (must stay below 1; a day's own mean renamed
gives infinity):

| Archive | Days | Sea level, max (median) | U, max (median) | V, max (median) |
|---|---|---|---|---|
| 6.25 km gate truth, dual files, every scored day (whole arrays, as the gate computes it) | 366-6205 | 0.278 (0.151), day 4130 | 0.640 (0.485), day 2659 | 0.697 (0.466), day 5097 |
| round-2 box truth, dg50 snapshots vs their daily-mean twin | 1-1094 | 0.764 (0.153), day 15 | 0.602 (0.381), day 12 | 0.574 (0.346), day 864 |
| 3.125 km sensitivity product vs its daily-mean twin | 1-364 | 0.765 (0.232), day 17 | 0.602 (0.373), day 12 | 0.471 (0.341), day 319 |
| dg50 nu=10 from day 3650, the driver's daily-mean stepper on CPU | 3651-3654 | 0.238 | 0.420 | 0.408 |

Two candidates were measured and rejected: the divergence-to-vorticity ratio
does not separate them (box truth day 400: 3.91e-4 snapshot, 3.80e-4 mean;
the flow is balanced), and the one-day volume budget with product-of-means
transports separates them only 6-17 fold (snapshot residual 1.1 % coarse,
3.1 % fine; daily mean 18-22 %) and needs a tolerance.

Single-sampling archives (the two controls and the round-5 box truth) hold
nothing a daily mean could not also hold, so they are admitted only under the
three manifests frozen in code (`:51`).  Their sampling is shown from the data
here, once:

- Controls: restoring the archived file of day d into the driver's own model
  (fp64, libm, CPU) and stepping one day reproduces the archived day d+1 bit
  for bit, for nu=10 and nu=80 at d = 366, 3650 and 6205 (6 of 6, zero
  differing values).  Control of the instrument: stepping from nu=10's
  day-3651 daily mean instead misses the archived day 3652 at every wet point
  (up to 0.17 m, 0.35 and 0.51 m/s).
- Round-5 box truth: the midpoint test against its daily-mean twin, all 1094
  days (table).

## Gate margins

From the frozen tables (template re-frozen) through `held_out_gate` itself
(`margins.py`, `margins_refrozen.json`; identical to the printed digits with
the old template numbers).  Margin = how far inside the gate (negative =
fails that metric).  Every allowance is the truth's half-range.

`symmetric` (box truth):

| Metric | Gate | Truth half-range | Perfect model (margin) | nu=10 (margin) | nu=80 (margin) |
|---|---|---|---|---|---|
| transport RMSE, m3/s | below 7.961e5 | 2.967e5 | 5.494e5 (+2.47e5) | 1.093e6 (-2.97e5) | 1.444e6 (-6.48e5) |
| KE spectrum L1 | below 0.1819 | 0.0126 | 0.0401 (+0.142) | 0.1945 (-0.0126) | 0.3214 (-0.140) |
| wet EKE-map NRMSE | below 0.2733 | 0.0352 | 0.2326 (+0.0407) | 0.3085 (-0.0352) | 0.6733 (-0.400) |
| EKE ratio | in [0.9311, 1.0262] | 0.0476 | 1.0039 (+0.0224) | 0.7483 (-0.183) | 0.4525 (-0.479) |
| top-octave KE fraction | at most 0.05414 | 4.75e-4 | 0.00865 (+0.0455) | 0.05366 (+4.75e-4) | 0.02738 (+0.0268) |
| top-octave enstrophy fraction | at most 0.4384 | 0.00575 | 0.1245 (+0.314) | 0.4327 (+0.00575) | 0.2940 (+0.144) |
| native grid-noise index | at most 0.9388 | 0.00827 | 0.4839 (+0.455) | 0.9306 (+0.00827) | 0.7053 (+0.234) |
| **verdict** | | | **PASS** | **FAIL** (skill, variability) | **FAIL** (skill, variability) |

`template` (filtered truth):

| Metric | Gate | Truth half-range | Perfect model (margin) | nu=10 (margin) | nu=80 (margin) |
|---|---|---|---|---|---|
| transport RMSE, m3/s | below 1.103e6 | 2.586e5 | 4.602e5 (+6.43e5) | 1.362e6 (-2.59e5) | 1.680e6 (-5.76e5) |
| KE spectrum L1 | below 0.4354 | 0.0141 | 0.0404 (+0.395) | 0.4496 (-0.0141) | 0.5421 (-0.107) |
| wet EKE-map NRMSE | below 0.3828 | 0.0433 | 0.2291 (+0.154) | 0.4261 (-0.0433) | 0.6014 (-0.219) |
| EKE ratio | in [0.9265, 1.0378] | 0.0557 | 1.0022 (+0.0356) | 1.0738 (-0.0360) | 0.6494 (-0.277) |
| top-octave KE fraction | at most 0.053662 | 1.80e-6 | 3.33e-5 (+0.0536) | 0.053660 (+1.80e-6) | 0.02738 (+0.0263) |
| top-octave enstrophy fraction | at most 0.43274 | 4.48e-5 | 7.00e-4 (+0.432) | 0.43270 (+4.48e-5) | 0.2940 (+0.139) |
| native grid-noise index | at most 0.93278 | 0.00222 | 0.2112 (+0.722) | 0.93057 (+0.00222) | 0.7053 (+0.227) |
| **verdict** | | | **PASS** | **FAIL** (skill, variability) | **FAIL** (skill, variability) |

- nu=10 fails skill by exactly the truth half-range on every skill metric: it
  is the better control, and the threshold is its median minus that range.
- The controls pass noise by construction: the worse control's median is the
  reference level.  Round 6 put the noise limits at 0.05445 / 0.4379 / 0.9367
  for both objectives.  Now `symmetric` tightens top-octave KE and loosens
  the other two (the box truth's spread is wider than nu=10's there), and
  `template` sits within 0.0022 of nu=10's medians (decision 2).

## Adequacy rerun and the year-1 bars

`adequacy_r7.json` (scratch): the round-6 operator command with the JSON
redirected; exit 0, `PASS measured=1 adequate=True`.  The year-1 numbers
equal the operator's round-6 run; only the early rotational-velocity
reduction changed.

| Metric | Year-1 score | Registered bar (largest pair, at least 10 %) | Pair 1-2 | Pair 2-3 | Bar from pair 2-3 only |
|---|---|---|---|---|---|
| kinetic energy | 1.13 % | 19.58 % | 19.58 % | 0.92 % | 10 % |
| eddy kinetic energy | 9.59 % | 18.59 % | 18.59 % | 2.42 % | 10 % |
| jet position / basin width | 3.63 % | 10 % | 4.02 % | 3.48 % | 10 % |
| KE spectrum L1 | 8.68 % | 19.81 % | 19.81 % | 8.84 % | 10 % |

- With the non-spin-up pair (years 2-3) every bar falls to the 10 % floor and
  the year-1 verdict is still PASS; eddy kinetic energy then passes by 0.41
  points.  The registered rule is unchanged (decision 3).
- The 20-year coarse restriction cannot be written by the coarse-graining
  command: it refuses a fine run without archived mean transports, and this
  run has none.  The bars need only the restricted snapshots, which the
  evaluation already makes in memory with the same box operator (days
  366-367 equal the stored box truth bit for bit,
  `test_swg_online_evaluation.py:624`).  Exact
  command, from the work copy root with the environment above
  (`year_pairs_command.sh` in scratch holds the same text):

```bash
python -c "
import sys, json, importlib.util as u
T = 'scripts/validate/ocean_fidelity/testcases/'
def load(n):
    s = u.spec_from_file_location(n, T + n + '.py'); m = u.module_from_spec(s); s.loader.exec_module(m); return m
ev = load('nemo_testcase_l1_swg_online_evaluation'); ad = load('nemo_testcase_l1_swg_online_reference_adequacy')
from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_DG50_NAMELIST as N, build_swg_card
card = build_swg_card(N); D = int(sys.argv[1]); Y = list(range(int(sys.argv[2]), int(sys.argv[3]) + 1))
stats = [ad.diagnose(ev._native_array_provider(f'year{y}', ev.restrict_fine_truth_window(
    ev.FINE_TRUTH_ROOT, (y - 1) * D + 1, y * D, objective='symmetric')),
    days=D, card=card, namelist=N, diagnostics='online') for y in Y]
pairs = {f'{a}-{a + 1}': ad.adequacy_metrics(p, q, basin_width_m=N.rn_domsiz_m)['scores']
         for a, p, q in zip(Y, stats, stats[1:])}
from pathlib import Path; out = Path(sys.argv[4]); out.parent.mkdir(parents=True, exist_ok=True)
json.dump(pairs, out.open('w'), indent=1, sort_keys=True)
print({m: max(p[m] for p in pairs.values()) for m in next(iter(pairs.values()))})
" 365 2 19 /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round7/reference/year_pairs_2_19.json
```

- Years 2-19 give 17 adjacent pairs; year 20 cannot be read, because its last
  snapshot's test needs day 7301.  Cost, measured: years 2-3 took 104 s wall
  (about 14 s start-up, about 45 s per year), so years 2-19 take about 14 min
  on this CPU in one process, reading about 180 GB.  Validated against a
  known answer: run on years 2-3 it reproduces the adequacy tool's pair 2-3
  from the stored coarse truth bit for bit (`year_pairs_2_3.json`).

## Launch prerequisites (each blocks scoring a trained run until met)

Read off the round-7 training driver (work copy `claude-swgonl-r7-driver`,
uncommitted) and checked against this scorer with files and manifests built
to that driver's keys (`claude_r7b/impl/driver_file_admission.py`,
`driver_manifest_binding.py`); no driver run was made.

1. **The closure checkpoint hash is not checked against a file.**  The
   binding checks only that `closure.checkpoint_sha256` is a 64-hex string
   that differs between seeds.  Nothing hashes a checkpoint file, so nothing
   ties a free run to the network that made it.  To remove: the score request
   names each seed's checkpoint file and the scorer hashes it against the
   manifest (the driver already records the path); tying the run's DATA to
   that network further needs a one-day replay from the checkpoint, as done
   for the controls.
2. **The driver and the scorer disagree on the checkpoint key.**  The driver
   writes `closure.checkpoint = {path, sha256}`; the scorer requires
   `closure.checkpoint_sha256`.  A manifest with the driver's closure block is
   refused: `manifest does not identify ['checkpoint_sha256']`.  Decision 4
   settles which side renames.
3. **Trained free runs must store daily means, and today nothing writes them
   for a trained run.**  The training driver's free run writes end-of-day
   snapshots only, under its own per-file schema
   `legoesm-swg-online-free-run-v1`, which this scorer refuses (`no known
   per-file sampling label`; a snapshot-only archive could not show its
   sampling anyway).  The only writer of daily means, the double-gyre driver,
   runs no closure and writes them only in its GPU mode (native
   transcendentals); it refuses them on the CPU.  Both frozen controls ran on
   the CPU (`cpu-scanned`, libm, snapshot-only).
   - What this implies: a trained run written on the GPU would differ from
     both controls in two things, its closure and its device and math
     library.  The device difference is round-off per step, which in this
     chaotic flow gives a different realisation; whether it moves two-year
     window statistics relative to the truth's spread is unmeasured.  So a
     skill or noise margin against the controls could not be attributed to
     the closure alone.
   - What removes it: the training driver's free run writes the day's mean
     beside each snapshot, with this scorer's dual per-file labels, on the CPU
     with libm, like the controls; seed and controls then differ only by the
     closure.  The round-7 driver verifier showed that driver's free run with
     no closure, on the CPU, reproduces the nu=10 control bit for bit on days
     1-2.  The alternative, re-running both controls on the GPU and
     re-freezing their tables, makes a new baseline.
4. **Free runs must reach day 6206.**  The last window ends on day 6205 and
   its last snapshot's test reads day 6206's daily mean.  All three driver
   configurations set 7300 days (met); a shorter run is refused (missing day
   file).
5. **The two round-7 pieces do not merge cleanly.**  The driver's test
   `test_free_runs_are_scored_by_the_harness_and_identify_the_closure` reads
   `FINE_SNAPSHOT_LABEL` (renamed `SNAPSHOT_LABEL` here in round 7) and loads
   the driver's snapshot-only files through this loader, which now refuses
   them; it must change with items 2 and 3.

Consequence of follow-up item 1, for the operator: a seed whose fields turn
non-finite during days 367-6206 is now refused at its last finite day
(`score` exits 2, naming the day), not scored as an unhealthy seed (exit 1),
and no report is written for the other seeds.  Either way it cannot pass.
Non-finite from day 366 or earlier fails health; after day 6206 it fails on
the manifest's whole-run health record.

## Decisions for the owner

1. **Trained free runs must be dual archives** (snapshot and daily mean in
   every file), or the gate cannot show their sampling from the data.
   Correction (follow-up): the round-7 option "(1) daily means on the GPU, no
   code change" was wrong: the training driver's free run writes no daily
   mean in any mode (launch prerequisite 3), so every option needs a driver
   change.  (1) the free run writes daily means on the CPU with libm, like
   the controls (my pick now: seed and controls differ only by the closure);
   (2) the same on the GPU (device and math library differ from the controls,
   effect unmeasured); (3) admit snapshot-only trained runs on their label
   alone (codex defect 4 stays open for them).
2. **The truth's half-range is almost zero for `template` noise** (1.8e-6,
   4.5e-5, 2.2e-3 against nu=10 medians of 0.054, 0.433, 0.931): a closure
   exactly as noisy as nu=10 then passes noise about half the time (the
   round-5 sampling concern returns, by design).  (1) keep, as codex requires
   (my pick: a closure clearly quieter than nu=10 passes reliably); (2) scale
   the truth's relative half-range to the control's level (still
   truth-derived; allowances 0.0029, 0.028, 0.0098 under `template`); (3) other.
3. **Adequacy year-1 bars** (round-6 decision 2, still open).  (1) keep the
   registered rule (bars 18.6-19.8 %); (2) non-spin-up pairs only (my pick:
   bars 10 %, verdict unchanged, EKE margin 0.41 points); (3) years 2-19 of
   the 20-year truth with the command above (about 14 min CPU).
4. **Trained-run manifest contract.**  Each manifest must carry
   `closure: {checkpoint_sha256, network_seed, objective, window_days}` and
   the request states `--objective` and `--window-days`.  The training and
   free-run driver (another piece) must write exactly these; the round-7
   driver writes `closure.checkpoint.{path, sha256}` instead, so today every
   driver manifest is refused (launch prerequisite 2).  (1) keep these names
   and change the driver (my pick); (2) rename to whatever that driver
   writes, in the same merge.
5. **Template re-freeze.**  (1) keep the re-measured numbers (my pick: same
   rule, still no trained model, changes at 5.6e-15); (2) revert to the
   round-6 numbers, and every `template` score and verify refuses.

## Choices made

| Choice | Asked? |
|---|---|
| Noise allowance = the truth's half-range above the worse control's median | ASKED (brief: every allowance from the truth) |
| Rotational velocity: one pooled vector RMS per day over both components and all wet points | ASKED (brief: one number, the pre-registered definition; matches the correlation's pooling and the predictability tool's RMS speed) |
| Data property = the midpoint test, per field, parameter-free, whole arrays | ASKED (brief delegated: "find a property ... and say which") |
| `template` record re-measured at the merged code (32 of 189 numbers, at most 5.6e-15 relative) | UNASKED; required for "frozen in code ... for both objective operators"; decision 5; offered for revert |
| Digest test pinning each truth operator's arithmetic (fine day 366) | UNASKED; the gate that would have caught the drift; offered for revert |
| Single-sampling archives admitted only under manifests frozen in code; trained runs must therefore be dual | UNASKED; decision 1; offered for revert |
| ~~A non-finite field skips the sampling test~~ (round 7) replaced in the follow-up: only a non-finite SNAPSHOT skips it; a finite snapshot beside a non-finite daily mean refuses | ASKED (follow-up brief).  Consequence, not asked: a seed turning non-finite during days 367-6206 refuses instead of failing health (launch prerequisites section) |
| Follow-up: the binding reads day files through the loader's own function, not "refuse the second key" | ASKED (brief offered both; reason in the follow-up table) |
| Follow-up: the shared function itself (4 lines) rather than copying the loader's expression into the binding | UNASKED; offered for revert to the copied expression |
| Follow-up: data distinctness = one SHA-256 per loaded window; any window shared by two seeds (any window of either) refuses; checked at score time, after loading and before the gate, so `bind_trained_runs` alone still admits re-saved copies | Refusing identical scored arrays: ASKED.  UNASKED: per-window granularity, the any-window rule, the placement, and the digest stored in every score row (`scored_data_sha256`, so the report carries it); offered for revert |
| Follow-up: decision-1 recommendation moved from GPU to CPU daily means | UNASKED (the round-7 premise "no code change" was false); a recommendation only |
| The fine-truth reader reads one more day per window; available windows require the next day pinned | UNASKED; consequence of the test; offered for revert |
| `closure.*` key names and the `--window-days` flag | UNASKED; decision 4 |
| Runs sharing any day-file digest refused (beyond the four named duplicates) | UNASKED; otherwise a copied run under a relabelled manifest passes; offered for revert |
| Roots compared as directories (device, inode), so a symlink or bind mount is the same root | UNASKED; offered for revert |
| Gate report schema renamed `v2` to `v3` (new fields) | UNASKED; offered for revert |
| Adequacy report gains per-pair year-to-year scores (pure addition) | UNASKED; offered for revert |

## Validation

From the work copy root, CPU, `JAX_ENABLE_X64=1`, pytest with
`-q -p no:cacheprovider`, one file at a time.

| Run | Decisive line |
|---|---|
| `pytest tests/ocean/fidelity/test_swg_online_evaluation.py` | `32 passed in 33.95s` |
| `pytest tests/ocean/fidelity/test_swg_online_reference_adequacy.py` | `9 passed in 24.77s` |
| `freeze --objective symmetric --verify` | `PASS freeze`, exit 0, 14.5 min; all six frozen keys equal; truth passes; nu=10, nu=80 and both plants fail |
| `freeze --objective template --verify`, round-6 numbers | `FAIL freeze`, exit 1: `truth_loo`, `controls`, `bars` unequal (at most 5.6e-15 relative); self-checks pass |
| `freeze --objective template --verify`, re-frozen | `PASS freeze`, exit 0, 28 min on the loaded machine; all six frozen keys equal; truth passes; nu=10, nu=80 and both plants fail |
| adequacy, registered inputs | `PASS measured=1 adequate=True`, exit 0 |
| year-pair command, years 2-3 | pair 2-3 bit-equal to the adequacy tool's; 104 s |
| reversions (15 trees) | each listed test fails; unmutated tree `41 passed` |
| `ruff check` on the four changed Python files | `All checks passed!` |
| Follow-up: `pytest tests/ocean/fidelity/test_swg_online_evaluation.py` | `36 passed in 28.10s` |
| Follow-up: `pytest tests/ocean/fidelity/test_swg_online_reference_adequacy.py` (file unchanged) | `9 passed in 20.60s` |
| Follow-up: verifier plants re-run (`plants.py`) | P1 REFUSED; P2 ADMITTED (the documented limit); P3 REFUSED; P4 through the binding alone ADMITTED (bytes-level by design), through scoring REFUSED (`test_swg_online_evaluation.py:498`) |
| Follow-up: reversions (5 trees, `mutate.py`) | each fails its test (table above); unmutated tree `36 passed in 28.55s` |
| Follow-up: in-window blow-up (`blowup_inside_window.py`: all fields non-finite from day 403) | refused at day 402, `the next day's daily mean is not`; non-finite throughout the window: admitted, arrays carry the NaN (fails health when scored) |
| Follow-up: `freeze --objective symmetric --verify` (fine days 366-6206 through the new non-finite rule) | `PASS freeze`, exit 0, 13 min; all six frozen keys equal; truth passes; nu=10, nu=80 and both plants fail.  `template` not re-run: same reader and check, only the restriction after it differs |
| Follow-up: `ruff check` on the evaluation and its test file | `All checks passed!` |

## Citation anchors

The citation gate's map has no entries for this lane's files (as in round 6).
The anchor of every cited line, for the gate's owner:

```text
nemo_testcase_l1_swg_online_evaluation.py:51    FROZEN_SINGLE_SAMPLING_MANIFESTS = frozenset(
nemo_testcase_l1_swg_online_evaluation.py:302   def _admit_day_file(saved, name: str):
nemo_testcase_l1_swg_online_evaluation.py:330   def _require_end_of_day_snapshots(name: str, snapshot, daily_mean,
nemo_testcase_l1_swg_online_evaluation.py:346   if not np.all(np.isfinite(s)):
nemo_testcase_l1_swg_online_evaluation.py:348   for which, mean in (("its own day's", m), ("the next day's", n)):
nemo_testcase_l1_swg_online_evaluation.py:385   _require_end_of_day_snapshots(*previous, daily_mean)
nemo_testcase_l1_swg_online_evaluation.py:395   def _day_file_pins(manifest: dict) -> dict:
nemo_testcase_l1_swg_online_evaluation.py:421   pins = _day_file_pins(manifest)
nemo_testcase_l1_swg_online_evaluation.py:426   if _sha256(manifest_path) not in FROZEN_SINGLE_SAMPLING_MANIFESTS:
nemo_testcase_l1_swg_online_evaluation.py:436   _require_end_of_day_snapshots(
nemo_testcase_l1_swg_online_evaluation.py:504   # Round 7 (2026-10-14): the ``template`` tables were re-measured the same way
nemo_testcase_l1_swg_online_evaluation.py:904   noise_limit[metric] = controls[worse][metric] + bars[metric]["half_range"]
nemo_testcase_l1_swg_online_evaluation.py:981   def available_windows(root: Path = FINE_TRUTH_ROOT) -> list[tuple[int, int]]:
nemo_testcase_l1_swg_online_evaluation.py:1062  rows[-1]["scored_data_sha256"] = hashlib.sha256(
nemo_testcase_l1_swg_online_evaluation.py:1067  def bind_trained_runs(seeds: dict[str, tuple[Path, str]], *, objective: str,
nemo_testcase_l1_swg_online_evaluation.py:1111  pins = _day_file_pins(manifest)  # exactly the files the loader reads
nemo_testcase_l1_swg_online_evaluation.py:1112  if not pins:
nemo_testcase_l1_swg_online_evaluation.py:1151  bindings = bind_trained_runs(seeds, objective=objective,
nemo_testcase_l1_swg_online_evaluation.py:1162  first_seen = {}
nemo_testcase_l1_swg_online_evaluation.py:1221  score.add_argument("--window-days", type=int, required=True,
nemo_testcase_l1_swg_online_reference_adequacy.py:105  axes = tuple(range(1, left_flat.ndim))
test_swg_online_reference_adequacy.py:66   def test_early_rotational_velocity_is_one_vector_rms_per_day():
test_swg_online_evaluation.py:150  def test_noise_allowance_is_the_truth_spread_above_the_worse_control(objective):
test_swg_online_evaluation.py:176  def test_no_threshold_moves_with_a_controls_spread(objective, monkeypatch):
test_swg_online_evaluation.py:360  def test_free_run_scoring_binds_manifest_and_marks_nonfinite_seed(tmp_path):
test_swg_online_evaluation.py:412  def test_trained_score_requires_three_distinct_identified_runs(tmp_path,
test_swg_online_evaluation.py:498  def test_runs_with_identical_scored_data_are_refused(tmp_path, monkeypatch):
test_swg_online_evaluation.py:541  def test_reviewer_renamed_daily_mean_payload_is_refused(tmp_path, schema):
test_swg_online_evaluation.py:562  def test_daily_mean_product_is_never_read_as_snapshots():
test_swg_online_evaluation.py:568  def test_dual_archive_snapshots_are_shown_from_the_data(tmp_path):
test_swg_online_evaluation.py:610  def test_finite_snapshot_beside_a_nonfinite_daily_mean_is_refused(
test_swg_online_evaluation.py:624  def test_fine_truth_snapshots_are_shown_from_the_data():
test_swg_online_evaluation.py:646  def test_truth_operator_arithmetic_is_the_frozen_records(objective):
test_swg_online_evaluation.py:656  def test_fine_truth_renamed_payload_is_refused(tmp_path):
test_swg_online_evaluation.py:680  def test_available_windows_need_the_next_day_pinned(monkeypatch):
```

## Open

- The adequacy tool's own inputs are single-sampling products and the tool
  still takes their sampling from the manifest (outside this round's
  defects).  From the data, both are snapshots: the midpoint test passes on
  every day of both (table above).
- The design note still describes the round-5 gate and none of round 7
  (codex defect 5; not my file).
- The gate truth is now complete (7300 days), so the nine-window re-freeze of
  round-6 decision 1, option 2, is possible; nothing here changes the
  eight-window windows.
- Follow-up: the data test cannot see the NEXT day's daily mean under the
  snapshot names (plant P2 admitted), and no other check in the scorer does.
- Follow-up: data distinctness catches exact copies only; a copy perturbed by
  one unit in the last place passes it, as it passes the byte checks.
- Follow-up: an in-window blow-up now refuses instead of failing health
  (launch prerequisites section).  If a scored FAIL is wanted there, the
  smallest rule found: accept a non-finite NEXT-day mean only when that day's
  snapshot is non-finite too, and fail that window's health.  Not built
  (not asked).

## UNVERIFIED

- No trained closure exists, so no trained run has been scored; the binding,
  the dual-archive requirement and the manifest contract are exercised only on
  synthetic manifests and on archives built from the registered truth.
- The midpoint test is measured on the archives above, not on any trained
  closure.  A closure with strong variability faster than a day could push a
  velocity ratio (at most 0.70 measured, gate truth V day 5097) towards 1 and
  be refused loudly.
- The dual-archive test fixtures pair the registered box-truth snapshots with
  the registered daily-mean product, a restricted daily mean, not the
  driver's own v2 daily mean; the driver's own dual output was measured only
  on four dg50 days.
- The replay evidence covers six days; every other control day is bound by
  the frozen manifest hashes, not replayed.
- The digest pins and the exact `freeze --verify` hold for this platform's
  numpy, JAX and CPU; another machine or library version may need a
  re-measurement (same caveat as the round-6 registered-default pin).
- The 20-year year-pair bars were not computed; the cost is extrapolated from
  the years 2-3 run.
- Follow-up: the launch prerequisites about the training driver are read off
  its uncommitted round-7 code and checked with a file and manifests built to
  its keys, not with files a driver run wrote.
