# NEMO testcase L1 SWG online learning — Round 6 evaluation receipt

Scope: the evaluation-side findings of the round-5 Claude review (lens
"evaluation-and-shared-code": EV-01 to EV-10, EV-N1, EV-N2).  Work copy
`claude-swgonl-r6-evaluation`, base `a0b5535b2`, changes uncommitted.  CPU
only, fp64 (`JAX_ENABLE_X64=1`).  Files changed: the held-out evaluation, the
reference-adequacy tool, the 20-year control-health reporter, and their four
test files.  The shared climate instrument's code is untouched; only its test
file gained a pin.

## Round 6 evaluation — verdict

- The held-out gate is rebuilt on the truth's own window-to-window spread and
  frozen in code before any trained model exists.  A perfect model (each
  truth window scored against the climatology of the other seven) now PASSES
  for both objectives; both no-closure controls FAIL.
- The gate scores against the truth made by the selected objective's own
  operator (`symmetric` = box, `template` = filtered); an unknown objective
  raises.
- The adequacy tool and the control-health reporter now select the online
  diagnostics explicitly.  The 500 km jet-shift plant the tip passed with
  exit 0 now fails with exit 1.
- Every new test fails in a scratch copy with its fix reverted (table below).
  The frozen round-5 control numbers are reproduced exactly.
- Two parts of the gate depart from the literal brief because the literal
  version fails the perfect model or leaves EV-N2 open; both are measured
  below and put to the owner.

## Truth data and sampling

| Product (under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3`) | Sampling | Days | Role |
|---|---|---|---|
| `swgdg_rounds/legoesm_runs/dg_gpu_7300d_truth`, 6.25 km, 20 y | v2 dual archive: labelled end-of-day snapshots AND daily means in every file; manifest says `daily_mean`; no mean transports | 6661 of 7300 pinned at the freeze, still running | gate truth: snapshot fields only, restricted on the fly |
| `swgonl_rounds/round2/conservative_truth_dg50_1095d/snapshots`, 50 km | snapshot | 1095 | round-5 record, adequacy candidate, real-field test fixtures |
| `.../conservative_truth_dg50_1095d/daily_means`, 50 km | product-of-means daily mean | 1095 | none; the gate's reader refuses it |
| `swgonl_rounds/round1/dg50_cpu_7300d_no_closure` (nu=10) | v1 snapshot | 7300 | gate control |
| `swgonl_rounds/round2/dg50_nu80_cpu_7300d` (nu=80) | v1 snapshot | 7300 | gate control |

- The fine truth's day files carry both samplings.  The gate reads only the
  labelled snapshot fields (`nemo_testcase_l1_swg_online_evaluation.py:269`),
  because both controls archive daily snapshots.  Box restriction of fine day
  400 equals the stored three-year coarse truth bit for bit.
- No 20-year coarse product exists and none is needed: the gate restricts the
  pinned fine snapshots itself with the shared restriction functions
  (`:286`).  The coarse-graining CLI could not build one anyway; it refuses a
  fine run without archived mean transports, and this run has none.
- Mixing check.  The round-5 record and the new gate read snapshots on both
  sides.  The climate CLI's comparison already refused mixed sampling.  The
  adequacy tool had no check; it now refuses a candidate/sensitivity mismatch
  (`nemo_testcase_l1_swg_online_reference_adequacy.py:201`).  The
  control-health report now records the sampling it read
  (`nemo_testcase_l1_swg_online_control_health.py:83`).  No reported number
  was found to have mixed them.

## The frozen gate

Windows: the eight non-overlapping two-year windows after the spin-up year
that the fine truth pins, days 366-6205.  Every candidate (network seed or
control) is a free run from rest, scored on those same windows against the
eight-window truth climatology of the selected operator, and summarised by
its median over windows.  The record (windows, input pins, the truth's
leave-one-out table, both controls' tables, derived bars) is
`nemo_testcase_l1_swg_online_evaluation.py:387`.  The bars are derived by
`:710` and applied by `:740`.

- Skill (transport RMSE, sine-transform KE-spectrum L1, wet EKE-map NRMSE):
  the median seed must beat BOTH controls by more than the truth's
  half-range (`:777`).
- Variability: every seed's EKE ratio inside the truth's own leave-one-out
  band, smallest to largest of the eight (`:784`).
- Spectral health: no seed noisier than the worse control (nu=10 on all three
  measures) by more than that control's own half-range (`:782`).
- Physical health: unconditional on every window and on the driver's
  whole-run record (`:922`).

Frozen thresholds, medians over the eight windows (printed from the gate
itself):

| Measure | symmetric (box truth): perfect / nu10 / nu80 / threshold | template (filtered truth): perfect / nu10 / nu80 / threshold |
|---|---|---|
| transport RMSE, m3/s | 5.494e+05 / 1.093e+06 / 1.444e+06 / below 7.961e+05 | 4.602e+05 / 1.362e+06 / 1.68e+06 / below 1.103e+06 |
| KE spectrum L1 | 0.04009 / 0.1945 / 0.3214 / below 0.1819 | 0.04037 / 0.4496 / 0.5421 / below 0.4354 |
| wet EKE-map NRMSE | 0.2326 / 0.3085 / 0.6733 / below 0.2733 | 0.2291 / 0.4261 / 0.6014 / below 0.3828 |
| EKE ratio | 1.004 / 0.7483 / 0.4525 / in [0.9311, 1.0262] | 1.002 / 1.074 / 0.6494 / in [0.9265, 1.0378] |
| native grid-noise index | 0.4839 / 0.9306 / 0.7053 / at most 0.9367 | 0.2112 / 0.9306 / 0.7053 / at most 0.9367 |
| top-octave KE fraction | 0.008651 / 0.05366 / 0.02738 / at most 0.05445 | 3.334e-05 / 0.05366 / 0.02738 / at most 0.05445 |
| top-octave enstrophy fraction | 0.1245 / 0.4327 / 0.294 / at most 0.4379 | 0.0006996 / 0.4327 / 0.294 / at most 0.4379 |

Under `template`, nu10's eddy energy is 1.07 of the filtered truth's, so it
fails the band from above; under `symmetric` it is 0.75 of the box truth's.

Perfect model: PASS on skill, variability and noise for both objectives.
nu10 and nu80: FAIL skill and variability for both.

Two departures from the literal brief, both measured, both for the owner:

- The skill margin is the truth's half-range, not its full range and not the
  bar.  With the full range the box-truth perfect model fails transport
  (5.494e5 against a required 4.994e5).  With the bar ("beat by more than the
  high quantile") it fails transport and EKE-map under `symmetric` and
  EKE-map under `template`.  The half-range is this file's existing
  `uncertainty_half_range` convention.
- The noise tolerance is the worse control's own half-range, not the
  truth's.  The filtered truth's top-octave spreads (1.8e-6 for KE, 4.5e-5
  for enstrophy) are 0.007 and 0.03 of the sampling noise of nu10's
  eight-window median (2.65e-4 and 1.39e-3), so a closure exactly at nu10's
  noise would fail half the time and EV-N2 would stay open.  nu10's own
  half-ranges are 3.0 and 3.7 times that noise.

## Disposition

Reversions were made in scratch copies only (one tree per reversion, the
work copy never touched): trees under
`/home/dbalwada/.claude/jobs/b3690d5d/tmp/claude_r6/evaluation/impl/mut/`,
exact text replacements in mutations.py there, full output in
mutations.log there.  The four unmutated trees give `21 passed`,
`8 passed`, `2 passed`, `14 passed`.

| Review finding | Disposition | Evidence (command, decisive output) | Reversion that makes its test fail |
|---|---|---|---|
| EV-01 (blocker): single-realisation bars; a perfect model fails, the control passes | Fixed: truth-spread gate frozen in code (`nemo_testcase_l1_swg_online_evaluation.py:387`, derived by `:710`, applied by `:740`) | `freeze --objective symmetric --verify` and `--objective template --verify`: `PASS freeze`, exit 0, every frozen key equal, `truth_passes: true`, both controls and both plants fail.  Tests `test_swg_online_evaluation.py:41`, `:54`, `:73`, `:91`, `:173` | `ev_bar_is_median`: `:41` fails (stored transport bar 935200.83, derived 549386.67).  `ev_margin_is_bar` and `ev_margin_full_range`: `:54` fails, the perfect model is rejected (full range: `symmetric` only).  `ev_no_skill_margin`: `:73` `assert True is False`.  `ev_band_fixed_20pct`: `:91` `assert True is False`.  `ev_loo_includes_self`: `:173` transport 1.72 against 2.29 |
| Gate binding and the mandatory plants (design) | Fixed: three seeds, one score per frozen window, unconditional health; collapsed variance and the face checkerboard go through the real scorer | `test_swg_online_evaluation.py:126`, `:249` | `ev_window_count_unchecked`: `:126` `DID NOT RAISE`.  `ev_gate_health_ignored`: `:126` `KeyError` on the blown-up window.  `ev_variability_ignored`: `:249` `assert (not False and not True)` |
| EV-02: the adequacy tool ran the registered diagnostics on dg50 fields | Fixed: explicit online selection (`nemo_testcase_l1_swg_online_reference_adequacy.py:194`); the spectrum drops only a zero-wavenumber bin (`:51`); mixed sampling is refused (`:201`) | Tests `test_swg_online_reference_adequacy.py:119`, `:157`, `:102`, `:140`: `8 passed in 21.09s`.  Real 500 km plant: jet 0.12426 against bar 0.10, `adequate False`, exit 1 | `ad_registered_default`: `:119` `KeyError: 'diagnostics'`, and `:157` `assert 0.015531949493805503 > 0.1`, i.e. the tip's "adequate, exit 0" verdict reproduced.  `ad_drop_first_bin`: `:102` `assert 0.0 == (3.0 / 9.0)`.  `ad_no_sampling_guard`: `:140` fails, nothing refuses |
| EV-03: the gate always scored against box truth | Fixed: the objective selects both the truth operator and the frozen record (`nemo_testcase_l1_swg_online_evaluation.py:219`); unknown raises | `test_swg_online_evaluation.py:141`; the `template` record verifies and its filtered-truth perfect model passes | `ev_objective_ignored`: `:141` box operator returned for `template`.  `ev_unknown_objective_accepted`: `:141` `DID NOT RAISE` |
| EV-04: the control-health reporter refused | Fixed (`nemo_testcase_l1_swg_online_control_health.py:75`); the report records its sampling (`:83`) | `test_swg_online_control_health.py:25`: `2 passed in 8.84s` | `he_registered_default`: `:25` `assert 2 == 0`, the refusal exit |
| EV-05: scorer tests could not fail for what they claimed | Fixed: plants on real dg50 truth fields through the real scorer; the white-noise tests are gone | `test_swg_online_evaluation.py:218`: clean noise index 0.48, under the 0.9367 limit; a 0.1 m/s divergent face checkerboard goes above it, a rotational one trips top-octave enstrophy.  `:262`: thickness -1.0 m and Courant 2.0 computed by the scorer itself | `cl_noise_cell_centred`: `:218` `assert 0.32586613018360633 > 0.9366697173668606`.  `cl_spectra_cell_centred`: `:218` `assert 0.02244922090280279 > 0.4378716667204242`.  `ev_health_blind`: `:262` `assert not True` |
| EV-06: the default-output pin was blind to dry cells, masks and v | Fixed in the test only; registered code untouched.  New case: land faces carrying garbage, nonzero v, non-uniform area; the branch-point file (8f8551c87) and this file give the same JSON | `test_nemo_swg_dg_climate.py:203`: `14 passed in 9.69s`; `branch-point summary == current registered summary: True` | `cl_registered_dry_nan`, `cl_registered_v_doubled`, `cl_cell_centre_ignores_masks`, `cl_registered_jet_u_plus_v`: `:203` fails for each (differing `mean_u_m_s`, `eke_map_m2_s2`, `jet_eastward_speed_m_s`, `jet_y_m`) |
| EV-07: the absent-closure stage test cannot fail | Deferred: that test lives in the unit tests outside this piece.  Suggested fix: compare the closure-free stage with a stored pre-seam output on a nonzero entry, plus a structural check that the absent path adds nothing | none | none |
| EV-08: adequacy year-1 bars include the spin-up change | Deferred: owner decision 2 | With the fix, measured bars: KE 0.196, EKE 0.186, spectrum 0.198, jet 0.10 | none |
| EV-09: no SSH term; the spectra see only rotational flow | No term added, by reasoning, not a test.  The template is rigid-lid (paper line 179) and scores the mean transport streamfunction, mean and eddy energy, EKE maps and energy spectra.  The closure forcing's volume transport is divergence-free to roundoff (design note), so it cannot change volume or push SSH directly.  An SSH grid mode needs divergent velocity noise, which the native index catches above about 0.03-0.05 m/s (reviewer p16; limit now 0.9367, was 0.9318).  The rotational-only spectra remain a characterisation | none | none |
| EV-10: no path scored a trained closure | Fixed: `score` binds each seed (label, root, manifest SHA-256), the objective's frozen windows and the pinned truth, recomputes truth and controls and refuses drift (`nemo_testcase_l1_swg_online_evaluation.py:941`).  A non-finite window or a failed whole-run health record fails the seed instead of crashing it (`:910`) | `test_swg_online_evaluation.py:286` | `ev_nonfinite_raises`: `:286` `ValueError: native trajectory contains non-finite fields`.  `ev_whole_run_ignored`: `:286` `assert not True`.  `ev_unbound_manifest`: `:286` `DID NOT RAISE` |
| EV-N1: no skill metric shown to move | Fixed: analytic answers on real fields (`nemo_testcase_l1_swg_online_evaluation.py:145`).  Velocities times 2 and 3 give spectrum and EKE-map errors 3 and 8, EKE ratios 4 and 9, and transport errors in ratio 2 | `test_swg_online_evaluation.py:190` | `ev_transport_zero`: `:190` `assert 0.0 > 0.0`.  `ev_spectrum_zero`: `:190` `0.0 == 3.0 ± 3.0e-12` |
| EV-N2: the noise bars were the control's own mean | Fixed: tolerance = the worse control's own half-range | `test_swg_online_evaluation.py:103`: nu10 itself keeps spectral health; noise half its spread above passes, 1.5 times fails | `ev_no_noise_tolerance`: `:103` `assert False is True`.  `ev_noise_tolerance_truth_spread`: `:103` fails for `template` only, the truth's spread being too small there |
| Snapshots and daily means never mixed | Guarded (`nemo_testcase_l1_swg_online_evaluation.py:269`) | `test_swg_online_evaluation.py:332` | `ev_reads_daily_means`: `:332` fails; the daily-mean product is then refused only by the accident of missing fields, and a mislabelled dual file would be read |
| Item 6: frozen round-5 control numbers | Reproduced exactly (`nemo_testcase_l1_swg_online_evaluation.py:329`): the whole controls JSON is byte-equal for nu10 and nu80, including 0.7323411189244107, 0.22300407134605818 and 0.9317922221566435 | `test_swg_online_evaluation.py:350`; `nu10 full controls JSON equal to frozen round-5 record: True`, nu80 likewise | `ev_spectrum_zero`: `:350` `assert 0.0 == 0.22300407134605818` |

## Acquisitions

Item 1 was run here, writing to the scratch directory instead of `round6`
(see Validation); items 2 to 4 were not.  Run from the work copy root with
`PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`
and the repository venv's `python`.

1. Re-verify the frozen gate from the pinned data, about 17 min CPU each.
   Exit 0 only on exact equality and all self-checks (perfect model passes,
   both controls and both plants fail):

```bash
python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py freeze --objective symmetric --verify --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round6/gate_verify_symmetric.json
python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py freeze --objective template --verify --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round6/gate_verify_template.json
```

2. Only if the owner chooses a nine-window freeze (decision 1, option 2):
   after the fine truth reaches 6935 days, re-measure and paste each output's
   `record` (keys `windows`, `fine_truth_pins_sha256`,
   `control_manifest_sha256`, `truth_loo`, `controls`, `bars`) into
   `FROZEN_GATE`, before any trained seed is scored:

```bash
python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py freeze --objective symmetric --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round6/gate_freeze9_symmetric.json
python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py freeze --objective template --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round6/gate_freeze9_template.json
```

3. Before any trained closure is scored: at least three 20-year free runs of
   the trained closure on the dg50 deck from rest (at least 6205 days),
   writing daily snapshots (v1, or v2 with labelled snapshots) and the
   driver's whole-run `numerical_health`, each bound by its manifest SHA-256.
   No driver can run a trained closure free yet (the driver is outside this
   piece), so no command for the runs themselves is given.  Scoring them:

```bash
python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py score --objective <selected objective> --seed s1 <root1> <manifest sha256 1> --seed s2 <root2> <manifest sha256 2> --seed s3 <root3> <manifest sha256 3> --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round6/gate_trained.json
```

4. The 3.125 km adequacy command is unchanged from the round-5 receipt and
   now runs the online diagnostics:

```bash
python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_reference_adequacy.py --candidate-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/conservative_truth_dg50_1095d/snapshots --sensitivity-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/reference/dg3125_conservative_dg50_365d/snapshots --years 1 --json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round6/reference_adequacy.json
```

## Decisions for the owner

1. Gate rule (EV-01).  Frozen: two-year windows, bar = largest of eight
   leave-one-out values (8/9 coverage for a new window), skill margin = truth
   half-range, EKE band = the truth's eight-window range, noise tolerance =
   worse control's half-range.  The truth climatology includes the training
   years, as Maddison's does (paper lines 765-768: valid "since long time
   averages under a statistically steady assumption are computed").
   (1) accept as frozen (my pick); (2) re-freeze with nine windows when the
   truth completes, before training (acquisition 2); (3) another margin.
2. Adequacy year-1 bars (EV-08) include the spin-up year-1-to-2 change, so
   they sit near 0.19-0.20 instead of 0.10.  (1) keep; (2) drop the spin-up
   pair (my pick: the rule then measures variability, not spin-up);
   (3) use adjacent years 2-18 of the 20-year truth.
3. The design note's "Controls and held-out gates" section still describes
   the round-5 gate (frozen nu=10 numbers, "within 20%").  I do not own that
   file.  Replacement text is in the last section here.

## Validation

All on CPU, fp64, from the work copy root with the environment given under
Acquisitions; pytest with `-q -p no:cacheprovider`, one file at a time.

| Run | Decisive line |
|---|---|
| pytest tests/ocean/fidelity/test_swg_online_evaluation.py | `21 passed in 26.40s` |
| pytest tests/ocean/fidelity/test_swg_online_reference_adequacy.py | `8 passed in 21.09s` |
| pytest tests/ocean/fidelity/test_swg_online_control_health.py | `2 passed in 8.84s` |
| pytest tests/ocean/fidelity/test_nemo_swg_dg_climate.py (registered-path pin file) | `14 passed in 9.69s` |
| freeze --objective symmetric --verify | `PASS freeze`, exit 0, 13 min; all six frozen keys equal; truth passes; nu10, nu80, both plants fail |
| freeze --objective template --verify | `PASS freeze`, exit 0, 14 min; same |
| round-5 record through the refactored scorer | `nu10 full controls JSON equal to frozen round-5 record: True`, nu80 the same |
| ruff check on the seven changed Python files | `All checks passed!` |
| pytest tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py | `2 failed, 15 passed`: the gate refuses to stamp an uncommitted tree (`refusing to stamp a0b5535b2e97: 7 tracked file(s) modified`); with the documented escape `LEGOESM_GATE_ALLOW_DIRTY=1`, `17 passed in 3.60s`.  Expected to pass unescaped once committed |
| citation gate on this receipt, heading "## Round 6 evaluation", shipped map only | `status FAIL, citations 40, unmapped 40`: the map has no entries for this lane's Python files |
| same, with the 40 entries below added in memory (gate script not edited) | `"status": "PASS"`, 0 unmapped, 0 failures, 0 map entries failing audit; a two-line planted shift of the first entry gives `FAIL ['SYMBOL-NOT-AT-LINE']` |

Artifacts (scratch, for the operator to move under `round6`):
`/home/dbalwada/.claude/jobs/b3690d5d/tmp/claude_r6/evaluation/impl/`
holds spread_measurement.json (the measured tables the literal was built
from), verify_symmetric.json and verify_template.json, gate_summary.log,
mutations.log and the mutation trees.

Entries for the gate's owner to add.  FILES keys map to
`scripts/validate/ocean_fidelity/testcases/<name>` for the three scripts and
to `tests/ocean/fidelity/<name>` for the four test files; each value below is
the one-line anchor of its cited line:

```text
nemo_testcase_l1_swg_online_evaluation.py:145  def score_statistics(
nemo_testcase_l1_swg_online_evaluation.py:219  def truth_restrictor(
nemo_testcase_l1_swg_online_evaluation.py:269  def _snapshot_fields(
nemo_testcase_l1_swg_online_evaluation.py:286  def restrict_fine_truth_window(
nemo_testcase_l1_swg_online_evaluation.py:329  def score_control(
nemo_testcase_l1_swg_online_evaluation.py:387  FROZEN_GATE: dict[str, dict] = {
nemo_testcase_l1_swg_online_evaluation.py:710  def gate_bars(
nemo_testcase_l1_swg_online_evaluation.py:740  def held_out_gate(
nemo_testcase_l1_swg_online_evaluation.py:777  required = {metric: min(row[metric]
nemo_testcase_l1_swg_online_evaluation.py:782  noise_limit[metric] = controls[worse][metric]
nemo_testcase_l1_swg_online_evaluation.py:784  band = (bars[VARIABILITY_METRIC]["floor"]
nemo_testcase_l1_swg_online_evaluation.py:910  def score_free_run(
nemo_testcase_l1_swg_online_evaluation.py:922  whole_run_ok = (
nemo_testcase_l1_swg_online_evaluation.py:941  def score_trained_closure(
nemo_testcase_l1_swg_online_reference_adequacy.py:51   resolved = wavenumber > 0.0
nemo_testcase_l1_swg_online_reference_adequacy.py:194  diagnostics="online"))
nemo_testcase_l1_swg_online_reference_adequacy.py:201  sampling mismatch: candidate
nemo_testcase_l1_swg_online_control_health.py:75   diagnostics="online")
nemo_testcase_l1_swg_online_control_health.py:83   "sampling": provider.sampling
test_*.py:<line>  def test_<the cited test's name>(   (21 test citations)
```

## Choices made

| Choice | Asked? |
|---|---|
| Two-year windows from day 366, every complete one the truth pins (8) | ASKED (brief) |
| Bar = largest of the eight leave-one-out values | ASKED (brief: "a stated high quantile") |
| Skill margin = truth half-range, not full range or bar | UNASKED; measured; decision 1 |
| Noise tolerance = worse control's half-range, not the truth's | UNASKED; measured; decision 1 |
| Candidates summarised by the median over windows | UNASKED; decision 1 |
| EKE band = truth leave-one-out range, replacing the fixed 20% | ASKED (brief: every metric from the truth's spread) |
| Truth climatology includes training-split years | ASKED (brief: every available window); decision 1 |
| Seeds also gated on the driver's whole-run health record | UNASKED; implements the design's "remain finite ... 20-year free run"; offered for revert |
| Adequacy spectrum drops only a zero-wavenumber bin | UNASKED; defect fix: the old `[1:]` dropped the gravest sine mode once the online spectrum was selected; offered for revert |
| Adequacy refuses mixed candidate/sensitivity sampling | ASKED (brief: never mix) |

## Design note replacement text

For the design note's "Controls and held-out gates" section, replacing its
second and third paragraphs (owner of that file to apply):

> Both no-closure controls and every trained seed are 20-year free runs from
> rest, scored by one harness on the eight non-overlapping two-year windows
> after year 1 that the 6.25 km truth pins (days 366-6205), against the truth
> climatology made by the selected objective's own operator (`symmetric`:
> box restriction; `template`: filtered restriction).  Every number is a
> median over those windows.  The bars are the truth's own window-to-window
> spread, frozen in code before any trained model exists: each truth window
> is scored against the climatology of the other seven, and the bar is the
> largest of the eight values.  The median seed must beat BOTH controls by
> more than the truth's half-range in live-face-transport RMSE, KE-spectrum
> L1 and wet EKE-map NRMSE; every seed's EKE ratio must lie in the truth's own
> eight-window band; no seed may exceed the worse control's top-octave
> fractions or grid-noise index by more than that control's own half-range.
> Finite fields, positive thickness and sub-unit Courant numbers are
> unconditional on every window and on the whole run.  Measured: the perfect
> model passes and both controls fail, for both objectives; `freeze --verify`
> recomputes the record from the pinned data and refuses any difference.

## UNVERIFIED

- No trained closure has been scored: no driver can run one free yet.  The
  scoring entry point has run only on the frozen tables and on a synthetic
  six-day bound root.
- The `template` gate truth is the offline filtered restriction of daily
  snapshots.  That it is the same operator as the in-scan sampler's filtered
  targets rests on the shared implementation; the dual-operator hourly
  segments that would let me compare them are still pending.
- The fine truth was still running (pending manifest).  The freeze binds the
  5840 day files it read through `fine_truth_pins_sha256`; if any is
  rewritten, `freeze --verify` and `score` refuse.
- The 8/9 coverage statement assumes the eight two-year windows are
  exchangeable (no drift after year 1).  The box-truth leave-one-out EKE
  ratios (0.931 to 1.026) show no monotone trend; this was not tested
  formally.
- The registered-default pin (EV-06) depends on this platform's numpy/JAX FFT
  bits, exactly like the pin that already existed.
- The citation gate's map has no entries for this lane's Python files; the
  receipt passes only with the in-memory entries given in Validation.
- The control-health test packages the stored truth's year 1 as a one-year
  dg50 archive.  It proves the reporter runs the online path; it says nothing
  about any real control.
- The "about 17 min per objective" timings were measured once on this CPU.
