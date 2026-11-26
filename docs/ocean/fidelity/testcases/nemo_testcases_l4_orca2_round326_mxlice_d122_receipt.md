# ORCA2 round 326 — the rung-10 comparison runs TKE's under-ice mixing length on SI3's recorded ice thickness (DECISION 154)

Date: 2026-10-23 (Claude round, resume 3; codex review: operator).

Status: **LANDED-pending** (operator review + push gate).

## Verdict

- The rung-10 NEMO comparison now runs nn_mxlice = 2 fed NEMO's recorded
  mean ice thickness each step. Decision 122 passes on all seven pairs,
  both labels, top drag on: 0 exact rows lost, first over-bar rows unchanged.
- It was the owner of rung 10's remaining gap under ice: given-entry kt-10
  max T error 0.197 -> 7.7e-5 K, RMS T 1.10e-3 -> 3.5e-7 K.
- The shipped rung-10 card keeps nn_mxlice = 0; selecting 2 without the
  record refuses on every run path (plant shown).
- No model code changed (`git diff 1cdab6a54 -- packages src` empty).

## The record (acquisition `1abfce93d`..`d89a98dd5`; operator `--run`)

- Admission re-run here: `PASS_R326_HMI_ADMISSION`, identical to the
  operator's file except its path key; all nine plants fire
  (`orca2_rounds/round326/admit/`).
- Preregistered A1-A4: CONFIRMED. fr_i and taum 0 unequal points on all 20
  records; even-step hm_i equals the previous odd step's on all 10 rows;
  hm_i max 2.797 m; 37,784 ice-covered wet points carry hm_i > 0.
- NEMO's statements: the CASE 2 surface length
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:621-622`; hm_i assigned only at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:186` through
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:260` on ice steps
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:186`); sbc precedes zdf_phy
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:151`,
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179`). The record is written in
  zdf_tke just before tke_avn at kt, so it is the value TKE used at kt.

## What changed (`d5376302a`)

- The ladder's `comparison_card` also selects the model's existing V1
  statement (TKE `nemo_mxlice = 2`; nothing re-implemented) and refuses a
  card that already selects it. `_surface_forcings` passes the admitted
  hm_i for kt only when the card selects it, and refuses unless the
  record's fr_i equals that step's surface frame bit for bit (a record fed
  at the wrong step refuses; tested with kt 3's fr_i at kt 2).
- Round-326 D122 gate: control = comparison card with nn_mxlice 0
  (round 325's comparison card), mxlice = comparison card; rung 0 /
  OMT-1..4 both arms the shipped card.
- Round 324's two comparison-card tests moved from VORTEX (no physics) to
  GYRE and now expect both selections. Gradient probe stamps nn_mxlice and
  whether hm_i was passed.
- Preregistration amendment 2 (M1-M5) frozen before any arm ran.

## Decision 122, seven pairs (arms at `d5376302a`, `orca2_rounds/round326/arms/`)

| pair | exact rows lost | first over-bar (D139 recomputed) | census toward/away (early, late) |
|---|---:|---|---|
| rung 0, OMT-1..4 | 0 | equal | 0/0 — both arms the shipped card; 0 of 960 rows differ |
| rung 10 independent | 0 | equal (kt 1 entry sea level) | 179/4 (39/4, 140/0) |
| rung 10 given entry | 0 | equal (kt 1 stage-1 v) | 183/0 (43/0, 140/0) |

`decide7_mxlice.json`: `PASS_R278_D122_DECISION`.

- Controls reproduce round 325's drag-on L3 arm: 0 of 1,360 rows differ.
- The four away rows (independent, sea-level RMS; max unchanged):
  kt-2 stage 1 0.01257681 -> 0.01257682 m (+5.5e-7 relative), stage 2
  +8.2e-7, stage 3 +1.6e-6, kt-3 entry +6.2e-9. That label's sea level is
  set by its own kt-1 entry sea level. Not a lean away.
- M2: in both labels the only kt-1 rows that differ are stage-3 T, S, u, v;
  entry, stage-1 and stage-2 rows are bit-identical. CONFIRMED.

## Rung 10, kt-10 errors (|card - NEMO|, `arms/kt10_rung10.json`)

Full domain, both ranks (kt-10 entry = after nine steps; 799,200 T points;
top drag on in both arms):

| label | nn_mxlice | max sea level (m) | max T (K) | RMS T (K) | max u / v (m/s) | max S |
|---|---|---:|---:|---:|---:|---:|
| independent | 0 (before) | 0.02256209 | 0.19673128 | 1.10251e-3 | 0.0472 / 0.0516 | 0.0478 |
| independent | 2 | 0.01853309 | 0.01142969 | 2.10776e-5 | 0.0053 / 0.0031 | 0.0019 |
| given entry | 0 (before) | 0.00706750 | 0.19674038 | 1.10230e-3 | 0.0472 / 0.0516 | 0.0478 |
| given entry | 2 | 3.73498e-5 | 7.70301e-5 | 3.4649e-7 | 0.0030 / 0.0031 | 1.22e-5 |

Kt-10 stage 3 (rank 0 only, NEMO's stage frames): given entry max sea level
0.00208835 -> 2.75771e-5 m, max T 0.183322 -> 1.07927e-4 K, RMS T
7.74385e-4 -> 5.87177e-7 K, max u / v 0.0275 / 0.0525 -> 0.0057 / 0.0037
m/s; independent RMS T 7.74538e-4 -> 1.27272e-5 K.

## Refusal plant (`tests/ocean/fidelity/test_orca2_round326_mxlice_replay.py`)

- GYRE with nn_mxlice = 2 and no hm_i: step (with and without forcing),
  step_checked, integrate (with and without inputs), integrate_scan (with
  and without) all refuse with the model's nn_mxlice refusal: 7 passed.
- Refusal disabled in the TKE anchor: 7 failed (`None is not a valid value
  for jnp.array` instead), `gates/plant_refusal_removed.log`; restored,
  `git status` clean.
- Shipped cards keep 0, arms move only rung 10, recorded hm_i alignment:
  2 passed. With round 324's file: 18 passed (`gates/new_tests.log`).

## Gradients (N = 1, comparison card: drag on, nn_mxlice = 2)

- Probe at clean `d5376302a` (`grad/grad_rung10_n1_mxlice_drag_on.json`):
  stamps drag on, rCdU_ice passed, nn_mxlice = 2, hm_i passed; 2,633 s,
  peak 73.4 GiB (one process; memory rule held).
- T loss w.r.t. initial T: 0 non-finite entries (sea-level loss also 0);
  one T direction reverse vs forward 1.2e-14 relative (sea level 2.3e-15).
- rn_Ud: reverse -4.5118108e-5, forward equal to 7.5e-16, best FD 9.2e-7
  (round 325, nn_mxlice 0: -4.5118105e-5). Sea-level loss: exactly 0 at one
  step, as before.
- Traced and production steps bit-identical (0 leaves); rebuilt-at-1e-4
  model equals the traced step (0 leaves).
- No model function touched, so every round-316 guard is unchanged.

## Predictions (`PREREG_nemo_testcases_l4_orca2_round326.md`, amendment 2)

- M1 D122 PASS, rung 0 / OMT identical: CONFIRMED.
- M2 kt-1 stage-1/2 identical, stage 3 differs: CONFIRMED.
- M3 kt-10 RMS T falls in both labels: CONFIRMED (1.10e-3 -> 2.1e-5 and
  3.5e-7 K).
- M4 gradients: CONFIRMED (0 non-finite; gaps 1.2e-14 and 7.5e-16).
- M5 refusal plant: CONFIRMED.

## Round-314 audit row

- nn_mxlice stays DIFFERS (b:D135) against the shipped card; reason now
  says the rung-10 comparison runs it with the recorded hm_i (D154).
  Counts unchanged (180/3 vs record). Audit test 9 passed.

## Choices made

- nn_mxlice = 2 in the rung-10 comparison harness only, fed the record at
  the step TKE read it — ASKED (DECISION 154, round prompt).
- Selection placed in the ladder's `comparison_card` (so every rung-10
  comparison, the gradient probe included, runs it, as D150's drag) —
  ASKED (prompt: "the rung-10 COMPARISON harness").
- Rung 0 / OMT-1..4 both arms the shipped card (no replayed ice record) —
  ASKED (prompt: rung-10 comparison only).
- Harness refuses unless the record's fr_i equals the frame's bit for bit —
  UNASKED, instrument guard; offered for revert.
- Refusal plant on GYRE, not the ORCA2 card (same TKE refusal, seconds not
  minutes; VORTEX has no physics) — UNASKED, instrument choice; offered for
  revert.
- Round 324's comparison-card tests moved to GYRE — UNASKED, test fixture
  only (forced by the new selection); offered for revert.

## OPEN

- Older round-324/325 D122 gates and the round-324 probe call
  `comparison_card`, so re-running them now includes nn_mxlice = 2.
- Rung 10's next first non-bit rows are unchanged (independent: kt-1 entry
  sea level; given entry: kt-1 stage-1 v signed zeros).

## Gates (x64, CPU; logs `orca2_rounds/round326/gates/`)

- New tests (round 326 replay + round 324 file): 18 passed
  (`new_tests.log`); plant with the refusal removed: 7 failed
  (`plant_refusal_removed.log`); audit test 9 passed (`audit_test_r326b.log`).
- Default gate (land.sh list), split for the 10-minute cap: citation-gate
  test + private-import ratchet + round-326 gate test 46 passed
  (`default_gate4_1.log`); TKE terms + recipe 96 passed (`_2`); freshwater
  + stage face mask + ORCA2 parallel receipt 20 passed (`_3`); prognostic
  barotropic 8 passed (`_4`) = 170 passed, 0 failed.
- Receipt citation gate on this receipt: PASS, 6 citations, 0 failures
  (`citation_gate_receipt_d122.log`).

codex review: operator
