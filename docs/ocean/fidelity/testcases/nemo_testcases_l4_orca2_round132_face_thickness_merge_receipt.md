# ORCA2 round 132 — shared reference-face thickness merge

Date: 2026-10-03. Base: `797a939b7`. Incoming GYRE/VORTEX tip:
`9cd35a16aa8376d8ba2788f0a735746a29fb4420`. Merge commit: `7dadad262b`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round132`.

## Verdict

**LANDED.** The incoming reference-U/V-face-thickness statement passes the
ORCA2, GYRE, VORTEX, tank, and DINO predicates on the combined tree. Both
ORCA2 ten-step ladders move 185/200 rows with no exact-row loss and with most
RMS rows toward NEMO. The rung-0 independent month remains unstable, but its
first non-finite boundary moves from step 16 to step 36. Round 129's complete
barotropic association arm remains held: its 31.36 salinity exposure survives
the corrected face thickness.

Every ORCA2 ladder number below is **given NEMO's recorded entry**. The month
result is **independent**, from legoESM's own rung-0 initial state, compared
with NEMO's own rung-0 from-rest run. The two classes are not mixed.

## Merge and citation-map union

`git merge --no-ff --no-commit 9cd35a16a` completed with **zero conflicted
hunks**. Therefore there was no manual conflict resolution to review: the
index was the automatic semantic union of both parents. Both parent SHAs are
ancestors of `7dadad262b`. The default citation gate passes 274 citations
with zero failures, zero unmapped citations, and zero audit failures before
this receipt was written (`citation_default_pre.{json,log}`).

The merged statement is source exact. The record's compiled seamount branch
constructs U and V reference thickness as the minimum of the neighbouring T
thicknesses and then exchanges the paired fields at
`VORTEX_SMT_R3_OMIP_L1_P3/BLD/ppsrc/nemo/usrdef_zgr.f90:240-247`. The shared
consumer now takes the already-attached NEMO-built face arrays and fails
closed if they are absent at `vertical.py:673-735`. The preregistration used
the unpreprocessed `MY_SRC` line numbers 225/228/231; those are **not** the
compiled line numbers, so this receipt corrects the citation rather than
silently carrying it forward.

Sea ice is unchanged. No selector, configuration field, stabilizer, carried
state, threshold, or `unmeasured_features` entry changed in this round.

## Frozen predictions: results

| ID | Result | Measurement |
|---|---|---|
| R132-P1 | **CONFIRMED** | both parents are ancestors; zero conflict hunks; clean default citation audit |
| R132-P2 | **CONFIRMED** | GYRE ladder 0 moved rows; year and three certified snapshot digests reproduce exactly |
| R132-P3 | **CONFIRMED** | rung 0 and rung 7 each move 185/200 rows, lose 0 exact rows, retain the same first non-bit statement, and have a majority of RMS rows toward NEMO |
| R132-P4 | **REFUTED** | the held arm's rung-0 kt=10 stage-3 S maximum is 31.3630667330, within 1% of 31.3617, not reduced 2x |
| R132-P5 | **CONFIRMED** | independent month first becomes non-finite at step 36, later than step 16 |
| R132-P6 | **CONFIRMED** | all ten VORTEX/tank comparisons, GYRE, and DINO pass their standing predicates |

The refuted prediction is retained. It changes the next walk, not the landed
NEMO statement.

## ORCA2 ladders — given NEMO's recorded entry

The first comparison attempt used round 129's file named
`rung0_ladder.json`; that artifact was the subsequently reverted held arm, not
the production base tree. Its 195/200 count is **RETRACTED as a confounded
comparison**. The controlled comparison reran both ladders from the untouched
base `797a939b7` and from the merge, then used the committed round-111
comparator (`orca_ladder_compare_controlled.{json,log}`).

| ladder | moved / total | exact-row losses | RMS toward / away | max toward / away / equal | first non-bit |
|---|---:|---:|---:|---:|---|
| rung 0 | 185 / 200 | 0 | 151 / 34 | 111 / 73 / 1 | kt=1 stage-1 T; held completed-RHS depth average unchanged |
| rung 7 | 185 / 200 | 0 | 154 / 31 | 131 / 49 / 5 | kt=1 stage-1 T; runoff remains ruled out |

Selected kt=10 stage-3 RMS rows, base to merge:

| ladder | T | S | u | v | ssh |
|---|---:|---:|---:|---:|---:|
| rung 0 | 6.3174932e-3 -> 5.9630215e-3 | 1.6155414e-3 -> 1.6088974e-3 | 4.5188498e-3 -> 4.3430736e-3 | 5.0426701e-3 -> 4.3565552e-3 | 2.8059431e-2 -> 2.8026458e-2 |
| rung 7 | 1.0203382e-2 -> 9.4370585e-3 | 2.8051322e-3 -> 2.7885154e-3 | 5.7499179e-3 -> 5.3928781e-3 | 5.7695713e-3 -> 5.3353296e-3 | 2.7582991e-2 -> 2.7594271e-2 |

The full per-row registry is the comparator JSON; no moved row is omitted.

## Held barotropic arm — given NEMO's recorded entry

The exact round-129 held patch was replayed in a disposable clone on top of
the merge (`/tmp/autopilot-orca2-r132-held-arm`, temporary commit
`3380d6f14`). At rung-0 kt=10 stage 3:

| state | S RMS | S maximum |
|---|---:|---:|
| base production | 1.6155414073e-3 | 0.4156325968 |
| merged production | 1.6088973699e-3 | 0.4156677936 |
| merge plus held arm | 1.3181922976e-1 | **31.3630667330** |

The corrected face operand does not remove the tracer compensation exposed by
that multi-statement arm. It remains held; nothing from it is in the landed
tree.

## Rung-0 month — independent

The committed round-130 month gate was run twice against the admitted
round-83 NEMO from-rest record. Both executions first refuse at **step 36**,
field T, zero-based index `[j=86, i=159, k=0]`, with NaN
(`rung0_month{,_repeat}.log`). The pre-merge boundary was step 16, T at
`[j=1, i=49, k=0]`. The merge therefore advances the run by 20 steps but does
not make the 240-step month finite. No month field score is claimed.

## Shared-card gates

GYRE's 70-row trajectory-only ladder is array-identical to the compatible
pre-merge report: 0 moved rows, zero ULP worsening, first-over-bar unchanged
at kt=3 for T/S/u/v/ssh (`gyre_ladder_compare_controlled.{json,log}`). Its
360-day run reproduces the certified values and bytes:

| day | wet 3-D T RMS versus NEMO | snapshot SHA-256 |
|---:|---:|---|
| 30 | 2.3432465132112266e-06 K | `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba` |
| 240 | 6.5817060949447300e-05 K | `8b9cd60475626373d9a2fec0baa508f06c1f91aed4c3d7a24fc5d8b3f5878c8a` |
| 360 | 5.4077419367442036e-05 K | `e3e0a068346c7866f0a318bb32141f2fb326cc05c158a1c95bc39b091f585e25` |

All ten generic-card comparisons report `ORACLE_RELATIVE_COMPARE PASS`, 50
rows each, maximum worsening 0 ULP: LOCK_EXCHANGE, OVERFLOW, the six flat
VORTEX/vector resolution cards, and both seamount cards (`cards/`). Existing
debt remains debt; these results claim no new worsening, not bit identity.

The DINO gate's planted violation fires. The real CPU month result is recorded
in `dino_month.log` and the test outcome below.

## Tests, review, and citation controls

The corrected real-process census printed 0 before each battery.

| control | result |
|---|---|
| focused ORCA2 push battery | **138 passed in 1033.79 s** |
| DINO planted violation | **FIRES** |
| DINO real CPU month | **PASS**, T3D RMS 2.053801168e-3 K against bar 2.244317642e-3 K |
| seamount geometry | **GEOMETRY IDENTICAL**, 17/17 rows on both cards |
| default citation gate | **PASS**, 274 citations, zero failures/unmapped/audit failures |
| this receipt's citation gate | **PASS**, zero failures and zero unmapped citations |
| rigid +2 compiled-source citation plant | **FIRES** with `SYMBOL-NOT-AT-LINE` |

The single required `tests/ocean/fidelity -n 12` battery reached 99% and then
entered the campaign's known silent xdist tail, so it is **incomplete, not
PASS**. A named rerun exposed seven markers. Six are the established red set
already recorded by ORCA2 rounds 97-98:

- `test_nemo_testcase_l2_gyre_round129_spread_floor_gate.py::test_record_backed_gate_passes`;
- `test_nemo_testcase_l2_gyre_round51_live_operands.py::test_live_trace_and_raw_history_arms_are_private_and_off_by_default`;
- `test_nemo_testcase_round35_stamp_scope.py::test_every_driver_that_arms_the_escape_scopes_it`;
- `test_recipe_case_board.py::test_every_oracle_comparison_has_a_row`;
- `test_nemo_testcase_worktree_stamp.py::test_every_report_emitter_stamps_the_worktree`;
- `test_nemo_si3_scalarmath_v2_gate.py::test_full_v2_gate_and_plants`.

The seventh was an incoming stale unit expectation, not a physics failure:
rounds 204/205 added `base_noadv.{u,v}` and `advtrend.{u,v}` to the committed
stage-1 walker, while round 200's tuple test still named the older 12 entries.
The test-only repair adds those four entries in their actual source order; the
isolated test passes (`isolated_vortex_round200_fixed.log`). No production
code changes in that repair.

The required separate `codex exec --sandbox read-only` review did not reach
the diff: `failed to initialize in-process app-server client: Read-only file
system`. Verdict: **independent review unavailable in-sandbox**
(`codex_review.log`).

## OPEN

1. **Rung-0 independent month:** walk the new first non-finite boundary at
   step 36, surface T `[86,159,0]`, in stage order. The merge improved the
   horizon but did not close it.
2. **Tracer compensation:** the round-129 complete barotropic association arm
   still exposes a 31.363 salinity maximum and remains held. Do not revive the
   arm as a unit; resume with one source-ordered statement after the month
   boundary is localized.
3. **Hierarchy:** terminal rung-0 month metrics remain unmeasured while the
   run is non-finite. After rung 0 is closed, merge the parked hierarchy-decks
   branch and climb to rung 1 (+ interior T/S damping).
4. The rung-7 first non-bit statement remains unattributed beyond runoff being
   ruled out. No rung-7 physics lands until the hierarchy reaches it.
