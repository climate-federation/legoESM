# Round 173 receipt — wet-only solve-input acquisition retry

**Status: STOPPED_FOR_RECORD.**  No production physics, configuration,
carried state, restart schema, card default, or immutable before arm changed.
The Round-172 e3t acquisition failed because its converter replaced all 3,120
dry active-level matrix weights with zero even though NEMO's values are
positive.  The e3t arm consequently produced NaNs and stopped at step 1082;
the content arm never ran.  Neither failed arm is scored.

A corrected operator-run retry is committed.  It reuses the unchanged
Round-172 binary and the admitted baseline, constructs every frame from
NEMO's own recorded operands, and replaces only the 18,000 wet active cells.
Its 360-frame preflight passes; its wet-cell plant changes all 18,000 selected
values, preserves all 3,120 dry values, prints `STATUS PLANT-FIRED`, and exits
`2`.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round173/`.  The frozen
preregistration is commit `2edcb9df9`; the final acquisition handoff is based
on commit `31afcb72c` plus this receipt.

The immutable production headline remains Round 163: first-over-bar kt3,
day-30 T RMS `6.572574374770603e-05` K, day-240
`1.644836070117868e-02` K, and day-360 `1.122566001855131e-02` K.  No accepted
candidate ran, so the ladder, month, year, DINO, generic GYRE, tanks, and ORCA2
rows have zero registered movement.

## Compiled statements

The failed build opens one two-field frame for steps 1081--1440 at
`GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:136-150`.
In the selected adaptive-implicit branch, NEMO's ordinary diagonal retains
reference thickness on dry cells through
`e3t_3d * (1 + r3t(Kaa) * tmask)`, while the failed arm selected the imported
array directly at
`GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:490-504`.
The following LU recurrence divides by the preceding diagonal at
`GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:561-565`.

The original magnitude comparison remains the Round-125 compiled pair:
matrix formation at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:468-474`,
content formation at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-567`, and solve
consumption at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:577-582`.
The retry changes only the wet cells supplied to those already-compiled
Round-172 selector sites; no NEMO source or executable changes.

## Failure discrimination

The Round-172 baseline reached step 1440 with `STOP 0`.  Its step-1080 and
step-1440 restarts are byte-identical to the Round-125 producing run.  The
failed e3t arm uses an executable byte-identical to both the baseline copy and
the installed Round-172 binary, so the failure is not a binary or source-card
drift.

The committed retry preflight reads the rejected raw payload and the admitted
Round-125 step-1081 record.  Its closed census is:

| item | count | verdict |
|---|---:|---|
| wet active cells selected per frame | `18,000` | PASS |
| dry active cells retained per frame | `3,120` | PASS |
| failed-input dry zeros at step 1081 | `3,120` | PASS |
| corresponding positive NEMO weights | `3,120` | PASS |
| failed NEMO step | `1082` | NaN refusal |

The old raw converter copied legoESM's mask-zeroed `e3t(Kaa)` into every
interior cell.  That representation is correct inside legoESM, whose solve is
wet-masked, but it is not an identity input for NEMO's dry-column arithmetic.
The corrected converter begins with NEMO's complete recorded `e3t_Kaa` and
`rhs_T`, overlays only `tmask == 1`, and proves every unselected cell remains
bit-identical.

## Frozen predictions and current verdicts

| preregistered item | verdict | result |
|---|---|---|
| corrected input has 18,000 wet and 3,120 retained dry cells | **CONFIRMED** | all 360 frames pass |
| existing baseline and executable are unchanged | **CONFIRMED** | both baseline restarts and all three binary copies are byte-identical |
| `e3t_wet` and `content_wet` reach step 1440 | **UNMEASURED** | operator acquisition required |
| both directed day-240 restarts differ from baseline | **UNMEASURED** | operator acquisition required |
| wet-scale plant moves 18,000 and no retained dry values | **CONFIRMED** | plant prints fired marker and exits `2` |
| e3t removes more day-240 RMS than content | **UNMEASURED** | ranking withheld |

The retry intentionally does not rebuild NEMO: the compiled selector is
unchanged and the defect is solely in the external raw input.  It writes new
`e3t_wet` and `content_wet` run directories under Round 173, preserving the
failed Round-172 arm and its evidence.

## Landing verdict and scope

No first non-bit production statement is named and no candidate lands.  The
paired arms remain forced-input magnitude sensitivities, not source-exact
landing proof.  Rule 12 and Decisions 43/45/55/59 therefore do not run.  The
required shared-card table is zero movement on every card because the model
implementation is unchanged.

No configuration or carried-state choice was made.  No stabilizer was added.
The pre-implementation search found and reused the Round-172 acquisition,
Round-125 self-describing reader, and existing record stamps; no second
scientific harness was created.

## Review, citations, and tests

The required separate review command was run with `codex exec --sandbox
read-only`.  Its exact disposition is:

> independent review unavailable in-sandbox

The terminal error was `failed to initialize in-process app-server client:
Read-only file system`; it emitted no SHIP or DO NOT SHIP verdict.

The receipt citation gate reports PASS with six citations, zero failures,
zero unmapped citations, and zero map-audit failures; all nine self-test
plants fire.  Its shifted compiled-source plant reports FAIL and exits `1`.

The focused citation, Round-35 reader/stamp, and year-owner files report
`111 passed in 62.17s`.

The mandated single `tests/ocean/fidelity tests/ocean/unit -n 12` battery was
run once.  Nine JAX/XLA workers aborted during unrelated compilations and were
replaced.  The 1,500-second bound ended the run at 96%; it emitted no terminal
pytest summary, so its interspersed failures and errors cannot be classified
against the known-red set.  This is incomplete full-tree coverage, not a
pass.  The directly relevant focused files have the clean 111-test summary
above.

The worktree-stamp file separately reports `1 failed, 9 passed in 3.07s`.
Its three offenders are byte-for-byte the same list produced at base commit
`821b2f82c` (`round50`, `round146`, and `round156` gates); the new acquisition
adds no offender.

## Evidence hashes

| artifact | SHA-256 |
|---|---|
| `acquisition_preflight.log` | `386ebd69228344ee809ba59e5c18eb39e4e68b4d89ed8ba3f0ac9b4ab5e89f99` |
| `acquisition_wet_mask_plant.log` | `21a8f44b4b95e0ae8204a378eb2946b2f204697b058afc0c6f637da160b73f81` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `focused_tests.log` | `c89a4598b515d0c429935bea25c01be03838382d988e1802d69f13008d81cc29` |
| `full_fidelity_unit_tests.log` | `3d8020bed43992c05d6962d8262099943fea48919b4187cf962e146abc890192` |
| `worktree_stamp_test.log` | `e3c49ec3252037006d844f7a15fec0232b54356fc63bcedd580a216875e2df47` |
| `worktree_stamp_test_base.log` | `e16dff6dc9753378b4ef4960098651e609c63455fa7ce9434aa2089fa04466a3` |
| retry `run.sh` | `5bc055793b6b344bb2a7f9de30e603cdf8756d69aa984a7bf87ade31b1c279bb` |

## OPEN — round 174

1. The operator runs the committed Round-173 retry.  It must not rebuild the
   unchanged Round-172 binary or overwrite the failed Round-172 arm.
2. Admit the two new arms only if each reaches step 1440 with `STOP 0`, both
   required restarts exist, the executable remains byte-identical, and each
   day-240 restart differs from the admitted baseline.
3. Score baseline, `e3t_wet`, and `content_wet` day-240 temperature against
   the same Round-125 restart.  Promote the larger reduction from the
   `1.241262968697578e-03` K complete-K/e3w remainder to its producer walk;
   do not treat the forced-input ranking as landing proof.
4. Keep Round 163 as the immutable production before arm.  No configuration
   or carried-state decision is requested.
