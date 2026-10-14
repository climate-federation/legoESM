# Grid ladder rung G3 — round 2 receipt (gridladder_r15)

Status: **STOPPED_FOR_RECORD**. Nothing landed in the model. The card
equals NEMO bit for bit at step 2 on every field and level; by step 10
u and v are over the bar. The step where they part, and the statement,
need the frame twins this round's acquisition runs.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g3_round2.md
(commit fb76506d4, before any comparison).
Code: ba8fd38e5 (acquisition, ladder arm, tests), 2c0df66a2 (kt = 11
witness, chain test), 388b7231b (review fixes), this receipt's commit.
Evidence: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round15/.
**Acquisition needed:**
scripts/validate/ocean_fidelity/testcases/nemo_testcase_g3_orca2ts/run.sh
(operator: `run.sh --run`; it writes .../round15/oracle_g3_r15).
Decision 120 (damping switch) still pending; all of this is the as-built
deck (no hand edits).

## Take-aways

1. **NEMO stops at kt = 11 on the meridional-velocity check alone.**
   |V| = 10.24 m/s at (ji, jj, jk) = (22, 84, 27), 122.0 °E 7.4 °N, a
   partial bottom face. Sea level 3.85 m, |U| 3.04 m/s, salinity: inside.
2. **The card stops on the same face with the same bits:**
   |V| 10.244435336848499 on both sides; |U| identical; |ssh| differs in
   the last digit, same cell.
3. **Step 2 is NEMO's, bit for bit:** sea level, both barotropic
   velocities, their six histories, and u, v, T, S on all 30 levels
   (430,552 wet T cells; NEMO moved every one of them since the start).
4. **Step 10 is not:** u and v over the bar (relative 1.6e-14 full cells,
   9.9e-15 partial); sea level, barotropic velocities, histories, T, S at
   the bar.
5. **The hand-edit cells play no part so far:** 0 unequal among them at
   step 10, all four fields.

## 1. NEMO's boundary (registered, not a failure)

The check that fired is `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stpctl.f90:245`
on the maximum of `:181`. Same physical boundary as the ORCA2 lane's
OMT-0: real density, no momentum advection, no lateral friction, no
drag. G3's record is kt = 1..10.

| stp_ctl quantity | NEMO (output.abort.nc) | card | limit |
|---|---|---|---|
| \|ssh\| max, cell | 3.8518689708730403, (9, 90) | 3.8518689708730394, (9, 90) | 20 |
| \|U\| max, face | 3.0407808304957555, (21, 84, 27) | same bits | 10 |
| \|V\| max, face | 10.244435336848499, (22, 84, 27) | same bits | 10 |

## 2. Measurements (INDEPENDENT label only)

The record binary and the frame twins never ran (round 1's script stopped
at the boundary), so there are no step frames and no consecutive
restarts. GIVEN-NEMO-ENTRY needs the new acquisition.

| check | witness | result |
|---|---|---|
| geometry, 36 mesh rows, all levels | smoke mesh file | 36 of 36 bit-identical |
| kt = 2, 13 rows (2-D, level 1, six histories) | smoke restart | 0 unequal cells |
| kt = 2, all levels, u v T S, both depth classes | smoke restart | 0 unequal, dry 0 |
| kt = 10, sea level / uu_b / vv_b | stopped run's restart | at bar; 7160 / 7755 / 7759 cells unequal; max 2.2e-15 / 3.7e-16 / 5.0e-16 |
| kt = 10, level-1 u / v | same | **over**; max 4.8e-14 / 4.9e-14, interior open water |
| kt = 10, all levels u | same | **over**; full 1.58e-14 (184,535 cells), partial 9.9e-15 (6,935) |
| kt = 10, all levels v | same | **over**; full 5.3e-15 (184,104), partial 2.9e-15 (6,840) |
| kt = 10, all levels T / S | same | at bar; 180 / 165 cells unequal |
| kt = 11, u / v / sea level | output.abort.nc | over (expected past the record) |

Region split at kt = 10, level-1 u (unequal cells): fold 108, seam 220,
south 38, coast 446, open 6,164; the worst cells are open-ocean, not the
fold, the seam or the wall. Worst level for u and v: 16 (full class).

## 3. Predictions

| prediction | outcome |
|---|---|
| P1 geometry bit-identical | HELD |
| P2 kt = 2 at the bar | HELD (bit-identical) |
| P2 u, v NOT bit-identical at kt = 2 | FALSIFIED: bit-identical |
| P3 unequal T/S only where sea level is unequal | vacuous at kt = 2 (none unequal); at kt = 10, 6 T and 3 S cells sit in equal-sea-level columns, but earlier steps' sea level is unseen, so not a test |
| P4 kt = 10 at the bar | FALSIFIED: u, v over |
| P5 worst u/v in partial cells | FALSIFIED: full class worse |

## 4. First non-bit statement

NOT NAMED. The first unequal step lies in kt = 3..10 and no NEMO state
exists in between. Every candidate of the preregistration (equation of
state, partial-cell pressure gradient, depth mean of the right-hand side)
ran twice on the real T/S at kt = 1 and 2 without one unequal bit,
so the difference is not a systematic order difference on every cell.
PLAUSIBLE only: a data-dependent last-place case at one step. The ORCA2
lane runs the same density path (shared code); it has no G3 measurement.
Nothing landed, so there was nothing to re-score.

## 5. The new acquisition (run.sh, round 2)

- Admits round 1's smoke and stopped run IN PLACE, before any run: the
  smoke as a finished pinned 2-step run; the stopped run only if it is the
  pinned 100-step deck, stopped at kt = 11 by |V| alone, with its step-10
  restart and its abort state and no later restart.
- Runs the record binary three times, all nn_itend = 10: twin a (restart
  10), twin b (2, 4, 6, 8, 10), twin c (3, 5, 7, 9). A stop before
  kt = 10 is refused with NEMO's stop lines quoted.
- Admission: G1's checker at 10 steps (all stage steps, substeps 1 and 2),
  twin a's step 10 equals the reference binary's (the stopped run's),
  twin b's step 2 equals the smoke's, frames of b and c equal a's.

| gate | result |
|---|---|
| preflight on the committed tree | PREFLIGHT_OK |
| admission on the real round-1 record | boundary and smoke admitted; refused at the (empty) twin a, exit 73 |
| planted: \|V\| 9.24, stop at 10, smoke unfinished, no abort file | each refused, exit 73 |

## 6. Changes to shared code

| code | change | effect on G2 |
|---|---|---|
| G2 ladder script | `--case` (G2 default), restart-scored arm, kt = 11 abort witness | none: G2's step-2 slow test passes |

No model code changed. Searched before building: the G2 ladder (its
restart reader, depth classes, region rows, reused), the ORCA2 reader
(hand-edit cells derived from it). New: the arm, the abort reader.

## 7. Tests (synchronous)

- G3 chain against NEMO, kt 2, 10, 11 (slow): 2 passed (with the
  abort-witness unit test), 124 s.
- Fast: G3 file, G2 ladder file, three citation-receipt files: 77 passed,
  8 deselected. Citation gate's own tests: 21 passed.
- G2 step-2 slow test on the edited script: 1 passed.
- Plants: the arm's sea-level and level plants each exit 1; an arm with
  no witness refuses.
- The scoring JSONs of section 2 were produced on the tree before
  ba8fd38e5; the committed chain test re-derives every number it pins.

## 8. Review

Single review (codex): **"VERDICT: findings"** (codex_review_r15.log).

- "boundary admission neither requires nor hashes `output.abort.nc`":
  fixed, required and hashed; plant refused.
- "missing restart/abort witnesses are silently skipped ... can pass
  without scoring anything": fixed, the arm refuses.
- "`case` silently defaults to `ORCA2G2-zps`": not silent, measured: the
  G2 card refuses a G3 deck (exit 1, "deck must state
  namtsd.ln_tsd_init = False"). No change.

## 9. Choices

ASKED (round brief): kt = 1..10 as G3's record; the boundary registered;
records to nn_itend = 10; smoke admitted in place; as-built deck.

UNASKED, each offered for revert:
1. The stopped 100-step run doubles as the reference binary's step-10
   witness instead of a new 10-step reference run.
2. Twin a's restart list is `10` only.
3. The kt = 11 abort state scored as a witness and its stp_ctl maxima
   compared.
4. New evidence directory round15/oracle_g3_r15.
5. Two citation-map entries for the stop check.

No gate exists for: dual review (single, lane rule 6), claim review
before code (not done), non-vacuity (plants above).

UNVERIFIED:
- that the restart of a 100-step deck at step 10 equals a 10-step deck's
  byte for byte (no variable depends on nn_itend by reading, not run);
- the new run branch on a real record.
