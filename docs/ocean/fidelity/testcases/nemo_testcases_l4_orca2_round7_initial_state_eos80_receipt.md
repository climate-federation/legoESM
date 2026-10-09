# NEMO testcase Lane 4 — ORCA2 card round 7 transcription receipt

Date: 2026-09-22

Starting tip: `90ab3e871a4d7cfbc11441b345b007016726d348`

Preregistration: `c05473544`

Status: **HELD.**  Both transcriptions round 6 left open are landed and gated.
The ORCA2 card now builds its OWN initial temperature and salinity exactly as
NEMO does — every one of the 799,200 cells of each field is bit-identical to
NEMO's step-1 state, with no operand loaded from the record — and the
production step now runs NEMO's EOS-80 stratification instead of refusing it.

The ladder consequently reaches the momentum tendencies for the first time and
stops there on the NEXT unbuilt statement: the four-cell vertex thickness the
EEN vorticity operator needs is not defined on a tripolar fold row.  No first
non-bit arithmetic statement exists yet, and the kt=10 magnitude therefore
stays **UNMEASURED** — not zero, not extrapolated.

No configuration, selector, default, carried state, stabilizer, NEMO source or
sea-ice registry entry changed.  Sea ice remains out of scope and the
six-entry `unmeasured_features` tuple is unchanged.

## 1. What this round landed

| statement | compiled owner | disposition |
|---|---|---|
| ORCA_R2 initial hand alterations | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:218-253` | transcribed; gated bit-exact on the full domain |
| mask AFTER those alterations | `:308-309` | transcribed; the alterations act on the unmasked field |
| global-to-local index map | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/mppini.f90:1587-1593` | read to derive the boxes, not fitted to the residual |
| initial-condition call site | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/istate.f90:117-118` | confirms the reader is what builds the state |
| EOS-80 coefficient initialization | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:2284-2293` | 126 numbers re-derived and compared bit for bit |
| shared expansion-coefficient case | `:1281-1330` | one polynomial serves both equation-of-state forms |
| buoyancy-frequency assembly | `:1587-1647` | carries no equation-of-state branch at all |
| vertex thickness for EEN | `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynvor.f90:912-937` | **NOT transcribed** — the round's stop |

## 2. The independent initial state, and Decision 52's labels

The labels are binding and are never mixed in one row.

| quantity | label | result |
|---|---|---|
| kt=1 entry temperature | independent | **0 / 799,200 unequal** |
| kt=1 entry salinity | independent | **0 / 799,200 unequal** |
| kt=1 entry zonal and meridional velocity | independent | 0 unequal each |
| kt=1 entry sea-surface height | independent | 16,433 / 26,640 unequal, max 0.015479333813968585 m |
| all five fields after the sea-surface-height bridge | given NEMO's entry | 0 unequal each |
| kt=10 magnitude of the first non-bit statement | — | **UNMEASURED**; no such statement is reached |

Decision 52's debt is therefore discharged for temperature and salinity: the
card does not need NEMO's recorded state for them and is not given it.  The
remaining sea-surface-height difference is the one the lane has recorded since
round 1 and is owned by the initial sea-ice category configuration, which is
out of scope; Decision 52's bridge stays the only explicit entry replacement.

**The ablation, not a shrinking count.**  Rebuilding the same initial state
through the same helper with the alterations REMOVED reproduces round 6's
residual exactly: 1,283 unequal temperature cells with maximum 20.515075852794034 °C
and 720 unequal salinity cells with maximum 0.3500000000000014 PSU.  So the
alterations own the entire residual rather than part of it, and the gate
refuses if that ablation ever becomes vacuous.

**Where the indices came from.**  The source writes its boxes in the global
halo-inclusive frame (`101 + nn_hls`, `141 + nn_hls - 1`), and the compiled
index map turns a global index into `index - nn_hls`, so every halo term
cancels and the boxes are halo-independent inner one-based ranges: rows 101 to
109 and columns 140 to 154 for the Alboran pair, rows 87 to 96 and columns 147
to 159 for the Red Sea assignment.  This deck has a two-cell halo.  Nothing was
tuned against the measured residual; the first evaluation of these bounds was
already bit-exact.

## 3. The EOS-80 stratification branch

The record's own namelist selects EOS-80 (`ln_eos80 = .true.` in the run
directory's `namelist_cfg`), and the domain file resolves the configuration to
ORCA index 2, which is also what makes the hand alterations live.

The transcription is a coefficient-set selection, not a formula, and the
compiled source is what says so: the expansion-coefficient routine runs ONE
polynomial case for both equation-of-state forms and the buoyancy-frequency
routine contains no such selection at all.  The two forms differ only in what
the initialization loads — the normalization (`rdeltaS` 20 rather than 32,
`r1_S0` one fortieth rather than the absolute-salinity scaling) and 122
polynomial coefficients.

The gate re-derives all 126 numbers from the compiled file on every run and
compares them to the committed set bit for bit: 52 density, 35 thermal, 35
haline and 4 normalization constants, **0 differing**.  Perturbing one of them
by a single representable step makes the gate refuse.

Nothing else moved: the stratification default stays the simplified form, the
GYRE card stays on TEOS-10, and an unknown selector still raises.

## 4. How far the ladder now reaches

The production step is entered at kt=1 with fp64, scalar-libm, CPU and JIT
retained, and with the exact recorded surface operands.  It passes the
stratification it used to refuse and stops inside the momentum tendencies:
the vertex thickness the EEN vorticity operator needs is not defined on a
tripolar fold row.

NEMO does not build that field with a special fold formula.  It computes the
masked four-cell average over the interior and then completes the fold row with
its ordinary F-point boundary exchange, and finally replaces any zero with the
reference thickness.  Transcribing that is the next statement in execution
order and is NOT attempted here.

The gate records this stop the way the previous one was recorded — named,
cited, with no magnitude registered — and exits non-zero.  Only a deliberate
"not built" refusal is caught; any other error propagates as a defect.

## 5. Prediction ledger

| ID | verdict | evidence |
|---|---|---|
| R7-P1 | **CONFIRMED** | The expansion-coefficient routine's polynomial case names both forms; the buoyancy-frequency routine has no such selection. |
| R7-P2 | **CONFIRMED** | 126 of 126 coefficients and normalization constants bit-identical to the compiled block. |
| R7-P3 | **CONFIRMED** | Independent kt=1 temperature and salinity are 0 / 799,200 unequal each; the ablation restores 1,283 and 720. |
| R7-P4 | **CONFIRMED** | The compiled bounds were bit-exact on first evaluation; no shift, transpose or level change was applied. |
| R7-P5 | **CONFIRMED** | GYRE base and tip agree with zero differing rows, array-equal residuals and byte-identical 30-day snapshots (section 6). |
| R7-P6 | **REFUTED / kt=10 UNMEASURED** | The first non-bit statement moved later but is still not reached: the ladder stops at the vertex thickness before any arithmetic row can be scored. |

R7-P6 is the honest outcome of a ladder that gained two statements and found a
third missing.  No magnitude is registered for a row that was never computed.

## 6. GYRE, which shares this implementation

Both changed model files serve GYRE too, so GYRE was re-measured at the round's
base tip and at its tip with the eval protocol held byte-identical.

| check | result |
|---|---|
| ten-step trajectory, certified rows compared | 70 |
| rows whose status changed | 0 |
| violations | 0 |
| first row over the bar, before and after | the same one, at step 2 |
| per-cell residual arrays (210 of them) | every one array-equal |
| 30-day member, daily state snapshots | 31 files, 0 differing bytes |
| 30-day member manifest | differs only in the recorded commit, path and wall time |

GYRE therefore did not move, which is what makes this a one-sided ORCA2
landing rather than a GYRE landing.

## 7. Controls, and what each one proves

| control | result |
|---|---|
| remove the hand alterations, rebuild through the same helper | restores 1,283 temperature and 720 salinity unequal cells, with round 6's maxima |
| perturb one representable kt=1 temperature value | gate refuses at the named entry-identity check |
| perturb one recorded surface-frame digest | gate refuses at the named admission check |
| perturb one EOS-80 coefficient by one representable step | coefficient audit refuses and names the coefficient |
| shift a receipt citation by two lines | citation gate fails |
| shift the alteration box by one row | the cell-level unit assertions fail |
| an unbuilt statement from another routine | the stop arm refuses to label it, rather than borrowing this round's citation |
| a non-bit entry field other than sea-surface height | the twin is reported NOT eligible |

The last two controls exist because the independent review found the first
version of the stop arm failing both; see section 8.

## 8. Implementation scope, review, gates and tests

Three model lines changed, in two files, plus one factored-out constructor: the
initial-state construction moved into a named helper so the gate could ablate
it, and two allowed-value sets gained the EOS-80 name.  No parallel ladder,
parser or launcher was created; the existing ORCA2 ladder was extended.  No
NEMO run was attempted and no acquisition is needed.

The required separate read-only independent review RAN ONCE this round, on
the committed diff, before the user paused that tool for the remainder of the
round; no second pass was run, and none was attempted after the pause.  Its
verdict: *"Found 2 defects ... No defect found in index/level mapping, array
aliasing, defaults, or GYRE routing. T/S match the emitted kt=1 oracle
bit-for-bit."*  Both defects were in the gate's new stop arm, neither in the
model, and both are fixed with tests that exercise the failing side: it
labelled every deliberate refusal as this round's gap, and it certified the
entry state unconditionally.  The reviewer also ran its own independent check
that the TEOS-10 stratification path is byte-unchanged.  Its full report is
retained with the round's evidence.

Because the pause landed after the fixes, those fixes are themselves
UNREVIEWED by a second party: they are small, they are covered by the two new
tests named above, and they only narrow what the gate will certify, but that
is the honest status and the next round should re-review them.

The receipt citation gate passes with every citation mapped, no failure and no
map-audit failure, and a rigid two-line shift of a round-7 citation makes it
fail.  One existing citation was rigidly re-anchored: inserting the helper
moved a recipe line from 1411 to 1493 with its text unchanged.

Repository ratchets: 5,241 passed, 3 failed — all three are the listed
pre-existing reds (two inline-coefficient rows and one parameter-spec row), and
no new failing identifier appeared.

The single required ocean-fidelity battery completed in 2,455.10 s:
**1 failed, 1518 passed, 7 skipped**.  That one failure is the listed
pre-existing sea-ice scalar-math provenance red (`A MY_SRC is not verbatim`);
no new failing identifier appeared.  The seven named push gates pass at the
round's final tip: **136 passed in 1,083.03 s**.

New direct tests: six on the hand alterations, pinning each box, each level
and each value, with a synthetic shifted box that must fail; three on the
EOS-80 stratification arm, including one that would catch it silently
returning the TEOS-10 answer and one that keeps the typo refusal; one on the
convection trigger's acceptance of the same arm; and four on the gate's
coefficient audit, its refusal parser and its eligibility helper.

## 9. OPEN

1. **The next statement in execution order is the vertex thickness on the
   tripolar fold row.**  NEMO builds it as the masked four-cell average over
   the interior, completes the fold row with its ordinary F-point boundary
   exchange, and replaces any remaining zero with the reference thickness
   (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynvor.f90:912-937`).
   That is a shared-model change and will need the full GYRE identity proof
   again, though GYRE's closed box never takes the fold branch.
2. Only after that can the ladder produce a first non-bit ARITHMETIC statement
   and a kt=10 magnitude.  Neither is registered here, and no number should be
   quoted for them.
3. The independent sea-surface height still differs by up to 1.55 cm on 16,433
   of 26,640 surface cells.  Its owner is the initial sea-ice category
   configuration, which is out of scope on this lane; Decision 52's bridge
   remains the only explicit entry operand replacement.
4. The recorded runoff tracer-source operands remain an explicit later
   boundary; this round did not reach them.
5. GitHub issue 1455 remains an operator-post action because no GitHub
   connector is installed in this environment.

## Choices

ASKED: Decision 52's sea-surface-height bridge remains the only explicit entry
replacement, and the independent initial state it mandated is now built.
UNASKED: none.  The EOS-80 arm is an addition to an allowed set, not a change
of any default; no scheme selection, tunable, threshold, cadence, resolution,
timestep, carried state, data source or previously-tolerated condition moved.
The one ablation switch added is a gate-only keyword whose default is what NEMO
executes, and no card, recipe or driver exposes it.
