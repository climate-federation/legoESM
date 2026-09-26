# NEMO testcase Lane 4 — ORCA2 card round 24 Decision-54 receipt

Date: 2026-09-26

Parent: `e3f2a7eb97bbaa09f6be0c9c065d0831adbdadef`

Experiment tip: `59f81ffbc2b7753c4df28da96746a08381f3b665`

Status: **HELD — Decision 54 does not land.**  The complete three-part
`dyn_ldf` attribution is bit-exact as an isolated operator on NEMO's recorded
kt=2 entry, but the independent trajectory moves at kt=1 stage 2 and becomes
nonphysical: steps 1--3 complete and step 4 is refused because the evolved
free surface makes the raw-mesh `e3w` stretch non-positive.  Every model and
test change was reverted; the final `packages/` and `tests/` trees are
identical to the parent.

All ladder numbers below are **independent with Decision-52 SSH**.  The direct
operator replay is separately labelled **given NEMO's entry**.  The six
sea-ice selectors and the card's `unmeasured_features` tuple were not changed.

## Compiled statements and experiment

The admitted build reads the three-dimensional T- and F-point coefficients
from `eddy_viscosity_3D.nc` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:348-353` and
multiplies the F coefficient by `fmask` once at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:387-393`.
The executing level operator then uses that already-masked coefficient with
the live F thickness, stored F-cell area reciprocal, circulation edge
lengths, live T/U/V thicknesses, and Kmm face divisors at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123-140`.

The experiment implemented those statements as one unit: no second F mask
for the file-read coefficient, raw NEMO T/U/V/F thickness operands, and the
native F-cell `e1f*e2f` reciprocal plus circulation edges.  An initially
broader tripolar-area change was retracted during development because it
altered unrelated consumers; the scored experiment uses an LDF-only metric
operand seam.  That retraction is present in the commit history and the final
tree contains neither version.

| card | which Decision-54 statements execute |
|---|---|
| ORCA2-zps | all three: single file-coefficient mask, live e3 weighting, native F area/edges |
| GYRE-zco | shared e3-weighted div/curl and metric path; not the file-coefficient mask arm |
| DINO `nemo_dino_kamm[_mlf]` | none of this e3-weighted arm (`lateral_viscosity_e3_weighting=off`) |
| LOCK_EXCHANGE-zco | none (`vector_laplacian`, e3 weighting off) |
| OVERFLOW-zps | none (`vector_laplacian`, e3 weighting off) |

## Given-NEMO-entry operator result

At kt=2 the production operator at experiment tip `59f81ffbc` is bit-exact
against a literal replay of the compiled statements: **0 / 411,736** scored U
cells and **0 / 412,537** scored V cells differ.  The live-operand replay is
also bit-exact, and all 26,640 native F-area reciprocals are `np.array_equal`.
The one-ULP U plant changes exactly one scored cell and returns `DEBT`.

This confirms the local transcription.  It does not authorize landing when
the trajectory gate fails.

## Independent ORCA2 ladder

The admitted round-23 artifact is the ten-step baseline.  The experimental
arm has a complete kt=1..3 artifact (12 checkpoints); the required kt=1..10
run fails closed while entering step 4 with:

`raw-mesh e3w_int must contain only finite values > 0`

The outcome gate registers all **45** observable moved field rows: 14 move
toward NEMO by maximum absolute error and 31 move away.  No formerly
bit-identical row leaves the bar and the first non-bit statement remains kt=1
stage-1 T: 233,341 / 399,600 cells, maximum
`0.0014770192519697467` degC.

The first movement is already kt=1 stage-2 U, not the preregistered kt=2
stage-3 bound.  Its maximum error grows from `0.06470386947382581` to
`0.10080009966621735` m/s.  The companion V row grows from
`0.034012848056840184` to `0.11456680400114852` m/s.  By kt=3 stage 3 the U
maximum is `271.2640512806289` m/s and V is `136.59561139623077` m/s.  The
step-4 refusal is therefore a measured downstream cancellation exposure, not
an operator-replay discrepancy and not permission to add a stabilizer.

The complete before/after row objects and their directions are in
`round24/result.json`; no post-refusal row is claimed or synthesized.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R24-P1 | **REFUTED** | The merged path did not already carry all six raw thickness operands and the native F metric; explicit LDF-only routing was required. |
| R24-P2 | **CONFIRMED** | The complete operator replay is bit-exact in U and V. |
| R24-P3 | **REFUTED** | First movement is kt=1 stage-2 U, earlier than kt=2 stage 3. |
| R24-P4 | **CONFIRMED** | First non-bit statement and its score are unchanged. |
| R24-P5 | **UNMEASURED AFTER STOP** | GYRE base/tip certification was not run because the ORCA2 ladder had already prohibited landing; the final package tree equals the certified parent. |
| R24-P6 | **UNMEASURED AFTER STOP** | DINO/LOCK/OVERFLOW were not run for the prohibited experiment; the final package tree equals the certified parent. |
| R24-P7 | **CONFIRMED** | The one-ULP operator plant returns `DEBT`; the citation plant also fires. |

Failed predictions were not rewritten.  P5 and P6 are not inferred passes.

## Gate, review, and tests

The round-24 outcome gate exits 2 with `HELD`, checks the admitted 40-row
baseline, registers all 45 moved rows through kt=3, verifies no AT-BAR row
left the bar, requires the exact step-4 refusal, and requires the operator
plant to fire.

The required separate `codex exec --sandbox read-only` review was attempted
at the committed final diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

The citation gate and test results are recorded below after their final runs.

Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round24/`.

## Choices

ASKED: Decision 54 authorized the three cited `dyn_ldf` statements only if
the full ORCA2 and cross-card gates passed.  They did not.

UNASKED: none.  No configuration choice, carried-state change, stabilizer,
sea-ice selector, or NEMO source edit was made.

## OPEN

1. Decision 54 remains open.  Before another landing attempt, substitute the
   three attributed statements separately on the independent trajectory to
   identify which component exposes the step-4 instability and measure the
   cancelling pair; do not relax the raw-`e3w` refusal.
2. The first-movement mechanism is now bounded to the stage-1 LDF tendency
   consumed by kt=1 stage 2; the preregistered kt=2-stage-3 assumption is
   retired.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. The independent ORCA2 year still depends on its scheduled independent
   initial-state completion.
6. The seven inherited duplicate citation-map literal keys remain open.
