# ORCA2 round 275 preregistration — certified-statement source-unit bisect

Date: 2026-10-13

Lane base: `b7bf36941dcf806ca28a6ff0f2157f2cb93dba75`

Label: **independent** rung 0.  The given-NEMO-entry and higher-card labels
are landing gates only after an independent rung-0 arm qualifies.

## Frozen question and source order

Round 274's complete thirteen-statement G1 arm failed Decision 96: 28 of
193 RMS-moving rows went toward NEMO and 165 went away, while the step-10
stage-3 state became explosive.  This round asks which compiled source unit
caused that result.  It does not carry the explosive step-10 state upstream
as an attribution.

The executing ORCA2 build resolves vector momentum advection: stage 1's RHS
is already complete and `dyn_adv` is called only at stages 2 and 3
(`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:360-366`,
`:368-417`).  It resolves the combined planetary-plus-relative EEN branch
(`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:737-770`) and its
partial-cell transports (`dynvor.f90:782-785`).  Its implicit vertical solve
uses the face and interface divisors at `dynzdf.f90:191-205` and `:360-375`.

The thirteen imported G1 selections are first measured one source unit at a
time, then individually inside the first moving unit:

1. barotropic EEN coefficients: ratio fold exchange, F-mask pivot row, and
   written ratio order;
2. RK3 stage update: whole stage-1 RHS and written velocity order;
3. EEN/stage-F: written EEN trend plus the interpolated F ratio;
4. vertical solve: literal barotropic residual removal;
5. completed-stage fold: top-V exchange;
6. live ratio/depth readers: stored T ratio, literal stage T ratio, rounded
   density depth, and literal stage U/V ratios.

The coordinate's stored-T-ratio selection is counted with unit 6, so the
thirteen-statement prose unit has fourteen selector fields.  The new G2
candidate, NEMO's two vertical-momentum thickness divisors, is a fourteenth
statement arm (fifteenth selector field) and is measured alone.  Its compiled
citations are the same `dynzdf.f90:191-205` and `:360-375` statements above.

## Frozen predictions and falsifiers

### R275-P1 — source-unit owner

Prediction: the RK3 stage-update unit is the first unit that reproduces the
round-274 explosive class, because the G1 proof exercised flux form while
ORCA2's compiled branch is vector form.  Within that unit,
`rk3_stage1_rhs_nemo_literal` is predicted to be the non-executing G1 branch;
it moves a strict majority of RMS rows away or produces growth by step 10.

Falsifier: that unit is finite and net-toward, or another singleton is the
first arm with the explosive class.  A falsifier is retained and the actual
unit/selector is reported.

### R275-P2 — singleton table

Prediction: every one of the thirteen G1 statement fields and the stored
ratio field is measured alone on the same committed tree, with toward / away
/ equal RMS counts, exact-row losses, first-over-bar movement, and step-10
stage-3 SSH/T/S/u/v maxima.  Any group that moves while all members alone are
clean is bisected by the smallest source-ordered pair sharing a consumer.

Falsifier: an arm cannot construct, does not complete 200 rows, or is scored
against a different baseline/protocol.  Such an arm is UNMEASURED-with-spec,
not silently omitted.

### R275-P3 — landing predicate

Prediction: at least one cited singleton is net-toward under Decision 96:
strict majority toward among RMS-moving rows, no exact-row loss, and the
first over-bar row toward or unchanged.  A qualifying singleton may land
only after OMT-1..4, rung 0, rung 10, GYRE year, DINO month, tanks, citations,
and tests are green.

Falsifier: no singleton qualifies, or any full landing gate is red.  In that
case the round is HELD with selectors off.

### R275-P4 — G2 divisor arm

Prediction: the G2 `zdf_momentum_nemo_face_thickness` statement is net-toward
on rung 0, because ORCA2 has the same 31-level partial-cell geometry and runs
the cited implicit solve.  It must not be selected on a card lacking the raw
NEMO face/interface operands.

Falsifier: the arm refuses the production ORCA2 operand route, loses an exact
row, or moves a majority away.  No alternate divisor or stabiliser is added.

### R275-P5 — controls

Prediction: the unchanged-selector control reproduces round 274's safe
independent rung-0 ladder, every classifier plant fires, citations remain
mapped, and sea ice plus `unmeasured_features` are untouched.

Falsifier: the control differs, a plant stays green, or any ice/configuration
choice moves.

## Landing rule

Measurement arms are private card selections.  A statement lands only under
R275-P3; otherwise all ORCA2/OMT selectors remain off.  No configuration
choice, carried-state change, floor, clip, or stabiliser is authorised.
