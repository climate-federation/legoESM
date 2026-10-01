# Preregistration — round 162, the cancellation and the T-point ratio

Committed before any round-162 measurement ran.  Entry for every
developed-state row: NEMO's admitted day-180 daily restart (step 1080, the
entry to step 1081).  Every model arm is one production step through
`LatLonCGridOceanModel.step` under production just-in-time compilation, never
an isolated closure (operator note L-amend).  Oracles: the admitted round-156
stage-2 record under `round157/oracle_developed_stage2`, the admitted
round-123 process-budget record, and the admitted round-125 vertical
matrix/solve record, all read through the readers earlier rounds already
share, on the same windows.

## What round 161 left, in one paragraph

Round 161 exonerated the face free-surface ratio: installing the oracle's own
recorded `r3u(Kmm)`/`r3v(Kmm)` in the velocity-indicator continuity solve
moved the developed stage-2 vertical-velocity residual from
`2.334682468902387e-13` to `2.334683948520e-13` m/s, a worsening of
`0.00006%`, so the residual is UNOWNED after four one-at-a-time oracle
substitutions.  It also decomposed the ratio difference itself: legoESM's
ratio statement on the oracle's own recorded `ssh(Kmm)` reproduces the
oracle's ratio to `4.04e-21` root mean square, so the STAGE SEA SURFACE HEIGHT
owns `1.0000000006` of the live `1.4516405645036015e-13` difference.  And it
measured, at the same developed entry, that round 160's second continuity
solve takes the one-step tracer ADVECTION row from `1.097459140409e-08` K to
`6.181193168124e-12` K while the VERTICAL DIFFUSION row grows from
`2.183408900044e-05` K to `2.183409436263e-05` K — opposite directions
CONFIRMED, cancellation PLAUSIBLE ONLY.

## Order A — the discriminating measurement for the cancellation

This is the input Decision 55 is waiting on and it is reported first.

Each row of round 152's ranking is legoESM-minus-NEMO for one isolated
one-step temperature contribution at the same day-180 entry.  Write the two
error fields as `e_adv` and `e_zdf`.  The claim under test is that they
CANCEL in production — that `e_adv` and `e_zdf` point in opposite directions
cell by cell, so removing `e_adv` (which the second continuity solve very
nearly does) leaves the combined error LARGER even though the advection row
alone collapses by a factor of 1,776.

Three scored quantities, all from the two production steps the ranking mode
already takes, no new record and no new run:

1. **the combined row** — legoESM's `advection + vertical_diffusion` one-step
   temperature increment scored against NEMO's `advection + vertical_diffusion`
   recorded trend, over the same wet mask, in BOTH arms;
2. **the cell-by-cell sign correlation** — the Pearson correlation of `e_adv`
   against `e_zdf` over the wet cells in the PRODUCTION arm, together with the
   fraction of wet cells on which the two carry opposite signs;
3. **the quadrature reference** — `sqrt(rms(e_adv)^2 + rms(e_zdf)^2)`, which is
   what the combined row would be if the two errors were independent.

### The verdict rule, fixed here

**CANCELLATION CONFIRMED** requires all three: the correlation is negative and
below `-0.1`; the production combined row is SMALLER than the production
vertical-diffusion row alone; and the corrected arm's combined row is LARGER
than the production arm's by more than ten times the `2e-10` K run-to-run
floor.  **REFUTED** if the correlation is above `+0.1` (the two reinforce), or
if it lies in `[-0.1, +0.1]` (they are independent and the advection error
simply does not matter at this size), or if the combined row does not grow.
An outcome that satisfies some but not all three is reported as such and
called neither.

### The arithmetic the verdict rests on, read off the definition, not measured

`rms(e_adv + e_zdf)^2 = rms(e_adv)^2 + 2*cov + rms(e_zdf)^2`, so with
`rms(e_adv) = 1.097e-08` K and `rms(e_zdf) = 2.183e-05` K the whole span the
correlation can move the combined row over is `+/- rms(e_adv)`, i.e.
`+/- 1.1e-08` K, or `5.0e-04` relative.  A perfectly anti-correlated pair puts
the production combined row at about `2.18231e-05` K.  This is why the
measurement is well posed at all: the cross term is four orders above the
`2e-10` K floor even though the advection row is three orders below the
vertical-diffusion row.

### Predictions

* **A1** The correlation is NEGATIVE.  REFUTED if it is `>= -0.1`.
* **A2** The production combined row is below the production
  vertical-diffusion row `2.183408900044e-05` K, and the corrected combined
  row is above the production combined row.  REFUTED if either comparison
  goes the other way.
* **A3** The decomposition control holds: the combined error field equals
  `e_adv + e_zdf` to within 8 units in the last place of its own size.
  REFUTED if it does not, and then the combined row is measuring a different
  pairing and nothing above is reported.
* **A4** The production arm reproduces round 152's published table to one part
  in a million, as round 161's run of the same mode did.  REFUTED otherwise,
  and the round stops there.

### If Order A confirms, what gets named

The cancelling partner is the vertical-diffusion path.  This round then names
its FIRST non-bit statement in compiled order at the same developed entry, by
scoring legoESM's own vertical-solve trace against the admitted round-125
record at step 1081, field by field in the order `tra_zdf` computes them:
the mixing coefficients, the live thicknesses, the tridiagonal matrix, the
content it acts on, and the solve.  The first field that is not bit-identical
names round 163's target and is cited to the compiled line of the build that
wrote the record.  Nothing is named if Order A refutes.

## Order B — the T-point ratio

The ONE operand of the velocity-indicator continuity solve that rounds
159-161 never substituted.  It enters twice on the compiled path: the whole
horizontal divergence is divided by the live thickness at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130` and the same
live thickness multiplies the divergence back at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:153`; legoESM forms the
ratio at `ocean_pe_latlon_cgrid.py:1715`, the thickness at
`ocean_pe_latlon_cgrid.py:1716` and hands it to the divergence block at
`ocean_pe_latlon_cgrid.py:1740`.  The substitution replaces that ONE operand,
for exactly one velocity-indicator call, by an observer that restores the
producer immediately — the technique rounds 152 and 161 already use, so no
production line changes and no card can reach it.

**Registered before measuring, read off the compiled source.**  The order for
this round named the stretching term at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:297-298` as a second
place the ratio at the now level appears.  It does not: that statement reads
the ratio at the AFTER and BEFORE levels, not the now level, and legoESM's
answering pair is `ocean_pe_latlon_cgrid.py:1758-1759`.  Round 160 already
measured the after-level pair inert, and the before level is the entry state's
own sea surface height, which this round's entry bridge loads bit-identically
from NEMO's restart.  So the stretching term is accounted for and Order B is
the divisor alone.

**The operand installed** is NEMO's own ratio statement
(`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90:257`, one product of
the height and the reciprocal reference depth) evaluated on the ORACLE'S OWN
recorded `ssh(Kmm)`.  It is NOT a directly recorded `r3t`; the record carries
the height, not the T-point ratio, and this is said here rather than left
implied.  Round 161's measurement is what makes that legitimate: on the same
record the same class of statement reproduced the oracle's recorded face ratio
to 1.2 units in its last place.

### Predictions

* **B1** The T-point ratio does NOT own the residual: the substitution removes
  less than 10% of the `2.334682468902387e-13` m/s residual in the corrected
  arm.  The reason is arithmetic, read off the code: the ratio enters as
  `1 + r3t` on both a divide and a multiply, so a relative ratio difference of
  order `1.5e-13` moves the divergence by that same relative amount, while the
  residual is `9.74e-08` of the vertical velocity's own size.  REFUTED if it
  removes 50% or more, in which case the T-point ratio IS the owner and round
  163 lands its producer.
* **B2** The two arms agree: the corrected arm (round 160's second solve) and
  the shared velocity-form arm (round 159's) remove the same fraction to
  within a factor of two.  REFUTED otherwise, which would mean the
  substitution reaches more than the one producer.
* **B3** The observer is passive and live: zero bytes move on every state leaf
  with the substitution off, and at least one cell moves with it on.
* **B4** The substitution does not reach the tracers: the tracer's own
  vertical velocity is bit-identical with and without it, 0 of 18,000.

## Order C — the stage sea surface height

Round 161 named the stage height as the owner of the ratio difference and
predicted its size from the ratio's own definition rather than measuring it.
This round measures it: the height legoESM hands the continuity producer at
the stage-2 velocity-indicator call, captured in the harness by the same
observer, scored against the oracle's recorded `ssh(Kmm)` on the same T
window.

* **C1** The difference is of order `6e-10` m root mean square — between
  `1e-10` and `5e-09` m.  REFUTED outside that band, and then round 161's
  arithmetic off the ratio statement was wrong and the receipt says so.
* **C2** The height difference DIVIDED by the reference depth reproduces the
  face-ratio difference round 161 measured, to within a factor of two:
  `1.4516405645036015e-13` at the u points.  REFUTED outside a factor of two,
  which would mean the height is not the operand round 161 said it was.
* **C3** If C1 and C2 hold, the walk's next target is the barotropic step that
  produces the after height — round 138's first non-bit boundary — and the
  receipt names it.

## Order D — the fail-closed plant path

For the third round running, the `STATUS PLANT-BLIND` path has been registered
as structurally distinct but never demonstrated.  This round demonstrates it
once: a deliberately blinded plant is committed to the tree, run, its output
quoted, and then removed in its own commit.  The blinded plant installs an
INERT substitution (legoESM's own live ratio in place of the oracle's) and is
watched by the tracer-identity control, which is the WRONG control for it and
cannot see it.  The walk must then print `STATUS PLANT-BLIND` and exit 2,
never the marker a caught plant prints.

* **D1** The blinded plant is not caught, the walk prints
  `STATUS PLANT-BLIND` and exits 2.  REFUTED if it prints `STATUS PLANT-FIRED`
  or exits 0 or 1, and then the fail-closed path is broken and that is the
  round's finding.
* **D2** The round's two real plants are caught, print `STATUS PLANT-FIRED`
  and exit 1.

## Landing rule

This is expected to be a MEASUREMENT round.  Anything lands only under the
FULL Decision 43/45 gate: day-30 temperature root-mean-square decreases, the
first-over-bar row is not earlier, no kt=1 at-bar row leaves the bar, every
moved row registered with the harness's run-to-run floor of about `2e-10` K
quoted next to it, DINO measured if the statement is shared and the card
census saying which cards execute the changed code, day 240
`1.6446718648e-02` K and day 360 `1.1223450850e-02` K not worse, owners ranked
at day 240, the push gate green, and a Claude reviewer's verdict quoted
verbatim.  Anything that fails a row is HELD behind its receipt with
production restored and proven.  A candidate that needs a carried-state change
or a configuration choice is written up as a DECISION_NEEDED instead of
landed.  Decision 55 is NOT reopened by this round; Order A is reported as an
input to it.

## Before arm

The tip this round starts from, `3f7b69ff3842f9aaf159f32819e7f133602e0aa8`.
The lane's inherited year rows are day 30 `6.88819351379691829e-05` K, day 240
`1.64467186440671112e-02` K and day 360 `1.12234508615602115e-02` K.
