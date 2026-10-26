# Preregistration — round 164, developed velocity and vertical-chain walks

Committed before any round-164 measurement.  The entry for every developed
row is NEMO's admitted day-180 restart (step 1080, entry to step 1081), and
every legoESM row is produced through `LatLonCGridOceanModel.step` under the
production JIT.  Isolated closures are calibration rows only.

The immutable production tip is `85cf8601e5`; its certified headline is day
30 `6.572574374770603e-05` K, day 240 `1.644836070117868e-02` K and day 360
`1.122566001855131e-02` K.  No configuration field changes this round.

## A. Stage-2 velocity-form control, then the stage-1 output walk

The authority is round 159's JSON, not its superseded prose.  The current
two-solve production arm must reproduce the velocity-form stage-2 momentum
vertical-velocity row `2.334682468902387e-13` m/s.  Installing NEMO's recorded
stage-2 entry velocity together with that velocity form must reproduce
`1.0492082366460718e-13` m/s: 55.06% of the residual removed by RMS.  The
null installs legoESM's own stage-1 output and must reproduce its baseline.
The substituted continuity call is selected by its velocity-indicator tag,
never by call count alone.

The stage-1 output is then walked in compiled order.  The admitted round-140
and round-146 records supply the step-entry momentum RHS and its HPG, LDF,
VOR, KEG and ZAD families; the round-156 record supplies the stage-1 output
`uu/vv(Kmm)`.  The shared production live-operand trace supplies legoESM's
completed stage-1 RHS, raw stage velocity, barotropic target and corrected
stage output.  Rows are scored in that order, U and V separately, with an
entry-velocity ULP plant that must move a registered output row and exit
nonzero.

Predictions and falsifiers:

1. The round-159 velocity-form authority and 55.06% entry-velocity removal
   reproduce to the stored digits.  REFUTED if the null differs, the tagged
   call is not unique, or either RMS differs.
2. The previously landed live-geometry LDF transcription is BIT on the
   current tip, so HPG and LDF are exact and VOR is the first non-bit RHS
   family, at the compiled-rounding scale.  REFUTED if an earlier family is
   non-bit or VOR is bit-exact; the measured first family replaces this
   prediction.
3. The first non-bit stage-1 output statement is named only after the
   completed RHS, raw update, barotropic target and correction are scored.
   An inherited non-bit operand is reported as inherited, not assigned to a
   downstream arithmetic statement.

## B. Developed tracer-ZDF/TKE chain

First re-run the one-step process ranking in the now-landed production arm.
It must reproduce round 161/162's corrected-arm values: advection
`6.181193168124e-12` K RMS, vertical diffusion
`2.183409436263e-05` K RMS, and combined advection plus vertical diffusion
`2.1834094298886284e-05` K RMS.  Vertical diffusion must remain the largest
one-step process row.  Any different value is a REFUTED prediction and stops
attribution until explained.

Then extend the existing developed-state process walk, not a second harness,
to compare its production vertical trace against the admitted round-125
record in compiled dataflow order: final heat diffusivity `avt`, isoneutral
diffusivity, effective coefficient, live thicknesses, content RHS, lower /
diagonal / upper matrix, and solved T.  A one-ULP change to one consumed heat
coefficient must move a matrix row and solved temperature and exit nonzero.

Prediction: the first directly observed non-bit boundary is the mixing
coefficient entering `tra_zdf`, before the matrix.  The round-125 record does
not carry the developed `zdf_tke` operands (`en`, `sh2`, `rn2`, individual
TKE sweep fields), so this round may name only the compiled `avt_k -> avt`
assignment as the first observed statement and say its operand is inherited.
If `avt` is bit-exact, the first later non-bit row is named instead.  If `avt`
is non-bit and no existing admitted record discriminates its upstream TKE
producer, the result is STOPPED_FOR_RECORD and a fail-closed acquisition
script is written; no TKE statement is guessed from attribution.

## C. ORCA2 blast radius and certified route

Add `ORCA2-zps` to the Decision-43 card census by building the real card from
the existing deck, and import the model's own `resolved` and `executes`
predicates.  Frozen expectation: it resolves the same two-solve program but
its explicit card choice remains `nemo_stage_momentum_wzv_split=False`, so
`executes_route=true` and `executes_at_this_tip=false`.  A plant that removes
or falsifies this row must exit nonzero.

Attempt only the existing certified ORCA2 gate against its own oracle record.
The constructor currently names six unresolved selected mechanisms and the
execution validator is designed to refuse such a card.  If it refuses before
the trajectory, report the certified route as UNMEASURED with the exact
reason; do not weaken validation and do not flip the card.  If it executes,
measure explicit false and explicit true arms and register every moved row.
Decision 58 remains pending; this round does not answer it by configuration.

## Verdict and landing rule

This is expected to be a measurement round.  No production physics lands
unless a one-variable NEMO statement passes Decision 43/45 as amended by
Decision 55: day 30 decreases, first-over-bar is not earlier, no kt=1 AT-BAR
row leaves, all moved rows are registered, all executing cards are measured,
and the year rows are handled exactly as note AT permits.  Otherwise status
is HELD, or STOPPED_FOR_RECORD if the first observed vertical boundary needs
new NEMO operands.  Citation-gate and shifted-citation plants, focused tests,
the full required ocean battery, and a separate read-only Codex review are
required before the receipt closes.
