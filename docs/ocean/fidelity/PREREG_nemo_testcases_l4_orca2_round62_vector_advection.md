# ORCA2 round 62 preregistration — vector-invariant stage-2 advection

Date: 2026-09-28

Base: `1efb941c314cba10f7b5b85034faebec1de27a8b`

Claim label: **given NEMO's recorded operands**.  Independent-start and
month-scale claims are out of scope for this measurement round.  Sea ice, its
six selectors, and the ORCA2 card's `unmeasured_features` tuple are frozen.

## Source-first scope

The existing four-card scope probe must print ORCA2-zps and GYRE-zco as
`vector_invariant`, and OVERFLOW-zps and LOCK-zco as `flux_form/nemo_up3`,
before any numerical score is accepted.  Therefore the round-56
`651 / 16,900` after-VOR to after-ADV count is an OVERFLOW boundary and is not
an ORCA2 measurement.

The producing ORCA2 configuration executes HPG, VOR, then the vector-form
advection call at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:402-420`.
The compiled vector dispatcher calls KEG and then ZAD at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynadv.f90:134-138`.
KEG forms `zu`, `zv`, `zhke`, then the two face updates at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynkeg.f90:117-130`.
ZAD forms its vertical transports, shear products, level updates, and separate
bottom update at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzad.f90:102-137`.

The admitted rank-0 stream writes the cumulative before/after boundaries at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:372-420`.
Its SHA-256 is frozen as
`18a7b4701f167c5fd17f93875c318eb21ea30ce79c523b269ac6d8abd2ebd35d`.

## Frozen measurements and falsifiers

1. **Card execution census.** Number: disagreements from the existing scope
   probe.  CONFIRM if zero and the four cards resolve to the two pairs above;
   REFUTE if any disagreement exists.  A refutation stops the round.
2. **Record admission.** Number: parsed fields and physical EOF in the
   rank-0 `NEMO_L2_RKTRM_1` stream.  CONFIRM at eight fields in the fixed order
   `before`, `after_hpg`, `after_vorticity`, `after_advection`, U then V for
   each boundary, with header `(1,1,2,3,2,94,152,31,64)` and the frozen digest.
   REFUTE on any header, order, length, digest, finite-payload, or EOF mismatch.
3. **ORCA2 after-VOR to after-ADV movement.** Numbers: bit-unequal active wet
   U and V cells and maximum absolute changes on rank 0.  CONFIRM that the
   vector operator executes if either unequal count is nonzero.  REFUTE the
   prior imported framing if the counts are not exactly `651 / 16,900` U with
   no active V faces; those values belong only to OVERFLOW and are not expected
   here.
4. **First statement, KEG.** Numbers: bit-unequal active wet U/V cells between
   a literal source-order replay of `dynkeg.f90:117-130` and legoESM's exposed
   production KEG component, both driven by the recorded NEMO stage-2 Kmm
   velocity and static ORCA2 metrics.  Prediction: **0 unequal U and 0 unequal
   V**.  CONFIRM only at zero; REFUTE on one bit.  A one-ULP change to one
   nonzero replay input must make its named row refuse.
5. **Second statement, ZAD / combined closure.** The current admitted stream
   has neither `after_keg` nor the `ww` consumed by ZAD.  Numbers: available
   field census for those two operands.  Prediction: both are absent, so ZAD
   statement fidelity is **UNMEASURED_WITH_SPEC**, not inferred from the
   combined endpoint.  REFUTE if a self-describing admitted stream actually
   carries both.  If absent, write one additions-only acquisition recording
   per rank and per stage-2 call: before KEG, after KEG, after ZAD, Kmm U/V,
   `ww`, effective `wsd`, live U/V thicknesses, metrics, masks, domain origin,
   and the executed flags.  The checker must parse the self-describing header
   and physical EOF; it may not predict a whole-file byte count.
6. **Landing.** No physics statement lands unless the first unequal statement
   is isolated and bit-exact given NEMO operands and the full ORCA2/GYRE/card
   gates pass.  With a missing ZAD operand stream, the frozen disposition is
   `STOPPED_FOR_RECORD` and `ACQUISITION_NEEDED` names the committed `run.sh`.

## Controls and scope

The gate must refuse a payload ULP plant, a field-name/order plant, and a
producer/digest plant.  The acquisition patch must add only write calls and a
writer module, change no compiled arithmetic line, use a fresh target name,
and contain named `REFUSE` exits.  The script is preflighted only; `mpirun` is
not attempted in the sandbox.

ASKED: return to ORCA2's executing vector-invariant path and walk KEG then ZAD.

UNASKED: none.  No configuration, selector, threshold, state, forcing,
stabiliser, sea-ice field, or NEMO arithmetic statement may change.
