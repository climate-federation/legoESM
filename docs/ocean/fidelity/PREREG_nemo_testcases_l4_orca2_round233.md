# ORCA2 round 233 preregistration — atomic stage-transport geometry unit

Date: 2026-10-10. Frozen base: `71c8b189f`. This round executes the round-232
OPEN and operator note B58 addendum 31. It first measures, privately and
atomically with the complete round-217 vector unit, NEMO's raw reference U/V
face thicknesses and its associated U/V/T/F masks. It does not choose a new
configuration, carried state, stabiliser, sea-ice selector, or
`unmeasured_features` entry.

Every number is reported separately as **independent** or **given NEMO's
entry**. Production JIT, CPU, fp64 and scalar-libm are mandatory. Existing
passive completed-stage states remain the only executable observer; no new
in-executable intermediate observer is permitted.

## Frozen source unit and implementation scope

The executing ORCA2 build reads `e3u_0/e3v_0` from the domain at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domzgr.f90:184-188`, and the
substitution header identifies those stored three-dimensional arrays at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/inc/domzgr_substitute.h90:114-121`. NEMO builds
`umask/vmask/fmask` from neighbouring `tmask` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:206-214`, applies the
one U/V/F mask association with sign `+1` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:228-232`, and the
executing T-pivot V-fold rule is
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:973-981`. The stage-1
transport consumes the live raw operands at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-285`.

The candidate must reuse `nemo_qco_resolved_mesh_operands`; a second face-
thickness builder is forbidden. The private arm routes the WS-RK3 stage
geometry through that builder and takes the raw mask bundle already attached
to the ORCA2 card. It is scored only together with the round-217 unit: exact
slow-V/raw-`ssvmask` pair, seven-array association, raw reference face depths,
unmasked/materialised V transport. A constituent never lands alone.

Before trajectory scoring, the gate compares raw versus current `e3u_0`,
`e3v_0`, `tmask`, `umask`, `vmask`, and `fmask` against the card's admitted
NEMO bundle. It must also replay the round-232 five-operand table on NEMO-live
support. One-bit plants for face thickness and masks, a reconstructed-source
plant, a missing-unit-member plant, and a claim-label plant must refuse.

## Frozen predictions and falsifiers

- **R233-P1 — instrument and resolved geometry.** Prediction: traced completed
  states remain array-identical to ordinary states under each label; the raw
  `e3u_0/e3v_0` and associated T/U/V/F masks are each bit-exact against the
  admitted NEMO bundle on their registered extents. Any changed state bit,
  missing raw array, wrong extent, non-firing plant, dirty/stale worktree,
  wrong backend/dtype, or a second geometry implementation REFUTES the
  instrument and stops the round.
- **R233-P2 — five-operand closure.** Prediction: at kt=1 stage 1, live `e3v`
  and `vmask` fall from 13/226,637 and 668/399,600 unequal to zero. If the
  geometry defects are upstream owners, `vn_adv` falls from 8,589 unequal to
  at most 35, and the complete candidate replays NEMO `zFv` bit-for-bit under
  both labels. `vn_adv` remaining of order 8,589, or a nonzero `e3v/vmask`
  census, REFUTES root ownership; the candidate remains private and the round
  is HELD with `DECISION_NEEDED` between landing the independently exact
  geometry unit and retaining the full cancelling unit.
- **R233-P3 — endpoint signature.** Conditional on P2: OMT-4 kt=1 stage-1
  fold-band S falls from 3.283356343139289 PSU to at most 0.0012 PSU and T
  from 0.17733430832081432 K to at most 0.0014 K under both labels; the
  complete unit finishes kt=8 and the full ten-step ladder. Failure of any
  endpoint REFUTES the landing premise.
- **R233-P4 — rung-0 transfer.** Conditional on P3: independent rung 0 also
  completes kt=8 and its ten-step ladder; the previous live-W-thickness
  refusal disappears. The first-over-bar row moves toward NEMO or is
  unchanged, a strict majority of RMS-moved rows move toward, no exact row is
  lost, and the certified SSH maximum does not worsen. A remaining refusal,
  majority-away result, earlier first debt, exact-row loss, or worse SSH
  maximum holds the unit.
- **R233-P5 — landing gates.** Only if P1-P4 confirm, score OMT-1 through
  OMT-4, OMT-4's independent 96-step month, rung 0's independent ten-step
  ladder and month, rung 10 given NEMO's entry, GYRE's certified ladder/year,
  DINO month, lock-exchange, overflow and generic/card gates. Decision 96 is
  applied with RMS-toward/away/equal and bit-moved-but-score-equal rows listed
  separately. Any gate red outside registered two-ULP near-zero cells stops
  the landing.
- **R233-P6 — shared implementation safety.** A package landing must preserve
  JIT, autodiff and pytree purity; the cumulative citation map is re-anchored
  for every edited model file, the default and round receipt citation gates
  pass with a firing rigid-shift plant, and focused plus ocean-fidelity tests
  follow the one-battery-at-a-time rule. GYRE movement is judged by its
  standing gate, not assumed absent.

## Outcomes

`LANDED` requires P1-P6 and the complete cited unit. `HELD` means a falsifier
names the next cancelling owner with all failed predictions retained.
`STOPPED_FOR_RECORD` is reserved for a genuinely absent operand after all
existing admitted records are checked. No new NEMO acquisition is presently
requested.

ASKED choices: Decisions 103, 109, 113, and standing Decision 96. UNASKED
choices: empty.
