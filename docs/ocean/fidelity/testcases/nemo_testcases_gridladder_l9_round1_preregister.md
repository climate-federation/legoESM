# Grid ladder rung L9 — round 1 preregistration (gridladder_r57, Codex)

Date: 2026-10-23

Starting commit: `bd83f3fd1eeb`

Decision 151 fixes the candidate: L9 is certified L8 plus the certified L3
eddy-induced velocity, with the complete five-key ORCA2 `namtra_eiv` group.
No mixed-layer eddy or other configuration switch is added.  Every existing
card and every library default remains unchanged.

NEMO is operator-only.  This round designs and builds L9 and its acquisition;
it does not build or run NEMO, admit a record, score a trajectory, or certify
L9.

## Oracle statement order, frozen before implementation

ORCA2 computes the standard isoneutral slope once from the before tracer state,
then refreshes the time-varying eddy coefficient before the RK stages
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:184-194`). During stage 3 the live
non-Shuman branch calls `tra_adv_trp`, then computes selected BBL coefficients
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:555`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:560`).

Inside that call, after the bottom transport slots are zeroed, NEMO calls the
eddy-induced-transport routine only at stage 3
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:248-265`).  The running RK3 overload
builds U/V bolus streamfunctions and adds their vertical differences to the
ordinary U/V transports
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:938-955`).  NEMO then recomputes
vertical velocity from the augmented U/V pair and materializes W transport
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:267-312`).

Only after those transports exist does the tracer program clear `Krhs`, pass
the augmented U/V/W transports to stage-3 FCT, and add the remaining terms
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:600-645`).  The remaining exact
order is QSR, isoneutral LDF, geothermal BBC, BBL, damping, then implicit ZDF
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:700-749`).  Thus the full L9
order is:

```text
ordinary U/V transport -> EIV U/V addition -> recomputed W transport ->
zero Krhs -> FCT advection -> SBC -> QSR -> isoneutral LDF -> geothermal BBC
-> diffusive BBL -> interior damping -> implicit ZDF
```

ORCA2 selects diffusive BBL option 1 and advective BBL option 0
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:265-272`).  `tra_bbl` therefore calls the
diffusive routine, skips the advective routine, and writes its bottom-cell
increment into the already accumulated `Krhs`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:182-202`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90:249-264`).  It neither modifies the
earlier EIV-augmented transports nor reorders the LDF/BBC/DMP writes.

## Frozen candidate and controls

The L9 selector is default false and legal only with the complete certified L8
selection plus the complete certified L3 Treguier-21, face-averaged,
bolus-through-FCT selection.  L8 without L3 remains legal.  L8 plus any partial
or differently configured eddy-induced velocity remains refused.  Decision
147's refusal of staged EIV without the stage-3 `Krhs` route remains unchanged.

One-step x64 controls use INEXACT binary64 operands.  They pin the EIV U/V
addition before the existing L8 `Krhs` words and pin all seven L8 post-write
boundaries.  Moving the EIV addition, or swapping any adjacent Krhs call, must
change the first displaced boundary bit for bit and fail.  A production-route
control also requires the live L9-only EIV addition to execute before the live
L8 finalizer and requires the exact objects produced by EIV, BBC, BBL and DMP
to reach their registered positions.

## Predictions and falsifiers

- **P1 — exact card union.** L9 differs from L8 only by ORCA2's complete L3
  five-key EIV group and corresponding certified model selectors.  Any other
  deck/config difference, or any change to an existing card, refutes P1.
- **P2 — exact source order.** Every INEXACT position pin passes under JIT and
  every moved-call plant fails at its first displaced boundary.  A move that
  survives refutes P2.
- **P3 — guarded composition.** L8 alone and L3 alone still construct.  Only
  L9 may construct with both selections.  Any partial EIV+L8 composition that
  constructs, or any formerly legal combination that becomes illegal, refutes
  P3.
- **P4 — forward and gradient preservation.** L9 adds no division, square root,
  or forward branch beyond the already certified L3/L8 paths.  Its new ordered
  addition has finite gradients and identical forward bits under gradient and
  ordinary evaluation.  Any new non-finite gradient or changed forward bit
  refutes P4.
- **P5 — acquisition preflight.** Read-only preflight proves all seven standing
  checklist items with a refusing plant each, syntax-checks every patched NEMO
  file, runs whitespace-tolerant selection checks on a real committed
  `ocean.output` excerpt and a stamped L9 deck, compiles a gfortran writer for
  the exact Python reader sequence, calls `check_acting` only on the smoke, and
  removes scratch through an `EXIT` trap.  Any surviving plant or created NEMO
  target refutes P5.

## Acquisition contract

The operator-only `--run` mode requires an explicit acknowledgement and runs a
two-step acting smoke, a writer-free boundary, and three independently staged
twins to one even endpoint no greater than ten steps.  The observers write full
`jpi*jpj*jpk` arrays only for NEMO-defined cells, canonicalize all excluded
cells to positive zero, and sit outside any branch that is false without XIOS.
Admission requires complete per-step L8 boundary records, an acting EIV
transport witness, exact twin agreement on the defined domain, and full-depth
restarts at every scored step.

## Review and choices

- codex review: operator
- independent review: operator (Claude)
- ASKED: Decision 151's L8+L3 composition, exact-order controls, committed
  acquisition, read-only preflight, and stop at the record boundary.
- UNASKED: none.
