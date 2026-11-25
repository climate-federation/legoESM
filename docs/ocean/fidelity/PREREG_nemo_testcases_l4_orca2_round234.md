# ORCA2 round 234 preregistration — complete external-mode association

Date: 2026-10-10. Frozen base: `afb2daf6b`. This round executes the external-
mode walk ordered by operator note B58 addendum 32. Decision 114 is unanswered,
so the independently exact raw face-thickness/mask geometry from round 233 is
not landed here. The candidate is measured privately and atomically with the
complete round-217 vector unit; no configuration, carried-state, stabiliser,
sea-ice selector, or `unmeasured_features` choice changes.

Every number is reported separately as **independent** or **given NEMO's
entry**. Production JIT, CPU, fp64 and scalar-libm are mandatory. Existing
passive completed-stage/substep states are the only executable observer; no
new in-executable intermediate observer is permitted.

## Frozen source statement and candidate

After every split-explicit substep the executing ORCA2 build updates face
depths and reciprocals at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:738-744`, then makes
one seven-array association at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:747-756`: `ua_e` and
`va_e` use sign -1; `hu_e`, `hv_e`, `hur_e`, `hvr_e`, and `ssha_e` use sign
+1. The executing T-pivot T and U rules are
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:945-972`.

The present compact helper associates U/V velocity and the three V arrays,
but gives `hu_e/hur_e` only their cyclic closure and returns `ssha_e`
unchanged. The private candidate completes that same helper with the U-fold
rule for `hu_e/hur_e`, the T-fold rule for `ssha_e`, and carries the associated
SSH into the next substep. It does not add a second association implementation.

## Frozen predictions and falsifiers

- **R234-P1 — instrument/passivity.** Prediction: ordinary and traced completed
  states are array-identical under both labels, the backend is CPU/JIT/fp64,
  and one-bit U-depth, SSH, label-coverage, and false-owner plants refuse. Any
  changed ordinary state bit or non-firing plant stops attribution.
- **R234-P2 — literal association.** Given NEMO's recorded pre-association
  operands, all seven candidate post-association arrays are bit-exact against
  the admitted NEMO substep record. Any unequal array refutes this
  transcription and the model candidate is removed.
- **R234-P3 — external-mode owner.** With the complete association inside the
  atomic round-217 unit, `vn_adv` falls from 8,589 / 8,589 unequal to at most
  35 under both labels. If it does, the association owns the independent
  external-mode debt. If `vn_adv` remains 8,589-scale, the statement is a
  measured null; it is retained as exact debt but does not land, and the walk
  advances to a per-substep operand table rather than trying a second blind
  statement.
- **R234-P4 — exposed tracer endpoint.** Conditional on P3, OMT-4 kt=1 stage-1
  fold-band S falls from 3.283356343139289 PSU to at most 0.0012 PSU and T from
  0.17733430832081432 K to at most 0.0014 K under both labels, and the atomic
  unit completes kt=8. Failure refutes the landing premise.
- **R234-P5 — landing gates.** Only if P2-P4 confirm, score OMT-1 through
  OMT-4, OMT-4's independent month, independent rung 0 ladder/month, rung 10,
  GYRE, DINO, lock-exchange, overflow and generic/card gates under Decision 96.
  Majority-away RMS rows, earlier first debt, an exact-row loss, worse SSH
  maximum, or any external red holds the unit. A package landing also requires
  GYRE's shared gate and citation-map re-anchoring.

## Outcomes

`LANDED` requires P1-P5. `HELD` means a falsifier or null names the next
source-ordered operand with the temporary model candidate removed.
`STOPPED_FOR_RECORD` is reserved for an operand absent from all admitted
records. Failed predictions remain in the receipt.

ASKED choices: Decisions 103, 109, 113, standing Decision 96, and pending
Decision 114. UNASKED choices: empty.
