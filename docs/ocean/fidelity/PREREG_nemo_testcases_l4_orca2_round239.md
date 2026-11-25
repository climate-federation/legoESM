# ORCA2 round 239 preregistration — completed V-RHS fold pair and atomic unit

Date: 2026-10-11. Frozen base: `1c0b948a3`. Decisions 115 and 114 are
landed. This round executes round 238's OPEN and operator note B58 addendum
39. No deck, carried state, stabiliser, sea-ice selector, or
`unmeasured_features` entry changes.

Every ORCA2 result is retained separately as **independent** or **given
NEMO's entry**. Execution is CPU, production JIT, fp64/libm. Only passive
completed-state traces and offline replay may be used; no new in-executable
observer is permitted.

## Frozen source unit and order

The executing OMT-4 build forms the completed vector-form V forcing from live
`e3v_3d`, the completed three-dimensional V RHS, `vmask`, and `r1_hv_0` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90:194-200`, then applies
drag and wind at `:218-230`. It copies `Ve_rhs` into `zv_frc` and subtracts
the two-dimensional Coriolis trend through raw `ssvmask` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:289-328`. The
vector update consumes that same raw V mask at `dynspg_ts.f90:669-682`; the
seven-array association is at `:747-756`; and the unmasked materialised V
transport and continuity consumer are at `:533-560`.

Round 238 made live `e3v` and `vmask` exact. The remaining measured boundary
is completed V RHS `668/1,050` unequal on the 35 fold-halo faces, `vn_adv`
`919/8,589` unequal, and `zFv` `19,516/226,637` unequal. Round 239 first
applies only the cited `Ve_rhs` fold pair to the completed V RHS. It then
scores the complete round-217 unit atomically: that pair, raw vector mask,
seven-array association, and unmasked/materialised V transport. No constituent
may land by itself.

## Frozen predictions and falsifiers

- **R239-P1 — admission/passivity.** The inherited slow-forcing, substep,
  stage and transport records parse from their self-describing headers, twin
  defined payloads are exact, and traced completed states equal ordinary
  states bit-for-bit under both labels. A malformed/missing stream, twin
  difference, moved state bit, wrong dtype/backend, stale tree, or a
  non-firing record/passivity plant REFUTES the instrument.
- **R239-P2 — Ve_rhs discrimination.** With landed raw `e3v`, the Ve_rhs fold
  pair alone is predicted to close the completed V RHS on the 35 faces:
  `668/1,050 -> 0/1,050` unequal under both labels. If it does not, retain the
  exact count/max and emit the source-ordered HPG, EEN vorticity, KEG, ZAD,
  LDF and drag operand table; the first unequal operator is the named OPEN and
  no production landing is attempted.
- **R239-P3 — atomic endpoint.** With the complete unit, kt=1 stage-1 fold-band
  S is predicted to fall by orders of magnitude from `3.283356343139289 PSU`
  but remain above `0.0012 PSU` while the upstream `vn_adv` debt remains.
  T, the five-operand `zFv` table, and the first live-thickness boundary are
  recorded under both labels. Geometry rows must stay exact; only `zvb` may
  remain nonzero in the five-operand table. The unit must finish kt=8 or move
  the refusal later than kt=8. A geometry regression or earlier refusal
  REFUTES the candidate.
- **R239-P4 — conditional closure.** If P2 reaches zero, the indivisible unit
  is predicted to reduce `vn_adv 919 -> <=35`, fold-band S to `<=0.0012 PSU`,
  and complete kt=8. Failure of any threshold HOLDS the unit and names the
  next recorded boundary; it may not land on trajectory averages alone.
- **R239-P5 — Decision-96 census.** Conditional on P2/P4, score both OMT-4
  labels, independent rung 0, and given-entry rung 10. Landing requires a
  strict majority of RMS-moved rows toward NEMO, first-over-bar toward or
  unchanged, zero exact-row losses, and no worse certified SSH maximum.
  Report toward/away/score-equal separately and register every bit-touched
  row. OMT-4's independent 96-step month must complete; rung 0 must not refuse
  before its registered step-100 boundary.
- **R239-P6 — shared gates.** A qualifying implementation must pass GYRE's
  certified ladder/year, DINO month, DINO/LOCK/OVERFLOW/VORTEX/tank/generic
  card gates, focused tests, the once-only ocean-fidelity battery, default and
  round citation gates with a firing rigid-shift plant, and independent
  read-only review. JIT, autodiff and pytree purity remain intact. Any
  non-Decision-96 red blocks landing.

## Outcomes

`LANDED` requires P1-P6 and the complete cited unit. `HELD` retains every
failed prediction and names the first remaining statement. No NEMO acquisition
is expected; existing admitted records cover the frozen measurements.

ASKED choices: Decisions 96, 103, 109, 113, 114 and 115. UNASKED choices:
empty.
