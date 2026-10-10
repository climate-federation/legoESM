# ORCA2 round 232 preregistration — stage-1 V-transport operands

Date: 2026-10-10. Frozen base: `0838ebc78`. This round is scoped to the
rank-zero operand split ordered by the round-231 OPEN and operator note B58
addendum 30. It does not change a card, deck, carried state, stabiliser,
sea-ice selector, or the ORCA2 card's `unmeasured_features` tuple.

All measurements are reported separately as **independent** and **given
NEMO's entry**. Production JIT, CPU, fp64 and scalar-libm are mandatory. The
candidate values come from the already-passive round-228/229 live-stage trace;
the gate must reproduce its completed ordinary state bit-for-bit before any
operand is read. No in-executable observer is permitted.

## Frozen source order and instrument

The compiled ORCA2 statement first constructs the V correction `zvb` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279`, then
constructs `zFv` in the operand order `e1v`, live `e3v(Kmm)`, `vv(Kmm)`,
`zvb`, `vmask` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`.
The admitted rank-zero
`oracle_rkstage1_transport_operands_kt00000001.bin` already records those five
operands and the correction inputs `vn_adv`, `r1_hv`, and `vv_b`; no new NEMO
record is requested first.

The candidate live thickness and corrected velocity are taken from the
existing passive stage geometry. The raw stage velocity, live reciprocal and
`zvb` are replayed offline by the already-shared literal helpers on the same
passive entry/barotropic state. The replay is admissible only if its corrected
V velocity and final `zFv` reproduce the candidate trace at zero unequal bits
on the scored rank-zero slab. A one-bit perturbation in each operand scorer and
one correction-replay operand must make the gate refuse.

## Frozen predictions and falsifiers

- **R232-P1 — instrument/passivity.** Prediction: independent and given-entry
  traced states equal their ordinary states in every state slot, and offline
  replay reproduces candidate corrected V and `zFv` bit-for-bit on rank zero.
  Any changed state bit, replay residual, wrong dtype/backend, dirty/stale
  worktree, missing operand, or non-firing plant REFUTES the instrument and
  stops attribution.
- **R232-P2 — five-operand owner.** Prediction: `e1v`, live `e3v(Kmm)`,
  `vv(Kmm)`, and `vmask` are bit-exact against NEMO; `zvb` is the first
  unequal operand in the compiled statement order under both labels. Any
  earlier unequal operand REFUTES and names that earlier statement. All five
  exact REFUTES the current record's sufficiency and requires a rank-complete
  operand record before attribution.
- **R232-P3 — correction split.** Conditional on R232-P2: compare `vn_adv`,
  `r1_hv_0/(1+r3v(Kmm))`, and `vv_b(Kmm)` in the compiled correction order.
  Prediction: the first unequal input is `vn_adv`; substituting NEMO's recorded
  `vn_adv` alone makes candidate `zvb` and rank-zero `zFv` bit-exact. An
  earlier reciprocal debt or a carried-`vv_b` debt REFUTES and names that
  operand instead; no single substitution closing both quantities REFUTES a
  single-statement landing.
- **R232-P4 — endpoint signature.** Conditional on a source-cited single
  operand closing the rank-zero boundary, score it atomically with the complete
  round-217 unit. Prediction: OMT-4 kt=1 stage-1 fold-band S falls from
  3.283356343139289 PSU to at most 0.0012 PSU, T falls from
  0.17733430832081432 K to at most 0.0014 K, and the unit completes kt=8.
  Failure of any endpoint REFUTES the landing premise; the candidate remains
  private and the round is HELD.
- **R232-P5 — landing gates.** If R232-P4 confirms, both OMT-4 ladders, the
  OMT-4 independent month, independent rung-0 kt=1..10 plus month, rung-10,
  GYRE, DINO and tanks are scored under Decision 96, with every moved row
  registered. Majority-away RMS rows, an earlier first debt, an exact-row
  loss, a worse certified SSH maximum, or any external gate red holds the
  change. If P2/P3 cannot isolate one statement, no model change or trajectory
  gate is run.

## Outcomes

`LANDED` requires P1-P5 and a cited single-statement transcription. `HELD`
means the comparison names the next operand/record without a qualifying model
change. `STOPPED_FOR_RECORD` is reserved for the all-five-exact or missing
rank-complete case. Failed predictions remain in the receipt as REFUTED.

ASKED choices: Decisions 103, 109, 113, and standing Decision 96. UNASKED
choices: empty.
