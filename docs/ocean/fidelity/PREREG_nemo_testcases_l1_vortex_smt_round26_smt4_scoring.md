# Preregistration — VORTEX_SMT round 26 (lane round 238): admit and score SMT-4

Frozen before reading an SMT-4 trajectory value. Base: `96204ad3d` (round
237 / VORTEX_SMT round 25). Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round238/`; the oracle records
remain the operator-completed, already self-described files under
`phase3/round237/oracle_vortex_smt4/`.

## Compiled program read first

The admitted target is
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3`. Its compiled `ldfdyn.f90:177-185` reads the
reference and card `namdyn_ldf` blocks in that order. Lines 221-278 resolve
the z-partial-step, level Laplacian selection to `np_lap`; lines 311-346 form
the mode-20 coefficient with `zUfac=0.5*rn_Uv` through `ldf_c2d`.
`dynldf.f90:81-90` dispatches `np_lap` to `dynldf_lev_lap`. The resolved
divergence/curl statement is `dynldf_lev.f90:121-140`: it reads the live
partial-cell `e3f`, `e3t(Kbb)`, `e3u/e3v(Kbb)`, and output divisors
`e3u/e3v(Kmm)`, then adds the result to `Krhs`. The compiled stage program
calls this operator before vertical momentum diffusion at
`stprk3_stg.f90:387-405`.

Pre-implementation search found the single shared VORTEX acquisition driver,
the existing self-describing parser, the SMT-0..3 card builder, the shared
geometry gate, the common 50-row trajectory gate, the existing 100-day
scorer, and the production `nemo_div_curl` operator. This round extends those
paths; it does not create a second driver, parser, geometry builder, scorer,
or momentum-diffusion operator.

## Frozen predictions and falsifiers

* **R26-P1 — driver repair.** The shared driver creates the free-space-check
  target directory before calling `df`. A committed regression test deletes
  that line and must fail. This changes no deck, source card, target name, or
  record. Failure to prove the old form red holds the repair.
* **R26-P2 — admission.** The operator-completed ten-step and 100-day reports
  are `ADMITTED`; the plain and instrumented step-10 restarts are byte
  identical; all named groups parse to EOF; header, field-name, and truncation
  plants exit nonzero; both runs contain `STOP 0`; the resolved output prints
  the complete SMT-4 tuple frozen in round 237. Any failed condition refuses
  the record. No NEMO rebuild or rerun is permitted for an admitted record.
* **R26-P3 — one-module card.** The explicit `VORTEX_SMT4_VEC-zps` card differs
  from SMT-3 only by `namdyn_ldf`: div-rot type 0, level Laplacian, coefficient
  mode 20, `rn_Uv=0.1 m/s`, `rn_Lv=10 km`, and `rn_ahm_b=0`. Geometry and the
  initial state are bit-identical to the admitted NEMO record. Any other
  resolved scientific difference or non-bit geometry row refuses the card.
* **R26-P4 — certified trajectory.** Every kt=1 T/S/u/v/ssh row is AT-BAR.
  The first new non-bit row is predicted at kt=2 momentum after `dyn_ldf`;
  pre-`dyn_ldf` stays at the recorded floor and post-`dyn_ldf` owns the new
  increment. A pre-LDF mismatch REFUTES that attribution and moves the walk
  to the earlier boundary. All 50 rows and the 100-day day-1/10/30/60/100
  checkpoints are registered without a direction claim.
* **R26-P5 — first statement.** If P4 confirms the post-LDF boundary, compare
  NEMO's compiled-order `zwf`, `zwt`, U update, and V update operands under
  production JIT from NEMO's recorded stage entry. The first non-bit operand
  or statement is the round's owner. A missing operand produces
  `STOPPED_FOR_RECORD` with a new additive acquisition; it is never inferred.
* **R26-P6 — disposition.** This is a measurement rung. No physics statement
  lands unless one existing, cited shared statement closes the owner and the
  full Decision-43/45/55/59/96 gates fit inside this round. Otherwise status
  is `HELD` with the named first boundary and an exact ORCA2 rung-0 pointer.

Production JIT is authoritative; eager rows are supporting measurements only.
Plants exit nonzero and print `STATUS PLANT-FIRED`. Every citation is mapped
against the compiled SMT-4 target and the shifted-citation plant must exit
nonzero.

## No hidden choices

Decision 93 authorises this exact SMT-4 rung and its mode-20 stand-in for
ORCA2's unavailable file-backed coefficient. Every companion value comes
from ORCA2 rung 0's resolved reference namelist as recorded in round 237.
No timestep, run length, scheme, coefficient, threshold, stabiliser, carried
state, record source, default, or card outside the authorised rung is changed.
If completing the card or walk requires another scientific choice, the round
stops with `DECISION_NEEDED`.
