# Preregistration: NEMO-testcases L2 GYRE round 112 FCT upstream source-order walk

Date: 2026-09-18. Frozen at incoming lane tip
`a6bb76339dec6a9395cdcad36cea37ac5c2a1b5a`, before any round-112
measurement or numerical edit. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round112/`.

Decision 43 remains the landing rule. The immutable before arm is the landed
round-110 trajectory at
`phase3/round110/candidate/{ladder.json,day_gap.json}`. Round 111 established
that the first directly scored FCT upstream output was non-bit but stopped
because its operands were absent. The operator has now run and admitted the
passive operand record. This round extends the existing round-67 production
capture; it does not create a second stage harness and it does not infer an
anti-diffusive operand from two oracle outputs.

## Immutable trajectory and admitted record

| row | immutable round-110 value |
|---|---:|
| kt2 T max abs | `1.4210854715202004e-14` K |
| kt2 S max abs | `2.1316282072803006e-14` psu |
| kt2 U max abs | `2.7377110452773967e-12` m/s |
| kt2 V max abs | `3.2849219221489645e-12` m/s |
| kt3 T max abs | `8.600419718618468e-7` K |
| kt3 S max abs | `6.979441735666114e-8` psu |
| day-30 T RMS | `6.890484901489568e-5` K |

The admitted record is
`phase3/round111/oracle_fct_writers/oracle_fct_writers_kt00000002_s3.bin`,
SHA-256 `157a5a0ec7cd8607c7c5ee924f788926db5cff5c62e969b853a42f623310505f`,
produced at `a6bb76339dec6a9395cdcad36cea37ac5c2a1b5a`. Its operator-run gate reports
`STATUS PASS`, 34 self-describing fields, 5,849,032 bytes, positive-zero T/S
entry RHS fields, and twin admission `exact=45/65 changed=20 admitted=132`.
These are acquisition facts supplied before the round, not round-112
measurements. Round 112 must independently re-run the admission before using
the values.

## Compiled source order

The record's compiled build enters the active-tracer FCT path and calls the
two-step upstream routine before the limiter
(`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:167-178`).
The routine materializes the half-step coefficient, then writes horizontal
and vertical first-upwind faces
(`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:502-534`). It
differences those faces, builds the Kmm-thickness midpoint tracer, and records
both results
(`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:536-549`). It
then replaces the faces with the arithmetic average of the Kbb and midpoint
upwind fluxes
(`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:572-608`),
differences the averaged faces, divides by the live Kmm thickness, and adds
the result to the existing RHS
(`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:609-623`).

The existing shared WS-RK3 transport builder already materializes NEMO's
metric-bearing horizontal transports before tracer multiplication; its own
source contract says dividing those transports back to metric-free flux and
multiplying again is not bitwise equivalent. The ordinary FCT helper currently
consumes the metric-free transports, forms tracer fluxes, and lets the generic
divergence multiply horizontal metrics later. The walk will score the
metric-bearing equivalent at every recorded boundary rather than comparing
quantities on different staggerings or in different units.

## Frozen predictions and falsifiers

**P1 -- record admission and calibration.** The round-111 gate will reproduce
the exact record digest, field schema, positive-zero RHS entries, `p2dt =
14400 s`, and `STATUS PASS`; its stamp, truncation, and RHS-entry-ULP plants
will each print `STATUS PLANT-FIRED` and exit nonzero. The inherited twin
admission will retain `exact=45/65 changed=20 admitted=132`. REFUTED by any
different digest/schema/count, a normal nonzero exit, or a plant exit of zero;
that stops the round before a numerical edit.

**P2 -- untouched production source-order walk.** With the exact recorded
Kbb tracer, metric-bearing transports, Kbb/Kmm free-surface ratios, masks,
thickness, reciprocal area, and positive-zero RHS injected at the stage-3 FCT
call inside the complete production step, the first non-bit row is predicted
to be the first horizontal upwind-face pair. The mechanism prediction is the
association boundary: NEMO multiplies its already-rounded metric-bearing
transport by the selected upwind tracer, while the current path multiplies a
metric-free transport by tracer before applying the metric. At least one T and
one S u- or v-face word will differ; the later midpoint, averaged faces,
divergence, and divided RHS will retain a nonzero discrepancy. REFUTED if the
first u/v pair is BIT, if an input row is already non-bit, or if W is earlier.
If refuted, the measured first non-bit row, not this mechanism, owns the walk.

The walk must report separate `production step JIT`, `production eager`, and
`isolated closure JIT` tables. Only the complete production-step JIT row may
qualify a candidate. A production-JIT one-ULP change to a finite nonzero exact
input must change the corresponding scored boundary, print
`STATUS PLANT-FIRED`, and exit nonzero. An inert plant or a plant that touches
a zero refutes the instrument.

**P3 -- eligible source candidate.** If P2 is confirmed, the sole candidate
is to route the WS stage's already-existing NEMO metric-bearing transports
through the shared FCT upstream program and preserve the compiled
max/min/product, face-difference, reciprocal-area, thickness-division, mask,
and RHS-add associations. It adds no scheme selector, coefficient, field,
stabilizer, carried state, or restart payload. It is eligible only if every
recorded upstream row -- first u/v/w faces, first divergence, midpoint,
averaged u/v/w faces, final divergence, and divided RHS for both tracers -- is
BIT under the production-step JIT and production eager arms. Any unequal word,
different eager/JIT endpoint, or unrecorded operand REFUTES eligibility.

If P2 instead names a later statement, P3 is replaced only by the literal
compiled statement at that measured boundary; no earlier BIT statement is
changed. If a missing operand prevents proof, the round stops for a passive
new-target record rather than guessing.

**P4 -- local magnitude forecast.** A source-exact upstream candidate is
predicted to reduce the round-111 direct T/S `adv_up1` maxima from
`6.545655547452618e-11` / `5.680531218351438e-12` per second to zero and the
complete local kt3 T/S maxima to at most `6.0e-8` / `7.0e-9`. The preserved
oracle-output kt3 T endpoint `5.760972143775689e-8` is a ceiling discriminator,
not an implementation input. The numerical forecast is REFUTED if either
direct boundary remains non-bit or either kt3 bound is exceeded.

**P5 -- Decision-43 trajectory.** Reached only after P3 closes. The candidate
must strictly reduce day-30 T RMS below `6.890484901489568e-5 K`, keep the
first-over-bar boundary no earlier than kt2, keep every kt1 AT-BAR row at bar,
and register every moved row including regressions. The forecast band is
`1e-6`--`6e-5 K`; outside it the numerical prediction is REFUTED, while an
equal or larger day-30 value rejects the landing. Predicted headline values
are kt2 U/V unchanged at `2.7377110452773967e-12` /
`3.2849219221489645e-12`, kt3 T/S no larger than the P4 bounds, and no kt1
class change.

**P6 -- shared-card risk.** The horizontal/vertical FCT upstream program is
shared by the certified LOCK_EXCHANGE and OVERFLOW WS-RK3 cards, so both tank
gates must be measured before and after any candidate; they are predicted
bit-identical because their metric-bearing transports follow the same source
association. DINO's two certified cards use the Euler/MLF tracer lane rather
than this WS stage route and are predicted not to execute the candidate; that
claim is resolved from their instantiated cards and executable-path tests,
not a hard-coded card list. If either DINO card executes, its cheapest
committed numerical gate becomes mandatory. ORCA2 remains
**UNMEASURED-WITH-SPEC**: resolve its tracer integrator and FCT dispatch, record
both FCT writes and every upstream operand for kt1--10, replay the production
fp64 JIT closure, and compare the next consumed T/S state.

## Measurement order and stopping rules

1. Commit this preregistration before creating round-112 evidence or parsing
   the scientific payload.
2. Re-run the normal record admission and all acquisition plants. Extend the
   existing round-67 production capture with the admitted reader and the
   compiled-order rows; do not build a parallel FCT harness.
3. Run untouched production-step JIT, production eager, isolated-JIT, and the
   production plant. Name the first non-bit statement from the table.
4. Implement only that statement's source transcription. Re-run the full walk;
   proceed only if every required upstream row is BIT in production JIT and
   eager.
5. If eligible, run the complete kt1--10 ladder, days 1--30, and the shared
   tank/card gates against the immutable round-110 arm. Apply all five
   Decision-43 conditions and register every moved row.
6. Run a separate read-only Codex refutation pass, the receipt citation gate
   and a shifted-citation plant, focused tests, and the complete ocean fidelity
   and unit trees with twelve workers. Quote every terminal pytest summary and
   diff failures against the frozen pre-existing list.

## Scope

CPU only, fp64, `JAX_ENABLE_X64=1`. NEMO source is read-only; neither
`makenemo` nor `mpirun` is run. No configuration/default, coefficient,
stabilizer, carried state, restart schema, year harness, reconciliation gate,
freshwater pair, #1484 guard, or immutable artifact is changed. No oracle
output enters the independent implementation. No configuration choice is
made.
