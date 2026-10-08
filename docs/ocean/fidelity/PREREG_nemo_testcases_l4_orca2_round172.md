# Preregistration — ORCA2 round 172 passive kt=8 RHS replay

Date: 2026-10-08. Frozen base: `aa7c7a03b`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round172/`.

Every ORCA2 number in this round is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the result. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple remain
unchanged.

## Frozen record and compiled order

The oracle input is the admitted rank-complete record under
`orca2_rounds/round170/acquisition/orca2_rung0_rhs8_ranked_10step_np2`.
Its checker must again parse the self-describing headers, cover 148 x 180
exactly once, expose all ten HPG/LDF/VOR/KEG/ZAD accumulator fields and prove
all 20 terminal restarts byte-identical to the round-169 baseline.

The exact compiled rung-0 program overwrites the accumulator with HPG, then
adds LDF, VOR, KEG and ZAD in that order at
`ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/stp2d.f90:145-179`. The replay will
use the existing production live-operand trace's already-materialised raw
HPG, LDF, vorticity, KEG and ZAD terms and add them offline in precisely that
order. It will not add a callback, returned component or observer to the kt=8
advancing executable.

The candidate kt=8 entry is produced by the unchanged complete private arm
from rounds 163--171. The existing early-return barotropic trace supplies its
completed three-dimensional RHS without reaching the later stage-3 refusal.
A separate offline tendency evaluation may expose the five raw terms only if
its completed U/V RHS is bit-identical both with and without component output
and to that passive trace. This is an instrument prerequisite, not a physics
result.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R172-P1 | The round-170 record remains admissible without reinterpretation. | Both rank files parse from their own headers, cover 148 x 180 exactly once, expose the registered ten fields, and all 20 terminal restarts are byte-identical to the admitted baseline. | Any header, registry, payload, placement, coverage or restart comparison fails. |
| R172-P2 | The existing live-operand trace is passive under the complete arm through kt=7. | At each kt=1..7, every prognostic field in `trace.state_after` is array-identical to the unchanged ordinary arm, and its after-ZAD U/V equals that step's ordinary completed RHS bit-for-bit. | Any state or completed-RHS bit moves. |
| R172-P3 | The offline kt=8 decomposition is admissible. | Its no-component total equals the passive barotropic trace's completed U/V RHS bit-for-bit, its component-return total equals the no-component total, and its five raw terms reproduce that same total in the production association. | Any closure bit moves; no scientific accumulator score is emitted. |
| R172-P4 | The first finite-scale-to-explosive U transition is VOR, as frozen in round 171. | HPG and post-LDF remain below `1e20 m s^-2`, post-VOR crosses it and is at least `1e12` times `max(NEMO max, 1)`, and later rows retain the class. | HPG or LDF is already explosive, VOR stays finite-scale, or no unique transition exists. |
| R172-P5 | This remains a measurement-only attribution round. | No `packages/`, card, selector, carried-state, stabiliser or sea-ice change lands; the gate reports both the first non-bit boundary and first explosive transition, and requests the first missing internal operand record rather than attributing below it. | A physics/configuration change lands, a stabiliser is introduced, or an internal statement is claimed without a recorded one-variable discriminator. |

The first non-bit accumulator is selected independently of R172-P4: HPG,
LDF, VOR, KEG and ZAD in compiled order, U before V at each boundary. A finite
upstream non-bit row is not hidden by the explosive classifier. Failed
predictions remain in the receipt.

## Controls and terminal rule

The gate must reject planted record placement, replay source order,
trace-state passivity, completed-RHS closure, first-boundary selection and
explosive classification changes. A one-ULP active-cell plant must move the
selected first non-bit boundary.

If the first non-bit operator's NEMO operands are absent, write a committed,
additions-only, self-describing, per-rank kt=8 acquisition for exactly that
operator and stop with `ACQUISITION_NEEDED`. Every artifact the launcher reads
must be committed and repo-relative; the checker parses names and payload
lengths from the record header and proves kt=10 restarts byte-identical to the
round-170 producer. No stabiliser, bar relaxation, configuration choice or
partial halo-unit landing is permitted.

ASKED choices: continue round 171's independent rung-0 compiled-source walk.
UNASKED choices: empty.

## Instrument correction after the first refused run

The first committed run completed kt=1, then refused at kt=2. The existing
live-operand trace returned a state array-identical to the ordinary complete
arm in all five prognostic fields, but its completed U/V RHS was not
array-identical to the separately compiled early-return barotropic trace.
Thus R172-P2's extra cross-graph RHS clause is **REFUTED**. No accumulator was
scored. This reproduces round 171's failure mode: changing a compiled return
graph moves an internal boundary even when the traced full-step state remains
bit-identical.

The user-specified passivity predicate in B57 addendum 2 is kt=1..7 traced
versus untraced **state** identity. The corrected bridge therefore keeps that
predicate and adds only same-graph closure: in each live trace its own
`after_ldf` boundary must equal its own published completed RHS bit-for-bit.
A separate offline component call must match every raw term and the completed
boundary of that same live trace at kt=1..7. Only after those two checks pass
may the identical offline call be evaluated on the kt=8 entry and accumulated
in NEMO source order. This never compares an internal array across two return
graphs.

The corrected predictions are additive; the failed originals remain above:

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R172-P2a | The existing live-operand trace is passive under B57's stated predicate and closes internally. | At kt=1..7 every returned state field equals the ordinary arm bit-for-bit, and the trace's own `after_ldf` U/V equals its own completed RHS. | Any state or same-graph closure bit moves. |
| R172-P3a | A standalone component evaluation is a bit-exact offline bridge to the existing trace. | At kt=1..7 all five raw U/V terms, `after_ldf` U/V and completed U/V equal the live trace's values; at kt=8 its own `after_ldf` equals its own completed total. | Any bridge or internal-closure bit moves; no scientific accumulator is scored. |

R172-P4, both explosive thresholds, source order, first-non-bit selection and
all terminal rules are unchanged.
