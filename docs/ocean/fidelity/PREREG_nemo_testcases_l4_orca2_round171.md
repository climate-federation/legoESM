# Preregistration — ORCA2 round 171 kt=8 RHS accumulator walk

Date: 2026-10-08. Frozen base: `cea3bdd83`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round171/`.

Every ORCA2 number in this round is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the result. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple remain
unchanged.

## Frozen record, order and classifier

The only new oracle input is the operator-admitted record under
`orca2_rounds/round170/acquisition/orca2_rung0_rhs8_ranked_10step_np2`.
Its checker must independently re-admit two self-describing rank records,
exactly-once global coverage, all ten HPG/LDF/VOR/KEG/ZAD accumulator fields
and 20 terminal restarts byte-identical to the round-169 baseline.

The compiled rung-0 program accumulates HPG, LDF, VOR, KEG and ZAD in that
order at `ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/stp2d.f90:145-179`.
The measurement will obtain legoESM's already-materialised stage-1 operator
components from its existing private live-operand trace, accumulate them in
that exact order, and compare both faces at every boundary. The observed
after-ZAD arrays must be bit-identical to the completed candidate RHS already
measured in round 170; otherwise this observer is not an admissible
instrument.

For this walk, a boundary is **explosive** only when its active-domain
candidate maximum is at least `1e20 m s^-2` and at least `1e12` times
`max(reference maximum, 1 m s^-2)`. The first transition from a preceding
non-explosive accumulator to an explosive accumulator is the producer
boundary. Non-bit but finite-scale upstream rows remain reported and are not
silently renamed as the explosive owner.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R171-P1 | The round-170 record is admissible without reinterpretation. | Both rank files parse from their own headers, cover 148 x 180 exactly once, expose the registered ten fields, and all 20 terminal restarts are byte-identical to the admitted baseline. | Any header, registry, payload, placement, coverage or restart comparison fails. |
| R171-P2 | The existing live-operand trace is passive and closes on round 170's completed RHS. | The traced and untraced completed kt=1..7 states are array-identical, and the trace's accumulated after-ZAD U/V equals the ordinary completed 3-D RHS bit-for-bit at kt=8. | Any earlier state or completed RHS bit moves. |
| R171-P3 | The first finite-scale-to-explosive U accumulator transition is the VOR call, consistent with the private halo unit first entering the cross-face EEN rotation. | HPG and LDF remain below the frozen threshold, after-VOR crosses it, and later rows retain the carried magnitude. | HPG/LDF is already explosive, VOR remains finite-scale, or no unique transition exists. |
| R171-P4 | NEMO's record itself stays finite-scale at every accumulator boundary. | Every owned recorded value is finite and every recorded active-domain maximum is below the frozen threshold. | Any NEMO accumulator is non-finite or explosive. |
| R171-P5 | This round remains measurement-only and names at most an operator boundary. | No `packages/`, card, selector, carried-state, stabiliser or sea-ice change lands; if the boundary's internal operands are absent, a new fail-closed acquisition is requested rather than attributing below the record. | A physics/configuration change lands, or an internal statement is claimed without a recorded one-variable discriminator. |

## Controls and terminal rule

The gate must reject planted rank placement, source order, observer closure,
explosive classification and passivity violations. A one-ULP known-answer
control must fire on one active accumulator cell. Failed predictions remain
in the receipt.

If the first explosive accumulator is located but the admitted streams do not
split that operator's live operands, write a committed, additions-only,
self-describing per-rank acquisition for precisely that operator at kt=8 and
stop with `ACQUISITION_NEEDED`. Do not infer the internal owner from the
candidate field alone. No stabiliser, clip, bar relaxation, configuration
choice or partial halo-unit landing is permitted.

ASKED choices: continue round 170's compiled-source independent rung-0 walk.
UNASKED choices: empty.

## Instrument correction after the first refused run

The first committed invocation completed kt=1..7, then refused because the
source-ordered reconstruction of the candidate after-ZAD RHS was not
bit-identical to the ordinary candidate RHS. No accumulator score was emitted.
This does **not** by itself show observer perturbation: round 41 already proved
that the same ten raw terms can remain bit-identical while changing only their
addition association. R171-P2 therefore remains frozen and will be recorded
as **REFUTED** if this mismatch reproduces.

The corrected classifier separates the two claims mechanically. The observer
is admissible only if its completed kt=1..7 states, kt=8 barotropic boundary,
and kt=8 completed RHS are all array-identical to the unobserved program. The
source-ordered reconstruction versus that exact ordinary RHS is reported as a
scientific association row; it is no longer mislabeled as observer passivity.
If actual observer passivity moves one bit, the gate still refuses and no
accumulator number is citable. The thresholds and all five predictions above
are unchanged.
