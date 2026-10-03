# Preregistration — ORCA2 round 131 rung-0 step-16 non-finite walk

Date: 2026-10-03. Base: `4a217f925`. Every number is **independent**:
legoESM starts from the hierarchy rung-0 card's climatological T/S, zero
velocity, and zero sea surface. Decision 52's recorded-entry bridge is not
used. The deck, zero forcing, configuration, carried state, stabilizers,
sea-ice selectors, and `unmeasured_features` remain frozen.

## Source order and measurement

NEMO advances RK stages 1, 2, and 3 in order
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:211-227`). Within stage 3 it forms
the tracer transport and accumulator before the implicit tracer solve
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:548-604`), then solves the
vertical tracer system (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:97-107`).

The committed probe advances the ordinary production-JIT CPU/fp64/x64/libm
trajectory through step 15, verifies all five prognostic fields are finite,
then evaluates step 16 through existing write-only hooks. It records the first
non-finite field/index at the stage-1 tracer, stage-2 tracer, stage-3 advection
content, complete pre-implicit content, pre-implicit concentration, and
returned-state boundaries. Diagnostic runs must return the same ordinary
step-16 state as the production reference.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R131-P1 | The ordinary trajectory is finite through step 15 and first becomes non-finite at step 16 in T at `(j,i,k)=(1,49,0)`. | Exact step/field/index repeats round 130. | Mark **REFUTED** and stop: the baseline is not reproducible. |
| R131-P2 | Stage-1 and stage-2 T/S stay finite; stage-3 tracer advection content is the first measured non-finite boundary and contains the target cell. | Earlier tracer boundaries are finite and stage-3 advection is non-finite at `(1,49,0)`. | Mark **REFUTED** and name the earliest observed boundary; do not skip forward. |
| R131-P3 | Complete pre-implicit content and the implicit-solve output inherit the stage-3 non-finite; the vertical solve does not create the first NaN. | Stage-3 advection is already non-finite and all later boundaries carry it. | Mark **REFUTED** if the first non-finite appears only in source association or the solve, then walk that boundary next. |
| R131-P4 | Every diagnostic hook is passive for the prognostic step-16 state. | Hook `state_after` fields equal the independently compiled ordinary result bit-for-bit, including NaN positions and finite bits. | Reject that hook's scientific output. |
| R131-P5 | Measurement-only work changes no `packages/` file and no card selector. | Final diff is probes, tests, preregistration, and receipt only. | Stop without a physics landing. |

## Round bar

This round names the earliest measured source boundary and its magnitude or
non-finite census. It does not reland round 129's held barotropic chain and
does not add a stabilizer. A valid result requires the committed probe, a
plant that moves an earlier-boundary classification and fires, focused tests,
the citation gate with a shifted-line plant, one ocean-fidelity battery, and a
separate read-only Codex diff review attempt.
