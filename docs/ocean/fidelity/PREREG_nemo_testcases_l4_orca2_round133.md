# Preregistration — ORCA2 round 133 rung-0 step-36 operand walk

Date: 2026-10-04. Base: `a21a3097b`. Every number is **independent**:
legoESM starts from the hierarchy rung-0 card's climatological T/S, zero
velocity, and zero sea surface. Decision 52's recorded-entry bridge is not
used. The deck, zero forcing, configuration, carried state, stabilizers,
sea-ice selectors, and `unmeasured_features` remain frozen.

## Source order and measurement

NEMO completes the external-mode solve before RK stage 1
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:200-212`). Stage 1 then forms the
barotropic velocity correction from the transport average, live face-depth
reciprocal, and carried barotropic velocity before multiplying the live face
thickness and corrected velocity into `zFu`/`zFv`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:265-284`).

The committed probe advances the ordinary production-JIT CPU/fp64/x64/libm
trajectory through step 35, requires all five prognostic fields to remain
finite, and repeats the ordinary step-36 failure. Existing write-only hooks
then expose stage-1 live U/V face thickness, transport average, corrected
velocity, and final metric transport after the full ordinary step completes.
The first non-finite operand is named in that source order. Every diagnostic
run must return the same ordinary step-36 state bit-for-bit, including NaN
payloads.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R133-P1 | The merged baseline is finite through step 35 and step 36 first returns non-finite T at `(j,i,k)=(86,159,0)`. | Exact step/field/index repeats round 132 twice. | Mark **REFUTED** and stop: the merged baseline is not reproducible. |
| R133-P2 | Stage-1 face thickness and transport average are finite; corrected velocity is the first non-finite exposed operand. | Both earlier operands have zero non-finite values and corrected U or V has at least one. | Mark **REFUTED** and name the earliest exposed non-finite operand; do not skip forward. |
| R133-P3 | The final stage-1 metric transport is non-finite wherever its first non-finite operand is consumed and therefore precedes the returned tracer failure. | The transport census is nonzero and the source-order ledger names the same or a downstream support. | Mark **REFUTED** if the final transport remains finite; continue from the first later measured boundary. |
| R133-P4 | All write-only operand exposures are passive for the ordinary step-36 prognostic result. | Each exposed run's ordinary returned state equals the independent ordinary repeat bit-for-bit. | Reject that exposure and every scientific number derived from it. |
| R133-P5 | Measurement-only work changes no `packages/` file and no card selector. | Final diff is probe, tests, preregistration, and receipt only. | Stop without a physics landing. |

## Round bar

This round names the first non-finite stage-1 transport operand and its full
field/index census. It does not reland round 129's held barotropic chain, add a
stabilizer, or alter any selector. A valid result requires the committed probe,
an earlier-boundary classification plant and a passivity plant that both fire,
focused tests, the citation gate with a shifted-line plant, one prescribed
ocean-fidelity battery, and a separate read-only Codex diff-review attempt.
