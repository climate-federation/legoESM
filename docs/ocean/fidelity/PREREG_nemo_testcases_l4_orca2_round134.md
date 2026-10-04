# Preregistration — ORCA2 round 134 rung-0 step-36 FCT arithmetic walk

Date: 2026-10-04. Base: `6b08fe8c3`. Every number is **independent**:
legoESM starts from the hierarchy rung-0 card's climatological T/S, zero
velocity, and zero sea surface. Decision 52's recorded-entry bridge is not
used. The deck, zero forcing, configuration, carried state, stabilizers,
sea-ice selectors, and `unmeasured_features` remain frozen.

## Compiled source order and measurement

The ORCA2 stage program dispatches stage 3 to FCT at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:529,531-532`. The compiled FCT
routine then executes, in order:

1. the two-step upstream predictor and averaged low-order fluxes
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:495-609`);
2. centred high-order minus low-order antidiffusive fluxes
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:193-197,260-266`);
3. the nonoscillatory limiter
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:316,743-938`);
4. the final corrected-flux divergence and RHS accumulation
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:320-329`).

The committed probe advances the ordinary production-JIT CPU/fp64/x64/libm
trajectory through step 35 and repeats round 133's ordinary step-36 result.
For one otherwise ordinary step it wraps the existing write-only
`return_nemo_trace` interface, records every already-materialized FCT array by
`jax.debug.callback`, and returns the unchanged ordinary divergence pair.
It scores active cell or incident-face support and separately records the six
faces incident to target `(j,i,k)=(86,159,0)`. The ordinary observed result
must be bit-identical to the unobserved result, including NaN payloads.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R134-P1 | The baseline remains finite through step 35 and step 36 first returns non-finite T at `(86,159,0)`. | Exact step/field/index repeats round 133 twice. | Mark **REFUTED** and stop: the baseline or instrument is not reproducible. |
| R134-P2 | The first active-support non-finite FCT row is the pre-limiter antidiffusive flux: the whole-step-entry low-order predictor remains finite while the stage-2 high-order flux overflows. | All source-earlier trace rows have zero non-finites and `anti_pre_{u,v,w}` has at least one. | Mark **REFUTED**, name the earliest source-ordered non-finite row, and do not skip forward. |
| R134-P3 | The target cell's first incident non-finite row is also the pre-limiter antidiffusive flux. | Earlier target-cell and incident-face values are finite; at least one target incident `anti_pre` value is non-finite. | Mark **REFUTED** and retain the distinct global and target boundaries. |
| R134-P4 | Every downstream limiter/final-RHS boundary remains non-finite after the first non-finite row. | Each later grouped row has a nonzero active-support census. | Mark **REFUTED** at the first later finite group; claim no propagation across it. |
| R134-P5 | The write-only observer is passive. | Observed and ordinary step-36 returned states are bit-identical in T/S/u/v/ssh, including NaN bits; duplicate FCT calls agree array-for-array. | Reject the instrument and every scientific number derived from it. |
| R134-P6 | Measurement-only work changes no `packages/` file, public card selector, or sea-ice tuple. | Final diff is preregistration, probe, tests, and receipt only. | Stop without a physics landing. |

## Round bar

This round names the first non-finite arithmetic statement inside stage-3 FCT,
both over active support and at the round-133 target cell. It does not alter
physics, reland round 129's held barotropic chain, add a stabilizer, change a
configuration choice, merge the hierarchy decks, or climb to rung 1. A valid
result requires the committed probe, a source-order plant and a passivity
plant that both fire, focused tests, the citation gate with a shifted-line
plant, the one prescribed ocean-fidelity battery, and a separate read-only
Codex diff-review attempt.

## Instrument refutation and frozen continuation

The first committed measurement at `db301a4ff` rejected itself with
`write-only FCT observer moved the ordinary step`. R134-P5 is therefore
**REFUTED** and retained: requesting `return_nemo_trace=True` inside the full
production step changes the compiled graph at this catastrophically amplified
state, even though the wrapper returns the same two divergence arrays.

The continuation changes only the observer, before taking another scientific
measurement. The production step now calls ordinary FCT exactly once and a
`jax.debug.callback` copies its ten already-present array inputs. After the
ordinary step and its bitwise passivity check complete, the existing FCT trace
is evaluated in a separate JIT call on those exact captured operands. This is
the established developed-state FCT instrument pattern; no operand is
reconstructed.

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R134-P7 | Copying the ten exact FCT input arrays is passive, and the separate JIT trace is deterministic across duplicate T/S calls. | Observed and ordinary returned states are bit-identical in all five fields; duplicate captured operands agree array-for-array. | Reject the continuation instrument and stop without a scientific boundary. |

The continuation measurement at `4f9b6335c` also rejected itself with the
same passivity refusal. R134-P7 is therefore **REFUTED** and retained. No
scientific boundary from either observer is admissible. The existing narrow
stage-transport exposure cannot replace the rejected observers: on this card
it exposes the post-partition explicit vertical transport, while the compiled
two-step upstream predictor also consumes the complementary implicit
transport. Reconstructing that missing operand would change the exact
association being measured.
