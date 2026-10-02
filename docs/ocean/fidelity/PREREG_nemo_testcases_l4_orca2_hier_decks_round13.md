# ORCA2 hierarchy decks round 13 preregistration — Decision 83 rung-6 uniform background

Date: 2026-10-02

Base: `34a6e615f` (`fidelity/nemo-testcases-orca2-decks`).

Scope is NEMO-side only.  Decision 83 fixes `nn_havtb=0` for hierarchy rungs
0 through 6 and makes the shipped equatorial background-diffusivity shape part
of rung 7's TKE settings.  This round preserves the admitted rung-6
`nn_havtb=1` record under `record_havtb1_superseded`, constructs the replacement
rung-6 deck by changing that one assignment, and prepares its independent
240-step, two-rank reacquisition.  Rungs 5 through 1 remain unchanged this
round and follow in order after rung 6 is admitted.  No legoESM package, card,
recipe, physics, configuration, threshold, or carried state changes.

Every run claim is labelled **independent**: NEMO starts from the rung's own
from-rest initialization.

## Frozen oracle reading

The compiled vertical-physics initializer reads `nn_havtb` with `rn_avm0` and
`rn_avt0`, initializes `avtb_2d` uniformly to one, and applies the equatorial
shape only when `nn_havtb == 1`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:149-149`,
`:205-228`).  Decision 83 therefore predicts that `nn_havtb=0` leaves the
background tracer diffusivity uniform while retaining the constant-mixing
closure and its GYRE-card coefficients.  Rung 7 remains at the shipped value
one, so the complete rung-7 to rung-6 boundary now includes the background
shape with TKE's shipped settings.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| HD13-P1 superseded evidence | The admitted rung-6 `nn_havtb=1` record can be preserved without alteration before the replacement run. | Its exact deck, admission, 480 frames, terminal restarts, month products, and SHA inventory validate at their pinned digests after migration to `record_havtb1_superseded`. | Any old product changes, is missing, or fails its historical gate after migration. |
| HD13-P2 one-line Decision-83 delta | The replacement rung-6 deck differs from the admitted superseded deck only by `namzdf.nn_havtb: 1 -> 0`. | Parsed assignments and physical lines report exactly that single change; build, inputs, CPP keys, run protocol, and every other assignment are byte-identical. | Any second assignment, line, build, input, CPP-key, or protocol change appears. |
| HD13-P3 resolved uniform arm | The replacement run resolves constant mixing, `rn_avm0=1.2e-4`, `rn_avt0=1.2e-5`, `nn_avb=0`, and `nn_havtb=0`; TKE stays off. | The runtime output prints those exact values and omits the TKE initializer. | Any value differs, the shaped arm resolves active, or TKE initializes. |
| HD13-P4 rung boundary | Rung 7 remains `nn_havtb=1`; rung 6 becomes `0`, so the rung-7 to rung-6 module delta is exactly constant-mixing-for-TKE plus the Decision-83 shape change. | The complete assignment diff contains only `ln_zdfcst`, `ln_zdftke`, and `nn_havtb`. | Any other hierarchy assignment differs. |
| HD13-P5 replacement record | The unchanged instrumented build produces 480 valid self-describing frames, finite T/U/V/W month products, finite fp64 step-240 ocean restarts, and a complete SHA-256 inventory. | Every header-driven, terminal, month, resolved-output, inventory, and synthetic-violation check passes/fires. | A product is absent, malformed, non-finite, at the wrong step/dtype, or a plant stays green. |
| HD13-P6 acquisition disposition | No `nn_havtb=0` rung-6 record exists before operator handoff. | Preflight passes and the round ends `STOPPED_FOR_RECORD` with a committed fail-closed launcher. | An already-existing admissible replacement record is found. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and landing predicate

The gate must refuse a second deck delta, a missing Decision-83 line, a changed
retained closure coefficient, a changed build/input/CPP/protocol pin, an
unpreserved superseded record, a resolved `nn_havtb` other than zero, malformed
or truncated self-describing payloads, missing/non-finite frames, non-finite or
wrong-step terminal states, and an incomplete SHA-256 inventory.  Every new
plant must fire.

This round lands only the preregistration, replacement deck/manifest, gate,
launcher, tests, and receipt if the one-line preflight and every synthetic
violation pass.  Rung 6 remains **UNMEASURED** until the operator run is
admitted.  No GYRE run is required because no model file changes.
