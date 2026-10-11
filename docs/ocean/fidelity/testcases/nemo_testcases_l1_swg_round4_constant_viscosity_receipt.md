# NEMO SWG level-1 round 4 — constant stage-3 viscosity

Date: 2026-10-11. Base: `e3817a8773bc48ad1a934113237a0ad8b9f0ac8c`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round4`.

## Preregistration

This section is frozen before any round-4 implementation or measurement. The
operator's independent round-3b review reproduced the committed ladder and
identified three blockers. This round changes only the stage-3 viscosity
coefficient branch, the ineffective exact-zero control, and two duplicated
configured values named by that review.

Frozen predictions and falsifiers:

1. The SWG card will explicitly select the shared constant-coefficient
   `nn_ahm_ijk_t=0` dispatch. The coefficient will be formed from the card's
   resolved `rn_Uv` and `rn_Lv`, and the stage-3 kernel will consume that
   selection. NEMO forms `zUfac = 1/2*rn_Uv`, then `zah0 = zUfac*rn_Lv`, and
   assigns `zah0` directly to `ahmt` and `ahmf` at compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90:311-322`; stage 3 calls this
   coefficient through `dyn_ldf` at compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:317`. A metric-derived
   coefficient, an implicit/default selector, or an unknown selector that does
   not raise **REFUTES** the transcription.
2. Given-entry `kt=2, stage=3` U is predicted bit-identical after the constant
   branch replaces the metric branch. Any unequal wet U cell **REFUTES** that
   prediction. If it clears, the next first non-bit source statement will be
   named without attributing ownership beyond what the record discriminates.
3. The given-entry and chained ladders will score every existing transition as
   before. No previously bit-identical row may move, and neither first-over-bar
   boundary may move earlier. Any such move **REFUTES** Rule 12 eligibility.
4. The exact-zero control will perturb a wet, consumed zero-valued U input by
   the smallest normal binary64 magnitude and score the changed production
   output. The output must differ bit-for-bit, the control row must be DEBT,
   and the plant command must exit nonzero. An unchanged output or a green
   control **REFUTES** the repaired instrument.
5. SWG geometry will derive the F-point base thickness from its configured
   one-layer thickness, and the independent wind will derive its meridional
   scale from the resolved `rn_domsiz_m`. Remaining literal copies of `500.0`
   in the `hf_0` construction or `2_000_000.0` in that wind expression
   **REFUTE** the configuration-value repair.
6. The six certified non-SWG card hashes must remain byte-identical, and the
   focused SWG, dispatch, geometry, citation, and plant gates must pass. A
   moved certified hash, a citation plant that exits zero, or a non-binding
   numerical plant **REFUTES** landing.

No configuration or carried-state choice is introduced. The coefficient mode,
`rn_Uv`, `rn_Lv`, layer thickness, domain size, masks, and forcing are all
fixed by the shipped deck and already stated in the resolved SWG card.

## Measurement

Preregistration commit: `933320102cb9323d3af0bec2ea19e62e53c14f30`.
Clean measured implementation: `65c0a29cbf0729943cc6c963ec7704354e3fba52`.

1. **CONFIRMED.** The card explicitly selects the shared `nemo_constant`
   coefficient source together with the e3-weighted div-curl operator. The
   builder forms its scalar from the resolved `rn_Uv=2.0 m/s` and
   `rn_Lv=500 m`; the shared dispatch assigns that scalar at every T and F
   latitude. Unknown sources raise at model construction, and the production
   SWE stage also refuses incompatible operators, e3 weighting, and sources.
2. **REFUTED.** The faithful constant branch does not clear the first debt.
   `given.kt2.s3.u_k` remains unequal on 6,320 wet U faces, with maximum
   absolute difference `1.7267697688394534e-13`. The entire given-entry row
   payload, excluding only the stamped legoESM commit, is byte-identical to
   round 3b: 306/338 AT-BAR and 32/338 DEBT. Artifact
   `swg_stages_given.json`, SHA-256
   `3370ae9366f88498de916e5fe3723863219469a823ab183661e73b76a30e5247`.
3. **CONFIRMED.** No certified row or frontier moved. The chained payload is
   likewise byte-identical to round 3b: 127/338 AT-BAR, 211/338 DEBT, first
   debt at kt=2 stage 3, worst maximum absolute difference
   `2.639979230778522e-7`. The independent ten-step payload remains 24/50
   AT-BAR and 26/50 DEBT, first debt at kt=2, worst maximum absolute difference
   `3.1954825074897997e-7`. Artifact SHA-256 values are respectively
   `ee8faa9ccb1be068399cc0cfbada41c322243eb409bb08290a8d7c14d4042b63`
   and `8f8ae3d5f9aceb2e4bf2e0fd813dc9764d2a8690444dceb9e3f5982426323f63`.
4. **REFUTED as preregistered, then repaired with the permitted field-scale
   form.** The smallest normal binary64 U input still produced identical SSH,
   so it is not credited as a control. The landed zero-state plant instead
   inserts one ULP at the admitted kt=2 U field's nonzero scale into a wet,
   consumed zero-valued U cell. It changes two scored SSH cells, maximum
   absolute difference `7.806255641895632e-18`, reports DEBT, and exits 1.
   The ordinary-one plant changes two SSH cells by
   `7.105427357601002e-15`, reports DEBT, and exits 1. Artifact SHA-256 values
   are respectively
   `0dfc461e1eddba0b00c4e20e77cffbda2d5a50807ca36cd5a95286294051fe12`
   and `19c72967fb284caf9e7996dff8d35d39b30ed1cf6c9399cffb35bfc09254bf74`.
5. **CONFIRMED.** The F-point base thickness now reads the already constructed
   one-layer thickness array; the related H/face/depth construction reuses the
   same array. The independent wind denominator reads resolved
   `rn_domsiz_m`. The former duplicate literals are absent from those paths.
6. **Pending final gates.**

## First non-bit statement

Stages 1 and 2 remain bit-identical through kt=10. The first non-bit boundary
is still kt=2 stage 3, whose first newly executed source statement is the
`dyn_ldf` call at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:317-317`. The active coefficient
producer is now transcribed from
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90:311-322`, yet the scored payload is
unchanged. Therefore the stage record still identifies `dyn_ldf` only as the
first source-order candidate; it does not attribute the residual to the
coefficient, the later viscosity body, wind, or drag.

## Rule 12 disposition

All three scored payloads are identical to round 3b after deleting only their
commit stamps. No AT-BAR row moved and neither first-over-bar frontier moved
earlier. The new coefficient branch is opt-in: SWG alone selects
`nemo_constant`; the shared default remains `nemo_ldf_c2d`, and the existing
certified-card configuration test confirms GYRE, LOCK_EXCHANGE, and OVERFLOW
retain that default. Final hash and focused-card results are pending below.

## Review input

The independent round-3b verdict was:

> DO NOT SHIP — The required zero plant is ineffective, and stage-3 viscosity does not transcribe NEMO’s active coefficient branch.

Both stated defects are fixed. The review's coefficient-owner hypothesis is
refuted by the unchanged admitted-record payload and is not carried forward as
an attribution.

## Gates

The focused SWG battery initially passes 172/172 after the implementation. The
geometry gate remains 34/34 bit-identical; its signed-y Coriolis, wet-cell and
layer-thickness plants each exit 1. Remaining final gates are pending.

## Independent review

Independent review run by the operator.

## OPEN

1. Continue inside the stage-3 `dyn_ldf` boundary: the constant coefficient is
   now source-faithful but did not move any scored value. Discriminate the
   viscosity body from the later centred-wind and before-velocity drag
   statements before changing another term.
2. Only after every given and chained stage row is bit-identical, run the
   30-day rung and then the shipped 172800-step statistics.

No acquisition or configuration/carried-state decision is requested.
