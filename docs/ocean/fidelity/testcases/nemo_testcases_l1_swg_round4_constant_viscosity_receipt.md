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

Pending.

## Independent review

Independent review run by the operator.

## OPEN

Pending measurement.
