# Round 57 preregistration — kt=2 TKE closure operand walk

Date: 2026-09-11. Frozen parent: `21e85d25284d`. CPU,
production JIT, fp64/scalar-libm only. NEMO source and records are read-only;
no NEMO integration, `makenemo`, or `mpirun` is run by the agent.

## Round-56 review repairs precede measurement

The independent round-56 verdict is DO NOT SHIP. Before any score, round 57
repairs the z-star-card surface-mask construction, registers the legacy
`nemo_v1` floor move, makes the shape plant reach the stamped-array-shape
guard, corrects receipt provenance/test status and source line 571, removes
the tautological row-size check, and routes the two inert DINO diagnostics
through the shared derived floor. The required recipe gate is exactly
`tests/ocean/unit/test_nemo_recipe.py`: 24 passed. Its new plant requires a
card whose coordinate has no `is_active` to receive `state.land_mask.data` as
NEMO `tmask(:,:,1)`; the round-56 code raises before this card can step.

## Frozen closure hypotheses and falsifiers

The oracle record is
`phase3/round56/oracle_tke_operands/oracle_tke_operands_kt00000002.bin`,
produced at `21e85d25284de001c575e23f351a45c426ba5e15` by compiled
`cfgs/GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/{zdftke,zdfphy}.f90`.

Calibration runs first. Rebuilding NEMO's `en`, `zmxlm`, `zmxld`, `avm`,
`avt`, and `dissl` from the record with the compiled statements and their
association must give **0 unequal cells** on every consumed wet interface.
Any unequal calibration cell invalidates the instrument and stops the walk.

The production-path walk then calls legoESM's shared TKE implementation with
the record's carried operands. NEMO operands are substituted cumulatively in
compiled source order: carried production/buoyancy/dissipation operands;
matrix/RHS and the `en` sweep/floor; surface `rn_ebb*taum` and
`taum*tmask(:,:,1)`; raw mixing length, derived floor and bounds; Prandtl;
`avm/avt/dissl`; and the `zdfphy` pre-EVD copy. The first cumulative row that
takes `avt` to 0 unequal cells names the statement eligible to land. No
tolerance may be called exact. A plant changes one wet recorded operand and
must make the gate exit nonzero.

Round-56 P1/P2 stand unchanged. P1: the shared derived floor is the sole
coefficient owner; any remaining unequal acquired `avt` cell refutes it. P2:
after landing the named statement, kt=3 maximum T/S residuals are exactly
`1.6275031290e-4 K` and `6.3278533133e-6 g/kg`; any larger value refutes the
quantitative prediction. The one-ULP round-55-to-56 component is below the
trajectory instrument's resolution and is not separately attributed.

## Frozen controlled comparison and Rule 12

The before arm is the recorded decision-35 after arm, never a scratch toggle:
`decision35_fix_eta_drift/after_kt1_10.json` and `after_day_gap.json`. It uses
the same certified ladder gate, scalar-math-v2 roots, day-gap tool, daily
cadence, six-step snapshots, and days 1--30 as the current-tip after arm. The
floor attribution combines the never-measured decision-35-to-round-55 and
round-55-to-round-56 moves. Every moved row is registered; no AT-BAR row may
leave the bar and the first-over-bar row may not move earlier.

| card | preregistered Rule-12 disposition |
|---|---|
| GYRE | kt=1..10 plus days 1--30 measured against the decision-35 recorded after arm; report kt3 T/S and day-30. |
| legacy `nemo_v1` | Floor moves `1e-8` to `9.99999999999999847e-03` (about `1e6` times). Its only card-owned gate is the 24-test recipe suite, which proves construction/stepping but does not bit-score TKE closure: **UNMEASURED-with-spec debt**, to score this card's own `en/avm/avt/dissl` against its compiled oracle operands. |
| LOCK_EXCHANGE | Resolved `ln_zdfcst=.true.` at `LOCK_EXCHANGE_OMIP_L1_P3_R33ZDF/EXP00/namelist_cfg:131`; shared TKE statement not executed. |
| OVERFLOW | Resolved `ln_zdfcst=.true.` at `OVERFLOW_OMIP_L1_P3_R33ZDF/EXP00/namelist_cfg:129`; shared TKE statement not executed. |
| DINO | Shares `tke.py` on a separate branch; statement-level risk is explicit. No DINO trajectory claim is made here. |
| ORCA2 | **UNMEASURED-with-spec:** bit-score the same compiled TKE statement using ORCA2's own operands, resolved closure and masks. |

ASKED: review repairs, calibration/substitution, shared TKE statement, GYRE
ladder/month, card audit, citation/stamp/tests. UNASKED: none. No configuration,
carried-state, freshwater-pair, year-harness, reconciliation-gate, or #1484
guard change is admitted by this preregistration.
