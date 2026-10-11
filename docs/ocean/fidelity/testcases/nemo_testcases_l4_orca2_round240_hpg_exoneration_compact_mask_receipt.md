# ORCA2 round 240 — HPG exonerated; extra compact V mask is first

Date: 2026-10-11. Frozen base: `f431cb7bf`. Preregistration commit:
`f679b98f9`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round240/`.

This round changes no production model file, card, deck, carried state,
stabiliser, sea-ice selector, or `unmeasured_features` entry. Its measurements
are labelled **independent hierarchy rung 0** and are not mixed with a
given-NEMO-entry result.

## Frozen statement and source branch

The resolved ORCA2 branch is `hpg_sco`; the preregistered `zps_hde` split is
therefore a dead arm. NEMO builds the V surface seed from the north/current
thickness-density products at
`ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:409-416`, accumulates the
same products through the interior at
`ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:445-453`, and writes the
raw HPG sum into `pvv` at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`.

NEMO then accumulates HPG, LDF, VOR, KEG and ZAD in source order at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:141-175`. Only afterward does
the vector branch depth-average the stored three-dimensional tendency with
its live `e3v_3d`, stored `vmask`, and `r1_hv_0` at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`.

legoESM instead applies a generic compact mask to every completed momentum
tendency at `ocean_pe_latlon_cgrid.py:5982-5983`. On the 68 northern-fold
faces this compact V mask is not NEMO's stored raw mask. That multiplication,
not an HPG arithmetic statement, is the first effective non-bit statement.

## Source-ordered split

The admitted rank-complete round-181 HPG operands and round-92 accumulator
record were replayed offline through the production Decision-114 geometry
route. Every HPG quantity is bit-exact on all 1,319 wet fold-row levels:

| statement or operand | unequal / support | maximum absolute difference |
|---|---:|---:|
| `zhpj` | 0 / 1,319 | 0 |
| `zvap` | 0 / 1,319 | 0 |
| `pvv = zhpj + zvap` | 0 / 1,319 | 0 |
| north `e3w`, north `rhd` | 0 / 1,319 each | 0 |
| current `e3w`, current `rhd` | 0 / 1,319 each | 0 |
| `r1_e2v` | 0 / 68 | 0 |

The raw literal HPG is likewise `0/1,319` unequal. The compact V mask differs
on `1,319/2,108` compared slots, maximum `1.0`. Multiplying the exact raw HPG
by that mask reproduces legoESM's recorded stage HPG component bit-for-bit
(`0/1,319` unequal). Against NEMO, that component differs on all `68/68`
fold-row faces, RMS `1.0748834693208086e-06 m s-2`, maximum
`2.4423127429248864e-06 m s-2` at `[147,135]`. LDF, VOR, KEG and ZAD leave
the same 68-face residual unchanged.

The ordinary and traced completed states are array-identical for T, S, u, v
and ssh. The offline completed forcing reproduces the live forcing exactly;
its maximum calibration residual and over-floor-set symmetric difference are
both zero.

## Round-239 retraction

Round 239 named HPG as the first operator from a replay that called
`nemo_qco_card_mesh_operands`, the retired reconstructed-geometry diagnostic.
The production stage uses `nemo_qco_resolved_mesh_operands` after Decision
114. The two routes differ on this fold support in `27/1,319` live `e3v`
values (maximum `437.2169154512344 m`), `1,319/2,108` V-mask values, and all
`68/68` reciprocal-depth values (maximum `0.03332976059679253 m-1`).

The retired route reproduces round 239's alleged HPG endpoint exactly:
`68/68` unequal, maximum `2.4423127429248864e-06 m s-2`. The production
Decision-114 route is `0/68` unequal at that endpoint. Therefore round 239's
claim that raw HPG owns the completed V-RHS debt is **RETRACTED**. Its measured
68-face number remains valid only as a stale-geometry diagnostic.

## Predictions

| prediction | disposition |
|---|---|
| R240-P1 | **CONFIRMED**: both records admit, the target is 68 faces, and the HPG component identity is exact. |
| R240-P2 | **REFUTED**: no HPG component or primitive leaves the floor; `zps_hde` is not executed. |
| R240-P3 | **NOT ACTIVATED**: P2 is false. |
| R240-P4 | **NOT ACTIVATED**: no source-faithful HPG repair exists to score. |
| R240-P5 | **CONFIRMED DOCUMENTARY**: no production package change or configuration choice was made. |
| R240-P6 | **CONFIRMED DOCUMENTARY**: focused controls, citation checks and the fidelity battery are reported below. |

The stale-route reconciliation and compact-mask classification are explicitly
post-hoc follow-ups to the refuted HPG prediction. They do not rewrite the
frozen prediction.

## Instrument failures retained

1. The first HPG invocation pointed the component admission at the round-83
   terminal root. It refused because the rank-complete RHS frames live under
   round 92; the corrected invocation changed only that record root.
2. The first corrected operator classifier expected the already-masked stage
   component itself to be exact and refused. Inspection showed that this
   component is the evidence for the extra mask: raw HPG times compact mask
   reproduces it exactly. The classifier was corrected and rerun from the
   same passive records.

## Controls and verification

All 18 admission, selector, source-order, route, raw-HPG, mask-effect,
masked-replay, first-owner and calibration plants refuse. Binding SHA-256
values are:

- HPG split: `76edd5e7e8a37e553a2fe0e94b6a36fbcbec10b987330292046ebcd9deab3902`
- round-239 reconciliation: `69709d76a07b55c60beefc6f76b36e56dd70ae8d7a6b39a8248ec1ab5dfec9bf`
- corrected operator table: `ca6a979db608830de0923070d6e1dc0407cffe861bd009a8cefcfc177d8a6684`
- plant log: `b4f90abb5838f4ec41ceb8024f44954dd6f2dc4c27cf9c570581a70568ff41c1`
- focused tests: `75a3eda3c37782ebda638b7cffd2116b8e4c6cd6b9cce64477fe1050e37ec8d4`

The round-240 and round-239 focused tests pass `27/27`. The receipt citation
gate and its rigid-shift plant, the default cumulative receipt citation gate,
and the one `tests/ocean/fidelity -n 12` battery are recorded below after
execution. A separate `codex exec --sandbox read-only` review was attempted;
it could not initialize its in-process client because the sandbox refused the
required read-only filesystem alias. Independent review is unavailable
in-sandbox (log SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`).

No `packages/` file changed, so the candidate trajectory, GYRE year, DINO,
tanks, rung-0/rung-10 ladders and sea-ice card are inherited rather than
relabelled as newly passed gates.

## OPEN

Round 241 measures one private, source-exact arm that preserves the literal
HPG value through the generic compact-mask boundary on the 68 fold faces,
then replays the completed V RHS and re-scores the complete round-217 unit
atomically. A partial mask or HPG operand does not land. If that arm does not
close the completed RHS, the next source-ordered owner after this boundary is
named from the passive accumulator record.

ASKED choices: Decisions 96, 103, 109, 113, 114 and 115. UNASKED choices:
empty.
