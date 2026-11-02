# SI3 main catch-up receipt — 2026-10-30

## Scope and immutable inputs

- Lane parent: `e1f1fda931584f935161ac6f7c7924580c28c82b`.
- Main parent: `b5adef4a5d7b817dbb485962cab6e7497d87ab56`.
- Catch-up 2 main parent: `edd34c24469c834d536c511160ea0c4aef8974ae`.
- Catch-up 2 merge: `8debf7c4c79c`, with first parent `ba69df5f55`
  and second parent `edd34c24469c834d536c511160ea0c4aef8974ae`.
- Merge base: `ca822f33d930b9df4987561cc781cd3c1028e0a6`.
- Merge mode: one `git merge --no-ff` commit; no rebase.
- Textual conflict markers counted before resolution: **198** across 25 files.
- Preservation rule: landed SI3 D5 and O1–O16 production selections and exactness tests remain selected exactly; main behavior is retained unless it is an obsolete implementation replaced by the certified lane.
- independent review: operator

## Conflict-hunk ledger

Every pre-resolution conflict marker block is listed once. “Main” and “lane” describe the two parent contributions; “kept” describes the merged result.

| # | File / hunk | Main changed | Lane changed | Kept | Why |
|---:|---|---|---|---|---|
| 1 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H01 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 2 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H02 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 3 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H03 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 4 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H04 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 5 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H05 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 6 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H06 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 7 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H07 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 8 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H08 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 9 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H09 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 10 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H10 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 11 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H11 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 12 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H12 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 13 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H13 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 14 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H14 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 15 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H15 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 16 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H16 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 17 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H17 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 18 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H18 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 19 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H19 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 20 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H20 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 21 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H21 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 22 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H22 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 23 | `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md` H23 | main shifted source-line pins after its ocean edits | lane carried the certified round-8 prose and SI3-era pins | kept the lane claim text and re-anchored every moved pin to the merged source | the claim is unchanged; the citation gate, not either parent’s line number, determines the live pin |
| 24 | `docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_round112_een_southern_ff_landing_receipt.md` H01 | main shifted the EEN helper pins | lane pinned the certified southern-fill implementation | kept the certified claim and re-anchored it to the merged helper | preserves the landed ORCA2 verdict while making the citation shift-sensitive |
| 25 | `packages/core/legoesm/core/bulk_flux.py` H01 | main added the NEMO source-rounding import | lane added the smooth quintic helper import | kept both imports | the helpers are independent and both merged call paths remain selectable |
| 26 | `packages/core/legoesm/thermo.py` H01 | main added transcendental/source-rounding support and older wording | lane added smooth helpers and certified 5.0.2 wording | kept both implementations and the 5.0.2 contract wording | the executable features compose; the lane wording is the certified oracle version |
| 27 | `packages/core/legoesm/thermo.py` H02 | main added transcendental/source-rounding support and older wording | lane added smooth helpers and certified 5.0.2 wording | kept both implementations and the 5.0.2 contract wording | the executable features compose; the lane wording is the certified oracle version |
| 28 | `packages/ice/legoesm/ice/bitz_lipscomb.py` H01 | main added an AD-safe double-`where` guard to `_piecewise_remap` for a zero-thickness column | lane deleted that duplicate remap and certified `_cumulative_remap` instead | kept the lane's live cumulative remap and applied a forward-identity `stop_gradient` guard to its no-material operands; did **not** revive `_piecewise_remap` | the original merge's “equivalent safe divisor” claim was false: the lane divisor was forward-safe but did not preserve main's zero cotangent, while reviving the old symbol violates the O8–O12 no-duplicate gate; the new guard changes no forward operand or output bit |
| 29 | `packages/ice/legoesm/ice/bitz_lipscomb.py` H02 | main added the nested safe divisor for open-water `h_total` so an inactive quotient cannot poison reverse mode | lane replaced the surrounding production chain, retained a guarded live divisor, but still evaluated inactive gathered `qns_ice`/`h_ice` cotangents | kept the lane arithmetic and main's live `h_total` guard, and made the inactive gathered `h_ice` and `qns_ice` operands forward-identical stopped values | the original equivalence claim was false: `jax.grad` of the merged production step was NaN on open water; the gather guard makes it finite without changing any forward value |
| 30 | `packages/ml/legoesm/training/param_collector.py` H01 | main added include-versus-exclude refusal | lane added explicit-only initialization and reachability validation | kept both validations | the checks cover independent invalid configurations |
| 31 | `packages/ocean/legoesm/ocean/advection.py` H01 | main added NEMO beta/stencil/UP1 trace plumbing | lane added the certified literal FCT/source-order implementation | kept the lane implementation with all main trace APIs in its generic fallback | the lane is a behavioral superset and certified exact paths stay byte-for-byte selected |
| 32 | `packages/ocean/legoesm/ocean/advection.py` H02 | main added NEMO beta/stencil/UP1 trace plumbing | lane added the certified literal FCT/source-order implementation | kept the lane implementation with all main trace APIs in its generic fallback | the lane is a behavioral superset and certified exact paths stay byte-for-byte selected |
| 33 | `packages/ocean/legoesm/ocean/advection.py` H03 | main added NEMO beta/stencil/UP1 trace plumbing | lane added the certified literal FCT/source-order implementation | kept the lane implementation with all main trace APIs in its generic fallback | the lane is a behavioral superset and certified exact paths stay byte-for-byte selected |
| 34 | `packages/ocean/legoesm/ocean/advection.py` H04 | main added NEMO beta/stencil/UP1 trace plumbing | lane added the certified literal FCT/source-order implementation | kept the lane implementation with all main trace APIs in its generic fallback | the lane is a behavioral superset and certified exact paths stay byte-for-byte selected |
| 35 | `packages/ocean/legoesm/ocean/advection.py` H05 | main added NEMO beta/stencil/UP1 trace plumbing | lane added the certified literal FCT/source-order implementation | kept the lane implementation with all main trace APIs in its generic fallback | the lane is a behavioral superset and certified exact paths stay byte-for-byte selected |
| 36 | `packages/ocean/legoesm/ocean/advection.py` H06 | main added NEMO beta/stencil/UP1 trace plumbing | lane added the certified literal FCT/source-order implementation | kept the lane implementation with all main trace APIs in its generic fallback | the lane is a behavioral superset and certified exact paths stay byte-for-byte selected |
| 37 | `packages/ocean/legoesm/ocean/advection.py` H07 | main added NEMO beta/stencil/UP1 trace plumbing | lane added the certified literal FCT/source-order implementation | kept the lane implementation with all main trace APIs in its generic fallback | the lane is a behavioral superset and certified exact paths stay byte-for-byte selected |
| 38 | `packages/ocean/legoesm/ocean/advection.py` H08 | main added NEMO beta/stencil/UP1 trace plumbing | lane added the certified literal FCT/source-order implementation | kept the lane implementation with all main trace APIs in its generic fallback | the lane is a behavioral superset and certified exact paths stay byte-for-byte selected |
| 39 | `packages/ocean/legoesm/ocean/advection.py` H09 | main added NEMO beta/stencil/UP1 trace plumbing | lane added the certified literal FCT/source-order implementation | kept the lane implementation with all main trace APIs in its generic fallback | the lane is a behavioral superset and certified exact paths stay byte-for-byte selected |
| 40 | `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py` H01 | main added periodic guards and explicit external-mode flux selection | lane added ORCA fold helpers, EEN fills, and certified broadcasting | kept both with periodic and fold behavior selected by the existing configuration | main’s general mode and the lane’s NEMO mode are independent |
| 41 | `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py` H02 | main added periodic guards and explicit external-mode flux selection | lane added ORCA fold helpers, EEN fills, and certified broadcasting | kept both with periodic and fold behavior selected by the existing configuration | main’s general mode and the lane’s NEMO mode are independent |
| 42 | `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py` H03 | main added periodic guards and explicit external-mode flux selection | lane added ORCA fold helpers, EEN fills, and certified broadcasting | kept both with periodic and fold behavior selected by the existing configuration | main’s general mode and the lane’s NEMO mode are independent |
| 43 | `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py` H04 | main added periodic guards and explicit external-mode flux selection | lane added ORCA fold helpers, EEN fills, and certified broadcasting | kept both with periodic and fold behavior selected by the existing configuration | main’s general mode and the lane’s NEMO mode are independent |
| 44 | `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py` H05 | main added periodic guards and explicit external-mode flux selection | lane added ORCA fold helpers, EEN fills, and certified broadcasting | kept both with periodic and fold behavior selected by the existing configuration | main’s general mode and the lane’s NEMO mode are independent |
| 45 | `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py` H06 | main added periodic guards and explicit external-mode flux selection | lane added ORCA fold helpers, EEN fills, and certified broadcasting | kept both with periodic and fold behavior selected by the existing configuration | main’s general mode and the lane’s NEMO mode are independent |
| 46 | `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py` H07 | main added periodic guards and explicit external-mode flux selection | lane added ORCA fold helpers, EEN fills, and certified broadcasting | kept both with periodic and fold behavior selected by the existing configuration | main’s general mode and the lane’s NEMO mode are independent |
| 47 | `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py` H08 | main added periodic guards and explicit external-mode flux selection | lane added ORCA fold helpers, EEN fills, and certified broadcasting | kept both with periodic and fold behavior selected by the existing configuration | main’s general mode and the lane’s NEMO mode are independent |
| 48 | `packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py` H09 | main added periodic guards and explicit external-mode flux selection | lane added ORCA fold helpers, EEN fills, and certified broadcasting | kept both with periodic and fold behavior selected by the existing configuration | main’s general mode and the lane’s NEMO mode are independent |
| 49 | `packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py` H01 | main added periodic and north-fold density/geometry handling | lane added certified north-neighbor and fold-grid operators | kept all static arguments and all three fold/periodic helpers | the merged operator retains both grid families without changing certified defaults |
| 50 | `packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py` H02 | main added periodic and north-fold density/geometry handling | lane added certified north-neighbor and fold-grid operators | kept all static arguments and all three fold/periodic helpers | the merged operator retains both grid families without changing certified defaults |
| 51 | `packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py` H03 | main added periodic and north-fold density/geometry handling | lane added certified north-neighbor and fold-grid operators | kept all static arguments and all three fold/periodic helpers | the merged operator retains both grid families without changing certified defaults |
| 52 | `packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py` H04 | main added periodic and north-fold density/geometry handling | lane added certified north-neighbor and fold-grid operators | kept all static arguments and all three fold/periodic helpers | the merged operator retains both grid families without changing certified defaults |
| 53 | `packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py` H05 | main added periodic and north-fold density/geometry handling | lane added certified north-neighbor and fold-grid operators | kept all static arguments and all three fold/periodic helpers | the merged operator retains both grid families without changing certified defaults |
| 54 | `packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py` H06 | main added periodic and north-fold density/geometry handling | lane added certified north-neighbor and fold-grid operators | kept all static arguments and all three fold/periodic helpers | the merged operator retains both grid families without changing certified defaults |
| 55 | `packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py` H07 | main added periodic and north-fold density/geometry handling | lane added certified north-neighbor and fold-grid operators | kept all static arguments and all three fold/periodic helpers | the merged operator retains both grid families without changing certified defaults |
| 56 | `packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py` H08 | main added periodic and north-fold density/geometry handling | lane added certified north-neighbor and fold-grid operators | kept all static arguments and all three fold/periodic helpers | the merged operator retains both grid families without changing certified defaults |
| 57 | `packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py` H09 | main added periodic and north-fold density/geometry handling | lane added certified north-neighbor and fold-grid operators | kept all static arguments and all three fold/periodic helpers | the merged operator retains both grid families without changing certified defaults |
| 58 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H01 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 59 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H02 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 60 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H03 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 61 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H04 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 62 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H05 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 63 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H06 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 64 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H07 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 65 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H08 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 66 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H09 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 67 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H10 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 68 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H11 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 69 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H12 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 70 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H13 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 71 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H14 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 72 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H15 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 73 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H16 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 74 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H17 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 75 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H18 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 76 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H19 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 77 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H20 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 78 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H21 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 79 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H22 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 80 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H23 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 81 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H24 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 82 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H25 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 83 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H26 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 84 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H27 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 85 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H28 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 86 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H29 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 87 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H30 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 88 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H31 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 89 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H32 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 90 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H33 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 91 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H34 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 92 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H35 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 93 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H36 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 94 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H37 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 95 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H38 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 96 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H39 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 97 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H40 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 98 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H41 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 99 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H42 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 100 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H43 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 101 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H44 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 102 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H45 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 103 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H46 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 104 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H47 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 105 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H48 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 106 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H49 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 107 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H50 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 108 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H51 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 109 | `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` H52 | main added raw-mesh guards, budget splits, periodic SWE routing, ENS/vorticity overrides, and AD-safe dry handling | lane added the certified WS-RK3/FCT, K2, drag, source-order, stage-geometry, and trace paths | composed the independent features; restored lane-exact CEN2 and QCO-rank expressions where a union would change certified arithmetic | all main features remain reachable while the O1-O16/D5 production selections retain their exact statement order |
| 110 | `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` H01 | main added ENS totals and vorticity override routing | lane added fold-halo, raw-mask, and certified face-geometry routing | kept both and routed the merged vorticity argument through the common override | the operands are complementary and the selector remains explicit |
| 111 | `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` H02 | main added ENS totals and vorticity override routing | lane added fold-halo, raw-mask, and certified face-geometry routing | kept both and routed the merged vorticity argument through the common override | the operands are complementary and the selector remains explicit |
| 112 | `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` H03 | main added ENS totals and vorticity override routing | lane added fold-halo, raw-mask, and certified face-geometry routing | kept both and routed the merged vorticity argument through the common override | the operands are complementary and the selector remains explicit |
| 113 | `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` H04 | main added ENS totals and vorticity override routing | lane added fold-halo, raw-mask, and certified face-geometry routing | kept both and routed the merged vorticity argument through the common override | the operands are complementary and the selector remains explicit |
| 114 | `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` H05 | main added ENS totals and vorticity override routing | lane added fold-halo, raw-mask, and certified face-geometry routing | kept both and routed the merged vorticity argument through the common override | the operands are complementary and the selector remains explicit |
| 115 | `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` H06 | main added ENS totals and vorticity override routing | lane added fold-halo, raw-mask, and certified face-geometry routing | kept both and routed the merged vorticity argument through the common override | the operands are complementary and the selector remains explicit |
| 116 | `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` H07 | main added ENS totals and vorticity override routing | lane added fold-halo, raw-mask, and certified face-geometry routing | kept both and routed the merged vorticity argument through the common override | the operands are complementary and the selector remains explicit |
| 117 | `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` H08 | main added ENS totals and vorticity override routing | lane added fold-halo, raw-mask, and certified face-geometry routing | kept both and routed the merged vorticity argument through the common override | the operands are complementary and the selector remains explicit |
| 118 | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` H01 | main added GYRE resolution, SWG cards, and raw-stage-mesh selection | lane added ORCA exact cards and SI3-certified gamma/geometry selections | kept both card families; retained the lane’s certified ORCA gamma and added main’s independent GYRE/SWG dispatch | card-specific behavior remains explicit and no certified card default changed |
| 119 | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` H02 | main added GYRE resolution, SWG cards, and raw-stage-mesh selection | lane added ORCA exact cards and SI3-certified gamma/geometry selections | kept both card families; retained the lane’s certified ORCA gamma and added main’s independent GYRE/SWG dispatch | card-specific behavior remains explicit and no certified card default changed |
| 120 | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` H03 | main added GYRE resolution, SWG cards, and raw-stage-mesh selection | lane added ORCA exact cards and SI3-certified gamma/geometry selections | kept both card families; retained the lane’s certified ORCA gamma and added main’s independent GYRE/SWG dispatch | card-specific behavior remains explicit and no certified card default changed |
| 121 | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` H04 | main added GYRE resolution, SWG cards, and raw-stage-mesh selection | lane added ORCA exact cards and SI3-certified gamma/geometry selections | kept both card families; retained the lane’s certified ORCA gamma and added main’s independent GYRE/SWG dispatch | card-specific behavior remains explicit and no certified card default changed |
| 122 | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` H05 | main added GYRE resolution, SWG cards, and raw-stage-mesh selection | lane added ORCA exact cards and SI3-certified gamma/geometry selections | kept both card families; retained the lane’s certified ORCA gamma and added main’s independent GYRE/SWG dispatch | card-specific behavior remains explicit and no certified card default changed |
| 123 | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` H06 | main added GYRE resolution, SWG cards, and raw-stage-mesh selection | lane added ORCA exact cards and SI3-certified gamma/geometry selections | kept both card families; retained the lane’s certified ORCA gamma and added main’s independent GYRE/SWG dispatch | card-specific behavior remains explicit and no certified card default changed |
| 124 | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` H07 | main added GYRE resolution, SWG cards, and raw-stage-mesh selection | lane added ORCA exact cards and SI3-certified gamma/geometry selections | kept both card families; retained the lane’s certified ORCA gamma and added main’s independent GYRE/SWG dispatch | card-specific behavior remains explicit and no certified card default changed |
| 125 | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` H08 | main added GYRE resolution, SWG cards, and raw-stage-mesh selection | lane added ORCA exact cards and SI3-certified gamma/geometry selections | kept both card families; retained the lane’s certified ORCA gamma and added main’s independent GYRE/SWG dispatch | card-specific behavior remains explicit and no certified card default changed |
| 126 | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` H09 | main added GYRE resolution, SWG cards, and raw-stage-mesh selection | lane added ORCA exact cards and SI3-certified gamma/geometry selections | kept both card families; retained the lane’s certified ORCA gamma and added main’s independent GYRE/SWG dispatch | card-specific behavior remains explicit and no certified card default changed |
| 127 | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` H10 | main added GYRE resolution, SWG cards, and raw-stage-mesh selection | lane added ORCA exact cards and SI3-certified gamma/geometry selections | kept both card families; retained the lane’s certified ORCA gamma and added main’s independent GYRE/SWG dispatch | card-specific behavior remains explicit and no certified card default changed |
| 128 | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` H11 | main added GYRE resolution, SWG cards, and raw-stage-mesh selection | lane added ORCA exact cards and SI3-certified gamma/geometry selections | kept both card families; retained the lane’s certified ORCA gamma and added main’s independent GYRE/SWG dispatch | card-specific behavior remains explicit and no certified card default changed |
| 129 | `packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py` H01 | main added live divisor-thickness selection | lane added reference-thickness and geometry-stage selections | kept all three selectors | they address different operands |
| 130 | `packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py` H01 | main added the live divisor block and operand precedence | lane added native-face/fold behavior and certified validation | kept both and explicitly selected nemo_qco_live for the ORCA recipe | the merged selection reproduces the certified lane while exposing main’s general option |
| 131 | `packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py` H02 | main added the live divisor block and operand precedence | lane added native-face/fold behavior and certified validation | kept both and explicitly selected nemo_qco_live for the ORCA recipe | the merged selection reproduces the certified lane while exposing main’s general option |
| 132 | `packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py` H01 | main changed the literal solve plumbing | lane added the certified optional K_2 heat/salt split | kept K_2 through every merged call and return arm | dropping K_2 would remove a landed exactness path |
| 133 | `packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py` H02 | main changed the literal solve plumbing | lane added the certified optional K_2 heat/salt split | kept K_2 through every merged call and return arm | dropping K_2 would remove a landed exactness path |
| 134 | `packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py` H03 | main changed the literal solve plumbing | lane added the certified optional K_2 heat/salt split | kept K_2 through every merged call and return arm | dropping K_2 would remove a landed exactness path |
| 135 | `packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py` H04 | main changed the literal solve plumbing | lane added the certified optional K_2 heat/salt split | kept K_2 through every merged call and return arm | dropping K_2 would remove a landed exactness path |
| 136 | `packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py` H01 | main made the post-solve dissipation branch AD-safe on inactive values | lane used safe_sqrt on the certified positive path | kept main’s guarded operands with lane safe_sqrt | inactive gradients stay finite and positive production arithmetic stays lane-exact |
| 137 | `packages/ocean/legoesm/ocean/state.py` H01 | main added the external-mode flux-form selector | lane added certified forcing fields and NEMO selectors | kept all fields without duplicating the auto-merged alpha field | the state remains a pure pytree and both selectors are explicit |
| 138 | `packages/ocean/legoesm/ocean/state.py` H02 | main added the external-mode flux-form selector | lane added certified forcing fields and NEMO selectors | kept all fields without duplicating the auto-merged alpha field | the state remains a pure pytree and both selectors are explicit |
| 139 | `packages/ocean/legoesm/ocean/state.py` H03 | main added the external-mode flux-form selector | lane added certified forcing fields and NEMO selectors | kept all fields without duplicating the auto-merged alpha field | the state remains a pure pytree and both selectors are explicit |
| 140 | `packages/ocean/legoesm/ocean/vertical.py` H01 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 141 | `packages/ocean/legoesm/ocean/vertical.py` H02 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 142 | `packages/ocean/legoesm/ocean/vertical.py` H03 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 143 | `packages/ocean/legoesm/ocean/vertical.py` H04 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 144 | `packages/ocean/legoesm/ocean/vertical.py` H05 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 145 | `packages/ocean/legoesm/ocean/vertical.py` H06 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 146 | `packages/ocean/legoesm/ocean/vertical.py` H07 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 147 | `packages/ocean/legoesm/ocean/vertical.py` H08 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 148 | `packages/ocean/legoesm/ocean/vertical.py` H09 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 149 | `packages/ocean/legoesm/ocean/vertical.py` H10 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 150 | `packages/ocean/legoesm/ocean/vertical.py` H11 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 151 | `packages/ocean/legoesm/ocean/vertical.py` H12 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 152 | `packages/ocean/legoesm/ocean/vertical.py` H13 | main added raw-stage faces and r3f override/return APIs | lane added fold ratios, native F geometry, fe3mask, and source association | kept both API aliases, rejected simultaneous overrides, and retained both return modes | this preserves old callers and both independently selected behaviors |
| 153 | `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round107_een_u_operand_walk.py` H01 | main updated the ORCA operand walk | lane carried the expanded certified probe | kept the lane superset | no measured field or plant is dropped |
| 154 | `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round107_een_u_operand_walk.py` H02 | main updated the ORCA operand walk | lane carried the expanded certified probe | kept the lane superset | no measured field or plant is dropped |
| 155 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H01 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 156 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H02 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 157 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H03 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 158 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H04 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 159 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H05 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 160 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H06 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 161 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H07 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 162 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H08 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 163 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H09 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 164 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H10 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 165 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H11 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 166 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H12 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 167 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H13 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 168 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H14 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 169 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H15 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 170 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H16 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 171 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H17 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 172 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H18 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 173 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H19 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 174 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H20 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 175 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H21 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 176 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H22 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 177 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H23 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 178 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H24 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 179 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H25 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 180 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H26 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 181 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H27 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 182 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H28 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 183 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H29 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 184 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H30 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 185 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H31 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 186 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H32 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 187 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H33 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 188 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H34 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 189 | `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` H35 | main added GYRE/TSUNAMI/SWG maps and shifted common pins | lane added the complete SI3/O7 map and re-anchor chain | kept both source maps, removed obsolete duplicate literal entries, and explicitly re-anchored all moved symbols | the full 3,755-entry audit and duplicate-key plant remain strict |
| 190 | `tests/ocean/fidelity/test_orca2_round103_rung0_ladder_gate.py` H01 | main added a ladder assertion | lane added a distinct certified assertion | kept both tests | no test from either parent is dropped |
| 191 | `tests/ocean/unit/test_gm_redi_latlon_cgrid.py` H01 | main added live-divisor coverage | lane added face/fold exactness coverage | kept both tests | the merged implementation supports both |
| 192 | `tests/ocean/unit/test_nemo_advection_off_arms.py` H01 | main added TSUNAMI off/upwind plants and validator coverage | lane added VORTEX explicit-card and zero-dispatch coverage | kept both fixtures and all tests | both cards and both planted violations must continue to fire |
| 193 | `tests/ocean/unit/test_nemo_advection_off_arms.py` H02 | main added TSUNAMI off/upwind plants and validator coverage | lane added VORTEX explicit-card and zero-dispatch coverage | kept both fixtures and all tests | both cards and both planted violations must continue to fire |
| 194 | `tests/ocean/unit/test_nemo_advection_off_arms.py` H03 | main added TSUNAMI off/upwind plants and validator coverage | lane added VORTEX explicit-card and zero-dispatch coverage | kept both fixtures and all tests | both cards and both planted violations must continue to fire |
| 195 | `tests/ocean/unit/test_nemo_advection_off_arms.py` H04 | main added TSUNAMI off/upwind plants and validator coverage | lane added VORTEX explicit-card and zero-dispatch coverage | kept both fixtures and all tests | both cards and both planted violations must continue to fire |
| 196 | `tests/ocean/unit/test_zdf_implicit_literal.py` H01 | main changed literal-solver coverage | lane added K_2 exactness coverage | kept the lane test alongside main’s auto-merged suite | the certified split solve remains tested |
| 197 | `tests/test_param_specs.py` H01 | main added declared-unset-field helpers | lane added an equivalent float-annotation helper | kept main’s shared helper and removed only the redundant duplicate definition | test coverage is unchanged and no test case is dropped |
| 198 | `tests/test_param_specs.py` H02 | main added declared-unset-field helpers | lane added an equivalent float-annotation helper | kept main’s shared helper and removed only the redundant duplicate definition | test coverage is unchanged and no test case is dropped |

## Catch-up 2

Catch-up 2 merged the supplied newer-main target
`edd34c24469c834d536c511160ea0c4aef8974ae` with one additional
`--no-ff` merge commit.  It produced exactly one textual conflict marker
block, recorded once here.

| # | File / hunk | Newer main changed | SI3 first parent changed | Kept | Why |
|---:|---|---|---|---|---|
| C2-1 | `packages/core/legoesm/core/bulk_flux.py` H01 (import block) | added `positive_part_c1` beside `smoothstep_quintic` | retained the certified NEMO source-rounding alias `sr` beside `smoothstep_quintic` | imported all three independent helpers | all three names have distinct live consumers; composing the imports preserves both parents without changing selection |

Catch-up 2 no-conflict semantic risks:

- `packages/core/legoesm/core/{smooth.py,bulk_flux.py}` and
  `packages/core/legoesm/thermo.py` change shared smooth/thermodynamic helpers;
  the import hunk above was the only textual conflict.
- `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py`
  changed the same tracer-RHS neighborhood without a catch-up-2 conflict.
  The merged helper now uses main's `active_t` output mask by default and the
  lane's raw inactive-storage/e3t-zero behavior only when the certified NEMO
  WS/FCT caller passes `_certified_nemo_storage=True`.  The two formulas are
  bit-identical at wet and partial cells with `tmask > 0, e3t > 0`; they differ
  at land/inactive cells (`tmask == 0`) whenever the raw divergence is nonzero,
  including inactive storage with either zero or positive `e3t`.  An active
  zero-thickness cell is an invalid geometry: the lane guard remains defined,
  while main's active divisor would be zero.
- `packages/ocean/legoesm/ocean/physics/combined.py`,
  `vertical_mixing/{epbl.py,integration.py,mpas_integration.py}`, and
  `packages/ocean/legoesm/ocean/scm.py` change shared vertical-mixing and SCM
  routing adjacent to NEMO physics.  Their changed/new named tests were run
  serially; all ocean files passed (`85`, `7`, and `17` tests respectively).
- New MOM6 ePBL fixtures and the changed atmosphere/unit tests were retained
  wholesale.  Every existing changed test file under `tests/ocean`,
  `tests/ice`, and `tests/unit` was run; the failures reproduced identically
  on standalone newer main and are enumerated below.

## Non-textual semantic-risk inventory

These main changes touch ocean/NEMO paths used by, or immediately surrounding, the certified lane without producing a textual conflict:

- `packages/ocean/legoesm/ocean/config.py` — main changes shared configuration construction used by certified recipes.
- `packages/ocean/legoesm/ocean/dynamics/barotropic_common.py` and `barotropic_implicit_latlon_cgrid.py` — main changes common/implicit external-mode utilities adjacent to certified explicit routing.
- `packages/ocean/legoesm/ocean/eos.py` — main moves and extends the NEMO RK3 stretch/EOS helpers used by exact cards.
- `packages/ocean/legoesm/ocean/experiments/dino.py` — main changes DINO recipe threading and therefore the broader NEMO-fidelity configuration surface.
- `packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py`, `nemo_state_bridge.py`, `time_levels.py`, and `provenance.py` — main changes recipe construction, state bridging, clock semantics, and SHA stamping.
- `packages/ocean/legoesm/ocean/init_latlon_cgrid.py` and `recipes.py` — main changes initialization/recipe plumbing consumed before certified production steps.
- `packages/ocean/legoesm/ocean/physics/combined.py` — main changes the combined physics dispatch surrounding certified vertical/lateral operators.
- `packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py` — main adds KPP EOS-salinity preprocessing; the merged ordering keeps the lane's fail-closed KPP/convective-viscosity validation ahead of closure evaluation.
- `packages/ocean/legoesm/ocean/physics/vertical_mixing/_shared.py`, `config.py`, and `integration.py` — main changes shared shear/config/integration code used by NEMO TKE and ZDF paths.
- `packages/ocean/legoesm/ocean/timestep.py` — main changes generic time-step routing adjacent to the certified RK3 path.
- `config/templates/ocean/*.yaml` — main updates shipped ocean templates; none silently selects a certified private arm.
- Added/changed SWG, TSUNAMI, GYRE-hires, VORTEX, and ORCA fidelity scripts/tests under `scripts/validate/ocean_fidelity/testcases/` and `tests/ocean/fidelity/` — retained wholesale; these are semantic-risk controls even where no production hunk conflicted.
- `tests/ocean/fidelity/test_post_20261029_provenance_sha_gate.py` — the lane's prospective gate initially treated the catch-up merge's independently authored second-parent history as new lane evidence.  Its later one-hop exemption was also too weak: a token could be re-edited, copied or retargeted on the same line, or moved between lines and then evade validation.  The exemption now maps the same logical line through every first-parent revision back to the pre-cutoff base and requires the same token at every revision.  Three committed bypass plants fail inheritance, while a same-line citation-only re-anchor remains valid.
- `packages/ice/legoesm/ice/{config.py,sea_ice.py}` and `scripts/run/run_omip_core2.py` — main's precipitation-closure additions changed the response pytree and pond/snow freshwater paths for every SI3 caller despite no textual conflict.  Per Decision 190, the `core2_precipitation_closure` selector defaults on.  Both CORE-II production constructors retain their explicit selections.  Nine certified NEMO pond/remap configurations are explicitly pinned False for literal configuration provenance even though these fixtures do not move bits when unpinned.  The four m6 parent/current control configurations do move response structure or bits and are pinned False for control validity.  Main's precipitation-routing test deliberately retains and exercises the committed True default.

## Citation reconciliation

The merged gate retains both parents’ independent maps. All moved citations were re-anchored by unique symbol and explicit extent. Obsolete duplicate literal entries were removed rather than allowed to shadow a key. The pre-commit audit result is 42 passing functional citation tests plus two expected provenance refusals caused solely by the uncommitted merge; the clean-commit rerun is recorded below.

## Pre-commit validation

- Required receipt-citation gate: `42 passed`, with only the two real-receipt
  tests refusing the intentionally dirty, uncommitted merge worktree.  There
  were no citation, map-cardinality, or source-anchor failures.
- Operator landing selection excluding that provenance-sensitive gate:
  `1 failed, 127 passed in 1025.99s`; the sole failure exposed the non-textual
  KPP validation-order interaction listed above.  After moving the unchanged
  refusal ahead of closure evaluation, its exact node emitted
  `1 passed in 1.47s`.
- D5 and O1–O16 production exactness selection: the combined run emitted
  `1 failed, 54 passed, 1 error in 839.78s`.  The O4 error was solely the
  package-only `PYTHONPATH` hiding the repository-local `scripts` package from
  its child process; with the repository root included, the same production
  scorer emitted `1 passed in 269.98s`.  The remaining O7
  `sum_first_category` planted-control failure was reproduced unchanged at the
  exact lane parent (`1 failed in 2.08s`) and is the pre-existing qsb
  tie-handling control already recorded by the SI3 receipts.  All three
  canonical O8–O16 representative score nodes and all eight finalizer tests
  passed.
- The clean-merge operator battery, provenance SHA gate, and merge/main-alone
  default ice selections are run after the merge object exists and are quoted
  in the landing handoff.
- The merge-aware prospective provenance SHA gate emits
  `1512 passed in 4.97s`; the exact lane parent emits
  `1401 passed in 2.28s`.
- The certified O12 parent/current comparison plus main's two CORE-II rain
  closure regressions emit `3 passed in 21.43s` after adding the explicit
  precipitation-closure selector.
## Decision 190 follow-up measurements and fixes

1. **Committed-default pond routing.**  The original filename-wide failure was
   caused by the four O12 m6 parent/current control configurations (slab,
   legacy dynamics, v2 one category, and v2 five categories): removing their
   explicit False selections makes the control fail on the changed response
   pytree.  By contrast, removing the other nine explicit-False selections
   from O12 production wiring, lid restart, multicategory JPL5, Decision 184c,
   lid-on one- and five-category validation, lid-off one- and five-category
   validation, and the O13 production remap call-site still emits `85 passed`.
   Those nine pins remain because certified NEMO-comparison configurations
   state NEMO's setting literally, not because these particular fixtures move
   bits.  The master scorer had one additional executed `si3_lev` configuration
   without the pin; with the True default its precipitation-concentration
   response is `0.8` instead of `None`.  It is now explicitly False and has a
   production-path assertion.  The remaining `si3_lev` omissions in O12 are
   validation-refusal inputs: the missing-`namthd_pnd` production check, and
   (for each of lids-on and lids-off) the all-stages composition, four
   dynamics/advection selections, two unsupported remap selections, the
   one-category remap selection, and the ridging selection.  Each is rejected
   before a sea-ice step can produce a response.  The grep found no other
   executed NEMO-comparison `si3_lev` configuration without the False pin.

2. **Inactive thermodynamics without forward drift.**  Resume 4 guarded the
   thermodynamic operands but then selected the live entry state as the
   *forward* value before open-water formation.  The operator's four-column,
   two-step probe proved that this changed the lane: among other differences,
   step-1 `S_bulk` moved from `0.1` to `0.0`, sub-epsilon enthalpy was erased,
   and step-2 freezing thickness moved from `0.04984625211138035` to
   `0.04976158051528507`.  Resume 5 restores the exact `920c61d884` expression
   and uses a custom reverse rule whose primal is exactly `post_da`.
   Hard-coded eager and JIT bit patterns cover all nine state leaves on open
   water, freezing open water, sub-epsilon ice, and freezing sub-epsilon ice
   for two steps.  Both plants failed on `ae5a8821d2` before the fix.

   For `qlead=-40`, the lane and main ice-volume forwards agree and the fixed
   derivatives with respect to `qlead`, bottom temperature, and salinity remain
   `(-1.3096887137695114e-08, 8.220414129866561e-07,
   4.470021523873651e-08)`, equal to main to roundoff and to central differences.
   On the measured lane-only sub-epsilon state (`concentration=5e-11`,
   `h_ice=0.5`), AD gives `1.0` for
   `d(concentration_out)/d(concentration_in)` and the lane forward's wholly
   sub-epsilon central difference gives `0.9999999999852521`.

3. **Non-vacuous AD plants.**  The remap and production-pytree plants remain.
   The helper `_guard_inactive_columns` still exists, so the deleted
   `test_inactive_column_guard_stops_h_ice_cotangent` is restored verbatim in
   intent: inactive cotangents are zero and active cotangents are one.  The
   formation test now compares the lane forward to its own finite difference,
   while retaining main's values only for the agreed ice-volume forward.

4. **Provenance SHA history.**  The one-hop exemption accepted all three
   reported bypass classes.  The replacement walks to the pre-cutoff base and
   requires token continuity on the mapped line at every revision.  The four
   original plants exercised only `_sha_survives_line_history`; reverting the
   real `inherited_parent_sha` entry point to one hop therefore still emitted
   `1516 passed`.  A temporary three-commit Git history now drives the bypass
   through `validate_document`; it fails under the one-hop revert and keeps the
   full-history implementation bound end to end.

5. **CEN2 certified direct callers.**  The ORCA2 round-257 writer gate and
   GYRE round-72 tracer-stage probe pass `_certified_nemo_storage=True`
   explicitly.  Resume 4 tested the wrappers themselves, so a direct caller
   could bypass a still-correct wrapper.  Resume 5 binds `measure` to all four
   ORCA2 wrapper calls and `_live_stage_rows` to the GYRE wrapper call.  Plants
   routing one ORCA2 call and the GYRE call directly to the lower helper fail
   with `3 == 4` and `0 == 1`, respectively.

6. **Slow parent controls.**  Project addopts deselect both controls under an
   ordinary run.  Explicit final-HEAD `-m slow` results are recorded below.

7. **Degenerate-column reach and NEMO semantics.**  On clean main
   `edd34c2446`, an instrumented real call emitted
   `IceColumnModel.step -> step_sea_ice -> _si3_step_with_trace ->
   si3_column_step_arrays` with entry concentration min/max `(5e-11, 5e-11)`.
   `IceColumnModel.create` exposes `concentration_init`, so main production can
   reach the disputed range.  `run_omip_core2` cannot currently do so: it
   constructs the default `SeaIceConfig` (`thermo_scheme="zero_layer"`) and a
   `DynamicSeaIceState`, whereas the SI3 dispatch requires
   `thermo_scheme="si3_bl99"` and `SI3ColumnState`.  Validation/scorer callers
   also reach the function, but the certified gates contain no near-empty
   columns.

   Neither the lane nor main degenerate forward is fully NEMO-literal.  NEMO
   gathers thermodynamics only for `a_i > epsi10`
   (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd.f90:181-222`), runs lead growth over the
   geographic grid at `:235`, and then calls correction at `:238`.
   `ice_var_zapsmall` first sets `h_i=0` for `a_i <= epsi10`
   (`icevar.f90:625-627`) and, when the minimum area/volume/thickness is below
   `epsi10`, clears energy and temperature (`:664-689`) and finally clears
   area, ice/snow volume, surface temperature, age, and ponds (`:695-720`).
   The production transcription has neither a literal inactive gather/scatter
   nor the final `ice_cor`/`ice_var_zapsmall` stage.  The merge kept the lane's
   already-certified forward, so resume 5 changes no primal value and registers
   the production-reachable semantic choice below.

## Resume 6 reverse-only follow-up

The production forward remains the certified lane expression.  The operator
probe is byte-identical to its `fwd_old.txt` reference, and both eager and JIT
forward bit cards remain unchanged.  Only reverse routing and dtype
normalization changed:

- the entry-concentration cotangent is now selected by structural carry
  (`post_dh.h_ice != 0`) rather than accidental value equality;
- open-water `age_volume` is carried from the live entry state, restoring its
  identity cotangent;
- the entry gather guard distinguishes nonzero thickness from the discarded
  thermodynamic lane, restoring the sub-epsilon thickness derivative;
- frazil/do forcing leaves are cast to the state dtype, restoring the
  certified mixed-float behavior and a finite float32 concentration gradient;
- the redundant active-column mask in the custom reverse rule was deleted;
  the cumulative-remap guard was also deleted after both its production sweep
  and tie-survivor plants stayed green without it; and
- the original `h_ice=0` formation fixture is restored, with `h_ice=0.5` kept
  as a second case.

At the pre-fix `6326624aac` tree, the committed sweep fails in eager mode:
the first 16 cases have worst relative error `1.0`, and the four positive-
concentration/zero-thickness cases are non-finite.  The focused defects are
`d(concentration_out)/d(concentration_in)=1` and
`d(age_volume_out)/d(concentration_in)=3600` at zero concentration and zero
thickness (both targets are zero),
`d(age_volume_out)/d(age_volume_in)=0` (target one), and a zero sub-epsilon
thickness derivative where the lane finite difference is approximately
`4.775e-6`.

The 4,968-pair oracle covers 24 production cases, nine exit leaves, and 23
entry-state/forcing leaves.  It uses main AD wherever the main and lane
forwards agree in a neighborhood, otherwise the lane forward: central
differences away from a boundary, a positive-side difference at concentration
zero, and a below-threshold difference at `epsi10`.  Eager and compiled values
are independently pinned; their one reviewed source-rounding difference is
the active case's `h_ice`-versus-concentration slope.

| concentration | h_ice | qlead | eager worst relative | JIT worst relative |
|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0 | `0.000e+00` | `0.000e+00` |
| 0 | 0.0 | -40 | `0.000e+00` | `0.000e+00` |
| 0 | 0.5 | 0 | `5.307e-15` | `0.000e+00` |
| 0 | 0.5 | -40 | `5.307e-15` | `0.000e+00` |
| 1e-12 | 0.0 | 0 | `0.000e+00` | `0.000e+00` |
| 1e-12 | 0.0 | -40 | `0.000e+00` | `0.000e+00` |
| 1e-12 | 0.5 | 0 | `5.307e-15` | `0.000e+00` |
| 1e-12 | 0.5 | -40 | `5.307e-15` | `0.000e+00` |
| 5e-11 | 0.0 | 0 | `0.000e+00` | `0.000e+00` |
| 5e-11 | 0.0 | -40 | `0.000e+00` | `0.000e+00` |
| 5e-11 | 0.5 | 0 | `5.307e-15` | `0.000e+00` |
| 5e-11 | 0.5 | -40 | `5.307e-15` | `0.000e+00` |
| 1e-10 | 0.0 | 0 | `0.000e+00` | `0.000e+00` |
| 1e-10 | 0.0 | -40 | `0.000e+00` | `0.000e+00` |
| 1e-10 | 0.5 | 0 | `5.307e-15` | `0.000e+00` |
| 1e-10 | 0.5 | -40 | `5.307e-15` | `0.000e+00` |
| 2e-10 | 0.0 | 0 | `0.000e+00` | `0.000e+00` |
| 2e-10 | 0.0 | -40 | `0.000e+00` | `0.000e+00` |
| 2e-10 | 0.5 | 0 | `2.715e-14` | `0.000e+00` |
| 2e-10 | 0.5 | -40 | `2.715e-14` | `0.000e+00` |
| 0.5 | 0.0 | 0 | `0.000e+00` | `0.000e+00` |
| 0.5 | 0.0 | -40 | `0.000e+00` | `0.000e+00` |
| 0.5 | 0.5 | 0 | `2.715e-14` | `0.000e+00` |
| 0.5 | 0.5 | -40 | `7.880e-13` | `0.000e+00` |

The maximum is `7.880e-13`, below the required `1e-6` relative threshold.
The remap-guard source removal changes three later parenthesis-occurrence
anchors without moving their lines; the executable citation map was updated
accordingly, and its full non-vacuity audit remains green.

## Resume 7 independent reverse oracle and remap correction (superseded in Resume 8)

Resume 6's two embedded reverse-oracle blobs were not independent: one was a
dump of `920c61d884` JIT AD and the other a dump of `0ae642c422` JIT AD.  Both
were deleted in Resume 7, and the cumulative-remap no-material guard was
restored in `packages/ice/legoesm/ice/bitz_lipscomb.py`.  However, Resume 7's
initial generator and production tiny-snow test had four residual defects
corrected in Resume 8 below: (1) `test_si3_column_step_remap_reverse_mode_finite_at_tiny_snow`
differentiated only `exit.h_snow`, which pruned the remap enthalpy branch in
`jax.grad` and passed even when the `_cumulative_remap` guard was deleted;
(2) `established_lane_branch` used the lane's own AD as the target for 35
entries across cases 18, 19, 22, and 23; (3) 19 `S_layers` (`input_index = 7`)
entries at `S = 0` used `main_ad` while the `1e-8` forward step disagreed with
`main_ad` by 2x to 2,070x across the `1.146e-11` nonlinearity scale; and (4) a
global `3e-6` absolute floor gated 4,614 of 4,968 entries.

## Resume 8 and Resume 9 oracle, tolerance, tiny-snow, and citation-span corrections

1. **Full-exit tiny-snow production test (`tests/unit/test_bitz_lipscomb.py`):**
   `test_si3_column_step_remap_reverse_mode_finite_at_tiny_snow` now
   differentiates the entire `.exit` state pytree via `jax.jacrev` over all
   nine leaves (`concentration`, `h_ice`, `h_snow`, `T_surface`, `e_ice`,
   `e_snow`, `S_bulk`, `S_layers`, `age_volume`) with respect to
   `entry.h_snow = 3e-200`, asserting every leaf Jacobian is finite and the
   surviving snow-thickness derivative is `[[1.0]]`.  Deleting the
   `_cumulative_remap` no-material guard (`bitz_lipscomb.py:987-999`) now fails
   immediately with `AssertionError: concentration: [[nan]]` (with `NaN` in all
   nine exit leaf Jacobians).
2. **Zero lane-AD targets and one-sided exclusion rule (`scripts/validate/ocean_fidelity/testcases/si3_reverse_sweep_oracle.py`):**
   `_sample` computes `jacrev` only when `sha == MAIN_SHA` (`has_jacobian=False`
   on `LANE_SHA`), and `build_oracle` raises `ValueError` if `"jacobian"` is
   present in the lane sample.  Every scored entry targets either `main_ad`
   (`3744` entries where main and lane forwards agree in the sampled
   neighbourhood and `main_ad` agrees with `lane_fd` within `1e-6` relative) or
   a finite difference of the lane forward (`389` `lane_fd_central`, `736`
   `lane_fd_forward`, `26` `lane_fd_backward`; `method_counts = {"main_ad": 3744, "lane_fd_central": 389, "lane_fd_forward": 736, "lane_fd_backward": 26, "excluded_nondifferentiable_boundary": 73}`).
   Specifically (regenerated verbatim from the committed fixture):
   - `[18,3,11]` (`T_surface / dqns_ice` at `(2e-10, 0.5, 0.0)`): `method = 2`
     (`lane_fd_forward` at `step = 0.0001`), `target = -0.00024972905521281064`,
     `fd_rounding_bound = 1.1368683772161603e-09`, `main_ad = -0.0002497283729832977`,
     `tip_eager = -0.0002497283729832977`, `tip_jit = -0.0002497283729832977`,
     `err_eager = 6.822295129543278e-10 <= fd_rounding_bound = 1.1368683772161603e-09`
     (`rel_eager = 2.7318788051032145e-06`), while `[18,3,9]` (`T_surface / qns_ice`
     at `(2e-10, 0.5, 0.0)`) has `method = 0` (`main_ad`),
     `target = 0.050971963040420495`, `tip_eager = tip_jit = 0.050971963040420495`,
     and `fd_central = 0.05097196293490924` (`rel(main_ad, fd_central) = 2.0699860515303513e-09`);
   - `[22,3,11]` (`T_surface / dqns_ice` at `(0.5, 0.5, 0.0)`): `method = 2`
     (`lane_fd_forward` at `step = 0.0001`), `target = -0.00024972905521281064`,
     `fd_rounding_bound = 1.1368683772161603e-09`, `main_ad = -0.0002497283729832977`,
     `tip_eager = -0.0002497283729832977`, `tip_jit = -0.0002497283729832977`,
     `err_eager = 6.822295129543278e-10 <= fd_rounding_bound = 1.1368683772161603e-09`
     (`rel_eager = 2.7318788051032145e-06`), while `[22,3,9]` (`T_surface / qns_ice`
     at `(0.5, 0.5, 0.0)`) has `method = 0` (`main_ad`),
     `target = 0.050971963040420495`, `tip_eager = tip_jit = 0.050971963040420495`,
     and `fd_central = 0.05097196293490924` (`rel(main_ad, fd_central) = 2.0699860515303513e-09`);
   - `[22,3,7]` (`T_surface / S_layers` at `(0.5, 0.5, 0.0)`): `method = 0`
     (`main_ad`), `target = 1014193911.7999892`, `tip_eager = 1014193911.8006747`,
     `tip_jit = 1014193911.8006747` (`rel_eager = rel_jit = 6.758603132412692e-13`),
     agreeing with `fd_central = 1014194316.4398981` at `step = 5e-15` to
     `rel(main_ad, fd_central) = 3.989767072745935e-07`; and
   - `[19,1,2]` (`h_ice / h_snow` at `(2e-10, 0.5, -40.0)`): `method = 3`
     (`lane_fd_backward` at `ref_step = 1e-06`), `target = 6.869234348005904e-06`,
     `fd_rounding_bound = 1.3877787807814457e-11`, `tip_eager = 6.869232846683468e-06`,
     `tip_jit = 6.869232846683468e-06`, `err_eager = err_jit = 1.5013224360536578e-12`,
     `rel_eager = rel_jit = 2.1855746361157155e-07` (after a 4-step backward FD
     refinement `1e-8`: `6.869504964868e-6`; `1e-7`: `6.869296798051039e-06`;
     `1e-6`: `6.869234348005904e-06`; `1e-5`: `6.869232960227e-6` confirmed the
     AD was right and the `1e-7` step difference was float64 subtraction noise on
     `exit.h_ice ~ 0.05`).
   For the `36` ice-thermodynamic entries `[c, 1|3|4, 2]` (`outputs h_ice,
   T_surface, e_ice` with respect to `h_snow` at `h_snow = 0.0` on the `12`
   `h_ice = 0.5` cases), the base point `h_snow = 0.0` executes the
   `h_snow <= 0.0` no-snow branch while `h_snow > 0.0` crosses the strict
   `h_snow > 0.0` guard into the snow-present branch (so the right FD disagrees
   on `31` of the `36` entries); as stated in `SELECTION_RULE` and
   `fixture["selection"]`, those `36` entries are scored against the left
   (backward) difference at the `1e-06` refinement step along the executed
   `h_snow <= 0.0` branch (`17` match `main_ad` with `method = 0` and `19` use
   `lane_fd_backward` with `method = 3`).
   Statement-boundary candidates (`[c, :, 1]` at `h_ice = 0.0`, and
   `[c, :, 16]`, `[c, 2, 2]`, `[c, 5, 2]` at `h_ice = 0.5`) are excluded only
   where measured left and right one-sided lane finite differences disagree by
   more than both `1e-6` relative error and `1x` the entry's own
   `fd_rounding_bound` (`|fd_right - fd_left| > max(1e-6 * max(|fd_left|,
   |fd_right|, tiny), fd_rounding_bound)`).  Entries `[16,1,1]` (`h_ice / h_ice`
   at `(2e-10, 0.0, 0.0)`) and `[20,1,1]` (`h_ice / h_ice` at `(0.5, 0.0, 0.0)`)
   have `fd_left = 0.0`, `fd_right = 1.3234889800848443e-16`, and
   `fd_rounding_bound = 1.3234889800848443e-16` (`step = 1e-07`), so
   `|fd_right - fd_left| <= fd_rounding_bound`; they are not excluded and are
   scored normally against `target = 1.3234889800848443e-16` (`method = 2`,
   `lane_fd_forward`), passing with `tip_eager = tip_jit = 0.0` and
   `err_eager = err_jit = 1.3234889800848443e-16 <= fd_rounding_bound = 1.3234889800848443e-16`.
   Only the `73` genuine non-differentiable entries are excluded (`method = 4`,
   `excluded_nondifferentiable_boundary`, `target = NaN`), each recorded by
   `[case, output, input]` index with both one-sided values in
   `oracle["excluded_nondifferentiable"]`.
3. **`S_layers` entries at `S = 0` (`input_index = 7`):**
   Setting `_INPUT_STEPS[7] = 5.0e-15` with a central stencil resolves the
   smooth `S_layers ~ 1.146e-11` nonlinearity scale at `S = 0`.  Across the
   `176` `S_layers` entries that use `method = 0` (`main_ad`) (including `16` of
   the `19` previously discrepant entries), `lane_fd` agrees with `main_ad` to a
   worst relative difference of `3.989767072745935e-07` (at `[2,3,7]`, where
   `target = main_ad = 1014193911.7999892` and `fd_central = 1014194316.4398981`).
   The remaining `40` `S_layers` entries use `method = 1` (`lane_fd_central`):
   `37` zero-target open-water entries (`err_eager = err_jit = 0.0`) plus `3`
   active-melt `h_ice / S_layers` entries at `exit.h_ice ~ 0.05` (`[7,1,7]` with
   `target = -0.646011022453763`, `tip_eager = tip_jit = -0.6454089331773042`,
   `err_eager = err_jit = 0.0006020892764587726 <= fd_rounding_bound = 0.0013877787807814457`,
   `rel_eager = rel_jit = 0.0009320108411956177`; `[11,1,7]` with
   `target = -32.270019989510956`, `tip_eager = tip_jit = -32.270295740909674`,
   `err_eager = err_jit = 0.00027575139871771626 <= fd_rounding_bound = 0.0013877787807814457`,
   `rel_eager = rel_jit = 8.545053349732427e-06`; and `[15,1,7]` with
   `target = -64.5407338684123`, `tip_eager = tip_jit = -64.54028348890222`,
   `err_eager = err_jit = 0.0004503795100845309 <= fd_rounding_bound = 0.0013877787807814457`,
   `rel_eager = rel_jit = 6.978221087519379e-06`), so across all `216`
   `S_layers` entries the worst relative difference is `0.0009320108411956177` at
   `[7,1,7]`, covered by its own `fd_rounding_bound = 0.0013877787807814457`.
4. **Per-entry FD rounding bound (`fd_rounding_bound`):**
   The global `3e-6` floor is removed.  Each entry stores its own float64
   subtraction rounding bound (`FD_ROUNDING_BOUND_FORMULA`), using
   `s(x+-d) = max(|f(x+-d)|, d)` on diagonal state entries (`output_index ==
   input_index`) where the perturbed state leaf enters the column step at scale
   `>= d` and `s(x+-d) = |f(x+-d)|` otherwise, with `s(x) = |f(x)|`:
   `(|spacing(s(x+d))| + |spacing(s(x-d))|) / (2*d)` for central differences,
   `(|spacing(s(x+d))| + |spacing(s(x))|) / d` for forward differences,
   `(|spacing(s(x))| + |spacing(s(x-d))|) / d` for backward differences, and the
   `0.5`-weighted sum of the two forward-step bounds for two-step averaged
   evaporation differences.  Every scored entry must satisfy `rel <= 1e-6` or
   `|err| <= 1.0 * fd_rounding_bound`.  Entry `[18,1,0]` (`h_ice / concentration`
   at `(2e-10, 0.5, 0.0)`, `target = 0.0`, `tip_eager = 4.76837158203125e-07`,
   `tip_jit = 5.873816633084061e-07`) has `exit.h_ice = 0.5`
   (`spacing(0.5) = 5.551115123125783e-17`) and central step `d = 2e-11`, so its
   own FD rounding bound is
   `(5.551115123125783e-17 + 5.551115123125783e-17) / (4e-11) = 2.7755575615628914e-06`,
   which covers both eager and JIT errors directly without any global floor or
   per-entry patch.
5. **Rebuilt sweep and restored citation spans:**
   On `0ae642c422`, the rebuilt sweep fails on `151` entries in eager mode and
   `151` entries in JIT mode (first failure `(2, 3, 7)` in both modes).  On the
   repaired HEAD, `0` entries fail in eager and `0` in JIT (`19 passed` across
   `tests/unit/test_bitz_lipscomb.py`).  With the `_cumulative_remap` guard
   restored in `bitz_lipscomb.py`, the file's parenthesis count matches
   `920c61d884`, so the three unmoved `bitz_lipscomb.py` citation spans in
   `si3_r21_o8_o12_score_preregister.md`, `si3_r21_o8_o12_score_receipt.md`,
   and `si3_r3_d5_receipt.md` (`1028-1277`, `1301-1320`, `1718-1783`) and their
   four `(')', count)` anchors in `nemo_testcase_receipt_citation_gate.py` are
   restored to their original full extents.

## Final clean-checkout verification

| Selection | Final line |
|---|---|
| Every requested `test_si3_o*.py` / `test_nemo_si3_*.py` file | each file started in a fresh serial process (excluding `test_si3_o6_rdgrft_cor.py` and `test_nemo_si3_o4_orca2_prather.py`, run by the operator with O6 already `28 passed` on `006f20b57c`): 36 files completed green; scalar math emitted its unchanged `1 failed, 4 passed in 1.11s`; O8–O16 acquisition emitted `27 passed` plus the `timeout=1200` subprocess limit on `test_preflight_compiles_and_runs_real_patched_source_without_nemo` (`1 failed, 27 passed in 1210.96s (0:20:10)`, previously `28 passed in 1114.96s (0:18:34)`) |
| `test_bitz_lipscomb.py` | `19 passed in 253.76s (0:04:13)` |
| Operator forward probe | byte-identical to `fwd_old.txt` in both eager and JIT (`eager_cmp_status=0 jit_cmp_status=0`) |
| `test_sea_ice_precip_routing.py` | `7 passed in 2.48s` |
| `test_si3_surface.py` | `1 failed, 23 passed, 1 xfailed in 38.94s`; only the reproduced lane-parent `sum_first_category` plant |
| Certified O8–O16 exactness scorer | `30 passed in 786.57s (0:13:06)` |
| O5 slow parent/current control | `1 passed in 599.22s` |
| O6 slow merge-parent/current control | `1 passed in 774.76s` |
| CEN2 direct-caller wrappers | ORCA2 `10 passed`; GYRE `4 passed` |
| Landing battery | 7/7 files in fresh serial processes; `172 passed` (component final lines: `44 passed in 9.32s`, `6 passed in 5.47s`, `71 passed in 43.45s`, `25 passed in 322.97s (0:05:22)`, `16 passed in 6.43s`, `2 passed in 537.90s (0:08:57)`, and `8 passed in 64.79s (0:01:04)`) |
| Receipt citation gate | `44 passed in 9.32s` |
| Provenance SHA gate | `1517 passed in 23.70s` |

## Failure inventory and attribution

Every observed stable failure is classified here.

| Test node(s) | Classification | Disposition / cause |
|---|---|---|
| `test_si3_surface.py::test_libm_pow_matches_glibc_and_differentiates` | merge-introduced, fixed here | original merge dereferenced lazy `_LIBM=None`; cherry-picked `e4be5cee4e29aa2c71bf5802e2ccee7df5ca0fb9`, using `glibc_libm().pow` |
| `test_bitz_lipscomb.py` open-water reverse-mode regressions | merge-introduced, corrected here | preserved the lane's exact `post_da` forward and routed only the valid inactive concentration cotangent to the entry value; eager/JIT two-step bits equal `920c61d884` |
| main's remap-only Bitz regression (`test_cumulative_remap_reverse_mode_finite_at_zero_thickness` / `test_si3_column_step_remap_reverse_mode_finite_at_tiny_snow`) | merge-introduced, guard restored in resume 7, production plant fixed in resume 8 | the helper guard is required, but resume 7's production test differentiated only `exit.h_snow` (which pruned the remap enthalpy branch and passed with the guard deleted); resume 8 differentiates the full `.exit` pytree via `jax.jacrev`, failing with `AssertionError: concentration: [[nan]]` (NaN in all 9 leaves) when the guard is deleted |
| landing battery KPP validation-order node recorded above | merge-introduced, fixed in the original catch-up | main closure evaluation ran before the lane's unchanged fail-closed refusal; ordering was corrected and the node passed |
| 15 nodes in `test_nemo_testcase_receipt_citation_gate.py` on the first clean follow-up run | follow-up source-shift, fixed here | the AD guards, dual CEN2 path, and catch-up-2 core changes moved 149 mapped endpoints; all audited Markdown citations and the executable map were re-anchored with literal insertion maps and explicit replacement anchors; pre-commit result is `42 passed` plus the two expected dirty-worktree provenance refusals |
| three bypass classes in `test_post_20261029_provenance_sha_gate.py` | follow-up weakness, fixed here | carry-forward now requires the identical token on one mapped logical line through the complete first-parent history to the pre-cutoff base; later re-edits, copied/retargeted tokens, and moved tokens fail committed plants |
| `test_si3_surface.py::test_production_path_plants_fire[sum_first_category]` | pre-existing lane | unchanged qsb tie-handling planted-control failure; reproduced at exact lane parent |
| `test_nemo_si3_o4_prather_acquisition.py::test_o4_production_scorer_and_plants` initial setup error | invocation/environment, fixed invocation | child process could not import repository-local `scripts` until `.` was included in `PYTHONPATH`; production scorer then passed |
| `test_si3_o6_rdgrft_cor.py` final whole-file run | runtime limit, no failure observed | 16 of 28 selected nodes emitted passing dots before the fresh 30-minute process timed out; the receipt records the timeout rather than claiming a completed file |
| `test_nemo_si3_o4_orca2_prather.py` final whole-file run | runtime limit, no failure observed | eight of nine nodes emitted passing dots; the last node did not finish within a fresh 30-minute process, so the receipt records the timeout rather than claiming a completed file |
| `test_nemo_si3_scalarmath_v2_gate.py::test_full_v2_gate_and_plants` | pre-existing external build-artifact drift | clean final code emits `A MY_SRC is not verbatim`; the same named file at original clean HEAD `920c61d884` emits the identical `1 failed, 4 passed`, and no NEMO executable was run |
| `test_flag_reachability_audit.py::test_every_resolver_read_reaches_the_leaf` | pre-existing on newer main | reproduced identically on `edd34c24469c`; main's CLUBB host table is rejected by its own strict validity check |
| four `test_pipeline_profile_convection.py` Zhang–McFarlane nodes | pre-existing on newer main | reproduced identically on `edd34c24469c`; strict validation rejects CMT on the cubed-sphere/CD-grid test host |
| `test_pipeline_profile_convection.py::TestMicrophysicsWiring::test_micro_steps_finite[p3]` | pre-existing on newer main/environment | reproduced identically on `edd34c24469c`; sandbox cannot create `/home/dbalwada/.cache/legoesm/p3` |
| `test_pipeline_profile_convection.py::test_bechtold_convective_rain_reaches_surface_precip_929` | pre-existing on newer main | reproduced identically on `edd34c24469c`; zero-efficiency expectation conflicts with the current positive precipitation output |
| `test_scm_rce_thermo_objective.py::test_the_cape_thresholds_are_trainable_not_tier_zero` | pre-existing on newer main | reproduced identically on `edd34c24469c`; six registered thresholds versus required eight |
| `test_zm_clubb_scalar_fields.py::test_tables_are_real_fields_appended_at_the_tuple_end` | pre-existing on newer main | reproduced identically on `edd34c24469c`; current `ExperimentConfig` tail ordering differs from the asserted table tail |

## Decision register

DECISION 190 applied: default True; every affected certified NEMO-comparison
configuration is explicitly False, while default-behavior tests retain True.

DECISION 193 applied (user, 2026-10-31): near-empty columns (concentration
<= epsi10) keep the lane's certified degenerate forward for now.  The complete
NEMO inactive gather/scatter plus `ice_cor`/`ice_var_zapsmall` is a separate
later round, and the production forward switches only after that round is
certified.  Main's pre-merge degenerate forward is not restored.

The options as registered by resume 5 were:

1. Keep the certified lane degenerate forward indefinitely.
2. Replace it with main's pre-merge degenerate forward.
3. Transcribe the complete NEMO inactive gather/scatter plus
   `ice_cor`/`ice_var_zapsmall`, certify near-empty records, and switch only
   with a new explicit decision.

Neither existing version is the complete NEMO statement sequence.  This round
implements no option: the certified lane forward remains bit-identical, and
the NEMO transcription is queued.
