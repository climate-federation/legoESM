# NEMO testcase L2 GYRE phase-3 round-49 receipt

Date: 2026-09-11. CPU/fp64/libm. Producer branch
`fidelity/nemo-testcases-l2-gyre-codex2`; evidence root
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round49/`.

## Verdict and compiled alignment

The task's EEN/ENS premise is retracted: this build selects ENE
(`GYRE_OMIP_L2_P3_SM_R46KT2/EXP00/namelist_cfg:165-167`). Stage 1 executes
HPG→LDF→VOR→KEG→ZAD with LDF `(Kbb,Kbb)`
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-175`); stages 2/3
execute HPG→VOR→ADV on Kmm
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:431-480`), and stage
3 later executes LDF `(Kbb,Kmm)` (`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:690-712`).

| ladder | compiled operands/statements | measured substitution result |
|---|---|---|
| LDF | live F thickness/curl, Kbb T/U/V thickness/divergence, Kmm U/V output divisors (`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`) | baseline s1 U/V `17400/17100`; full thickness `10626/10411`; literal order `974/866`. Stage 3 literal `302/428`. Recorded stage-3 r3f did not move them: **REFUTED**, no production route. |
| VOR | CRV curl+f, live e3f division, `(metric*e3)*velocity`, pair sums, stored reciprocal association (`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynvor.f90:536-573`) | baseline residual stages 1/2/3 U/V `4/2,1/0,2/1`; literal source order yields `0/0,0/0,0/0`: **CONFIRMED and landed shared**. |

Artifacts: `round49_ldf_ene_ladder_refuted.json`,
`round49_literal_candidate.json`, `round49_recorded_r3f.json`, and clean-stamped
`round49_ene_confirmed_a2f43e.json`. ENE and LDF wet-row plants exit 1 and
move respectively 17,400 cells (`7.925e-6`) and 17,400/17,100 cells
(`2.721e-7/2.660e-7`). Header, truncation, and false-stamp plants also exit 1.

## Rule 12, trajectory, and ZAD

| card | ENE change disposition |
|---|---|
| GYRE-zco | TESTED: six admitted post-VOR rows bit-exact; 14-row trajectory move gate PASS (largest worsening 1 row ULP; limit 2) |
| LOCK_EXCHANGE-zco | VALUE-INERT: instantiated flux-form card never calls vector PV flux |
| OVERFLOW-zps | VALUE-INERT: same flux-form exclusion |
| DINO | VALUE-INERT: instantiated `een_total`; new guard is ENE/ENE-total only |
| ORCA2 | UNMEASURED-WITH-SPEC: no native card on this branch; acquire stage-entry U/V, e3f/r3f, all four metrics and reciprocals, post-VOR U/V accumulators, and kt=1..10 native trajectory |

Round-48 before and round-49 after first-over-bar are both kt=2 U/V; kt=2
maxima remain `2.7478404751243857e-12 / 3.305560306813421e-12`. The after gate
ran through kt=10 and remains DEBT. ZAD was not re-run: the frozen trigger was
“after both candidates land,” and LDF was refuted; round 47's 57 worsened rows
remain controlling.

Focused suites: 15 operator tests and 49 card/routing tests pass. Adversarial
review before landing caught the wrong EEN premise, two false causal
predictions, and an inert ENE plant; after repair, citation-map audit is clean,
plants are red, LDF is not production-selected, and ORCA2 remains explicit.

## Disposition

ASKED: source transcription, LDF/VOR ladder, conditional ZAD retest, kt=1..10,
Rule 12. UNASKED: no bar/config/card guard/NEMO-run changes. Open: isolate the
remaining LDF last-ULP association; build the native ORCA2 card/record above;
only then reconsider ZAD.
