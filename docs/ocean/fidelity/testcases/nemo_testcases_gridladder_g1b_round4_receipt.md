# Grid ladder rung G1b — round 4 receipt (gridladder_r27)

Status: **LANDED** (one cited, G1b-selected source statement and one corrected
instrument). G1b is **NOT certified**: GIVEN-NEMO-ENTRY remains at the bar over
the admitted record, but INDEPENDENT first exceeds the 1e-15 bar at kt 27 and
has a worst sea-level score of 1.70e-14.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1b_round4.md`
(`6cd3a4335`, before any round-4 measurement). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round27/`.
Record build (`R4REF`):
`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo`; full-card Decision-122 pointer
(`ORCA2`): `ORCA2_OMIP_L4/BLD/ppsrc/nemo`.

## Results

1. **HPG/KEG split — no production defect landed.** On G1b at kt 2, both
   operands are nonzero: HPG/KEG maxima are 9.99e-9/1.44e-10 on U and
   8.52e-9/1.68e-10 on V. Each operand is bit-identical to the offline replay
   on every wet face. The first non-bit operation is the diagnostic `HPG + KEG`
   addition: 5 U faces, 8.27e-25; after the production HPG → VOR → KEG
   accumulator, the stage-1 RHS remains 4 U / 0 V faces, maximum 1.65e-24.
   NEMO's executed calls are `R4REF/stp2d.f90:138`, `:144`, `:161`; the
   in-place KEG subtraction is `R4REF/dynkeg.f90:128-130`. The same full-ORCA2
   statements are `ORCA2/stp2d.f90:141`, `:147`, `:164` and
   `ORCA2/dynkeg.f90:128-130`. An **unregistered follow-up** replaced selected
   routine barriers with source-round materializations; 4/0 and 1.65e-24 were
   unchanged. The RHS residual is still OPEN.
2. **Unregistered instrument correction and retraction.** The round-3
   instrument always replayed the
   thickness-weighted `ELSE` statement. G1b actually executes the preceding
   `ln_dynadv_vec` velocity branch. Its one-level barotropic correction nearly
   canceled the wrong replay and manufactured the reported 35 U / 49 V row at
   3.5e-18. The tool now selects the card's compiled branch and cannot print
   that withdrawn claim. Citation: `R4REF/stprk3_stg.f90:366-370`; the same
   line in the Decision-122 full build is `ORCA2/stprk3_stg.f90:467-471`.
3. **Vector stage update (fe88ed406), with comparisons separated.** The
   one-change production comparison holds the G1b card, entry and selector
   fixed: before the vector arm implemented the selected written order, the
   kt-2 stage fields had 8 U / 3 V unequal at stage 1, 48/87 at stage 2 and
   57/81 at stage 3; afterward each is 0/0. Separately, the raw-operator
   comparison holds the traced operands fixed and stops before the barotropic
   correction: with `rk3_stage_velocity_written_order` off, the raw
   multiply/add/mask differs from one-operation-at-a-time NumPy on 107 U / 192
   V faces (max 1.7e-18 / 8.7e-19); with it on, raw is 0/0. Turning that flag
   off in the complete stage also changes the written-order barotropic
   correction, so the resulting 28 U / 21 V final row is a two-consumer
   ablation and is not used to attribute the stage-update change. The selected
   vector path materializes NEMO's multiply, then add, then mask through the
   existing shared `rk3_stage_velocity_update`. The library default stays
   `False`; the ORCA2G ladder builder is the only setter, and only G1b takes
   its vector branch.

## Both labels, before and after the landing

The bar is the registered normalized maximum 1e-15. The kt-1 T/S land cells in
the entry-frame arm are the known NEMO-held-land convention and are excluded,
as in rounds 1–3.

| label / scope | before (`427f325de`) | after (`fe88ed406`) |
|---|---:|---:|
| GIVEN entry frames, kt 1–50, first over bar | none | none |
| GIVEN entry dynamic unequal cells | 140,599 | 2,320 |
| GIVEN restart, steps 3–100, first over bar | none | none |
| GIVEN restart dynamic unequal cells | 307,908 | 5,181 |
| INDEPENDENT first over bar, ssh / u / v | 20 / 36 / 36 | 27 / 63 / 62 |
| INDEPENDENT worst ssh / u / v | 1.46e-14 / 1.95e-15 / 2.14e-15 | 1.70e-14 / 2.09e-15 / 2.28e-15 |
| INDEPENDENT exact rows / total unequal cells | 34 / 6,724,204 | 46 / 5,508,585 |

The later first step is not used as evidence by itself. The evidence for the
landing is the raw cited statement becoming exact (107/192 → 0/0), the
fixed-selector production stages moving 8/3, 48/87 and 57/81 → 0/0, and the
strict GIVEN-row and aggregate improvements. The INDEPENDENT worst maxima move
slightly upward while its exact-row and unequal-cell counts improve; this is
registered under D96 as a last-place trajectory lottery. G1b therefore remains
OPEN, not certified.

## Controls, predictions, and provenance

- **R4-I1 — REFUTED.** Registered prediction: "the split observer's HPG + KEG
  is bit-identical to its pre-existing combined KE_PGF bucket on U and V."
  Its registered falsifier, "any unequal wet face," fired on 5 U faces.
- **R4-I2 — CONFIRMED.** Registered prediction: "a one-cell perturbation of
  either published half moves its named split row and makes the gate exit
  nonzero." From clean commit `f70edbc32`, the plant advances a nonzero HPG
  value (U/V reference maxima 9.99e-9/8.52e-9) by one representable value.
  The HPG rows each gain exactly one unequal face, the split row moves 5→6 U
  and 0→1 V, and the CLI exits 1 with `STATUS PLANT-FIRED`.
- **R4-A1 — REFUTED.** Registered prediction: "kt-2 KEG is bit-identical to
  the offline replay; HPG is the first non-bit operand and carries the
  five-face fused-bucket residual." KEG is bit-identical, but HPG is also
  bit-identical, so the registered "HPG is bit-identical" falsifier fired; the
  five faces arise only in the separately evaluated diagnostic sum.
- **R4-A2 — NOT REACHED.** Registered prediction: "the cited HPG transcription
  makes the kt-2 stage-1 RHS bit-identical on both faces (4/0 to 0/0) and loses
  no exact GIVEN row." R4-A1 found no HPG defect, so no HPG transcription was
  made. The separate routine-materialization check above is unregistered and
  is not R4-A2.
- **R4-B1 — REFUTED.** Registered prediction: "after A, the direct vector
  velocity write on the card's own RHS remains 35 U / 49 V unequal at about
  3.5e-18; source-operation materialisation makes its raw U/V outputs
  bit-identical." Its registered "counts differ materially" falsifier fired:
  after correcting the dead-arm replay, the executed raw boundary is 107/192,
  not 35/49. Separately, the **unregistered post-hoc materialization result**
  is that the corrected raw 107/192 boundary becomes 0/0.
- **R4-L1 — CONFIRMED.** Registered prediction: "after each landed statement,
  GIVEN-NEMO-ENTRY remains at the bar over entry frames 1..50 and restarts
  3..100; unequal-face counts and maxima fall at the named statement." Both
  GIVEN arms remain at the bar, and the named raw and stage rows become exact.
- **R4-L2 — CONFIRMED.** Registered prediction: "after each landed statement,
  INDEPENDENT remains not certified; first-over-bar and worst values may move
  by last-place lottery and will be registered without treating a later first
  step alone as progress." INDEPENDENT remains uncertified and its movements
  are registered above without using the later first step as evidence.
- Search-before-build found and reused the production HPG/KEG component bundle,
  `nemo_source_round`, `rk3_stage_velocity_update`, and the existing
  `rk3_stage_velocity_written_order` card selector. No parallel implementation
  or new configuration field was added.
- independent review: operator (Claude), **HOLD** on the receipt and tests;
  the model change was found correct. This revision addresses every confirmed
  finding; operator re-review remains external to this round.

Focused CPU tests, synchronous: with `JAX_ENABLE_X64=1`, the new traced
synthetic test, G1/G1b split and stage routes, and the card selector ratchet are
**7 passed**; the shared stage-arm ratchets are **22 passed**; the citation gate
is **17 passed**. With x64 disabled, the synthetic test is **1 skipped**. A
single-process attempt at the combined files reached the repository's documented
per-process JAX compiler limit and aborted; every focused group was rerun in a
fresh process for the verdicts above. No nested Codex review was run, per the
round instruction; the operator reviews the diff before landing.

Evidence SHA-256 (first 16): `rhs_terms_kt2_before.json` b9578e80c3201c2f;
`rhs_terms_split_plant.json` a6b9a70626ce4adb (clean producer `f70edbc32`);
`stages_kt2_without_written_order.json` dae05726b769a68c;
`stages_kt2_after.json` e944543b13dda41f;
`l0/stages_kt2.json` 4bf5f38d4c7b2f0f; `l0/indep100.json` c4f3f889d9626f59; `l0/given_entry50.json`
3b00630966aec414; `l0/restart_entry100.json` b20c1a2670f98060;
`l1/indep100.json` a247aa796967e0ea; `l1/given_entry50.json`
3040e3186588f3d0; `l1/restart_entry100.json` 2a587da38e667847.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r27/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — the cited vector stage update is exact; G1b remains uncertified on the INDEPENDENT label.
