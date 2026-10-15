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

1. **HPG/KEG split — no production defect landed.** At kt 2, HPG is 0/15,853
   unequal U faces and KEG is 0/15,853. The first non-bit operation is the
   diagnostic `HPG + KEG` addition: 5 U faces, 8.27e-25; after the production
   HPG → VOR → KEG accumulator, the stage-1 RHS remains 4 U / 0 V faces,
   maximum 1.65e-24. NEMO's executed calls are `R4REF/stp2d.f90:138`, `:144`,
   `:161`; the in-place KEG subtraction is `R4REF/dynkeg.f90:128-130`. The
   same full-ORCA2 statements are `ORCA2/stp2d.f90:141`, `:147`, `:164` and
   `ORCA2/dynkeg.f90:128-130`. Replacing the selected routine barriers with
   source-round materializations left all counts and maxima unchanged, so that
   candidate was removed. The RHS residual is still OPEN.
2. **Stage-1 replay retraction.** The round-3 instrument always replayed the
   thickness-weighted `ELSE` statement. G1b actually executes the preceding
   `ln_dynadv_vec` velocity branch. Its one-level barotropic correction nearly
   canceled the wrong replay and manufactured the reported 35 U / 49 V row at
   3.5e-18. The tool now selects the card's compiled branch and cannot print
   that withdrawn claim. Citation: `R4REF/stprk3_stg.f90:366-370`; the same
   line in the Decision-122 full build is `ORCA2/stprk3_stg.f90:467-471`.
3. **Vector stage update (fe88ed406).** With the corrected replay and the
   card's existing `rk3_stage_velocity_written_order` switched off, the raw
   multiply/add/mask differs on 107 U / 192 V faces (max 1.7e-18 / 8.7e-19),
   leaving 28 U / 21 V final faces at 1.7e-18 after the barotropic correction.
   The selected path materializes NEMO's multiply, then add, then mask through
   the existing shared `rk3_stage_velocity_update`; raw, corrected, and all
   three kt-2 stage fields become 0 unequal. The deselected arm is the plant.
   The library default stays `False`; the ORCA2 card builder already selected
   the option explicitly, and only G1b takes its vector branch.

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
landing is the cited statement becoming exact (107/192 raw and 28/21 final →
0/0), plus the strict GIVEN-row and aggregate improvements. The INDEPENDENT
worst maxima move slightly upward while its exact-row and unequal-cell counts
improve; this is registered under D96 as a last-place trajectory lottery. G1b
therefore remains OPEN, not certified.

## Controls, predictions, and provenance

- The split plant advances one wet HPG cell by one representable value. Both U
  and V HPG rows report exactly one unequal face and the derived sum row fires;
  the CLI exits 1 with `STATUS PLANT-FIRED`.
- R4-A1 (split closure exact) and R4-A2 (HPG owns the five faces) are
  **REFUTED**: both halves are exact; their separately evaluated diagnostic
  sum is the first non-bit operation. R4-A3 (routine materialization closes
  the four RHS faces) is **REFUTED**: 4/0 and 1.65e-24 are unchanged.
- R4-B1 is **CONFIRMED after correcting the dead-arm replay**: the original
  35/49 row was not the executed statement; the true raw 107/192 and final
  28/21 statement rows close to 0/0 with source-order materialization.
- R4-L1 (both GIVEN arms remain at bar), R4-L2 (INDEPENDENT remains
  uncertified), and R4-C1 (not certified this round) all **HOLD**.
- Search-before-build found and reused the production HPG/KEG component bundle,
  `nemo_source_round`, `rk3_stage_velocity_update`, and the existing
  `rk3_stage_velocity_written_order` card selector. No parallel implementation
  or new configuration field was added.
- independent review: operator (Claude)

Focused fp64 CPU tests, synchronous: RHS split + plant and G1b stage route +
deselection plant, **5 passed**; G1b card and the shared stage-arm suite,
**36 passed**; citation gate after the repository re-anchor pass, **17 passed**.
The re-anchor moved 60 citation-map lines and 16 audited-receipt lines after
the helper insertion; its two exact-string mutation controls were updated and
both replanted defects fire. No nested Codex review was run, per the round
instruction; the operator reviews the diff before landing.

Evidence SHA-256 (first 16): `rhs_terms_kt2_before.json` b9578e80c3201c2f;
`rhs_terms_split_plant.json` 1dccb46b0657109d;
`stages_kt2_without_written_order.json` dae05726b769a68c;
`stages_kt2_after.json` e944543b13dda41f;
`l0/indep100.json` c4f3f889d9626f59; `l0/given_entry50.json`
3b00630966aec414; `l0/restart_entry100.json` b20c1a2670f98060;
`l1/indep100.json` a247aa796967e0ea; `l1/given_entry50.json`
3040e3186588f3d0; `l1/restart_entry100.json` 2a587da38e667847.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r27-1059729479
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — the cited vector stage update is exact; G1b remains uncertified on the INDEPENDENT label.
