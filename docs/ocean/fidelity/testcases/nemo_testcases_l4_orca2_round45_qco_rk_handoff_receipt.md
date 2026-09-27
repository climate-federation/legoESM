# ORCA2 round 45 — stage-1 QCO/RK operand handoff

Date: 2026-09-27  
Card: `orca2_vector_een_c2`  
Claim label: **given NEMO's entry** (Decision 52)

## Verdict

**HELD.**  The first non-bit input to the stage-1 tracer QCO/RK assignment is
the Kaa stretch: 2 / 8,613 support-safe wet columns differ by at most
`1.1102230246251565e-16`.  NEMO interpolates the already-formed endpoint r3t
ratios; legoESM interpolates SSH first and then forms the ratio.  Replaying
NEMO's r3t interpolation closes all 8,613 columns exactly.

The following QCO/RK assignment has a separate arithmetic-association debt.
With recorded operands, the fused JAX expression differs on 57,141 temperature
and 57,169 salinity cells, while the source-ordered replay is exact on all
228,641 scored cells for both tracers.  Therefore no single statement closes
the full stage-1 row, and no model change lands in this round.

No configuration, selector, carried-state rule, sea-ice field, score domain,
scientific threshold, or stabiliser changed.  The card's six sea-ice
`unmeasured_features` entries remain frozen.

## Preregistered predictions

| prediction | outcome |
|---|---|
| R45-P1 round-44 boundary reproduces | CONFIRMED: after-SBC T/S remain exact; production stage-1 remains 57,160 T and 57,180 S unequal with maxima `7.105427357601002e-15` and `1.4210854715202004e-14` |
| R45-P2 record, execution policy, scorer, and plant bind | CONFIRMED |
| R45-P3 literal compiled assignment closes Kaa | CONFIRMED: 0 / 228,641 unequal for T and S |
| R45-P4 operands remain exact until an arithmetic boundary | **REFUTED**: Kbb and Kmm stretches are exact, but Kaa stretch is already 2 / 8,613 unequal |
| R45-P5 one QCO/RK statement repair closes production stage 1 and lands | **REFUTED**: the upstream Kaa operand and downstream assignment are two distinct compiled statements; no model edit was attempted |

The failed predictions remain frozen in the preregistration.

## Ordered statement walk

All rows are **given NEMO's entry**, CPU, fp64, scalar libm, and use the
existing record-backed support-safe rank-0 wet interior.

| boundary or discriminator | T / shared unequal | S unequal | maximum absolute difference |
|---|---:|---:|---:|
| production after-SBC Krhs | 0 / 228,641 | 0 / 228,641 | 0 |
| current Kbb stretch | 0 / 8,613 | — | 0 |
| current Kmm stretch | 0 / 8,613 | — | 0 |
| current Kaa stretch | 2 / 8,613 | — | `1.1102230246251565e-16` |
| source-ordered Kaa r3 interpolation | 0 / 8,613 | — | 0 |
| recorded-operand NumPy literal QCO/RK | 0 / 228,641 | 0 / 228,641 | 0 |
| recorded-operand fused JAX QCO/RK | 57,141 / 228,641 | 57,169 / 228,641 | `3.552713678800501e-15` / `1.4210854715202004e-14` |
| recorded-operand source-ordered JAX QCO/RK | 0 / 228,641 | 0 / 228,641 | 0 |
| current-Kaa-only literal replay | 16 / 228,641 | 13 / 228,641 | `3.552713678800501e-15` / `7.105427357601002e-15` |
| production stage 1 | 57,160 / 228,641 | 57,180 / 228,641 | `7.105427357601002e-15` / `1.4210854715202004e-14` |

The recorded Kbb endpoint ratio is independently reconstructed exactly from
the card's entry SSH and reference-depth reciprocal.  This validates the
endpoint-ratio instrument before the interpolation result is used.

## First non-bit statement

The compiled HYB stage-1 branch first computes the endpoint r3 arrays and then
sets Kaa to `r2_3 * r3t(Kbb) + r1_3 * r3ta` at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:160-179`.
That source-ordered replay is exact.  The current model instead forms
`eta_before + (eta_after - eta_before) / 3` and calls the generic ratio
constructor; algebraic equivalence is not bit identity.

After an exact Kaa operand, NEMO's next assignment multiplies Kbb and Krhs by
their respective stretches, adds those rounded products, and divides by the
Kaa stretch at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:670-681`.
The recorded-operand replay proves that assignment has an additional fused
arithmetic debt.  It is downstream and is not combined with the first
statement in this held round.

## Controls, tests, and review

- The one-ULP tracer plant exits nonzero with `REFUSE: planted QCO/RK tracer
  cell rejected through scorer`.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).
- No `packages/` file changed, so neither the ORCA2 landing ladder nor the GYRE
  trajectory non-regression gate is invoked as a landing claim.

## Evidence

Durable artifacts are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round45/`.
The decisive files are `qco_rk_final.json`, `qco_rk_final.log`,
`qco_rk_plant.log`, and `codex_review.log`.

## OPEN

1. Preregister and land the first statement only: construct stage-1 Kaa r3t
   by source-ordered interpolation of the endpoint r3t ratios, then run the
   full ORCA2 ladder and required GYRE gates.  Do not fold the downstream
   QCO/RK arithmetic repair into that change.
2. After the Kaa operand lands, preregister the separately proven QCO/RK
   source-order statement and require production stage-1 T/S to become exact.
3. The first whole-card non-bit checkpoint remains kt=1 stage-1 temperature at
   `0.0014770192519700243 K`; round-20 slow forcing and the Decision-52
   **independent** initial state/year remain open.
4. Sea ice remains out of scope and exactly the frozen six-item
   `unmeasured_features` tuple.
