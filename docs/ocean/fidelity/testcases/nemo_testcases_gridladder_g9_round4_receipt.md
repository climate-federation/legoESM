# Grid ladder rung G9 — round 4 receipt (gridladder_r49)

Status: **LANDED, G9 NOT CERTIFIED.**  One NEMO statement, G9-selected only:
stage 3 builds NEMO's tracer Krhs and trazdf's content.  kt-1 step-end T
60,401 → 1,076 cells off, S 1,414 → 587.  The first differing stage-3
statement is earlier, inside the FCT's upstream pass, and is not landed.

- Record: round 46's (`gridladder_rounds/round46/oracle_g9_r46/twin_a`),
  unchanged.  No acquisition.
- Item-1 test `2d3e9e7d4`; preregistration `1b95667f4` (moved `77322a1f0`),
  before any item-2 measurement; landing `02573a4c4`; test `5d99d5179`;
  re-anchor and test follow-ups `4d675a4c6`, `cd6f3ed73`.
- codex review: operator.
- Evidence: `gridladder_rounds/round49/` (manifest `manifest.sha256`, first 16
  `342213cbf602129b`; pickles and arrays excluded).
- NEMO lines are cited in ORCA2_OMIP_L4.  `stprk3_stg.f90`, `trazdf.f90`,
  `traadv_fct.f90`, `trasbc.f90` are byte-identical to the record build
  ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1 (`cmp`).

## 1. Round-48 review item: the stage-2 wzv assertion is value-based

The "≤ 100 cells" bound is gone.  kt 1, stage 2: the 73 zFw cells off NEMO's
record all sit in the 10 columns next to the stage-2 zFu (28 cells) and zFv
(9 cells) that already differ upstream of wzv; every other column is exact.

New assertion (jit, state and forcing traced, x64), stages 1 and 2: every
zFu, zFv, zFw cell at the bar against the record; zFw equal to NEMO's on every
column whose own and neighbouring zFu/zFv equal NEMO's; stage 1 also has
zFu, zFv exact.

| copy (`round49/plants/`) | result |
|---|---|
| restored | 1 passed |
| stage-2 clock pair removed | failed: 165,653 clean-column cells |
| stage-1 clock pair removed | failed: 355,104 cells |

## 2. Preregistration outcome

| prediction | outcome |
|---|---|
| P1 content `zrhs` differs | CONFIRMED: 68,045 T / 1,738 S cells (≤1.4e-16) |
| P2 NEMO's `zrhs` substituted → step-end exact | CONFIRMED: 0 / 0 |
| P3 card FCT rate = NEMO's after-advection Krhs | **REFUTED**: 404,775 T / 415,895 S cells |
| P4 card algebra, not XLA | CONFIRMED (section 3) |

So the content is not the first statement; the FCT is.

## 3. The walk (kt 1, stage 3, offline replays)

NEMO's stage 3 zeroes Krhs
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:604`), adds advection
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:637`), the surface terms
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:645`) and qsr
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:717`), then tra_zdf
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:749`) forms
`zrhs = e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:288`) and solves.  The
preregistration cited the tra_zdf call as :753; it is :749.

1. Solve.  NEMO's literal `zrhs` (numpy, from the record's Kbb, after-qsr
   Krhs, r3t, e3t_0) substituted as the card's pre-implicit content: step-end
   T, S exact on every wet cell.  The implicit solve is exact.
2. Content.  The card built `e3t(Kbb)*T − dt*(flux divergence) +
   (dt*e3t(Kmm))*sources`, not NEMO's form: 68,045 T / 1,738 S cells off.
3. Rate.  The shared FCT's NEMO-order trace (upstream write, then limited
   write), replayed jitted on the card's own stage-3 FCT inputs, misses
   NEMO's after-advection Krhs on 404,775 T / 415,895 S cells (median 9 / 15
   last places).  Instrument: a literal numpy replay of NEMO's FCT on NEMO's
   operands (the card's limiter coefficients, north-fold rows excluded)
   matches NEMO's rate on all but 2,848 T / 4,511 S cells.
4. First statement.  The card's upstream predictor `zta_up1`
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:538`) already differs from
   the literal one on 111,748 T / 83,834 S cells; its rate after the upstream
   write (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:605`) on 400,147 /
   411,486.  NEMO forms the upstream fluxes from the transports
   (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:502`) and divides by
   `e1e2t`, then `e3t`; the card's generic FCT forms them from
   transport/edge-length and a generic divergence.  An algebraic form, not
   XLA: NEMO's build has no `-march`/FMA flags, and the card's statements
   evaluated jitted on identical operands match numpy.
5. Downstream statement landed (section 4): with the card's own rate and
   NEMO's association, an offline estimate gave 1,632 T / 1,170 S step-end
   cells; the landed code gives 1,076 / 587.

New field `rk3_stage3_tracer_krhs_nemo_order` (default False, G9 only).  The
FCT rate is the shared FCT's existing NEMO-order trace, returned by a static
flag through the two advection helpers; the stage-3 terms are the card's
existing written heat, salt, runoff and qsr rates kept apart and added in
NEMO's order (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:307`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:308`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:323`).  Searched first:
`stage3_content_written_order`, `_source_order_sum`, `return_nemo_trace`,
`_nemo_cen2_tracer_rhs`, `nemo_fct` — the trace and the ordered-sum helper
existed; no literal stage-3 FCT and no stage-3 Krhs path existed.

| kt-1 quantity (cells off NEMO) | before | now |
|---|---|---|
| stage-3 pre-implicit content T / S | 68,045 / 1,738 | 1,226 / 762 |
| step-end T / S | 60,401 / 1,414 | 1,076 / 587 |

Residual: 58 T content cells (jitted) differ by a last place even where the
card's FCT rate and FCT e3t(Kmm) equal NEMO's — one card operand of the
content (PLAUSIBLE: its stored e3t(Kmm) or e3t(Kbb)) is not NEMO's there.
Not walked.

## 4. Score at `02573a4c4` (both labels, kt 1..10)

Rows ssh, uu_b, vv_b, u, v, T, S (level 1 or 2-D), normalised max.
Flushed subnormals 0; dry-cell mismatches 0.

| label | kt | exact / AT-BAR / DEBT | worst | round 48 worst |
|---|---|---|---|---|
| independent | 1 | 5 / 2 / 0 | T 2.4e-16 | T 2.4e-16 |
| independent | 2..5 | 0 / 0 / 7 | 2.7e-14 → 5.5e-14 | 3.1e-14 → 8.2e-14 |
| independent | 6..10 | 0 / 0 / 7 | v 7.6e-14 → 2.5e-13 | v 7.6e-14 → 2.0e-13 |
| given-NEMO-entry | 1, 3, 5, 7, 9 | 5 / 2 / 0 | ≤4.7e-16 | ≤3.5e-16 |
| given-NEMO-entry | two-step 2..10 | 0 / 0 / 7 | ≤3.1e-14 | ≤3.1e-14 |

Against round 48, 212 rows: statuses identical (AT-BAR 58, DEBT 154);
38 identical, 96 lower, 16 count-only, 62 higher; cell counts lower on
almost every row.  All 62 higher rows are DEBT both sides except one AT-BAR
row (given kt-3 T 3.5e-16 → 4.7e-16, cells 3,717 → 253).  Largest: independent
kt-10 v 1.62e-13 → 2.45e-13.  First over-bar step unchanged (kt 2).

Score SHA-256 (first 16): `indep_kt1_5.json` `5e7e9cc8949d43e9`,
`indep_kt6_10.json` `5315bd20caf48522`, `given_e135.json` `198534886394070f`,
`given_e79.json` `8a87ecaf21c4dcf5`.  Row diff `score/cmp.txt`.

**Not certified.**  So this still does not certify, for the ORCA2 lane's
fold import, the stage-1 V transport mask, the round-46 exchanges, the
round-47 FCT fold statements, or round 48's end-of-stage fold and stage pair.
What stands: every single step from NEMO's state reproduces NEMO's u and v
exactly at every level and T, S within the bar (fold rows included), and the
kt-1 tracer debt is now owned by the stage-3 FCT's form.

Other cards: traced step jaxpr, addresses masked, identical before and after —
G5 `339a8016d262c4f6`, G8 `03ccdb94113eab61` (`round49/jaxpr/`).

## 5. Tests and plants (CPU, fp64, jit, traced inputs)

- `test_g9_stage_transport_wzv_reads_nemos_stage_clock_pair`: section 1.
- `test_g9_stage3_tracer_content_is_nemos_trazdf_zrhs`: kt-1 content at the
  bar against NEMO's literal `zrhs` and equal to it on all but ≤1,300 T /
  ≤800 S cells (measured 1,226 / 762).  This is a measured count, because
  the residual's owner is not walked.  Plant (stage-3 terms not passed): failed, 68,045 T.

Two existing tests follow the new field: the G9-vs-G5 config diff lists it;
the pivot-row FCT test reads its FCT inputs from a flag-off step (the inputs
sit upstream of the new path, whose side output refuses them).

## 6. Gates (`JAX_ENABLE_X64=1`, `land.sh` PYTHONPATH; `round49/final/`)

| battery | result |
|---|---|
| G9 file | 19 passed |
| G2 ladder file | 11 passed |
| recipe, freshwater, barotropic state | 49 passed |
| citation-gate test, private-import ratchet, TKE | 91 passed |
| face-mask rank (`-n 2`) | 2 passed |

Before the citation-gate test passed: 86 older citation lines re-anchored with
the existing tool (base `2ae9b6774`); three map extents grew by the lines
inserted between unchanged endpoints.

## Choices

ASKED: the items (brief); G9's record; the bar and labels (D127).

UNASKED, offered for revert:
- landed a downstream cited statement (the content) while the first
  differing statement (the FCT upstream pass) stays open; statuses unchanged,
  62 DEBT maxima higher (DECISION_NEEDED line).
- new field default False, G9 only (no-silent-default rule).
- the new content test uses a measured count (section 5).
- the item-1 support dilates the dirty columns by one cell, cyclic in both
  directions.

UNVERIFIED: the literal FCT instrument is validated on 99.3% of cells only
(card limiter coefficients); north-fold rows excluded from the walk; the 58
residual content cells' operand.

## 7. Receipt citation gate

`--from-heading` this title, at `089368d9d`: `PASS`, 12 citations,
0 failures, 0 unmapped, 0 map entries failing the audit.  Its shift plant on
the trazdf.f90:288 citation exits 1.  Nine map entries added (one pinned to
its second occurrence).  Citation-gate test, private-import ratchet, TKE
re-run after them: 91 passed (`round49/final/`).
