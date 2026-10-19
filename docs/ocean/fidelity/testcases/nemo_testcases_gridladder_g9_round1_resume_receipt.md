# Grid ladder rung G9 — round 1 resume receipt (gridladder_r46, scoring)

Status: **LANDED, G9 NOT CERTIFIED.** Two cited NEMO fold statements land
(G9-selected only).  They remove the pivot-row salinity anomaly.  Every kt
1..10 is still DEBT under both labels, from the stage-3 FCT update on the
T-pivot row.

- Record: `gridladder_rounds/round46/oracle_g9_r46`, run by the operator at
  `83fbfd888`: `G9_PREFLIGHT_OK`, `PASS_G5_ADMISSION unequal_rows=None`,
  `G9_ACQUISITION_PASS steps=1..10 restarts=2..10 reference_completed=21`.
- Landing commit `c8783354a`.  Scores at that commit: `gridladder_rounds/round46/score/final/`.
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.

## 1. Predictions scored

| prediction | result |
|---|---|
| P1 log prints FCT order 2/2, `ln_traadv_OFF = F` | held (preflight and acquisition log checks passed) |
| P2 record admitted, passivity PASS | held |
| P3 kt-1 stage-1 `zFv` bit-identical on row 147 with the resolved mask | **falsified**: 1,319 cells differ (the card's was 0 there) |
| P4 G9 not bit-identical kt 1..10 | held |
| P5 NEMO's reference stops at kt 11 | **falsified**: it completed 21 steps |

## 2. The pivot-row salinity anomaly: NEMO does not show it

NEMO's own stage-1 tracer dump (`oracle_rktracer_operands_kt00000001_s1.bin`):
pivot-row surface salinity after stage 1 spans 26.50–33.50 PSU, and its
advection trend there is at most 7.7e-6 PSU/s.  The card's spanned
22.28–41.09 PSU.

The walk, kt 1, stage 1, in NEMO's order:

- `zFu` bit-identical everywhere.
- `zFv` differs on exactly 1,319 cells, all on row 147, max 4.0e5 m3/s.
- Of its operands (the record's stage-1 transport dump), the corrected
  velocity and the time-mean barotropic transport `vn_adv` differ on row 147.
  The card's `vn_adv` is 0 there; NEMO's is nonzero on the 68 wet surface
  faces.
- NEMO's row-147 `vn_adv` equals the fold image of its own row 146, using
  the card's fold permutation and sign: 0 cells differ.

**First statement where the card leaves NEMO:** the fold exchange of the
time-mean transports after the substep loop,
`CALL lbc_lnk( 'dynspg_ts', un_adv, 'U', -1._wp, vn_adv, 'V', -1._wp )`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:844`, SUBROUTINE `dyn_spg_ts`
at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:136`).

With that image, the pivot row's `zFv` was still 1–3 ulps off at stage 1 and
O(1) off at stage 2.  The second statement is the fold exchange of the
after ratios,
`CALL lbc_lnk( 'stp_RK3_stg', r3ua, 'U', 1._wp, r3va, 'V', 1._wp, r3fa, 'F', 1._wp )`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:169`, SUBROUTINE `stp_RK3_stg`
at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:78`).  The card's
stage-transport faces now read the existing ratio-fold helper there.

`dynspg_ts.f90` and `stprk3_stg.f90` are byte-identical in the record build
and ORCA2_OMIP_L4, so these lines hold in both.

## 3. What changed (`c8783354a`)

- New config field `rk3_stage_transport_fold_exchanges`, default False.
  When True, every stage transport reads `vn_adv` through the existing
  `_rk3_fold_exchange_top_v` and builds its V faces with the existing ratio
  fold exchange.  Both wirings sit on existing lines.
- Only the G9 card selects it, next to `rk3_stage_transport_raw_vmask`,
  with a selector row `stage transport fold exchanges`.
- Searched for an existing implementation: `vn_adv`, `Hv_avg`, `fold_exchange`,
  `_rk3_fold_exchange_top_v`, `_nemo_ratio_fold_exchange`, `fold_grid`.
  Both exchanges already existed; nothing new was written.
- Other cards: traced step jaxpr, addresses masked, unchanged — G5
  `339a8016d262c4f6`, G8 `03ccdb94113eab61` (`score/fix2/jaxpr_*.txt`).

## 4. Stage-level effect at kt 1 (card against NEMO's dumps)

| quantity | before | after |
|---|---|---|
| stage-1 `zFv`, row 147 | 1,319 cells, 4.0e5 m3/s | 0 cells |
| stage-2 `zFv`, row 147 | 1,319 cells, 4.0e5 m3/s | 0 cells (9 last-place cells elsewhere) |
| stage-1 S after the stage | 3.3 PSU | 7.1e-15 PSU |
| stage-2 S after the stage | 5.1 PSU | 7.1e-15 PSU |
| stage-3 transport, row 147 | — | exact (last-place cells elsewhere) |

The stage-1 continuity solve's vertical transport (`zFw`) differs in last
places almost everywhere (normalised 8e-16).  An offline replay of NEMO's
`div_hor` + `wzv` from NEMO's own transports and stored ratios reproduces
NEMO's `zFw` off the runoff columns (0 cells differ).  The card's departs on
271,084 of those cells.  Not walked further: under the bar, and not what
drives the DEBT.

## 5. Score (both labels, kt 1..10, every field, wet and dry cells)

Row = one field (ssh, uu_b, vv_b, u, v, T, S), level 1 or 2-D.  Levels = the
all-level rows for u, v, T, S where a restart exists.  Exact = bit-identical
or a D127 flushed subnormal.  Flushed subnormals: 0 in every row.  Dry-cell
mismatches: 0 in every all-level row.

| label | kt | exact / AT-BAR / DEBT rows | worst row (normalised) | all-level rows |
|---|---|---|---|---|
| independent | 1 | 3 / 1 / 3 | S 5.9e-6 | none (no kt-1 restart) |
| independent | 2..10 | 0 / 0 / 7 each | S or T, 2.9e-5 → 3.2e-4 | all DEBT |
| given-NEMO-entry | 1, 3 | 3 / 1 / 3 | T 5.3e-5 (kt 3) | kt 3: u AT-BAR, v/T/S DEBT |
| given-NEMO-entry | 5, 7, 9 | 3 / 0 / 4 | S ≤ 9.5e-5 | all DEBT (u, v ≤ 7.8e-15) |
| given-NEMO-entry | two-step 2..10 | 0 / 0 / 7 each | ≤ 1.1e-4 | all DEBT |

Before the statements (independent, kt 1): S 9.7 PSU, T 0.52 K.  After:
S 2.4e-4 PSU, T 9.2e-5 K.

Score SHA-256 (first 16): `indep_kt1_5.json` `7c824990628635b6`,
`indep_kt6_10.json` `6c808fa28f605e09`, `given_e135.json` `e2b81b94c1efce66`,
`given_e79.json` `7891edff5ad889b9` (`score/final/summary.txt`).

**Not certified.**  So this does not yet certify the stage-1 V transport mask
for the ORCA2 lane (standing note 2026-10-16 22:58).  What it does show: on
G9's record, the stage-1 and stage-2 3-D V transport on row 147 equals NEMO's
only with the fold-associated mask AND the two exchanges above.  The ORCA2
card has neither exchange in its stage transport; a Decision-122 import
would need both.

## 6. Where it still leaves NEMO (next walk)

After stage 3 of kt 1 the remaining error is on rows 145–147 only: 1,326
pivot-row cells (T up to 1.6e-4 K, S up to 2.4e-4 PSU), 31–33 on row 146, 1–2
on row 145.  NEMO's row 147 is self-mirrored after every stage; the card's
is self-mirrored after stages 1–2 but not after stage 3 (1,366 cells; probe
`score/probe/mirror.py`).  Its mirror mean is still wrong, so both halves of
the card's stage-3 row differ.

So the next first-differing statement is inside the stage-3 FCT on the fold
row.  The card's FCT has no fold handling at all (the shared advection
module only has a fail-fast note).  NEMO statements to walk, in order:
upstream flux `fct_up1_2stp`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:171`), the antidiffusive-flux
fold exchange
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:309-311`), and `nonosc`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:316`).  PLAUSIBLE until
measured: the record holds no FCT internals, so the next acquisition needs
the record binary to write `zwi`, `ztFv` and the limiter coefficients on rows
145–147 at kt 1 stage 3.

## 7. Tests and plants (CPU, fp64)

New value test `test_g9_stage_transport_on_the_pivot_row_is_nemos`: kt 1 of
the jitted G9 step (state traced, x64) on NEMO's surface input.  Stage-1 and
stage-2 `zFv` must equal NEMO's on row 147, every level, wet and dry.  Stage
1 must be bit-identical everywhere.  Stage 2's row 147 holds 375 zeros that
differ only in sign; the test compares values there.

Plants, each a scratch worktree at `c8783354a` with one wiring line removed
(`score/plants/*.diff`, `*.log`):

| copy | result |
|---|---|
| restored | 1 passed |
| `vn_adv` exchange off | 1 failed: stage 1, 1,319 pivot cells |
| `r3v` exchange off | 1 failed: stage 1, 692 pivot cells |

Gates at `c8783354a` (`JAX_ENABLE_X64=1`, `land.sh` PYTHONPATH):

| battery | result |
|---|---|
| default gate: citation gate, private-import ratchet, TKE, recipe, freshwater, barotropic state | 140 passed |
| default gate: face-mask rank (`-n 2`) | 2 passed |
| G9 file | 14 passed |
| G2 ladder file (config-tail positions updated) | 11 passed |

This receipt's citation gate: section 8.

## 8. Receipt citation gate

`nemo_testcase_receipt_citation_gate.py --receipt <this file> --from-heading
"# Grid ladder rung G9 — round 1 resume receipt (gridladder_r46, scoring)"`.
Result in `gridladder_rounds/round46/score/final/receipt_citation_gate.json`.

- At `8337e2191` (the receipt commit): `PASS`, 7 citations, 0 failures, 0 unmapped,
  0 map entries failing the audit.
- Its shift plant on `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:844` exits 1, `FAIL`.
- Citation-gate test + private-import ratchet after the map edit: 23 passed
  (`final/final_gates.log`).
- Evidence manifest `gridladder_rounds/round46/score/manifest.sha256`
  (`42c090a0f5d0e33b`, 43 files; pickles and arrays excluded).

## Choices

ASKED: G9's definition and record (the brief); the bar and labels (D127).

UNASKED, offered for revert:
- the new field defaults False and only G9 selects it (no-silent-default rule);
  importing it to ORCA2 is a Decision 122 question;
- the two exchanges share one field (one selector, two wiring lines, each
  planted separately);
- the stage-2 pivot comparison is value equality (signed zeros ignored).

UNVERIFIED: the stage-3 FCT fold statements (no internals in the record);
the stage-1 `wzv` last-place departure (unwalked, under the bar).
