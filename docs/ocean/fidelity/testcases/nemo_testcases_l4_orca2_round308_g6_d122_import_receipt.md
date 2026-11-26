# ORCA2 round 308 — rung G6's certified statements imported under Decision 122

Date: 2026-10-17 (Claude round; codex review: paused (quota)).

Status: **LANDED-pending** (operator review + push gate).

## Headline

- All seven ORCA2 card/label pairs now select eight of G6's statements: the
  vector Krhs order HPG -> VOR -> KEG with the KEG product rounded, vor_een's
  written order, np_CRV's top-F U fold halo, the wzv stored r3 ratios,
  dyn_zad's written association, the stage-2 clock pair, the stored Kmm
  ratios, and Decision 132's carried after-SSH slot.
- Each passes D122 (c)+(d) alone and all eight pass together, on all seven
  pairs: 0 exact rows lost, first over-bar row unchanged.
- G6's fold statements stay off.  Alone, the fold unit and the raw pivot V
  mask refuse; HPG fold geometry moves no row; top-V passes the rule but
  grows the kt-10 sea-level maximum 0.39 -> 7.4 m (Decision 125 holds it).
- The fold band is unchanged: OMT-4 fold U stays at 85-166% relative error
  kt 1-10 (the selection moves the 4th digit).  The private fold unit cannot
  become public yet: its partner, the stage-1 3-D V transport's associated V
  mask, is certified by no rung.

## Provenance

- Base `8007f03c5`.  Import `a4a1bd214`; preregistration + gate `ef4b40001`
  (frozen before measuring); KMM/SEL arms `6721a2635`; selection `16124b704`.
- Import = package hunks of gridladder commits `b1a4fe915c fe88ed4067
  5812f22a63 840e6efeb2 ded95e4698` (G1b vector Krhs order + KEG rounding)
  and `9f8316faba dc38c2e8cb e31b570a09 24690e4329 da61d1c107 55b7f55e13
  bac05518aa 23dfc33b06 4054231a7a` (G6).  Applied with `git apply --3way`;
  conflicts were resolved by keeping this branch's cards and taking
  gridladder's code.  Every selector's lines match the gridladder tip
  (`9616955124`); `rk3_vector_rhs_nemo_order` differs only in line
  wrapping.  Not imported: G6's card selection, the G1b observer payload,
  and G1b's subnormal-KEG workaround `3f00fe185b` (an XLA flush workaround;
  Decision 127 already scores flushed subnormals as matching).
- Certification: gridladder receipts `nemo_testcases_gridladder_g6_round1..8`
  (addendum 104).
- Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round308_g6_d122_gate.py`
  (round-285 ladder, round-276 classifier, round-278 decision).  Records as
  round 307.  CPU, fp64, libm, production JIT.  Evidence
  `orca2_rounds/round308/` (`arms/`, `arms2/`, `final/`).

## Map: G6 statement -> ORCA2 setting -> compiled line (ORCA2_OMIP_L4 ppsrc)

| arm | setting(s) | NEMO line run by ORCA2 |
|---|---|---|
| VECORD | `rk3_vector_rhs_nemo_order` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:141`, `:147`, `:164` |
| EENORD | `een_vorticity_written_order` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762-766` |
| TOPF | `een_nemo_top_f_u_halo` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762-766` |
| WZVR | `nemo_wzv_stored_r3_ratios` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sshwzv.f90:297-298` |
| ZADA | `nemo_zad_written_association` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzad.f90:112-126` |
| CLOCK | `nemo_stage2_momentum_wzv_half_step` (+WZVR) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:188-189`, `:221`, `:327` |
| KMM | `nemo_stage_momentum_wzv_stored_kmm_ratios` (+WZVR) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:327` |
| CARRY | `nemo_first_wzv_after_ssh='rk3_extrapolated_carried'` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:237`; read at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:152`, `:156` |
| FOLDUNIT | `barotropic_literal_fold_unit_vector_form` | (round 303 ladder map) |
| TOPV | `rk3_fold_exchange_top_v_row` | (round 303) |
| RAWV | `rk3_vector_raw_vmask` (also the vorticity fold halo, ZAD pivot face and stage-ratio fold exchange: one switch in the imported code) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:283` is the transport line it does NOT cover |
| HPGF | `nemo_hpg_sco_fold_geometry` | — |

VECORD as G6 runs it also turns off `rk3_stage1_rhs_nemo_literal`.  On
vector cards that flag's only consumer is already forced on by the vector
stage update (read off the code), so the landed set leaves it as it was.  SEL
measures exactly the landed set.

GAP (addendum 104): the stage-1 3-D V transport's associated V mask is not
certified by any rung; it stays off, and with it every fold statement.

## D122 table (control = the shipped card at the same tree)

Census = rows toward / away / equal (registered, never a veto).  First
over-bar row, every arm and pair: kt-1 stage-1 T (rung 10 independent: kt-1
entry ssh), direction `equal`.

| arm | rung 0 | OMT-1 | OMT-2 | OMT-3 | OMT-4 | rung 10 indep | rung 10 given | lost | D122 |
|---|---|---|---|---|---|---|---|---:|---|
| VECORD | 51/110/24 | 26/65/47 | 66/85/29 | 73/65/40 | 60/98/27 | 66/112/3 | 93/81/9 | 0 | PASS |
| EENORD | 76/82/27 | 40/49/54 | 54/86/46 | 72/69/36 | 90/79/16 | 76/95/7 | 99/69/14 | 0 | PASS |
| TOPF | 72/113/0 | 27/114/0 | 39/142/0 | 40/141/0 | 68/117/0 | 68/117/0 | 68/117/0 | 0 | PASS |
| WZVR | 68/69/26 | 21/15/45 | 28/44/54 | 37/31/68 | 82/54/29 | 58/104/15 | 90/75/23 | 0 | PASS |
| ZADA | 68/76/48 | 30/32/82 | 49/56/68 | 70/56/59 | 82/56/56 | 91/67/19 | 83/69/26 | 0 | PASS |
| CLOCK | 72/65/26 | 14/17/58 | 42/23/70 | 34/41/63 | 85/51/29 | 69/96/12 | 90/73/24 | 0 | PASS |
| KMM | 70/69/24 | 21/15/45 | 28/44/54 | 38/29/66 | 84/52/29 | 58/104/15 | 90/75/23 | 0 | PASS |
| CARRY | 111/64/0 | 110/25/0 | 140/35/0 | 132/43/0 | 133/42/0 | 95/80/0 | 112/63/0 | 0 | PASS |
| **SEL (landed)** | 115/70/10 | 112/29/6 | 141/40/6 | 133/48/8 | 136/49/10 | 98/89/2 | 116/72/2 | 0 | **PASS** |
| HPGF | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0 | PASS (inert) |
| TOPV | 23/165/0 | 100/42/0 | 111/71/0 | 119/63/0 | 23/165/0 | 24/160/4 | 24/160/4 | 0 | PASS, held (D125) |
| FOLDUNIT | refuses in kt 8 | runs | runs | runs | refuses | refuses | refuses | — | FAIL |
| RAWV | refuses in kt 4 | non-finite u kt 8 | refuses | refuses | refuses | refuses | refuses | — | FAIL |
| FOLD | refuses | runs | runs | runs | refuses | refuses | refuses | — | FAIL |
| ALL | refuses | runs | runs | runs | refuses | refuses | refuses | — | FAIL |

Refusals are the live-W-thickness guard (`raw-mesh e3w_int must contain only
finite values > 0`), as rounds 282/303 found for the fold unit.

Size, kt-10 stage-3 sea-level maximum (m), control -> SEL: rung 0 0.394296
-> 0.394008; OMT-1 (kt 8) 0.747152 -> 0.74699; OMT-2 0.348877 -> 0.348812;
OMT-3 0.363401 -> 0.363164; OMT-4 0.394127 -> 0.393839; rung 10 independent
0.277223 -> 0.276969; given entry 0.261704 -> 0.26145.  TOPV: rung 0 and
OMT-4 0.394 -> 7.41, rung 10 0.277 -> 3.27.

Instrument checks:
- control ladders at `ef4b40001` and `6721a2635` are identical on all seven
  pairs;
- the shipped card at `16124b704` reproduces the SEL arm row for row on all
  seven pairs (1,360 rows; `final/instrument_check.txt`), so SEL's verdict is
  the shipped card's (the classifier refuses cross-tree pairs, so it is not
  re-scored).
- CARRY moves no kt-1 row on any pair (first moved kt = 2 everywhere).

Not measured this way: OMT-0 inherits the selection from the shared ORCA2
builder but is outside D122's seven pairs.  The given-entry rung-10 ladder
bridges NEMO's entry each step; whether the carried slot is reseeded from
NEMO's `ssha` there was not traced (its numbers are the label's own).

## Fold band now (round-302 history probe, shipped OMT-4 card, kt 1-10)

Relative RMS on NEMO's wet support, step entry, base `8007f03c5` -> tip
`16124b704`:

| kt | fold U | interior U | fold V | fold ssh | interior ssh |
|---:|---|---|---|---|---|
| 2 | 104.8 -> 104.8 | 4.821 -> 4.821 | 86.39 -> 86.39 | 195.4 -> 195.4 | 5.513 -> 5.513 |
| 5 | 165.8 -> 165.8 | 6.237 -> 6.236 | 104.9 -> 104.9 | 119.4 -> 119.4 | 5.432 -> 5.432 |
| 8 | 115.4 -> 115.3 | 4.714 -> 4.696 | 92.9 -> 92.9 | 88.64 -> 88.65 | 4.295 -> 4.296 |
| 10 | 116.3 -> 116.3 | 6.122 -> 6.129 | 101.6 -> 101.6 | 66.3 -> 66.31 | 5.284 -> 5.284 |

First over-bar: kt-1 stage 1 in every field and region, both trees.  The
fold debt is owned by the fold statements left off.  The private fold unit
cannot become public: alone it refuses (kt 8), with top-V + raw V mask +
HPG geometry it still refuses on rung 0, OMT-4 and rung 10; round 303 showed
it completes only with the uncertified transport mask.

## GYRE, DINO, VORTEX (import tree `ef4b40001` vs base)

- GYRE ten-step residual archive identical (210 arrays, sha256
  `3b3d865cf5c82d38` both, same as round 307); day-30: 30/30 snapshots
  byte-identical (manifest excluded).  The selection commit edits only the
  ORCA2 card builder, which GYRE does not call.
- Step jaxprs identical base vs import: DINO (123,236 lines), VORTEX zco
  flux/vector, VORTEX seamount flux/vector.

## Predictions

| ID | Verdict | Result |
|---|---|---|
| R308-P1 | **CONFIRMED** | rounding-form arms: 0 exact rows lost, first over-bar equal, all seven pairs (KMM after adding the WZVR flag it requires; first KMM run refused on that dependency). |
| R308-P2 | **CONFIRMED** | FOLDUNIT refuses on rung 0, OMT-4, rung 10 (both labels). |
| R308-P3 | **REFUTED** | TOPV passes (c)+(d) on all seven pairs; its growth (0.39 -> 7.4 m) shows only in size.  Stays off by Decision 125. |
| R308-P4 | **CONFIRMED** | CARRY: no kt-1 row moves on any pair. |
| R308-P5 | **CONFIRMED** | FOLD refuses on rung 0, OMT-4, rung 10. |
| R308-P6 | **CONFIRMED** | GYRE, DINO, VORTEX identical. |

## Gates and tests

- New `tests/ocean/fidelity/test_orca2_round308_g6_selection.py` (14 tests):
  reverting the selection fails 6 (tree restored, `git status` clean).
- Receipt citation gate on this receipt: PASS, 13 citations, failures 0.
- Default gate (land.sh battery, JAX_ENABLE_X64=1): 159 passed (citation
  gate + private-import ratchet + selection 37; recipe 25; TKE/freshwater/
  face-mask/barotropic 95; ORCA2 parallel receipt 2).  The import's line
  shifts were re-anchored in the citation map and the default receipt.
- codex review: paused (quota).  The operator runs an independent Claude
  review before landing.

## Configuration choices

- ASKED (D122/D132, taken): the eight selections above; the carried
  after-SSH slot adds one carried field to the ORCA2 state (D132).
- UNASKED: none.  `rk3_stage1_rhs_nemo_literal` left as it was (inert on
  vector cards); no runoff (D126) or TKE switch changed.

## OPEN

1. The stage-1 3-D V transport's associated V mask needs a ladder rung; until
   then the fold unit, raw V mask, HPG fold geometry and top-V stay off.
2. V3 import (addendum 105); D135 (TKE penetration/mixing length) pending.
