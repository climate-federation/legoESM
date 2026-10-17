# ORCA2 round 303 — hooked run after the EVD fix, and the fold-band stage-1 pair

Date: 2026-10-16 (Claude round; codex review: paused (quota)).

Status: **HELD** (no model or card change; citation-map entries and one
committed arm driver).

Correction (2026-10-16, after the independent Claude review, HOLD): the
exchange first cited here (`:584` / `:570`) sits inside `IF( ln_shuman )` and
does not run (ocean.output:220 `ln_shuman = F`).  The exchange that runs is
after the tracer update; the citations, order, history and fold-unit
statements below are corrected in place.

## Headline

- The fold-band blow-up **persists** without EVD momentum mixing: the hooked
  run (round 249's private fold unit + round 245's barotropic surface-mask
  arm) reaches fold U 1550% at kt-7 entry and refuses during kt 8 on
  non-positive live W thickness (`raw-mesh e3w_int must contain only finite
  values > 0`).  The interior improved (kt-2 entry U 1.72% -> 2.7e-14%).
- Its earliest over-the-bar row is kt-1 stage 1, fold band, and it lives on
  ONE row: the last T row (row 147 = the T-pivot row), 68 wet columns,
  levels 0-28, 1,319 cells each for V, T and S.  Every other kt-1 stage-1
  T/S difference is <= 3.6e-15 (rounding).
- Two NEMO statements own it, and they only work together:
  1. the stage-1 3-D V transport multiplies by NEMO's laterally associated
     V mask (`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:291`,
     `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:283`, SUBROUTINE
     `stp_RK3_stg`; mask built at
     `ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dommsk.f90:211` and associated at
     `ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dommsk.f90:232`, likewise
     `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:211` and
     `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:232`, SUBROUTINE `dom_msk`).
     The card's compact mask keeps the pivot-row V faces dry, so the
     tracer update on row 147 misses its north-face transport;
  2. the end-of-stage sign-minus V exchange
     (`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:826-827`,
     `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:795-796`, SUBROUTINE
     `stp_RK3_stg`).  It is the `ELSE` (non-Shuman) branch's inner `ELSE`:
     `ln_shuman = F` (ocean.output:220) and `ln_zdfcst = T`
     (ocean.output:557) makes `l_zdfsh2 = .FALSE.` in `zdf_phy_init`, so
     neither `:822` / `:791` (the `avm_k` form) nor the Shuman call runs.
     It runs AFTER the tracer update, so it sets the next stage's V, not
     this stage's tracer transport.  It writes row 147's V as the fold
     image of row 146; the card leaves 0.  Measured on NEMO's own frames:
     the top row equals sign x permuted row 146 in value in all 40 OMT-4
     frames (kt 1-10, stages 0-3, 0 value differences); the 5,400 bit
     differences are all signed zeros.
- With BOTH added to the hooked run, kt-1 stage 1 is exact in every field
  and the run completes kt 1-10 with no refusal: kt-10 entry fold U 4.56%,
  interior 0.78% (hooked alone: refuses in kt 8).  Each alone refuses
  earlier than neither (transport mask alone in kt 5; top-V alone in kt 7).
- This shows only that the pair beats either statement alone on the hooked
  run.  It does NOT explain the top-V growth Decision 125 held: that was
  measured on the shipped card, where this pair has never been run.
- History: round 288 did NOT measure this pair.  It paired top-V with round
  286's STAGE-UPDATE mask, on rung 0.  The pair here (stage-TRANSPORT mask
  + top-V) has never been run on the shipped card.
- No landing: both statements are measured only on the private fold unit.
  The holder is the fold unit (see "What keeps it private").

## Provenance

- Base `710787b0c`; preregistration `35bcf9c39` (frozen before measuring).
- Instrument: round 302's history probe, arm `unit`, measured at clean tree
  `35bcf9c39` (CPU, fp64, libm, production JIT).  Card = round 223's OMT-4
  card; record = round 222's `orca2_omt4_frames_10step_a_np2`.  Fold band =
  last three T rows; interior = rows 0-144; relative = RMS(card - NEMO) /
  RMS(NEMO) on NEMO's wet support.
- The probe discards history on a refusal, so the kt-10 request was re-run
  at `--kts 7` (`measure_unit_kt10_refusal.log` keeps the refusal).
- Arms: committed driver
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round303_fold_pair_arms.py`
  (arms `hooked`, `topv`, `tmask`, `pair`, plus `--mirror`; history kept on
  refusal; test
  `tests/ocean/fidelity/test_nemo_testcase_l4_orca2_round303_fold_pair_arms.py`).
  Re-measured at clean commits: `hooked`/`topv`/`tmask` at `15cb400a2`,
  `pair` and `--mirror` at `e03e5b314` (the two differ only in how the
  refusal text is stored).  Every stage row is identical to the first
  (scratch) measurement; refusals kt 8 / 7 / 5 / none, all
  `raw-mesh e3w_int must contain only finite values > 0`.  Evidence
  `orca2_rounds/round303/committed/` (`evidence.sha256`).
- top-V = `rk3_fold_exchange_top_v_row=True` on the card config (existing
  selector, Decision 125 held).
- Transport mask = round 258's `_resolved_stage_vmask_arm` with the D114
  resolved V mask.  It replaces the V-mask argument of EVERY call of the
  stage-transport function, at all three stages.  Inside that function the
  mask feeds the qco V face thickness, the corrected stage V, the 3-D V mass
  flux and the continuity W: the whole stage transport, not only the `zFv`
  line at `:291` / `:283`.
- Round-286 stage-update mask = commit `a1dc5d235d`'s hunk on scratch
  commit `f4166a28f` in a separate clone (not committed; one arm row only).
- Evidence: `orca2_rounds/round303/`.

## 1. Hooked run, per kt (step entry, relative RMS %, fold / interior)

| kt | U | V | sea level | T | S |
|---:|---|---|---|---|---|
| 2 | 12.3 / 2.7e-14 | 62.6 / 2.1e-14 | 0 / 0 | 2.47 / 1.9e-14 | 1.69 / 9.3e-15 |
| 3 | 60.4 / 1.71 | 78.7 / 1.96 | 57.9 / 2.09 | 2.78 / 2.4e-4 | 1.89 / 8.1e-5 |
| 4 | 126 / 2.15 | 110 / 3.05 | 51.1 / 2.48 | 3.32 / 8.2e-4 | 2.25 / 2.9e-4 |
| 5 | 210 / 2.95 | 193 / 3.31 | 60.8 / 2.86 | 4.49 / 1.4e-3 | 3.07 / 4.0e-4 |
| 6 | 287 / 3.29 | 281 / 2.73 | 52.2 / 2.45 | 7.75 / 2.0e-3 | 5.31 / 5.0e-4 |
| 7 | 1550 / 3.24 | 3280 / 3.78 | 85.2 / 2.17 | 32.3 / 2.7e-3 | 22.5 / 6.0e-4 |
| 8 | refuses during kt 8 | | | | |

Round 302 (with momentum EVD), same run: kt 2 interior U 1.72%, kt 7 fold U
1170% / interior 4.94%.  kt-1 stages 1-2 are identical to round 302's in
every field and region (EVD acts at stage 3 only).

## 2. The earliest row and its statements (kt 1, stage 1, fold band)

| field | unequal (fold) | rows | max abs | interior |
|---|---:|---|---|---|
| V | 1,319 | 147 only | 0.027227136128767482 m/s | exact |
| T | 1,319 | 147 only | 0.17735225873138316 K | <= 3.6e-15 (rounding, 244 cells) |
| S | 1,319 | 147 only | 3.283688696376231 PSU | <= 7.1e-15 (rounding, 49 cells) |
| U, sea level | 0 | — | 0 | exact |

- Compiled order inside `stp_RK3_stg` (R299W7): transport (:291) -> V
  update (:480) -> correction (:549) -> tracer update (:707) -> exchange
  (:826).  Row 146 V is exact, so the exchange's source row is clean in the
  hooked run
  (rounds 289-290 held the exchange because that row carried barotropic
  debt on the shipped card; the fold unit removes that debt).
- NEMO's row-147 T is self-mirrored (pivot row); the card's is not, and 668
  of 683 wet west-half cells differ — the 668 slots rounds 247/258 traced to
  the transport's V mask.
- One-variable arms on the hooked run (fold / interior, %):

| arm | kt-1 s1 V fold unequal | kt-1 s1 T fold | refusal | kt-3 entry U | kt-5 entry U |
|---|---:|---|---|---|---|
| hooked (none) | 1,319 | 0.835 | in kt 8 | 60.4 / 1.71 | 210 / 2.95 |
| + round-286 stage-update mask | 1,319 | 0.835 | in kt 6 | 376 / 3.08 | 44,100 / 16.1 |
| + transport mask | 1,319 | 0 | in kt 5 | 93.8 / 2.22 | 3.0e7 / 4.46 |
| + top-V exchange | 0 | 0.835 | in kt 7 | 92.5 / 0.435 | 930 / 3.92 |
| + transport mask + top-V | 0 | 0 | none (kt 1-10) | 1.95 / 0.204 | 5.21 / 0.674 |

Hooked + pair, kt 6-10 entry U: 3.48 / 0.73, 2.71 / 0.65, 5.37 / 1.10,
6.00 / 0.82, 4.56 / 0.78; kt-10 stage-3 U 8.67 / 1.01, sea level 0.656 /
0.212.  Its first over-bar row moves to kt-1 stage 2 (fold U max
5.17e-6 m/s; T 6.4e-9 K, S 1.4e-8 PSU on the same 1,319 pivot cells).

## What keeps the fold unit private, and what would let it land

- It exists only as test hooks (`_NEMOWSRK3TestHooks`: external-mode
  association, reference-face-depth override, unmasked + materialised
  barotropic V transport, atomic fold unit, literal HPG V mask) plus round
  245's monkeypatched surface-mask context.  Rounds 249-259 held it because
  it refused on live W thickness (kt 4-8) and failed Decision 96 as a
  partial unit.
- Correction: card selectors DO reach parts of it.  Ladder map:
  - Matches: its barotropic core is the same code the public card setting
    `barotropic_literal_fold_unit_vector_form` switches on (the hook and the
    setting enter the same literal-fold-unit branch of the external mode);
    G1b and G6 set it on, OMT-4 sets it off.  Top-V is the same setting and
    function (`rk3_fold_exchange_top_v_row`) as on G1 and G6.
  - Gaps: this round's stage-transport V mask has no ladder counterpart (G6
    keeps the compact mask in the transport).  G6's raw RHS / stage-update V
    mask, HPG fold geometry, vorticity fold halo, ZAD pivot face and
    stage-ratio exchange do not exist in this lane.
  - Source: the barotropic-core match is read off the code here; the G6
    entries are the independent review's map, not re-measured in this
    clone.
- Measured now: with the pair above it runs OMT-4 kt 1-10 for the first
  time.  To land: promote the hook set + the two stage statements to cited
  card selectors as ONE atomic unit, then score Decision 96 on all seven
  card/label pairs (rung 0, OMT-1..4, rung 10 both labels) with GYRE
  byte-identical.  Not attempted this round (a multi-statement package
  change; the probe is OMT-4 only).

## Three certified-card fingerprint tests (red at base, unchanged)

`tests/ocean/unit/test_nemo_vortex_card.py::test_the_certified_cards_are_untouched_by_the_vortex_card`
with `[GYRE-zco]`, `[LOCK_EXCHANGE-zco]`, `[OVERFLOW-zps]`.  The digest
hashes the printed full config of each card.  Bisected (2,014 commits): last
green `674f3b0377`, first red `e487b3b42` (VORTEX_SMT round 215, barotropic
entry forcing, 2026-10-03), which added shared config fields.  Between the
last green commit and this tip, each of the three cards gains 35 config
fields (all off, or `nemo_literal` / `nemo_replace` / filter alpha / drag
`uc0` values), changes 0 and removes 0; every non-config part of the digest
is identical (`e6ee250b515d2ecf`, `8a835fbd22b2fb27`, `082c50472753aa23` at
both).  Not re-pinned: the expectation is stale in form, but whether each of
the 35 added fields is inert on LOCK_EXCHANGE and OVERFLOW is not measured
(GYRE's numbers are guarded by its byte-identical trajectory gate).

## Gates

- No `packages/` change: GYRE trajectory and day-30 gates not required.
- Citation gate (receipt commit `d59cc819c`): this receipt PASS (8
  citations, each pinned to its subroutine, 0 unmapped); default receipt
  PASS (274).  Line plant on
  `ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:826-827` fails
  (`SYMBOL-NOT-AT-LINE`, exit 1); routine plant on
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:211` fails
  (`ROUTINE-CONTEXT-MISMATCH`, exit 1).
- Tests: citation-gate tests, private-import ratchet, round-302 probe test:
  `26 passed`.
- codex review: paused (quota).  The operator runs an independent Claude
  review before landing.

## Prediction ledger

| ID | Verdict | Result |
|---|---|---|
| R303-P1 | **CONFIRMED** | kt-1 stage-1/2 rows identical to round 302's. |
| R303-P2 | **CONFIRMED** | kt-2 entry interior U 2.7e-14% (< 1.72%). |
| R303-P3 | **CONFIRMED** | kt-7 fold U 1550%; refuses during kt 8. |
| R303-P4 | **CONFIRMED** | first over-bar kt-1 stage-1 fold V (and T, S), 1,319 cells. |
| R303-P5 | **CONFIRMED, refined** | no separable statement: the owner is the transport-mask + top-V pair, which needs the private fold unit; the round-286 stage-update mask is not the owner (no effect on stage-1 T, worse later). |

## Configuration choices

- ASKED: none.  UNASKED: none.  No card, selector, default, D126 runoff or
  D132 after-SSH change; all arms were scratch.

## OPEN

1. Promote the fold unit + transport-mask + top-V pair to cited card
   selectors as one atomic unit and score Decision 96 on all seven pairs.
2. First over-bar row of the hooked + pair run: kt-1 stage-2 fold U
   (5.17e-6 m/s), walk it in compiled order.
