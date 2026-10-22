# ORCA2 round 323 — rung 10 takes L8's stage-3 Krhs route and L5's mixed-layer eddies; L3 refused by L8's guard

Date: 2026-10-22 (Claude round; codex review: operator).

Status: **LANDED-pending** (operator review + push gate + DINO month gate).

## Verdict

- Rung 10 now runs lane A's L8 stage-3 tracer route (diffusive BBL in Krhs,
  isoneutral, geothermal, interior damping, NEMO's order) and lane B's L5
  mixed-layer eddies. Both pass Decision 122 on all seven pairs: 0 exact
  rows lost, first over-bar row unchanged, census toward.
- L3 (eddy-induced velocity) stays off: lane A's own L8 guard refuses it
  (L8 admits only the plain L2 isoneutral block). L8 + L3 is certified on no
  rung. Decision below.
- L7 river-mouth mixing: nothing imported; ORCA2's river-mouth map is zero
  at all 26,640 points, audit rows now INERT.
- Gradients on the shipped card (N = 1): 0 non-finite, reverse = forward.

## Provenance (commits, this branch, base `ea02b6e6f`)

- `a4fa89432` import, default-off: lane B L1 `40b5e44139`, `53624609d0`,
  `3cfd91d074` (BBL trend in stage-3 Krhs, pivot ghost; BBL module already
  here); lane A L4b `f0549b581d`, `ff080c094c`, `288311ef9b`; L6
  `11c7147545`, `945f885f94`; L8 `ce9636bc1f` (lane A tip `bd83f3fd1e`).
  Package hunks, `git apply --3way`. Recipe: only the shared
  climatology-at-step and SHA-pinned damping builder; lane A's ladder-card
  tables not imported. Resolutions: the stage-3 finaliser call keeps lane
  A's BBL arguments and this branch's live Krhs terms (lane A tip order);
  four operator-file and one helper conflicts keep this branch's later
  versions; a duplicated end-of-stage pivot-fold field the 3-way apply
  added was dropped (this branch's position kept; lane A's tip carries the
  same duplicate). Ladder: geothermal heat flow (sha256 `e369762f...6fffb`,
  native float32 widened) and the per-step damping target are passed when a
  card selects them.
- `5437190778` lane B L5 `acda81beb`, `bedbcf650`, `7b09b4654`, `43071fbf2`,
  `1075ccf0f` (lane B tip `de1adde51a`); keeps this branch's DECISION 147
  backstop; one order-preserving line (top-stress numerator before divisor)
  so rung 10's program is unchanged. Public `orca2_l5_mle()` (lane B's L5
  arguments) and `orca2_l8_krhs_selection()` (lane A's L8 selectors);
  preregistration and gate.
- `439644db6` `import hashlib` for the imported damping builder (first L8
  arm hit NameError; nothing measured before the fix).
- `ff26b7cdd` selection (rung 10), validator rn_gambbl 10.0, rung 0 /
  OMT-1..4 state every new selector off; `51144f58f`, `99be4dd80`,
  `6f9be3312`, and the table commit: round-314 audit; `c75b708f0` rung-10
  step callers pass the damping target, older selection tests keep their
  round-322 card, new selection test.
- Gate `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round323_l8_l3_l5_d122_gate.py`
  (round-285 ladder, round-318 later-first classifier = DECISION 139's
  safeguard, round-278 decision). CPU, fp64, libm, production JIT. Evidence
  `orca2_rounds/round323/` (`jaxpr/`, `fp/`, `arms/`, `final/`, `grad/`,
  `gates/`).

## Flag-off identity

| tree vs base `ea02b6e6f` | normalized step programs | rung-10 values |
|---|---|---|
| L1/L4b/L6/L8 import | 11/11 identical | — |
| + L5 import | 10/11 (rung 10: two independent statements swapped) | 72/72 leaf-steps bit-identical (3 steps) |
| + order-preserving line | 11/11 identical | (same program) |
| selection | 10/10 non-rung-10 identical | rung 10 moves (selected) |

The 11: rung 0, OMT-0..4, rung 10, GYRE, VORTEX SMT flux and vector, DINO.

## Which records run each piece

Rung 10 only: its record states ln_trabbl, ln_trabbc (nn_geoflx = 2),
ln_tradmp (nn_zdmp = 0), ln_ldfeiv and ln_mle all T (EXP00
`ORCA2_OMIP_L4/EXP00/namelist_cfg:256`, `ORCA2_OMIP_L4/EXP00/namelist_cfg:312`,
`ORCA2_OMIP_L4/EXP00/namelist_cfg:317`, `ORCA2_OMIP_L4/EXP00/namelist_cfg:328`);
rung 0 and OMT-1..4 state all five F. Same compiled lines on the full card:
tra_bbc then tra_bbl then tra_dmp at stage 3
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:738-740`); MLE and EIV
transports at stage 3 (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:255-260`).
resto.nc sha256 `f4b6aec4...8b634` (lane A's pin, checked at read).

## Decision 122, seven pairs (arms at `439644db6`)

| step | pair | exact rows lost | first over-bar | census toward/away (early, late) |
|---|---|---:|---|---|
| L8 | rung 0, OMT-1..4 | 0 | equal | 0/0 (cards unchanged) |
| L8 | rung 10 independent | 0 | equal (kt 1 entry ssh) | 116/63 (18/21, 98/42) |
| L8 | rung 10 given entry | 0 | equal (kt 1 stage-1 T) | 134/45 (25/14, 109/31) |
| L5 on L8 | rung 10 independent | 0 | equal | 118/61 (23/16, 95/45) |
| L5 on L8 | rung 10 given entry | 0 | equal | 121/58 (22/17, 99/41) |
| L8 + L5 | rung 10 independent | 0 | equal | 122/57 (21/18, 101/39) |
| L8 + L5 | rung 10 given entry | 0 | equal | 130/49 (26/13, 104/36) |

All three: `PASS_R278_D122_DECISION`. L5's increment is scored against a
copy of the L8 arm relabelled control (rows untouched; `arms/*L8base.json`).
Final tree `ff26b7cdd`, arm D122 vs rebuilt control: PASS, the same
numbers; D122 rows equal the L8 + L5 arm rows and the rebuilt control rows
equal the `439644db6` control rows on all seven pairs. OMT-3/4 arms first
refused to stamp (tree edited mid-run; nothing measured) and were rerun
from a clean worktree at `439644db6`.

## Rung 10, kt-10 stage-3 errors (|card - NEMO|)

| state | label | max sea level (m) | max T (K) | RMS T (K) | RMS S |
|---|---|---:|---:|---:|---:|
| before (round 322) | independent | 0.18349806 | 0.20180847 | 3.59966e-3 | 2.26655e-3 |
| L8 | independent | 0.18349769 | 0.20180847 | 3.59963e-3 | 2.26655e-3 |
| L8 + L5 | independent | 0.18349390 | 0.20180779 | 3.53189e-3 | 2.26276e-3 |
| before | given entry | 0.16859805 | 0.20181174 | 3.59978e-3 | 2.26666e-3 |
| L8 | given entry | 0.16859778 | 0.20181174 | 3.59975e-3 | 2.26667e-3 |
| L8 + L5 | given entry | 0.16859636 | 0.20181106 | 3.53214e-3 | 2.26284e-3 |

L3: refused at construction, no number.

## Predictions (PREREG_nemo_testcases_l4_orca2_round323.md)

- P1 L8 completes, finite, both labels: CONFIRMED.
- P2 L8 passes D122: CONFIRMED.
- P3 L8 lowers RMS T (independent): CONFIRMED, barely (3.59966e-3 ->
  3.59963e-3); ten steps of geothermal and damping are small.
- P4 sea-level change < 20 %: CONFIRMED (-0.0002 %).
- P5 L8 + L5 completes and passes D122 against L8: CONFIRMED.
- P6 L5 lowers RMS T: CONFIRMED, 3.59963e-3 -> 3.53189e-3 (-1.9 %).
- P7 gradients: CONFIRMED (below).

## Gradients (shipped card, L8 + L5, N = 1; `grad/grad_rung10_n1_L8L5.json`)

- T loss w.r.t. initial T: 0 non-finite entries (sea-level loss also 0);
  one T direction reverse vs forward 2.0e-15 (sea level 4.0e-14).
- rn_Ud: reverse -4.51364e-5, forward equal to 7.5e-16, best FD 2.8e-6
  (round 322: -4.51375e-5). Sea-level loss: exactly 0 at one step.
- Traced and production steps bit-identical (0 leaves); rebuilt-at-1e-4
  model equals the traced step (0 leaves). Peak 67.7 GiB.
- Every round-316 guard is present in each merged function (counts
  unchanged in all five files; lane B adds one in the MLE path).

## Round-314 audit rows that change

- ln_trabbc, nn_geoflx, ln_tradmp, nn_zdmp, ln_tsd_dmp, ln_mle, nn_mle,
  rn_ce, rn_lat, rn_rho_c_mle, nn_mld_uv, nn_conv: -> MATCH. rn_lf and
  rn_time: INERT (nn_mle = 0 only). River-mouth rows (ln_rnf_mouth, rn_hrnf,
  rn_avt_rnf, sn_cnf): INERT. File-descriptor rows: forcing input.
- Counts vs record 164/5 -> 176/4 MATCH/DIFFERS; vs EXP00 163/6 -> 175/5.

## Choices made

- L8 + L5 selected on rung 10 — ASKED (round prompt; D122 passes).
- rn_gambbl 0.0 -> 10.0 on rung 10 (record value; inert with advective BBL
  off; L8's guard requires it; validator updated) — UNASKED, inside the L8
  selection; offered for revert.
- Rung 0 / OMT-1..4 builders restate the new selectors off — keeps their
  pre-round values; no behaviour change.
- L5's increment scored against a relabelled L8 arm — instrument choice;
  the final-tree D122 run does not depend on it.

## OPEN

- DECISION: L3 with L8 on rung 10. (1) keep L3 off until a ladder rung
  certifies L8 + L3 (lane A or B); (2) widen L8's guard to admit L3's block
  and measure under D122 (uncertified composition). Pick 1.
- Rounds 319/322 gradient tests on the selected card: result in Gates.
- Every rung-10 step caller must now pass tra_dmp's step target; a future
  rung-10 trainer must too (the gradient probe shows how).

## Gates (x64, CPU; logs `orca2_rounds/round323/gates/`)

- ORCA2 selection tests (rounds 300-323, excluding 319/322): 143 + 18 passed
  after fixes (first run: 12 failed, 10 errors, all from the selection: no
  damping target passed, or expectations of the round-322 card).
- Default gate (land.sh list): private-import ratchet 6, TKE terms 71,
  recipe 25, freshwater 16, prognostic barotropic 8, ORCA2 parallel
  receipt 2 = 128 passed; remainder below.
