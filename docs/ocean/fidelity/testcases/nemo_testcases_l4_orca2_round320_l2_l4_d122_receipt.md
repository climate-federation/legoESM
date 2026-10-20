# ORCA2 round 320 — lane B's L2 isoneutral diffusion lands on rung 10; lane A's L4 geothermal refuses

Date: 2026-10-20 (Claude round; codex review: operator).

Status: **LANDED-pending** (operator review + push gate + DINO month gate).

## Verdict

- Isoneutral (L2) lands on rung 10 under Decision 122: all seven pairs
  pass, 0 exact rows lost, first over-bar row unchanged, census leans
  toward (139/40 independent, 146/33 given entry).
- Rung 10's kt-10 max temperature error falls 0.360 -> 0.202 K; max
  sea-level error moves 0.1832 -> 0.1835 m (+0.16 %).
- Geothermal (L4) refuses on rung 10, alone and with L2: lane A's own guard
  refuses it whenever tracer lateral diffusion runs, and rung 10 always runs
  it. It stays off (decision below).
- Rung 0's deck also runs isoneutral diffusion, but L2 refuses there (it needs
  the stored stage-3 tracer thickness, which rung 0 keeps off). Rung 0 stays
  as shipped, without any isoneutral block: a card/deck mismatch, now named.

## Provenance (commits, this branch)

- `246f73429` item 0: round-319 receipt records DECISION 143 (floor frozen
  out of every rung-10 trainer at 1e-10); the 238 GiB peak's cause is
  labelled PLAUSIBLE.
- `fc4d58556` import, default-off. L2 = lane B package hunks of
  `3cfd91d074..51b2582f5e` (3b2e24fdbf, 4b1bc73142, 29ae92272a, 026b3cc8dc,
  30934c2bd2, 95981cb330, 51b2582f5e), `git apply --3way`. L4 = lane A
  package hunks of `7a31b67ef..b4f70973d0` (763a8625f3, 8918d537af,
  588c0e3da2). Byte-identical to the lane tips: the isoneutral module, its
  config, the grid operators, the BBL module (taken whole: its north-neighbour
  helper moved to the grid module, plus L1's off-by-default pivot ghost) and
  the geothermal module. Conflicts: one signature in the vertical module
  (both arguments kept), one field order in the state module. Not imported:
  the ladder-card recipe hunks (this branch has no G/V/L rung tables); only
  the shared ORCA2 isoneutral constructor and VORTEX SMT-3's use of it.
- `8a6fbf66e` two order-preserving lines in imported code, so every flag-off
  program stays identical (a static +1 sign no longer adds a multiply; one
  north neighbour evaluated in its old position).
- `3da2f6835` preregistration + gate; adds the public L2 selection
  constructor (lane B's argument list).
- `50388ab5d` selection (rung 10) + round-314 audit rows; `2113923a5`
  audit table/receipt; `dd63567e4` selection test; `16ba169d2` gate control
  = card before the selection (below); `45db33d21` citation re-anchor;
  `629127391` lane B's isoneutral unit tests; `6583a4439` rung-10
  isoneutral scalar declared inert.
- Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round320_l2_l4_d122_gate.py`
  (round-285 ladder, round-318 later-first classifier = DECISION 139's
  safeguard, round-278 decision). CPU, fp64, libm, production JIT. Evidence
  `orca2_rounds/round320/` (`jaxpr/`, `probes/`, `arms/`, `final/`, `land/`).

## Step programs (normalized jaxprs, byte compare)

| tree vs base `246f73429` | identical |
|---|---|
| import, as applied | 3/11 (GYRE, DINO, rung 0, OMT-1..4, rung 10 differed) |
| import + `8a6fbf66e` | 11/11 |
| after selection | 10/11 (only rung 10 differs) |

The 11: rung 0, OMT-0..4, rung 10, GYRE, VORTEX SMT, VORTEX SMT vector, DINO.

## Which decks set each piece

- Isoneutral: rung 0 and rung 10 (lap + iso + MSC, mode 20, rn_Ud = 0.018;
  `ORCA2_OMIP_L4/EXP00/namelist_cfg:302`, `ORCA2_OMIP_L4/EXP00/namelist_cfg:304`,
  `ORCA2_OMIP_L4/EXP00/namelist_cfg:306`); OMT-1..4 state it off.
  Rung 0's NEMO run executes it (its ocean.output: ln_traldf_iso = T).
- Geothermal: rung 10 only (`ORCA2_OMIP_L4/EXP00/namelist_cfg:256`).

## Feasibility (trace only, observed before the preregistration)

| card | piece | result |
|---|---|---|
| rung 10 | L2 | traces |
| rung 0 | L2 | refuses: Kmm isoneutral geometry needs the stored stage-3 tracer thickness |
| rung 10 | L4 | refuses: "not supported with tracer lateral diffusion" |
| rung 10 | L2 + L4 | refuses (same) |

The L4 refusal is true in this model: NEMO adds tra_bbc after tra_ldf
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:738-740`); this model adds the
isoneutral increment after the stage-3 source assembly that L4 joins (read
off the step function's statement order).

## Decision 122, seven pairs (one tree `16ba169d2`, arm D122)

Control = the card before the selection; on rung 10 its isoneutral block is
rebuilt (equal to the `246f73429` card's block). Instrument check: on all
seven pairs the rebuilt control reproduces the pre-selection control
(measured at `3da2f6835`) row for row, and D122 reproduces the arm.

| pair | exact rows lost | first over-bar | census toward/away (early, late) | kt-10 max ssh before -> after (m) |
|---|---:|---|---|---|
| rung 0 | 0 | equal (kt 1 stage-1 T) | 0/0 | 0.00701981 -> same |
| OMT-1 (kt 8) | 0 | equal | 0/0 | 2.90886e-05 -> same |
| OMT-2 | 0 | equal | 0/0 | 4.30139e-05 -> same |
| OMT-3 | 0 | equal | 0/0 | 7.46458e-13 -> same |
| OMT-4 | 0 | equal | 0/0 | 1.89112e-04 -> same |
| rung 10 independent | 0 | equal (kt 1 entry ssh) | 139/40 (32/7, 107/33) | 0.183209 -> 0.183498 |
| rung 10 given entry | 0 | equal (kt 1 stage-1 T) | 146/33 (39/0, 107/33) | 0.168432 -> 0.168598 |

Decision: `PASS_R278_D122_DECISION`. The later-first safeguard was not
needed (no first row moved). Exact rows: rung 10 4 / 5, unchanged.

## Rung 10, kt-10 stage-3 errors (|card - NEMO|, round 318's instrument)

| state | label | max sea level (m) | max T (K) | RMS T (K) |
|---|---|---:|---:|---:|
| before | independent | 0.18320913 | 0.35954065 | 3.9642e-3 |
| after L2 | independent | 0.18349806 | 0.20180847 | 3.5997e-3 |
| before | given entry | 0.16843220 | 0.35956484 | 3.9644e-3 |
| after L2 | given entry | 0.16859805 | 0.20181174 | 3.5998e-3 |
| after L4, after both | either | refused, no number | | |

RMS S 2.284e-3 -> 2.267e-3 (both labels).

## Predictions (PREREG_nemo_testcases_l4_orca2_round320.md)

- P1 L2 completes on rung 10, both labels: CONFIRMED.
- P2 passes D122 (c)+(d): CONFIRMED.
- P3 lowers rung 10's kt-10 RMS T (independent): CONFIRMED, 3.964e-3 ->
  3.600e-3.
- P4 sea-level change under 20 %: CONFIRMED, +0.16 %.

## Round-314 audit rows that change

- Isoneutral MSC split, coefficient shape (mode 20) and rn_Ud: DIFFERS ->
  MATCH. Counts vs record 161/8 -> 164/5 MATCH/DIFFERS; vs EXP00 160/9 ->
  163/6. Table regenerated by the audit script at `50388ab5d`.
- Geothermal rows stay NOT-MODELLED; their reason now names the L4 refusal.

## Not done, and why

- Geothermal's input file is not added to rung 10's declared inputs:
  nothing on the card reads it while L4 refuses. It is in place for when L4
  lands: the rung-10 deck already links geothermal_heating.nc to the
  on-disk ORCA2_ICE_v5.0.0 copy, sha e369762f068a7fbc1561be310fb8ded844233da37e6a0c6ffc10805e4996fffb
  (matches the brief).
- L1 (BBL) commits are not imported beyond the BBL module's off-by-default
  helper change; L3 not imported (brief).

## Gates (x64, CPU; logs `orca2_rounds/round320/gates/`)

- Receipt citation gate on this receipt: PASS, 5 citations, 0 failures,
  0 map entries failing (`citation_gate_receipt.json`). The imports shifted
  89 map entries; re-anchored to the lines their symbols now identify (map +
  default receipt, `45db33d21`).
- Default gate (land.sh list), split across calls: citation-gate test 17,
  private-import ratchet 6, TKE terms 71, recipe 25, freshwater closure 16,
  stage face mask 2, prognostic barotropic 8, ORCA2 parallel receipt 2 =
  147 passed, 0 failed.
- Nearby: lane B's isoneutral unit tests (imported) 100, round-318
  selection 15, round-315 gradient probe 11, card constructibility 22,
  recipe unit 31, round-320 selection 3, round-314 audit 9 passed.
- Round-319 traced scalars: 12 passed, 1 failed before the fix: rung 10's
  isoneutral scalar kappa_Redi now moves 0 values raised and 0 lowered (the
  executed coefficient comes from rn_Ud). Declared inert; rerun 13 passed.
- Non-vacuity: with the card selection reverted, 2 of the 3 round-320
  selection tests fail (`nonvacuity.log`).

## OPEN

- DECISION: geothermal on rung 10. (1) lane A certifies L4 on a rung that
  also runs isoneutral diffusion (NEMO's tra_ldf -> tra_bbc order), then
  import; (2) write that order on this card now, measured under D96 only;
  (3) leave off. Pick 1.
- Rung-10 trainers: the isoneutral knob that moves the state is now rn_Ud,
  not kappa_Redi (inert). Whether a trainer exposes rn_Ud is the user's call.
- Rung 0 runs no isoneutral diffusion although its NEMO run does; L2 needs
  the stored stage-3 tracer thickness there.
- The geothermal-refusal test is not plant-checked (its guard is lane A's).

## Choices

- ASKED (brief): L2 and L4 imports; L2 selected on rung 10.
- UNASKED: geothermal input not added to rung 10's declared inputs (piece
  refused). Offered: add it now, sha-pinned, if wanted.
- UNASKED: the gate's control rebuilds rung 10's pre-selection isoneutral
  block on the selected tree (instrument only; checked row for row).
- UNASKED: two order-preserving edits to imported lane code (no value
  changes; lane tips differ from this branch by those two lines).
