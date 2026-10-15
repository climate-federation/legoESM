# Grid ladder — G4 round 8 (gridladder_r28)

Status: **LANDED, DEBT. G4 is not certified.** GIVEN-NEMO-ENTRY stays at the
bar for every field kt 1..10. INDEPENDENT now first exceeds the bar at kt 9
(ssh, u, v), up from kt 6 (u, v) in round 7.

Preregistration `PREREG_nemo_testcases_gridladder_g4_round8.md` (`08cabcb5c`,
before any measurement). Record `gridladder_rounds/round21/oracle_g4_r21`.
Evidence `gridladder_rounds/round28/` (`manifest.sha256`).

## Round 28 — housekeeping

- G1b round-2 Decision-122 pointer A cited the full-ORCA2 `dynvor.f90:648` as
  "vor_een np_CRV CASE"; that line is the np_CRV case of `vor_ens`. Corrected
  to :762-766 in that receipt with a Rule-11 note. Every other ORCA2-build
  pointer in the G1b and G4 receipts was opened at its enclosing SUBROUTINE
  (29 pointers) and names the routine its label says. No gate code changed.
- The synthetic G1b written-order jit test now asserts at least one fused
  element (was exactly 465). Round-4 receipt gains `l0/stages_kt2.json`
  4bf5f38d4c7b2f0f.

## Where the INDEPENDENT chain first leaves NEMO

Both labels share kt 1, so the first unequal value of the independent chain
is at kt 1. Stages 1 and 2 are equal in every value (56 u signed-zero bits).
Stage 3 temperature is the first leave: 489 cells, max 3.552713678800501e-15;
stage-3 salinity, ssh, uu_b, vv_b, u, v are value-identical. Exact salinity
rules out the shared implicit solve and points at a temperature-only stage-3
Krhs statement.

## Stage-3 walk in compiled order (kt 1, given-entry, stage 3 T)

| arm | T unequal | T max abs |
|---|---:|---:|
| before (option deselected; control) | 489 | 3.552713678800501e-15 |
| + NEMO-written non-solar heat (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trasbc.f90:307`) | 345 | 7.105427357601002e-15 |
| + solve RHS rounded as written (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trazdf.f90:288-289`) — landed | 200 | 7.105427357601002e-15 |
| + literal `e3t_0*(1+r3t)` thicknesses, frozen runoff depth | 3438 | 4.802366831668792e-05 |
| + literal thicknesses, live runoff depth (three variants) | 181 | 9.784763888021075e-09 |
| landed + materialized `Krhs + qsr` (review) | 200 | 7.105427357601002e-15 |

Every other stage-3 field stayed value-identical in every arm.

1. **Non-solar heat.** NEMO clears Krhs, adds `r1_rcp*qns*z1_rho0_e3t` on the
   Kmm top cell, then runoff, then Kmm solar
   (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/traqsr.f90:393`). The card derived the
   heat as (Kbb tendency − Kbb solar) × h(Kbb)/h(Kmm) from `q_net`, which is
   `round(qns+qsr)`, not `qns`. The card now receives NEMO's recorded `qns` in a
   new optional forcing field and transcribes the statement.
2. **Solve RHS.** `e3t(Kbb)*T + p2dt*e3t(Kmm)*Krhs` was fused by XLA; each
   product and the sum are now rounded as written.
3. **Refuted (Rule 11).** I claimed NEMO's runoff depth is frozen at run start
   (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcrnf.f90:489`, set once in
   `sbc_rnf_init`) and is carried state the restart does not hold. The frozen
   depth made 3342 runoff cells wrong by up to 4.8e-5; the live Kmm depth is
   what matches. Retracted.
4. **Refuted.** NEMO's literal `e3t_0*(1+r3t)` at Kbb/Kmm in the heat divisor,
   solar kernel and solve RHS made 90 runoff cells wrong by up to 9.8e-9 in
   every variant tried (stage-2 stored ratio, ratio of the mean sea level, live
   or literal runoff depth). The card's solve matrix is built on its own
   thickness ladder, so the RHS must use the same. Not landed.

Decision 122: the same statements run on the full card at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:307` and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:288-289`. ORCA2's card does not
select the option; UNMEASURED there.

Selection: `rk3_stage3_heat_nemo_written` defaults False; only G4 selects it
(through the same per-rung table as round 7's tracer order); G1, G1b, G2, G3
unchanged. With it on, a missing `qns` or a non-NEMO solar scheme raises.

## Both labels, kt 1..10 (scored at `7770b0f12`)

Cells are `status max_abs/unequal`; A = AT-BAR, D = DEBT; bar 1e-15 normalized.

### INDEPENDENT

| kt | ssh | uu_b | vv_b | u | v | T | S |
|---:|---|---|---|---|---|---|---|
| 1 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/200 | A 0/0 |
| 2 | A 0/0 | A 0/0 | A 0/0 | A 2.914e-16/2 | A 3.469e-16/2 | A 7.105e-15/308 | A 0/0 |
| 3 | A 0/0 | A 0/0 | A 0/0 | A 3.053e-16/5 | A 3.400e-16/6 | A 7.105e-15/389 | A 0/0 |
| 4 | A 0/0 | A 0/0 | A 0/0 | A 3.331e-16/5 | A 3.192e-16/6 | A 7.105e-15/472 | A 0/0 |
| 5 | A 1.110e-16/51 | A 6.939e-18/43 | A 4.337e-18/54 | A 2.776e-16/15 | A 2.845e-16/26 | A 7.105e-15/541 | A 0/0 |
| 6 | A 4.996e-16/816 | A 4.163e-17/834 | A 2.776e-17/855 | A 5.274e-16/296 | A 9.992e-16/369 | A 1.066e-14/632 | A 0/0 |
| 7 | A 6.661e-16/2803 | A 5.551e-17/2902 | A 6.592e-17/2956 | A 6.106e-16/1275 | A 9.992e-16/1435 | A 1.066e-14/688 | A 7.105e-15/1 |
| 8 | A 2.498e-15/6040 | A 1.249e-16/6304 | A 1.388e-16/6374 | A 1.665e-15/3719 | A 1.776e-15/3740 | A 1.421e-14/752 | A 7.105e-15/2 |
| 9 | D 4.585e-14/9138 | A 4.753e-16/9712 | A 7.772e-16/9614 | D 1.647e-14/6685 | D 1.642e-14/6855 | A 1.421e-14/800 | A 7.105e-15/2 |
| 10 | D 5.596e-14/11577 | A 8.882e-16/12067 | A 1.110e-15/12072 | D 3.758e-14/8877 | D 3.694e-14/9250 | restart | restart |

Surface rows 68: 62 AT-BAR, 6 DEBT (was 53/15). All-level rows: 4 DEBT (kt 9,
10 u and v; worst relative 1.26e-14 at kt 10 u partial), was 10. Every row
DEBT now was DEBT in round 7; no AT-BAR row left the bar. Region worst:
open interior 1.515e-14 (kt 9 ssh, 8230 cells), seam 4.97e-15 (kt 10 u), fold,
south and coast at the bar.

### GIVEN-NEMO-ENTRY (entries 1/3/5/7/9 + two-step successors)

Every field AT-BAR at kt 1..10, surface and all-level rows. Worst: T
7.105e-15 (≤ 326 cells per kt); u/v ≤ 3.469e-16; ssh, uu_b, vv_b, S exact.

The review fix (materialized `Krhs + qsr`) moved no kt-1 stage-3 bit; the
ladders were not re-run after it.

## Verdict and OPEN

G4 is **DEBT**: INDEPENDENT ssh/u/v exceed the bar at kt 9-10. The first leave
is still kt 1 stage-3 T, now 200 cells. Next open, in compiled order: the
card's stage thickness ladder (`h_k_old`, midpoint Kmm, and the solve
diagonal's Kaa) against NEMO's stored `e3t_0*(1+r3t)` — it must change for the
whole solve together, not the RHS alone (finding 4).

## Tests and review

- Fast G4 + G2-ladder + private-import + dispatch: 67 passed, 1 failed (the
  known pre-existing slab-land dispatch guard).
- Slow G4 stage route: 1 passed (282 s); asserts 200 with the option and 489
  deselected (the control).
- New solve-RHS unit test: passes; XLA does not fuse that small program, so it
  pins the written arithmetic only (removing the rounding still passes). The
  route control carries non-vacuity.
- Citation tests: 196 passed after the mechanical reanchor (three span extents
  grew; endpoints unchanged).
- Single review (codex): "DO NOT SHIP" — (1) midpoint Kmm thickness instead of
  `e3t_0*(1+r3t(Kmm))`: measured in finding 4, makes 90 cells worse by up to
  9.8e-9; not acted on, registered as the next open. (2) final `Krhs + qsr`
  not materialized: fixed. It confirmed card isolation, sign, level-1 heat,
  runoff before solar, `c_sw = 3991.86795711963`, no other temperature term
  dropped, salinity unchanged.

ASKED: housekeeping; first-leave walk; landing cited statements (G4-selected);
both labels; tests; one review. UNASKED, offered for revert: the new optional
forcing field `q_nonsolar` (needed to transcribe `qns`; inert unless the G4
option is on).

UNVERIFIED: the full ORCA2 card's response (pointer only); whether the kt 9
open-interior debt is owned by the thickness ladder or a momentum statement.
