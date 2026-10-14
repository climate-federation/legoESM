# ORCA2 round 278 — Decision 122 landing

**Status: LANDED.**  Fourteen geometry-ladder-certified units (fifteen
statements) are selected on the production ORCA2 card.  Singly and together
they lose no exact row and keep the first over-bar row unchanged or toward
on rung 0, OMT-1..4 and rung 10 under both labels.  Two certified statements
stay off: G1's EEN vorticity order fails (b), and the completed-stage top-V
fold exchange meets (c)+(d) but grows the ten-step SSH maximum ~20x on rung 0
and OMT-4, so it is held for a user decision.

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round278.md`
(commit `18663a801`, before any ladder).  Gate:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round278_d122_landing_gate.py`
(round 277's measurement, round 276's classifier; `--decide` applies (c)+(d)
over the seven card/labels).  Singles measured at `18663a801`, the final set
and its control at `2418d64d3`, each against its own same-tree control.
Counts are RMS directions `toward / away / score-equal`, registered only.
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round278/`.

## (1) Candidates

(a) is the grid-ladder receipt (gridladder branch).  (b) was checked
mechanically: each cited ORCA2G1 build line was found verbatim in the
shipped ORCA2 build, and the enclosing branch is the one ORCA2's resolved
namelist runs (`ln_eos80`, `ln_drgimp`, `ln_dynspg_ts`, `ln_dynadv_vec`,
`ln_dynvor_een` all true; stage ratios in the default `np_HYB` case).
All fifteen selectors were off on the ORCA2 card before this round.

| unit | ladder receipt | ORCA2 build line |
|---|---|---|
| G2 dyn_zdf live face / 1-D interface divisors | G2 round 2 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:192-198`, `:361-367` |
| G3 stored stage ratio r3u(Kmm) (with its divisor host) | G3 round 3 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221-223` |
| G2 tra_zdf rounded live T thickness | G2 round 3 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:231-233` |
| G2 ascending-level RHS depth sum | G2 round 4 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:198-199` |
| G1 EEN ratio fold exchange + fe3mask pivot row (atomic) | G1 round 6 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:169`; `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:232`, `:258` |
| G1 EEN ratio written order | G1 round 8 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:266-269` |
| G1 stage-1 RHS handoff | G1 round 6 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:141`, `:147` |
| G1 stage velocity written order | G1 round 6 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:474-479`, `:514-515`, `:536-537` |
| G1 ZDF barotropic removal | G1 round 7 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:158-161` |
| G1 stage F ratio | G1 round 7 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:224`, `:245` |
| G1 stage T ratio; stored T-ratio product | G1 rounds 9, 8 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221-223` |
| G1 EOS depth product | G1 round 9 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:357`, `:646` |
| G1 stage U/V ratio | G1 round 9 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:178-179` |
| G1 completed-stage top-V fold exchange | G1 round 7 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:795-796` |

Not a candidate: G1's EEN vorticity written order certifies the planetary
`np_COR` line `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:745`; ORCA2's
vector-form call runs `np_CRV`, so (b) fails and Decision 96 governs it.

## (2) Census, singly and as the final set

Every arm: zero exact rows lost on every card; first over-bar row (kt=1
stage-1 T; rung 10 independent: its own kt=1 entry SSH) unchanged or toward.
All fifteen units and the final set meet (c)+(d).  Round 275-277 counts
reproduce exactly (e.g. divisors 132/47/0, ascending sum 91/86/18 and
78/92/10).

| unit | rung 0 | OMT-1 | OMT-2 | OMT-3 | OMT-4 | rung 10 given | rung 10 indep | (c)+(d) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| G2 dyn_zdf divisors | 132/47/0 | 61/78/0 | 85/94/0 | 81/98/0 | 104/75/0 | 75/104/0 | 72/107/0 | pass |
| G3 stored r3u (+ divisor host) | 132/47/0 | 61/78/0 | 85/94/0 | 81/98/0 | 104/75/0 | 75/104/0 | 72/107/0 | pass |
| G2 tra_zdf rounding | 76/103/0 | 36/103/0 | 122/57/0 | 94/82/3 | 50/128/1 | 112/67/0 | 84/94/1 | pass |
| G2 ascending RHS sum | 91/86/18 | 80/68/5 | 89/76/30 | 80/66/48 | 103/80/12 | 78/92/10 | 58/116/6 | pass |
| G1 EEN fold pair | 114/61/0 | 57/78/0 | 103/72/0 | 107/68/0 | 107/68/0 | 116/79/0 | 115/80/0 | pass |
| G1 EEN ratio order | 59/70/32 | 26/30/47 | 23/57/59 | 35/39/66 | 65/69/30 | 59/69/25 | 41/77/13 | pass |
| G1 stage-1 RHS | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | pass |
| G1 stage velocity order | 89/82/18 | 64/27/48 | 48/101/33 | 75/45/63 | 94/74/21 | 75/95/19 | 71/107/11 | pass |
| G1 ZDF barotropic removal | 77/93/7 | 54/35/45 | 61/84/31 | 83/53/40 | 86/76/17 | 80/83/14 | 68/100/10 | pass |
| G1 stage F ratio | 39/73/34 | 14/19/52 | 32/53/41 | 52/30/66 | 61/53/41 | 88/49/28 | 78/59/20 | pass |
| G1 stage T ratio | 68/66/34 | 12/37/64 | 63/70/42 | 28/53/74 | 83/64/26 | 68/71/27 | 72/68/14 | pass |
| G1 stored T-ratio product | 73/67/29 | 30/37/68 | 57/76/50 | 58/34/84 | 87/63/31 | 68/83/20 | 78/70/21 | pass |
| G1 EOS depth product | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | pass |
| G1 stage U/V ratio | 59/74/15 | 42/42/49 | 87/58/23 | 44/41/45 | 72/67/15 | 84/60/18 | 83/53/13 | pass |
| G1 top-V fold exchange | 23/165/0 | 99/43/0 | 118/64/0 | 127/55/0 | 23/165/0 | 24/160/4 | 23/161/4 | pass |
| **final set (14 units)** | 139/52/4 | 68/80/5 | 88/99/8 | 89/99/7 | 114/79/2 | 81/114/0 | 79/116/0 | pass |

| final set: card (label) | early | late | first away kt | first over-bar row | kt=10 SSH max before -> after (m) |
|---|---:|---:|---:|---|---|
| rung0 (independent) | 28/23/4 | 111/29/0 | 1 | kt=1 stage1 T: toward | 0.384033466829 -> 0.383539664887 |
| omt1 (independent) | 18/30/5 | 50/50/0 | 1 | kt=1 stage1 T: equal | 0.746821544969 -> 0.746717960307 |
| omt2 (independent) | 19/28/8 | 69/71/0 | 1 | kt=1 stage1 T: equal | 0.34648943844 -> 0.346451366223 |
| omt3 (independent) | 18/30/7 | 71/69/0 | 1 | kt=1 stage1 T: equal | 0.363539967902 -> 0.363550379935 |
| omt4 (independent) | 29/24/2 | 85/55/0 | 1 | kt=1 stage1 T: toward | 0.393740852853 -> 0.393531689285 |
| rung10 (given_nemo_entry) | 20/35/0 | 61/79/0 | 1 | kt=1 stage1 T: toward | 0.287207342138 -> 0.28674475751 |
| rung10 (independent) | 25/30/0 | 54/86/0 | 1 | kt=1 entry ssh: equal | 0.292694913522 -> 0.293848416123 |

The stage-1 RHS and EOS depth product units are bit-inert on all seven
card/labels (no row moves).  The G3 pair's votes equal the divisors' on every
card, as in round 277.  R278-P1, P2 **CONFIRMED**; P4 **CONFIRMED** (the
final set passes without bisection).

**Top-V fold exchange (R278-P3 CONFIRMED, held):** kt=10 SSH maximum
0.384 -> 8.388 m on rung 0, 0.394 -> 16.334 m on OMT-4, 0.287 -> 1.245 m
(given) and 0.293 -> 1.261 m (independent) on rung 10; OMT-1..3 unchanged.
It meets (c)+(d), so Decision 122 as written would land it; per the
preregistration it stays off pending the user's decision.

## (3) Landed selectors

Production ORCA2 card (`build_orca2_zps_card`); the rung-0 and OMT cards are
built from it and inherit the same selections, as for every earlier ORCA2
landing.  The landed card's model config and coordinate equal the measured
final-set arm exactly (checked on rung 10 and OMT-4).

- zdf_momentum_nemo_face_thickness: false -> true on the ORCA2 production card
- zdf_momentum_stage_r3u_nemo_literal: false -> true on the ORCA2 production card
- zdf_tracer_nemo_rounded_thickness: false -> true on the ORCA2 production card
- barotropic_rhs_ascending_level_sum: false -> true on the ORCA2 production card
- barotropic_een_ratio_fold_exchange: false -> true on the ORCA2 production card
- barotropic_een_fe3mask_pivot_row: false -> true on the ORCA2 production card
- barotropic_een_ratio_written_order: false -> true on the ORCA2 production card
- rk3_stage1_rhs_nemo_literal: false -> true on the ORCA2 production card
- rk3_stage_velocity_written_order: false -> true on the ORCA2 production card
- zdf_barotropic_removal_nemo_literal: false -> true on the ORCA2 production card
- rk3_stage_r3f_nemo_literal: false -> true on the ORCA2 production card
- rk3_stage_r3t_nemo_literal: false -> true on the ORCA2 production card
- nemo_r3t_stored_product: false -> true on the ORCA2 production card
- nemo_eos_gdept_rounded_product: false -> true on the ORCA2 production card
- rk3_stage_r3uv_nemo_literal: false -> true on the ORCA2 production card

Package defaults are unchanged (all false).  Other cards whose NEMO program
compiles the same lines (selection is a separate decision): GYRE (RK3 build:
dyn_zdf, tra_zdf, stp2d and stprk3_stg lines) and DINO (dyn_zdf, tra_zdf and
stp2d lines).

## Gates and tests

- GYRE ten-step trajectory: 70 rows, 0 ULP worsening, first over-bar kt=3
  unchanged; residual archive SHA256
  `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`, equal
  to rounds 276-277 (the gate's own exit 1 is its registered debt rows, as
  before).  CPU 30-day member: all 30 daily snapshots byte-identical to the
  pre-round hashes; day 30
  `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.
- Focused tests: 91/91 pass (G1 transfer, round-276/277 imports, G1
  ORCA2-grid ladder).  Updated: the card test now pins the fifteen selectors
  on and the two held ones off; the four "only the G1 card selects" source
  ratchets now name exactly two selecting builders (G1 card, ORCA2 card) for
  landed statements and one for held ones; the round-277 precondition now
  reads the landed card.  Non-vacuity: with only the card edit reverted, all
  five updated tests fail; restored and verified.
- The full `tests/ocean/fidelity` battery was not run.
