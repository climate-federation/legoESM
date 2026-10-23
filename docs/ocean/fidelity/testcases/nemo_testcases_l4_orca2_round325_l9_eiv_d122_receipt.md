# ORCA2 round 325 — rung 10 runs L3's eddy-induced velocity on lane A's L9 route (DECISION 151)

Date: 2026-10-23 (Claude round; codex review: operator).

Status: **LANDED-pending** (operator review + push gate + DINO month gate).

## Verdict

- Rung 10 now runs ORCA2's eddy-induced velocity (L3) on lane A's
  certified L9 route. Decision 122 passes on all seven pairs, both labels,
  in BOTH drag arms: 0 exact rows lost, first over-bar row unchanged.
- The two arms agree, so no baseline choice is needed.
- With the top drag on (the NEMO comparison), kt-10 stage-3 RMS T error
  halves: 1.578e-3 -> 7.74e-4 K (both labels). Sea level moves <= 0.4 %.
- Gradients with L3 on (N = 1, both drag arms): 0 non-finite,
  reverse = forward.

## Provenance (base `1da92cbf3`)

- `10f26fc87` import, default-off, from fidelity/nemo-testcases-orca2-gridladder
  (L9 certified, round 57): the selector `rk3_stage3_l9_eiv_nemo_order`
  (`e4f204272d`) and the guard hunk as left at lane A tip `e3aed2ffc6`
  (`f644ffd530`, `287994dbe4` did not change it): L8's composition admits L2's
  block or L9's exact L3 selection; L9 refuses without the complete L8.
  Not imported: lane A's `_check_l3_krhs_route` and its step / inner calls
  (this branch already carries DECISION 147's guard, round 321, same
  condition), lane A's citation spacer lines and EIV-site comments, and
  lane A's ladder-card tables (as in round 323).
- The eddy-velocity arithmetic is NOT re-imported: this branch runs lane B's
  L3 (round 321) through `_nemo_ws_add_stage3_bolus`, which does lane A's
  inline L9 statements in the same order (add to the stage-3 U/V pair, then
  one transport-indicator wzv on the updated pair). Read off the code, not
  measured against lane A's record.
- Lane A's L9 card and this card build the isoneutral block with the same
  constructor body and the same `_SMT3_RN_UD/LD` (text diff); the L9 guard's
  every condition holds for `orca2_l3_eiv_gm_redi()` (kappa_GM = aei0 =
  3000, Treguier on, kappa_min 0, bolus through FCT, face-averaged,
  `nemo_left`, `nemo_forward_exact`).
- NEMO order with both bolus terms on: EIV then MLE at stage 3
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:255-260`); this branch calls
  the EIV addition before the MLE addition.
- `7efb33e9b` preregistration. `f57604a68` selection + gate
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round325_l9_eiv_d122_gate.py`).
  `0ea810cfe` the rung-0 builder (rung 0 and OMT-1..4 derive from the rung-10
  card) states the L9 selector off; without it the first 20 non-rung-10 arms
  refused at construction (nothing measured; `arms_refused_l9_rung0/`).
  Tests, audit and citation re-anchor: `a56f1686a`, `19c56e947`, the audit
  pin and re-anchor commits.

## Selection (rung 10 only)

`gm_redi = orca2_l3_eiv_gm_redi()` and `rk3_stage3_l9_eiv_nemo_order = True`,
as ORCA2's `&namtra_eiv` resolves: ln_ldfeiv = T
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:317`), nn_aei_ijk_t = 21
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:319`), rn_Ue = 0.03 and rn_Le = 200 km
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:322-323`; aei0 = 1/2 rn_Ue rn_Le = 3000
m2/s), ln_eke_equ = F (`ORCA2_OMIP_L4/EXP00/namelist_ref:980`). Rung 0
and OMT-1..4 state ln_ldfeiv = F and are unchanged.

## Flag-off identity (import `10f26fc87` vs base; `orca2_rounds/round325/jaxpr/`)

- Normalized step programs: 11/11 identical (rung 0, OMT-0..4, rung 10,
  GYRE, VORTEX SMT flux and vector, DINO).
- Rung 10, 3 production steps: 72/72 leaf-steps bit-identical (SHA-256).
- Round 323's probe stepped rung 10 without its per-step damping target,
  which the card refuses since round 323; the copy here passes it.

## Decision 122, seven pairs (arms at `0ea810cfe`, `orca2_rounds/round325/arms/`)

| drag arm | pair | exact rows lost | first over-bar (D139) | census toward/away (early, late) |
|---|---|---:|---|---|
| off (shipped) | rung 0, OMT-1..4 | 0 | equal | 0/0 (cards unchanged) |
| off | rung 10 independent | 0 | equal (kt 1 entry ssh) | 140/39 (28/11, 112/28) |
| off | rung 10 given entry | 0 | equal (kt 1 stage-1 T) | 155/24 (31/8, 124/16) |
| on (comparison) | rung 0, OMT-1..4 | 0 | equal | 0/0 (cards unchanged) |
| on | rung 10 independent | 0 | equal (kt 1 entry ssh) | 144/35 (32/7, 112/28) |
| on | rung 10 given entry | 0 | equal (kt 1 stage-1 v) | 179/0 (39/0, 140/0) |

`decide7_l3.json`, `decide7_l3_drag.json`: both `PASS_R278_D122_DECISION`.

- Controls reproduce round 324: drag-off control vs round 324 control and
  drag-on control vs round 324 drag arm, 0 of 1,360 rows differ each.
- The classifier admits only a control labelled `control`; the drag-on
  censuses score copies of the `control_drag` files relabelled (one key
  changed, checked; `*.as_control.json`), as round 323 did.

## Rung 10, kt-10 errors (|card - NEMO|; `arms/kt10_rung10.json`)

Full domain, both ranks (kt-10 entry = after nine steps; 799,200 T points):

| drag | label | arm | max sea level (m) | max T (K) | RMS T (K) | max u / v (m/s) | max S |
|---|---|---|---:|---:|---:|---:|---:|
| off | independent | before | 0.30894873 | 0.26943744 | 4.33878e-3 | 0.2215 / 0.1711 | 0.2012 |
| off | independent | L3 | 0.30901579 | 0.26949912 | 4.03572e-3 | 0.2216 / 0.1711 | 0.2012 |
| off | given entry | before | 0.32475315 | 0.26947486 | 4.33901e-3 | 0.2217 / 0.1711 | 0.2012 |
| off | given entry | L3 | 0.32482026 | 0.26953655 | 4.03578e-3 | 0.2217 / 0.1711 | 0.2012 |
| on | independent | before | 0.02255388 | 0.19457645 | 1.92985e-3 | 0.0472 / 0.0516 | 0.0478 |
| on | independent | L3 | 0.02256209 | 0.19673128 | 1.10251e-3 | 0.0472 / 0.0516 | 0.0478 |
| on | given entry | before | 0.00705928 | 0.19458572 | 1.93002e-3 | 0.0472 / 0.0516 | 0.0478 |
| on | given entry | L3 | 0.00706750 | 0.19674038 | 1.10230e-3 | 0.0472 / 0.0516 | 0.0478 |

Kt-10 stage 3 (rank 0 only, 399,600 points; NEMO wrote stage frames for
rank 0 alone):

| drag | label | arm | max sea level (m) | max T (K) | RMS T (K) | max u / v (m/s) |
|---|---|---|---:|---:|---:|---:|
| off | independent | before | 0.18349390 | 0.20180779 | 3.53189e-3 | 0.1536 / 0.1409 |
| off | independent | L3 | 0.18349380 | 0.20177144 | 3.25144e-3 | 0.1535 / 0.1410 |
| off | given entry | before | 0.16859636 | 0.20181106 | 3.53214e-3 | 0.1535 / 0.1409 |
| off | given entry | L3 | 0.16859913 | 0.20177471 | 3.25153e-3 | 0.1535 / 0.1409 |
| on | independent | before | 0.01751246 | 0.18333344 | 1.57797e-3 | 0.0276 / 0.0525 |
| on | independent | L3 | 0.01750930 | 0.18332139 | 7.74538e-4 | 0.0276 / 0.0525 |
| on | given entry | before | 0.00208004 | 0.18333441 | 1.57795e-3 | 0.0275 / 0.0525 |
| on | given entry | L3 | 0.00208835 | 0.18332235 | 7.74385e-4 | 0.0275 / 0.0525 |

Full-domain max T rises slightly with L3 in the drag-on arm (0.1946 ->
0.1967 K); where that point sits: not located.

## Gradients (L3 on, N = 1; `grad/grad_rung10_n1_l3_drag_{off,on}.json`)

Round 324's committed probe, unchanged, at clean `f57604a68` (the rung-10
card there is the shipped L3 card; the later commits touch rung 0's builder,
tests and docs only).

| drag | T non-finite | T dir rev vs fwd | ssh dir rev vs fwd | dL/dln(rn_Ud) rev | rev vs fwd | best FD | peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 0 | 5.5e-15 | 4.0e-14 | -4.511811e-5 | 1.5e-15 | 2.2e-6 | 73.3 GiB |
| on | 0 | 3.2e-15 | 2.3e-15 | -4.511811e-5 | 9.0e-16 | 1.8e-6 | 72.7 GiB |

- Sea-level loss: 0 non-finite; its rn_Ud gradient exactly 0 at one step,
  as in rounds 323-324. Traced vs production step: 0 non-bit leaves;
  rebuilt-at-1e-4 vs traced: 0 non-bit leaves.
- rn_Ud: -4.5118e-5 vs round 324's -4.5136e-5 (-0.04 %).
- Round-316 guards: the model-file change is the host-side construction
  guard only; no traced function changed, so every guard is unchanged.

## Predictions (`PREREG_nemo_testcases_l4_orca2_round325.md`)

- P0 flag-off identity: CONFIRMED (11/11, 72/72).
- P1 L3 arms complete, finite: CONFIRMED.
- P2 D122 PASS, both drag arms, five non-rung-10 pairs unmoved: CONFIRMED.
- P3 first over-bar rows equal: CONFIRMED (all four rung-10 pairs).
- P4 drag-on stage-3 RMS T falls: CONFIRMED (-51 %, both labels).
- P5 max sea level moves < 10 %: CONFIRMED (<= 0.5 % everywhere).
- P6 gradients: CONFIRMED (table above; rn_Ud within 1 %).

## Round-314 audit rows

- ln_ldfeiv: DIFFERS -> MATCH. nn_aei_ijk_t, rn_Ue, rn_Le: NOT-MODELLED ->
  MATCH (rn_Ue and rn_Le compared through aei0 = 1/2 rn_Ue rn_Le).
  ln_eke_equ: MATCH, unchanged.
- Counts vs record 176/4 -> 180/3 MATCH/DIFFERS; vs EXP00 175/5 -> 179/4.
  The run-length row keeps its committed DECISION 142 annotation.

## Plants

- Selection reverted (recipe line removed): round-325 selection test
  3 failed (`gates/plant_selection_reverted.log`); restored, `git status`
  clean.
- The L9 guard refuses L3 without the L9 selector, L9 with L2's block, and
  L9 with a changed Treguier evaluation (in the same test).

## Choices made

- L3 on rung 10 on the L9 route — ASKED (DECISION 151, round prompt).
- Both drag arms measured — ASKED (round prompt).
- Rung-0 builder states the L9 selector off (pre-round value; needed for
  rung 0 / OMT to construct) — UNASKED, no behaviour change; offered for
  revert.
- Drag-on censuses on relabelled control copies — UNASKED, instrument
  choice (round-323 precedent); offered for revert.
- Gradient check run in both drag arms (prompt named one check) — UNASKED,
  extra measurement only.
- Older selection tests (rounds 320, 321, 323) keep their earlier card via
  `without_l3`; round 323's refusal text follows the imported guard — tests,
  not configuration.

## OPEN

- The rung-10 card still lists `staged_gm_eiv` among its declared
  unmeasured features (and four gates pin that list); now stale. Not
  changed here.
- Full-domain kt-10 max T rises 0.002 K with L3 (drag on); not located.

## Gates (x64, CPU; logs `orca2_rounds/round325/gates/`)

- Selection battery (rounds 300-325, gradient files apart): rounds 300-312
  files all passed, 0 failed (`sel_battery.log`; the 10-minute cap cut it
  inside round 313's salt-matrix file); rounds 313 (both), 314, 318: 34
  passed, 1 failed = the round-314 audit pin's expected counts (the intended
  change; pin updated, 9 passed, `audit_test.log`); rounds 320, 323, 325: 9
  passed; rounds 321, 324: 12 passed.
- Gradient tests on the shipped card (L3 on): round 319 13 passed
  (`long_grad319.log`); round 322 12 passed, including its rn_Ud gate
  (within 1 % of -4.5e-5; `long_grad322.log`). Round 323 step paths: 3
  passed; its kt 2-3 scan gradient: see `long_steppaths.log`.
- Default gate (land.sh list), split for the 10-minute cap: TKE terms +
  recipe 96 passed (`default_gate_1.log`, whose 3 citation failures were the
  shifted anchors), citation-gate test 17 + private-import ratchet 6 passed
  after the re-anchor (`default_gate_1b.log`), freshwater + stage face mask
  + ORCA2 parallel receipt 20 passed (`default_gate_2.log`), prognostic
  barotropic 8 passed (`default_gate_3.log`) = 147 passed, 0 failed.
- Citation re-anchor (round 321's tool vs `1da92cbf3`): 62 map lines and
  17 lines of the GYRE round-8 receipt, rigid shifts.
- Receipt citation gate on this receipt: see `citation_gate_receipt.log`.

codex review: operator
