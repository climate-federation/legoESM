# ORCA2 round 319 — TKE bottom-BC drag traceable; rung 10 has three inert scalars and one one-sided floor

Base `dd74a18bf08b` (round 318). Prereg `e64b6e382a02` (frozen before any
measurement). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round319/`.
CPU, fp64, x64, production JIT, libm policy.

## Verdict
- Drag is now a traced input on rung 10: the five `float()` casts in the
  TKE bottom boundary condition are gone (`7f4c7f360050`). Forward bits
  identical on all 13 cards (P1 confirmed).
- Rung 10's trainable scalars: cd0, uc0, isoneutral diffusivity, lateral
  viscosity scale and TKE minimum move the state both ways; cd_max, z0 and
  ke0 move nothing (the card's linear drag never reads them) and are
  declared inert; the background diffusivity floor kappaH_min is declared
  one-sided (gradient exactly 0), pending the user's decision.
- Rung-10 isoneutral gradient through the copy the step reads: finite at
  N = 1, 2, 5, 10; reverse = forward mode to <= 1.5e-11; no
  flip-free FD epsilon in the sweep.
- FINDING: the background diffusivity floor kappaH_min sits one ulp below
  the TKE branch at its rung-10 value. Raising it moves the state; lowering
  it moves nothing; its derivative is exactly 0 (round 317's "AD exactly
  0" was this). A gradient trainer cannot move it from 1e-10.
- At those cells rung 10 computes NEMO's own arithmetic (16,444 of 16,444
  floor cells equal); the eager form of the same function would not.

## 1. The fix (`7f4c7f360050`)
`ocean_model_latlon_cgrid.py:11751-11755`: `cd0`, `cd_max`, `z0`, `ke0`,
`uc0` pass the config values straight through (round 316 did the same in
the momentum drag). The config fields are Python floats, so `float(x)` was
the identity on every card.

Forward bits (round 316's instrument, 10 steps, every state leaf's raw
bytes at every kt, base `e64b6e382a02` vs fix `7f4c7f360050`):
- rung 0, OMT-1..4, rung 10, GYRE, VORTEX zco flux/vector, VORTEX seamount
  flux/vector, OVERFLOW, LOCK_EXCHANGE: BIT-IDENTICAL, 13 of 13, 0
  leaf-steps moved; the step jaxpr is identical on all 13
  (`forward_bits_summary.txt`).
- The seven ORCA2 pairs replay these six cards' step: same program, so
  same rows; the ladders were not re-run.
- GYRE ten-step check at the fix: archive SHA-256
  `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`
  (= round 316's), status DEBT (unchanged).
- DINO month gate: runs at landing (operator).

Tests (jit, x64):
- `test_tke_nemo_terms.py::...::test_drag_coefficient_is_a_traced_input`:
  d(sum of bottom TKE)/d(cd0) = sum of unfloored values / cd0 to 1e-12.
  With the casts restored: ConcretizationTypeError (fails).
- `test_orca2_round319_traced_scalars.py::test_cd0_traces_through_rung10_step_under_jit_grad`:
  `make_jaxpr(jit(grad))` of a rung-10 one-step loss w.r.t. traced cd0
  (226,830 jaxpr lines). With the casts restored it fails with
  ConcretizationTypeError raised at `ocean_model_latlon_cgrid.py:11751`
  (`jobs/plant_casts_back.log`).

Not in scope, named only: the C-D grid and MPAS drag paths still cast the
same five coefficients.

## 2. Rung-10 scalars: three inert, one one-sided
Instrument `nemo_testcase_l4_orca2_round319_traced_scalars.py` (scan mode,
`jobs/scan2_rung10.json`): one jitted production step body with every
scalar a traced scale (= 1); each scalar alone x (1 + 1e-3) and x (1 - 1e-3);
values that changed, all state leaves, kt 1 + kt 2. Control: traced step
at s = 1 vs production `model.step`, kt 1: 0 values differ. Card: drag
scheme `nemo_linear` (rate = cd0 * uc0).

| scalar (rung-10 value) | moved, raised (kt 1 + 2) | moved, lowered | verdict |
|---|---|---|---|
| cd0 (1e-3) | 5,823,006 | 5,822,861 | live |
| uc0 (0.4) | 5,823,006 | 5,822,861 | live |
| cd_max (0.1) | 0 | 0 | INERT: not read by nemo_linear |
| z0 (3e-3) | 0 | 0 | INERT: not read by nemo_linear |
| ke0 (2.5e-3) | 0 | 0 | INERT: not read by nemo_linear |
| isoneutral diffusivity (1000, top-level copy) | 4,284,667 | 4,284,592 | live |
| lateral viscosity scale (ahm max 4e4) | 2,305,403 (kt 2 only) | 2,305,514 (kt 2 only) | live |
| background diffusivity floor kappaH_min (1e-10) | 859,521 | 0 | ONE-SIDED: gradient exactly 0, pending the user's decision |
| TKE minimum tke_background (1e-10) | 5,760,162 | 5,737,905 | live |

- P3 falsified twice: kappaH_min moves when raised (predicted inert), and
  the viscosity scale moves nothing at kt 1.
- The first version of this section claimed no inert trainables; it
  counted movement upward only, which kappaH_min passes with a zero
  gradient. Retracted.
- Tests (rung 10, two steps): a scalar not declared must move the state
  both raised and lowered; cd_max, z0, ke0 are declared inert (must move
  neither way); kappaH_min is declared one-sided (must move raised, not
  lowered). 13 passed (`jobs/test_r319_v2.log` 11, `jobs/test_r319_cd0.log`
  2). Plant: kappaH_min removed from the one-sided declaration — its test
  fails, "moves 859521 values raised, 0 lowered"
  (`jobs/plant_onesided.log`); plant file deleted. Earlier plant (still
  valid): the instrument ignoring the viscosity scale fails its test
  (`jobs/plant_inert.log`).
- A trainer on rung 10 must freeze cd_max, z0 and ke0 out.
- DECISION 143 (2026-10-20, user: "Keep fixed, out of training"): the
  floor kappaH_min is frozen out of every rung-10 trainer at NEMO's value
  1e-10. Not trained, not moved (recorded in round 320).

Linearity check (`--linearity`, one step, max |change| over all T cells,
`jobs/lin2_rung10.json`, `jobs/lin2_rung10_neg.json`): state at s = 1 + r
and 1 + 10 r vs s = 1, and the forward-mode tangent times r. State, dt and
forcing are now arguments of the jitted step, as in production: traced
step at s = 1 vs `model.step`, 0 values differ. (The first form closed over
them; its step differed from production in last bits, e.g. kappaH_min
+1e-3 gave 4.6636e-11 K; the reviewer counted ~97k velocity values.)
| scalar | T change, r = 1e-3 | r = 1e-2 | tangent x 1e-3 |
|---|---|---|---|
| isoneutral (control) | 7.372e-5 K | 7.355e-4 K | 7.373e-5 K |
| TKE minimum (control) | 5.799e-10 K | 5.788e-9 K | 5.800e-10 K |
| kappaH_min | 4.664e-11 K | 4.666e-10 K | 0 |
| kappaH_min, r = -1e-3, -1e-2 | 0 | 0 | 0 |

- kappaH_min lowered: no state value moves, tangent 0 (every leaf); raised:
  response x10 for x10. One-sided: CONFIRMED in production form. The two
  controls match their tangent to <= 0.02 %.
- Not the cold-start seed: rung 10 enters with a carried diffusivity
  (minimum 1.2e-6, no cell at 1e-10).

Mechanism (CONFIRMED; reviewer finding reproduced, `jobs/floor_cells_rung10.json`,
`--floor-cells`: the closure's inputs recorded inside the jitted production
step; control: recorded step = unrecorded step, 0 values differ; the
recorded closure output = stored diffusivity in all 414,119 wet cells).
The round's earlier "the floor ties the other branch" was wrong.
- Statement (the non-literal fallback): `tke.py:2329` floors K_M / Pr, with
  Pr = 1 / pdlr (`tke.py:2174`, `tke.py:2175`). In the 16,444 cells within 0.1 % above the
  floor, Pr = 10 exactly, so pdlr = 0.1 (its floor). With NEMO's forced
  constants the TKE branch's minimum is 0.1 x 1e-3 x sqrt(1e-10) x 0.1 =
  1e-10, the floor, in exact arithmetic.
- In fp64 the jitted step lands ONE ULP ABOVE the floor in 7,634 cells and
  never at or below it: XLA rewrites K_M / (1 / pdlr) as K_M x pdlr, and
  0.1 x 1e-9 = 1e-10 + 1 ulp. The same function evaluated eagerly gives
  exactly 1e-10 there (a tie: gradient 0.5 per cell).
- So raising the floor by 0.1 % moves 16,444 diffusivities (60,397 state
  values); lowering it never binds; forward and reverse derivatives are
  exactly 0.0.

NEMO's statement in those cells (measured on rung 10's own inputs; NEMO is
built -O3 without fast-math, so each product is one correctly rounded fp64
operation):
- NEMO floors before the Prandtl factor: zav at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:708`, floored at
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:710`, then multiplied by pdlr
  and floored again at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:717`;
  pdlr >= 0.1 by `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:405`; the floor
  1e-10 is IWM's `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:442`.
- Result: NEMO's value equals rung 10's jitted value in 16,444 of 16,444
  cells (7,634 at 1e-10 + 1 ulp). The eager form would match only 5,346.
  Rung 10 does NOT depart from NEMO there. NEMO's own value sits one ulp
  above its floor, so lowering NEMO's avtb would not bind there either
  (read off the arithmetic, not run).
- Certified form computing it: the vmix ladder's V4 (V3 + internal-wave
  mixing, avtb = 1e-10; gridladder-vmix round 14) and V5, bit-for-bit with
  NEMO in avt kt 1..10; the vmix lane's closure carries the same two
  statements (read off the code). Floor-cell counts on V4 not measured.
- The match relies on XLA's rewrite of the reciprocal division; no fix this
  round (named only).

## 3. Round 317's detached results
| run | status | finite | reverse vs forward (worst) | process peak, guard RSS (GiB) |
|---|---|---|---|---|
| OMT-3 probe | complete, 8,453 s | yes, N = 1, 2, 5, 10 (56 rows) | 1.2e-13; 3 of 4 parity controls (drag: as rung 0) | 117 GiB |
| OMT-4 probe | stopped at its 160 GiB cap, 7,451 s | yes, 32 rows (T directions and isoneutral, N = 1..10) | 8.5e-14 | 161 GiB |

OMT-4's viscosity, floor and drag rows were not reached.

No-remat cost rows (fused fwd+bwd, T loss; remat values in round 317's table).
Units: "run growth" = the instrument's sampled RSS growth during the run, GB
(1e9 B); "process peak" = the memory guard's whole-process RSS peak, GiB
(2^30 B), compile included.
| card | N | fwd+bwd | run growth (GB) | process peak (GiB) |
|---|---|---|---|---|
| rung 0 | 2 | 20.9 s | 16.8 | 73 |
| rung 0 | 5, 10 | guard-killed at its 150 GiB cap | — | 152, 189 |
| rung 10 | 2 | 64.4 s | 21.9 | 85 |
| rung 10 | 5, 10 | guard-killed at its 150 GiB cap | — | 179, 224 |

- Remat and no-remat gradients agree at N = 2: 6.2e-16 (rung 0), 3.5e-13
  (rung 10), relative to max |g|.
- Without remat, N >= 5 does not fit under 150 GiB on either card. Whether
  the kills came during compile or during the run is not recorded; the
  cost mode now writes XLA's memory estimate to its JSON before executing
  (a row marked "compiled" without run fields = a run-time stop; no row =
  a compile-time stop).
- For comparison, all fused remat N = 10, process peak: 76 GiB (rung 0) and
  89 GiB (rung 10) — round 317's 82 / 96 GB in decimal units.
- Section 4's 238 GiB (guard process peak) is a different instrument: the
  chained per-step probe. Cause PLAUSIBLE (read off the probe's structure,
  not measured): it holds every step's pullback during its reverse phase.

## 4. Rung-10 isoneutral gradient at N = 2, 5, 10
Round 315's probe, `--rung rung10 --directions 0 --scalars kappa_redi
--remat`, at the prereg commit (top-level isoneutral copy, round 317's
wiring), `jobs/redi_rung10_n10.json`: 8,329 s, guard process peak 238 GiB
(chained per-step probe, reverse phase; a first run
stopped at its 220 GiB cap, `jobs/capstop220/`). Forward parity
bit-identical; construction parity (model rebuilt at s = 1 + 1e-5)
bit-identical to the traced path.

| loss | N | reverse | reverse vs forward | FD best (eps) |
|---|---|---|---|---|
| T | 1 | -3.2176504e-5 | 2.7e-15 | 7.2e-8 (1e-2) |
| T | 2 | -6.3816880e-5 | 2.5e-12 | 1.9e-6 (1e-4) |
| T | 5 | -1.5901266e-4 | 1.1e-12 | 1.4e-5 (1e-5) |
| T | 10 | -3.1101906e-4 | 1.2e-11 | 1.4e-5 (1e-5) |
| SSH | 1 | 0 | 0 | 0 (FD = 0) |
| SSH | 2 | 3.6290348e-8 | 4.8e-13 | 1.6e-6 (1e-2) |
| SSH | 5 | 4.6135930e-7 | 5.4e-12 | 2.2e-4 (1e-5) |
| SSH | 10 | -6.3076545e-6 | 1.5e-11 | 1.9e-5 (1e-5) |

- P4: finite, reverse = forward <= 1e-10, SSH nonzero from N = 2:
  CONFIRMED. FD <= 1e-6 at N = 2: missed (1.9e-6 T, 1.6e-6 SSH).
- Flip-free epsilon: none in the sweep (1e-2 .. 1e-7). The FD miss grows
  as epsilon shrinks (T, N = 10: 1.4e-5 at 1e-5, 1.2e-4 at 1e-6, 2.0e-3 at
  1e-7), roughly as 1/eps. PLAUSIBLE: branch switches (the pattern round
  316 measured on OMT-2); plain rounding noise in the loss fits the same
  1/eps growth. Branch flips not counted here.
- Same run, prereg commit (casts present): the traceability check reports
  cd0 NOT traceable on rung 10 — the defect this round's fix removes.
- Same run: the background floor's reverse gradient is exactly 0 at every
  N and loss (section 2's one-sided floor).

## 5. DECISION 142 (docs/comments only)
- Card run-length declaration (`nemo_testcase_recipe.py:1985`): one-line
  comment with the decision.
- Round-314 audit table, nn_itend row: both status cells now read
  "DIFFERS — deliberate (DECISION 142)"; the audit script's reason string
  says so too (regeneration keeps the reason). No value changed.

## Gates (x64, CPU, `jobs/gate_*`)
Default gate files at `3c9f2c1991d1` (one process each): citation-gate
test 17 passed after the re-anchor below, private-import ratchet 6, TKE
terms 71, recipe 25, freshwater 16, WS stage face-mask 2, barotropic state
8, ORCA2 parallel receipt 2 — 0 failed. Also: round-314 audit 9,
round-315 probe 11, round-319 scalar tests 13.
- Re-anchor (`ccec4003606d`): the round-317 citation of the TKE bottom-BC
  drag lines anchored on the cast text; it now anchors on the arguments.
- Citation gate on this receipt: PASS, failures 0 (`gates/citation_gate_receipt.json`).

## Dual review and this resume
- codex: LAND. Independent Claude review: HOLD, five findings; each
  reproduced before fixing, and none changes a forward value.
  1. Test called a zero-gradient scalar live (upward-only count): fixed,
     section 2 (two-sided test, kappaH_min declared, plant fails).
  2. Floor mechanism was one ulp, not a tie: reproduced and measured
     against NEMO's statement, section 2.
  3. `--linearity` was not the production program: now bit-identical;
     "lowered: nothing moves" holds in production form.
  4. Memory numbers: labelled by instrument and unit, section 3.
  5. FD 1/eps growth: "switches" now PLAUSIBLE, section 4.
- Gates re-run after the resume at `d9067e78d` (x64, CPU,
  `jobs/gates2/`, one process each): private-import ratchet 6, TKE terms
  71, recipe 25, freshwater 16, WS stage face-mask 2, barotropic state 8,
  ORCA2 parallel receipt 2, round-316 instruments 9 — 0 failed. At
  `4910fdccd` (clean tree): citation-gate test 17 passed; this receipt's
  gate PASS, 11 citations, failures 0 (`gates/citation_gate_receipt_resume.json`).

## OPEN
- The background floor kappaH_min is one-sided on rung 10 (and in NEMO's
  own arithmetic): closed by DECISION 143 (frozen out of training at 1e-10).
- The floor-cell match with NEMO relies on XLA rewriting the reciprocal
  division; the eager form departs by one ulp in 7,634 cells.
- Floor-cell counts on V4 itself (not measured).
- Branch-flip count for the rung-10 isoneutral FD (no flip-free epsilon).
- OMT-4's viscosity, floor and drag probe rows (cap stop).
- The C-D grid and MPAS drag paths keep their casts.

## Choices
- ASKED (operator): the cast removal.
- UNASKED: none. The inert and one-sided lists in the test record
  measurements; they change no configuration, and kappaH_min's status is
  left to the user.

codex review: operator
