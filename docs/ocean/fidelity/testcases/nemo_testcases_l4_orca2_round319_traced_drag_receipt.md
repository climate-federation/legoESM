# ORCA2 round 319 — TKE bottom-BC drag traceable, no inert trainables on rung 10

Base `dd74a18bf08b` (round 318). Prereg `e64b6e382a02` (frozen before any
measurement). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round319/`.
CPU, fp64, x64, production JIT, libm policy.

## Verdict
- Drag is now a traced input on rung 10: the five `float()` casts in the
  TKE bottom boundary condition are gone (`7f4c7f360050`). Forward bits
  identical on all 13 cards (P1 confirmed).
- Rung 10's trainable scalars: cd0, uc0, isoneutral diffusivity, lateral
  viscosity scale, TKE minimum and the background diffusivity floor all
  move the state; cd_max, z0 and ke0 move nothing (the card's linear drag
  never reads them) and are declared inert in the test.
- Rung-10 isoneutral gradient through the copy the step reads: finite at
  N = 1, 2, 5, 10; reverse = forward mode to <= 1.5e-11; no
  flip-free FD epsilon in the sweep.
- FINDING: the background diffusivity floor kappaH_min sits exactly on a
  kink at its rung-10 value. Raising it moves the state linearly; lowering
  it moves nothing; its derivative under jit is exactly 0 (round 317's
  "AD exactly 0" was this). A gradient trainer cannot move it from 1e-10.

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

## 2. No inert trainables on rung 10
Instrument `nemo_testcase_l4_orca2_round319_traced_scalars.py --scan`: one
jitted production step body with every scalar a traced scale (= 1);
each scalar alone x (1 + 1e-3); values that changed, all state leaves.
Control: traced step at s = 1 vs production `model.step`, kt 1: 0 values
differ. Card: drag scheme `nemo_linear` (rate = cd0 * uc0).

| scalar (rung-10 value) | values moved, kt 1 | kt 2 | verdict |
|---|---|---|---|
| cd0 (1e-3) | 2,300,277 | 3,522,729 | moves |
| uc0 (0.4) | 2,300,277 | 3,522,729 | moves |
| cd_max (0.1) | 0 | 0 | INERT: not read by nemo_linear |
| z0 (3e-3) | 0 | 0 | INERT: not read by nemo_linear |
| ke0 (2.5e-3) | 0 | 0 | INERT: not read by nemo_linear |
| isoneutral diffusivity (1000, top-level copy) | 861,058 | 3,423,609 | moves |
| lateral viscosity scale (ahm max 4e4) | 0 | 2,305,403 | moves from kt 2 |
| background diffusivity floor kappaH_min (1e-10) | 60,397 | 799,124 | moves |
| TKE minimum tke_background (1e-10) | 2,143,707 | 3,616,455 | moves |

- P3 falsified twice: kappaH_min moves (predicted inert), and the
  viscosity scale moves nothing at kt 1.
- Tests: one per scalar on rung 10 (two steps); cd_max, z0, ke0 are in a
  declared inert list that fails if they ever start moving. 13 passed
  (600 s). Plants: viscosity scale ignored by the instrument, and
  kappaH_min added to the inert list — each fails its test (2 failed,
  `jobs/plant_inert.log`); plant tree restored clean.
- A trainer on rung 10 must freeze cd_max, z0 and ke0 out.

Linearity check (`--linearity`, one step, max |change| over all T cells,
`jobs/lin_rung10.json`): state at s = 1 + 1e-3 and 1 + 1e-2 vs s = 1, and
the forward-mode tangent times 1e-3.
| scalar | T change, +1e-3 | T change, +1e-2 | tangent x 1e-3 |
|---|---|---|---|
| isoneutral (control) | 7.372e-5 K | 7.355e-4 K | 7.373e-5 K |
| TKE minimum (control) | 5.799e-10 K | 5.788e-9 K | 5.800e-10 K |
| kappaH_min | 4.664e-11 K | 4.666e-10 K | 0 |

| kappaH_min, lowered (`jobs/lin_rung10_neg.json`) | 0 at -1e-3 | 0 at -1e-2 | 0 |

- kappaH_min: raised, the response is linear (x10 for x10) and the
  diffusivity moves by exactly the floor's change (1.000e-13 at +1e-3)
  where it binds; lowered, no state value moves. One-sided: CONFIRMED.
  The two controls match their tangent to <= 0.02 %.
- Not the cold-start seed: rung 10 enters with a carried diffusivity
  (minimum 1.2e-6, no cell at 1e-10).
- PLAUSIBLE mechanism: in a quiescent cell at the TKE minimum and the
  minimum mixing length, the closure's raw diffusivity is
  0.1 x 1e-3 m x sqrt(1e-10) / Prandtl 10 = 1e-10, the floor itself, so
  the floor ties the other branch. The closure alone (toy column, rung-10
  TKE settings): strictly binding, the floor's tangent is 1e-10 eager and
  jit; at the tie it is 0.5 x 1e-10 eager and 0 under jit. Rung-10 tie
  cells not counted.

## 3. Round 317's detached results
| run | status | finite | reverse vs forward (worst) | peak |
|---|---|---|---|---|
| OMT-3 probe | complete, 8,453 s | yes, N = 1, 2, 5, 10 (56 rows) | 1.2e-13; 3 of 4 parity controls (drag: as rung 0) | 117 GiB |
| OMT-4 probe | stopped at its 160 GiB cap, 7,451 s | yes, 32 rows (T directions and isoneutral, N = 1..10) | 8.5e-14 | 161 GiB |

OMT-4's viscosity, floor and drag rows were not reached.

No-remat cost rows (fused fwd+bwd, T loss; remat values in round 317's table):
| card | N | fwd+bwd | run memory growth | process peak (GiB) |
|---|---|---|---|---|
| rung 0 | 2 | 20.9 s | 16.8 GB | 73 |
| rung 0 | 5, 10 | stopped at the 150 GiB cap | — | 152, 189 |
| rung 10 | 2 | 64.4 s | 21.9 GB | 85 |
| rung 10 | 5, 10 | stopped at the 150 GiB cap | — | 179, 224 |

- Remat and no-remat gradients agree at N = 2: 6.2e-16 (rung 0), 3.5e-13
  (rung 10), relative to max |g|.
- Without remat, N >= 5 does not fit under 150 GiB on either card.

## 4. Rung-10 isoneutral gradient at N = 2, 5, 10
Round 315's probe, `--rung rung10 --directions 0 --scalars kappa_redi
--remat`, at the prereg commit (top-level isoneutral copy, round 317's
wiring), `jobs/redi_rung10_n10.json`: 8,329 s, peak 238 GiB (a first run
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
- Flip-free epsilon: none in the sweep (1e-2 .. 1e-7). The FD miss does
  not fall as epsilon shrinks (T, N = 10: 1.4e-5 at 1e-5, 1.2e-4 at 1e-6,
  2.0e-3 at 1e-7), the switch pattern round 316 measured on OMT-2. Branch
  flips not counted here.
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

## OPEN
- The background floor's kink on rung 10: count the tie cells; whether a
  trainer exposes this floor at all is a user decision.
- Branch-flip count for the rung-10 isoneutral FD (no flip-free epsilon).
- OMT-4's viscosity, floor and drag probe rows (cap stop).
- The C-D grid and MPAS drag paths keep their casts.

## Choices
- ASKED (operator): the cast removal.
- UNASKED: none. The inert list in the test records a measurement; it
  changes no configuration.

codex review: operator
