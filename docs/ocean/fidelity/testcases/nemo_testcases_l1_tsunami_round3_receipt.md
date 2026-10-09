# RECEIPT — TSUNAMI lane, round 3 (measure the RK3 record)

Date 2026-10-08. Lane tip at start `6b4b3ea33b83`. Preregistered in
`PREREG_nemo_testcases_l1_tsunami_round3.md` (commit `7405a37c0`), after one
disclosed exploratory kt = 1 probe (prereg section 0).

Status: **STOPPED_FOR_DECISION.** The first non-bit statement is closed by a
one-field card selection that already exists and is certified elsewhere; it
changes which scheme the card selects, so it is asked, not made.

Take-aways:

1. **Record admitted, geometry identical.** 110 records re-admitted; 35 mesh
   rows equal NEMO's `mesh_mask.nc` bit for bit; a planted element turns the gate red.
2. **The first non-bit statement is the stp_2D pressure gradient.** NEMO's
   kt = 1 right-hand side is `eos` + `hpg_sco` and nothing else (an offline
   replay reproduces it bitwise). The card differs from it by 7.7e-11 of 8.7e-8
   (346 cells), most likely because it evaluates density at the static depth; selecting the
   existing `geometric` depth arm, as a measurement arm only, takes that to
   1.1e-19 and the whole kt = 1 step to 7.6e-17 on ssh.
3. **From kt = 2 the whole-step error is the card's momentum advection (B6).**
   Scaling NEMO's entry velocity by 0.25, 0.5, 1 scales the right-hand-side
   error by exactly 4.000000 per doubling, with a 1.1e-19 control at zero velocity.
4. **TSUNAMI's one wet level makes every stage's u, v the external uu_b, vv_b**
   (the barotropic correction), so the stage momentum internals cannot be
   tested here (section 5).

## 1. Admission and geometry (task 1, 2)

| gate | result | evidence (sha256 prefix) |
|---|---|---|
| `check_records.py --program rk3` | `ADMITTED 110 records`, json byte-identical to the operator's | `tsunami_round3_admission.json` `a0f77990211a2e59` |
| groups present | kt 1..10: 60 step groups (`e_ b_ 1_ 2_ 3_ f_`) and an 8-substep record; kt 11..100: the six `f_` groups; output identity `sossheig souubaro sovvbaro` | same |
| geometry identity | 35 rows EXACT, 0 differing: glam/gphi t,u,v,f; e1/e2 t,u,v,f (operand and grid); ff_t, ff_f (source, grid, operand); tmask, umask, vmask, fmask; e3t, e3w, gdept, gdepw at the executed level | `tsunami_round3_geometry.json` `01a33c1f1fe3b7fd` |
| geometry plant | `--plant "grid dx_u"` moves one element: `GEOMETRY DIFFERS`, exit 1; perturbing `e2f`, `glamf`, `ff_t`, `tmask`, `gdept_1d` in a copy of the mesh each turns exactly that row red (test) | `tests/ocean/fidelity/test_nemo_tsunami_geometry_gate.py` |

The dummy bottom record (jpk = 2) is reported in the json, not scored: the card does not carry it.
Evidence directory: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round3/`.

## 2. The ladder, kt = 1..10 (task 3)

One shared harness (`nemo_testcase_l1_tsunami_ladder.py`): the VORTEX card
model and private hooks, the trajectory gate's row scorer (bar 1e-15
normalised), the VORTEX walk's substep boundary order. Searched before
writing: `scripts/validate/ocean_fidelity` for a TSUNAMI or step-record
reader (only the admission parser exists, reused) and for ladder tools (the
VORTEX kt2 walk and spgts walk, reused for hooks and boundary order; their
`run` entry points select a VORTEX card and record, so only the TSUNAMI
readers and arms are new).

**INDEPENDENT** (card's step chained from the card's initial state; max abs
error vs NEMO's after-step state; ssh is unequal at every kt):

| kt | ssh | uu_b = vv_b = u = v | T | S |
|---|---|---|---|---|
| 1 | 1.17e-07 | 6.00e-08 | 4.59e-03 | 6.89e-03 |
| 2 | 2.04e-04 | 4.76e-05 | 1.68e-02 | 2.52e-02 |
| 3 | 4.55e-04 | 1.23e-04 | 2.94e-02 | 4.41e-02 |
| 5 | 1.24e-03 | 1.12e-04 | 2.28e-02 | 3.42e-02 |
| 10 | 1.05e-03 | 1.20e-04 | n/a (no kt = 11 entry record) | n/a |

First over the bar: kt = 1, ssh (the first row in field order; T and S are over it too).

**GIVEN-NEMO-ENTRY, whole step** (card handed NEMO's entry state of step kt;
not comparable with the table above). Card as it ships, and the
measurement arm `eos_depth = geometric` (card unchanged, one field):

| kt | ssh as shipped | ssh geometric | u as shipped | u geometric |
|---|---|---|---|---|
| 1 | 1.17e-07 | 7.63e-17 | 6.00e-08 | 5.20e-18 |
| 2 | 1.33e-05 | 1.34e-05 | 3.20e-06 | 3.23e-06 |
| 3 | 3.00e-05 | 3.00e-05 | 4.21e-06 | 4.20e-06 |
| 5 | 2.69e-06 | 2.70e-06 | 1.13e-06 | 1.11e-06 |
| 10 | 7.71e-07 | 7.56e-07 | 3.53e-07 | 3.43e-07 |

**GIVEN-NEMO-ENTRY, in NEMO's order** (stp_2D right-hand side, then
dyn_spg_ts, then the handoff, then the stages). Max abs, u (v identical):

| kt | stp_2D rhs as shipped | rhs geometric | handoff uu_b as shipped | handoff uu_b geometric |
|---|---|---|---|---|
| 1 | 7.70e-11 | 1.09e-19 | 6.00e-08 | 5.20e-18 |
| 2 | 4.93e-09 | 4.97e-09 | 3.20e-06 | 3.23e-06 |
| 3 | 8.18e-09 | 8.19e-09 | 4.21e-06 | 4.20e-06 |
| 10 | 4.78e-10 | 4.67e-10 | 3.53e-07 | 3.43e-07 |

**First non-bit statement per kt** (record resolution: the right-hand side is
the first thing NEMO records inside the step; the substep rows come after it):

| kt | first non-bit | statement | reading |
|---|---|---|---|
| 1 | stp_2D right-hand side, 346 cells, 7.70e-11 | `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stp2d.f90:137-138` (eos + dyn_hpg) | density depth, section 3 |
| 2..10 | stp_2D right-hand side, 4.7e-10 .. 8.2e-09 | same two calls | momentum advection in the card, section 4 |

**dyn_spg_ts substeps** (8 substeps per kt, 179 boundaries each). "own" = the
card's own entry forcing; "NEMO" = NEMO's recorded entry forcing substituted
(`zu_frc`, `zv_frc`):

| kt | own: first unequal | own: rows over bar / 179 | NEMO forcing: first unequal | NEMO forcing: rows over bar, max normalised |
|---|---|---|---|---|
| 1 | entry forcing `zu_frc`, 7.7e-11 | 154 | substep 1 velocity update, 4.3e-19 | 0, 9.8e-16 |
| 2 | entry forcing, 4.9e-09 | 154 | substep 1 velocity update, 3.5e-18 | 6, 1.4e-15 |
| 5 | entry forcing, 1.5e-09 | 154 | substep 1 velocity update, 2.6e-18 | 9, 1.7e-15 |
| 9 | entry forcing, 5.7e-10 | 154 | substep 1 velocity update, 1.7e-18 | 9, 1.6e-15 |
| 10 | entry forcing, 4.8e-10 | 170 | substep 1 entry ssh, 2.6e-318 (denormal) | 8, 1.4e-15 |

With NEMO's forcing the substep loop is NOT bit-identical (126..149 of 179
boundaries unequal at every kt) and sits at 1.0e-15 .. 1.7e-15 normalised,
within a factor 1.7 of the bar: a few units in the last place. (kt = 10's
first unequal row is a far-field denormal value, 44 cells at 1e-318.) Under
`geometric`, the "own" arm at kt = 1 drops from 154 rows over the bar (5.1e-06)
to 14 (7.3e-15).

**Stage-local** (NEMO's external handoff and NEMO's stage entry; 150 rows,
19 bit-identical). Range over kt = 1..10, max abs:

| stage | ssh | u, v | T | S |
|---|---|---|---|---|
| 1 | 1.7e-18 .. 1.4e-17 | 0 .. 1.7e-18 | 6.0e-04 .. 4.2e-03 | 9.0e-04 .. 6.3e-03 |
| 2 | 0 .. 9e-309 | 8.7e-19 .. 1.7e-18 | 9.0e-04 .. 6.3e-03 | 1.3e-03 .. 9.5e-03 |
| 3 | 0 | 8.7e-19 .. 3.5e-18 | 1.8e-03 .. 1.3e-02 | 2.7e-03 .. 1.9e-02 |

Same to rounding under `geometric` (the pressure gradient does not reach the stage
output, section 5). Instrument note: the per-cell first-unequal location of
every row is in the json; the stage u, v rows are the correction arithmetic only.

## 3. The first non-bit statement (CONFIRMED / PLAUSIBLE)

- **CONFIRMED** NEMO's kt = 1 right-hand side is the pressure-gradient trend
  alone. Instrument: `nemo_testcase_l1_tsunami_hpg_replay.py`, numpy float64
  in compiled operand order, constants parsed from the reference run's
  `ocean.output` (not from the card), fed only NEMO's recorded entry T, S,
  ssh, r3t. It reproduces `b_uu_rhs_k1` and `b_vv_rhs_k1` with 0 unequal
  cells (`hpg_replay_kt1.json` `8eea642ff392f4ec`). Plants: moving `mu2`,
  `rho0` or `a0` by 1e-9 relative turns it red (1.7e-19, 8.7e-17).
- **CONFIRMED** the card's value differs from it by 7.70e-11 at 346 cells; the
  largest card term at kt = 1 is that pressure gradient (velocity is zero).
- **CONFIRMED** (one variable) setting `eos_depth = geometric` for the run,
  nothing else, takes the right-hand side to 1.09e-19, the handoff uu_b from
  6.00e-08 to 5.20e-18 and the whole kt = 1 step from 1.17e-07 to 7.63e-17 on ssh.
- **PLAUSIBLE** mechanism: NEMO's density uses the live stretched depth,
  `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/eosbn2.f90:361`, and again in the
  correction term built from the stretched depth less ssh, `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynhpg.f90:387`;
  the card's `insitu` arm evaluates density at the static depth. An exploratory replay (NOT committed) with the
  static depth landed 8.6e-13 from the card, not at the 1e-19 floor, so the
  static depth is the dominant piece, not shown to be the whole of it.
- The `geometric` arm already exists and is certified for `nemo_seos`
  (guard comment in `ocean_model_latlon_cgrid.py` at the `_geometric_certified_eos` set; DINO
  certificate). The TSUNAMI card selects `insitu`.

## 4. From kt = 2: momentum advection (B6)

Given NEMO's entry state at kt = 2, 5 and 10 with `geometric`, the stp_2D
right-hand side minus NEMO's prediction `hpg_replay + lambda * (NEMO rhs - hpg_replay)`
(entry velocity scaled by lambda, T, S, ssh unscaled):

| kt | lambda 0 (control) | 0.25 | 0.5 | 1.0 | E(1)/E(0.5) | E(0.5)/E(0.25) |
|---|---|---|---|---|---|---|
| 2 | 1.09e-19 | 3.11e-10 | 1.24e-09 | 4.97e-09 | 4.000000 | 4.000000 |
| 5 | 1.09e-19 | 9.40e-11 | 3.76e-10 | 1.50e-09 | 4.000000 | 4.000000 |
| 10 | 1.09e-19 | 2.92e-11 | 1.17e-10 | 4.67e-10 | 4.000000 | 4.000000 |

**CONFIRMED** at kt = 2, 5, 10, in the maximum norm (one scalar per lambda; cell
maps are not stored): the residual scales quadratically in velocity to six
digits and no linear part is visible above the 1.09e-19 floor. Other kt are not
tested. PLAUSIBLE: the quadratic term is the card's UP3 flux-form advection, which
NEMO does not run (the card's term decomposition lumps advection with Coriolis,
so it was not isolated). NEMO's OFF branch:
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynadv.f90:185` and the depth-average case
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stp2d.f90:176`. The floor is identical at
kt = 1, 2, 5, 10, so it is a fixed-location rounding floor, not a state-dependent term.
Instrument defect found and fixed on the way: the right-hand-side observer is
an asynchronous callback; read without an effects barrier it returned stale
values (ratios 1.5 and 2.0 in the first run). The committed arm waits, and a
test pins the 4.0.

## 5. One wet level (CONFIRMED)

In a one-level column every stage ends with the barotropic correction,
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:413-414`, which sets u, v to the
external uu_b, vv_b. Measured by an exploratory probe that is NOT committed (stage 2
and stage 3, NEMO's external handoff handed in, one cell bumped by 1.0 in u, in v,
in T and in entry ssh, each separately): u, v and ssh are bit-unchanged; only T, S
respond, to the velocity. The committed plant (a test) is one: u by 1e-3 at stage 2,
read on T. So TSUNAMI proves the
external mode, the hybrid update arithmetic, the correction and the tracer
thickness-ratio step; it cannot test the stage 3-D momentum step (flux-form
(1 + r3) update, dyn_zdf, stage-2/3 pressure gradient). The ladder's
stage-entry plant is therefore read on T, not on u.

## 6. When the card "refuses" (task 4)

The card does NOT refuse at a statement: its step runs kt = 1..10 with the
carrier arms. What refuses is `validate_nemo_testcase_card_for_execution`,
which lists the four declared blockers; the ladder above bypasses it and
labels every row. How far each blocker lets the ladder go, and the smallest
card-selected arm that closes it (for round 4):

| blocker | binds | measured | smallest arm | NEMO's OFF branch |
|---|---|---|---|---|
| B6 momentum advection | kt = 2, every kt after | right-hand side error quadratic in velocity, 4.97e-09 at kt = 2; whole-step ssh 1.3e-05 | a "no momentum advection" selection (NEMO `np_LIN_dyn`) for stp_2D and the stages | `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynadv.f90:185` |
| B7 tracer advection | kt = 1 on T, S (4.59e-03, 6.89e-03) | the card's T, S stay uniform (20, 30 to 1e-14, read at kt = 1) where NEMO's move by up to 4.6e-03, 6.9e-03 (thickness ratio). Independent kt = 2 ssh 2.04e-04 against given-entry 1.33e-05 | "no tracer advection, thickness-ratio step only" (NEMO `np_NO_adv`), `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:503-505` | `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/traadv.f90:444` |
| B4j j-seam wall | not within kt <= 10 (PLAUSIBLE: no card A/B with a corrected j path exists) | NEMO's ssh support (`ladder_front.json`, abs ssh > 1e-12) is 62, 55, 49, 45, 40, 36, 32, 28, 24, 20 cells from the j-seam at kt = 1..10, about 4 per step (reaches it near kt 15, extrapolation) | y-wrap in the barotropic path | n/a |
| B4 i-seam | crossed from kt = 5 (entry velocity at the seam columns 3.8e-18 at kt 5, 3.1e-12 at 6, 7.4e-4 at 10) | no step in the given-entry error across kt 5..6 (geometric ssh 2.70e-06 then 1.70e-06); B6 at 1e-6 would hide anything smaller (PLAUSIBLE) | none until B6 is closed | n/a |

Order of binding in the ladder: depth in the pressure gradient (kt 1), B7 (kt 1, tracers), B6 (kt 2).

## 7. Predictions scored

| id | prediction | result |
|---|---|---|
| P1 | geometry identity, plant red | **CONFIRMED** 35/35 rows, plant red |
| P2 | admission, 110 records, byte-identical json | **CONFIRMED** |
| P3 | independent never bit-identical; T, S over bar from kt = 1 | **CONFIRMED** |
| P4 | substep loop bit-identical given NEMO's entry forcing | **FALSIFIED** as worded: 126..149 of 179 boundaries unequal; ownership by the forcing holds (own arm 154 rows over the bar, NEMO-forcing arm 0..11, max 1.7e-15) |
| P5 | stages within bar on ssh, velocity; T, S over by >= 1e-4 | **CONFIRMED** (ssh <= 1.4e-17, u, v <= 3.5e-18, T >= 6.0e-4); u, v are the correction of one column (section 5) |
| P6 | offline replay reproduces NEMO's rhs bitwise | **CONFIRMED**, plants red |
| P7 | B7 binds first; B4j not before kt 8 | B7 **CONFIRMED**; B4j not within kt 10 **PLAUSIBLE** (support argument only); B6 prediction "inert" **FALSIFIED**: it binds at kt = 2 |

## 8. Review

Single review (codex), verdict **DO NOT SHIP** on `6b4b3ea33..a5ad18686`
(`codex_review.txt`). Disposition:

- Geometry blind spots: CONFIRMED in part, fixed. The vertical rows compared
  column (0,0) only; they now compare every column. The `glam/gphi` rows are
  the helper arrays the card's grid is built from (the card holds no second
  copy): stated, not closed.
- Unasked choices: the EOS arm is a flag on a measurement run, the card is
  unchanged, every dependent number is labelled; REJECTED as a defect and
  offered for revert. The seam padding was measured: inert through kt 8,
  consequential from kt 9 (section 9).
- B6 overclaim: CONFIRMED, fixed by weakening (maximum norm; kt 2, 5, 10; the
  UP3 attribution PLAUSIBLE).
- B4j "CONFIRMED": CONFIRMED, relabelled PLAUSIBLE.
- Stage plants not run: CONFIRMED; the 1.0 bumps were an uncommitted probe and
  are labelled so.
- Uncitable numbers: the front-support numbers are now an arm
  (`ladder_front.json`); the 8.6e-13 static-depth replay is labelled exploratory.
- Citations through the generic source: CONFIRMED; the two stage-routine
  citations now point at the compiled TSUNAMI file (`:413-414`, `:503-505`).

## 9. Choices made this round

| choice | ASKED or UNASKED |
|---|---|
| `eos_depth = geometric` as a harness-only measurement arm, card unchanged (the `with_first_wzv_after_ssh` pattern: the card states its own form, the arm scores it under the other) | UNASKED (reversible: a flag); every number that depends on it is labelled |
| card selection of `geometric` | NOT MADE, asked below |
| stage-entry seeding pads the redundant west/south face record with the periodic wrap (VORTEX pads zero) | UNASKED. Measured (`--seam-pad zero`, geometric, whole step): bit-identical rows through kt 8; at kt 9 ssh 8.74e-07 becomes 5.00e-05 and at kt 10 7.56e-07 becomes 6.50e-04, so the card consumes that record at the seam and the choice matters from kt 9 |
| ladder runs `meridional_periodicity(card.j_periodic)` scope for every step | UNASKED, round-2 mechanism |
| harness reads the observer with an effects barrier | UNASKED (instrument fix) |
| citation gate: five TSUNAMI compiled-source files and six entries added | UNASKED |

Every UNASKED item is offered for revert. Caller grep for the new option: it
exists only in the harness (`--eos-depth`, default none); library defaults and
every other card are unchanged.

## 10. Gates

(filled in the final commit)

## 11. ORCA2 pointer

Same program ORCA2 rung 0 runs: `stp2d.f90:137-138` (`eos` then `dyn_hpg` in
the external mode), the `hpg_sco` surface term (`jk = 1`; ORCA2's `jk >= 2`
loop is not exercised here), the live-depth density line (ORCA2's TEOS-10 arm
has the same stretched depth), the dyn_spg_ts substep loop and the handoff.
Given NEMO's entry, the split-explicit loop is at a few ulp (<= 1.7e-15
normalised) at kt = 1..10, and the external-mode error at kt = 1 is entirely
the pressure-gradient forcing: PLAUSIBLE evidence that ORCA2's external-stage
ssh debt is in the right-hand side feeding the loop, not in the loop. Not
transferable: the one-level stage internals (section 5) and the filter weights (nn_bt_flt = 1 here).

## 12. What is open, for round 4

1. Card selection `eos_depth` (the decision below), then re-score the ladder
   (expected: kt = 1 within 1e-16, kt >= 2 still B6).
2. B6 and B7 arms (section 6), then B4/B4j once B6 no longer hides 1e-6 effects.
3. Residual after the depth fix: the 8.6e-13 piece (section 3) and the 1.09e-19 floor.

## Citations

The first non-bit statement: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stp2d.f90:137-138`.
Its density depth: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/eosbn2.f90:361`;
its surface term: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynhpg.f90:387`.
Momentum advection OFF: `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynadv.f90:185` and
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stp2d.f90:176`. Tracer advection OFF:
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/traadv.f90:444`, with the thickness-ratio
step `TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:503-505`. The one-level correction:
`TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:413-414`.
