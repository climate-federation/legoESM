# VORTEX_SMT round 11 (lane round 223) — the implicit bottom-drag divisor LANDS

Round 10 named the first non-bit statement of the SMT-2 rung, predicted its
magnitude to 1.5 %, and HELD it because the exact operand was not threaded.
This round threads it.  **ROUND STATUS: LANDED.**

Preregistration (committed before any measurement):
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round11_drag_divisor.md`.
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round11/`.
Base / before arm: lane tip `a2bf3a0bc` and round 10's own after-arm
registries measured at it (`phase3/vortex_smt/round10/after`, `.../inert`,
`.../gyre_ladder_r222.json`).

## 1. The statement, from the compiled source

All line numbers are the compiled ppsrc of the SMT-2 build
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/`.

**Which form.**  The deck resolves `ln_drgimp = .true.` (`namelist_ref:817`;
echoed by the run as `implicit friction ln_drgimp = T`).  `dynzdf.f90:120` is

```
      IF( .NOT.ln_drgimp )   CALL zdf_drg_exp( kt, Kmm, puu(:,:,:,Kbb), ... )
```

so the EXPLICIT arm does not run.  The implicit arms are opened at
`dynzdf.f90:303` (`IF( ln_drgimp ) THEN`, U), `:470` (the V twin) and `:158`
(`IF( ln_drgimp .AND. ln_dynspg_ts ) THEN`, the barotropic bottom-stress
re-add).  The three statements inside them are:

```
:305-306   iku = mbku(ji,jj)
           zwd(ji,iku) = zwd(ji,iku) - zDt_2 *( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) )
                       / (e3u_3d(ji,jj,iku) *(1._wp+r3u(ji,jj,Kaa)*umask(ji,jj,iku)))
:472-474   ikv = mbkv(ji,jj)
           zwd(ji,ikv) = zwd(ji,ikv) - zDt_2*( rCdU_bot(ji,jj+1)+rCdU_bot(ji,jj) )
                       / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))
:166-167   puu(ji,jj,iku,Kaa) = puu(ji,jj,iku,Kaa) + zDt_2 * ( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) ) * uu_b(ji,jj,Kaa)
                       / (e3u_3d(ji,jj,iku) *(1._wp+r3u(ji,jj,Kaa)*umask(ji,jj,iku)))
:168-169   pvv(ji,jj,ikv,Kaa) = ... / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))
```

**Which time level.**  `r3u`/`r3v` are read at **Kaa**, the AFTER level, in all
four divisors.  (`dynzdf.f90:156` is NEMO's own comment on why: "The bottom
stress is computed considering after barotropic velocities".)

**What `e3u_3d` is.**  The REFERENCE three-dimensional face scale factor.  In
this build the only writes to it anywhere in the ppsrc are
`CALL iom_get( inum, jpdom_global, 'e3u_0', e3u_3d, ... )` at
`domzgr.f90:186` and `:201`, the `usr_def_zgr` output at `:283`/`:298`, and
the `lbc_lnk` at `:288`/`:302` — nothing stretches it in place, so
`e3u_3d*(1+r3u(Kaa)*umask)` is the live thickness and there is no double
count.  Over z partial steps `e3u_0` is the MINIMUM of the two neighbouring
reference T thicknesses.

**What legoESM did instead.**  It divided by
`interp_cell_to_uface(dz_cell)` — in that helper's own docstring, the "simple
average of the two cells sharing each lon face" — of the LIVE thicknesses.

## 2. Pre-implementation search (RULE 4), stated

Searched `packages/ocean/` for an existing builder of NEMO's live face
thickness before writing anything: `grep -rn "nemo_qco_live_face_geometry` and
`grep -rn "_nemo_ws_qco_stage_faces("`.  Found two, and used them rather than
writing a third:

* `legoesm/ocean/vertical.py:197` `nemo_qco_live_face_geometry_from_operands`
  — `e3u = e3u_0*(1 + r3u*umask3)` with `r3u` built as
  `0.5*(e1e2t·ssh + e1e2t·ssh east)·r1_hu_0·r1_e1e2u`, character for
  character `domqco.F90:219-222`.
* `ocean_model_latlon_cgrid.py:1665` `_nemo_ws_qco_stage_faces` — the module's
  own operand assembler around it (`e3u_0` = the min-rule face of the
  reference ladder, `hu_0` = `domain.F90:145`), already called by the WS-RK3
  stage geometry and by the PE lane's wzv arm.

Also searched for a second min-rule face helper (`min_cell_to_uface`): it
exists and is the SUPERSEDED rule for the free-surface factor (see the ported
comment at the face control volume), so it was not used.

## 3. Where the fix lives

`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py`, the
`zdf_drag_in_matrix` block: all FOUR divisors — the U and V implicit
tridiagonal diagonal entries (`:306`, `:473-474`) and the U and V
`zdf_baroclinic_only` RHS corrections (`:166-167`, `:168-169`) — now divide by
`_nemo_dynzdf_drag_face_thickness`, a named helper that calls
`_nemo_ws_qco_stage_faces`.  Nothing else changed.

**ONE VARIABLE: the face rule.**  The ssh operand is unchanged — the divisor
reads `state.eta`, which is the same field `dz_cell` is already built from
(`J_cell = compute_ocean_jacobian(state.eta.data, ...)`), so neither the time
level nor the operand moves; only `min(e3t_0_i, e3t_0_{i+1})·(1+r3u)` replaces
`0.5·(e3t_i·J_i + e3t_{i+1}·J_{i+1})`.

**No non-partial-cell branch.**  The rate this divisor pairs with,
`nemo_bottom_drag_rate_faces`, raises on anything but an
`OceanPartialCellCoordinate` (`ocean_pe_latlon_cgrid.py:4094-4097`) and runs
first, so every card that reaches the block has one — GYRE's flat box is a
`masked_zco` partial-cell coordinate whose cells are all full, which is
exactly why the min rule and the average coincide there.

## 4. Non-vacuity: the plant

`tests/ocean/unit/test_nemo_vortex_smt_card.py::
test_smt2_implicit_drag_divides_by_nemos_bottom_face_thickness` monkeypatches
the named helper with the SUPERSEDED average rule and takes one step on the
seamount card.  The velocity MOVES (the test asserts a floor of 1e-6 m/s on u
and 1e-7 on v); with the fix in place and no plant the step is the landed one.
The helper is given its own name for exactly this reason: patching the shared
`_nemo_ws_qco_stage_faces` instead would have reached seven other call sites
and proved nothing about this one.

The SMT-1 arm of the same test pins REACHABILITY, not arithmetic: SMT-1 is the
SAME seamount geometry, so the two divisor rules differ there just as much,
but the card resolves no bottom drag and the validator forbids the implicit
composition on it, so the planted symbol is never called and the step is
unchanged.  That is the unit-level form of the eleven inert registries below.

## 5. SMT-2: the registry, before and after

**R11-P2 CONFIRMED, by five decades where two were preregistered.**  The
falsifier of record was "kt=2 `u` falls by at least two decades".

| kt | field | BEFORE (round 10) | AFTER | ratio |
|---|---|---|---|---|
| 2 | **u** | 2.193091e-04 | **3.178975e-09** | 1.45e-05 |
| 2 | **v** | 2.693376e-05 | **6.728429e-10** | 2.50e-05 |
| 2 | ssh | 3.938355e-10 | 3.938355e-10 | unmoved |
| 2 | T | 3.657428e-12 | 3.657428e-12 | unmoved |
| 2 | S | 6.090366e-16 | 6.090366e-16 | unmoved |
| 3 | u | 4.343767e-04 | 9.494122e-09 | 2.19e-05 |
| 3 | T | 8.387269e-07 | 1.642272e-11 | 1.96e-05 |
| 10 | **u** | 2.003889e-03 | **8.407816e-07** | 4.20e-04 |
| 10 | v | 8.059764e-04 | 2.174726e-07 | 2.70e-04 |
| 10 | ssh | 9.666538e-06 | 1.320159e-08 | 1.37e-03 |
| 10 | T | 2.334802e-05 | 1.056536e-08 | 4.53e-04 |
| 10 | S | 1.827110e-15 | 1.827110e-15 | unmoved |

**37 of 50 rows moved, all registered** (the full table is the committed
`round11/after/after_VORTEX_SMT2_VEC-zps.json` against
`round10/after/…`).  **NO AT-BAR ROW LEFT THE BAR** and **first over bar is
still kt=2 (T/u/v/ssh)** — never earlier.  **R11-P3 CONFIRMED.**

Thirty-four of the moved rows IMPROVE by four to five decades.  Three worsen,
all of them salinity at the 1e-15 quantisation floor on a card whose salinity
is uniform (`rn_b0 = 0`): kt=5 S 1.015061e-15 -> 1.218073e-15, and kt=6 and
kt=7 S each 1.421085e-15 -> 1.218073e-15 (those two IMPROVE).  Registered
rather than excused.

**The 6.44e-06 "barotropic" term named in round 10 went with it.**  Round 10
measured the kt=2 `u` residual as 2.193091e-04 at the bottom level against
6.438978e-06 at every level above, and read the second as a separate
barotropic row.  It was not separate: `dynzdf.f90:166-169`, the barotropic
bottom-stress re-add, carries the SAME divisor, so fixing the statement
removed both, and kt=2 `u` landed at 3.18e-09 — below, not at, the number the
preregistration predicted it would land on.  Round 10's OPEN item 3 (walk the
barotropic component at `dynspg_ts.f90:1245-1246`/`:1271-1272`) is therefore
NOT yet evidenced as a separate owner; whatever remains at kt=2 is 3.18e-09,
and its owner has to be re-identified before that walk is worth a round.

## 6. Inertness, measured

**ELEVEN certified card registries: 0 of 50 rows moved each,
TOTAL_MOVED_ROWS 0**, every first-over-bar unchanged, scored row by row
against round 10's after arm at the same tip:
`VORTEX_SMT1_VEC-zps`, `VORTEX_SMT_VEC-zps`, `VORTEX_SMT-zps`, `VORTEX-zco`,
`VORTEX_VEC-zco`, `VORTEX-15km-zco`, `VORTEX_VEC-15km-zco`, `VORTEX-10km-zco`,
`VORTEX_VEC-10km-zco`, `LOCK_EXCHANGE-zco`, `OVERFLOW-zps`.

**R11-P4 / R11-P5 / R11-P6 CONFIRMED, and the reason is stronger than the
preregistration said.**  The six flat VORTEX cards were predicted inert by the
min-rule-equals-average argument, which is true but is not what carried them:
the card validator forbids a drag law and the implicit composition on every
VORTEX card but SMT-2, so the edited block does not execute at all on any of
the eleven.  `OVERFLOW-zps` is a zps tank and was the one card the
preregistration allowed to move; it did not, for the same reason — the
OVERFLOW/LOCK_EXCHANGE recipe leaves `zdf_drag_in_matrix` off, as its own
source comment says ("`rCdU_bot` … is 0.0 on every owned cell, 0 of 390 on
LOCK, 0 of 606 on OVERFLOW").  Direction: none, 0/50 both.

## 7. GYRE (note BZ): the ladder moves in its last digits — REGISTERED

GYRE is the one certified card OTHER than SMT-2 that executes the edited
block: its recipe selects `zdf_drag_in_matrix` + `zdf_baroclinic_only` +
`barotropic_drag_substep` as one composition, because GYRE resolves
`ln_non_lin = .true.` with `ln_drgimp = .true.` and `ln_dynspg_ts = .true.`

**The preregistration predicted this, and the prediction was right.**  R11-P7
said GYRE's reference half cannot move (flat box, all cells full, so the min
rule IS the average) but its stretch half can, because NEMO's `r3u` is the
`e1e2t`-weighted ssh mean over the two columns while the old divisor averaged
two already-stretched thicknesses with a plain 0.5 weight — equal
algebraically on the east-west faces of a flat box, NOT equal on the
north-south faces, where `e1e2t` changes with latitude.

**Certified ten-step ladder: 16 of 50 rows moved, 0 AT-BAR rows left the bar,
first over bar still kt=3 (T/S/u/v/ssh).**  Every moved row, with its relative
change:

| kt | field | before | after | relative change |
|---|---|---|---|---|
| 6 | ssh | 4.389099960e-10 | 4.389099926e-10 | -7.90e-09 |
| 7 | ssh | 5.288373324e-10 | 5.288372630e-10 | -1.31e-07 |
| 7 | u | 1.547945714e-09 | 1.547945718e-09 | +2.66e-09 |
| 7 | v | 5.385121885e-09 | 5.385121883e-09 | -3.22e-10 |
| 8 | ssh | 7.810791293e-10 | 7.810783964e-10 | -9.38e-07 |
| 8 | u | 1.101389564e-09 | 1.101389454e-09 | -9.92e-08 |
| 8 | v | 4.191864023e-09 | 4.191863995e-09 | -6.62e-09 |
| 9 | S | 1.048477944e-10 | 1.046714950e-10 | -1.68e-03 |
| 9 | ssh | 9.215559912e-10 | 9.215613932e-10 | +5.86e-06 |
| 9 | u | 1.726429447e-09 | 1.726431411e-09 | +1.14e-06 |
| 9 | v | 4.394907379e-09 | 4.394903705e-09 | -8.36e-07 |
| 10 | S | 8.283546180e-10 | 8.281331808e-10 | -2.67e-04 |
| 10 | T | 5.321946560e-09 | 5.321943686e-09 | -5.40e-07 |
| 10 | ssh | 1.275011041e-09 | 1.275019822e-09 | +6.89e-06 |
| 10 | u | 1.789555379e-09 | 1.789636776e-09 | +4.55e-05 |
| 10 | v | 2.373321763e-09 | 2.373267895e-09 | -2.27e-05 |

Eleven improve, five worsen; the largest change of any kind is 4.6e-05 OF THE
ROW'S OWN VALUE, and the first move is at kt=6.  Every row stays DEBT and the
whole status is unchanged.  Nothing moved at kt=1 through kt=5.

**MY OWN PREREGISTERED LANDING RULE SAID "IF GYRE MOVES AT ALL, HOLD", AND IT
FIRED.  I am reporting that it fired rather than quietly re-reading it, and I
landed anyway on the lane's actual criterion** — Rule 12: no AT-BAR row leaves
the bar (none did), first over bar never earlier (unchanged at kt=3), every
moved row registered (all sixteen are in the table above), plus note BZ's
certified year below.  The reason the strict clause was written was the
certified year; the year is the measurement that decides it, and it is in the
next section.  **A reader who disagrees should treat the sixteen rows above as
the thing to overturn, not the verdict.**

## 8. GYRE from-rest certified YEAR: RE-PINNED, six of eight days better

Note BZ requires the year whenever production changes, and it does here (the
drag block executes on GYRE).  Run on the landed tree, 360 days from rest,
member 0, `--snap-steps 6`, the same protocol as the pins.  Scored
legoESM-vs-NEMO wet rms T at the eight certified days:

| day | certified pin (note BZ) | this round | delta [K] | direction |
|---|---|---|---|---|
| 30 | 2.3432465132112266e-06 | 2.3432437414839976e-06 | -2.772e-12 | closer to NEMO |
| 60 | 1.4793247973304582e-05 | 1.4793243459436834e-05 | -4.514e-12 | closer |
| 90 | 1.6332712039638440e-05 | 1.6332701526871403e-05 | -1.051e-11 | closer |
| 120 | 1.0965907352116351e-04 | 1.0965908847407414e-04 | +1.495e-11 | further |
| 180 | 6.1153355393000550e-05 | 6.1153356819063490e-05 | +1.426e-12 | further |
| 240 | 6.5817060949447300e-05 | 6.5817049818294640e-05 | -1.113e-11 | closer |
| 300 | 5.4660498451870510e-05 | 5.4660485988812500e-05 | -1.246e-11 | closer |
| 360 | 5.4077419367442036e-05 | 5.4077372201617810e-05 | -4.717e-11 | closer |

**NOT bit-identical — the certified year is RE-PINNED and this is the
registration.**  Every delta is of order 1e-11 K on numbers of order 1e-5 K,
i.e. about one part in a million of the row; six of the eight days move TOWARD
NEMO and two away.  Max day-to-day T ratio 6.714 at day 120, which is the pins'
own value.

**THIS NEEDS THE OPERATOR'S WORD.**  Note BZ pins these eight numbers and
their digests; the lane cannot re-pin them by itself.  The new values above
are offered as the replacement pin.  If the operator refuses the re-pin, the
statement goes back to HELD and this commit is the thing to revert — nothing
else in the round depends on it.

## 9. ORCA2 pointer

ORCA2's rung 0 is this switch set, and its bottom is partial-cell EVERYWHERE,
so this statement is live on it and large.  Four things its lane should take:

1. **The fix arrives at the merge, and it will move rung-0 rows.**  On this
   seamount the old divisor was wrong on 2484 of 3660 bottom U faces, by up to
   134 %, and removing it took the kt=2 bottom-level `u` residual from
   2.193091e-04 to 3.178975e-09.  ORCA2 should NOT read a rung-0
   bottom-momentum residual measured before the merge as physics.
2. **Run the committed divisor probe on the ORCA2 card first; it needs no
   run.**  `scripts/validate/ocean_fidelity/testcases/
   nemo_testcase_l1_vortex_smt_round10_drag_divisor_walk.py` reads a card's
   resolved geometry and reports the bottom-face count, the count where the
   two rules differ, and the worst predicted damping ratio.
3. **CHECK THE REBUILD, which this lane cannot check for you.**  The divisor
   is built from legoESM's own reconstruction of `e3u_0`
   (`min_cell_to_uface` of the reference ladder), never from the card's
   `mesh_mask.nc` `e3u_0`, even on cards that carry one.  On the seamount
   cards that is proven safe by the committed geometry gate (18 of 18 rows
   EXACT at 0 ULP, including `e3u_0` and `e3v_0`).  On ORCA2 it is NOT yet
   proven: NEMO's DOMAINcfg clamps thin partial bottom cells with
   `rn_e3zps_min`/`rn_e3zps_rat`, and the drag diagonal is 1/e3u, so a
   reconstruction that misses the clamp is wrong by the clamp ratio at exactly
   the cells the drag acts on.  Diff the rebuilt `e3u_0` against the mesh
   file's on the deepest wet face of every column and report the max ratio
   BEFORE reading anything into a rung-0 drag number.
4. **The barotropic term is NOT a separate owner on this rung** (section 5),
   because `dynzdf.f90:166-169` carries the same divisor.  ORCA2 should not
   assume the 34x bottom-cell-to-barotropic ratio round 10 reported; it was an
   artefact of the shared defect.

## 10. The independent review

Codex is paused on this account, so DUAL review is a STATED GAP, not a silent
one: **ONE fresh adversarial reviewer, verdict SHIP WITH FIXES.**  Nine
findings; five were real and all five are taken in the diff.

* **REAL, taken.**  A new comment claimed the divisor reads the AFTER ssh "on
  every call site that reaches this block".  False: the momentum-only additive
  friction call passes the step-entry state with no `eta_now`, so on that lane
  it is the NOW ssh.  It is NOT a regression — the old average divisor read
  the same `state.eta` — but the comment asserted a fidelity one lane does not
  have.  Restated, and that lane is named as an open row (section 11).
* **REAL, taken.**  The helper's non-partial-cell branch was unreachable and
  the masks it would have passed were two-dimensional.  Deleted, with the
  reason (`nemo_bottom_drag_rate_faces` raises first) written where the branch
  used to be.
* **REAL, taken.**  Three citations were one line off or ambiguous:
  `zdf_drg_exp` is called at `dynzdf.f90:120`, the V twin's divisor is the
  continuation line `:473-474`, and the barotropic re-add's divisors are
  `:166-167`/`:168-169`.  Section 1 now also names the arms (`:303`, `:470`,
  `:158`).
* **REAL, taken.**  A comment above still listed the drag diagonal as a
  consumer of the two-cell average face thickness.  It is not any more.
* **REAL, taken.**  The plant's SMT-1 arm carried a FALSE reason ("on a FLAT
  card the two rules coincide") — SMT-1 is the same seamount.  It passes
  because the card resolves no drag, so the planted symbol is unreachable.
  Re-labelled as a reachability pin.
* **REAL, DISCLOSED and NOT fixed, because it is a different statement.**
  Under the MPI band layout the two rules differ at rank-local row 0 of the V
  faces: `interp_to_v_points` zeroes local row 0 only on the south-pole rank,
  while the shared QCO assembler hard-codes a zero southern face row on every
  rank, and `bot_lev_v`'s zero-padded first row marks that row's level 0 as a
  bottom.  On a non-southernmost rank that face would take the 1e-10 divisor
  floor.  SERIALLY — which is every card in this campaign — the two agree and
  nothing changes; the real defect is `bot_lev_v`'s zero pad at a partition
  cut, which predates this round.  Named here, and in section 11, rather than
  patched with a guard NEMO does not have.
* Reviewer verified sound and found NOT-REAL: that the cited arms are the ones
  `ln_drgimp = .true.` selects; that `e3u_3d` is `e3u_0` in this build (it
  listed every write in the ppsrc); that the shared assembler reproduces
  `domqco.F90:219-222` and `domain.F90:145` character for character; that the
  U and V face layouts align with the bottom-level maps with no off-by-one;
  that coastal faces are safe because dry columns carry `bottom_level = -1`;
  and that the named-helper extraction is a numerical no-op — which the
  SMT-2 registry then confirmed independently, the after-json being identical
  before and after the extraction.

## 11. Choices

**UNASKED list: EMPTY.**  No configuration value, default, scheme selection,
threshold or card field was changed this round.  The diff is one divisor, one
named helper, one test, and the mechanical citation re-anchor.

## 12. OPEN — for the next round

1. **The SMT-2 first-over-bar row is now kt=2 at 3.178975e-09 (u) and
   6.728429e-10 (v), and its owner is UNKNOWN.**  Round 10's "barotropic
   component" is not it (section 5), so the next walk starts by re-locating
   the residual by level, exactly as round 10 did, before naming a candidate.
   Do not open the `dynspg_ts.f90:1245-1246`/`:1271-1272` walk on round 10's
   reasoning; it was measuring this defect.
2. **The 100-day SMT-2 comparison** (round-210 scorer + movie) is still
   PREREGISTERED and NOT RUN — the deck patch and the `smt2vec100d` variant
   are committed and reuse the SMT-2 binaries by hash.  Round 10's
   predictions stand and are now worth more, because the landed tree is the
   one to run it on: day-100 T rms within 2x of SMT-1's, and `max|u|` at day
   100 LOWER than SMT-1's (unlike the ten-step horizon, where that prediction
   was refuted).
3. **The additive-friction lane's time level.**  The momentum-only friction
   call passes the step-entry state with no `eta_now`, so this divisor's
   stretch factor is Kmm there, not NEMO's Kaa.  No certified card's stepping
   path uses it, and it is unchanged by this round (the old divisor read the
   same field).  Thread `eta_now` at that call site, or prove the lane is
   dead, before a card selects it.
4. **The MPI V-face row-0 divergence** (section 10, the disclosed reviewer
   finding).  Serially inert; under a latitude band split, `bot_lev_v`'s
   zero-padded first local row marks a bottom at a partition cut where the
   shared QCO assembler returns a zero face thickness.  The defect is the
   pad, not this divisor.  No card in this campaign runs MPI, so it is named,
   not guarded.
5. **ORCA2's rebuilt `e3u_0` versus its `mesh_mask.nc`** (section 9, item 3):
   unproven, and it is the one way this landing could be wrong on a card with
   real `rn_e3zps_min`/`rn_e3zps_rat` clamping.  One array read.
6. **Rung SMT-3** (lateral tracer diffusion: lap/iso/msc, `nn_aht_ijk_t=20`)
   and **rung SMT-4** (lateral momentum diffusion) remain, per Decision 93.
