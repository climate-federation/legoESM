# RECEIPT — TSUNAMI lane, round 1 (survey + card + acquisition)

Date 2026-10-08. Lane tip at start `0575754c061c`. Preregistered in
`PREREG_nemo_testcases_l1_tsunami_round1.md`, committed before any record.

Status: **STOPPED_FOR_RECORD.** The card is transcribed and refuses to
execute (five named blockers); the acquisition is written and preflighted;
NEMO has not been run.

Two things to take away first:

1. **TSUNAMI does not run ORCA2's program.** It compiles no `key_RK3`, so
   NEMO runs the leapfrog step, which the case's own source reduces to the
   split-explicit external mode alone. ORCA2 runs the RK3 step. The sub-step
   loop is shared code; the window set-up and closing are not.
2. **NEMO's metric lags its surface height here** (read off the code,
   PLAUSIBLE until the record is in): the "now" metric at step k comes from
   step k-4's ssh. The card must carry that; no legoESM program does.

## 1. What was built

| artefact | what it is |
|---|---|
| the TSUNAMI card | additive in the shared recipe module: a resolved-namelist record with every switch stated, the geometry and initial ssh transcribed from the case's user routines, its own validator branch, five declared gaps. Not added to the shared card dispatch, so no existing gate can reach it |
| one opt-in flag in the shared coordinate builder | lets a card build a ONE-level column; default unchanged (caller grep: the TSUNAMI card is the only caller) |
| the card test | 10 tests: construction and shapes, every switch against NEMO's own resolved TSUNAMI namelists and cpp keys, the lookup refusing a planted drift, the validator refusing a forcing switch turned on, the model config, the f-plane, the initial ssh against the formula at three cells, the execution gate's refusal, a planted drift refused, the single-level flag opt-in only |
| the acquisition | builds the shipped case twice (reference and instrumented), refuses if the cpp keys stop selecting the leapfrog step, runs both, admits by bit-identity of NEMO's own ssh and barotropic-velocity output |
| the record writer | a new read-only per-step writer for the leapfrog step (kt 1..100 after-step ssh/uu_b/vv_b; kt 1..10 also every time-level slot and the metric), plus VORTEX's sub-step writer reused unchanged, gated to kt 1..10 |
| the record checker and its test | self-described headers, the exact group set of every record (per sub-step frame for the sub-step file), finite owned values, output identity; 11 tests including nine planted refusals and a one-ULP output difference |

## 2. The survey and the step program

The full table is in the preregistration's section 1. The resolved values
that matter: 201 x 201 doubly periodic cells of 10 km, ONE 100 m level, f-plane
at 38.5 N, 100 steps of 1000 s, split-explicit free surface with forward
integration, boxcar filter and nn_e = 6 sub-steps, EEN barotropic Coriolis,
zero forcing, no drag. Every other operator the deck names is either OFF or
never called by this case's step.

What the case changes in NEMO's step program (`diff -u` against `src/OCE`):
its leapfrog step keeps only surface forcing, the metric update, the
split-explicit external mode, output and the time-level rotation. Its RK3
step compiles but is never called. Its output routine writes no state.

Geometry: uniform z levels, not sigma. One wet level of exactly 100 m; the
reference depths are 50/150 m at T and 0/100 m at W; every horizontal scale
factor is 10 000 m; no land.

## 3. Blockers

| id | missing piece | smallest card-selected addition |
|---|---|---|
| B1 | a step program that runs the leapfrog external mode alone | a whole-step identity selected only by this card that calls the existing split-explicit window once per step, with the leapfrog window set-up and closing |
| B2 | the lagged metric | B1 carries the metric's own time-level slots |
| B3 | j-periodicity as card data | legoESM's y-wrap is a process-global flag today |
| B4 | the NEMO-literal barotropic arms across an open i-seam | measure; every certified card has a closed ring |
| B5 | a one-level column in the model step | the coordinate now builds behind the opt-in flag; the step is unproven |

Found while building, not by reading: B5. The shared coordinate builder
refused a one-level column outright; the card could not be constructed until
the opt-in flag existed.

## 4. Choices made this round

| choice | ASKED or UNASKED | note |
|---|---|---|
| drop key_xios from both acquisition builds | UNASKED, VORTEX precedent | no XIOS in this toolchain; the case guards only a context finalise with it |
| ln_meshmask on in the acquisition deck | UNASKED, VORTEX precedent | geometry receipt |
| passivity judged on NEMO's own output, not a restart | UNASKED, forced | the case's step never writes a restart |
| the card's barotropic arm carried from the VORTEX flux identity, gaps declared | UNASKED, VORTEX round-1 precedent | execution is refused, so nothing runs under it |
| one-level opt-in flag in the shared coordinate builder | UNASKED | pure addition, default unchanged, selected only by this card |
| card kept out of the shared dispatch | UNASKED | keeps every existing card gate unchanged |
| re-anchor tool rewrote three spans of the round-8 receipt | rule-driven | the committed tool's own scope |

Every UNASKED item is offered for revert.

## 5. Gates

| gate | result |
|---|---|
| card test + record checker test | `21 passed` |
| planted defect: the ssh bump radius computed over a shifted distance | RED (1 failed, 6 passed), restored |
| planted: card's rn_atfp moved 0.1 -> 0.2 | RED, the NEMO-namelist test (`1 failed, 9 passed`), restored |
| planted: sub-step checker reverted to "exit ssh only" | RED (`3 failed, 8 passed`), restored |
| card, checker, recipe module, citation gate test, private-import ratchet (CPU, x64, at `4840353cd`) | `69 passed in 244.26s`, log sha256 `6b852cbab0201bf5` |
| citation gate, this receipt | `PASS`, 29 citations, 0 failures, clean tree at `4840353cd41b`, json sha256 `c52e74ce77577b41` |
| citation gate, planted shift of `stpmlf.F90:118` | `FAIL` (`SYMBOL-NOT-AT-LINE`), exit 1, json sha256 `9cc3a8cb705051ae` |
| acquisition preflight on the committed tree | `PREFLIGHT_OK`, exit 0 |
| acquisition preflight with key_RK3 planted in the cpp keys | `REFUSE`, exit 68 |
| new Fortran writer, syntax only against stub modules | `WRITER_SYNTAX_OK` |
| every instrument patch, `patch --dry-run --fuzz=0` on fresh copies | apply clean |
| pre-existing red, not this round's | the constants ratchet on `tests/unit/test_jra55_do.py`, a file this round does not touch |

Review: single review (codex), verdict **DO NOT SHIP** on the pre-fix diff
(the first codex run died mid-exploration with no verdict; the re-run, log
sha256 `1aa5577f46d48c48`, is the one quoted). Disposition:

- CONFIRMED, fixed: the record checker admitted a sub-step file holding only
  the exit ssh. It now requires every group the patch writes, per frame.
- CONFIRMED, fixed: twelve selectors dyn_spg_ts reads were inherited, not
  stated. The card now states them (all `.false.`) and the validator refuses
  any switched on.
- CONFIRMED, fixed: the switch test compared literals with literals. It now
  reads NEMO's resolved TSUNAMI namelists and cpp keys.
- PLAUSIBLE, registered debt D-TSU-1: the acquisition does not grep the
  preprocessed dyn_spg_ts for the sub-step writer, as it does for the step
  writer. The tightened checker refuses any record with a group missing.
- Codex confirmed with no finding: TSUNAMI runs stp_MLF; the four-step metric
  lag; the 201 x 201 indexing and the vertical ladder; the record header
  counts; the writer's passivity.

## 6. ORCA2 pointer

ORCA2 rung 0 runs the RK3 program with nn_bt_flt = 3. TSUNAMI runs the
leapfrog program with nn_bt_flt = 1. A bit-exact TSUNAMI sub-step loop is
evidence for ORCA2's loop body only where a statement does not read the
filter weights; the window set-up, closing and the lagged metric do not
transfer. A key_RK3 build of the same case would run the case's own RK3 step
(ORCA2's program with every physics term off). That is a configuration
change, asked in the final lines, not made.

## 7. How to acquire

```
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tsunami/run.sh
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tsunami/run.sh --run
```

Evidence lands in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round1/oracle_tsunami_r1`.
This round's own evidence (review text, test logs, gate json) is in the
parent directory.

## 8. Open for round 2

Registered debt D-TSU-1 (codex, PLAUSIBLE): add a preprocessed-source grep
for the sub-step writer to the acquisition's build guard.

Score G1-G6 and L1-L4 against the record; build the kt = 1..10 scorer for the
new record family; decide B1's shape with the record in hand (it is the
largest piece); answer the key_RK3 question.

## Citations

The case's user routines. Domain size: `tests/TSUNAMI/MY_SRC/usrdef_nam.F90:92-93`
and one wet level, `:98`. The bump's origin indices:
`tests/TSUNAMI/MY_SRC/usrdef_hgr.F90:79-80`; positions in kilometres, `:93-94`;
the f-plane, `:121-122`. The level thickness:
`tests/TSUNAMI/MY_SRC/usrdef_zgr.F90:128`; the flat, landless bottom,
`:187-189`. Uniform temperature and salinity:
`tests/TSUNAMI/MY_SRC/usrdef_istate.F90:65-66`; the ssh bump, `:94-101`, with
its radius from the maximum distance, `:96`. Zero forcing:
`tests/TSUNAMI/MY_SRC/usrdef_sbc.F90:60-65`.

The step program. NEMO calls the leapfrog step without key_RK3:
`nemogcm.F90:186`. The case's step: forcing, `tests/TSUNAMI/MY_SRC/stpmlf.F90:111`;
the metric from the after slot, `:118`; the external mode, `:126`; the
time-level rotation, `:131-134`.

The external mode. Fresh window every step: `dynspg_ts.F90:202`. The
Coriolis coefficient from the now metric: `:355`. The window starts from the
now ssh and now face depth: `:486` and `:490`. The un_adv filter:
`:920`. The automatic sub-step count: `:1240`.

Geometry and initial levels. Periodic wrap of the positions before the ssh
routine reads them: `domhgr.F90:114`. The 1-D ladder's end records:
`depth_e3.F90:68` and `:73`. All three ssh slots start equal:
`restart.F90:466` and `:481`. All three metric slots start equal:
`domqco.F90:126` and `:99`.
