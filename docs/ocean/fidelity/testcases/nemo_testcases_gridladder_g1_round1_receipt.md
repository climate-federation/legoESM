# Grid ladder rung G1 — round 1 receipt (gridladder_r1)

Status: **STOPPED_FOR_RECORD** (DECISION 118 applied; measurement setup only;
nothing measured, no NEMO run). Earlier status STOPPED_FOR_DECISION is closed. Preregistration:
docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1_round1.md (committed
first). Branch fidelity/nemo-testcases-orca2-gridladder from b65a32ff0.

## What exists after this round

- The G1 NEMO configuration, as files (the operator's run.sh creates
  cfgs/ORCA2G1_OMIP_L1 and ORCA2G1_OMIP_L1_P3 from them):
  scripts/validate/ocean_fidelity/testcases/nemo_testcase_g1_orca2grid/
  namelist_cfg, domaincfg_namelist_cfg, usrdef_istate.F90, usrdef_sbc.F90,
  restart_g1_usrdef_ssh.patch, g1_step_record.F90,
  stprk3_g1_step_record.patch, make_bathy_level.py, check_records.py, run.sh.
- The explicit card `build_orca2g1_flat_card(deck_root)` with its deck reader,
  bump and deck-bound validator pair, dispatched as "ORCA2G1-flat", declared
  NOT execution-ready (two unmeasured features). The shared validator's table
  does not list it and refuses it as unknown (tested).
- One shared-model addition: the one-level opt-in on the reference-coordinate
  builder, default off, selected only by this card. Caller grep
  (`grep -rn allow_single_level packages`): the definition and the G1 card,
  nothing else. Library defaults and every other card are unchanged.
- Tests: tests/ocean/unit/test_nemo_orca2g1_card.py and
  tests/ocean/fidelity/test_nemo_testcase_g1_orca2grid_tools.py.

## Findings (detail and citations in the preregistration, section 2)

1. An all-wet flat ocean on ORCA2's grid cannot run (mesh-pole cells 0.17 m
   wide; external Courant number 14,891). Built with ORCA2's coastline; the
   Courant number is then 0.0781 (SUPERSEDED 5400 s / 65 value; 0.7812 at
   10800 s, see the D118 section). Offline numbers from the shipped domain file.
2. ORCA2's RK3 deck runs 10800 s with automatic substeps; 5400 s is the
   reference namelist's value. Built as briefed (5400, 65 stated) —
   SUPERSEDED by D118: 10800, automatic, resolves 13.
3. NEMO zeroes the sea level on a read domain and never calls the user bump;
   one added statement in a case-local copy applies it.
4. The TSUNAMI card, its one-level step support and its acquisition scripts
   are not on this branch (PR #1920's branch).

The citation gate passes on the preregistration (40 citations, 0 unmapped,
whole-map audit clean; its shifted-citation plant exits 1). This receipt
itself carries no line citations, so its own gate pass is empty.

## DECISION 118 applied (user, 2026-10-12)

(1) ORCA2's coastline over the flat 100 m floor is kept (already built).
(b) The time step is ORCA2's deck value, 10800 s
(ORCA2_OMIP_L4/EXP00/namelist_cfg line 28), with everything derived from it
as ORCA2's deck resolves it.

| item | SUPERSEDED | in force |
|---|---|---|
| time step | 5400 s | 10800 s |
| substeps | 65, stated, automatic off | automatic (ceiling 0.8); resolves 13 |
| external Courant number, 100 m | 0.0781 | 0.7812 (NEMO stops at 0.9) |
| barotropic step | 83.08 s | 830.77 s |
| wave reaches fold / coast / seam | kt 40 / 45 / 66 | kt 20 / 23 / 33 |
| forcing call period, run lengths 2 / 10 / 100, restarts every 10 | — | unchanged |

The Courant number is offline (shipped cell sizes, ORCA2's mask), not NEMO
output; the largest value sits at (ji,jj) = (50,2) on the Antarctic coast.
It is pinned by a committed test written in NEMO's own order of operations
and frozen as prediction P2 (NEMO prints 13 and 0.7812 in its own log; the
acquisition already copies those lines). Gate for "the old step is gone":
restoring 5400 in the deck turns 2 tests red (shown, then restored).

Card change: the substep count is no longer read from the deck; it is
resolved from the domain file with the existing shared helper (no second
implementation). The deck reader now requires automatic substeps and reads
the ceiling. No library default and no other card changed.

Choices this pass — ASKED: 10800 s; automatic substeps; unchanged run
lengths. UNASKED: none. (The deck still states the unused count as 30, the
reference value ORCA2's deck leaves in place; NEMO overwrites it.)

## What was checked, and by what

| claim | gate | result |
|---|---|---|
| both source patches apply unfuzzed and only add lines | `patch --fuzz=0` on fresh copies; run.sh preflight | applies; 0 removed lines |
| writer and user routines are valid Fortran against NEMO's modules | `gfortran -fsyntax-only` with the TSUNAMI RK3 build's module directory | 3 of 3 clean |
| the committed deck is the 100-step deck and renders to the 2- and 10-step decks by 3 lines | run.sh preflight | pass |
| deck values; every one of the 54 required switches refused when moved | pytest, parametrised | pass |
| automatic substeps 13 and Courant number 0.7812 at 10800 s | pytest, NEMO's order of operations on the stand-in domain | pass |
| card builds, is fp64, one level, 201-cell bump; fifteen planted disagreements refused | pytest on a stand-in domain synthesised from ORCA2's shipped file | pass |
| the one-level opt-in is non-vacuous | reverting it turns 2 tests red | shown |
| record checker: clean record admitted; five plants exit nonzero | pytest | pass |
| geometry identity against NEMO's mesh | pytest | SKIPPED: no record yet |
| DOMAINcfg builds and runs; NEMO builds and runs | NO GATE this round | unexecuted |

The stand-in domain is NOT DOMAINcfg output and supports no geometry claim.
run.sh has only been preflighted; its build and run branches are unexecuted.
The geometry-identity test body has never run.

Focused tests: 68 passed, 1 skipped. Neighbouring suites (recipe, advection
OFF arms, constants / dispatch / private-import ratchets): 5748 passed, 3
skipped, 2 failed. Both failures are in files this round did not touch
(tests/unit/test_jra55_do.py literal; land/slab_land.py dispatch guard).

## ORCA2 pointer (skeleton; every statement proved on G1 is an ORCA2 deliverable)

| statement | ORCA2 owner (round) | G1 status |
|---|---|---|
| T/U/V/F fold masks, raw face thickness | D114, round 238 | to measure (geometry identity) |
| required barotropic filter alpha | D115, round 237 | to measure (kt = 1 substeps) |
| fold exchange of ssh and the barotropic pair | rounds 205-206, 244 (by round title) | to measure (from kt about 40) |
| cyclic i-exchange | TSUNAMI round 4 | to measure (from kt about 66) |
| split-explicit substeps, EEN planetary | TSUNAMI round 5 | to measure (interior, kt = 1) |
| coast: south wall and lateral walls | — | to measure (from kt about 45) |

## Line-stability note

The new card code sits at the end of the recipe file and the one-level opt-in
keeps the coordinate file's line count, because the first layout moved five
line citations of older receipts and turned the citation gate's whole-map
audit red. The first 3379 lines of the recipe file are byte-identical to the
base commit (checked by script). The preregistration's four Python line
citations were re-pointed after the move; nothing else in it changed.

## Review

Single review (codex): **INCOMPLETE, no verdict.** The one `codex exec` review
of `git diff b65a32ff0..HEAD` was ended by its own 590 s limit (exit 124)
while still reading sources; it was not re-run. Log:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round1/codex_review_r1.log.
Its two interim notes, verbatim: "The source-level formula and DOMAINcfg path
are holding up so far." and "I'm now narrowing the review to failure-path
behavior in the acquisition script and selectors the card claims to
bind—those are where the substantive contradictions appear."

Acted on the second note with an own dump of the card's resolved
configuration. ONE confirmed defect: the card inherited ORCA2's enhanced
vertical diffusion (scheme "enhanced_diffusion", 100 m2/s) although the G1
deck states `ln_zdfevd = .false.`; nothing in the validator saw it. Fixed (the
card now selects no convection scheme), with a validator row and a planted
refusal; rows for the vorticity thickness option and lateral mixing were added
at the same time. The acquisition script's failure paths were NOT reviewed by
anyone but the author. Treat run.sh as unreviewed.

## OPEN for round 2

1. CLOSED by D118 (coastline kept; 10800 s).
2. Operator: merge the TSUNAMI branch (PR #1920) into this branch, so the
   one-level step, the y-wrap scoping and the TSUNAMI card are present.
3. Operator: `run.sh --run`; record the DOMAINcfg namelist and the SHA-256 of
   domain_cfg.nc (run.sh prints it) here.
4. Round 2: admit the record, run geometry identity (P3), then the kt = 1..10
   and 100-step ladders with the section-7 region counts.
5. Named, not done: the bump is far from every boundary, so stage frames
   (kt 1..10) never see the fold at finite amplitude; a second record with
   stage frames around kt 20-25 would (kt 40-50 at the SUPERSEDED step).
