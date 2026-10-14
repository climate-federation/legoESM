# Grid ladder rung G1 — round 1 receipt (gridladder_r1)

Status: **STOPPED_FOR_DECISION** (measurement setup only; nothing measured, no
NEMO run). Preregistration:
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
  bump and validator, registered as "ORCA2G1-flat", declared NOT
  execution-ready (two unmeasured features).
- One shared-model addition: the one-level opt-in on the reference-coordinate
  builder, default off, selected only by this card. Caller grep
  (`grep -rn allow_single_level packages`): the definition and the G1 card,
  nothing else. Library defaults and every other card are unchanged.
- Tests: tests/ocean/unit/test_nemo_orca2g1_card.py and
  tests/ocean/fidelity/test_nemo_testcase_g1_orca2grid_tools.py.

## Findings (detail and citations in the preregistration, section 2)

1. An all-wet flat ocean on ORCA2's grid cannot run (mesh-pole cells 0.17 m
   wide; external Courant number 14,891). Built with ORCA2's coastline; the
   Courant number is then 0.0781. Offline numbers from the shipped domain file.
2. ORCA2's RK3 deck runs 10800 s with automatic substeps; 5400 s is the
   reference namelist's value. Built as briefed (5400, 65 stated).
3. NEMO zeroes the sea level on a read domain and never calls the user bump;
   one added statement in a case-local copy applies it.
4. The TSUNAMI card, its one-level step support and its acquisition scripts
   are not on this branch (PR #1920's branch).

## What was checked, and by what

| claim | gate | result |
|---|---|---|
| both source patches apply unfuzzed and only add lines | `patch --fuzz=0` on fresh copies; run.sh preflight | applies; 0 removed lines |
| writer and user routines are valid Fortran against NEMO's modules | `gfortran -fsyntax-only` with the TSUNAMI RK3 build's module directory | 3 of 3 clean |
| the committed deck is the 100-step deck and renders to the 2- and 10-step decks by 3 lines | run.sh preflight | pass |
| deck values; every one of the 54 required switches refused when moved | pytest, parametrised | pass |
| card builds, is fp64, one level, 201-cell bump; ten planted disagreements refused | pytest on a stand-in domain synthesised from ORCA2's shipped file | pass |
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

## Review

Single review (codex): see the end of this file.

## OPEN for round 2

1. DECISION_NEEDED answers (coastline; time step) before the acquisition runs.
2. Operator: merge the TSUNAMI branch (PR #1920) into this branch, so the
   one-level step, the y-wrap scoping and the TSUNAMI card are present.
3. Operator: `run.sh --run`; record the DOMAINcfg namelist and the SHA-256 of
   domain_cfg.nc (run.sh prints it) here.
4. Round 2: admit the record, run geometry identity (P3), then the kt = 1..10
   and 100-step ladders with the section-7 region counts.
5. Named, not done: the bump is far from every boundary, so stage frames
   (kt 1..10) never see the fold at finite amplitude; a second record with
   stage frames around kt 40-50 would.
