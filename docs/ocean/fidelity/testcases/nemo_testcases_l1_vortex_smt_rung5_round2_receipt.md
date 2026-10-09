# Receipt — SMT-RUNGS round 2: SMT-5 build (T/S damping deck, NEMO dump, legoESM arm)

**Status: STOPPED_FOR_RECORD.** Deck, NEMO-side inputs and the legoESM arm are
built and committed; the SMT-5 NEMO record is the operator's
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smtrungs_round2_smt5/run.sh`
`--smoke`, then `--run`). Decision 107 fixes every deck value. Base `ea7f91213`.
Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smtrungs_round2_smt5.md`
(committed before any record exists).

**ORCA2 pointer.** This is ORCA2 rung 1's exact program: ORCA2_OMIP_L4's compiled
`tradmp.f90`, `dtatsd.f90`, `fldread.f90` and `daymod.f90` are byte-identical
to the SMT-4 build's (`cmp` silent on all four), and the deck's namtsd/namtra_dmp
lines are rung 1's (namelist_cfg lines 49, 51, 55-56, 345-346). The one
regional difference is the resto field (rung 1: 691 Mediterranean/Red Sea
columns; SMT-5: every wet cell, Decision 107b). Every statement below carries
to the ORCA2 rung-1 walk unchanged.

## Round 2 — deck (SMT-5 vs SMT-4, namtsd and namtra_dmp lines only)

SMT-4 has neither block (both read namelist_ref defaults: OFF). The ONE added
hunk, applied after SMT-4's three patches:

```
+&namtsd
+   ln_tsd_init = .false.   (initial state stays usr_def_istate; D107 "else = SMT-4")
+   ln_tsd_dmp  = .true.    (ORCA2 rung 1 cfg:49)
+   cn_dir      = './'      (cfg:51)
+   sn_tem = 'data_1m_potential_temperature_nomask', -1., 'votemper', .true., .true., 'yearly', '', '', ''
+   sn_sal = 'data_1m_salinity_nomask'             , -1., 'vosaline', .true., .true., 'yearly', '', '', ''
+&namtra_dmp
+   ln_tradmp   = .true.    (cfg:345)
+   nn_zdmp     = 0         (cfg:346, D107c)
+   cn_resto    = 'resto.nc' (rung 1 resolves this reference default)
```

Run blocks unchanged: kt=1..10 (nn_itend = nn_stock = 10), 100 days
(nn_itend = 3000, nn_stock = 30), smoke 2 steps (nn_itend = nn_stock = 2, the
only smoke hunk). nn_fsbc = 1, so every length and cadence is a multiple; no
restart list is used (0 <= 10 entries). Preflight checks all of this
mechanically and prints `SMT5_DECK_CADENCE_OK` per arm.

## Round 2 — NEMO writes its own inputs (Decision 107e)

A card-local module, `vortex_smt5_target_dump.F90`, is called from
`usr_def_istate` right after T and S are set (additive patch, 3 added lines,
no shipped line changed). It writes, in the run directory: votemper and
vosaline as 12 identical records `(time_counter, z, y, x)` (ORCA2's names and
axis order), and `resto(z, y, x)` = tmask x (1/86400) s-1. Single rank,
interior points, all jpk levels, `NF90_NOCLOBBER`. Order is NEMO's own:
`istate_init` (`VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:399`) runs
before `tra_dmp_init` reads resto (`VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:432`,
`VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tradmp.f90:306`), and the target is
first opened by `fld_read` at kt = 1. Both builds carry the dump (it is a deck
input, not a record writer); after each run the driver checks the files'
CONTENT (names, axes, 12 identical double records, resto = tmask/86400, S =
35 tmask) and that both arms dumped byte-identical files.

| file (committed) | sha256 |
|---|---|
| vortex_smt5_target_dump.F90 | `50b8cc52cb0f1d2e431d3792bb867034f54fc240c413479fc3dfcd9597c54d86` |
| usrdef_istate_smt5_target_dump.patch (on shipped usrdef_istate.F90 `a2746520…6d4a39`) | `665614e7b3a432be372223274a9b7de5496a75c859e8549c0d3b2ba2e5379276` |
| namelist_cfg_smt5_tracer_damping.patch | `a562e3b21167ca997f945781967766f909ec47eab736081079e566c37326c446` |
| namelist_cfg_smt5_smoke_2step.patch | `604b96570d9f6cbbcc8838245d0c3cfb419371422efcca4d39a8f397d491d04e` |

The dumped files' own sha256 are written by the acquisition
(`smt5_dumped_inputs.sha256` in each evidence directory); none exist yet.

Shared driver: three new variants. `smt5vecsmoke` builds
VORTEX_SMT5_VEC_R8_OMIP_L1{,_P3} (the SMT-4 stage writer, unchanged) and runs 2
steps; `smt5vec` (kt=1..10) and `smt5vec100d` reuse those binaries, proved by
hash against the smoke manifest. Preflight: `bash -n` clean; all patches
`--fuzz=0` dry-run clean; `GFORTRAN_COMPILE_PASS` for the dump module and
`GFORTRAN_PATCHED_NEMO_SYNTAX_PASS usrdef_istate.f90` against the SMT-4 build's
module files; `SMTRUNGS_R2_SMT5_PREFLIGHT_PASS`. Existing variants' preflight
re-run after the edit: smt4vec, smt4vec100d, smt3vec, smtvec, flux all
`PREFLIGHT_OK`. makenemo/mpirun not run (operator's).

## Round 2 — the legoESM arm, statement by statement

All citations: SMT-4 build compiled source (identical to ORCA2's, above).

| NEMO statement | citation | legoESM |
|---|---|---|
| stage 3 only, after `tra_ldf`, before `tra_zdf` | `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:513`, `:526`, `:529`, `:538` | increment added to the stage-3 rate after the Redi (tra_ldf) term, inside the WS-RK3 tracer branch; stages 1-2 untouched |
| `Krhs += resto*(zts_dta - pts(Kbb))`, `nn_zdmp = 0` | `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tradmp.f90:190`, `:193-194` | `nemo_tra_dmp_rates` on the step-entry T/S |
| target at kt | `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tradmp.f90:181`, `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dtatsd.f90:212` | `nemo_fld_time_interpolate` |
| raw copy, then times tmask (z/zps) | `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dtatsd.f90:261`, `:308` | same order |
| `ztinta`, `ztintb = 1. - ztinta`, `fnow = ztintb*b + ztinta*a` | `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/fldread.f90:244`, `:245`, `:246` | same, each product materialised (no fused multiply-add) |
| after record = first centre >= time | `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/fldread.f90:309`, `:313` | searchsorted left; NaN outside the records |
| previous-year records before Jan 15.5 | `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/fldread.f90:281`, `:916` | timeline starts at year 0's January |
| record centre = integer rounded average | `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/fldread.f90:935` | Fortran truncating division + MOD |
| 365-day months | `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/daymod.f90:182` | other calendars raise |
| model time: half step before nit000, + ndt each step, + 0 for kn_fsbc = 1 | `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/daymod.f90:92`, `:140`, `:240`, `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/fldread.f90:205` | isecsbc = NINT(0.5 rn_Dt) + elapsed seconds at the step's start (1440 s at kt = 1) |

The model-time formula is MEASURED against a NEMO record, not only read: ORCA2
rung 1's ocean.output prints kt = 1 at 0.0625 days and kt = 2 at 0.1875 days
for rn_Dt = 10800 (half a step, then + one step), records b/a 12/1 at
-15.5/+15.5 days; the new function reproduces that weight (249/496, the value
the ORCA2 card already pins) bit for bit.

Selection: new config field `nemo_tracer_damping`, default None (OFF). Caller
grep: `grep -rn "nemo_tracer_damping=" packages/ src/ scripts/` returns ONE
line, the SMT-5 card builder. The card validator refuses the field on any
other card and requires it on VORTEX_SMT5_VEC-zps; the model refuses it off
the WS-RK3 momentum+tracer program. The card reads NEMO's dumped files
(`deck_root` required) and refuses a resto that is not tmask/86400 or records
that differ.

Why the existing mechanisms are NOT reused (left untouched): the sponge
(`SpongeForcing`) enters the generic explicit tendency with no RK3 stage
placement and a fixed target with no time interpolation; surface restoring
acts on the top layer only. Neither is NEMO's statement.

## Tests (focused, fp64, CPU)

New file `tests/ocean/fidelity/test_nemo_testcase_l1_vortex_smt5_tracer_damping.py`:
10 passed. It covers record centres; the ORCA2 kt = 1 weight; NaN outside the
records; the rate equal to resto*(T_dta - T_Kbb) bit for bit on a synthetic
column at three model times; refusal without model time; the SMT-5 config =
SMT-4 config plus the field; validator both directions; and the PRODUCTION
placement: the stage-3 rate the tracer program consumes equals the rate before
it plus resto*(T_dta - T_Kbb) bit for bit, the increment is nonzero on every
wet cell (target offset +0.25 K so the plant is not perturbing a zero), the
stage-1 tracer equals SMT-4's bit for bit, and the stage-1 PLANT breaks both
(the stage-3 identity raises; the stage-1 tracer moves). Instrument test 2
passed; `tests/ocean/unit/test_nemo_fld_read.py` 4 passed.

Closed cards bit-identical, MEASURED: state sha256 (u, v, T, S, ssh) after 3
production steps at the parent code vs after the arm, on GYRE, VORTEX,
VORTEX_VEC, SMT, SMT_VEC, SMT-1..4: `CLOSED_CARDS_BIT_IDENTICAL` on all nine
(evidence `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/round2/`
baseline_{a,b,c}.json vs after_{a,b,c}.json; sha256 of baseline_a
`96c383d4…a6e06e`, after_a `d6238789…5e02`). The CI test of that script proves
only that the digest separates two states; the bit-identity is this
measurement.

Pre-existing reds, unchanged by this diff (verified at the parent in a
separate worktree): the VORTEX card-digest pins for GYRE, LOCK_EXCHANGE and
OVERFLOW (already stale at the parent; this change moves them only by the
added None field — the field-stripped digests equal the parent's exactly), a
hard-coded-constant hit in an unrelated JRA55 test, the round-51 trace-field
order test, and the federation-plan "taxonomy" module.

## Choices made this round

| choice | status |
|---|---|
| deck values (target, resto, nn_zdmp, 12 records, namtsd settings, NEMO dump) | ASKED (Decision 107) |
| ln_tsd_init = .false. stated explicitly | ASKED (Decision 107: everything else = SMT-4) |
| target files written as double, not ORCA2's float32 | UNASKED — follows from D107a (target = the analytical state itself; float32 would differ by ~1e-7 relative). Revert = NF90_FLOAT in the dump |
| NEMO writes the inputs inside the production run (no separate pre-run) | UNASKED mechanism for D107e — revert = a 1-step pre-run arm |
| test-only hooks (stage-1 plant, stage-3 exposure) | exempt (tests) |

## Review, citations

Single review (codex), verdict before fixes: "DO NOT SHIP", 3 findings, all
CONFIRMED, all acted on: (1) admission checked only that the dumped files
exist — now checks their content, and a planted T/S swap fails the check;
(2) the closed-card test cannot detect a trajectory change — correct, it is
labelled an instrument test and the bit-identity is quoted as a measurement;
(3) Python half-even round used for NINT — replaced by Fortran NINT (inert at
rn_Dt = 2880).

Citation gate: run from this receipt's "## Round 2 —" heading; result and
shifted-citation plant quoted in the commit that adds this receipt.

## OPEN for round 3

1. Operator runs `run.sh --smoke` then `--run`; score P1-P7 of the prereg.
2. The 100-day and kt walk harnesses must pass `t_seconds` (elapsed seconds at
   step start) for the SMT-5 card; the model refuses to step it without.
3. ORCA2 rung 1 can reuse the record-centre/weight functions in place of its
   hard-coded 249/496 (one line; not done here).

UNVERIFIED: the NEMO side (dump, deck, variants) is proved by syntax compile and
preflight only, not by a run; netCDF dimension naming is assumed to be
irrelevant to iom_get (ORCA2's files are read by the same routine; the smoke run
decides); XLA is assumed not to re-fuse the materialised products.
