# ORCA2-DECKS round 19 receipt — rung 1 admitted, NEMO-side hierarchy complete

Date: 2026-10-02

Disposition: **LANDED**.  Decision 83's replacement rung-1 independent record
is admitted.  Every NEMO-side rung 1 through 10 now has its authorized deck,
two-rank self-describing operand record, 240-step from-rest output, terminal
restart, and checksum inventory.  The rung-1/main-rung-0 physical boundary is
exactly interior T/S damping; the remaining rows are run protocol and names of
independently exact-zero files.

Base: `5dd16b6f3`.  Preregistration: commit `929151958`, before this round's
fresh record validation or hierarchy inventory.  Completion gates: commits
`1f5b81f56`, `ef72d59c0`, and `90cc1f754`.  Every run claim below is
**independent**: NEMO starts from the rung's own from-rest initialization.

## Rung-1 admission

The operator ran round 18's committed launcher at producer commit
`5dd16b6f3` and it reached `STOP 0`, step 240, clean admission, and
`ORCA2_HIERARCHY_RUNG1_HAVTB0_ACQUISITION_PASS`.  Round 19 then independently
reran all 28 inherited record/deck plants; every plant printed
`STATUS PLANT-FIRED`.  Fresh clean validation reproduced the canonical
scientific payload exactly after excluding only the worktree stamp and printed
`PASS_RUNG1_HAVTB0_RECORD`.

The admitted record contains:

- 480 self-describing frames: 3,840 finite PRESENT payloads and 960 ABSENT
  payloads, exactly the owner-off runoff fields;
- one exact-zero fp64 five-field surface input on the 148x180 grid;
- eight finite T/U/V/W month files with 86 floating variables;
- two finite fp64 step-240 ocean restart shards and no ice product; and
- 549 regular files in a complete SHA-256 inventory.

Admission SHA-256 is
`f28561d93512774304cef16f6b1031c760156fb0e73385e472ed5da9f57c5b26`;
record-inventory SHA-256 is
`5efea04fc4d6c6ebe34f87fb89c1f9ef928e44a12e23b329f1f1f978d3893e77`.
HD19-P1 and HD19-P2 are **CONFIRMED**.

## Compiled branch and Decision-83 replacement

NEMO reads `nn_havtb` with the vertical-physics namelist, initializes the
background multiplier uniformly, enters the equatorial shaping arm only for
selector value one, and applies the multiplier to tracer diffusivity
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`,
`:205-228`).  The replacement rung-1 deck and preserved value-1 evidence
differ on exactly one parsed and physical line:

```diff
-   nn_havtb    =    1         !  horizontal shape for avtb (=1) or not (=0)
+   nn_havtb    =    0         !  horizontal shape for avtb (=1) or not (=0)
```

The completion gate pins both admissions and both inventories and refuses if
the preserved `record_havtb1_superseded` evidence disappears.  HD19-P3 is
**CONFIRMED**.

## Complete adjacent-rung namelist diff

The completion gate parses every current exact deck and requires these to be
all differing assignments.  Values are upper rung to lower rung.

| edge | assignment | upper | lower |
|---|---|---|---|
| 10 -> 9 | `namsbc.nn_ice` | `2` | `0` |
| 9 -> 8 | `namsbc_rnf.ln_rnf_mouth` | `.true.` | `.false.` |
| 9 -> 8 | `namzdf.ln_zdfddm` | `.true.` | `.false.` |
| 9 -> 8 | `namzdf_iwm.ln_tsdiff` | `.true.` | `.false.` |
| 8 -> 7 | `namzdf.ln_zdfiwm` | `.true.` | `.false.` |
| 7 -> 6 | `namzdf.ln_zdfcst` | `ABSENT` | `.true.` |
| 7 -> 6 | `namzdf.ln_zdftke` | `.true.` | `.false.` |
| 7 -> 6 | `namzdf.nn_havtb` | `1` | `0` |
| 6 -> 5 | `namsbc.ln_rnf` | `.true.` | `.false.` |
| 5 -> 4 | `namsbc.ln_traqsr` | `.true.` | `.false.` |
| 5 -> 4 | `namtra_qsr.ln_qsr_rgb` | `.true.` | `.false.` |
| 5 -> 4 | `namtra_qsr.nn_chldta` | `1` | `0` |
| 4 -> 3 | `namsbc.ln_abl` | `ABSENT` | `.false.` |
| 4 -> 3 | `namsbc.ln_blk` | `.true.` | `.false.` |
| 4 -> 3 | `namsbc.ln_cpl` | `ABSENT` | `.false.` |
| 4 -> 3 | `namsbc.ln_dm2dc` | `ABSENT` | `.false.` |
| 4 -> 3 | `namsbc.ln_flx` | `ABSENT` | `.true.` |
| 4 -> 3 | `namsbc.ln_mixcpl` | `ABSENT` | `.false.` |
| 4 -> 3 | `namsbc.ln_ssr` | `.true.` | `.false.` |
| 4 -> 3 | `namsbc.ln_usr` | `ABSENT` | `.false.` |
| 4 -> 3 | `namsbc.nn_fwb` | `2` | `0` |
| 4 -> 3 | `namsbc_flx.cn_dir` | `ABSENT` | `'./'` |
| 4 -> 3 | `namsbc_flx.sn_emp` | `ABSENT` | `'rung3_zero_flux', -12., 'emp', .false., .true., 'yearly', '', '', ''` |
| 4 -> 3 | `namsbc_flx.sn_qsr` | `ABSENT` | `'rung3_zero_flux', -12., 'qsr', .false., .true., 'yearly', '', '', ''` |
| 4 -> 3 | `namsbc_flx.sn_qtot` | `ABSENT` | `'rung3_zero_flux', -12., 'qtot', .false., .true., 'yearly', '', '', ''` |
| 4 -> 3 | `namsbc_flx.sn_utau` | `ABSENT` | `'rung3_zero_flux', -12., 'utau', .false., .true., 'yearly', '', '', ''` |
| 4 -> 3 | `namsbc_flx.sn_vtau` | `ABSENT` | `'rung3_zero_flux', -12., 'vtau', .false., .true., 'yearly', '', '', ''` |
| 4 -> 3 | `namsbc_ssr.ln_sssr_bnd` | `.true.` | `.false.` |
| 3 -> 2 | `namtra_eiv.ln_ldfeiv` | `.true.` | `.false.` |
| 3 -> 2 | `namtra_mle.ln_mle` | `.true.` | `.false.` |
| 2 -> 1 | `nambbc.ln_trabbc` | `.true.` | `.false.` |
| 2 -> 1 | `nambbl.ln_trabbl` | `.true.` | `.false.` |

Decision 83 deliberately makes the 7 -> 6 TKE boundary include the shipped
background shape alongside the closure switch.  The BBL initializer returns
before allocation when its selector is false, and the geothermal initializer
allocates and reads its input only when its selector is true
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/trabbl.f90:540-564`,
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/trabbc.f90:200-251`).
The stage calls are guarded by the same selectors
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:430-458`,
`:523-529`).

Every rung retains `namagrif.ln_spc_dyn=.true.`.  The reused compiled build has
no `key_agrif` (`cpp_ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE.fcm:1`), so the line is
retained but inert.  HD19-P4 is **CONFIRMED**.

## Rung-1 to main-rung-0 boundary

The pinned main-lane rung-0 deck differs from rung 1 on exactly these rows:

| class | assignment | rung 1 | rung 0 |
|---|---|---|---|
| damping | `namtra_dmp.ln_tradmp` | `.true.` | `.false.` |
| damping | `namtsd.ln_tsd_dmp` | `.true.` | `.false.` |
| protocol | `namrun.ln_rst_list` | `ABSENT` | `.true.` |
| protocol | `namrun.nn_itend` | `240` | `10` |
| protocol | `namrun.nn_stock` | `240` | `1` |
| protocol | `namrun.nn_stocklist` | `ABSENT` | `1,2,3,4,5,6,7,8,9,10` |
| zero-file name | `namsbc_flx.sn_emp` | `rung3_zero_flux` | `rung0_zero_flux` |
| zero-file name | `namsbc_flx.sn_qsr` | `rung3_zero_flux` | `rung0_zero_flux` |
| zero-file name | `namsbc_flx.sn_qtot` | `rung3_zero_flux` | `rung0_zero_flux` |
| zero-file name | `namsbc_flx.sn_utau` | `rung3_zero_flux` | `rung0_zero_flux` |
| zero-file name | `namsbc_flx.sn_vtau` | `rung3_zero_flux` | `rung0_zero_flux` |

The gate requires each zero-file operand to become identical after replacing
only its filename.  No other physical assignment differs.  The hierarchy's
rung-1 -> rung-0 module boundary is therefore exactly the interior T/S damping
specified by Decision 80.

## Complete admitted hierarchy

| rung | canonical status | frames | payload form | `nn_havtb` | inventory files |
|---:|---|---:|---|---:|---:|
| 1 | `PASS_RUNG1_HAVTB0_RECORD` | 480 | 3,840 PRESENT / 960 ABSENT | 0 | 549 |
| 2 | `PASS_RUNG2_HAVTB0_RECORD` | 480 | 3,840 PRESENT / 960 ABSENT | 0 | 545 |
| 3 | `PASS_RUNG3_HAVTB0_RECORD` | 480 | 3,840 PRESENT / 960 ABSENT | 0 | 542 |
| 4 | `PASS_RUNG4_HAVTB0_RECORD` | 480 | 3,840 PRESENT / 960 ABSENT | 0 | 541 |
| 5 | `PASS_RUNG5_HAVTB0_RECORD` | 480 | 3,840 PRESENT / 960 ABSENT | 0 | 536 |
| 6 | `PASS_RUNG6_HAVTB0_RECORD` | 480 | 4,800 finite PRESENT | 0 | 540 |
| 7 | `PASS_RUNG7_RECORD` | 480 | 4,800 finite PRESENT | 1 | 538 |
| 8 | `PASS_RUNG8_RECORD` | 480 | 4,800 finite PRESENT | 1 | 538 |
| 9 | `PASS_RUNG9_RECORD` | 480 | 4,800 finite PRESENT | 1 | 534 |
| 10 | `PASS_RUNG10_RECORD` | 480 | 4,800 finite PRESENT; 200/200 kt1-10 comparisons bit-exact | 1 | 539 |

All ten admissions say `claim_label=independent`, all month products are
finite, and every inventory is complete.  The completion gate pins every
admission and inventory hash, requires all six superseded value-1 records to
remain present, validates rung 1 afresh, and prints
`PASS_ORCA2_NEMO_HIERARCHY_RUNGS_1_10`.  Its seven completion-layer plants all
fire.  HD19-P5 is **CONFIRMED**.

## Gates, tests, review, and scope

- Rung 1: all 28 inherited plants fire; clean fresh validation passes.
- Completion: all seven new plants fire; clean hierarchy gate passes.  The
  clean evidence JSON SHA-256 is
  `757a913b90ec478b659ca1582561d677c183feed25232c48510933b9c844de08`.
- The receipt citation gate passes all seven compiled-source citations with
  zero failures and zero unmapped citations; the shifted `zdfphy` span plant
  fires.  The cumulative default receipt gate also passes with zero failures
  and zero unmapped citations.
- Complete hierarchy battery: **507 passed** in 53.01 s.
- The single permitted `tests/ocean/fidelity -n 12` battery selected 2,684
  tests and reached 99%.  Its only visible failure was the declared
  pre-existing SI3 scalar-math provenance gate; the known late-suite no-output
  tail produced no terminal summary and was interrupted.  No second broad
  battery ran, so this battery is not claimed PASS.
- Separate `codex exec --sandbox read-only` review returned before reading the
  diff: `failed to initialize in-process app-server client: Read-only file
  system`.  Verdict: **independent review unavailable in-sandbox**.
- No file under `packages/` or `src/` changed.  No NEMO source or record was
  modified in this round.  GYRE is byte-identical by construction; its year
  gate was not rerun.

ASKED choices: Decisions 80 and 83 define every deck boundary and assign
`nn_havtb=0` to rungs 0 through 6.  UNASKED choices: none.

## Prediction ledger

| prediction | status |
|---|---|
| HD19-P1 rung-1 plants | **CONFIRMED**; 28/28 print `STATUS PLANT-FIRED` |
| HD19-P2 rung-1 admission | **CONFIRMED**; fresh complete record validation passes |
| HD19-P3 authorized replacement | **CONFIRMED**; one line, preserved value-1 evidence pinned |
| HD19-P4 rung boundaries | **CONFIRMED**; every adjacent diff and the rung-0 boundary exact |
| HD19-P5 hierarchy completion | **CONFIRMED**; ten canonical independent records admitted |

## OPEN

1. Rung-0 card construction and cross-model scoring remain main-lane work and
   are outside this NEMO-only side lane.
2. The broad ocean-fidelity battery has no terminal summary; this does not
   block the record-only completion, but it is not a green-suite claim.

## UNVERIFIED

- No legoESM card has been built or scored here; this lane was NEMO-side only.
- The broad ocean-fidelity battery did not produce a terminal summary.
