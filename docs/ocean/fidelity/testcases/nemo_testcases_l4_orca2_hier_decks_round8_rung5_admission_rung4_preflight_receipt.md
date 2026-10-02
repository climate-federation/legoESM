# ORCA2-DECKS round 8 receipt — rung 5 admitted, rung 4 preflight

Date: 2026-10-01

Disposition: **STOPPED_FOR_RECORD**.  The round-7 admission crash is repaired,
the existing operator-produced rung-5 record is admitted without rerunning
NEMO, and the rung-4 deck and acquisition are preflight-clean.  The rung-4
oracle record does not yet exist.

Base: `a4df4b624`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round8.md`, commit `ebb8c46f9`,
before the admission repair, rung-5 measurement, or rung-4 source reading.
Every run claim is labelled **independent**: NEMO starts from that rung's own
from-rest initialization.

## Rung inventory

| rung | NEMO deck | independent record | status / unmeasured specification |
|---:|---|---|---|
| 10 | shipped one-category ocean-ice deck | 480 frames + 240-step month | **ADMITTED** round 2 |
| 9 | rung 10 without ice | 480 frames + 240-step month | **ADMITTED** round 3 |
| 8 | rung 9 without DDM, river-mouth diffusivity, differential T/S mixing | 480 frames + 240-step month | **ADMITTED** round 4 |
| 7 | rung 8 without internal-wave mixing/background reset | 480 frames + 240-step month | **ADMITTED** round 5 |
| 6 | rung 7 with constant mixing replacing TKE | 480 frames + 240-step month | **ADMITTED** round 6 |
| 5 | rung 6 without runoff | 480 version-2 frames + 240-step month | **ADMITTED** this round |
| 4 | rung 5 without penetrative chlorophyll shortwave | absent | **STOPPED_FOR_RECORD**; needs 480 header-valid frames, finite T/U/V/W month files, and finite fp64 step-240 ocean restarts |
| 3 | not built; rung 4 minus bulk/restoring/freshwater budget | absent | **UNMEASURED WITH SPEC**; clean unforced formulation must be source-resolved |
| 2 | not built; rung 3 minus GM/MLE | absent | **UNMEASURED WITH SPEC** |
| 1 | not built; rung 2 minus BBL/geothermal | absent | **UNMEASURED WITH SPEC** |
| 0 | main-lane owned | not compared | **OUT OF SCOPE** |

Every rung retains `ln_spc_dyn=.true.` exactly as the source deck carries it;
the reused binary has no `key_agrif`, so the assignment remains inert.

## Loud correction and rung-5 admission

Round 7's first admission plant did not test its planted field name.  It
crashed first because `validate_calibration` passed the deleted keyword
`tke_active` to the rung-6 validator.  The rung-6 validator already encodes
constant mixing and TKE-off checks.  The repair deletes only that keyword and
a regression test parses the call to require the current zero-keyword
interface.

The fixed field-name plant reaches the self-describing parser and fires.  All
sixteen admission plants fire, then the clean gate reports
`PASS_RUNG5_ABSENT_RECORD`.  The independent rung-5 record contains:

- 480/480 frames, 3,840 finite PRESENT fields, and 960 ABSENT fields;
- exactly `rnf` and `rnf_tsc` ABSENT in every frame;
- two finite fp64 step-240 ocean restart shards and no ice product;
- eight finite T/U/V/W month files with 92 floating variables; and
- a complete 536-regular-file SHA-256 inventory.

The repaired recorder is passive on active runoff: all 480 rung-6 calibration
frames and both terminal restart files are byte-identical to the admitted
rung-6 record.  HD8-P1 through HD8-P4 are **CONFIRMED**.

## Compiled rung-4 shortwave boundary

NEMO reads the reference `namtra_qsr` group first and the configuration group
second, so omitted values inherit reference defaults
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/traqsr.f90:1101-1103`).
Those defaults disable `ln_traqsr`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/EXP00/namelist_ref:202`) and disable RGB,
two-band, five-band, and bio penetration while setting `nn_chldta=0`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/EXP00/namelist_ref:424-431`).

There is therefore **no two-band default**.  When penetration is enabled, the
compiled initializer counts all four scheme switches, refuses unless exactly
one is selected, and maps a selected two-band switch explicitly to `np_2BD`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/traqsr.f90:1123-1137`).
The preregistered alternative “leave `ln_traqsr` true and inherit two-band” is
**REFUTED**.  It would stop in initialization.

Rung 4 thus disables the whole penetrative-shortwave module.  The stage call is
guarded by `ln_traqsr`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:525`).
With it false, NEMO adds `qsr` to `qns` and zeros `qsr` at stage 1
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/trasbc.f90:267-270`), and
sets `fraqsr_1lev` to one
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcssm.f90:284`).

## Complete rung-4 namelist delta

The complete physical-line diff from rung 5 is one module and three explicit
assignments:

```diff
-   ln_traqsr   = .true.    !  Light penetration in the ocean            (T => fill namtra_qsr)
+   ln_traqsr   = .false.   !  Light penetration in the ocean            (T => fill namtra_qsr)
-   ln_qsr_rgb  = .true.       !  RGB light penetration (Red-Green-Blue)
+   ln_qsr_rgb  = .false.      !  RGB light penetration (Red-Green-Blue)
-   nn_chldta   =      1       !  RGB : Chl data (=1) or cst value (=0)
+   nn_chldta   =      0       !  RGB : Chl data (=1) or cst value (=0)
```

These are physical lines 90, 137, and 139 of the generated deck.  The exact
deck SHA-256 is
`96796ce98339c744f2d488cc8bbc962023ca769877de82a1ee7ee7e579b0a14c`.
All other parsed assignments, CPP keys, inputs, repaired record binary,
from-rest mode, two-rank layout, 240-step length, output cadence, and the
retained unread TKE sentinel are unchanged.

The preflight gate pins the compiled sources, reference namelist, repaired
binary, admitted upper-rung record, exact three-assignment delta, and full
selector-file census.  Four plants fire: unrelated deck change, missing
penetration selector, invented two-band fallback, and changed build pin.
HD8-P5 is **CONFIRMED**.

## Validation, review, and scope

- Focused round-7/round-8 controls: **17 passed**.
- Ruff passes the new gate/test; the launcher passes `bash -n` and reports
  `ORCA2_HIERARCHY_RUNG4_PREFLIGHT_READY`.
- The required separate `codex exec --sandbox read-only` review failed before
  reading the diff: `failed to initialize in-process app-server client:
  Read-only file system`.  Verdict: **independent review unavailable
  in-sandbox**.
- No file under `packages/` or `src/` changed.  GYRE is byte-identical by
  construction; its year gate was not rerun.
- No NEMO source tree was modified.  The launcher reuses the admitted repaired
  binary and requests no rebuild.

ASKED choices: rung ordering and source-decided non-penetrative behavior are
fixed by Decisions 79/80 and side-lane note B23.  UNASKED choices: none.

## OPEN

1. The operator runs the committed rung-4 launcher.  Rung 4 remains
   UNMEASURED until all fifteen admission plants fire and the clean gate
   admits 480 frames, the month products, and terminal restarts.
2. After rung 4 admits, source-resolve rung 3's zero-surface-forcing
   formulation.  If neither `ln_usr` nor `ln_flx` is side-effect-free, stop for
   `DECISION_NEEDED` rather than selecting one.
3. Rungs 2 and 1 remain unbuilt; main-lane rung 0 remains out of scope.

Acquisition launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round8_acquisition/run.sh`.

Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung4/record`.

## UNVERIFIED

- The rung-4 NEMO run has not been executed.
- Its 480-frame census, month products, terminal restarts, and SHA-256
  admission are unmeasured.
