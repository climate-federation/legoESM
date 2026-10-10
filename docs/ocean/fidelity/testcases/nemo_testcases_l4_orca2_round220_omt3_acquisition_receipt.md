# ORCA2 round 220 — OMT-3 lateral-diffusion acquisition handoff

Date: 2026-10-10. Frozen base: `fee85863a`. Preregistration commit:
`1909548b8`. Acquisition implementation commit: `f6d92b9ae`. Status:
**STOPPED_FOR_RECORD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round220/`.

This round changes no package model file, shipped ORCA2 card, carried state,
stabiliser, sea-ice selector, or `unmeasured_features` tuple. It creates the
Decision-109 OMT-3 deck and a fail-closed operator acquisition. No NEMO run was
attempted in the sandbox. OMT-3 trajectories are therefore **UNMEASURED WITH
SPEC**, and no independent/given-entry numbers are mixed or reported.

## Exact OMT-2 to OMT-3 edge

The canonical deck is rendered from the admitted round-218 OMT-2 deck,
SHA-256 `bc5fc68188e4d79479acce38085e497f9f2543a36d6544f1dc1cea95bf51d169`.
The gate reports exactly one changed assignment and one removed override:

| assignment | admitted OMT-2 | OMT-3 | provenance |
|---|---:|---:|---|
| `namdyn_ldf.ln_dynldf_lap` | `.false.` | `.true.` | rung-0 shipped module |
| `namdyn_ldf.ln_dynldf_off` | explicit `.true.` | absent, resolving `.false.` from `namelist_ref` | rung-0 shipped module |

Vector momentum advection, linear implicit bottom drag, tracer-advection OFF,
and tracer-diffusion OFF remain unchanged. The rendered deck SHA-256 is
`a4c023c1b9a667cb0ba6ea839b62fe38059cb4050e8cec9a057fd710c6382715`.
The deck preflight JSON SHA-256 is
`02b9950ef2ae6d12f77586955b41c04a195b135ee2ae6b4fe3a4974e05085760`.
Both extra-delta and wrong-selector plants exit nonzero.

The compiled build reads both namelists at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:177-184`, requires one
and only one OFF/laplacian/bilaplacian selector at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:221-228`, and maps the
partial-step iso-level laplacian at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:250`. It allocates
the viscosity arrays only when diffusion is active at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:297-305`.

The stage program calls momentum LDF at stage 3,
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:488-493`; the
dispatcher selects the iso-level laplacian at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynldf.f90:81-85`; and the live
div-curl statement is
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynldf_lev.f90:123-140`.

Search-before-build found and reused the round-218 OMT-2 binary-reuse
launcher, deck gate, and rank-complete frame gate. No numerical implementation
or NEMO source patch was added.

## Acquisition contract

The launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round220_omt3_frames_acquisition/run.sh`.
It reuses the admitted uninstrumented binary SHA-256
`c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343`
and additions-only P3 binary SHA-256
`5b82a3254c40f71186af159b93cba419709ccf49cf4172ad3d44440b8fb1d895`;
it does not call `makenemo`.

The committed run order is:

1. legal two-step smoke with `nn_fsbc=2`, `nn_stock=2`;
2. uninstrumented ten-step calibration;
3. two independent rank-complete P3 twins through kt=10;
4. uninstrumented 96-step boundary run with the compiled ten-entry restart
   list `10,20,...,90,95`.

Admission requires 80 self-described frames per twin, both ranks, four stage
boundaries, five fp64 fields, exact twin equality, and byte-identical kt=10
terminal restarts against the uninstrumented calibration. The month either
completes kt=96 or records an exact compiled `stp_ctl` boundary strictly after
kt=10; an earlier boundary refuses the record. The frame preflight JSON
SHA-256 is
`7656d30171d250fd8ae5aa4dc081a74ed9b7bd1baab0b2fe034bbdff1d2a550f`;
the full preflight log SHA-256 is
`d8f90cf52474e1f7a3a3a373a1bfc6e31d77e425a9f93b3988e731a39d5a184e`.

## Preregistered predictions

| ID | disposition |
|---|---|
| R220-P1 | **CONFIRMED**: exact one-module deck edge; both deck plants fire. |
| R220-P2 | **UNMEASURED WITH SPEC**: ten-step NEMO calibration/twins await the operator. |
| R220-P3 | **UNMEASURED WITH SPEC**: 96-step completion awaits the operator; an earlier compiled boundary will remain a refutation. |
| R220-P4 | **UNMEASURED WITH SPEC**: the complete vector unit cannot be scored without the OMT-3 record. |
| R220-P5 | **PARTLY CONFIRMED**: deck and inventory plants fire in preflight; record-content and future trajectory plants await real output. |

## Validation and review

Focused round-220 plus citation-gate tests pass 25/25 (log SHA-256
`0190cd2518229d8d336254c25b7fefb35dddec933048ebbbac243eb306aabc76`).
Independent review was attempted separately
with `codex exec --sandbox read-only` and exited 1 before reading the diff:
`failed to initialize in-process app-server client: Read-only file system (os
error 30)`. Independent review is unavailable in-sandbox; this is not a PASS.
The review log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

On clean receipt/map commit `c0cf865f7`, the round citation gate passes seven
citations with zero failures, unmapped entries, or map-audit failures; the
cumulative default gate passes 274 citations with the same zero counts. The
explicit two-line shift plant on
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynldf.f90:81-85` fails with one
finding.
Round/default/plant JSON SHA-256 values are
`dbd9ba210ae1a168d07df8e84f0cc8f872653a0b0ca47ba138c0dfcb30f78625`,
`822bcba7de0054e3d9c38d7243faed26a6d9db1c12662d433526289aeaac2bbb`,
and `e36a4504993036b997210afb2e3c04bd21f342de09f98895028211b916e300f9`.

The one prescribed `tests/ocean/fidelity -n 12` battery collected 3,068
tests and reached 99% before a bounded interrupt: 3,039 PASS, seven SKIP, four
registered pre-existing FAIL lines, and 18 tests unclassified. It is not called
PASS. The four failures are the GYRE round-129 spread-floor record stamp,
allow-dirty scope, worktree-stamp ratchet, and SI3 scalar-math provenance gate.
The unfinished `test_prediction_plant_is_fail_closed` was replayed alone and
timed out without a verdict at 15 minutes; it is unrelated to this round's
files. Full-battery and isolated logs have SHA-256
`b3d3ad503503f118a726137c3f0ad05ccd9612160a0f081abc28f5beab2543f5`
and `fd01b65d6fe039cabb36640d3f81768dfec661d3b190e4e91c26434bc4f24ed0`.

## OPEN

1. The operator runs the committed launcher with `--run`. The expected target
   is
   `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round220/acquisition/orca2_omt3_frames_10step_a_np2`.
2. Admit the record and disposition R220-P2/P3. For a checker mismatch, repair
   the checker against the self-described record and use `--admit-existing`;
   never rerun NEMO for a check-only refusal.
3. Build the gate-local OMT-3 card, run both labelled ladders, and score the
   complete round-217 vector unit atomically. If the kt=8 live-W refusal first
   returns here, walk lateral momentum diffusion to its first cited non-bit
   statement before OMT-4. If it remains clean, OMT-4 is next.
4. Do not promote any constituent of the unit or touch the shipped rung-10
   card. Sea ice and its `unmeasured_features` tuple remain unchanged.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
