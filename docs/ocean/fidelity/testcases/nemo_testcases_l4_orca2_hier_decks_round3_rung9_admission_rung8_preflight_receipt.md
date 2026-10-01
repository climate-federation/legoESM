# ORCA2-DECKS round 3 receipt — rung 9 admitted, rung 8 mixing-off preflight

Date: 2026-10-01

Base: `86a2b4e5c`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round3.md` (committed as
`3ac7fe772` before rung-9 admission and rung-8 construction).

Disposition: **STOPPED_FOR_RECORD**.  The operator-produced rung-9 no-ice
record is admitted under round 2's frozen predicates.  The rung-8 deck and
acquisition pipeline are preflight-clean, but no rung-8 oracle record exists.

Every run claim is labelled **independent**: NEMO starts from that rung's own
from-rest T/S initialization.  No given-entry result appears in this receipt.

## Rung inventory

| rung | NEMO deck | record | status / unmeasured specification |
|---:|---|---|---|
| 10 | exact shipped one-category ocean-ice deck | 480 operand frames + 240-step month | **ADMITTED** in round 2 |
| 9 | rung 10 with only `nn_ice: 2 -> 0` | 480 operand frames + 240-step month | **ADMITTED** this round |
| 8 | rung 9 with double diffusion, river-mouth diffusivity, and differential internal-wave T/S mixing off | absent | **STOPPED_FOR_RECORD**; needs 480 header-valid finite frames, finite T/U/V/W month files, and finite fp64 step-240 ocean restarts |
| 7 | not built; rung 8 minus internal-wave mixing and its background reset | absent | **UNMEASURED WITH SPEC** |
| 6 | not built; rung 7 replaces TKE by GYRE constant mixing | absent | **UNMEASURED WITH SPEC** |
| 5 | not built; rung 6 minus runoff | absent | **UNMEASURED WITH SPEC** |
| 4 | not built; rung 5 minus RGB shortwave | absent | **UNMEASURED WITH SPEC**; compiled fallback must be cited before construction |
| 3 | not built; rung 4 minus bulk/restoring/freshwater budget | absent | **UNMEASURED WITH SPEC**; clean unforced formulation must be cited before construction |
| 2 | not built; rung 3 minus GM/MLE | absent | **UNMEASURED WITH SPEC** |
| 1 | not built; rung 2 minus BBL/geothermal | absent | **UNMEASURED WITH SPEC** |
| 0 | main-lane owned | not visible at this side-lane tip | **OUT OF SCOPE**; compare against rung 1 when available |

Every rung retains `ln_spc_dyn=.true.` exactly as the source deck carries it.
The reused binary still has no `key_agrif`, so the assignment remains inert.

## Rung-9 admission

The committed round-2 gate passes on the operator record, and all nine
record-stage plants have `STATUS PLANT-FIRED`.  Admission reports:

- 480/480 self-describing rank-step frames and 4,800 finite named payloads;
- two finite fp64 step-240 ocean restart shards and no ice product;
- eight finite T/U/V/W month output shards containing 96 floating variables;
- resolved from-rest, `nn_ice=0`, `nn_mxlice=0`, implicit ice drag false,
  ocean-only freshwater-budget mode, no SI3 initialization, and the committed
  invalid ice-namelist sentinel unread; and
- a complete 534-file SHA-256 record inventory.

HD2-P3, HD2-P4, and HD2-P5 are therefore **CONFIRMED**.  The record producer
is `86a2b4e5c22eab5f33207f48f2b52167b615e22d`; binary, deck, manifest, and
run protocol equal the round-2 pins.

## Complete rung-8 namelist delta

The complete physical-line diff from rung 9 is:

```diff
-   ln_rnf_mouth = .true.    !  specific treatment at rivers mouths
+   ln_rnf_mouth = .false.   !  specific treatment at rivers mouths
-   ln_zdfddm   = .true.    ! double diffusive mixing
+   ln_zdfddm   = .false.   ! double diffusive mixing
-   ln_tsdiff   = .true.    !  account for differential T/S mixing (T) or not (F)
+   ln_tsdiff   = .false.   !  account for differential T/S mixing (T) or not (F)
```

These are physical lines 162, 394, and 425 of the generated ocean namelist.
The complete parsed assignment maps differ on exactly those three fully
qualified keys.  The rung-8 ocean namelist SHA-256 is
`6d9679c9ce9cae41c36a6d0a3dd4c996921302fba677b36975452c9a8b99656d`.
The retained ice namelist, binary, CPP keys, inputs, from-rest mode, two-rank
layout, 240-step length, output cadence, and all other ocean assignments are
unchanged.

The four subordinate values remain byte-identical and are gated:

| retained assignment | rung 9 | rung 8 disposition |
|---|---:|---|
| `rn_hrnf` | 15 m | inert because the river-mouth selector is false |
| `rn_avt_rnf` | 1e-3 m2/s | inert because the river-mouth selector is false |
| `rn_avts` | 1e-4 m2/s | inert because double diffusion is false |
| `rn_hsbfr` | 1.6 | inert because double diffusion is false |

## Compiled branch and resolved predictions

The vertical-physics manager reads the double-diffusion selector and its two
parameters together at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`.
Its resolved report distinguishes the selected and unselected arms at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:257-258`.
At execution, false skips `zdf_ddm` and copies `avs=avt` before wave mixing;
the same block also shows the river-mouth increment and retained internal-wave
call at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:353-372`.
The implicit tracer solver consequently builds a separate salinity matrix only
when double diffusion is true, at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/trazdf.f90:175-180`.

The runoff initializer reads the selector and its two numeric parameters at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcrnf.f90:308-320`.
Its false arm zeros the 2-D and depth masks and the enhanced-layer count at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcrnf.f90:504-535`.
The gate inventories every compiled file carrying `ln_rnf_mouth`, `rn_hrnf`,
or `rn_avt_rnf`; only `sbcrnf.f90` and the guarded `zdfphy.f90` consumer exist.

Internal-wave mixing remains selected by the unchanged manager dispatch at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:283-284`.
Its own namelist loader and molecular background reset remain active at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfiwm.f90:421-447`.
Only its differential T/S branch changes: false sets the salt/heat ratio to
exactly one before adding the same wave diffusivity to salt and heat, at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfiwm.f90:301-316`.
The gate's compiled-file census finds `ln_tsdiff` only in that file.

Thus rung 8 removes the one Decision-80 module without changing TKE, internal
waves, molecular backgrounds, runoff itself, sea-surface restoring, or any
other hierarchy module.  Runtime confirmation remains **UNMEASURED** until the
operator record exists.

## Prediction ledger

| prediction | status | evidence |
|---|---|---|
| HD3-P1 rung-9 admission | **CONFIRMED** | clean pass; all nine prior record plants fire |
| HD3-P2 one-module deck delta | **CONFIRMED** | exactly three assignments and physical lines; generated SHA pinned |
| HD3-P3 compiled consequences | **PARTLY CONFIRMED / runtime UNMEASURED** | exact compiled branches and resolved-log predicates pinned; needs rung-8 run |
| HD3-P4 retained parameters inert | **CONFIRMED** | values unchanged; guarded consumer inventory closed |
| HD3-P5 record completeness | **UNMEASURED** | no rung-8 record exists |
| HD3-P6 acquisition disposition | **CONFIRMED** | preflight passes, four plants fire, record target absent |

## Validation, review, and scope

Rung-8 preflight reports `PREFLIGHT_PASS_RUNG8`; the extra-deck-delta,
missing-selector, changed-retained-parameter, and changed-build-pin plants all
fire.  The focused rung-10/rung-9/rung-8 plus citation battery passes **44
tests**.

The required `tests/ocean/fidelity -n 12` battery ran once.  It reached 99%
and then produced no output for a bounded minute at the already-known late
stage-sweep hang, so it was interrupted rather than reported as complete.
All new hierarchy tests had passed.  The three visible failures reproduce in
isolation and are pre-existing files unchanged by this round: six legacy
report emitters missing worktree stamps; case-board omission of
`hires_lane_surface`; and the SI3 scalar-math gate's `A MY_SRC is not
verbatim` refusal.

The default citation audit passes with 274 citations, zero failures, zero
unmapped citations, and zero map-audit failures before this receipt is added.
The round receipt's own citation gate and rigid-shift plant are run after the
receipt is committed.

The required separate `codex exec --sandbox read-only` review failed before
reading the diff with `failed to initialize in-process app-server client:
Read-only file system`.  Verdict: **independent review unavailable in-sandbox**.

No file under `packages/` or `src/` changed.  GYRE is byte-identical by
construction and its year gate was not rerun.  No legoESM card, recipe,
physics, config, selector, threshold, carried state, or sea-ice declaration
changed.

ASKED choices: the three rung-8 switches and their top-down position are fixed
by Decision 80 and the side-lane instructions.  UNASKED choices: none.

## OPEN

1. Operator runs the committed rung-8 launcher.  Rung 8 stays UNMEASURED until
   all nine admission plants fire and the clean admission passes.
2. The next side-lane round admits rung 8, then reads the compiled
   `ln_zdfiwm=false` branch before constructing rung 7.  It must prove the
   resolved `rn_avt0`, `rn_avm0`, and `rn_emin` backgrounds with internal-wave
   mixing off and print the complete one-module diff.
3. Rungs 6 through 1 remain unbuilt.  Main-lane rung 0 remains out of scope.
