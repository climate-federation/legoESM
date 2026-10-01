# ORCA2-DECKS round 5 receipt — rung 7 admitted, rung 6 constant-mixing preflight

Date: 2026-10-01

Base: `c3089b635`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round5.md` (commit
`787be39b3`, before the admission repair, admission, or rung-6 construction).

Disposition: **STOPPED_FOR_RECORD**.  The existing operator-produced rung-7
run is admitted after correcting one source-refuted decimal-rendering
predicate.  The rung-6 deck and acquisition pipeline are preflight-clean, but
no rung-6 oracle record exists.

Every run claim is labelled **independent**: NEMO starts from that rung's own
from-rest T/S initialization.  No given-entry result appears in this receipt.

## Rung inventory

| rung | NEMO deck | record | status / unmeasured specification |
|---:|---|---|---|
| 10 | exact shipped one-category ocean-ice deck | 480 operand frames + 240-step month | **ADMITTED** in round 2 |
| 9 | rung 10 with only `nn_ice: 2 -> 0` | 480 operand frames + 240-step month | **ADMITTED** in round 3 |
| 8 | rung 9 without double diffusion, river-mouth diffusivity, or differential internal-wave T/S mixing | 480 operand frames + 240-step month | **ADMITTED** in round 4 |
| 7 | rung 8 without internal-wave mixing or its background reset | 480 operand frames + 240-step month | **ADMITTED** this round |
| 6 | rung 7 with constant mixing replacing TKE | absent | **STOPPED_FOR_RECORD**; needs 480 header-valid finite frames, finite T/U/V/W month files, and finite fp64 step-240 ocean restarts |
| 5 | not built; rung 6 minus runoff | absent | **UNMEASURED WITH SPEC** |
| 4 | not built; rung 5 minus RGB shortwave | absent | **UNMEASURED WITH SPEC**; compiled fallback must be cited before construction |
| 3 | not built; rung 4 minus bulk/restoring/freshwater budget | absent | **UNMEASURED WITH SPEC**; clean unforced formulation must be cited before construction |
| 2 | not built; rung 3 minus GM/MLE | absent | **UNMEASURED WITH SPEC** |
| 1 | not built; rung 2 minus BBL/geothermal | absent | **UNMEASURED WITH SPEC** |
| 0 | main-lane owned | not visible at this side-lane tip | **OUT OF SCOPE**; compare against rung 1 when available |

Every rung retains `ln_spc_dyn=.true.` exactly as the source deck carries it.
The reused binary still has no `key_agrif`, so the assignment remains inert.

## Rung-7 admission and loud correction

The NEMO process had completed all 240 steps.  The round-4 post-check alone
refused because it required the printed decimal
`rmxl_min=1.0000000000000000E-002`.  That was not the compiled statement:
the false-IWM branch evaluates `1.e-6/(rn_ediff*sqrt(rn_emin))`, and this run's
binary64 operands print `rn_ediff=0.10000000000000001` and
`rn_emin=9.9999999999999995E-007`, yielding
`9.9999999999999985E-003`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdftke.f90:835-841`).

**REFUTED:** HD4-P4's exact decimal-rendering prediction.  It is retained here
rather than silently edited.  The branch prediction and physical 0.01 m value
stand.  The gate now requires NEMO's exact source-produced print, and a new
control replacing it by the formerly expected rounded decimal makes the gate
refuse.

`run.sh --admit-existing` performed no NEMO run.  The corrected admission and
all ten record-stage plants pass/fire.  It reports:

- 480/480 self-describing rank-step frames and 4,800 finite named payloads;
- two finite fp64 step-240 ocean restart shards and no ice product;
- eight finite T/U/V/W month files with 94 floating variables;
- from-rest, no ice, no internal-wave initializer/reset, retained
  `rn_avm0=1.2e-4`, `rn_avt0=1.2e-5`, and `rn_emin=1e-6`, the source-computed
  standard mixing-length minimum, and both invalid inactive-group sentinels
  unread; and
- a complete 538-regular-file SHA-256 inventory.

HD5-P1 and HD5-P2 are **CONFIRMED**.  The record producer is
`c3089b6359388a4d8deee3a96d067439c390ca2c`; binary, deck, manifest, inputs,
and run protocol retain round 4's pins.

## Complete rung-6 namelist delta

The complete physical-line diff from rung 7 is:

```diff
 &namzdf        !   vertical physics manager                             (default: NO selection)
 !-----------------------------------------------------------------------
-   ln_zdftke   = .true.       !  Turbulent Kinetic Energy closure       (T =>   fill namzdf_tke)
+   ln_zdfcst   = .true.       !  constant mixing
+   ln_zdftke   = .false.      !  Turbulent Kinetic Energy closure       (T =>   fill namzdf_tke)
```

The added line is physical line 390 and the changed line is 391 of the rung-6
namelist.  In rung 7, `ln_zdfcst` was absent from `namelist_cfg` and therefore
resolved from `namelist_ref` as false.  The complete parsed assignment maps
differ only on `namzdf.ln_zdfcst` false-to-true and
`namzdf.ln_zdftke` true-to-false.  The rung-6 ocean namelist SHA-256 is
`d2828565a16bd79bd7f89a39297b59d3f9b953583294f2f1769f6a66c8fca331`.

The retained coefficients are exactly `rn_avm0=1.2e-4`,
`rn_avt0=1.2e-5`, and `nn_avb=0`.  As Decision 80 requires every unnamed
switch to remain unchanged, `nn_havtb=1` also remains: this is constant
vertical closure with the inherited equatorial tracer-background shape, not a
new uniform-horizontal choice.  The TKE namelist group stays byte-retained in
the deliverable deck.  The execution copy replaces its first value by a
syntactically invalid sentinel and is SHA-pinned; a successful run will prove
the inactive group was not read.

The retained ice namelist, binary, CPP keys, inputs, from-rest mode, two-rank
layout, 240-step length, output cadence, and every other ocean assignment are
unchanged.

## Compiled closure and every selector consumer

The manager reads both closure selectors and the retained coefficients in one
namelist (`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`).
It maps `ln_zdfcst` directly to `np_CST`, calls `zdf_tke_init` only when
`ln_zdftke` is true, requires exactly one closure, and disables the
shear-production calculation for constant closure
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:262-277`).
At each step, only the `np_TKE` arm calls `zdf_tke`; the `np_CST` arm makes no
closure call and retains the initialization-time arrays
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:330-343`).

Those arrays are built from `rn_avm0`, `rn_avt0`, `nn_avb`, and `nn_havtb`;
the last selector applies the inherited equatorial tracer shape before the
masked `avt_k`/`avm_k` initialization
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:205-228`).
The main restart writer calls `tke_rst` only under `ln_zdftke`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:380-385`).

The complete compiled selector census is closed.  `ln_zdfcst` occurs only in
`zdf_oce.f90` and `zdfphy.f90`; their declarations are adjacent
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdf_oce.f90:35-37`).
`ln_zdftke` additionally guards assimilation-background TKE read/write
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/asmbkg.f90:100-118`)
and every 25-hour TKE allocation, accumulation, normalization, output, and
reset site
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/dia25h.f90:142-172`,
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/dia25h.f90:244-246`,
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/dia25h.f90:286-288`,
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/dia25h.f90:339-343`,
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/dia25h.f90:369-371`).
These diagnostic/assimilation effects remain behind their own outer runtime
triggers; the hierarchy gate pins the guards rather than claiming those outer
paths execute in this deck.

Thus rung 6 removes Decision 80's TKE closure module and selects the specified
constant closure without changing enhanced convection, the background-shape
selector, runoff, surface forcing, or another hierarchy module.  Runtime
confirmation remains **UNMEASURED** until the operator record exists.

## Prediction ledger

| prediction | status | evidence |
|---|---|---|
| HD5-P1 rung-7 admission repair | **CONFIRMED** | clean existing-record pass; all ten admission plants fire; no NEMO rerun |
| HD5-P2 rung-7 consequences | **CONFIRMED** | corrected exact print plus every prior selector/sentinel predicate passes |
| HD5-P3 one-module rung-6 delta | **CONFIRMED** | exactly two selector assignments and two physical diff rows; generated SHA pinned |
| HD5-P4 constant closure | **PARTLY CONFIRMED / runtime UNMEASURED** | compiled dispatch and complete selector census pinned; needs rung-6 run |
| HD5-P5 retained coefficients | **PARTLY CONFIRMED / runtime UNMEASURED** | exact deck values and source initialization pinned; resolved log needs rung-6 run |
| HD5-P6 rung-6 record completeness | **UNMEASURED** | no rung-6 record exists |
| HD5-P7 acquisition disposition | **CONFIRMED** | preflight passes, five plants fire, record target absent |

## Validation, review, and scope

Rung-6 preflight reports `PREFLIGHT_PASS_RUNG6`; the extra-deck-delta,
missing-constant-selector, missing-TKE-selector, changed-retained-coefficient,
and changed-build-pin plants all fire.  The focused rung-10 through rung-6
battery passes **46 tests**.  The repaired rung-7 gate alone passes **9 tests**,
including the new exact-print control.

The final focused hierarchy plus citation battery passes **63 tests**.  The
default citation audit passes with 274 citations, zero failures, zero unmapped
citations, and zero map-audit failures.  This receipt's own audit passes all 13
citations with zero failures/unmapped entries; its rigid two-line shift of the
new vertical-closure key fires and exits 1.

The required `tests/ocean/fidelity -n 12` battery ran once, reached 99%, and
then emitted no output for a bounded 60 seconds at the known late-suite hang,
so it was interrupted rather than reported as complete.  Four visible
failures reproduce in one isolation battery and are the branch's recorded
pre-existing files, unchanged by this round: six legacy report emitters
missing worktree stamps; case-board omission of `hires_lane_surface`; the SI3
scalar-math gate's `A MY_SRC is not verbatim` refusal; and the GYRE round-129
certified stepping-gate pin.  All new hierarchy and citation tests had passed.

The required separate `codex exec --sandbox read-only` review failed before it
could read the diff with `failed to initialize in-process app-server client:
Read-only file system`.  Verdict: **independent review unavailable
in-sandbox**.

No file under `packages/` or `src/` changed.  GYRE is byte-identical by
construction and its year gate was not rerun.  No legoESM card, recipe,
physics, config, selector, threshold, carried state, or sea-ice declaration
changed.

ASKED choices: the rung-6 closure switches, coefficients, and top-down
position are fixed by Decisions 79/80 and the side-lane instructions.  UNASKED
choices: none.

## OPEN

1. Operator runs the committed rung-6 launcher.  Rung 6 stays UNMEASURED until
   all ten admission plants fire and the clean admission passes.
2. The next side-lane round admits rung 6, then constructs rung 5 by removing
   `ln_rnf` only after reading every compiled runoff consequence.  It must
   retain `ln_rnf_mouth=false` from rung 8 and every unnamed switch.
3. Rungs 4 through 1 remain unbuilt.  Main-lane rung 0 remains out of scope.
