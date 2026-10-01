# ORCA2-DECKS round 6 receipt — rung 6 admitted, rung 5 runoff-off preflight

Date: 2026-10-01

Base: `42466030b`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round6.md` (commit
`e839ce86d`, before the admission repair, record admission, or rung-5
construction).

Disposition: **STOPPED_FOR_RECORD**.  The operator-produced rung-6 record is
admitted after correcting one over-scoped inherited TKE-output predicate.  The
rung-5 deck and acquisition pipeline are preflight-clean, but no rung-5 oracle
record exists.

Every run claim is labelled **independent**: NEMO starts from that rung's own
from-rest T/S initialization.  No given-entry result appears in this receipt.

## Rung inventory

| rung | NEMO deck | record | status / unmeasured specification |
|---:|---|---|---|
| 10 | exact shipped one-category ocean-ice deck | 480 operand frames + 240-step month | **ADMITTED** in round 2 |
| 9 | rung 10 with only `nn_ice: 2 -> 0` | 480 operand frames + 240-step month | **ADMITTED** in round 3 |
| 8 | rung 9 without double diffusion, river-mouth diffusivity, or differential internal-wave T/S mixing | 480 operand frames + 240-step month | **ADMITTED** in round 4 |
| 7 | rung 8 without internal-wave mixing or its background reset | 480 operand frames + 240-step month | **ADMITTED** in round 5 |
| 6 | rung 7 with constant mixing replacing TKE | 480 operand frames + 240-step month | **ADMITTED** this round |
| 5 | rung 6 with runoff disabled | absent | **STOPPED_FOR_RECORD**; needs 480 header-valid finite frames, finite T/U/V/W month files, and finite fp64 step-240 ocean restarts |
| 4 | not built; rung 5 minus RGB shortwave | absent | **UNMEASURED WITH SPEC**; compiled fallback must be cited before construction |
| 3 | not built; rung 4 minus bulk/restoring/freshwater budget | absent | **UNMEASURED WITH SPEC**; clean unforced formulation must be cited before construction |
| 2 | not built; rung 3 minus GM/MLE | absent | **UNMEASURED WITH SPEC** |
| 1 | not built; rung 2 minus BBL/geothermal | absent | **UNMEASURED WITH SPEC** |
| 0 | main-lane owned | not visible at this side-lane tip | **OUT OF SCOPE**; compare against rung 1 when available |

Every rung retains `ln_spc_dyn=.true.` exactly as the source deck carries it.
The reused binary still has no `key_agrif`, so the assignment remains inert.

## Rung-6 admission and loud correction

The NEMO process completed all 240 steps.  The round-5 post-check alone
refused because the inherited no-ice gate required the printed line
`nn_mxlice=0`.  The reset and print both live inside `zdf_tke_init`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdftke.f90:785-787` and
`:796-798`).  The compiled dispatcher calls that initializer only for
`ln_zdftke=true`; rung 6 selects constant mixing and never calls it
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:262-277`).

**REFUTED:** the round-5 application of the active-TKE `nn_mxlice` print
predicate to rung 6.  The rung-9 through rung-7 measurement remains valid
because those rungs run TKE.  The repaired gate requires the print to be
present when TKE is active and absent when TKE is inactive; injecting the line
into a rung-6 fixture makes it refuse.

`run.sh --admit-existing` performed no NEMO run.  The corrected admission and
all ten inherited record-stage plants pass/fire.  It reports:

- 480/480 self-describing rank-step frames and 4,800 finite named payloads;
- two finite fp64 step-240 ocean restart shards and no ice product;
- eight finite T/U/V/W month files with 94 floating variables;
- from-rest, no ice, constant mixing true, TKE false, retained
  `rn_avm0=1.2e-4`, `rn_avt0=1.2e-5`, `nn_avb=0`, `nn_havtb=1`, and both
  invalid inactive-group sentinels unread; and
- a complete 540-regular-file SHA-256 inventory.

HD6-P1 and HD6-P2 are **CONFIRMED**.  The record producer is
`42466030b1082f2712bc9d84436667f902afe76f`; binary, deck, manifest, inputs,
and run protocol retain round 5's pins.

## Complete rung-5 namelist delta

The complete physical-line diff from rung 6 is:

```diff
-   ln_rnf      = .true.    !  runoffs                                   (T => fill namsbc_rnf)
+   ln_rnf      = .false.   !  runoffs                                   (T => fill namsbc_rnf)
```

This is physical line 92 of the generated ocean namelist.  The complete parsed
assignment maps differ only on `namsbc.ln_rnf`.  The rung-5 ocean namelist
SHA-256 is
`f537e3d29a6e2472f89cc9a8d23ec70d18756e3ad112088e7256804698bc1d59`.

The retained values include `ln_rnf_mouth=false`, `rn_hrnf=15`,
`rn_avt_rnf=1e-3`, `rn_rfact=1`, `ln_blk=true`, `ln_ssr=true`, `nn_fwb=2`,
`ln_traqsr=true`, `ln_zdfcst=true`, and `ln_zdftke=false`.  The retained ice
namelist, binary, CPP keys, inputs, from-rest mode, two-rank layout, 240-step
length, output cadence, and every other ocean assignment are unchanged.

## Compiled runoff branch and selector census

The surface manager reads `ln_rnf` in `namsbc` and prints its resolved value
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:159-165` and
`:183-207`).  It then calls the runoff initializer unconditionally
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:359-374`).
The initializer reads and echoes the complete `namsbc_rnf` group before
testing the selector
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcrnf.f90:308-320`).
With runoff false it forces the four runoff sub-switches false and returns
before allocation or file input
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcrnf.f90:322-330`).

**REFUTED:** HD6-P4's preregistered proposal to prove the group is read by
placing a value in the active-runoff control print.  That print is downstream
of the false-arm return, so it cannot appear.  The non-vacuous replacement is
the exact `&NAMSBC_RNF` group in `output.namelist.dyn`, emitted by the compiled
write before the return; deleting that echo makes the resolved gate refuse.

The active per-step runoff call is guarded by the selector
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:517`).  So are
the RK3 tracer heat/salt source
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/trasbc.f90:318-324`), both
continuity runoff-divergence sites
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/divhor.f90:142` and
`:199`), and the external-mode surface-height source
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/stp2d.f90:241-245`).
All are inactive at rung 5.

The gate also enumerates every compiled file containing `ln_rnf`: 24 files,
with no unaccounted occurrence.  This is a symbol census, not a claim that all
24 paths execute on this deck.  HD6-P3 and HD6-P5 are **CONFIRMED**;
HD6-P4 is **CONFIRMED with its preregistered control method refuted and
replaced**.  Runtime confirmation remains **UNMEASURED** until the operator
runs rung 5.

## Prediction ledger

| prediction | status | evidence |
|---|---|---|
| HD6-P1 rung-6 admission repair | **CONFIRMED** | existing record admits; all inherited admission plants fire; no NEMO rerun |
| HD6-P2 rung-6 record completeness | **CONFIRMED** | 480 frames, 4,800 payloads, 94 month variables, two restarts, 540-file inventory |
| HD6-P3 one-module rung-5 delta | **CONFIRMED** | exactly one assignment and one physical line; generated SHA pinned |
| HD6-P4 runoff false branch | **PARTLY CONFIRMED / control method REFUTED / runtime UNMEASURED** | compiled branch and group echo pinned; active-runoff print is correctly absent; needs rung-5 run |
| HD6-P5 retained modules | **CONFIRMED** | exact assignment-map equality outside `ln_rnf` |
| HD6-P6 rung-5 record completeness | **UNMEASURED** | no rung-5 record exists |
| HD6-P7 acquisition disposition | **CONFIRMED** | preflight passes, four plants fire, record target absent |

## Validation, review, and scope

Rung-5 preflight reports `PREFLIGHT_PASS_RUNG5`; the extra-deck-delta,
missing-runoff-selector, changed-retained-selector, and changed-build-pin
plants all fire.  The final focused hierarchy-rounds-1-through-6 and citation
test battery passes **73 tests** (including the 29-test repair/rung-5 subset).
Ruff passes the new gate and test; the launcher passes `bash -n`.

The round receipt citation gate passes with 13 citations, zero failures, and
zero unmapped citations.  Its planted bad span exits 1.  The default receipt
gate also passes with 274 citations, zero failures, and zero unmapped
citations/map-audit failures.

The one broad `tests/ocean/fidelity -n 12` battery reached 99%, reproduced the
listed pre-existing SI3 scalar-math provenance failure
(`test_nemo_si3_scalarmath_v2_gate.py::test_full_v2_gate_and_plants`), then
made no progress for 60 seconds in the known late-suite stall and was
interrupted.  It therefore has no complete-suite verdict; no second broad
battery was started.

The required separate `codex exec --sandbox read-only` review failed before it
could read the diff with `failed to initialize in-process app-server client:
Read-only file system`.  Verdict: **independent review unavailable
in-sandbox**.

No file under `packages/` or `src/` changed.  GYRE is byte-identical by
construction and its year gate was not rerun.  No legoESM card, recipe,
physics, config, selector, threshold, carried state, or sea-ice declaration
changed.

ASKED choices: the runoff selector and its top-down position are fixed by
Decisions 79/80 and the side-lane instructions.  UNASKED choices: none.

## OPEN

1. Operator runs the committed rung-5 launcher.  Rung 5 stays UNMEASURED until
   all twelve admission plants fire and the clean admission passes.
2. The next side-lane round admits rung 5, then resolves from compiled source
   whether rung 4 disables light penetration or selects NEMO's two-band
   fallback before constructing any deck.  Two defensible readings require
   `DECISION_NEEDED`, not a silent selection.
3. Rungs 3 through 1 remain unbuilt.  Main-lane rung 0 remains out of scope.
