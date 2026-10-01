# ORCA2-DECKS round 4 receipt — rung 8 admitted, rung 7 internal-wave-off preflight

Date: 2026-10-01

Base: `9599298bfd`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round4.md` (commit
`1ae9d9283`, before record admission or deck construction).

Disposition: **STOPPED_FOR_RECORD**.  The operator-produced rung-8 record is
admitted under round 3's frozen predicates.  The rung-7 deck and acquisition
pipeline are preflight-clean, but no rung-7 oracle record exists.

Every run claim is labelled **independent**: NEMO starts from that rung's own
from-rest T/S initialization.  No given-entry result appears in this receipt.

## Rung inventory

| rung | NEMO deck | record | status / unmeasured specification |
|---:|---|---|---|
| 10 | exact shipped one-category ocean-ice deck | 480 operand frames + 240-step month | **ADMITTED** in round 2 |
| 9 | rung 10 with only `nn_ice: 2 -> 0` | 480 operand frames + 240-step month | **ADMITTED** in round 3 |
| 8 | rung 9 without double diffusion, river-mouth diffusivity, or differential internal-wave T/S mixing | 480 operand frames + 240-step month | **ADMITTED** this round |
| 7 | rung 8 without internal-wave mixing and its background reset | absent | **STOPPED_FOR_RECORD**; needs 480 header-valid finite frames, finite T/U/V/W month files, and finite fp64 step-240 ocean restarts |
| 6 | not built; rung 7 replaces TKE by GYRE constant mixing | absent | **UNMEASURED WITH SPEC** |
| 5 | not built; rung 6 minus runoff | absent | **UNMEASURED WITH SPEC** |
| 4 | not built; rung 5 minus RGB shortwave | absent | **UNMEASURED WITH SPEC**; compiled fallback must be cited before construction |
| 3 | not built; rung 4 minus bulk/restoring/freshwater budget | absent | **UNMEASURED WITH SPEC**; clean unforced formulation must be cited before construction |
| 2 | not built; rung 3 minus GM/MLE | absent | **UNMEASURED WITH SPEC** |
| 1 | not built; rung 2 minus BBL/geothermal | absent | **UNMEASURED WITH SPEC** |
| 0 | main-lane owned | not visible at this side-lane tip | **OUT OF SCOPE**; compare against rung 1 when available |

Every rung retains `ln_spc_dyn=.true.` exactly as the source deck carries it.
The reused binary still has no `key_agrif`, so the assignment remains inert.

## Rung-8 admission

The committed round-3 gate passes on the operator record, and all nine
record-stage plants have `STATUS PLANT-FIRED`.  Admission reports:

- 480/480 self-describing rank-step frames and 4,800 finite named payloads;
- two finite fp64 step-240 ocean restart shards and no ice product;
- eight finite T/U/V/W month output shards containing 96 floating variables;
- from-rest execution with all three rung-8 selectors false, internal-wave
  mixing retained, the unit salinity/heat wave ratio, the internal-wave
  background reset, and the invalid ice-namelist sentinel unread; and
- a complete 538-file SHA-256 record inventory.

HD3-P3 and HD3-P5 are therefore **CONFIRMED**.  The record producer is
`9599298bfd`; binary, deck, manifest, inputs, and run protocol equal the
round-3 pins.

## Complete rung-7 namelist delta

The complete physical-line diff from rung 8 is:

```diff
-   ln_zdfiwm   = .true.       ! internal wave-induced mixing            (T =>   fill namzdf_iwm)
+   ln_zdfiwm   = .false.      ! internal wave-induced mixing            (T =>   fill namzdf_iwm)
```

This is physical line 397 of the generated ocean namelist.  The complete
parsed assignment maps differ only on `namzdf.ln_zdfiwm`.  The rung-7 ocean
namelist SHA-256 is
`2a36188c237b8d1dcfd52f1fb1a009fd0f45634eacc36b42ff28340f64edd27d`.
The retained ice namelist, binary, CPP keys, inputs, from-rest mode, two-rank
layout, 240-step length, output cadence, and every other ocean assignment are
unchanged.

The internal-wave namelist group is byte-retained in the deliverable deck.
For execution, the gate replaces its first value with a syntactically invalid
sentinel and pins the resulting whole-file SHA-256 to
`4804d56f905bfedf124ed3472cfd0589dfceb6c88aeaedc77db9451715f365f8`.
Successful `STOP 0` will therefore prove the inactive group was not read.

## Compiled branch and resolved predictions

The vertical-physics manager reads the selector and both background values in
one namelist (`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`).
It first builds `avmb` and `avtb` from the retained `rn_avm0=1.2e-4` and
`rn_avt0=1.2e-5`; `nn_avb=0` keeps the vertical values constant while
`nn_havtb=1` retains the equatorial tracer-background shape
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:205-228`).

The only compiled initializer call is guarded by the selector
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:283-284`), as is
the per-step internal-wave increment
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:353-372`).
Turning the selector off therefore removes both.  The skipped initializer is
the only branch that overwrites the backgrounds with molecular values and a
uniform horizontal shape
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfiwm.f90:421-447`).

The inherited TKE namelist reports `rn_emin=1e-6`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdftke.f90:776`).  Its
internal-wave true arm alone forces `rn_emin=1e-10` and `rmxl_min=1e-3`; the
false arm instead retains `rn_emin=1e-6` and computes
`rmxl_min=1e-6/(0.1*sqrt(1e-6))=1e-2`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdftke.f90:835-841`).
These resolved values remain **UNMEASURED** until the operator runs rung 7.

The compiled-file census finds `ln_zdfiwm` only in `zdf_oce.f90`,
`zdfphy.f90`, and `zdftke.f90`; the two executable internal-wave calls occur
only in the guarded manager sites above.  Thus rung 7 removes Decision 80's
one internal-wave module without changing TKE, enhanced convection, runoff,
surface forcing, or another hierarchy module.

## Prediction ledger

| prediction | status | evidence |
|---|---|---|
| HD4-P1 rung-8 admission | **CONFIRMED** | clean pass; all nine prior record plants fire |
| HD4-P2 one-module deck delta | **CONFIRMED** | exactly one assignment and physical line; generated SHA pinned |
| HD4-P3 internal-wave arm absent | **PARTLY CONFIRMED / runtime UNMEASURED** | exact compiled guards pinned; needs rung-7 run |
| HD4-P4 inherited backgrounds restored | **PARTLY CONFIRMED / runtime UNMEASURED** | retained values and false branches pinned; resolved log needs rung-7 run |
| HD4-P5 retained wave namelist inert | **UNMEASURED** | invalid execution sentinel is pinned; needs rung-7 run |
| HD4-P6 record completeness | **UNMEASURED** | no rung-7 record exists |
| HD4-P7 acquisition disposition | **CONFIRMED** | preflight passes, four plants fire, record target absent |

## Validation, review, and scope

Rung-7 preflight reports `PREFLIGHT_PASS_RUNG7`; the extra-deck-delta,
missing-selector, changed-retained-background, and changed-build-pin plants all
fire.  The focused rung-10/rung-9/rung-8/rung-7 plus citation battery passes
**53 tests**.

The required `tests/ocean/fidelity -n 12` battery ran once.  It reached 99%
and then produced no output for a bounded minute at the known late-stage hang,
so it was interrupted rather than reported as complete.  All new hierarchy
and citation tests passed.  Five visible failures reproduce in isolation and
are pre-existing files unchanged by this round: six legacy report emitters
missing worktree stamps; case-board omission of `hires_lane_surface`; the
stale GYRE live-trace six-field expectation; the SI3 scalar-math gate's
`A MY_SRC is not verbatim` refusal; and the GYRE round-129 certified stepping
gate pin.

The default citation audit passes with 274 citations, zero failures, zero
unmapped citations, and zero map-audit failures; all nine built-in plants fire.
This receipt's own citation gate is run after this receipt is committed, with
a two-line shift plant required to refuse.

The required separate `codex exec --sandbox read-only` review failed before it
could read the diff with `failed to initialize in-process app-server client:
Read-only file system`.  Verdict: **independent review unavailable in-sandbox**.

No file under `packages/` or `src/` changed.  GYRE is byte-identical by
construction and its year gate was not rerun.  No legoESM card, recipe,
physics, config, selector, threshold, carried state, or sea-ice declaration
changed.

ASKED choices: the rung-7 selector and its top-down position are fixed by
Decision 80 and the side-lane instructions.  UNASKED choices: none.

## OPEN

1. Operator runs the committed rung-7 launcher.  Rung 7 stays UNMEASURED until
   all ten admission plants fire and the clean admission passes.
2. The next side-lane round admits rung 7, then constructs rung 6 by replacing
   TKE with constant vertical mixing at the GYRE-card values
   `rn_avt0=1.2e-5` and `rn_avm0=1.2e-4`, after reading the compiled dispatch
   and all TKE-dependent consequences.
3. Rungs 5 through 1 remain unbuilt.  Main-lane rung 0 remains out of scope.
