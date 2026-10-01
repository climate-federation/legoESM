# ORCA2-DECKS round 2 receipt — rung 10 admitted, rung 9 no-ice preflight

Date: 2026-10-01

Base: `432d704a7`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round2.md` (committed as
`6256e9438` before the rung-9 gate was run).

Disposition: **STOPPED_FOR_RECORD**.  The operator-produced rung-10 record is
admitted under round 1's frozen predicates.  The rung-9 no-sea-ice deck and
acquisition pipeline are preflight-clean, but no rung-9 oracle record exists.

All run claims are labelled **independent**: each NEMO run starts from its own
deck's from-rest T/S initialization.  No given-entry and independent numbers
are mixed here.

## Rung inventory

| rung | NEMO deck | record | status / unmeasured specification |
|---:|---|---|---|
| 10 | exact shipped one-category deck | 480 operand frames + 240-step month | **ADMITTED** |
| 9 | rung 10 with only `namsbc.nn_ice: 2 -> 0` | absent | **STOPPED_FOR_RECORD**; needs 480 header-valid finite frames, finite T/U/V/W month files, and finite fp64 step-240 ocean restarts |
| 8 | not built; rung 9 minus double diffusion, river-mouth diffusivity, and differential T/S mixing | absent | **UNMEASURED WITH SPEC**: one-module deck diff, two-rank kt=1..10 record, independent month |
| 7 | not built; rung 8 minus internal-wave mixing/background reset | absent | **UNMEASURED WITH SPEC** |
| 6 | not built; rung 7 replaces TKE by GYRE constant mixing | absent | **UNMEASURED WITH SPEC** |
| 5 | not built; rung 6 minus runoff | absent | **UNMEASURED WITH SPEC** |
| 4 | not built; rung 5 minus RGB shortwave | absent | **UNMEASURED WITH SPEC**; compiled fallback must be cited before construction |
| 3 | not built; rung 4 minus bulk/restoring/freshwater budget | absent | **UNMEASURED WITH SPEC**; clean unforced formulation must be cited before construction |
| 2 | not built; rung 3 minus GM/MLE | absent | **UNMEASURED WITH SPEC** |
| 1 | not built; rung 2 minus BBL/geothermal | absent | **UNMEASURED WITH SPEC** |
| 0 | main-lane owned | not visible at this side-lane tip | **OUT OF SCOPE**; compare against rung 1 when available |

Every rung retains `ln_spc_dyn=.true.` exactly as the source deck carries it.
The compiled binary has no `key_agrif`, so round 1's gate continues to classify
that line as retained but inert.  No hierarchy switch outside Decision 80 was
changed.

## Rung-10 admission

The committed round-1 launcher was rerun with `--admit-existing`; all eight
real-record controls fired before the clean pass.  Admission reports:

- 480/480 self-describing rank-step frames, 4,800 finite field payloads;
- 200/200 field comparisons at kt=1..10 raw-bit equal to the round-69 source,
  including signed zero;
- two ocean plus two ice step-240 restart shards payload-identical to the
  source except permitted timestamp metadata, with `sshn`, `un`, `vn`, `tn`,
  and `sn` finite fp64;
- eight finite T/U/V/W month products and a complete 539-file SHA-256
  inventory; and
- resolved from-rest, step/restart 240, `nn_ice=2`, `jpl=1`, `STOP 0`.

Thus HD1-P3, HD1-P4, the measurable part of HD1-P5, and HD1-P6 are
**CONFIRMED**.  HD1-P5's old `ln_spc_dyn` print prediction remains
**REFUTED** as recorded in round 1; it is not silently rewritten.

## Oracle source and exact rung delta

The upper deck's only changed line is
`rung10_namelist_cfg:86`:

```diff
-   nn_ice      = 2         !  =0 no ice boundary condition
+   nn_ice      = 0         !  =0 no ice boundary condition
```

The gate compares complete parsed assignment maps and all physical lines.  It
reports exactly `{namsbc.nn_ice: [2, 0]}` and no second delta.  The compiled
manager reads this selector in `ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:159-165`
and defines zero as the no-ice arm at `:247-259`.  That arm fixes ice fraction
to zero at `:274-280`, allocates only placeholder arrays instead of calling
SI3 at `:376-382`, and has no `ice_stp` case at `:499-503`.

The only load of the ice configuration occurs inside `ice_init` at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/icestp.f90:285-305`.
Therefore all 91 parsed assignments in the byte-retained
`namelist_ice_cfg` become inert together.  The gate emits every fully-qualified
key in `preflight.json`; their groups are `namdia`, `namdyn`, `namdyn_adv`,
`namdyn_rdgrft`, `namdyn_rhg`, `namini`, `namitd`, `nampar`, `namsbc`,
`namthd`, `namthd_do`, `namthd_pnd`, `namthd_sal`, and `namthd_zdf`.
The launcher proves non-reading with a committed syntactically invalid
execution sentinel; successful `STOP 0` is required and any SI3 initialization,
ice namelist output, ice initialization file, ice restart, or ice month product
is a refusal.

Three ocean-side ice-coupled values become inert/resolved differently as the
compiled consequence of the same one-module switch; they are deliberately not
additional deck edits:

| namelist line | rung 10 | required rung-9 resolution | compiled source |
|---|---:|---:|---|
| `rung10_namelist_cfg:416`, `namzdf_tke.nn_mxlice` | 2 | 0 | `ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdftke.f90:785-787` |
| reference `namdrg.ln_drgice_imp` | true | false | `ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfdrg.f90:371-380` |
| reference `namsbc_fwb.nn_fwb_voltype` | 1, ice+ocean | 2, ocean-only | `ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcfwb.f90:149-156` |

`namsbc.ln_ice_embd` is already false in the inherited reference namelist, so
it needs no deck edit.  The complete ocean-deck diff is the single line above;
the byte-retained ice deck has zero differing lines.  CPP keys, binary, input
manifests, two-rank layout, from-rest mode, 240-step length, output cadence,
and `ln_spc_dyn` are unchanged.

## Prediction ledger

| prediction | status | evidence |
|---|---|---|
| HD2-P1 rung-10 admission | **CONFIRMED** | clean pass after all eight record plants fired |
| HD2-P2 one-module deck delta | **CONFIRMED** | one assignment, one physical line, exact generated-deck SHA-256 `9177246f...eb15` |
| HD2-P3 ice namelist unopened | **UNMEASURED** | source path and sentinel gate established; needs operator run |
| HD2-P4 compiled consequences | **UNMEASURED** | source branches pinned; resolved values need operator run |
| HD2-P5 record completeness | **UNMEASURED** | no rung-9 record exists |
| HD2-P6 acquisition disposition | **CONFIRMED** | rung-9 target absent; preflight reports `ORCA2_HIERARCHY_RUNG9_PREFLIGHT_READY` |

## Validation, review, and scope

The rung-10/rung-9 focused battery passes **18 tests**.  Rung-9 preflight and
its three real preflight plants pass/fire.  Synthetic tests prove the nine
record-stage violations fire: malformed name, truncation, missing frame,
non-finite frame, non-finite terminal state, wrong terminal step, sentinel
read, wrong resolved no-ice consequence, and incomplete SHA inventory.

Independent diff review: **PENDING at receipt draft**.

No file under `packages/` or `src/` changed.  GYRE is byte-identical by
construction and its year gate was not rerun.  This round changes no legoESM
card, recipe, physics, config, selector, threshold, or carried state.  It makes
no sea-ice physics claim: rung 9 removes SI3 from NEMO; it does not alter the
existing legoESM ORCA2 card or its `unmeasured_features`.

## OPEN

1. Operator runs the committed rung-9 launcher; rung 9 stays UNMEASURED until
   every admission plant fires and the clean admission passes.
2. The next side-lane round admits rung 9, then constructs rung 8 by removing
   exactly `ln_zdfddm`, `ln_rnf_mouth`, and `ln_tsdiff`, after reading their
   compiled dispatch and required resolved backgrounds.
3. Rungs 7 through 1 remain unbuilt.  Main-lane rung 0 remains out of scope.
