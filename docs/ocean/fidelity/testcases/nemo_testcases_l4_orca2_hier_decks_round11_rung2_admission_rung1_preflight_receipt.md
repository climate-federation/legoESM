# ORCA2-DECKS round 11 receipt — rung 2 admitted, rung 1 preflight

Date: 2026-10-01

Disposition: **STOPPED_FOR_RECORD**.  The operator-produced rung-2 record is
complete and admitted.  The rung-1 deck, gate, manifest, and fail-closed
acquisition are preflight-clean.  The rung-1 oracle record does not yet exist.

Base: `713a3e86b1`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round11.md`, commit `406a2f421`,
before this round admitted rung 2, read the compiled BBL/geothermal branches,
or constructed rung 1.  Every run claim is labelled **independent**: NEMO
starts from that rung's own from-rest initialization.

## Rung inventory

| rung | NEMO deck | independent record | status / unmeasured specification |
|---:|---|---|---|
| 10 | shipped one-category ocean-ice deck | 480 frames + 240-step month | **ADMITTED** round 2 |
| 9 | rung 10 without ice | 480 frames + 240-step month | **ADMITTED** round 3 |
| 8 | rung 9 without DDM, river-mouth diffusivity, differential T/S mixing | 480 frames + 240-step month | **ADMITTED** round 4 |
| 7 | rung 8 without internal-wave mixing/background reset | 480 frames + 240-step month | **ADMITTED** round 5 |
| 6 | rung 7 with constant mixing replacing TKE | 480 frames + 240-step month | **ADMITTED** round 6 |
| 5 | rung 6 without runoff | 480 version-2 frames + 240-step month | **ADMITTED** round 8 |
| 4 | rung 5 without penetrative chlorophyll shortwave | 480 version-2 frames + 240-step month | **ADMITTED** round 9 |
| 3 | rung 4 with exact-zero surface fluxes and no restoring/freshwater budget | 480 version-2 frames + 240-step month | **ADMITTED** round 10 |
| 2 | rung 3 without GM eddy-induced velocity or mixed-layer eddies | 480 version-2 frames + 240-step month | **ADMITTED** this round |
| 1 | rung 2 without BBL or geothermal heating | absent | **STOPPED_FOR_RECORD**; needs 480 header-valid frames, finite T/U/V/W month files, and finite fp64 step-240 ocean restarts |
| 0 | main-lane owned dynamics-core deck | main-lane record in progress | **OUT OF SIDE-LANE SCOPE**; deck comparison below |

Every side-lane rung retains `ln_spc_dyn=.true.` exactly as the source deck
carries it.  The reused binary's pinned CPP-key line has no `key_agrif`
(`cpp_ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE.fcm:1`), and its compiled source has no
`ln_spc_dyn` occurrence, so the assignment remains inert.  This is unchanged
from the gated round-1 finding at `rung10_namelist_cfg:234`.

## Rung-2 admission

The operator-produced target existed before this round.  The committed round-10
launcher ran in admit-only mode: all nineteen plants printed
`STATUS PLANT-FIRED`, the clean gate printed `PASS_RUNG2_RECORD`, and no NEMO
run occurred.  The independent record contains:

- 480/480 self-describing frames, 3,840 finite PRESENT fields, and 960 ABSENT
  fields, exactly `rnf` and `rnf_tsc` absent in every frame;
- an fp64 exact-zero five-field surface-flux file with its SHA-256 pinned;
- two finite fp64 step-240 ocean restart shards and no ice product;
- eight finite T/U/V/W month files with 86 floating variables; and
- a complete 545-regular-file SHA-256 inventory.

HD11-P1 and HD11-P2 are **CONFIRMED**.

## Compiled rung-1 boundary

NEMO always calls both initializers from `nemo_init`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/nemogcm.f90:428-434`).
The BBL initializer reads its namelist, prints the selector, and returns before
allocation when `ln_trabbl=false`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/trabbl.f90:540-564`).
The geothermal initializer reads its namelist but allocates/reads the heat-flow
field only inside `ln_trabbc=true`; the false arm prints that no geothermal
flux is used
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/trabbc.f90:200-251`).

At runtime, both possible BBL coefficient call sites are guarded by
`ln_trabbl` and stage 3
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:430-458`).
The stage-3 geothermal and BBL tracer calls are separately guarded by
`ln_trabbc` and `ln_trabbl`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:523-529`).
Thus Decision 80's rung-1 boundary is namelist-only and needs no CPP-key or
binary change.  HD11-P3 is **CONFIRMED**.

## Complete rung-1 namelist delta

These are every physical line differing from rung 2.  The retained geothermal
and BBL parameters are inert behind the false selectors.

```diff
-   ln_trabbc   = .true.    !  Apply a geothermal heating at the ocean bottom
+   ln_trabbc   = .false.    !  Apply a geothermal heating at the ocean bottom
-   ln_trabbl   = .true.    !  Bottom Boundary Layer parameterisation flag
+   ln_trabbl   = .false.    !  Bottom Boundary Layer parameterisation flag
```

The exact rung-1 deck SHA-256 is
`44a15a9d7c8e1b77bebf8704d6455edc83ba7ebea93f358d8bbe093d5e1f0670`.
All other parsed assignments, inputs, CPP keys, repaired record binary,
from-rest mode, two-rank layout, 240-step length, output cadence, exact-zero
surface file, and retained unread TKE sentinel are unchanged.

## Required rung-1 versus main-lane rung-0 deck diff

The main lane now has a rung-0 deck at round 90, SHA-256
`d25c69958aeb7d4dffeeab6b08c89f6b314dd7c6d130ed94acfee7cb90643c2c`.
After stripping comments and whitespace, it differs from this rung-1 deck in
21 assignments.  This is **not yet a one-module boundary**.  The gate pins and
prints the complete semantic diff:

| assignment | side-lane rung 1 | main-lane rung 0 |
|---|---|---|
| `namagrif.ln_spc_dyn` | `.true.` | `.false.` |
| `namrun.ln_rst_list` | absent | `.true.` |
| `namrun.nn_itend` | `240` | `10` |
| `namrun.nn_stock` | `240` | `1` |
| `namrun.nn_stocklist` | absent | `1..10` |
| five `namsbc_flx.sn_*` file operands | `rung3_zero_flux` | `rung0_zero_flux` |
| `namsbc_ssr.ln_sssr_bnd` | `.false.` | `.true.` |
| `namtra_dmp.ln_tradmp` | `.true.` | `.false.` |
| `namtra_qsr.nn_chldta` | `0` | `1` |
| `namtsd.ln_tsd_dmp` | `.true.` | `.false.` |
| six explicit inactive `namzdf.ln_zdf*` selectors | absent | `.false.` |
| `namzdf.nn_havtb` | `1` | `0` |

The two damping rows are the intended rung-1 to rung-0 module removal.  Four
rows are the main lane's ten-step restart protocol and five select its separately
named zero-flux file.  Ten additional resolved/declaration differences remain:
the special-dynamics selector, restoring bound, chlorophyll-data selector, six
explicit inactive vertical-mixing selectors, and the background-diffusivity
shape.  This side lane changes none of them: note B23 orders unnamed switches
to remain exactly as the record deck carries them, while the main lane owns
rung 0.  The discrepancy is recorded as OPEN for the cross-lane handoff.

## Gates, review, and scope

- The focused round-11 gate battery passes **9 tests**; all four preflight
  plants fire.
- The committed launcher passes shell syntax, Python compilation, stages the
  exact deck idempotently, and prints `ORCA2_HIERARCHY_RUNG1_PREFLIGHT_READY`
  without invoking NEMO.  HD11-P4 is **CONFIRMED**.
- The required separate `codex exec --sandbox read-only` review failed before
  reading the diff: `failed to initialize in-process app-server client:
  Read-only file system`.  Verdict: **independent review unavailable
  in-sandbox**.
- No file under `packages/` or `src/` changed.  GYRE is byte-identical by
  construction; its year gate was not rerun.
- No NEMO source tree was modified.  The rung-1 launcher reuses the admitted
  repaired binary and requests no rebuild.

ASKED choices: rung ordering and removal of BBL plus geothermal heating are
fixed by Decision 80 and side-lane note B23.  The side-lane switches retained
across the rung-1/main-rung-0 diff are also fixed by B23.  UNASKED choices:
none.

## Prediction ledger

| prediction | status |
|---|---|
| HD11-P1 rung-2 record | **CONFIRMED**; clean admit-only pass |
| HD11-P2 admission non-vacuity | **CONFIRMED**; all nineteen plants fire |
| HD11-P3 rung-1 boundary | **CONFIRMED**; exact two-selector namelist-only delta |
| HD11-P4 rung-1 preflight | **CONFIRMED** |

## OPEN

1. The operator runs the committed rung-1 launcher.  Rung 1 remains
   UNMEASURED until all twenty admission plants fire and the clean gate admits
   480 frames, exact-zero input, month products, terminal restarts, and the
   complete SHA-256 inventory.
2. The main and side lanes must reconcile the ten non-protocol/non-file-name
   rung-1/rung-0 deck differences before calling that edge a one-module damping
   boundary.  This side lane makes no choice for main-lane-owned rung 0.
3. Rung 0's record and legoESM card remain main-lane work and out of scope.

Acquisition launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round11_acquisition/run.sh`.

Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung1/record`.

## UNVERIFIED

- The rung-1 NEMO run has not been executed.
- Its 480-frame census, resolved inactive BBL/geothermal arms, month products,
  terminal restarts, and SHA-256 admission are unmeasured.
- The physical significance of the ten extra rung-1/rung-0 deck differences is
  not attributed here; only the mechanically parsed configuration diff is
  claimed.
