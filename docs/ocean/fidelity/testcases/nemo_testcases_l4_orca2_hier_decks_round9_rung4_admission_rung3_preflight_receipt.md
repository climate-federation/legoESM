# ORCA2-DECKS round 9 receipt — rung 4 admitted, rung 3 preflight

Date: 2026-10-01

Disposition: **STOPPED_FOR_RECORD**.  The existing operator-produced rung-4
record is independently admitted, and the rung-3 exact-zero surface-flux deck
and acquisition are preflight-clean.  The rung-3 oracle record does not yet
exist.

Base: `d8edcb850`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round9.md`, commit `dce235fab`,
before this round ran the rung-4 gate, read the compiled surface dispatcher, or
constructed rung 3.  Every run claim is labelled **independent**: NEMO starts
from that rung's own from-rest initialization.

## Rung inventory

| rung | NEMO deck | independent record | status / unmeasured specification |
|---:|---|---|---|
| 10 | shipped one-category ocean-ice deck | 480 frames + 240-step month | **ADMITTED** round 2 |
| 9 | rung 10 without ice | 480 frames + 240-step month | **ADMITTED** round 3 |
| 8 | rung 9 without DDM, river-mouth diffusivity, differential T/S mixing | 480 frames + 240-step month | **ADMITTED** round 4 |
| 7 | rung 8 without internal-wave mixing/background reset | 480 frames + 240-step month | **ADMITTED** round 5 |
| 6 | rung 7 with constant mixing replacing TKE | 480 frames + 240-step month | **ADMITTED** round 6 |
| 5 | rung 6 without runoff | 480 version-2 frames + 240-step month | **ADMITTED** round 8 |
| 4 | rung 5 without penetrative chlorophyll shortwave | 480 version-2 frames + 240-step month | **ADMITTED** this round |
| 3 | rung 4 with exact-zero surface fluxes and no restoring/freshwater budget | absent | **STOPPED_FOR_RECORD**; needs 480 header-valid frames, finite T/U/V/W month files, and finite fp64 step-240 ocean restarts |
| 2 | not built; rung 3 minus GM/MLE | absent | **UNMEASURED WITH SPEC** |
| 1 | not built; rung 2 minus BBL/geothermal | absent | **UNMEASURED WITH SPEC** |
| 0 | main-lane owned | not compared | **OUT OF SCOPE** |

Every rung retains `ln_spc_dyn=.true.` exactly as the source deck carries it;
the reused binary has no `key_agrif`, so the assignment remains inert.

## Rung-4 admission

The operator's round-8 launcher finished with
`ORCA2_HIERARCHY_RUNG4_ACQUISITION_PASS`.  This round reran the committed clean
gate directly against that existing record and obtained `PASS_RUNG4_RECORD`:

- 480/480 frames, 3,840 finite PRESENT fields, and 960 ABSENT fields;
- exactly runoff-owned `rnf` and `rnf_tsc` ABSENT in every frame;
- two finite fp64 step-240 ocean restart shards and no ice product;
- eight finite T/U/V/W month files with 92 floating variables; and
- a complete 541-regular-file SHA-256 inventory.

All fifteen admission plants fired before the clean rerun: field name,
truncation, non-finite frame, ABSENT-as-zero, owner-on, missing frame,
non-finite terminal, wrong terminal step, ice/TKE sentinel reads, inherited
resolved consequence, runoff-group read, active-runoff print, shortwave
consequence, and SHA inventory.  HD9-P1 and HD9-P2 are **CONFIRMED**.

## Compiled rung-3 boundary

The compiled dispatcher requires exactly one surface formulation; `ln_usr`,
`ln_flx`, and `ln_blk` are distinct mutually exclusive selectors
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:299-306`).
At runtime it calls only the selected routine
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:440-449`).

`ln_usr` is **not** a zero-flux arm in this build.  Its compiled routine applies
the GYRE temperature/shortwave/freshwater forcing
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/usrdef_sbc.f90:124-135`)
and nonzero analytical wind stress
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/usrdef_sbc.f90:182-191`).
Therefore that alternative is **REFUTED**.

The unique no-physics arm is `ln_flx=true` with literal-zero input fields.
The compiled flux routine reads exactly `utau`, `vtau`, `qtot`, `qsr`, and
`emp`, then maps those operands directly to the surface fields
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcflx.f90:158-204`).
The selected file schema is not new: the pre-implementation search found the
main lane's round-82 rung-0 exact-zero file, which later NEMO records proved
loadable.  This round reuses its 180x148 grid, one climatological time slice,
fp64 type, and five exact-zero fields.

Bulk and restoring initialization are directly guarded by their switches
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:365-369`).
The per-step restoring and freshwater-budget calls are guarded independently
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:517-521`).
The retained restoring-bound parameter can alter `emp` and `qns` only inside
the disabled restoring routine
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcssr.f90:187-198`),
and the freshwater-budget namelist is read only inside the disabled budget
routine
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcfwb.f90:149-177`).
HD9-P3 is **CONFIRMED**: no CPP-key or rebuilt binary is needed and there is no
second defensible zero-forcing reading.

## Complete rung-3 namelist delta

Every physical line differing from rung 4 is printed below.  The first block
turns off bulk/restoring/budget, explicitly selects the only clean surface
dispatcher arm, and names its exact-zero operands.  The last line explicitly
turns off the bound belonging to the removed restoring module.

```diff
-   ln_blk      = .true.    !  Bulk formulation                          (T => fill namsbc_blk )
+   ln_blk      = .false.     !  Bulk formulation                          (T => fill namsbc_blk )
-   ln_ssr      = .true.    !  Sea Surface Restoring on T and/or S       (T => fill namsbc_ssr)
+   ln_ssr      = .false.     !  Sea Surface Restoring on T and/or S       (T => fill namsbc_ssr)
-   nn_fwb      = 2         !  FreshWater Budget:
+   nn_fwb      = 0           !  FreshWater Budget:
+   ln_usr         = .false.
+   ln_flx         = .true.
+   ln_abl         = .false.
+   ln_cpl         = .false.
+   ln_mixcpl      = .false.
+   ln_dm2dc       = .false.
+&namsbc_flx    ! hierarchy rung-3 exact-zero flux formulation (ln_flx=T)
+   cn_dir      = './'
+   sn_utau     = 'rung3_zero_flux', -12., 'utau', .false., .true., 'yearly', '', '', ''
+   sn_vtau     = 'rung3_zero_flux', -12., 'vtau', .false., .true., 'yearly', '', '', ''
+   sn_qtot     = 'rung3_zero_flux', -12., 'qtot', .false., .true., 'yearly', '', '', ''
+   sn_qsr      = 'rung3_zero_flux', -12., 'qsr',  .false., .true., 'yearly', '', '', ''
+   sn_emp      = 'rung3_zero_flux', -12., 'emp',  .false., .true., 'yearly', '', '', ''
-      ln_sssr_bnd =  .true.   !  flag to bound erp term (associated with nn_sssr=2)
+      ln_sssr_bnd =  .false.     !  flag to bound erp term (associated with nn_sssr=2)
```

The exact deck SHA-256 is
`6dbb38d77721e49dec63e452aca2c6e58cd159028a23eb5643016c7b40f6d6ab`.
All other parsed assignments, inputs, CPP keys, repaired record binary,
from-rest mode, two-rank layout, 240-step length, output cadence, and retained
unread TKE sentinel are unchanged.  The generated zero file is added to the
per-run input manifest and its full payload is checked for fp64 exact zero.

## Gates, review, and scope

- Rung-3 direct controls: **7 passed**; four preflight plants fire.
- The committed launcher passes `bash -n`, stages the exact deck idempotently,
  and reports `ORCA2_HIERARCHY_RUNG3_PREFLIGHT_READY` without invoking NEMO.
- No file under `packages/` or `src/` changed.  GYRE is byte-identical by
  construction; its year gate was not rerun.
- No NEMO source tree was modified.  The launcher reuses the admitted repaired
  binary and requests no rebuild.

ASKED choices: top-down rung order and removal of the bulk/restoring/freshwater
module are fixed by Decisions 79/80 and side-lane note B23.  The `ln_flx`
selection and exact-zero schema are source-decided and reuse the authorized
main-lane implementation.  UNASKED choices: none.

## Prediction ledger

| prediction | status |
|---|---|
| HD9-P1 rung-4 record | **CONFIRMED** |
| HD9-P2 admission non-vacuity | **CONFIRMED**; all fifteen plants fire |
| HD9-P3 unforced boundary | **CONFIRMED**; `ln_usr` refuted, exact-zero `ln_flx` unique |
| HD9-P4 rung-3 preflight | **CONFIRMED** |

## OPEN

1. The operator runs the committed rung-3 launcher.  Rung 3 remains
   UNMEASURED until all eighteen admission plants fire and the clean gate
   admits 480 frames, exact-zero input payloads, month products, and terminal
   restarts.
2. After rung 3 admits, construct rung 2 by removing only `ln_ldfeiv` and
   `ln_mle` under the same compiled-source and resolved-deck gates.
3. Rung 1 remains unbuilt; main-lane rung 0 remains out of scope.

Acquisition launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round9_acquisition/run.sh`.

Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung3/record`.

## UNVERIFIED

- The rung-3 NEMO run has not been executed.
- Its 480-frame census, exact-zero run input, resolved output, month products,
  terminal restarts, and SHA-256 admission are unmeasured.
