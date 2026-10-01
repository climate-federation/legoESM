# ORCA2-DECKS round 7 receipt — rung-5 recorder ABSENT repair

Date: 2026-10-01

Disposition: **STOPPED_FOR_RECORD**.  The rung-5 SIGSEGV is source-resolved to
the inherited recorder's unconditional runoff-field dereference.  An
additions-only repair, self-describing checker, active-runoff calibration, and
fresh operator launcher are committed and preflight-clean.  Neither the
repaired rung-6 calibration nor rung-5 NEMO run has been executed.

Base: `769526b4c`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round7.md`, commit `206e6c88f`,
before the repair or synthetic measurements.  Every run claim is labelled
**independent**: each NEMO rung starts from its own from-rest initialization.

## Independent crash resolution and complete field-owner census

The operator's original rung-5 run stopped with SIGSEGV before `time.step`
existed.  Both ranks' unsymbolized backtraces are preserved in
`rung5/record/run.user.stdout.log`; no model output from that failed run is
admitted.

The compiled recorder writes ten fields in order.  Its first runoff-owned call
is `rnf`, followed later by `rnf_tsc`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/l4_r69_surface.f90:65-88`).
NEMO allocates `rnf` only under `ln_rnf`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbc_oce.f90:195-216`) and
allocates `rnf_tsc` only in the runoff allocator
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcrnf.f90:146-152`).
Rung 5 resolves `ln_rnf=false`, so both arrays are
unallocated when the inherited recorder reaches them.

The complete current-schema owner table is:

| fields | owner | allocation statement | repaired record state at rung 5 |
|---|---|---|---|
| `utau`, `vtau`, `emp`, `fr_i`, `qns`, `qsr`, `sfx`, `taum` | surface core | `ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbc_oce.f90:195-216` | PRESENT |
| `rnf` | `ln_rnf` | `ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbc_oce.f90:195-216` | ABSENT |
| `rnf_tsc` | `ln_rnf` | `ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcrnf.f90:146-152` | ABSENT |

Thus HD7-P1 is **CONFIRMED by compiled-source reading** and remains
**UNMEASURED at runtime** until the operator run passes.  The repair names the
owner switch beside each conditional writer statement.

### Correction to the operator note's future-field prediction

The ten-field recorder has no chlorophyll, eddy-induced-velocity,
mixed-layer-eddy, bottom-boundary-layer, or geothermal payload.  Those modules
cannot cause this recorder to dereference their arrays on lower rungs.  Its
`qsr` is the core surface flux allocated unconditionally at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbc_oce.f90:195-216`, not a
conditionally allocated chlorophyll field.
Therefore the preregistered current-schema census finds exactly two
module-gated fields, both owned by `ln_rnf`; inventing ABSENT entries for fields
the schema does not contain would change the instrument rather than repair it.
Any later schema extension is required to declare and gate its owner first.

## Additions-only repair and passivity predicate

The committed patch applies at fuzz zero to the exact round-69 writer and has
zero removed source lines.  The active-runoff path remains the inherited
version-1 writer byte-for-byte.  Only the false branch calls the new version-2
writer; it emits the same field names and header-derived PRESENT payloads for
the eight core fields, and emits `(rank,n1,n2,n3)=(0,0,0,0)` with no payload
for `rnf` and `rnf_tsc`.

This is the main lane's round-87 ABSENT convention applied to the hierarchy's
existing schema, not a second record format rule.  The checker parses magic,
header integers, every ordered name, each header-derived payload length, fp64
finiteness, and physical EOF.  ABSENT is accepted only when the exact rung-5
deck, execution namelist, and `ocean.output` all resolve `ln_rnf=false`.

HD7-P3's schema and HD7-P4 are **CONFIRMED offline**: a synthetic
eight-PRESENT/two-ABSENT record passes, while changed name, truncation,
non-finite PRESENT payload, ABSENT-as-zero, and owner-on plants all refuse.
No synthetic result is promoted to a NEMO record claim.

## Acquisition order and frozen admission

The launcher creates one fresh instrumented build and two fresh run targets.
It runs the exact admitted rung-6 deck first.  Before rung 5 may start, all 480
surface frames must compare byte-for-byte with the admitted rung-6 record, and
both step-240 ocean restart shards must compare byte-for-byte.  This is the
required proof that the additions-only branch and rebuilt binary are passive
when runoff is active.  Any difference refuses the pipeline.

Only after that calibration passes does the same binary run rung 5.  Final
admission requires 480 version-2 frames with 3,840 PRESENT and 960 ABSENT field
entries, finite month products, finite fp64 step-240 ocean restarts, the
resolved runoff-off consequences, a complete SHA-256 inventory, and every
round-6 plant plus the five new schema plants.  HD7-P2 and HD7-P5 remain
**UNMEASURED**.

`run.sh --preflight-only` applies the patch at fuzz zero, compiles the patched
Fortran against the record build's module tree, and reports:

```text
ORCA2_HIERARCHY_RUNG5_ABSENT_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung5/record_absent_v2
```

HD7-P6 is **CONFIRMED**: neither fresh run target existed at handoff.  No
`makenemo` or `mpirun` command was executed in the sandbox.

## Validation, review, and scope

- Direct gate/parser/runner controls: **9 passed**.
- Ruff passes the new gate and test; the launcher passes `bash -n`; the patch
  passes Fortran syntax compilation against the pinned record build.
- The separate read-only `codex exec` review failed before reading the diff
  with `failed to initialize in-process app-server client: Read-only file
  system`.  Verdict: **independent review unavailable in-sandbox**.
- No file under `packages/` or `src/` changed.  GYRE is byte-identical by
  construction and its year gate was not rerun.
- No NEMO source tree was modified by this agent.  The committed patch is
  applied only by the operator launcher to a fresh target configuration.

ASKED choices: explicit ABSENT for an off owner and calibration-before-record
are fixed by operator note B24 and the main lane's round-87 convention.
UNASKED choices: none.

## Prediction ledger

| prediction | status |
|---|---|
| HD7-P1 crash owner | **CONFIRMED by source / runtime UNMEASURED** |
| HD7-P2 active-runoff passivity | **UNMEASURED**; operator calibration required |
| HD7-P3 inactive-runoff schema | **CONFIRMED offline / NEMO record UNMEASURED** |
| HD7-P4 checker non-vacuity | **CONFIRMED offline**; five plants fire |
| HD7-P5 rung-5 completeness | **UNMEASURED** |
| HD7-P6 acquisition disposition | **CONFIRMED**; fresh targets absent, preflight clean |

## OPEN

1. Operator runs the committed launcher.  It must pass rung-6 calibration
   before it creates the rung-5 record.
2. The next round admits the two runs.  A failed rung-6 byte comparison means
   this repair does not land; a rung-5 failure after calibration names another
   defect and remains recorded.
3. Only after rung 5 admits: read the compiled shortwave fallback and construct
   rung 4.  Rungs 3 through 1 remain unbuilt; main-lane rung 0 remains out of
   scope.

Acquisition launcher:
`/tmp/autopilot-orca2-1824514632/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round7_acquisition/run.sh`.

Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung5/record_absent_v2`.

## UNVERIFIED

- The repaired NEMO target has not been built or run.
- Rung-6 frame/restart identity and rung-5 runtime success are unmeasured.
- The rung-5 480-frame census, month products, terminal restarts, and SHA-256
  admission are unmeasured.
