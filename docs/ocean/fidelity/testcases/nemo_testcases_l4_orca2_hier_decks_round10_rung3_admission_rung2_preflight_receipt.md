# ORCA2-DECKS round 10 receipt — rung 3 admitted, rung 2 preflight

Date: 2026-10-01

Disposition: **STOPPED_FOR_RECORD**.  The operator-produced rung-3 run is
complete and admitted after two admission-only checker repairs.  The rung-2
deck, gate, manifest, and fail-closed acquisition are preflight-clean.  The
rung-2 oracle record does not yet exist.

Base: `582485ac9`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round10.md`, commit `8abee0ba3`,
before this round read the operator log, inspected the rung-3 target, repaired
either checker, admitted rung 3, or constructed rung 2.  Every run claim is
labelled **independent**: NEMO starts from that rung's own from-rest
initialization.

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
| 3 | rung 4 with exact-zero surface fluxes and no restoring/freshwater budget | 480 version-2 frames + 240-step month | **ADMITTED** this round |
| 2 | rung 3 without GM eddy-induced velocity or mixed-layer eddies | absent | **STOPPED_FOR_RECORD**; needs 480 header-valid frames, finite T/U/V/W month files, and finite fp64 step-240 ocean restarts |
| 1 | not built; rung 2 minus BBL/geothermal | absent | **UNMEASURED WITH SPEC** |
| 0 | main-lane owned | not compared | **OUT OF SCOPE** |

Every rung retains `ln_spc_dyn=.true.` exactly as the source deck carries it;
the reused binary has no `key_agrif`, so the assignment remains inert.

## Loud correction and rung-3 admission

The operator's NEMO run did not fail.  Its only stdout lines at the wrapper
level were `STOP 0` and the line-108 refusal.  The target has all 480 operand
frames (240 per rank), both step-240 restarts, and all eight month files.  The
refusal occurred in the admission loop after NEMO exited successfully.

The first failed plant, `terminal-nonfinite`, correctly raised the rung-9
terminal validator's `GateError`, but the rung-3 command-line gate did not
catch that deeper inherited exception class.  It emitted a traceback instead
of the required `STATUS PLANT-FIRED` marker.  A regression test raises that
exact class through `main()`: it failed before the catch-list repair and passes
after it.

The repaired plant loop then exposed a clean-gate defect.  Rung 3 resolves the
freshwater-budget selector to zero, so its compiled step routine does not call
the budget routine (`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:517-521`).
The inherited no-ice validator nevertheless required a print emitted only by
that inactive routine.  The repaired predicate reads the resolved selector:
nonzero requires the no-ice volume-control print; zero requires its absence.
The active upper rungs retain the original requirement, and a planted inactive
record carrying the forbidden print is rejected.

All eighteen admission plants then fired and the clean gate printed
`PASS_RUNG3_RECORD`.  The independent rung-3 record contains:

- 480/480 self-describing frames, 3,840 finite PRESENT fields, and 960 ABSENT
  fields, exactly `rnf` and `rnf_tsc` absent in every frame;
- an fp64 exact-zero five-field surface-flux file with its SHA-256 pinned;
- two finite fp64 step-240 ocean restart shards and no ice product;
- eight finite T/U/V/W month files with 86 floating variables; and
- a complete 542-regular-file SHA-256 inventory.

No record product was regenerated.  HD10-P1 through HD10-P3 are
**CONFIRMED**.

## Compiled rung-2 boundary

The compiled EIV initializer reads `ln_ldfeiv`; when false it takes the explicit
not-used arm, while allocation occurs only in the true arm
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/ldftra.f90:572-614`).
At stage 3, tracer advection calls the EIV transport and MLE transport under
their two independent selectors
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/traadv.f90:253-262`).
The MLE initializer likewise reads `ln_mle`, prints the explicit not-used arm
when false, and allocates its working arrays only in the true arm
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/tramle.f90:643-684`).

Thus Decision 80's rung-2 boundary has one source reading and needs no CPP-key
or binary change.  The main-lane rung-0 deck independently carries both false
selectors.  HD10-P4 is **CONFIRMED**.

## Complete rung-2 namelist delta

Every physical line differing from rung 3 is printed below.  Retained EIV and
MLE parameters are inert behind their false selectors and remain byte-exact.

```diff
-   ln_mle      = .true.   ! (T) use the Mixed Layer Eddy (MLE) parameterisation
+   ln_mle      = .false.   ! (T) use the Mixed Layer Eddy (MLE) parameterisation
-   ln_ldfeiv   = .true.    ! use eddy induced velocity parameterization
+   ln_ldfeiv   = .false.    ! use eddy induced velocity parameterization
```

The exact rung-2 deck SHA-256 is
`fa8d0a34d6f3bce8cce4c98ae56fefa9d2a0f5e51e4ba582aa695362154d75b7`.
All other parsed assignments, inputs, CPP keys, repaired record binary,
from-rest mode, two-rank layout, 240-step length, output cadence, exact-zero
surface file, and retained unread TKE sentinel are unchanged.

The committed preflight gate pins the admitted rung-3 record, all three
compiled source files, exact two-assignment delta, binary/build/input
identities, and run protocol.  Its unrelated-deck, source-pin, and
upper-admission plants fire.  The committed launcher stages the exact deck
idempotently and prints `ORCA2_HIERARCHY_RUNG2_PREFLIGHT_READY` without
invoking NEMO.

## Validation, review, and scope

- Focused hierarchy rounds 1 through 10 plus citation controls: **106 passed**.
- The receipt citation gate passes with three mapped citations, zero failures,
  and zero unmapped citations; its shifted-line plant fires.  The cumulative
  default-receipt gate passes with 274 citations and zero failures or unmapped
  citations.
- The one allowed `tests/ocean/fidelity -n 12` battery selected 2,266 tests,
  reached 99%, and was interrupted after the documented late-suite no-output
  stall.  The two visible failures are known pre-existing reds: the worktree-
  stamp ratchet and stale GYRE round-51 trace assertion.  No second broad
  battery ran.
- Ruff, Python compilation, shell syntax, and diff whitespace: **PASS**.
- The required separate `codex exec --sandbox read-only` review failed before
  reading the diff: `failed to initialize in-process app-server client:
  Read-only file system`.  Verdict: **independent review unavailable
  in-sandbox**.
- No file under `packages/` or `src/` changed.  GYRE is byte-identical by
  construction; its year gate was not rerun.
- No NEMO source tree was modified.  The rung-2 launcher reuses the admitted
  repaired binary and requests no rebuild.

ASKED choices: rung ordering and removal of GM eddy-induced velocity plus MLE
are fixed by Decisions 79/80 and side-lane note B23.  UNASKED choices: none.

## Prediction ledger

| prediction | status |
|---|---|
| HD10-P1 failure boundary | **CONFIRMED**; NEMO reached `STOP 0`, admission plant reporting failed |
| HD10-P2 existing-record disposition | **CONFIRMED**; all plants fire and the completed record admits in place |
| HD10-P3 repair scope | **CONFIRMED**; two checker-only repairs, no deck/schema/threshold change |
| HD10-P4 rung-2 boundary | **CONFIRMED**; exact two-selector namelist-only delta |

## OPEN

1. The operator runs the committed rung-2 launcher.  Rung 2 remains
   UNMEASURED until all nineteen admission plants fire and the clean gate
   admits 480 frames, exact-zero input, month products, terminal restarts, and
   the complete SHA-256 inventory.
2. After rung 2 admits, construct rung 1 by removing only `ln_trabbl` and
   `ln_trabbc`, with every necessary coupled line source-resolved and printed.
3. Main-lane rung 0 remains out of scope; when its deck exists, diff it against
   admitted rung 1 as required by note B23.

Acquisition launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round10_acquisition/run.sh`.

Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung2/record`.

## UNVERIFIED

- The rung-2 NEMO run has not been executed.
- Its 480-frame census, resolved inactive GM/MLE arms, month products,
  terminal restarts, and SHA-256 admission are unmeasured.
