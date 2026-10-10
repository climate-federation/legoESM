# ORCA2 round 218 — OMT-2 linear-drag acquisition handoff

Date: 2026-10-10. Frozen base: `cc4226e1a`. Preregistration commit:
`5839744f8`. Acquisition implementation commit: `10c36e2e8`. Status:
**STOPPED_FOR_RECORD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round218/`.

This round changes no package model file, shipped ORCA2 card, carried state,
stabiliser, sea-ice selector, or `unmeasured_features` tuple. It creates the
Decision-109 OMT-2 deck and a fail-closed operator acquisition. No NEMO run was
attempted in the sandbox. OMT-2 trajectories are therefore **UNMEASURED WITH
SPEC**, and no independent/given-entry numbers are mixed or reported.

## Exact OMT-1 to OMT-2 edge

The canonical deck is rendered from the admitted round-211 OMT-1 deck, SHA-256
`d0cccebd76c9c4631c91e51a1af3a9b552d97f307c7c523020160c883fb5e73f`.
The gate reports exactly one changed assignment and one removed override:

| assignment | admitted OMT-1 | OMT-2 | provenance |
|---|---:|---:|---|
| `namdrg.ln_lin` | `.false.` | `.true.` | rung-0 shipped module |
| `namdrg.ln_drg_off` | explicit `.true.` | absent, resolving `.false.` from `namelist_ref` | rung-0 shipped module |

All momentum-LDF, tracer-advection, and tracer-LDF OFF selectors remain
unchanged. The rendered deck SHA-256 is
`bc5fc68188e4d79479acce38085e497f9f2543a36d6544f1dc1cea95bf51d169`.
The deck report finishes `PASS_R218_OMT2_DECK`, SHA-256
`04caa134872711fd4f02f4a2669547f87a90b7e6cb759e9464fde702da9d31fe`.
Both the extra-delta and wrong-selector plants exit nonzero.

The compiled source first constructs constant linear drag at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/zdfdrg.f90:258-284` and selects the
linear initialisation at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/zdfdrg.f90:540-547`. With implicit
split-explicit friction, NEMO removes/restores the barotropic component at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:158-169`, inserts the U
and V partial-cell diagonals at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:303-312` and
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:470-474`, and initialises
the split-explicit face coefficients and baroclinic residual at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1404-1463`.

Search-before-build found the same full composition already implemented and
selected by the seamount SMT-2 card: `nemo_linear`, the resolved coefficients,
implicit 3-D matrix drag, baroclinic-only residual, and barotropic substep drag.
No duplicate numerical implementation was added. The OMT-2 card and ladder
remain downstream of record admission.

## Acquisition contract

The launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round218_omt2_frames_acquisition/run.sh`.
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
kt=10; an earlier boundary refuses the entire record. The preflight finishes
`PASS_R218_OMT2_FRAME_PREFLIGHT`, SHA-256
`3b0edcb1def498b7128ddddd891609e349e34e3f7d898a7a81981946acc73366`.

## Preregistered predictions

| ID | disposition |
|---|---|
| R218-P1 | **CONFIRMED**: exact one-module deck edge; both deck plants fire. |
| R218-P2 | **UNMEASURED WITH SPEC**: ten-step NEMO calibration/twins await the operator. |
| R218-P3 | **UNMEASURED WITH SPEC**: the 96-step completion prediction awaits the operator; any earlier compiled boundary will be retained as REFUTED. |
| R218-P4 | **UNMEASURED WITH SPEC**: no OMT-2 card or entry trajectory was scored from OMT-1 data. |
| R218-P5 | **UNMEASURED WITH SPEC**: the complete vector unit was not scored without the OMT-2 record. |
| R218-P6 | **PARTLY CONFIRMED**: deck and inventory plants fire in preflight; record-content plants await real output. |

## Validation and review

The citation gate passes all six compiled-source citations with zero unmapped
citations and zero map-audit failures. Its explicit `zdfdrg.f90:258-284`
two-line shift plant exits 1 with `FAIL`; the cumulative default receipt also
passes with zero unmapped citations. The retained artifacts are
`citation_gate.json`, `citation_gate_plant.json`, and
`citation_gate_default.json` under the evidence root. The citation gate's
focused tests pass 17/17 (log SHA-256
`7f8720e3b77e4ca05d21ca77bf66f5af4a32d9dd5c50faa2b2973503e9be6ebd`).

Focused round-218 tests pass 7/7. The one prescribed
`tests/ocean/fidelity -n 12` battery collected 3,056 tests and reached 99%.
After no pytest process remained, the retained controller was bounded. It is
not called a full PASS: 3,035 passed, seven skipped, four registered
pre-existing failures emitted verdicts, and ten remained unclassified. The
four failures are the GYRE round-129 spread-floor record stamp, allow-dirty
scope, worktree-stamp ratchet, and SI3 scalar-math provenance gate. Log
SHA-256:
`e50d96aaffc4a1f6f90cf9c692542448d08e1eb6bf3b5397ff8e5c9c101044e2`.

Independent review was attempted separately with `codex exec --sandbox
read-only` and exited 1 before reading the diff: `failed to initialize
in-process app-server client: Read-only file system (os error 30)`.
Independent review is unavailable in-sandbox; this is not a PASS. Log SHA-256:
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

1. The operator runs the committed launcher with `--run`. The expected target
   is
   `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round218/acquisition/orca2_omt2_frames_10step_a_np2`.
2. Re-admit the record and disposition R218-P2/P3. Never rerun NEMO for a
   checker mismatch; fix the checker against the self-described header and use
   `--admit-existing`.
3. Build the gate-local OMT-2 card from OMT-1 plus the already-shared exact
   linear-drag composition, run both labelled kt=1..10 ladders, and score the
   complete round-217 vector unit atomically. Re-test any qualifying statement
   on independent rung 0 before a production landing.
4. OMT-3 waits. No configuration decision is requested.

ASKED choices: Decisions 103, 109, and standing Decision 96. UNASKED choices:
empty.
