# Receipt — VORTEX_SMT round 15 (lane round 227): internal LDF acquisition

**Status: STOPPED_FOR_RECORD.**  No physics, card, carried state, trajectory,
or certified number changed.  The existing SMT-3 acquisition now has additive
self-describing seams for `ldf_slp` and `traldf_iso_lap`; its dry preflight and
focused controls are green.  The sandbox did not run NEMO, so the internal
first non-bit statement remains UNMEASURED and the operator-facing `run.sh` is
the only acquisition requested.

Base: `e052580833d6f55f165a9f9395bbe3bbc7687777` (round 226).
Preregistration: `184cb8805`.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round227/`.

## 1. Compiled program and magnitude

Round 226 measured the stage-3 post-LDF temperature boundary at
`2.0915088416728622e-07 K`, after a pre-LDF discrepancy of
`7.418332614861356e-11 K`; those are the magnitude rows this record is built
to split.  They are not reinterpreted here.

The admitted build computes density and N2, then calls the standard slope
producer before the RK3 stages in
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:159-177`.
The resolved tracer-diffusion selector calls `traldf_iso_lap` in
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90:105-110`.
Inside that program, the slope walk forms and bounds the U/V gradients, applies
the mixed-layer and Shapiro operations, and produces the W slopes in
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfslp.f90:186-338`.
The LDF operator computes A33/MSC in
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:780-829`, then
the tracer gradients, tensor factors, horizontal and vertical fluxes, and the
RHS divergence in
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:154-304`.
That source order, rather than a hypothesised owner, defines the next walk.

## 2. Acquisition delivered

The new target pair is
`VORTEX_SMT3_VEC_R15_OMIP_L1{,_P3}`.  It copies the admitted SMT-3 namelist,
cpp keys and ten-step cadence exactly.  The reference target carries no new
writer; the P3 target adds only three source patches and the read-only writer
module.  Each patch removes zero NEMO lines.

Two new records extend the existing record suite:

| record | named groups | purpose |
|---|---:|---|
| `oracle_ldf_slope_kt00000001.bin` | 16 | density/N2, live U/V thickness, raw and bounded gradients, mixed-layer values, filtered U/V/W slopes |
| `oracle_ldf_iso_kt00000001.bin` | 50 | A33/MSC inputs and outputs, tracer gradients, A11/A22/A13/A23, horizontal fluxes, A31/A32, vertical fluxes, geometry and RHS increment |

Both use the established `(name, rank, n1, n2, n3, payload)` stream to EOF.
The checker predicts neither a byte size nor a header tuple; it requires only
the magic and the registered names, then derives every payload length from its
own group.  It also rebuilds `rhs_after - rhs_before` exactly and refuses
non-finite payloads.  Header, field-name and truncation plants are registered.

Admission is fail-closed on the plain/instrumented step-10 restart comparison.
If the restart differs, the writer perturbs NEMO and the record is refused.
On success, each internal record receives a SHA-256/producer-commit/name stamp;
the script verifies all three fields before printing READY.  Every unexpected
shell failure prints a named `REFUSE` line.

## 3. Preregistered predictions

| prediction | result | verdict |
|---|---|---|
| R15-P1 one record extension, no physics | three patches are additions-only; deck/cpp/run settings are unchanged by construction | CONFIRMED by preflight; restart passivity awaits the record |
| R15-P2 sufficient seams | parser requires all 16 slope and 50 ISO group names | CONFIRMED structurally; payload values await the record |
| R15-P3 passive writer | plain/instrumented restart identity and in-run payload calibration | UNMEASURED pending acquisition |
| R15-P4 format and provenance | self-describing parser, three synthetic plants, commit-stamp check | CONFIRMED structurally; real stamps await acquisition |
| R15-P5 stop at acquisition boundary | no `mpirun`, trajectory score or physics verdict in this round | CONFIRMED |

No failed prediction is hidden: P1/P2/P4 are only structural until the NEMO
record exists; P3 is explicitly UNMEASURED.

## 4. Preflight, tests, review, and citations

The committed acquisition's `--preflight` output is retained at
`round227/preflight.log`.  Its decisive lines are:

```text
GFORTRAN_SYNTAX_PASS .../vortex_r18_tracer_terms.F90
GFORTRAN_SYNTAX_PASS .../vortex_r23_ldf_terms.F90
GFORTRAN_PATCHED_NEMO_SYNTAX_PASS ldfslp.f90
GFORTRAN_PATCHED_NEMO_SYNTAX_PASS traldf_iso.f90
PREFLIGHT_OK  variant smt3vecint: instrument and deck patches apply to the shipped sources
ROUND227_LDF_INTERNAL_PREFLIGHT_PASS .../round227/oracle_vortex_smt3_ldf_internal
```

The final focused CPU/fp64 parser, acquisition-control and receipt-citation
suite reported:

```text
34 passed in 3.58s
```

The receipt citation gate reports five citations, zero failures, zero
unmapped citations and zero audit failures.  Its planted shift of the compiled
A33/MSC span exits 1 and reports `SYMBOL-NOT-AT-LINE`; the control cannot pass
against its planted source displacement.

The required separate `codex exec --sandbox read-only` review was invoked on
the clean committed diff.  It exited 1 before sampling and printed, verbatim:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

It produced no shipping verdict: **independent review unavailable
in-sandbox**.  This environment failure is retained at
`round227/independent_review.log`; it is not replaced by an author review.

## 5. Blast radius and choices

No `packages/` or `src/` file changes.  No existing recipe or card executes
new arithmetic; therefore GYRE, DINO, the tanks, flat VORTEX and ORCA2 cannot
move in this round.  The new instrumented NEMO target is not a production
card.  DINO's month gate and the trajectory batteries are consequently not
applicable until a model statement is proposed.

**UNASKED list: EMPTY.**  Decision 93 already authorises the SMT-3 rung.  No
scheme, coefficient, time level, stabiliser, threshold, carried state or
default is selected.

## 6. OPEN

1. Operator: run the committed acquisition script.  Admission requires NEMO
   `STOP 0`, restart byte identity, exact RHS-increment calibration, all named
   groups, all three nonzero plants, and both commit stamps.
2. Next round: admit the record and first reproduce round 226's stage-entry,
   pre-LDF and post-LDF rows.  Walk the new seams in the compiled order above
   through the production-JIT closure.  Isolated eager/JIT rows are supporting
   evidence only.
3. The first seam non-bit given NEMO's preceding seam names the statement.  A
   production change may land only under the standing SMT/flat-VORTEX/tank,
   GYRE ladder/year, generic-card, DINO-month, citation, plant and independent
   review gates.
4. After SMT-3 is closed or explicitly held, continue Decision 93 with SMT-4
   lateral momentum diffusion.
