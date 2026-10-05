# Receipt — VORTEX_SMT round 13 (lane round 225): SMT-3 acquisition retry

**Status: STOPPED_FOR_RECORD.**  The operator-run command exited zero but did
not run NEMO: it selected the wrapper's dry default.  This round retracts the
reported acquisition, makes the operator-facing default execute both arms,
retains dry validation as explicit `--preflight`, and returns the same target
for acquisition.  No model, physics, card, record, or trajectory changed.

Base commit: `0dee7cd1c06abcb525fea6a036c297d2eb1cf26f` (round 224).
Preregistration commit: `4e82a0f46`.
Implementation commit: `8f3372367`.
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round225/`.

## 1. Retraction and diagnosis

The operator report said the acquisition exited zero.  The complete supplied
log has only 85 lines.  Its two driver invocations end with `DRY RUN. Re-run
with --run`, and its final line is
`ROUND224_SMT3_PREFLIGHT_PASS .../oracle_vortex_smt3`.  There is no `makenemo`,
`mpirun`, NEMO `STOP 0`, admission JSON, or
`ROUND224_SMT3_RECORD_READY` line.  At the initial inspection, the round-224
evidence root also had no `oracle_vortex_smt3` directory.  Round 225's explicit
preflight subsequently created that empty parent, but neither `kt1_10` nor
`day100` exists and no record or admission was produced.

**RETRACTED:** “the round-224 SMT-3 acquisition ran successfully.”  What ran
successfully was the dry source/deck preflight.  Round 224 predictions P2--P6
remain UNMEASURED.

The cause is direct and one-variable.  The committed wrapper initialized its
run selector false and required the extra `--run` argument; the operator
protocol executed the returned path without arguments.  Empty arguments now
select acquisition.  `--run` remains an accepted explicit spelling, and the
old dry behavior is available only as `--preflight`.  No line below the mode
selection changed.

## 2. Preregistered predictions

| prediction | result | verdict |
|---|---|---|
| R13-P1 one-variable wrapper repair | diff changes only default/mode parsing; targets, decks, patches, admissions, formats and evidence roots are unchanged | CONFIRMED |
| R13-P2 explicit preflight | both variants print `PREFLIGHT_OK`; final line is `ROUND224_SMT3_PREFLIGHT_PASS`; no arm/record appears, but the shared driver creates the empty parent evidence directory | **REFUTED as written**: its no-mutation falsifier fired |
| R13-P3 acquisition still needed | no admitted record exists before or after preflight | CONFIRMED |
| R13-P4 controls | shell syntax passes; 9 focused tests pass; citation control results are in section 5 | CONFIRMED |

R13-P2 is kept as a failed prediction.  The empty parent is harmless because
the acquisition refuses only an existing `kt1_10` or `day100` arm without its
admission; neither exists.  The first preflight attempt correctly refused the
dirty tree before the wrapper/test commit; the clean committed-tree rerun is
the result used above.

## 3. Executed control, including non-vacuity

The focused test copies the real wrapper into a temporary git repository and
executes it against a stub shared driver.  With no arguments the stub receives
exactly `--variant smt3vec --run` followed by
`--variant smt3vec100d --run`, produces two temporary admission markers, and
the wrapper prints `ROUND224_SMT3_RECORD_READY`.  With `--preflight`, the stub
receives the same variants without `--run`, creates only the empty parent (the
shared driver's measured behavior), and the wrapper prints its preflight
marker.  The no-argument assertion fails on the superseded wrapper, so it
controls the defect rather than inspecting a source token.

The focused suite's decisive line is:

```text
============================== 9 passed in 0.14s ===============================
```

The final combined focused suite, including all receipt-citation-gate tests,
reported **26 passed in 3.50s**.

The committed-tree preflight ends:

```text
PREFLIGHT_OK  variant smt3vec100d: instrument and deck patches apply to the shipped sources
ROUND224_SMT3_PREFLIGHT_PASS /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round224/oracle_vortex_smt3
```

It re-runs gfortran syntax validation for the additive writer.  The requested
branch remains the already-cited laplacian coefficient-mode-20 call in
`VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:354-390`; this receipt
does not claim that the absent SMT-3 compiled target has executed it.  The
round following acquisition must cite the new target's own ppsrc.

## 4. Scientific and landing table

| required row | round-225 result | verdict |
|---|---|---|
| SMT-3 kt=1..10 registry | no record, card, or run | UNMEASURED |
| SMT-3 100-day trajectory | no record, card, or run | UNMEASURED |
| first post-LDF boundary | no record | UNMEASURED |
| GYRE ladder/year | no production change; not rerun | UNCHANGED BY CONSTRUCTION |
| SMT-1/SMT-2, flat VORTEX, tanks, generic card | no production change; not rerun | UNCHANGED BY CONSTRUCTION |
| DINO month | no production change; not rerun | UNCHANGED BY CONSTRUCTION |
| ORCA2 | no code/card change | UNMEASURED-with-spec |

No first non-bit statement is named, no Rule-12 or Decision-43/45 row moved,
and no landing is attempted.

## 5. Citation gate and review

The receipt citation gate is run on this receipt from section 3.  Its one
compiled-source citation is already path-pinned to the SMT-2 build and must
resolve without an unmapped citation.  Shifting that citation by two lines is
the plant.  The real receipt reported **PASS, 1 citation, 0 unmapped, 0
failures**.  The shifted plant exited **1** with
`SYMBOL-NOT-AT-LINE`, so it fires.

The separate read-only review command exited 1 before reading the commits.  Its
verbatim result is:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

**Independent review unavailable in-sandbox.**  No `SHIP`, `HOLD`, or
`DO NOT SHIP` verdict is invented.  This is a wrapper/interface correction and
record stop, not a physics landing.

## 6. Choices and OPEN — round 226

**UNASKED list: EMPTY.**  No scientific option, threshold, target, record
format, target name, or carried state changed.  Reusing the target names is
safe because the dry invocation neither built nor ran them.

The operator runs the committed wrapper with no arguments.  Round 226 then:

1. admits the record only if restarts are byte-identical, every named group
   parses, all plants fire, both runs reach `STOP 0`, and the resolved tuple is
   exact;
2. cites the new target's compiled source, builds the explicit SMT-3 card, and
   proves geometry/initial-state identity to SMT-2;
3. scores the 10-step and 100-day arms, registers every row, and walks the
   first post-LDF boundary exactly as round 224's OPEN section specifies.

No new NEMO target is requested: the existing names remain unused.
