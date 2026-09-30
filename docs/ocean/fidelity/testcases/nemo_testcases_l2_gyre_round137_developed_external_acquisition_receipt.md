# NEMO testcase L2 GYRE phase 3 — round 137 developed external acquisition receipt

Date: 2026-09-21

Incoming lane tip: `e4854dfe7a5f4e2bdd4f949e10d5e01dc040b448`

Preregistration commit: `f7ba856dd`

Acquisition-script commit: `d0eadebc4`

Status: **STOPPED_FOR_RECORD — the required developed-state external-mode
record does not exist because the sandbox refused creation of the new NEMO
configuration with `Read-only file system`. No NEMO target, run directory, or
scientific measurement was produced. The additive writer, extended admitted
reader, fail-closed controls, and operator-ready acquisition are committed;
the next round must admit that record before interpreting any external-step
row.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round137/`

## Outcome first

Round 137 preregistered a compiled-order walk upstream of Round 136's
599-column QCO mismatch. The parent card's compiled program performs the
single external solve and then enters RK3 stage 1 at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3.f90:188-201`; its executing
dispatch calls `dyn_spg_ts` at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stp2d.f90:291-298`.

No existing admitted stream contains that developed solve. The round therefore
implemented a passive acquisition by extending the existing Round-81 binary
layout rather than adding another model stepper. It records all 50 external
substeps, appends the final external `pssh`, and records the immediately
consumed `ssha`, `r1_ht_0`, and `r3ta` in a separate stage-1 QCO stream. The
placement is source-bound: the external routine finalizes its accumulators and
assigns `pssh(:,:,Kaa)` at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/dynspg_ts.f90:797-827`, then stage
1 copies that field into `ssha` and calls `dom_qco_r3c_RK3` at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:150-180`. The
T-point compiled QCO statement is
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/domqco.f90:237-257`.

The first in-sandbox acquisition attempt reached `makenemo`, but the host
filesystem denied every creation under the NEMO `cfgs` directory. It created
neither `GYRE_OMIP_L2_P3_SM_R137EXT` nor
`round137/oracle_developed_external`. After the failed attempt, the run script
was tightened to check directory writability before invoking `makenemo` and to
verify the target directories immediately afterward. The retained retry exits
64 with the single named line:

```text
REFUSE: NEMO configuration directory is not writable in this sandbox
```

The campaign instruction requires an operator acquisition at this boundary;
no alternate build tree, inferred `pssh`, or scratch record was substituted.

## Frozen predictions: all still pending

No scientific prediction in the preregistration was measured. In particular,
this receipt makes **no** claim about the first non-bit external substep, the
599-column final-SSH set, or whether the model's production-JIT QCO multiply is
bit-exact. The frozen predictions remain unchanged for the admitted record:

| frozen item | round-137 disposition |
|---|---|
| one external record and one QCO record | `UNMEASURED — RECORD ABSENT` |
| step-1080 restart and step-1081 process record remain byte-identical | `UNMEASURED — RUN ABSENT` |
| external final `pssh` equals stage-1 `ssha` | `UNMEASURED — RECORD ABSENT` |
| NEMO `r3ta = ssha*r1_ht_0` replay is BIT | `UNMEASURED — RECORD ABSENT` |
| final model `pssh` differs in exactly the Round-136 599-column set | `UNMEASURED — WALK NOT RUN` |
| first model external mismatch is substep-1 slow momentum forcing | `UNMEASURED — WALK NOT RUN` |
| SSH first differs no earlier than substep 2 | `UNMEASURED — WALK NOT RUN` |

The Round-123 writer independently brackets the next-stage geometry in its
step-1081 process record at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-830`. The
operator script uses byte identity of that existing record, plus the
step-1080 restart, as its non-vacuous passivity admission.

## Record and control contract

The committed `run.sh` uses a new target name, copies the Round-123 `EXP00`,
`MY_SRC`, and preprocessor card file by file, applies two additive Fortran
patches, changes only `nn_itend` from 1440 to 1081, and proves both modified
translation units with `gfortran -fsyntax-only` before `makenemo`. It uses
Bash `SECONDS`, never `/usr/bin/time`.

The external stream retains the admitted 37 arrays for each of 50 substeps and
adds one 32-by-22 final-SSH array. Its exact expected size is 10,444,796 bytes.
The QCO stream has a 16-byte magic, seven 32-bit header integers, and three
36-by-26 fp64 arrays; its exact expected size is 22,508 bytes. The reader
rejects wrong magic, step, stage, dimensions, scalar kind, field count,
truncation, trailing bytes, non-finite values, time-level registration, and
commit stamps.

Controls actually executed before the stop:

| control | result |
|---|---|
| dry application of both additive source patches | PASS with zero removed Fortran lines |
| compiled-form syntax proof | `SYNTAX_PROOF_PASS dynspg_ts.f90`; `SYNTAX_PROOF_PASS stprk3_stg.f90` |
| writer/reader byte arithmetic | PASS: 10,444,796 and 22,508 bytes |
| source-layout plant | exit 69, `STATUS PLANT-FIRED: layout` |
| synthetic final-`pssh` one-ULP control | PASS: validator refused the planted row |
| synthetic QCO-result one-ULP control | PASS: validator refused the planted replay |
| sandbox acquisition retry | exit 64 with the named refusal above |

The record-header, truncation, replay, swap, final-SSH, QCO, commit-stamp, and
passive-admission plants embedded in `run.sh` are **not reported as fired**:
they require the missing NEMO record and will execute in the operator run.

## Campaign surfaces and headline numbers

No production implementation, recipe, card, carried state, default, or
stabilizer changed. Consequently no Decision-43/45 candidate exists and no
Rule-12 or year landing table was run. GYRE, DINO, LOCK_EXCHANGE, OVERFLOW,
the tanks, and ORCA2 are unchanged by this diagnostic-only commit.

For continuity, the incoming certified values—not remeasurements—remain:

| row | incoming value | this round |
|---|---:|---|
| kt2 U | `2.7377110452773967e-12 m/s` | unchanged; not rerun |
| kt2 V | `3.2849219221489645e-12 m/s` | unchanged; not rerun |
| kt3 T | `8.659373840202989e-7 K` | unchanged; not rerun |
| kt3 S | `7.027291104577671e-8 g/kg` | unchanged; not rerun |
| day-30 T3D RMS | `6.890431487825909e-5 K` | unchanged; not rerun |
| day-240 T3D RMS | `1.644674193e-2 K` | unchanged; not rerun |
| day-360 T3D RMS | `1.122357391e-2 K` | unchanged; not rerun |

## Verification and independent review

The final focused invocation covered the extended external/QCO gate, every
receipt-citation control, and the complete time-level registry unit file. Its
exact summary was `48 passed in 2.22s`. The clean-tree citation pass found six
citations, zero failures, zero unmapped citations, and zero failing map
entries. Shifting the final-`pssh` citation by two lines exited 1 with
`SYMBOL-NOT-AT-LINE`.

Retained evidence digests are:

| artifact | SHA-256 |
|---|---|
| `preflight_after_failfast.log` | `956eaa8465879a2ec6a7327e49e66ee0432a010e0be7be75bfe3e370a7e02093` |
| `layout_plant.log` | `560544dc32525db18084fd2471040aa79f15f28dcd249c4934340c43e9f7947f` |
| `acquisition_retry.log` | `a18526cea02650c9ef475a33a09ec4f9ef0a2da234f8c09f72b3b08f7cfee998` |
| `codex_review.log` | `8e3330eeb312ddb37b9da73c3ece4a8aa7e2ade4b7bb255573aa487bc89f0f58` |
| `focused_tests.log` | `bc57012b7f21161880d7e87a443f1ea3aab7af227f2931fa523d97fe8d4f5095` |

The required separate Codex command was invoked. Its complete verdict-bearing
output was unavailable because the read-only sandbox prevented its in-process
app-server client from initializing. Its output was, verbatim:

```text
Error while loading conda entry point: conda-anaconda-tos (cannot import name 'validate_prefix_exists' from 'conda.cli.install' (/home/dbalwada/miniconda3/lib/python3.13/site-packages/conda/cli/install.py))
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Independent review unavailable in-sandbox; the command emitted no verdict.

## OPEN — round 138 after acquisition

1. Run the committed Round-137 acquisition and require
   `ROUND137_DEVELOPED_EXTERNAL_READY`. Do not interpret either record unless
   the step-1080 restart and step-1081 process record are byte-identical to
   Round 123 and every embedded plant exits nonzero with its marker.
2. Re-anchor every citation to the new target's compiled `ppsrc` before
   claiming a measured statement. The parent-card citations in this stopped
   receipt prove the planned insertion boundaries, not the as-built target.
3. Extend the existing Round-82 production-JIT external walk using Round 136's
   admitted complete restart bridge. Compare all 50 substeps, final `pssh`,
   stage-1 `ssha`, and the QCO result. Run an independent ordinary production
   call and require returned-state bit identity.
4. Report the first non-bit boundary, every moved row, and exact set equality
   against Round 136's 599 wet columns. If final `pssh` is already non-bit,
   exonerate the ratio multiply only after exact-input production proof and
   continue upstream. Do not land a downstream QCO, FCT, EVD, or TKE change.
5. Preserve every frozen prediction and falsifier from the Round-137
   preregistration. Failed predictions must remain labeled `REFUTED`.

`ACQUISITION_NEEDED`:
`/tmp/autopilot-work-syZnXCH4/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round137_developed_external/run.sh`

`DECISION_NEEDED`: `NONE`

`ROUND_STATUS`: `STOPPED_FOR_RECORD`
