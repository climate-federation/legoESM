# Grid ladder rung G9 — round 7 resumed receipt (gridladder_r52)

Status: **LANDED KT-1 FIX; G9 NOT CERTIFIED; FOLLOW-UP ACQUISITION NEEDED.**
The admitted `trazdf` record rules out the matrix, solution, stage update, and
end-of-stage fold as owners of the registered 105 T / 96 S kt-1 residue.  The
first owning category is the **RHS terms**: the tracer advection RHS used a
vertical transport formed with a re-derived Kmm T ratio.  Routing NEMO's
stored stage ratio removes the complete scored kt-1 residue.  A two-label
kt 1..10 rescore nevertheless finds the first remaining debt at kt 4, so G9
and its proposed ORCA2 fold import are not certified.

- codex review: operator
- independent review: operator (Claude)
- evidence: `gridladder_rounds/round52/`
- implementation: G9-selected tracer WZV Kmm ratios only

## 1. Admitted record and owner

The operator run completed with 42 / 42 pre-existing records byte-identical
and admitted the four-frame
`oracle_g9_trazdf_zrhs_kt00000001.bin`.  The record SHA-256 is
`f4e1dfcc27ef8cc106973aa56d92763f75cb1873aecd87d080a3427493fa5c32`.
All ten arrays are finite and read completely.

The recorded `zrhs_content` is bit-identical on every wet level for T and S
to the source-ordered reconstruction from the passivity-identical round-46
Kbb tracer, after-qsr Krhs, Kbb/Kmm thickness ratios, and the 10800 s step.
That is the recurrence at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:284-290`.  The recorded matrix is
bit-identical to the prior matrix operands, the recorded solution is
bit-identical to the prior stage-3 Kaa tracer, and substituting the recorded
content closes the solve.  The first owner in the registered choice list is
therefore **RHS terms**, not `trazdf`'s zrhs, the matrix, the solution, the
stage update, or the end-of-stage fold.

The specific upstream RHS defect is in the tracer-owned continuity solve.
NEMO stores the stage-2 N+1/3 ratios at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:177-179`
and the stage-3 N+1/2 ratios at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:221-223`.
The transport-form divergence divides by stored `r3t(Kmm)` at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/divhor.f90:132-138`, and
the bottom-up W recurrence consumes that divergence at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/sshwzv.f90:297-298`.
The card had instead rebuilt Kmm's ratio from the real-equal interpolated sea
level.

## 2. G9-only fix and exact control

The new default-off configuration field selects stored Kmm ratios only for
G9's tracer-owned stage-2 and stage-3 WZV calls.  The shared helper receives
the stage index and forms NEMO's already-existing source-ordered ratio blend;
no other transport operand or consumer changes.  G5 is false, G9 is true,
and the selector census proves every default false.

The x64 value test uses an ordinary production `jax.jit` step with traced
state and forcing.  It reconstructs NEMO's recorded `zrhs_content` exactly on
all wet levels, proves the record's downstream solution equals the earlier
passive Kaa frame, and proves G9's scored level-1 T/S result equals that
solution bit-for-bit.  Removing only the new selector restores exactly 105 T
and 96 S unequal values.  The test passes.

The option is a static G9-only branch.  The 26 non-G9 cards retain their
round-50 address-masked jaxprs; G9 alone reaches the new call arguments.
The measured comparison reports 26 byte-identical graphs and zero different.
Its current-output and comparison-summary SHA-256 hashes are respectively
`43bbaae929734c09a725e0a88b998caadec361ece0eb31c31b9bae527a4cfa11`
and `cae0425898533f43f389659d4bb9061f2642c7e38ead6688e2401de42d03ac74`.

## 3. Both-label kt 1..10 rescore

The rescore used CPU production JIT, x64, and the admitted round-46 restart
and forcing roots.  Rows are ssh, `uu_b`, `vv_b`, u, v, T, and S.  Complete
evidence and hashes are:

- independent: `resume/g9_independent_kt1_10.{json,log}`, JSON SHA-256
  `41d5a48e085c750afc9cff4c0734e1b7637e0af5288b57a08389002d562cf18a`;
- given entry: `resume/g9_given_entry_kt1_10.{json,log}`, JSON SHA-256
  `47a7009913f5347367d26fced938c52398f16dffabed821740b000873d7fea62`.

| label | kt | exact / AT-BAR / DEBT | rows over the bar |
|---|---:|---:|---|
| independent | 1 | 7 / 0 / 0 | none |
| independent | 2..3 | 5 / 2 / 0 each | none |
| independent | 4 | 0 / 6 / 1 | ssh |
| independent | 5 | 0 / 2 / 5 | ssh, `uu_b`, `vv_b`, u, v |
| independent | 6..10 | 0 / 0 / 7 each | all seven rows |
| given entry | 1 | 7 / 0 / 0 | none |
| given entry | 3 | 5 / 2 / 0 | none |
| given entry | 5 | 6 / 1 / 0 | none |
| given entry | 7 | 5 / 2 / 0 | none |
| given entry | 9 | 6 / 1 / 0 | none |
| given-entry two-step | 2 | 5 / 2 / 0 | none |
| given-entry two-step | 4 | 0 / 6 / 1 | ssh |
| given-entry two-step | 6, 8, 10 | 3 / 4 / 0 each | none |

The first debt is the same kt-4 SSH row in the independent and given-entry
two-step labels, normalised maximum `1.211064769032015e-15`.  The admitted
kt-2 stage frames establish an exact stage 1 and first show three interior
one-ULP T differences in stage 2, before the subsequent two-step amplification.
They do not contain the stage-2 internal transport/RHS boundaries needed to
name another cited owner.

## 4. Certification and ORCA2 import

G9 is **not certified** because both labels contain the kt-4 debt row.  Under
Decision 122, this certifies **none** of the proposed ORCA2 fold import yet.
The full ORCA2 build executes the corresponding stored-ratio and divergence
statements at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:177-179`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221-223`, and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/divhor.f90:132-138`, but the lane-wide import
remains withheld until every G9 row is exact or AT-BAR.

## 5. Follow-up passive acquisition

The committed operator-only tool `run_r52_stage2_tracer_walk.sh` has read-only
default preflight, `--run`, and `--admit-existing` modes.  At kt 2, stage 2 it
records five frames: zFu/zFv/zFw transport, stored Kbb/Kmm/Kaa T ratios,
after-advection T/S RHS, after-SBC T/S RHS, and the final T/S stage update.
These boundaries will distinguish WZV/transport, centred-advection RHS, SBC,
and update without fitting an unobserved value.

The observer is additive and WRITE-only.  Admission requires a successful
two-step run, all 12 arrays finite and complete, no trailing bytes, exactly
one new record, and all 42 prior records byte-identical.  Preflight passed its
synthetic Fortran layout reader, truncated-final-frame plant, and
non-additive-source plant, ending with:

`G9R52_STAGE2_PREFLIGHT_OK base=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1 target=ORCA2G9_OMIP_L4_R52STG2 commit=2c73ac2560f6ba91920856de66fe7d225459727c`

The preflight log SHA-256 is
`48dc9e74caad0f9953c7bc9725fb17013a0cfee36c2e999e7e75ff63777ae53e`.
No NEMO build or run occurred in the agent session.

## 6. Gates and disposition

The receipt-specific citation gate found 8 citations, 0 failures, 0 unmapped
citations, and 0 failing map entries.  Its complete unit suite passed 17 / 17.
The combined default gate and private-import ratchet, run with CPU x64 and the
prescribed `PYTHONPATH`, passed 144 / 144 in 1045.90 s.  The combined log
SHA-256 is
`1fd7b5f638637a2b326519c03ec98c5608f80f124b41ca47735b1a612f71ab4e`.

ASKED: admit and diagnose the kt-1 stage-3 `trazdf` content record; fix the
first cited owner G9-only; exact-test selected and removed wiring; rescore both
labels kt 1..10; preserve all 26 non-G9 graphs; state the ORCA2 import result;
write and preflight another acquisition if the certification boundary needs
one; run the gates; commit.

UNASKED, offered for revert: none.

UNVERIFIED: the internal owner of kt-2 stage-2's three interior T differences.
It is the exact boundary requested by the follow-up acquisition and is not
represented as measured.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r52/repo
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r52/repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_g9_orca2tadv/run_r52_stage2_tracer_walk.sh --run
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — kt-1 RHS residue fixed exactly; G9 remains uncertified at kt 4 pending the preflighted kt-2 stage-2 tracer walk.
