# NEMO-testcases L2 GYRE round 71: FCT Kmm input receipt

Date: 2026-09-12. Final disposition: **STOPPED FOR RECORD / MAGNITUDE
PREDICTION REFUTED; no production physics change landed**. Oracle producer
`3b3b045bd9e03b60330204e7590e4c4470b7a0ca`; 132 values admitted (43 of 63
inherited records exact and 20 changed).

## Verdict and first non-bit boundary

The preregistered Kmm input substitution **REFUTED** Kmm as the magnitude owner
of the round-70 FCT/LDF cancellation. Supplying the recorded NEMO stage-3 Kmm
to the verified LDF-only arm changes the kt3 T maximum only from
`8.916073106490785e-7 K` to `8.916073035436511e-7 K`, and changes the S maximum
from `7.235656340753849e-8` to `7.235658472382056e-8`. It removes only 3 of the
1,375 retained T cells and 14 of the 1,891 retained S cells, not the frozen 50%
for either tracer. The twofold maximum-error prediction also fails for both.

Kmm itself is conclusively non-bit: 17,994 of 18,000 T cells differ with
maximum `8.369461070856232e-7 K`, and 16,769 S cells differ with maximum
`6.794565621248694e-8`. It is the first unequal input boundary in the
preregistered FCT source walk, but it is not an eligible implementation and is
not the large-error owner. The compiled FCT first guess consumes Kmm while
building its upstream face transports and tracer content
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:503-510`,
`:532-540`), after the tracer RHS is cleared and advection is called
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-828`, `:860`).

The next upstream producing association is mechanically identified but cannot
yet be replayed. The stage-2 thickness-weighted update writes Kaa
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:899-901`), and the
following time-level swaps pass that result as Kmm to stage 3
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:204-215`). Existing
records contain the stage-3 Kmm result but not kt=2 stage-1/stage-2 tracer RHS
and update operands. Therefore the stage-2 statement itself is **UNMEASURED**,
not inferred non-bit, and source order forbids moving on to FCT transport or
metric inputs. This round stops for that exact record.

## Frozen causal measurement

The clean report `round71_kmm_before.json` is stamped to instrumentation commit
`586458f3c3f3313c03e8e8770a39908a346c62fb`, has status **REFUTED**, and has
SHA-256 `7fb384f9cb28a59cfff2631a764b5080472a7e3c698f5b630a3f4ef6abcbae07`.
All live content, geometry, Kmm, and admitted arrays are float64. Every active
arm moves its selected tracer; cross-tracer isolation, selective/joint
identity, exact injected targets, exact same-step LDF, finite values, and exact
reproduction of all round-70 rows and criteria pass.

| tracer and arm | content max versus NEMO | kt3 max versus NEMO | kt3 movement from LDF-only |
|---|---:|---:|---:|
| T LDF-only | `5.954039670541533e-5` | `8.916073106490785e-7` | exact reference |
| T Kmm+LDF | `5.954045877842873e-5` | `8.916073035436511e-7` | 14,980 cells; max `7.368072374447365e-9` |
| T round-70 output pair | `5.885129041871551e-7` | `5.760972143775689e-8` | remains better |
| S LDF-only | `7.651457053725608e-6` | `7.235656340753849e-8` | exact reference |
| S Kmm+LDF | `7.651471605640836e-6` | `7.235658472382056e-8` | 10,908 cells; max `1.9028334463655483e-10` |
| S round-70 output pair | `2.0635297914850526e-8` | `6.410431296899333e-9` | remains better |

The T worsening set changes from 8,263 to 8,261 cells; 1,372 of the frozen
1,375 remain and one of round 70's 1,800 newly worsened cells is also worsened
by Kmm. The S set changes from 5,831 to 5,823; 1,877 of 1,891 remain and 8 of
the 6,374 new cells are worsened by Kmm. Aggregate movement therefore does not
waive the cellwise Rule-12 failure.

Both preregistered controls fail closed. The null-Kmm plant makes both active
content and active state movement predicates false and exits nonzero
(`plant_kmm_null.json`, SHA-256
`7e5cd6a41749bb573636c5a002ed868dcdbcafaae6aa052a0842029d2a520117`).
The content-ULP plant creates exactly one unequal T target cell with maximum
`3.552713678800501e-15 K`, makes the exact-target predicate false, and exits
nonzero (`plant_kmm_content_ulp.json`, SHA-256
`025a57a01ec2c35cc7181a1bdd9fbfe8891c3c1672592f9eac6e8762021a2a52`).
Inherited dry-cell division warnings are diagnostic only; every scored row is
wet-cell masked.

## Acquisition card

The requested next record is fully prepared at
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round71_fct_stage2/run.sh`.
It creates the new target `GYRE_OMIP_L2_P3_SM_R71FCTST2` from GYRE_PISCES,
copies the admitted round-64 EXP00 and MY_SRC cards file by file, changes no
namelist row, and applies one instrumentation-only patch to the existing
tracer writer. Four write predicates are widened through kt=2; no physical
statement is removed or replaced. The card records both kt=2 stage-1 and
stage-2 zero RHS, transports, post-advection/post-SBC RHS, Kbb/Kmm/Kaa and r3t
levels. Its independent gate requires exact stage bridges and has stamp,
one-ULP bridge, and truncation plants. Twin admission permits only the two new
record names and requires the final restart and mesh mask bit-identical.

The exact dry-applied source passed `gfortran -fsyntax-only` with the R64 build
includes; the preprocessed file contains all four widened predicates. Per the
campaign prohibition, this agent did not invoke makenemo or mpirun. The
operator must run the committed card before the next walk.

## Rule 12 card

| card | changed statement | Rule-12 disposition |
|---|---|---|
| GYRE | none in production; private Kmm causal instrument and a write-only acquisition card | **REFUTED before trajectory**: the twofold and 50% gates fail, so kt1--10 and days 1--30 are **UNREACHED** |
| LOCK_EXCHANGE | none | tracer FCT Kmm association is not reached by the tank's active statement path; production behavior is unchanged |
| OVERFLOW | none | tracer FCT Kmm association is not reached by the tank's active statement path; production behavior is unchanged |
| DINO | none | **UNREACHED**. The shared tracer/FCT statement is live there, so the measured cancellation and per-cell worsening are explicit risk; a future faithful edit requires DINO's own row gate. |
| ORCA2 | none; native stage record absent | **UNMEASURED WITH SPEC**: resolve its compiled configuration; record stage-1/2 Kbb, Kmm, Krhs, Kaa, transports and r3t, plus stage-3 complete FCT inputs/output; replay the update and each FCT input in source order; require exact statement replay, every moved row registered, no AT-BAR loss, and no earlier first-over-bar boundary. |

No AT-BAR row moved because no production statement changed. The first-over-bar
boundary cannot move. No configuration/default, carried state, stabilizer,
NEMO source/build/run, year harness, reconciliation gate, freshwater pair,
#1484 guard, or held manifest changed.

## Review and focused checks

The required separate review was invoked on clean acquisition commit
`836b15b3c4d01fdcc4c1535440943bbb096a001e` with
`codex exec --sandbox read-only`. It exited 1 before reviewing. Its terminal
result, verbatim, was **“Error: failed to initialize in-process app-server
client: Read-only file system (os error 30)”**. There is no verdict to quote;
the review requirement is **UNMET/BLOCKED**, and the absence of a verdict is
not approval. The full log is `codex_round71_review.log`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
No production diff is being landed, so this blocked review cannot silently
approve one.

The round-71 instrument/helper/acquisition tests, the entire receipt-citation
test file, Python compilation, Ruff on new Python files, shell parsing,
`git diff --check`, and the dry Fortran syntax proof pass. The operator's first
required repair is isolated in commit `69609b91f0b9`: the default real-receipt
citation test now reports all 16 tests passed. The combined focused run reports
27 passed (`focused_tests.xml`, SHA-256
`4d04018d10781cd2d8706f8d1549ff497f1a71119c2774a2b96608d503f5e4e5`).

The receipt citation gate is stamped to clean commit
`f2545f0ac7539edc93f80ce3662584665e1b2e93`; it audits all six round-71
compiled-source citations and the complete map with no failure
(`round71_citation_gate.json`, SHA-256
`2dabccfc4bccfe8a815e82b50b99892409d6d00cad5f863ddc151a92c96e6816`).
Shifting the stage-2 update citation by two lines produces
SYMBOL-NOT-AT-LINE and exits 1 (`round71_citation_plant.json`, SHA-256
`ecd989b0e4a14fcdd31b7d591d9b7172e54745a4aba1ffb621c9f646e56400d3`).

## ASKED / UNASKED and OPEN

| state | item | disposition |
|---|---|---|
| ASKED | configuration choice | none encountered |
| UNASKED | configuration, carried state, stabilizer, NEMO, or harness change | none performed |

OPEN for round 72: first admit the new R71 record and verify its compiled
writer locations and exact stage bridge. Replay the kt=2 stage-2
thickness-weighted tracer update in compiled operation order for T and S,
including Kbb, Kmm, accumulated post-SBC Krhs, tmask and all three r3t levels.
Stop at its first non-bit input or operation. Preserve the round-71 LDF-only
reference and retained-cell sets. Only after that producing statement is exact
may the walk proceed to FCT transports/metrics, which remain the leading
magnitude candidates after Kmm was refuted. Do not infer missing operands, use
the round-46 stage record as a substitute, or retry the held round-70 output
pair.
