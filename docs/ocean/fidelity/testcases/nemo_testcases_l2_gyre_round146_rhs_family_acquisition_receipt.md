# NEMO testcase L2 GYRE round 146 — developed RHS-family acquisition

Date: 2026-09-21

Incoming lane tip: `39ea1c80863d8924b5b2fd71a0a6ec99c6970f0e`  
Preregistration commit: `b13587663`  
Built acquisition commit: `44a0d6e8146ae8919b4cda26653fe979186a724c`  
Status: **STOPPED_FOR_RECORD — sandbox refused `mpirun` before NEMO
started; no scientific row was measured**

## Result

Round 146 found that the admitted Round-140 record cannot rank the HPG, LDF,
VOR, KEG, and ZAD families: it contains only the completed three-dimensional
RHS.  A new additive writer was preregistered, syntax-proven, compiled, and
staged.  The sandbox then refused PMIx listener-socket creation with
`pmix_ifinit: socket() failed with errno=1`; `mpirun` returned 213 before NEMO
started.  The committed run script now auto-detects that exact staged build,
verifies it fail-closed, skips `makenemo`, and resumes at the run/admission
tail.

No family ranking, first non-bit statement, day-240 sensitivity, candidate,
or landing verdict exists this round.  The preregistered HPG prediction remains
**UNMEASURED**, not confirmed or refuted.

## Compiled source and passive record

The uninstrumented admitted build calls HPG, LDF, VOR, KEG, and ZAD in compiled
order at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:142-169`, then consumes
the completed `Krhs` in the depth average at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:210-228`.

The new compiled target preserves those calls and writes the native U/V
cumulative arrays immediately after each one at
`GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/stp2d.f90:145-192`.  The source
card removes no line, assigns no model operand, adds no model call, and has one
open, ten array writes, and one close only at `kt=1081`.

The record contract is `NEMO_L2_R146FAM`, version 1, `kt=1081`, `Kbb=1`,
`Krhs=3`, `36 x 26 x 31`, fp64.  Its ten registered arrays are the U/V pair
after each compiled family; expected size is exactly `2,321,368` bytes.

## Acquisition controls completed before the sandbox stop

| control | result |
|---|---|
| preregistration before measurement | PASS; committed before source card and gate |
| additive patch | PASS; zero removed source lines |
| preprocessing and `gfortran -fsyntax-only` | `SYNTAX_PROOF_PASS stp2d.f90` |
| writer field census | PASS; exactly five U/V writes |
| writer-layout plant | `STATUS PLANT-FIRED`, exit 69 |
| reader size agreement | PASS; writer and reader both `2,321,368` bytes |
| new target | `GYRE_OMIP_L2_P3_SM_R146RHSFAM` |
| compiled binary SHA-256 | `8270a36f619c46c196e7389bf063b4ac2c8ead418e66ffd3f1357b772de5d250` |
| vector-math exclusion | PASS; no `_ZGV` dynamic symbol |
| NEMO integration | REFUSED before start by PMIx socket permission, exit 213 |

The parser's closed-census/header/truncation focused suite reported `3 passed
in 0.06s`.  The staged target retains the refused stdout/time logs; resume mode
moves them to named `sandbox_refusal` evidence before the operator run.

## Controls awaiting the record

The no-argument run script verifies the complete build commit, binary digest,
compiled writer layout, byte-identical staged inputs, and absence of a prior
record.  After NEMO runs it requires byte-identical step-1080/1081 restarts and
all inherited Round-140 records.  Header, truncation, final-boundary ULP,
missing-field, and passive-admission plants must each print
`STATUS PLANT-FIRED` and exit nonzero.  The final ZAD arrays must be BIT against
Round 140's completed RHS.  Any failure prints a named `REFUSE` line.

## Frozen campaign rows

Because no production code changed and no scientific record was admitted, the
immutable rows remain:

| row | unchanged value |
|---|---:|
| kt2 U RMS | `2.7377110452773967e-12` |
| kt2 V RMS | `3.2849219221489645e-12` |
| kt3 T RMS | `8.659373840202989e-7 K` |
| kt3 S RMS | `7.027291104577671e-8` |
| day-30 T3D RMS | `6.890431487825909e-5 K` |
| day-240 T3D RMS | `1.644674023317539e-2 K` |
| day-360 T3D RMS | `1.122357124784366e-2 K` |

DINO, LOCK_EXCHANGE, OVERFLOW, tanks, and ORCA2 execute no changed production
path.  ORCA2 remains `UNMEASURED-WITH-SPEC`.  No configuration, default,
carried state, scheme, stabilizer, canonical NEMO source, or immutable before
arm changed.

## Review and gates

The required separate read-only Codex pass was attempted against the committed
diff.  It produced no scientific verdict because the in-process app-server
could not initialize in the read-only sandbox.  Its verbatim terminal finding
was: `Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)`.  Therefore **independent review was unavailable
in-sandbox**; this is not represented as a SHIP verdict.  The round changes no
production physics and remains stopped before scientific measurement.

The preregistration citation gate passed with two citations and no failures;
the receipt gate passed with three citations and no failures.  Shifting the
new compiled-writer citation by two lines made the gate print `STATUS FAIL`
and exit 1.  The combined parser/citation focused suite reported `19 passed in
2.11s`.

## Evidence

| artifact | SHA-256 |
|---|---|
| `preflight.log` | `3d43b0f1913c6820589a4671762c712b98c803ee512a96b3b4675c8a49092cbb` |
| `layout_plant.log` | `9cfb3218842167328a98359a061a2e39ca1611d611abcc6062c777fd55636197` |
| `acquisition.log` | `acd1af2dce1235a6dc07f64f6b9933cf03eaad465852a5de3aabfb313f641d36` |
| `parser_tests.log` | `3125ea7be43eaff8638305b0397e992840a6e097fd68025e13f60403a0adc1a5` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `prereg_citation_gate.log` | `b8307d283e692836c4d90da680efa7ec35d746d36df3ab0eae7296199adac1ca` |
| `receipt_citation_gate.log` | `937c71cd9bb41f4e6d087ef7c095cd573df4fc4dcf6deee79f755a973e38e27e` |
| `receipt_citation_plant.log` | `6a72db7057a8c85214c350b9bb14daa92466cb1d46ec7b872206c39b5d38ff8d` |
| `focused_tests.log` | `a4c4056dee2c654e53a8c35159ccec79471f724b0747d05f553642e8b7acf4fb` |

## OPEN — round 147

Run the reported acquisition script with no arguments.  It resumes the exact
already-built target and must print
`ROUND146_DEVELOPED_RHS_FAMILIES_READY`.  Then execute the frozen Round-146
directed ranking: replace one RHS-family addend at a time through the complete
production-JIT step from NEMO's day-180 entry, register ordinary plus five
arms, and name the largest unambiguous family.  Carry only that family to the
day-240 year sensitivity and Decision-43/45 gate; do not revive the refused
downstream wind candidate or chase the `1e-22` Coriolis floor.

`DECISION_NEEDED`: NONE.

## Post-round admission amendment (Round 147)

The operator subsequently ran the acquisition to `STOP 0`; its original
post-run check exited 71 because it compared uninitialized full-array halo
storage byte-for-byte.  Round 147 decoded the entire inherited record and
bound ownership to the compiled loops.  Both restarts and the other five
inherited records are byte-identical.  All 704 inner-domain values of `CdU_u`,
`CdU_v`, `utauU`, and `vtauV` are BIT; only their 232-cell excluded halos
move.  The compiled drag routine declares `CdU_u`/`CdU_v` full-domain output
arrays but explicitly computes the inner domain only at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/dynspg_ts.f90:1460-1493`.
Round 147 therefore retracts the whole-array passivity test, preserves its
failed JSON, and admits the existing record without rebuilding or rerunning
NEMO.  The authoritative corrected admission is documented in the Round 147
receipt; it does not change this round's original `STOPPED_FOR_RECORD` verdict.
