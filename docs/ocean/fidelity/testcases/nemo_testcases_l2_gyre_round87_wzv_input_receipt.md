# NEMO testcase L2 GYRE phase 3 — round 87 WZV-input receipt

## Outcome

**STOPPED FOR DECISION; no production physics landed.** The first scalar-
calibrated non-bit statement in the stage-1 WZV walk is the separate
`e3t*hdiv` materialization, at `1.6940658945086007e-21` over 6,109 of 18,000
wet point-levels. The large owner is later: legoESM's reconstructed Kaa SSH
differs from NEMO's already-stored Kaa scratch slot at all 600 wet cells, with
maximum `1.144318797451864e-02 m`. That Kaa replacement removes 100 percent of
the Round-86 ZAD maximum on U and V, but the unmodified shared WZV helper still
leaves 2,939 W cells unequal at `2.117582368135751e-22`.

The frozen prediction is **REFUTED and retained** because Kaa was not the first
non-bit statement and Kaa-only W did not become bit-exact on the landed path.
A source-round candidate did make Kaa-injected W and downstream ZAD bit-exact,
but it could not reconstruct NEMO's actual Kaa scratch operand: the executing
RK3 program reads that slot before the external solve. The candidate was
therefore held and production was restored byte-for-byte to the Round-86 tip.
Carrying an equivalent Kaa scratch through steps and restarts is a new
carried-state representation, so the user must authorize it before the walk
can continue.

## Frozen registration and evidence

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round87.md`, committed as
`56ee3473c57e` before any Round-87 scientific comparison. The final
fail-closed instrument commit is `ed355edd27c4`; evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round87/`. All executions used
CPU, JAX fp64/libm, production JIT, and the admitted Round-64 record producer
`3b3b045bd9e03b60330204e7590e4c4470b7a0ca`.

The decisive ordinary artifact is `final_wzv_inputs.json`, SHA-256
`ba27c5d8bceb2eda35c735a0cc43d9732cf8975c329ff4f2393022668b07ba88`.
It exits 1 with `REFUTED`, as required. Admission retains 43 byte-identical
records, 20 classified changed records, 132 admitted differences, and zero
owned-and-defined differences in the consumed kt2 stage-1 projection. The
literal scalar WZV replay on NEMO inputs has zero unequal cells.

The installed GitHub CLI remains unauthenticated and no GitHub connector is
available. Rounds 85--87 could not be posted to issue #1455; no external state
was mutated.

## Ordered result

All rows before `hdiv` are bit-exact: Kbb U/V, U/V metrics, live face
thickness, metric-thickness products, U/V fluxes, zonal and meridional
differences, their sum, reciprocal T area, and live T thickness. `hdiv` itself
is bit-exact. The next statement materializes `e3t*hdiv` and is the first
non-bit row:

| boundary | unequal / active | absolute maximum | disposition |
|---|---:|---:|---|
| `hdiv` | 0 / 18,000 | 0 | bit-exact |
| `e3t*hdiv` | 6,109 / 18,000 | `1.6940658945086007e-21` | first non-bit statement |
| Kbb r3 | 0 / 600 | 0 | bit-exact |
| `r1_Dt` | 0 / 1 | 0 | bit-exact |
| Kaa SSH | 600 / 600 | `1.144318797451864e-02` | first magnitude-bearing input |
| Kaa r3 | 600 / 600 | `2.6607671590766353e-06` | magnitude-bearing derived input |
| final W | 18,000 / 18,000 | `7.946658315637966e-07` | inherited pre-external boundary |
| actual post-solve transport W | 18,000 / 18,000 | `7.908320113361555e-07` | distinct later operand |

The final instrument's scalar reference records every intermediate and
requires its own final W to equal NEMO bitwise. This calibration is what kills
two earlier diagnostic artifacts: eager/JIT asymmetry first reported non-bit
`e3div` at `1.69e-21`, then equal-JIT vector replay reported non-bit `hdiv` at
`1.06e-22`. Neither was citable. Only the scalar trace, whose final W is exact,
names the table above.

Kaa substitution in the unmodified shared helper reduces W to 2,939 unequal
cells and `2.117582368135751e-22`. Replaying ZAD with that W is nevertheless
bit-exact on all 17,400 U and 17,100 V wet point-levels, so both face residuals
move from `1.9220297482797664e-09` / `1.966061294804274e-09` to zero. The
separate post-solve transport W leaves `1.9149450031535433e-09` U and
`1.953787036547103e-09` V; it is not NEMO's pre-solve stage-1 operand.

A post-hoc record invariant is retained but not generalized: at this kt2
capture, stored Kaa r3 equals `Kbb+Kbb` bitwise on all 600 wet cells. No source
statement establishes that as an all-step identity, so `2*Kbb` is not an
eligible reconstruction and was not implemented.

## Compiled-source basis

The executing `stp_2D` branch assigns `r3t(Kaa)` from the already-present
`ssh(Kaa)`, then calls velocity-form WZV, KEG, and ZAD in that order
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-176`). The external
mode solve that writes the next Kaa state occurs only afterward
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:301-311`). Thus a
post-solve transport is not the consumed stage-1 operand.

The compiled velocity-form divergence forms the face products and hdiv, then
materializes `pe3divUh = hdiv*live_e3t` as a separate statement
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/divhor.f90:123-153`). WZV adds the
Kaa-minus-Kbb stretch and carries bottom-up
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:293-300`). Later,
`stp_RK3_stg` explicitly skips another WZV call for vector-invariant stage 1
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:326-339`). These
are the compiled statements that execute under the admitted GYRE header.

## Candidate and retraction

Candidate commit `002b7f6d0432` replaced stripped optimization barriers with
the repository's surviving IEEE source-round identity at `e3t*hdiv` and the
WZV recurrence, based the reconstructed SSH forecast on current eta, and
removed the later stage-1 transport-W substitution. With NEMO's recorded Kaa
injected, W and both ZAD faces became bit-exact. The full candidate remained
**REFUTED**, however: its own reconstructed Kaa still differed at all 600 wet
cells. NEMO does not execute that reconstruction at this boundary; it reads an
already-carried scratch slot. Commit `ed355edd27c4` restores production, and a
direct diff against the pre-candidate `7a07811bfc5b` is empty for both
production files.

This explicitly retracts the provisional hypothesis that changing the
forecast base from `eta_before` to `eta_now` could recover the stored slot. It
also corrects Round 86's instrumentation scope: Round 86 scored the captured
pre-external W but did not separately identify the later `_g0` W that the
current production stage-1 override consumes. Round 87 scores both; neither is
exact, and the compiled source permits only the former boundary.

## Controls, review, and verification

All three registered one-ULP controls print `PLANT_FIRED` and exit 1:

- `flux-ulp.json`, SHA-256 `dc95c9efa711bee789d1fcc6ade86be9d0075f12ff173286ec2b21d8a2b16e2a`;
- `kaa-ulp.json`, SHA-256 `d62f60d7499c72dbf3198cd1cc5548f4769f5487d1777d441468c05bfe3bfa1e`;
- `carry-ulp.json`, SHA-256 `6f5dd8ec4a4bdc3f3cb01b1b4b90a20b23a6e812f0a66cc4a17d7873f5d666b4`.

Two pre-result instrument failures are retained: an unnamed rank mismatch,
then the named `metric_u` `(22,32)` versus 3-D-mask mismatch. Neither wrote a
scientific JSON. The earlier vector-reference JSONs are retained as retracted
instrument outputs, not evidence.

The required separate read-only Codex command failed before a reviewer model
started. Its terminal verdict is quoted verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Per the operator instruction, **independent review unavailable in-sandbox**;
work continued. No `SHIP` verdict is claimed. Complete output is
`round87_codex_review.txt`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

Focused CPU/fp64 verification passes 50 tests covering the Round-46 literal
stage replay, Round-84 RHS walk, Round-86 ZAD operand walk, Round-87 scalar
calibration and controls, the receipt citation map, and ZAD/dynZDF composition.
The decisive line is `50 passed in 32.66s`; log SHA-256 is
`4a02858ea112e9816492f6b9847b93507533f4a399dcb13dab8f4eaac70d4405`.
Python compilation, `git diff --check`, and the production-file restoration
diff pass. The known unrelated `test_rk3_ws_differs_from_rk3_and_is_finite`
failure was not encountered.

The receipt citation gate passes all 5/5 compiled-source citations with zero
unmapped citations, failures, or global map-audit failures; artifact SHA-256 is
`7ca5fa38e03cb62b653bd5166c5591e0e9882d408d4359870f168258d0879468`.
Shifting the `divhor` citation by two lines exits 1 with
`SYMBOL-NOT-AT-LINE`; plant SHA-256 is
`9d14528c83026797efca94ca28844f45bea53090f4f52be08dab9c7a4d8406b2`.
Two earlier citation attempts were rejected: first for an off-by-one stp2d
heading plus ambiguous repeated anchors, then for off-by-one extents. No
citation from either failed attempt was accepted.

## Rule-12 disposition

| lane | Round-87 result |
|---|---|
| GYRE kt2 WZV/ZAD | **MEASURED / HELD.** Given-input scalar WZV and Kaa-injected source-round W are exact; the landed helper remains non-bit and lacks NEMO's stored Kaa scratch. |
| GYRE kt=1..10 | **PRESERVED, NOT RERUN.** No production change. Before remains `round85/bundle_kt1_10.json`; first-over-bar remains kt2 U/V. |
| Every moved GYRE row | **NONE.** Final production is byte-identical to the Round-86 tip. |
| GYRE days 1..30 | **PRESERVED, NOT RERUN.** Before remains `round85/bundle_day_gap.json`; day-30 T RMS remains `1.2397011295506804e-02 K`. |
| LOCK_EXCHANGE-zco | **PRESERVED, NOT RERUN.** No shared statement changed; Round-85's 50 certified rows remain the control. |
| OVERFLOW-zps | **PRESERVED, NOT RERUN.** No shared statement changed; Round-85's 50 certified rows remain the control. |
| DINO | **SHARED-STATEMENT RISK.** Its leapfrog path uses the shared WZV implementation. No neutrality is inferred; 96--98% regional cancellation remains explicit. |
| ORCA2 | **UNMEASURED WITH SPEC.** Carry and align pre/post-external Kaa SSH, Kbb U/V, every velocity/transport WZV input and carry, ZAD, cumulative RHS, and T/S/U/V/SSH on native masks through kt1..10 in fp64. Reject any wet-input mismatch, AT-BAR loss, or earlier first-over-bar. |

No configuration/default, coefficient, timestep, stabilizer, restart format,
year harness, reconciliation gate, freshwater pair, #1484 guard, held
manifest, NEMO source, or NEMO executable changed. The proposed Kaa scratch is
not implemented pending the decision below.

## OPEN — exact handoff after the decision

1. If authorized, add one shared NEMO-identity RK3 Kaa SSH scratch field with
   loud restart loading/failure semantics; cite and reproduce NEMO's slot
   initialization, stage rotation, and restart behavior before using it.
2. Bundle that proven operand with the already-proven source-round
   `e3t*hdiv`/recurrence association and removal of the non-executed post-solve
   stage-1 W override. Require W and ZAD to be bit-exact before the ladder.
3. Run the GYRE kt1..10 ladder against Round 85, then days 1--30, both tanks,
   and a numerical DINO shared-statement score. Keep LDF's smaller first
   cumulative boundary registered.
4. ORCA2 remains UNMEASURED under the specification in the Rule-12 table.

ACQUISITION_NEEDED: NONE

DECISION_NEEDED: May the shared NEMO-identity state add and restart-load NEMO's pre-solve RK3 Kaa SSH scratch slot, failing loudly when absent? My pick: YES, because the compiled path consumes that carried slot and no source-proven reconstruction exists.
