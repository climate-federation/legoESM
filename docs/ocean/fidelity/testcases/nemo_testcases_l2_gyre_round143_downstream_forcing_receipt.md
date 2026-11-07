# NEMO testcase L2 GYRE round 143 — downstream forcing receipt

Date: 2026-09-21  
Incoming lane tip: `478f8abf2b70a106f9910190ca66199767bc3a63`  
Preregistration commit: `590394b33`  
Measurement implementation and stamp: `2926841cc498ae0d2be31b6f64c4c2ae84e9f0d2`  
Initial receipt and citation-map commit: `ecb3f9d14`
Rigid citation re-anchor commits: `ed6bc015e`, `985f13751`
Status: **HELD — no production physics landed**

## Result

The first unresolved compiled statement after an exact post-drag input is the
wind update. Exact NEMO post-depth inputs leave the Round-142 residual
unchanged; exact post-drag inputs change it by less than 0.2%; exact post-wind
inputs make both incoming slow-forcing rows BIT. Thus the wind family carries
the remaining developed-state `1e-13 m s-2` magnitude, but this round does not
yet distinguish a stress, density reciprocal, live-depth reciprocal, or
written-association owner inside that statement.

The frozen depth prediction and the frozen post-drag exactness prediction are
**REFUTED**. The frozen terminal calibration prediction is **CONFIRMED**. No
result is inferred by residual subtraction: each row comes from a separate
production-JIT directed arm.

## Compiled-source statement

The admitted target computes the depth average at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:210-228`, calls and
records the drag boundary at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-239`, then records
the wind operands and result at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:241-260`. The first
unresolved U/V statements are the two wind additions at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:249-250`.

The record supplies both pre- and post-statement boundaries and every direct
operand: `r1_rho0`, face stress, and the live U/V inverse depths. This round
uses only the boundary pair. It therefore names the statement reached, not
which operand or association is wrong.

## Record, production, and passivity controls

The exact 1,486,548-byte Round-140 record was readmitted with SHA-256
`3523ad8eba0b53f4462e85237d85a066195437e6ad1c867e4cb0179d1764a8f5`.
Its header, literal depth and wind replays, same-run Round-139 terminal
identity, closed manifest, inherited byte identities, completion markers, and
stored plants all passed unchanged.

The ordinary arm reproduced Round 142's four maxima exactly. The completed-
RHS arm reproduced its four maxima exactly. In all five arms, the callback
step was BIT against an independently compiled plain step across 198,956 cells
and 23 returned-state leaves. Every callback final U/V pair was BIT against
the actual external-call pair. The two new private hooks default to `None`,
are absent from `LatLonCGridOceanConfig`, and replace only owned native faces.

## Registered production-JIT rows

CPU, fp64/x64/libm; entry is NEMO's admitted completed step 1080; all rows are
scored at step 1081. Signed zero is non-bit.

| directed arm | incoming U unequal / 580; max | incoming V unequal / 570; max | final U unequal / 580; max | final V unequal / 570; max |
|---|---:|---:|---:|---:|
| ordinary | 580; `4.2854247978022983e-13` | 570; `4.433308633699682e-13` | 580; `4.2854247978022983e-13` | 570; `4.4333086294645174e-13` |
| exact completed RHS | 580; `1.3307884389737387e-13` | 570; `1.089946783815101e-13` | 580; `1.3307884389737387e-13` | 570; `1.089946783815101e-13` |
| exact post-depth | 580; `1.3307884389737387e-13` | 570; `1.089946783815101e-13` | 580; `1.3307884389737387e-13` | 570; `1.089946783815101e-13` |
| exact post-drag + live wind | 580; `1.3293041661996061e-13` | 570; `1.0882954083811340e-13` | 580; `1.3293041661996061e-13` | 570; `1.0882954083811340e-13` |
| exact post-wind | 0; `0` | 0; `0` | 268; `6.88214269644119e-22` | 247; `8.470329472543003e-22` |

The post-wind final rows retain only the already measured downstream Coriolis
residue. They do not weaken the incoming-boundary wind-family verdict.

## Frozen predictions and falsifiers

| preregistered claim | observed result | verdict |
|---|---|---|
| post-depth reduces both completed-RHS residuals by at least 1,000x to at most `1e-18` | both maxima are bit-for-bit unchanged from the completed-RHS arm | REFUTED |
| exact post-drag plus live wind makes both incoming rows BIT | 580 U and 570 V faces remain unequal at `1.3293e-13` / `1.0883e-13` | REFUTED |
| exact post-wind makes both incoming rows BIT | 0 / 580 U and 0 / 570 V unequal | CONFIRMED |
| ordinary and Round-142 controls reproduce exactly | all eight pinned maxima reproduce | CONFIRMED |

## Plants, review, and tests

The closed 20-row registry omission plant exited 1 with
`ROUND143 DOWNSTREAM DOWNSTREAM-MISSING-ROW STATUS PLANT-FIRED`. The consumed
post-depth U one-ULP plant changed one word at native index `[13, 1]`, crossed
the complete production step, and moved exactly one incoming U face by
`1.6940658945086007e-21 m s-2`; it exited 1 with
`ROUND143 DOWNSTREAM DOWNSTREAM-DEPTH-ULP STATUS PLANT-FIRED`.

The required separate read-only Codex review emitted exactly:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

It exited 1 and supplied no `SHIP`, `HOLD`, or `DO NOT SHIP` verdict.
Independent review was unavailable in-sandbox; the standing operator rule
permits continuation. No production physics is landed.

The pre-measurement focused test run reported `28 passed in 1.24s`. The first
four-file focused run after adding the new hooks reported `1 failed, 75 passed
in 2.44s`: the only failure was the expected cumulative-receipt citation drift
caused by the 27 inserted lines. After rigidly shifting the nine affected
receipt references, the same run reported `76 passed in 2.45s`.

The preregistration citation gate passed with 5 citations, 0 unmapped, 0
failures, and 0 failing map entries. The receipt gate passed with 4 citations,
0 unmapped, 0 failures, and 0 failing map entries. Shifting the exact wind
statement citation by two lines made exactly one citation fail; the plant
exited 1 as required.

## Evidence artifacts

| artifact | SHA-256 |
|---|---|
| `downstream_directed.json` | `3b093c12aa9dfc5568b3ce08a1e8b83ad55f96963f02a00b4813f5ae2aa3b351` |
| `downstream_directed.log` | `a949fa271de814d90eb9b9e302b1ec84852c9fa134c5be28755984d2127ba99f` |
| `missing_row_plant.json` | `ce50fdf89b2c9f2a35cc240b03c3751b4135b14dda69e55d60f663cc93f745c8` |
| `missing_row_plant.log` | `3262e0ec29c608ba280b6d6354cd290a9682de6e2b36dce0b95e82c60914eb5d` |
| `depth_ulp_plant.json` | `7a72ac0922ecb0d1e5b4b88b98324284613c6ce5924fac9b9a0e485be852d22e` |
| `depth_ulp_plant.log` | `740e006e76579099eac8c2907c829740533ec83ce5bc6cc6130835ee4e80c3e9` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `focused_tests_clean.log` | `7a75419e240ea2597f786f041e25f98480a4bb927d3a089f52162f84a549c907` |
| `prereg_citation_gate_clean.json` | `6ddb02b1112622684ce951a99af481b39ed78acaedd8892a8751723c5c6735d0` |
| `citation_gate_clean.json` | `17a5fc251643279ca3cc98ae60ef0860358c2f446970ff34535d5faa0c730518` |
| `citation_gate_shifted_plant_clean.json` | `f4111964b36b716004b626e2742b544537c830e29c54845f6c2ebcafd1536f42` |

## Decision-43/45 gate and campaign scope

There is no source-exact candidate, so no ladder, month, year, DINO,
LOCK_EXCHANGE, OVERFLOW, tank, or ORCA2 trajectory was rerun. Immutable
headline values remain:

| row | unchanged value |
|---|---:|
| kt2 U RMS | `2.7377110452773967e-12` |
| kt2 V RMS | `3.2849219221489645e-12` |
| kt3 T RMS | `8.659373840202989e-7 K` |
| kt3 S RMS | `7.027291104577671e-8` |
| day-30 T3D RMS | `6.890431487825909e-5 K` |
| day-240 T3D RMS | `1.644674193e-2 K` |
| day-360 T3D RMS | `1.122357391e-2 K` |

DINO, LOCK_EXCHANGE, OVERFLOW, and tanks execute no changed production path.
ORCA2 remains `UNMEASURED-WITH-SPEC`: it requires a card-specific developed
entry and native forcing registry. No configuration, default, carried state,
scheme, stabilizer, canonical NEMO source, or immutable before arm changed.

## OPEN — round 144

Stay on the admitted day-180 record and the no-observer directed path. Walk
the wind statement's inputs in compiled order under production JIT: density
reciprocal, U/V stress, live U/V inverse depth, then the written multiplication
and addition association. Substitute one operand family at a time from the
same record and register every incoming/final row. The first exact operand
that moves the post-drag residual names the first non-bit input; if all inputs
are exact, test the written association. Do not infer an operand from the
post-wind boundary alone. A source-exact statement still requires the full
Decision-43/45 ladder, month, year, and shared-card checks before landing.

`ACQUISITION_NEEDED: NONE`  
`DECISION_NEEDED: NONE`  
`ROUND_STATUS: HELD`
