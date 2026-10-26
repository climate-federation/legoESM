# NEMO testcase L2 GYRE round 141 — developed 3-D RHS observer receipt

Date: 2026-09-21  
Incoming lane tip: `245e5cfe98b50fa548daa36ea740e48e10d75872`  
Preregistration commit: `e63660a26`  
Verdict: **HELD — no scientific RHS row scored and no physics landed**

## Result

The operator-provided Round-140 record is valid and its literal depth and wind
replays are BIT. The required production-JIT observer is not passive. Even an
unordered callback carrying only one already-computed face RHS changes the
returned production state and that face's external forcing. The fail-closed
gate therefore publishes an empty scientific registry and names no HPG, LDF,
VOR, KEG, or ZAD owner.

This is an instrument refutation, not evidence that the completed RHS is exact
or non-exact. The preregistered claim that completed `Krhs` U would be first is
**UNMEASURED**. Day-240 carry is **UNMEASURED**.

## Compiled program and admitted record

The record producer's compiled `stp2d` accumulates HPG, LDF, VOR, KEG, and ZAD
in that order at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:142-169`. It writes
thickness, the completed U/V `Krhs`, and masks at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:182-192`, computes and
writes the depth means at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:210-228`, then applies
drag and wind and writes the terminal pair at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-260`.

P0 readmission passed:

| control | result |
|---|---:|
| record size | 1,486,548 bytes |
| record SHA-256 | `3523ad8eba0b53f4462e85237d85a066195437e6ad1c867e4cb0179d1764a8f5` |
| header | `NEMO_L2_R140RHS`, version 3, kt 1081, Kbb 1, Krhs 3, 36 x 26 x 31, fp64 |
| literal U depth replay | BIT, 0 / 580 |
| literal V depth replay | BIT, 0 / 570 |
| literal U wind replay | BIT, 0 / 580 |
| literal V wind replay | BIT, 0 / 570 |
| Round-140 post-wind vs Round-139 incoming U | BIT, 0 / 580 |
| Round-140 post-wind vs Round-139 incoming V | BIT, 0 / 570 |
| completion | `STOP 0`; `RUN_DONE` |

The inherited header, truncation, replay-ULP, stamp, and passive-admission
plants all exited nonzero with their named `STATUS PLANT-FIRED` markers.

## Production-JIT passivity result

The ordinary eight-field external observer remains passive. The new RHS
observer was progressively reduced from two faces, to one ordered face, to
one unordered face. The smallest form still failed:

| run | returned-state unequal / 198,956 | max abs | observed-face incoming unequal | observed-face final unequal | boundary max abs |
|---|---:|---:|---:|---:|---:|
| ordinary control | 0 | 0 | 0 | 0 | 0 |
| U-only RHS observer | 45,012 | 2.2475077354755513e-14 | 174 / 726 | 160 / 726 | 1.6940658945086007e-21 |
| V-only RHS observer | 45,816 | 1.2256862191861728e-13 | 168 / 736 | 150 / 736 | 1.6940658945086007e-21 |

The opposite face remains BIT in each one-face run, which localizes the
movement to the added callback's compilation/fusion effect rather than record
ancestry. Because the observer fails the preregistered state and boundary
identity conditions, the gate reports `scientific_rows_withheld=true`,
`rows={}`, and `first_non_bit_operand=null`.

The missing-row registry plant exits 1 with
`ROUND141 RHS RHS-MISSING-ROW STATUS PLANT-FIRED`. The RHS-ULP scientific
plant is **UNMEASURED**: passivity fails before any RHS row is admissible, so
running that plant could not validate a claim-bearing row.

## Frozen predictions

| prediction | verdict | evidence |
|---|---|---|
| P0 record/readmission passes | CONFIRMED | exact digest, header, closure, literal replay, ancestry, completion, and inherited plants |
| production observer returns a BIT-identical state | REFUTED | U and V observers move 45,012 and 45,816 cells |
| production observer preserves its external boundary | REFUTED | U and V move at 1.6940658945086007e-21 |
| geometry rows are BIT | UNMEASURED | scientific registry withheld before scoring |
| completed `Krhs` U is first non-bit | UNMEASURED | scientific registry withheld before scoring |
| cumulative first term is HPG U | UNMEASURED | conditional acquisition was forbidden after P1 failed |
| day-240 carry | UNMEASURED | no admissible candidate or substitution arm |

No failed prediction has been rewritten as a scientific RHS verdict.

## Review, controls, and tests

The separate read-only Codex review emitted exactly:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

It exited 1 and supplied no `SHIP`, `HOLD`, or `DO NOT SHIP` verdict.
Independent review was unavailable in-sandbox; the standing operator rule
permits continuation. No production physics is landed.

Adding the private hook shifted existing citations in the same source file.
Every affected endpoint was rigidly moved by the corresponding +6, +16, or
+22 lines; cited extents and endpoint symbols are unchanged. The final
citation gate passed with 4 citations, 0 unmapped citations, 0 failures, and
0 failing map entries. Shifting the compiled-order citation by two lines made
the gate fail and exit 1 as required.

Focused CPU/x64 tests before the final citation pass reported:

```text
32 passed in 0.84s
77 passed in 2.66s
```

The first combined pass reported `3 failed, 74 passed in 2.61s`, solely the
expected pre-commit shifted-citation audit. The rigid re-anchor was committed
before the final clean test run; no failure is carried.

| evidence artifact | SHA-256 |
|---|---|
| `developed_rhs_jit.json` | `a8f412a7edcace3f1ebafcf66bdaf5687a95d6c3d6cd3e8d50d7674620dce00f` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `focused_tests_final.log` | `a1fd36cecde19442d4e8211fc0c807b21c04c1650bba039b5d7d3304a64c226f` |
| `citation_gate_final.json` | `28304077e2af55e5cbcc10bcae7cf3c8df88a142d856c6262d6a451ed89f90cd` |
| `citation_gate_shifted_plant_final.json` | `0e4523c256b2aee85b5cdc64542b7f03932a2a602385bbf2be763649efb4a7eb` |

## Certified trajectory and scope

There is no candidate and no Decision-43/45 run. Immutable headline values
remain:

| row | unchanged value |
|---|---:|
| kt2 U RMS | 2.7377110452773967e-12 |
| kt2 V RMS | 3.2849219221489645e-12 |
| kt3 T RMS | 8.659373840202989e-7 K |
| kt3 S RMS | 7.027291104577671e-8 |
| day-30 T3D RMS | 6.890431487825909e-5 K |
| day-240 T3D RMS | 1.644674193e-2 K |
| day-360 T3D RMS | 1.122357391e-2 K |

DINO, LOCK_EXCHANGE, OVERFLOW, and the tanks have no production change and
were not rerun. ORCA2 is `UNMEASURED-WITH-SPEC` for this GYRE-only diagnostic.
There is no configuration, default, carried-state, scheme, stabilizer, or
immutable-arm change.

## OPEN — round 142

Do not add another callback. Use a directed, no-observer substitution at the
existing production `du_dt`/`dv_dt` boundary:

1. prove that the ordinary production step is unchanged when no override is
   supplied;
2. substitute the admitted NEMO completed U/V `Krhs` directly, run through
   the production step, and score whether it closes the incoming 2-D forcing;
3. only if that confirms the completed RHS family, acquire cumulative
   HPG/LDF/VOR/KEG/ZAD boundaries and use one-term directed substitutions to
   rank their day-240 sensitivity. Do not infer an owner by residual.

No NEMO acquisition is needed for the first discriminator.

`ACQUISITION_NEEDED: NONE`  
`DECISION_NEEDED: NONE`  
`ROUND_STATUS: HELD`
