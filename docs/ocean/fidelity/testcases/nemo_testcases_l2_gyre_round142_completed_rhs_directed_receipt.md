# NEMO testcase L2 GYRE round 142 — completed-RHS directed receipt

Date: 2026-09-21  
Incoming lane tip: `a46cc1b14b7cebf18113bc54205900fc4a439267`  
Preregistration commit: `8587a16e8`  
Measurement implementation commit: `8dc328446`  
Final plant implementation and measurement stamp: `7be6a7546d1bc848fecd711dfa23da14b147774c`  
Citation re-anchor commit: `fa5e64e32`  
Status: **HELD — no production physics landed**

## Result

Replacing legoESM's completed three-dimensional U/V momentum RHS by NEMO's
same-step fields removes 68.946% of the U incoming-forcing maximum and 75.415%
of the V maximum. It does not close either boundary: all 580 wet U faces and
all 570 wet V faces remain unequal. The exact-closure prediction is REFUTED.

The preregistered completed-RHS-family criterion is also REFUTED. U improves
only 3.220x, from `4.2854247978022983e-13` to
`1.3307884389737387e-13 m s-2`; V improves only 4.067x, from
`4.433308633699682e-13` to `1.089946783815101e-13 m s-2`. This misses both
the required 1,000x reduction and the `1e-18 m s-2` terminal bound.

The completed RHS is therefore a measured partial contributor to the local
day-180 forcing gap, not its sufficient owner. Per preregistration, no
cumulative HPG/LDF/VOR/KEG/ZAD acquisition was run and no day-240 carry was
assigned. No first non-bit operator statement is named this round: the first
unresolved compiled boundary after the exact RHS input is the depth average,
but it was not directly scored and is not silently promoted to an owner.

## Compiled-source citations

The acquired target's exact compiled program accumulates HPG, LDF, VOR, KEG,
and ZAD into `Krhs` at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:142-169` and writes the
completed U/V fields at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:182-192`. It next
depth-averages those fields at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:210-228`, then applies
the drag and wind terms at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-260`.

The directed model arm changes no compiled NEMO source. It replaces only the
owned model-native U `[:, 1:, :]` and V `[1:, :, :]` completed-RHS faces at
the corresponding pre-reduction boundary. The excluded boundary faces retain
the values production computed. The hook defaults to `None`, is absent from
the public model configuration, and returns or callbacks no RHS value.

## Record and ordinary-path controls

The exact 1,486,548-byte Round-140 record was re-admitted at the measurement
commit:

| control | result |
|---|---:|
| record SHA-256 | `3523ad8eba0b53f4462e85237d85a066195437e6ad1c867e4cb0179d1764a8f5` |
| producer | `245e5cfe98b50fa548daa36ea740e48e10d75872` |
| header | `NEMO_L2_R140RHS`, version 3, kt 1081, Kbb 1, Krhs 3, 36 x 26 x 31, fp64 |
| NEMO literal depth replay | BIT on 580 U and 570 V wet faces |
| NEMO literal wind replay | BIT on 580 U and 570 V wet faces |
| Round-140 post-wind vs Round-139 incoming | BIT on both faces |
| closed run | 20 manifest members; `STOP 0`; `RUN_DONE` |

Header, truncation, replay-ULP, record-stamp, and passive-admission controls
all retain their named `STATUS PLANT-FIRED` markers.

The ordinary no-override arm reproduces all four Round-140 maxima exactly.
Its passivity comparison is BIT across 198,956 cells and 23 returned-state
leaves. The directed arm's established eight-field callback is independently
BIT against a plain directed step across the same census, and its reported
final U/V fields are BIT against the actual external-call operands. Thus the
directed state movement is the intended substitution, not observation:
96,892 / 198,956 returned-state cells move, maximum
`7.489275866134903e-9`.

## Eight registered rows

All rows are production JIT on CPU, fp64/x64/libm, driven from NEMO's admitted
completed step-1080 state. Signed zero is non-bit.

| registered row | unequal / wet | max abs vs NEMO |
|---|---:|---:|
| ordinary incoming U | 580 / 580 | `4.2854247978022983e-13` |
| ordinary incoming V | 570 / 570 | `4.433308633699682e-13` |
| ordinary final U | 580 / 580 | `4.2854247978022983e-13` |
| ordinary final V | 570 / 570 | `4.4333086294645174e-13` |
| NEMO-RHS-directed incoming U | 580 / 580 | `1.3307884389737387e-13` |
| NEMO-RHS-directed incoming V | 570 / 570 | `1.089946783815101e-13` |
| NEMO-RHS-directed final U | 580 / 580 | `1.3307884389737387e-13` |
| NEMO-RHS-directed final V | 570 / 570 | `1.089946783815101e-13` |

The final U equality to incoming U is exact at the printed maximum; final V
differs from ordinary incoming V only through the already measured downstream
Coriolis last-bit residue. Neither changes the family verdict.

## Frozen predictions and falsifiers

| preregistered claim | result | verdict |
|---|---|---|
| record and inherited controls pass | exact digest/header/replays/ancestry/completion | CONFIRMED |
| default arm reproduces Round 140 | all four maxima exact; callback/plain state BIT | CONFIRMED |
| NEMO completed RHS makes incoming U/V BIT | 580 and 570 faces remain unequal | REFUTED |
| completed RHS reduces each maximum >=1,000x to <=1e-18 | 3.220x U and 4.067x V; residual ~1e-13 | REFUTED |
| HPG has largest day-240 sensitivity | conditional acquisition forbidden | UNMEASURED |

The failed predictions remain explicit. The measured 69-75% local reduction
is not relabelled as exactness or as a year-scale owner.

## Plants and instrument retraction

The eight-row omission plant exited 1 with
`ROUND142 RHS RHS-DIRECTED-MISSING-ROW STATUS PLANT-FIRED`.

The first RHS-ULP selector used NEMO's serial `SUM` replay. Its selected word
survived that isolated expression but disappeared under the complete
production compile; the gate exited 1 with
`GATE FAILED: Round-142 one-ULP RHS plant did not reach incoming forcing`.
That control is RETRACTED and the failed log is retained.

The corrected selector uses the model's exact stacked depth-reduction
expression only to choose one candidate word; acceptance still requires the
change to cross the complete production step. It changed exactly the wet U
word at owned index `[1, 15, 0]` and then changed exactly one incoming U face
by `3.308722450212111e-24 m s-2`. The process exited 1 with
`ROUND142 RHS RHS-DIRECTED-ULP STATUS PLANT-FIRED`.

## Review, citation gate, and focused tests

The required separate read-only Codex invocation emitted exactly:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

It exited 1 and supplied no `SHIP`, `HOLD`, or `DO NOT SHIP` verdict.
Independent review was unavailable in-sandbox; the operator's standing rule
permits continuation. No production physics is landed.

The first focused combined run reported `3 failed, 50 passed in 2.60s`.
All three failures were the expected shifted-citation audit after adding four
hook-definition and five implementation lines. Every affected citation was
rigidly re-anchored with its extent and endpoint symbols unchanged. The
committed citation unit rerun reported `16 passed in 1.95s`.

The receipt citation gate passed with 4 citations, 0 unmapped citations,
0 failures, 0 failing map entries, and 0 failed self-controls. Shifting the
compiled HPG-through-ZAD citation by two lines made exactly one citation fail;
the plant exited nonzero as required. The final four-file focused rerun
reported `53 passed in 2.55s`.

## Evidence artifacts

| artifact | SHA-256 |
|---|---|
| `rhs_directed_commit7be6.json` | `3f1c92503f4641a8cf4aac9c4f9d621852004bcbbe5ead875c3c53116d8273d5` |
| `rhs_directed_commit7be6.log` | `81758b7780888feb8f3c4e88aac2fc2f7f6d917937bba49c376ce1f83f6d0033` |
| `rhs_ulp_plant_commit7be6.json` | `381c2372daeb9396847671ee76b2762e5928fb9ea9ea896ec13655f1c554b170` |
| `rhs_ulp_plant_commit7be6.log` | `288960afa52c08604f356df0765b060bcb7ac44aa06693313fd8cce5b1800d4f` |
| `missing_row_plant_commit7be6.json` | `9a54e407c84fecc0a6e8de62d02dece4dbd3bdf729546f307bfd0cb022d9b469` |
| `missing_row_plant_commit7be6.log` | `0e7dbd46102f0e6c931853d6ec2b1236a197ae4f2f3ee3cef94a2a546b135b8c` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `citation_gate.json` | `25a769a657de089fbddbaa9ee1b0cf5fa5c083c94d160ed7260900d42a7de6f4` |
| `citation_gate_shifted_plant.json` | `6ca6ac36fb66ac86867ed42954f6a85c1cb43ba438b65570e78a4d8c781f55a8` |
| `focused_tests_final.log` | `3adf41fc072bebff81016370ab5ec94c6be4d661624e1c0551c4e68e50735324` |

## Decision-43/45 gate and campaign scope

There is no candidate physics change, so no ladder, month, year, DINO,
LOCK_EXCHANGE, OVERFLOW, tank, or ORCA2 trajectory was rerun. The immutable
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
ORCA2 remains `UNMEASURED-WITH-SPEC`: it needs a card-specific developed
entry and native forcing registry. No configuration, default, carried state,
scheme, stabilizer, canonical NEMO source, or immutable before arm changed.

## OPEN — round 143

Stay on the same admitted day-180 record and add no observer. Use directed
substitutions at the already recorded downstream boundaries, in compiled
order:

1. replace the post-depth-average U/V pair, leaving model drag and wind live;
2. replace the post-drag pair, leaving model wind live; and
3. reuse the admitted post-wind/incoming override as the terminal control.

Each arm must preserve its unowned faces, keep the ordinary path unchanged,
and pass the same callback/plain-state and external-call identity checks. The
first directed boundary that removes the residual names the downstream owner
family; it does not yet name a statement inside that family. Only after the
post-RHS chain closes may a cumulative HPG/LDF/VOR/KEG/ZAD record be acquired
and scored on the year. No bit walk or rest-state patch substitutes for this
developed-state source-order discriminator.

`ACQUISITION_NEEDED: NONE`  
`DECISION_NEEDED: NONE`  
`ROUND_STATUS: HELD`
