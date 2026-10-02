# NEMO testcase L2 GYRE round 144 — developed wind operand receipt

Date: 2026-09-21  
Incoming lane tip: `a41bc5ed36574da1532d9369bee81a80c901a2c9`  
Preregistration commit: `d0148080c5e85f2360644f6e1e92c67c22519b47`  
Measurement implementation commit: `657aa3bd6cb41dc2ccbb9dd478508b8b1814bc9a`  
Accepted measurement stamp: `68791d46f26c604a5729f5776365efaa60216867`  
Accepted plant stamp: `3409f43aedd7676ea2825c71345548cfdf8d06b9`  
Status: **HELD — diagnostic instrumentation only; no production physics landed**

## Result

The first non-bit input to NEMO's developed-state wind statement is the live
U/V inverse depth. Replacing only that pair by NEMO's recorded
`r1_hu_0/(1+r3u(Kbb))` and `r1_hv_0/(1+r3v(Kbb))` makes both incoming slow
forcing rows BIT: U moves from 580 / 580 unequal at
`1.3293041661996061e-13 m s-2` to 0, and V moves from 570 / 570 unequal at
`1.0882954083811340e-13 m s-2` to 0. Density reciprocal and U/V stress
substitutions are each inert on both faces.

The preregistered stress-owner prediction is **REFUTED**. Density-inert and
terminal-control predictions are **CONFIRMED**. Exact inverse depth and all
exact inputs close the incoming boundary independently; neither result is
inferred by residual subtraction.

The statement named for round 145 is the inverse-depth construction/routing:
NEMO consumes its stored reference reciprocal divided by the step-entry QCO
face ratio, while legoESM's live wind path recomputes the reciprocal from the
min-face summed thickness. This round does not silently promote the diagnostic
override to production. It has not passed the Decision-43 ladder/month gate,
Decision-45 year gate, or the shared DINO check, so no landing is eligible.

## Compiled-source statement and model boundary

The admitted compiled program constructs `r1_hu_0`/`r1_hv_0` once at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domain.f90:212-215`. It constructs
the step-entry `r3u(Kbb)`/`r3v(Kbb)` from the step-entry SSH at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90:160-182`, using the
surface-area-weighted face statement at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90:188-218`. The exact wind
write and consumption are in
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:241-260`; its U/V
additions are exactly
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:249-250`.

The shared implementation already constructs NEMO QCO face geometry for the
same step-entry state at `ocean_model_latlon_cgrid.py:5357-5412`, but the wind
path instead derives its inverse depth from `H_u_pre/H_v_pre` at
`ocean_model_latlon_cgrid.py:5603-5644`. Round 145 must route the existing
shared QCO reciprocals to the wind statement; it must not add a second formula
or a card-specific selector.

## Registered production-JIT rows

CPU, fp64/x64/libm; entry is NEMO's admitted completed step 1080; all rows are
step 1081. Cells compare signed-zero bit patterns. Each table cell is
`unequal/scored; max abs; RMS abs; first unequal [j,i]`.

| arm | incoming U | incoming V | final U | final V |
|---|---|---|---|---|
| live | `580/580; 1.3293041661996061e-13; 2.3711894014130254e-14; [1,1]` | `570/570; 1.0882954083811340e-13; 2.5882578135201913e-14; [1,1]` | `580/580; 1.3293041661996061e-13; 2.3711894000479070e-14; [1,1]` | `570/570; 1.0882954083811340e-13; 2.5882578135657948e-14; [1,1]` |
| exact density only | same as live | same as live | same as live | same as live |
| exact stress only | same as live | same as live | same as live | same as live |
| exact inverse depth only | `0/580; 0; 0; none` | `0/570; 0; 0; none` | `268/580; 6.882142696441190e-22; 1.5741149555179925e-22; [1,1]` | `247/570; 8.470329472543003e-22; 1.4996792790117193e-22; [1,1]` |
| all exact inputs | `0/580; 0; 0; none` | `0/570; 0; 0; none` | `268/580; 6.882142696441190e-22; 1.5741149555179925e-22; [1,1]` | `247/570; 8.470329472543003e-22; 1.4996792790117193e-22; [1,1]` |
| exact post-wind terminal | `0/580; 0; 0; none` | `0/570; 0; 0; none` | `268/580; 6.882142696441190e-22; 1.5741149555179925e-22; [1,1]` | `247/570; 8.470329472543003e-22; 1.4996792790117193e-22; [1,1]` |

The remaining final rows are the already measured downstream Coriolis residue;
they do not weaken the incoming-boundary inverse-depth verdict. Every arm's
callback step is BIT against its independently compiled plain step across
198,956 cells and 23 returned-state leaves. Every callback final pair is BIT
against the actual external-call pair.

## Frozen predictions and falsifiers

| preregistered claim | observed result | verdict |
|---|---|---|
| density reciprocal is inert | both density substitution effects are BIT | CONFIRMED |
| stress is first non-bit input | stress substitution is inert; inverse depth moves all 580/570 faces | REFUTED |
| exact stress makes incoming U/V BIT | both remain 580/570 unequal at the live maxima | REFUTED |
| exact inverse depth is a calibration row expected BIT | both incoming rows are BIT | CONFIRMED |
| all exact inputs are BIT | both incoming rows are BIT | CONFIRMED |
| terminal post-wind control is BIT | both incoming rows are BIT | CONFIRMED |

## Plants and instrument retraction

The closed 24-row registry omission plant exited 1 with
`ROUND144 WIND WIND-MISSING-ROW STATUS PLANT-FIRED`.

The first stress-ULP selector used NEMO's recorded inverse depth to select the
word but executed the stress-only arm with legoESM's live inverse depth. Its
selected word did not survive that different context; the gate failed with
`GATE FAILED: Round-144 stress ULP did not reach incoming forcing`. That
mixed-context control is **RETRACTED** and is not cited as non-vacuity.

The corrected plant holds density and inverse depth exact in both selector and
executed arm. It changes one U-stress word at native `[1,9]`; the complete
production step changes exactly one incoming U face by
`2.6469779601696886e-23 m s-2`. It exits 1 with
`ROUND144 WIND WIND-STRESS-ULP STATUS PLANT-FIRED`.

## Review, citation gate, and focused tests

The required separate read-only Codex review emitted exactly:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

It exited 1 and supplied no `SHIP`, `HOLD`, or `DO NOT SHIP` verdict.
Independent review was unavailable in-sandbox; the standing operator rule
permits continuation. No production physics is landed.

The preregistration citation gate passed with 3 citations, 0 unmapped, 0
failures, and 0 failing map entries. The receipt gate passed with 7 citations,
0 unmapped, 0 failures, and 0 failing map entries. Shifting the exact U/V wind
statement citation by two lines made exactly one endpoint fail; the plant
exited 1 as required.

The final four-file focused suite reported `78 passed in 2.63s`:
`test_nemo_testcase_l2_gyre_round83_slow_forcing_walk.py`,
`test_nemo_testcase_l2_gyre_round51_live_operands.py`,
`test_nemo_testcase_receipt_citation_gate.py`, and
`test_fidelity_time_levels.py`.

## Evidence artifacts

| artifact | SHA-256 |
|---|---|
| `wind_operands_final.json` | `31525d3bbb7a38a995e90eb4756a771297c6e6ef6e8cd6163f3442388675b726` |
| `missing_row_plant.json` | `9b6dcfb4a69eba102c205d4b09f08b74a26802b11879a45b15ac21a3a0008c56` |
| `missing_row_plant.log` | `88acecc7cefe66cd6b88946db0907d5e654a6b1cd29eceeff6c68570172e1a56` |
| `stress_ulp_plant_final.json` | `0b055981b3fe219921850c4b5cad8961177be7d7e09900124adadf726d677c44` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `focused_tests_final.log` | `edf475123ed1b7e283713da2ea6b319b16ae5d461c0169a56c92d265b711fe97` |
| `prereg_citation_gate_final.json` | `1dc139f32d5f5ae3cf155b4100305669a038dccd026c816733ba00f361595ce0` |
| `receipt_citation_gate_final.json` | `0fc7ee9faa3ce5c6a46dfc58ee149d6f0be70c122c1f425fae39df47de87844e` |
| `receipt_citation_shift_plant.json` | `f345cc7591862a13553d6353141cdadd2f7406111cbbdfa80c31401b707546cc` |

The earlier `wind_operands.json` is preliminary and superseded because it did
not include the preregistered RMS and first-index fields. The zero-byte
`wind_operands.log` belongs to the manually interrupted first attempt and is
not evidence.

## Decision-43/45 gate and campaign scope

There is no production candidate in this diff, so the ladder, month, year,
DINO, LOCK_EXCHANGE, OVERFLOW, tank, and ORCA2 trajectories are not rerun.
Immutable values remain:

| row | unchanged value |
|---|---:|
| kt2 U RMS | `2.7377110452773967e-12` |
| kt2 V RMS | `3.2849219221489645e-12` |
| kt3 T RMS | `8.659373840202989e-7 K` |
| kt3 S RMS | `7.027291104577671e-8` |
| day-30 T3D RMS | `6.890431487825909e-5 K` |
| day-240 T3D RMS | `1.644674193e-2 K` |
| day-360 T3D RMS | `1.122357391e-2 K` |

DINO shares the wind/QCO statement and is explicitly **UNMEASURED** for this
diagnostic-only round; it must be measured with the candidate in round 145.
LOCK_EXCHANGE, OVERFLOW, and tanks execute no changed production path. ORCA2
remains `UNMEASURED-WITH-SPEC`: it needs its own developed native-forcing
registry before this GYRE operand result can be generalized. No configuration,
default, carried state, scheme, stabilizer, canonical NEMO source, or immutable
before arm changed.

## OPEN — round 145

Build the one-variable production candidate by asking the existing shared QCO
face-geometry helper for its stored `r1_hu_0/(1+r3u(Kbb))` and
`r1_hv_0/(1+r3v(Kbb))` outputs and routing those to the wind statement. Do not
add a new formula or selector. First prove the ordinary production-JIT
post-drag-plus-wind incoming U/V rows BIT on the admitted day-180 record, with
a production plant. Then run the complete Decision-43 ladder/month registry,
the Decision-45 360-day rows, and a before/after DINO gate because the
statement is shared. Land only if every landing condition passes; otherwise
restore production and preserve a held patch.

`ACQUISITION_NEEDED: NONE`  
`DECISION_NEEDED: NONE`  
`ROUND_STATUS: HELD`
