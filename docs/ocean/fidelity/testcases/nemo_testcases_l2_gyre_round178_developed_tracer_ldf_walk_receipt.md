# Round 178 — developed-state tracer-LDF statement walk

**Status: HELD.** No physics, card, carried state, default, or trajectory
baseline changed.  The admitted developed-state record moves the first
non-bit boundary earlier than the preregistration predicted: NEMO's `uslp`
output differs in 16,820 of 17,400 active U-face cells, maximum
`1.84356055604884082e-04`, under the production JIT closure.  Eager execution
names the same boundary.  The compiled write is
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:262-268`.

## Frozen preregistration and admission

The predictions and falsifiers were committed in
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round178.md` before the
record was parsed or the production walk was run.

The operator-reported Round-177 header `(1,1081,1,2,3,...)` is correct.  The
compiled writer serialises `Kbb,Kmm,Krhs` in that order at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:403-419`;
therefore the parser's old expected `Kmm=1` was stale.  The inherited
Round-123 calibration header was independently stale in the same way and was
corrected from its compiled writer before admission.  The existing record was
not rebuilt.

Admission reproduced:

- step-1080 and step-1440 restarts byte-identical to the uninstrumented daily
  record;
- exact 8,903,548-byte layout, 18,000 wet cells, and finite registered rows;
- Round-123 lateral-diffusion increment calibration: zero unequal cells;
- record-magic plant: `STATUS PLANT-FIRED`, exit 69;
- current-commit daily audit: 360/360 boundaries, 12/12 monthly overlaps,
  `STATUS ADMITTED`.

The active program is not inferred from a generic source tree.  The compiled
step calls the standard slope producer at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/stprk3.f90:173-180`, stage 3
calls `tra_ldf` at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/stprk3_stg.f90:928-934`, and
the dispatcher selects the standard iso-neutral Laplacian at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf.f90:105-118`.

## Production-step result

Both modes start from NEMO's admitted day-180 restart and execute the complete
`LatLonCGridOceanModel` step.  The JIT mode is authoritative.  The diagnostic
step and a separately compiled ordinary step have zero unequal carried-state
bytes in both modes.

| row | JIT unequal | JIT max abs | eager unequal | eager max abs |
|---|---:|---:|---:|---:|
| `T_Kbb` | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `tmask` | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `umask` | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `vmask` | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `wmask` | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `ahtu` | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `ahtv` | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| **`uslp`** | **16,820** | **1.84356055604884082e-04** | **16,820** | **1.84356055603913938e-04** |
| `vslp` | 16,530 | 2.73965483207163664e-04 | 16,530 | 2.73965483211071996e-04 |
| `wslpi` | 17,400 | 2.09879881944775512e-04 | 17,400 | 2.09879881945532719e-04 |
| `wslpj` | 17,400 | 2.40647324556912678e-04 | 17,400 | 2.40647324560430698e-04 |
| `dit` | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `djt` | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `dkt` | 0 | 0.00000000000000000e+00 | 0 | 0.00000000000000000e+00 |
| `e3t_live` | 18,000 | 4.28635735261195805e-06 | 18,000 | 4.28635735261195805e-06 |
| `e3u_live` | 17,400 | 3.21099863447216194e-03 | 17,400 | 3.21099863447216194e-03 |
| `e3v_live` | 17,100 | 3.53166745742328203e-03 | 17,100 | 3.53166745742328203e-03 |
| `A11` | 17,400 | 3.21099863447216194e-03 | 17,400 | 3.21099863447216194e-03 |
| `A22` | 17,100 | 3.53166745742328203e-03 | 17,100 | 3.53166745742328203e-03 |
| `hmsku` | 580 | 2.50000000000000000e-01 | 580 | 2.50000000000000000e-01 |
| `hmskv` | 570 | 2.50000000000000000e-01 | 570 | 2.50000000000000000e-01 |
| `A13` | 16,820 | 4.88543547352943364e+00 | 16,820 | 4.88543547350371909e+00 |
| `A23` | 16,530 | 7.26008530498982907e+00 | 16,530 | 7.26008530509339778e+00 |
| `fu` | 17,671 | 1.29145213722323097e+04 | 17,671 | 1.29145213722853659e+04 |
| `fv` | 17,352 | 1.40430068562951783e+04 | 17,352 | 1.40430068563528621e+04 |
| `vmsku` | 600 | 5.00000000000000000e-01 | 600 | 5.00000000000000000e-01 |
| `vmskv` | 600 | 5.00000000000000000e-01 | 600 | 5.00000000000000000e-01 |
| `ahu_w` | 600 | 1.00000000000000000e+03 | 600 | 1.00000000000000000e+03 |
| `ahv_w` | 600 | 1.00000000000000000e+03 | 600 | 1.00000000000000000e+03 |
| `A31` | 18,000 | 5.56181687153655366e+03 | 18,000 | 5.56181687155662075e+03 |
| `A32` | 18,000 | 6.37715410075818363e+03 | 18,000 | 6.37715410085140320e+03 |
| `fw_lower` | 17,400 | 7.11053448342296178e+03 | 17,400 | 7.11053448336325528e+03 |
| `fw_upper` | 17,409 | 7.11053448342296178e+03 | 17,409 | 7.11053448336325528e+03 |
| `Krhs_increment` | 18,000 | 2.47371009922240928e-08 | 18,000 | 2.47371009924515741e-08 |

NEMO builds the density-gradient, limiter, mixed-layer recurrence, and
unfiltered `zwz` at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:222-268`, then the
first observed non-bit output is the Shapiro-filtered `uslp` assignment at
`:262-268`.  This names the first **observed output statement**, not yet its
causal operand: the Round-177 stream does not contain `zgru`, `zdzr`, `pn2`,
`nmln`, `zwz`, or the mixed-layer accumulator.  Claiming the Shapiro
association itself as the owner would therefore be post-hoc and unsupported.

Within `traldf_iso`, the compiled order is A33 setup at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:167`, tracer
gradients at `:215-250`, horizontal tensor/flux writes at `:272-299`, vertical
tensor/flux writes at `:311-344`, and final RHS accumulation at `:346-367`.
The table scores every production-returned operator row needed to locate the
first boundary.  The record's raw base metrics/ratios and its separately
computed `ah_wslp2`/`akz` pair remain record-admitted but are not falsely
labelled independent production outputs; they occur downstream of the already
non-bit slope boundary.

## Prediction verdict and controls

- **REFUTED:** the preregistered prediction that the first non-bit row would be
  live QCO `e3u/e3v`.  `uslp` is earlier in compiled execution and much larger.
- **CONFIRMED:** entry temperature, four masks, both static diffusivities, and
  all three direct tracer-gradient rows are bit-exact.
- **CONFIRMED:** JIT and eager name the same first boundary.
- **CONFIRMED:** the diagnostic return is passive: zero unequal carried-state
  bytes in both modes.
- **PLANT-FIRED:** moving one active JIT `e3u` value by one ULP moves one `A11`
  cell, one `zfu` cell, and two final tendency cells, while `q`, `dit`, `djt`,
  and `dkt` remain unchanged.  Exit 1 is the required plant result.

## Landing and trajectory gates

There is no candidate patch: only a write-only diagnostic return and a private
plant override were added, both default-off, with an independent ordinary
step proving carried-state identity.  Consequently the certified GYRE ladder,
month, year, tanks, DINO, generic recipe, and ORCA2 trajectories were not
re-run and no before arm moved.  The receipt carries forward the current
immutable values solely as context: kt2 U/V
`8.330180021824198e-17 / 9.714451465470120e-17`, kt3 T/S
`4.9400710722e-7 / 4.0086298725e-8`, day-30 T RMS
`6.57257437477e-5 K`, day-240 `1.64483607012e-2 K`, and day-360
`1.12256600186e-2 K`.  A future slope candidate is shared-risk for DINO and
must run the recipe-derived card census plus the full Decision-43/45/55/59
gate before it can land.

## Independent review

The required read-only `codex exec` pass was attempted.  It produced no
scientific verdict because the in-process app-server could not initialise in
the read-only sandbox.  Its terminal line, quoted verbatim, was:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore: **independent review unavailable in-sandbox**.  This is not reported
as SHIP, HOLD, or DO NOT SHIP.

## Evidence and tests

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round178/`.

| artifact | SHA-256 |
|---|---|
| `admission.log` | `e0d6634c037e5fbbe06b01442723a980bbb7f08c98dbc583d86a0b046b9e51fc` |
| `admission_plant.log` | `63090aaeb297008e10f2d49d25538dbcf5a42640d8da96475946f0fabae80c0a` |
| `daily_record_audit_final.json` | `19bfd7b2c0fbfada8bfb7a8ef3a1334ff51be8065cbeefa4ae1cbc888d46a69c` |
| `developed_tracer_ldf_walk_final.json` | `210e09ea0452eae56c14cf5217cc263230d83719dd410c6c0aab814acbfa0766` |
| `developed_tracer_ldf_plant_final.log` | `48d1474c1251bf2dd07c9ef5c7c7fb3dd2f0e1e724e8fa167739eaf0ea8e8b13` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |

Focused harness suite before the final receipt/citation commit: **45 passed,
1 skipped**.  The final citation gate and its shifted-citation plant are run
against the committed receipt; their exit statuses are carried in the final
round report without rewriting this evidence-bearing receipt afterward.

## OPEN — Round 179

Stay on the magnitude owner and walk `ldf_slp` from NEMO's same day-180 entry.
Extend the existing Round-178 harness; do not create a second harness.  Acquire
or reuse one passive record containing the compiled producer inputs and
ordered intermediates `prd`, `pn2`, `nmln`, `r3t/r3u/r3v`, `zgru`, `zdzr`,
`zau/zbu`, limiter choices, mixed-layer indices/accumulators, `zwz`, and the
filtered `uslp`.  Under production JIT and eager execution, name the first
non-bit operand or statement before `uslp`.  The record admission is restart
identity plus in-run calibration under note AS.  Do not change the live
thickness route or any downstream tensor statement while this earlier slope
boundary remains open.  If the first statement is one-variable and NEMO-exact,
run the full trajectory/card/tank gate; otherwise remain HELD with the named
operand and a fail-closed plant.
