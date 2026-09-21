# NEMO testcase L2 GYRE phase 3 — round 138 developed external walk receipt

Date: 2026-09-21

Incoming lane tip: `4be746b6f8022672458a8349624cbb2f65b497f2`

Preregistration commit: `440de909e`

Measurement-instrument commit: `755c6028a28ea007d468e1c4bbbcf061cf7c2f41`

Citation-map commit: `580654ece`

Status: **HELD — no production physics, card, carried state, or configuration
changed. At NEMO's exact day-180 entry, the production-JIT external solve is
BIT through substep 1's entry, histories, midpoint, continuity, pressure,
Coriolis, and drag boundaries. Its first non-bit statement boundary is the
frozen slow-forcing subtraction: U differs in all 580 wet faces by at most
`4.2854247978022983e-13`, and V differs in all 570 wet faces by at most
`4.4333086294645174e-13`. The completed external `pssh` differs in all 600 wet
columns, not the preregistered 599. The downstream T-point QCO statement is
BIT under exact model and NEMO `pssh` inputs, so it is exonerated. The record
does not split the incoming slow RHS from the initializing `dyn_cor_2D` term;
no further upstream owner is claimed.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round138/`

## Outcome first

The executing dispatch calls the one split-explicit solve at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stp2d.f90:291-298`. NEMO initializes
the frozen SSH and momentum forcing, evaluates `dyn_cor_2D`, and subtracts that
term from `zu_frc`/`zv_frc` at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:289-325`. The
Round-137 stream writes those final two frozen forcing fields after the
pressure, Coriolis, and drag operands at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:631-684`.

That output is the first production-JIT non-bit boundary:

| row | wet cells unequal | max absolute | RMS | first unequal |
|---|---:|---:|---:|---|
| `slow_u`, substep 1 | 580 / 580 | `4.2854247978022983e-13` | `7.001320686853405e-14` | `[1,1]` |
| `slow_v`, substep 1 | 570 / 570 | `4.4333086294645174e-13` | `6.730382981311166e-14` | `[1,1]` |
| `u_exit`, substep 1 | 580 / 580 | `1.234202341021673e-10` | `2.0163803584552738e-11` | `[1,1]` |
| `v_exit`, substep 1 | 570 / 570 | `1.2767928901646908e-10` | `1.93835029722698e-11` | `[1,1]` |

In compiled order, every substep-1 row before `slow_u` is BIT. The U statement
at line 323 is therefore the first observed non-bit statement; the V twin at
line 324 is non-bit too. This does **not** prove the subtraction arithmetic is
wrong: the record contains its output but not the separate incoming
`Ue_rhs`/`Ve_rhs` and initializing `zu_trd`/`zv_trd` operands. The next walk
must split those operands before naming an upstream momentum operator.

NEMO's substep program advances midpoint, face geometry, transport, and fresh
SSH in
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:462-591`, applies
the pressure/Coriolis/drag/frozen-forcing velocity update in
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:631-684` and
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:697-759`, then rotates
the three absolute histories through
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:811-844`. The measured
causal order follows it: substep-1 SSH is BIT; all 600 SSH columns first become
non-bit at substep 2; all remain registered before NEMO finalizes and writes
`pssh(:,:,Kaa)` at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:857-890`.

The final production-JIT `pssh` result is:

| row | wet columns unequal | max absolute | RMS | first unequal |
|---|---:|---:|---:|---|
| legoESM final external `pssh` vs NEMO | 600 / 600 | `5.3786374312747576e-9 m` | `1.7490351967052955e-9 m` | `[1,1]` |

This **REFUTES** the frozen 599-column prediction. The ordinary Round-136
consumed `q_Kaa` result is reproduced exactly as 599 / 600 unequal, max
`1.2506662372402388e-12`, RMS `4.066795963906242e-13`; its unequal set is not
the 600-column raw-final-`pssh` set. No equality is asserted between those
sets.

## Record readmission and entry completeness

The exact acquired external record is 10,444,796 bytes with SHA-256
`6bc0f990ccba183a48ad09549b72b549603694e917389eb4f8b9c03c88ceaf09`.
The QCO record is 22,508 bytes with SHA-256
`626d21e229f7ced8f606f6385e04224fb2e086cef92d81d95dff7e2ef3e90878`.
Both stamps name producer commit
`4be746b6f8022672458a8349624cbb2f65b497f2`; the closed output manifest has 15
members. The passive step-1080 restart and step-1081 process stream reproduce
the earlier records at SHA-256
`6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976`
and `526d1fc73faeda990c661f2363a5bb328168bae17d4aea315daf05c35b4cd7b0`.

All 50 record-local midpoint, continuity, pressure, trend, update, and swap
replays are BIT. External final `pssh`, final-substep SSH, and stage-1 `ssha`
are BIT to each other. The recorded NEMO multiplication `ssha*r1_ht_0` is BIT
to recorded `r3ta`. All eight acquisition controls end in their required
`STATUS PLANT-FIRED` marker: header, truncation, replay ULP, swap ULP, final
`pssh` ULP, QCO ULP, stamp, and passive admission.

The admitted step-1080 restart bridge maps T, S, u, v, SSH, TKE, `avm`, `avt`,
`dissl`, the surface viscosity, independent depth means, and all six absolute
barotropic histories with zero unequal cells. The certified step-1081 forcing
is rebuilt by the existing campaign path. The observed production call and an
independently compiled ordinary production call return identical pytrees: 0
of 198,956 array cells differ across 23 leaves.

## Full production-JIT external registry

The authoritative `developed_external_jit.json` contains all 50 × 44 rows,
including cells unequal, max absolute, RMS, and first unequal index per row;
signed zero is non-bit. The compact table below lists every registered
boundary, the first non-bit substep (`BIT` means all 50), and the maxima over
the 50 substeps. Transport rows retain their native metric-volume units; all
other units follow their named field.

| boundary | first non-bit substep | max cells unequal | max absolute | max RMS |
|---|---:|---:|---:|---:|
| `back_coefficient_0` | BIT | 0 | `0.0` | `0.0` |
| `back_coefficient_1` | BIT | 0 | `0.0` | `0.0` |
| `back_coefficient_2` | BIT | 0 | `0.0` | `0.0` |
| `back_coefficient_3` | BIT | 0 | `0.0` | `0.0` |
| `continuity_div` | 2 | 600 | `1.1619611662483856e-11` | `1.5678950996759241e-12` |
| `cor_u` | 2 | 580 | `2.0565833317320886e-13` | `3.005620775073032e-14` |
| `cor_v` | 2 | 570 | `2.1359484353003e-13` | `3.2310963515496526e-14` |
| `depth_u_mid` | 3 | 580 | `5.365109245758504e-9` | `1.726622509509538e-9` |
| `depth_v_mid` | 3 | 570 | `4.6820787247270346e-9` | `1.6781104147388802e-9` |
| `drag_coefficient_u` | BIT | 0 | `0.0` | `0.0` |
| `drag_coefficient_v` | BIT | 0 | `0.0` | `0.0` |
| `eta_b` | 4 | 600 | `5.351997942049103e-9` | `1.7685841105007346e-9` |
| `eta_bb` | 5 | 600 | `5.3283263079517695e-9` | `1.7685841105007346e-9` |
| `eta_continuity` | 2 | 600 | `5.3786374312747576e-9` | `1.7685841105007346e-9` |
| `eta_entry` | 3 | 600 | `5.370718349217984e-9` | `1.7685841105007346e-9` |
| `eta_mid` | 3 | 600 | `5.960375547287011e-9` | `1.769200179713827e-9` |
| `eta_pgf` | 2 | 600 | `5.375264379436917e-9` | `1.7674109377532443e-9` |
| `inverse_depth_u` | 3 | 580 | `2.8961750532519037e-16` | `9.328648742702312e-17` |
| `inverse_depth_v` | 3 | 570 | `2.506404372243365e-16` | `9.068376359189805e-17` |
| `mid_coefficient_1` | BIT | 0 | `0.0` | `0.0` |
| `mid_coefficient_2` | BIT | 0 | `0.0` | `0.0` |
| `mid_coefficient_3` | BIT | 0 | `0.0` | `0.0` |
| `pgf_u` | 2 | 580 | `4.786176033136965e-13` | `7.326422809196388e-14` |
| `pgf_v` | 2 | 570 | `4.5060603824778134e-13` | `5.94085152272788e-14` |
| `slow_u` | 1 | 580 | `4.2854247978022983e-13` | `7.001320686853405e-14` |
| `slow_v` | 1 | 570 | `4.4333086294645174e-13` | `6.730382981311166e-14` |
| `ssh_forcing` | BIT | 0 | `0.0` | `0.0` |
| `swap_eta` | 2 | 600 | `5.3786374312747576e-9` | `1.7685841105007346e-9` |
| `swap_u` | 1 | 580 | `7.3136166532650204e-9` | `7.63302396406555e-10` |
| `swap_v` | 1 | 570 | `6.457154786559949e-9` | `7.807869740526549e-10` |
| `transport_u` | 2 | 580 | `3.301065686158836` | `0.3444979801436973` |
| `transport_v` | 2 | 570 | `2.9122789558023214` | `0.3523883885275589` |
| `trd_u` | 2 | 580 | `2.0564423219221944e-13` | `3.0055912008438613e-14` |
| `trd_v` | 2 | 570 | `2.1361149217436653e-13` | `3.2311249256371154e-14` |
| `u_b` | 3 | 580 | `7.024893868001758e-9` | `7.329182689634015e-10` |
| `u_bb` | 4 | 580 | `6.881277051645762e-9` | `7.177681979377295e-10` |
| `u_entry` | 2 | 580 | `7.168976440263819e-9` | `7.480895516839125e-10` |
| `u_exit` | 1 | 580 | `7.3136166532650204e-9` | `7.63302396406555e-10` |
| `u_mid` | 2 | 580 | `7.241148649445028e-9` | `7.5568209827855e-10` |
| `v_b` | 3 | 570 | `6.180208806522147e-9` | `7.49442748617581e-10` |
| `v_bb` | 4 | 570 | `6.04208071422474e-9` | `7.337250914664094e-10` |
| `v_entry` | 2 | 570 | `6.318807393990156e-9` | `7.651367557714219e-10` |
| `v_exit` | 1 | 570 | `6.457154786559949e-9` | `7.807869740526549e-10` |
| `v_mid` | 2 | 570 | `6.388238941307245e-9` | `7.729788945956343e-10` |

The production-eager control is intentionally not the production verdict. It
first differs at substep-1 `ssh_forcing`: 180 / 600 columns, max
`3.308722450212111e-24`, RMS `9.321206939732753e-25`. Production JIT keeps
that boundary BIT and first differs at `slow_u`. Eager final `pssh` still
differs in 600 / 600 columns, max `5.378637389641394e-9`, RMS
`1.7490351962101603e-9`. This is the required eager/JIT rounding distinction,
not a second owner claim.

## QCO exact-input adjudication

After the external solve, stage 1 passes completed `ssha` to QCO and installs
the returned after-level ratios at
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stprk3_stg.f90:176-205`. The
compiled T-point statement is
`GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/domqco.f90:237-258`.

Two production-JIT directed arms replace only the already-completed external
SSH endpoint and reuse the ordinary model's transport averages:

| arm | consumed `q_Kaa` vs `1 + pssh*r1_ht_0` | vs recorded NEMO `1+r3ta` | observer returned state |
|---|---|---|---|
| model captured final `pssh` | BIT, 0 / 600 | not applicable | BIT, 0 / 198,956 |
| NEMO recorded final `pssh` | BIT, 0 / 600 | BIT, 0 / 600 | BIT, 0 / 198,956 |

The ordinary process-trace observer also returns a BIT-identical state, 0 /
198,956 cells. Therefore the QCO multiplication is not the source of the
developed geometry discrepancy. Its ordinary 599-column result inherits the
upstream external endpoint.

## Frozen predictions and falsifiers

| preregistered item | result | disposition |
|---|---|---|
| exact record sizes, stamps, replay, passivity, and eight plants | all exact/fired | CONFIRMED |
| ordinary and external-observed production states are BIT | 0 / 198,956 unequal | CONFIRMED |
| final `pssh` differs in exactly 599 columns | 600 / 600 | **REFUTED** |
| final-`pssh` unequal set equals Round-136 `q_Kaa` set | 600-column set vs 599-column set; unequal | **REFUTED** |
| first production-JIT mismatch is substep-1 `slow_u` or `slow_v` | `slow_u`, then `slow_v` | CONFIRMED |
| substep-1 SSH is BIT; first SSH mismatch is no earlier than substep 2 | first mismatch substep 2 | CONFIRMED |
| every final unequal column appears inside the registered loop | all 600 first appear at substep 2 | CONFIRMED |
| production QCO is exact under model and NEMO endpoint inputs | both exact; NEMO arm exact to record | CONFIRMED |
| entry/final/registry plants exit nonzero | all three exit 1 with `STATUS PLANT-FIRED` | CONFIRMED |

The failed predictions remain failed; the receipt does not rewrite 600 as
599 or call the two unequal sets equivalent.

## Controls, review, and verification

The round's three controls were run from clean commit
`755c6028a28ea007d468e1c4bbbcf061cf7c2f41`:

| control | exit | result |
|---|---:|---|
| remove one compiled boundary from the 44-row registry | 1 | `ROUND138 MISSING-BOUNDARY PLANT STATUS PLANT-FIRED` |
| move one consumed wet entry SSH by one ULP through production JIT | 1 | `ROUND138 ENTRY-SSH-ULP PLANT STATUS PLANT-FIRED` |
| move one recorded final `pssh` by one ULP | 1 | `ROUND138 FINAL-PSSH-ULP PLANT STATUS PLANT-FIRED` |

The required separate read-only Codex command was invoked against the whole
round diff and the evidence. It could not initialize its in-process app-server
inside the read-only sandbox, so it emitted no verdict. Its output was,
verbatim:

```text
Error while loading conda entry point: conda-anaconda-tos (cannot import name 'validate_prefix_exists' from 'conda.cli.install' (/home/dbalwada/miniconda3/lib/python3.13/site-packages/conda/cli/install.py))
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Independent review unavailable in-sandbox; per the standing operator rule the
round continued. No `DO NOT SHIP` verdict exists.

The clean-tree focused invocation covered the Round-81 parser, the extended
Round-82 walk, the developed year-owner bridge, every receipt-citation control,
and the complete fidelity time-level unit file. Its exact summary was
`86 passed in 33.46s`.

The clean-tree receipt citation gate found 9 citations, zero unmapped
citations, zero failures, and zero failing map entries. Shifting the
frozen-forcing citation by two lines exited 1 with
`SYMBOL-NOT-AT-LINE`; all of the gate's own planted controls fired too.

Evidence digests:

| artifact | SHA-256 |
|---|---|
| `developed_external_jit.json` | `d395cb706696963fd7f87176342e17d84b20c8d34e9a8ae3875351007a1877c6` |
| `developed_external_eager.json` | `597a3288632940e3a6c9b3e765c99b880c728f3f9dcbca13ae74a1e9cb6811ac` |
| `boundary_summary_jit.json` | `75bbe5dd3bae92cf5528840fcf32011730d229326563fe8d88c51609091df8c7` |
| `missing-boundary_plant.json` | `9ee65bab9b89cc58dedb8312b885de4c901f90d4c7bb9a20a9f1e06b94c93e0d` |
| `entry-ssh-ulp_plant.json` | `71924decb9154b8949a30617bac714980c3135a2c85a71c266e5939d1e736ed3` |
| `final-pssh-ulp_plant.json` | `7b14802925bbb30dce10b17d9772ddf822df7b718c6c174ab826a2f483582b39` |
| `focused_tests.log` | `1eeb1de11609c332db187a1cc7d7ab6000821767f6b1bc25759aed2f66822736` |
| `citation_gate_draft.json` | `b70d8a842a7e87fc4c27fab6e680417a5b7860db5229fd7c9ff4b76e6e9640e5` |
| `citation_gate_shifted_plant_draft.json` | `5a9cb683dd9252647eda0ddc433d676dcfb102456291ab4ceac4c8ef61dc14bd` |
| `codex_review_unavailable.txt` | `92015cb4c0f52be782d229ae8a686707c0122077dd13c515153d648e2a22da67` |

## Campaign surfaces and headline numbers

This diagnostic round changes only a committed gate, its unit tests, citation
pins, preregistration, and this receipt. It changes no production module,
recipe, card, public/private configuration default, carried-state schema,
restart contract, stabilizer, or NEMO source. Therefore no Decision-43/45
candidate exists, and rerunning the ladder/month/year acceptance arms would
measure the same production commit.

The campaign headline values remain incoming certified values, not round-138
remeasurements:

| row | value | round-138 disposition |
|---|---:|---|
| kt2 U | `2.7377110452773967e-12 m/s` | unchanged |
| kt2 V | `3.2849219221489645e-12 m/s` | unchanged |
| kt3 T | `8.659373840202989e-7 K` | unchanged |
| kt3 S | `7.027291104577671e-8 g/kg` | unchanged |
| day-30 T3D RMS | `6.890431487825909e-5 K` | unchanged |
| day-240 T3D RMS | `1.644674193e-2 K` | unchanged |
| day-360 T3D RMS | `1.122357391e-2 K` | unchanged |

DINO, LOCK_EXCHANGE, OVERFLOW, and the tank statements execute no changed
production code and therefore retain their bits. ORCA2 remains
`UNMEASURED-WITH-SPEC`: this round adds diagnostic support for the resolved
GYRE forward/RK3 card only and makes no ORCA2 identity claim.

## OPEN — round 139

1. Stay at the first production-JIT non-bit statement. Preregister a developed
   extension of the existing Round-83 slow-forcing walk; do not create a
   second model stepper.
2. At step 1081, passively record the separate incoming `Ue_rhs`/`Ve_rhs` and
   the `zu_trd`/`zv_trd` returned by the initializing `dyn_cor_2D` immediately
   before the line-323/324 subtraction, plus the final `zu_frc`/`zv_frc`.
   Reuse the Round-137 executable inputs and admission identities under a new
   target name. This is the minimum record needed to decide which operand
   first differs; no current admitted developed stream contains it.
3. Drive the existing production-JIT slow-forcing operand registry from the
   same exact step-1080 restart. Require ordinary-state identity and a
   production plant. If incoming `Ue_rhs` is first, continue its existing
   compiled-order RHS decomposition; if the initializing Coriolis result is
   first, walk that operand. Do not infer either result from the final
   subtraction.
4. Do not revisit QCO, FCT, EVD, TKE, or a held rest-state patch: the exact-input
   QCO result is closed and the developed external solve is upstream.

`ACQUISITION_NEEDED`: `NONE` — round 139 may write and run the bounded GYRE
acquisition itself under the current campaign policy.

`DECISION_NEEDED`: `NONE`

`ROUND_STATUS`: `HELD`
