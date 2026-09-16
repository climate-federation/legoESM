# GYRE parallel TKE K_H held-candidate receipt

Date: 2026-09-16. Production parent: `3e7a15c1e64e`; evidence branch
`held/gyre-tke-kh-evidence`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/parallel/tkekh/tke_kh_evidence/`.
The production patch is **HELD**: Decision 41 permits re-evaluation only when
the stage walk reaches this candidate's owning stage, and the Rule-12 ladder
below fails the no-worsening gate. Nothing in this receipt was landed or
pushed.

## 1. Receipt recovery and reproduction

The historical item is recorded in two receipts:

- `nemo_testcases_l2_gyre_decision36_nemo_face_shear_receipt.md:75` records
  K_H moving from `5,325` unequal cells / `7.763531835e-03` maximum absolute
  residual to **`5,310 / 17,400`** / `5.636255351e-13`.
- `nemo_testcases_l2_gyre_round62_residual_receipt.md:62-71` identifies the
  admitted round-59 TKE record, the carried-entry differences, the exact
  matrix, the 238-cell RHS boundary, and repeats the **5,310-cell** K_H
  baseline.

The replay used
`round59/oracle_tke_operands/oracle_tke_operands_kt00000002.bin`, SHA-256
`b8f3bead4f30b78257153a0c40c5dd77656aee227b1a94a46625a5604faaca5d`,
whose producer stamp is `1a695951be1396abdd4d7a85f57b31de9c0917df`.
The record resolves `kt=2`, `Kbb=Kmm=3`, binary64, `nn_mxl=3`, `nn_pdl=1`,
`ln_mxl0=.true.`, `rn_ediff=0.1`, and
`rmxl_min=0.009999999999999998`.

The preregistered historical-count prediction was **REFUTED on the current
tip**: `tip_statement_walk.json` gives `5,312 / 17,400` unequal K_H cells and
maximum absolute residual `5.573874695130598e-13`, not 5,310 and
`5.636255351326724e-13`. This is reconciled, not relabeled: comparison with
the immutable round-62 `kt1_closure_walk_v4.json` changes exactly one carried
operand score, `sh2` maximum residual
`1.517631578573490e-18 -> 1.5098031412562882e-18`; all other carried-operand
score rows are identical. The historical receipt remains the receipt for the
named 5,310-cell item; the current-tip reproduction is the 5,312-cell row.

## 2. Executing compiled path and first non-bit statement

The executing GYRE binary calls `tke_tke` and then `tke_avn` at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:190-201`.
The recorded `nn_mxl=3`, `nn_pdl=1`, and `ln_mxl0=.true.` prove that the
surface arm at `:589,612-620`, raw buoyancy-length loop at `:627-630`,
case-3 bounds at `:669-683`, closure assignments at `:690-695`, and inverse
Prandtl update at `:699-702` execute. The source-ordered walk supplies NEMO's
recorded `en_post_sweep`, `rn2`, `rn2b`, `e3t`, stress, masks, floors, and
carried coefficients at every boundary; equality is uint64 equality over the
17,400 consumed wet interfaces.

| execution order | compiled statement | unequal | max abs | verdict |
|---:|---|---:|---:|---|
| 1 | surface length, `:589,612-620` | 0 | `0` | bit-exact |
| 2 | `zrn2=MAX(rn2,rsmall)` then `zmxlm=MAX(rmxl_min,SQRT(2*en/zrn2))`, `:627-630` | **13,947** | `7.975002901582895e+06` | **FIRST NON-BIT** |
| 3 | `nn_mxl=3` slope bounds, `:669-683` | 13,658 | `7.105427357601002e-15` | downstream non-bit |
| 4 | `zsqen=SQRT(en)`, `:691` | 0 | `0` | bit-exact |
| 5 | `zav=rn_ediff*zmxlm*zsqen`, `:692` | 3,634 | `6.938893903907228e-18` | non-bit |
| 6 | viscosity floor/mask, `:693` | 3,634 | `6.938893903907228e-18` | non-bit |
| 7 | diffusivity floor/mask before Prandtl, `:694` | 11,946 | `6.938893903907228e-18` | non-bit |
| 8 | `dissl=zsqen/zmxld`, `:695` | 12,886 | `2.6020852139652106e-18` | non-bit |
| 9 | inverse-Prandtl K_H update, `:699-702` | 3,443 | `2.6020852139652106e-18` | non-bit |

Therefore the first non-bit statement is unambiguously the raw
buoyancy-length assignment at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:627-630`.
The large raw maximum is real: the current GYRE card selects the historical
factored expression with an `N2` floor of `1e-12`, whereas compiled NEMO uses
`rsmall = 0.5*EPSILON` and the literal multiply/divide association.

Ownership is pre-stage vertical physics at kt=1. Compiled
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:154-168` constructs
`rn2b`, copies it to `rn2`, and calls `zdf_phy(kstp,Nbb,Nbb,Nrhs)` before the
WS-RK3 staged tendencies consume the resulting mixing coefficients. The
manifest header therefore names **kt=1 pre-stage vertical physics/TKE
closure, feeding stage 1** as the owning stage.

## 3. Prediction 3 refuted; separate post-hoc aggregate

`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_parallel_tke_kh.patch`
contains, but does not apply, the tied candidate:

1. select the literal `SQRT((2*en)/MAX(rn2,rsmall))` GYRE arm;
2. execute the first `nn_mxl=3` bottom-up iteration for carried `jpkm1` from
   the untouched `jpk` floor plus `e3t(jpk)`
   (`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:674-676`);
3. retain and directly multiply NEMO's inverse-Prandtl `pdlr`, avoiding the
   extra `pdlr -> Pr -> division` round trip (`:394-413,699-702`).

**Frozen Prediction 3 — REFUTED.** It required every statement row through
final K_H to have zero unequal cells. The archived
`candidate_statement_walk.json` (SHA-256
`4341cc292ae0889b2a1dfed09ca10e87ec8b2c2cc57f5697e7e9f5dc35aa37d2`)
instead records
`model_substitution_walk.kh_statement_walk.all_exact=false`. Its seven
non-exact statement rows are below. All compiled mappings refer to
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90`.

| statement row | compiled mapping | unequal / compared | max abs |
|---|---|---:|---:|
| `raw_buoyancy_length` | `:627-630` | **2,098 / 17,400** | `1.4210854715202004e-14` |
| `bounded_mixing_length` | `:634-683` | **2,052 / 17,400** | `3.552713678800501e-15` |
| `raw_zav` | `:692` | **558 / 5,723** | `3.469446951953614e-18` |
| `avm_floor_and_mask` | `:693` | **558 / 17,400** | `3.469446951953614e-18` |
| `avt_floor_and_mask_before_prandtl` | `:694` | **1,694 / 17,400** | `3.469446951953614e-18` |
| `dissipation_length_output` | `:695` | **1,844 / 17,400** | `1.734723475976807e-18` |
| `avt_inverse_prandtl_update` | `:699-702` | **522 / 17,400** | `1.734723475976807e-18` |

**Post-hoc whole-closure aggregate — CONFIRMED, not Prediction 3.** Applied
only in scratch commit `3849b99199c6`, the later
`source_order_closure_k_h` aggregate evaluates the whole closure in one
expression. The distinct
`model_substitution_walk.kh_statement_walk.local_candidate_row` field in the
same archived artifact records **0 / 17,400 unequal K_H cells, max abs 0**
given NEMO's recorded entry operands. This aggregate was not the frozen
statement-chain prediction and does not fulfil it.

The aggregate's separately archived non-vacuity evidence is
`candidate_ulp_plant.log` (SHA-256
`a71febe6df7c7955787351560e1acb665535247e9682464c6a7ac9ed6e3bf278`).
Raising recorded `en_post_sweep[1,1,0]` by exactly one binary64 ULP changes
that aggregate to **1 / 17,400 unequal cells** and makes the gate exit
nonzero:

```text
STATUS FAIL: planted one-ULP TKE entry violation detected at (1, 1, 0): source_order_closure_k_h has 1 unequal cell(s)
```

## 4. Information-only kt=1..10 ladder against tip

Both sides used
`nemo_testcase_l2_gyre_phase3_gate.py --trajectory-only --max-step 10`, the
same admitted round-59 root for oracle/stage2/stage3, and clean
CPU/fp64/x64/libm stamps. The tip reproduces the supplied controls exactly:
kt2 U/V `2.7377110452773967e-12 / 3.284922138989399e-12`; kt3 T/S
`1.627497246303733e-4 / 6.327735185607253e-6`. The supplied day-30 T RMS
`1.2397011295506804e-2 K` is a recorded control only; no candidate day-30 run
was requested or claimed.

The certified comparison is **FAIL**: 70 rows compared, 62 move, the largest
tip-to-candidate move is `78,428.984375` row-scale oracle ULP on kt10 V, and
many cells worsen by more than the two-ULP allowance. First-over-bar remains
kt2 U/V on both sides. `bit-moved` is a direct uint64 comparison of the two
gate residual sidecars; improved/worsened and the ULP scale are the gate's
oracle-relative values.

| kt | row | bit-moved | improved | worsened | max tip-to-candidate move | max move (row ULP) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `before.T` | 0 | 0 | 0 | `0` | `0` |
| 1 | `before.S` | 0 | 0 | 0 | `0` | `0` |
| 1 | `before.u` | 0 | 0 | 0 | `0` | `0` |
| 1 | `before.v` | 0 | 0 | 0 | `0` | `0` |
| 1 | `before.ssh` | 0 | 0 | 0 | `0` | `0` |
| 1 | `after.uu_b` | 0 | 0 | 0 | `0` | `0` |
| 1 | `after.vv_b` | 0 | 0 | 0 | `0` | `0` |
| 2 | `before.T` | 12 | 4 | 7 | `7.105427358e-15` | `2` |
| 2 | `before.S` | 15 | 5 | 9 | `1.421085472e-14` | `2` |
| 2 | `before.u` | 560 | 273 | 287 | `4.857225733e-17` | `0.21875` |
| 2 | `before.v` | 531 | 291 | 240 | `4.857225733e-17` | `0.21875` |
| 2 | `before.ssh` | 0 | 0 | 0 | `0` | `0` |
| 2 | `after.uu_b` | 575 | 330 | 245 | `5.271933064e-18` | `0.0237427` |
| 2 | `after.vv_b` | 566 | 260 | 306 | `4.662069342e-18` | `0.0209961` |
| 3 | `before.T` | 197 | 74 | 123 | `7.105427358e-15` | `2` |
| 3 | `before.S` | 158 | 72 | 86 | `1.421085472e-14` | `2` |
| 3 | `before.u` | 16,914 | 9,408 | 7,506 | `8.118505868e-15` | `36.5625` |
| 3 | `before.v` | 16,202 | 8,420 | 7,782 | `8.305856003e-15` | `37.4062` |
| 3 | `before.ssh` | 600 | 303 | 297 | `4.732528930e-15` | `21.3134` |
| 3 | `after.uu_b` | 579 | 220 | 359 | `4.662069342e-17` | `0.209961` |
| 3 | `after.vv_b` | 568 | 280 | 288 | `3.526367566e-17` | `0.158813` |
| 4 | `before.T` | 927 | 450 | 477 | `2.669864330e-11` | `7,515` |
| 4 | `before.S` | 652 | 336 | 316 | `4.261124786e-11` | `5,997` |
| 4 | `before.u` | 17,336 | 8,653 | 8,683 | `1.621025905e-13` | `730.045` |
| 4 | `before.v` | 16,982 | 8,712 | 8,270 | `1.621018857e-13` | `730.042` |
| 4 | `before.ssh` | 600 | 285 | 315 | `6.829714745e-14` | `307.583` |
| 4 | `after.uu_b` | 579 | 291 | 288 | `3.029260870e-16` | `1.36426` |
| 4 | `after.vv_b` | 570 | 292 | 278 | `2.042636893e-16` | `0.919922` |
| 5 | `before.T` | 4,753 | 2,221 | 2,532 | `4.598987857e-11` | `12,945` |
| 5 | `before.S` | 2,030 | 1,050 | 980 | `4.755662530e-11` | `6,693` |
| 5 | `before.u` | 17,393 | 7,930 | 9,463 | `2.975120150e-13` | `1,339.88` |
| 5 | `before.v` | 17,086 | 8,397 | 8,689 | `3.184362496e-13` | `1,434.11` |
| 5 | `before.ssh` | 600 | 321 | 279 | `5.645577322e-13` | `2,542.54` |
| 5 | `after.uu_b` | 580 | 317 | 263 | `1.669129245e-15` | `7.51709` |
| 5 | `after.vv_b` | 570 | 257 | 313 | `2.046472258e-15` | `9.21649` |
| 6 | `before.T` | 10,746 | 5,193 | 5,553 | `6.392042451e-11` | `17,992` |
| 6 | `before.S` | 5,050 | 2,614 | 2,434 | `1.027586904e-10` | `14,462` |
| 6 | `before.u` | 17,399 | 8,737 | 8,662 | `6.964463728e-13` | `3,136.52` |
| 6 | `before.v` | 17,097 | 8,255 | 8,842 | `7.452389400e-13` | `3,356.26` |
| 6 | `before.ssh` | 600 | 282 | 318 | `6.811927324e-13` | `3,067.82` |
| 6 | `after.uu_b` | 580 | 278 | 302 | `2.622576635e-15` | `11.811` |
| 6 | `after.vv_b` | 570 | 315 | 255 | `2.574221218e-15` | `11.5933` |
| 7 | `before.T` | 14,846 | 7,315 | 7,531 | `7.537082070e-11` | `21,215` |
| 7 | `before.S` | 8,611 | 4,280 | 4,326 | `1.246505121e-10` | `17,543` |
| 7 | `before.u` | 17,400 | 8,830 | 8,570 | `1.358508792e-12` | `6,118.18` |
| 7 | `before.v` | 17,100 | 8,443 | 8,657 | `3.384410807e-12` | `15,242` |
| 7 | `before.ssh` | 600 | 296 | 304 | `1.758056374e-12` | `7,917.58` |
| 7 | `after.uu_b` | 580 | 243 | 337 | `4.334206605e-15` | `19.5195` |
| 7 | `after.vv_b` | 570 | 278 | 292 | `3.943785402e-15` | `17.7612` |
| 8 | `before.T` | 16,339 | 8,120 | 8,219 | `8.235190307e-11` | `23,180` |
| 8 | `before.S` | 11,962 | 6,012 | 5,934 | `1.424069751e-10` | `20,042` |
| 8 | `before.u` | 17,400 | 8,541 | 8,859 | `3.426887246e-12` | `15,433.3` |
| 8 | `before.v` | 17,100 | 8,326 | 8,774 | `1.339730410e-11` | `60,336.1` |
| 8 | `before.ssh` | 600 | 277 | 323 | `2.074896244e-12` | `9,344.5` |
| 8 | `after.uu_b` | 580 | 321 | 259 | `7.013093990e-15` | `31.5842` |
| 8 | `after.vv_b` | 570 | 265 | 305 | `6.254545493e-15` | `28.168` |
| 9 | `before.T` | 16,865 | 8,564 | 8,301 | `1.083257928e-10` | `30,491` |
| 9 | `before.S` | 13,503 | 6,749 | 6,737 | `1.833910801e-10` | `25,810` |
| 9 | `before.u` | 17,400 | 8,501 | 8,899 | `4.873629278e-12` | `21,948.9` |
| 9 | `before.v` | 17,100 | 8,411 | 8,689 | `1.646365683e-11` | `74,145.7` |
| 9 | `before.ssh` | 600 | 299 | 301 | `2.619236425e-12` | `11,796` |
| 9 | `after.uu_b` | 580 | 306 | 274 | `1.397558284e-14` | `62.9404` |
| 9 | `after.vv_b` | 570 | 266 | 304 | `8.887828624e-15` | `40.0272` |
| 10 | `before.T` | 17,197 | 8,486 | 8,711 | `1.320970000e-10` | `37,182` |
| 10 | `before.S` | 14,626 | 7,254 | 7,358 | `1.851745424e-10` | `26,061` |
| 10 | `before.u` | 17,400 | 8,658 | 8,742 | `5.546993420e-12` | `24,981.4` |
| 10 | `before.v` | 17,100 | 8,814 | 8,286 | `1.741473285e-11` | `78,429.0` |
| 10 | `before.ssh` | 600 | 306 | 294 | `3.075113189e-12` | `13,849.1` |
| 10 | `after.uu_b` | 580 | 287 | 293 | `1.079518419e-14` | `48.6172` |
| 10 | `after.vv_b` | 570 | 264 | 306 | `8.631984016e-15` | `38.875` |

The ladder is information only and does not authorize landing. It strengthens
the Decision-41 hold: the frozen full statement-chain prediction is refuted;
the separately labelled post-hoc equal-input aggregate is exact, but the
candidate is not trajectory-monotone when introduced ahead of its owning
stage.

## 5. Evidence, validation, and disposition

| artifact | SHA-256 |
|---|---|
| `tip_statement_walk.json` | `b404179c3c03163ed48726312636f319e0aeb28f827b3b5566b900eb046aa0d6` |
| `candidate_statement_walk.json` | `4341cc292ae0889b2a1dfed09ca10e87ec8b2c2cc57f5697e7e9f5dc35aa37d2` |
| `candidate_ulp_plant.log` | `a71febe6df7c7955787351560e1acb665535247e9682464c6a7ac9ed6e3bf278` |
| `tip_kt1_10.json` | `51dfbcc65b9481e429cc319f61f9f73af4f77bb50e1da6bb8260bf9efe7f8cfd` |
| `candidate_kt1_10.json` | `b07ef59f5bca56aa573c474d7d9ab28cd7810add7fa05016c0dec572a2badc2b` |
| `ladder_comparison.json` | `400089614cf950578bcbf21ad8a39101e4b3c9847f36f82007ede95eb68cf54e` |

Validation: 125 candidate TKE/instrument tests pass in the clean scratch tree;
28 evidence-tree instrument/citation tests pass; the manifest applies cleanly
to the current evidence tip; the one-ULP plant exits nonzero; both ten-step
runs carry clean stamps; and the oracle-relative comparison exits nonzero with
the expected FAIL verdict. The citation gate passes all 15 mapped citations,
and shifting
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:627-630` by two lines
makes it exit nonzero.

Independent review: **DO NOT SHIP** on the prior receipt because it treated
the exact post-hoc aggregate as fulfilment of the refuted full-chain
Prediction 3. This fix records the refutation and separates the aggregate and
its evidence.

Disposition: **HELD, not landed**. Re-evaluate this manifest only when the
Decision-41 stage walk reaches kt=1 pre-stage vertical physics/TKE closure.
There is no user decision needed now.
