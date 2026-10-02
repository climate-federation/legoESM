# NEMO testcase L2 GYRE — decision 36 receipt: RK3 face-native shear

Date: 2026-09-12. Before-arm `24ed85bf6e9f`; after-arm
`f78547b752f7` (commits `cb5028d2dd2f`, `138de77eb757`, and
`f78547b752f7`). CPU, production JIT, fp64/scalar-libm. The predictions and
falsifiers in
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_decision36_nemo_face_shear.md`
were frozen before the after-arm ran. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/decision36_nemo_face_shear/`;
the before operand record is
`phase3/round61/kt2_operand_split_with_velocity_v6.json`.

## 1. The decision (ASKED)

The user answered **Yes** to selecting NEMO's compiled shear-production
statement on the GYRE NEMO-identity card. Final decision: **SHIP the three
reviewed commits plus this receipt**. P1 and P2 confirm the mechanism and
one-step propagation. P3 misses its ambitious month target, but Rule 12 is
satisfied: no AT-BAR row leaves the bar and first-over-bar does not move.

## 2. What NEMO does (compiled `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo`)

- `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:168` calls
  `zdf_phy(kstp,Nbb,Nbb,Nrhs)`: under RK3, Kmm and Kbb are both the whole-step
  entry level.
- `GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:83-114` sums `avm` on
  each uw/vw face, multiplies the Kmm and Kbb vertical velocity differences,
  divides by the two live `e3w_1d*(1+r3u/r3v)` factors, applies
  `wumask/wvmask`, then combines adjacent faces with `0.25` and the
  `2-mask*mask` coast factor. The executing no-Stokes arm is lines 97-109;
  the same face/metric/combine statement is visible across the cited compiled
  branch.

## 3. What changed and independent review

| commit | reviewed result |
|---|---|
| `cb5028d2dd2f` | GYRE alone selects face-native production, face `avm`, and live QCO metric. Its initial `nemo_face_native` spelling was not RK3-constructible; the next commit repairs that intermediate state. |
| `138de77eb757` | Adds/selects the explicit RK3 `nemo_face_native_now2` variant. Final card selectors are pinned at `nemo_testcase_recipe.py:236-238`. |
| `f78547b752f7` | The live Kbb face metric is statically selected from step-entry eta for `now2`; the leap-frog `nemo_face_native` arm still requires the carried eta and has no presence-based fallback (`ocean_model_latlon_cgrid.py:9148-9165`, `ocean_model_latlon_cgrid.py:9046-9119`). |

Statement walk: the helper forms `wumask/wvmask`, the live two-factor
denominator, face-summed `avm`, source-ordered products, coast factors, and
the `0.25` T-point combine in
`packages/ocean/legoesm/ocean/physics/vertical_mixing/_shared.py:384-454`.
The step-entry result is frozen and consumed by both the Prandtl calculation
and TKE production. Given NEMO operands, no arithmetic difference remains:
the NEMO-velocity replay is bit-exact (`0/17,400` unequal). The sole textual
implementation difference is the helper's defensive positive-denominator
clamp; it is inactive for valid positive GYRE face thicknesses. With legoESM
velocities, the remaining `1.518e-18` sh2 residual is upstream input debt, not
this statement.

No other current card changes behavior. DINO MLF already resolves
`nemo_face_native/nemo_face/nemo_qco_live_face`; DINO Kamm resolves
`squared_centered/tpoint` (so the face-metric branch is not reached); the
legacy `nemo_v1` path and both tanks resolve none of `now2`/live-face TKE.
ORCA2 exposes `now2` as an opt-in, but no default card selects it.

The new two-step test at
`tests/ocean/fidelity/test_nemo_testcase_l2_gyre_card_reconciliation.py:193-216`
is non-vacuous: it passes at the after tip (`1 passed`) and, in a clean
same-tip clone with only `f78547b752f7`'s model hunk reversed, fails nonzero at
kt=2 with the old `eta_before` `ValueError`. The operand walk is a real
production measurement: it requires fp64, constructs the printed card, and
enters the production routines (`nemo_testcase_l2_gyre_round54_tke_operands.py:514-552`);
its after artifact is clean-stamped `f78547b752f7`, and its plants fail.

Review verdict: **no blocking finding on the final three-commit stack**.

## 4. Frozen predictions

| prediction | before -> after | verdict against frozen claim/falsifier |
|---|---|---|
| P1 sh2 mechanism | max abs `4.102873968e-08 -> 1.517631579e-18`; unequal `17,400 -> 17,400`. K_H max abs `7.763531835e-03 -> 5.636255351e-13`; unequal `5,325 -> 5,310`. NEMO-velocity sh2 replay `0`, `0/17,400`. | **CONFIRMED**: below `1e-15`; the `>1e-12` falsifier does not fire. |
| P2 ladder | kt3 T abs `8.741316412e-03 -> 1.627511418e-04` K (normalised `3.722344243e-04 -> 6.930486749e-06`); S abs `1.356888521e-03 -> 6.327755180e-06`. The kt2 rows are unchanged; first-over-bar remains kt2 u/v. | **CONFIRMED**: T improves `53.7x` and is below `1e-3`; neither falsifier fires. |
| P3 month | day-30 3-D T rms `1.424101926e-02 -> 1.239756827e-02` K (`-12.94%`). | **REFUTED**: it does not reach `<5e-3`. The narrower “unchanged to 3 significant digits” falsifier also does not fire. |

## 5. Registered ladder rows (Rule 12)

**RETRACTION:** “All five kt2 rows are bit-identical” was false. Decision 36
left the rows unchanged: T/S are AT-BAR but non-bit, u/v are DEBT, and ssh is
exact. No AT-BAR row moves out of the bar; first-over-bar remains kt2 u/v and
barotropic kt2 uu_b/vv_b. Every one of the 39 moved regular rows improves.
Values below are normalised max abs (also the absolute max for u/v/ssh).

| kt | T before -> after | S before -> after | u before -> after | v before -> after | ssh before -> after |
|---:|---:|---:|---:|---:|---:|
| 3 | `3.722344243e-4 -> 6.930486749e-6` | `3.683245746e-5 -> 1.717656019e-7` | `7.193412342e-4 -> 1.245883653e-5` | `8.606873053e-4 -> 2.378741217e-5` | unchanged |
| 4 | `1.066212982e-3 -> 1.612684500e-5` | `8.238303196e-5 -> 4.033179923e-7` | `7.078776146e-3 -> 1.990446612e-5` | `1.409023481e-2 -> 1.904931432e-5` | `5.299303842e-7 -> 4.579229660e-7` |
| 5 | `3.059799209e-3 -> 2.449185898e-3` | `8.606416292e-5 -> 6.918625113e-5` | `9.209064428e-3 -> 9.153143673e-3` | `4.572156426e-2 -> 3.897881216e-2` | `9.473441543e-7 -> 7.731999804e-7` |
| 6 | `3.814546901e-3 -> 4.285250028e-4` | `1.278684851e-4 -> 1.017863388e-5` | `2.776572656e-2 -> 3.581970324e-3` | `6.209258255e-2 -> 1.260581325e-2` | `3.264872899e-6 -> 3.159416556e-6` |
| 7 | `4.541588625e-3 -> 1.356258216e-3` | `1.220353175e-4 -> 3.618772685e-5` | `3.604448103e-2 -> 7.816320814e-3` | `6.143938863e-2 -> 4.215021192e-3` | `6.748530600e-6 -> 6.728172383e-6` |
| 8 | `5.091248046e-3 -> 3.112920734e-4` | `1.357129927e-4 -> 7.204219372e-6` | `4.300559863e-2 -> 3.790292655e-3` | `6.203408527e-2 -> 2.412587622e-3` | `1.181473276e-5 -> 1.039449361e-5` |
| 9 | `5.435204894e-3 -> 3.628384841e-4` | `1.467046467e-4 -> 7.803669765e-6` | `4.806957096e-2 -> 2.025590653e-3` | `1.431705404e-2 -> 1.571557988e-3` | `2.166989447e-5 -> 1.436123018e-5` |
| 10 | `5.622586430e-3 -> 3.295360119e-4` | `1.543517595e-4 -> 7.090406871e-6` | `5.140358324e-2 -> 1.512706127e-3` | `8.868701469e-3 -> 5.837655355e-4` | `3.330594187e-5 -> 1.889566787e-5` |

All 16 barotropic moved rows (normalised/absolute max):

| kt | uu_b before -> after | delta | vv_b before -> after | delta |
|---:|---:|---:|---:|---:|
| 3 | `1.143453982e-7 -> 1.178364067e-7` | **+3.053%** | `7.055464101e-8 -> 7.089241596e-8` | **+0.479%** |
| 4 | `1.565440304e-7 -> 1.403524984e-7` | -10.343% | `9.577737085e-8 -> 9.411572344e-8` | -1.735% |
| 5 | `4.802626918e-7 -> 2.974609164e-7` | -38.063% | `2.509532369e-7 -> 2.197046269e-7` | -12.452% |
| 6 | `9.655733155e-7 -> 2.165363069e-7` | -77.574% | `5.139922650e-7 -> 1.306633554e-7` | -74.579% |
| 7 | `1.339961220e-6 -> 1.983263781e-7` | -85.199% | `8.751936384e-7 -> 1.492469420e-7` | -82.947% |
| 8 | `1.482587073e-6 -> 1.619217910e-7` | -89.078% | `1.073696797e-6 -> 1.491658788e-7` | -86.107% |
| 9 | `1.004516310e-6 -> 1.130643858e-7` | -88.744% | `9.305462078e-7 -> 2.324371840e-7` | -75.021% |
| 10 | `8.636846190e-7 -> 8.739959114e-8` | -89.881% | `7.199790082e-7 -> 1.898336377e-7` | -73.633% |

The only worsened ladder rows are kt3 uu_b and vv_b; both were and remain
DEBT, so they do not violate Rule 12.

## 6. Every moved day-gap statistic, days 1–30

Scored with `nemo_testcase_l2_gyre_year_owners.py --day-gap` from daily
members. Every one of the 240 statistics moves. An upward arrow marks the 25
worsenings; all unmarked cells improve. Units are the scorer's RMS units.

|day|T before -> after|S before -> after|u before -> after|v before -> after|ssh before -> after|T 0–100 m before -> after|T 100–1000 m before -> after|T >1000 m before -> after|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|`2.6994482e-3 -> 3.4102734e-4`|`2.0391299e-4 -> 1.4240025e-5`|`5.1958143e-4 -> 8.6064599e-5`|`7.4568357e-4 -> 5.2145658e-5`|`1.3518969e-6 -> 1.2616782e-6`|`5.2274173e-3 -> 6.6006467e-4`|`1.8631341e-5 -> 1.8694804e-5` ↑|`9.7280534e-7 -> 9.7766335e-7` ↑|
|2|`2.7856066e-3 -> 1.4445561e-4`|`1.4377632e-4 -> 6.7487420e-6`|`8.6901570e-4 -> 9.3089454e-6`|`1.8742702e-4 -> 1.0375870e-5`|`8.1644209e-6 -> 7.3577597e-6`|`5.3933600e-3 -> 2.6015834e-4`|`9.0197042e-5 -> 9.1907700e-5` ↑|`2.9335581e-6 -> 2.7923042e-6`|
|3|`2.8185201e-3 -> 2.7090068e-4`|`1.5600372e-4 -> 2.0235997e-5`|`5.5283561e-4 -> 2.0247238e-5`|`2.2112130e-4 -> 3.7482925e-5`|`1.7805887e-5 -> 1.6402782e-5`|`5.4539068e-3 -> 4.7815384e-4`|`1.8986064e-4 -> 1.9291918e-4` ↑|`5.7700404e-6 -> 5.6899952e-6`|
|4|`3.3104521e-3 -> 4.2961404e-4`|`1.9929520e-4 -> 2.3024076e-5`|`5.7926094e-4 -> 3.4261666e-5`|`2.8858229e-4 -> 3.4588548e-5`|`2.8945235e-5 -> 2.7198914e-5`|`6.4014422e-3 -> 7.5504529e-4`|`3.0724827e-4 -> 3.1228638e-4` ↑|`9.5352956e-6 -> 9.4508279e-6`|
|5|`3.8895058e-3 -> 6.3172211e-4`|`2.2571523e-4 -> 4.7348471e-5`|`7.4592579e-4 -> 5.5190666e-5`|`4.7500373e-4 -> 3.7375281e-5`|`4.1285204e-5 -> 3.9184772e-5`|`7.5159897e-3 -> 1.1171469e-3`|`4.3868748e-4 -> 4.4561060e-4` ↑|`1.4100103e-5 -> 1.4022054e-5`|
|6|`4.1957888e-3 -> 8.7633160e-4`|`2.8293570e-4 -> 1.4579406e-4`|`7.6618712e-4 -> 8.9383844e-5`|`3.5979115e-4 -> 7.3569558e-5`|`5.4623732e-5 -> 5.1965684e-5`|`8.0989823e-3 -> 1.5634700e-3`|`5.8195424e-4 -> 5.8980855e-4` ↑|`1.9368559e-5 -> 1.9287946e-5`|
|7|`4.4081862e-3 -> 1.0866715e-3`|`2.3556324e-4 -> 6.0531176e-5`|`7.7226024e-4 -> 7.6247968e-5`|`2.2425711e-4 -> 6.9753708e-5`|`6.8648275e-5 -> 6.5457107e-5`|`8.4965752e-3 -> 1.9325497e-3`|`7.3629234e-4 -> 7.4431945e-4` ↑|`2.5151101e-5 -> 2.5074257e-5`|
|8|`4.6568651e-3 -> 1.3494357e-3`|`2.5223081e-4 -> 7.0813231e-5`|`7.1844159e-4 -> 1.0446776e-4`|`4.6194871e-4 -> 7.4849618e-5`|`8.3165278e-5 -> 7.9487773e-5`|`8.9622322e-3 -> 2.4099725e-3`|`8.9482492e-4 -> 9.0298445e-4` ↑|`3.1371391e-5 -> 3.1295238e-5`|
|9|`4.9326765e-3 -> 1.6251576e-3`|`2.6366733e-4 -> 8.4122617e-5`|`7.2367629e-4 -> 1.0930881e-4`|`4.3774677e-4 -> 1.4676436e-4`|`9.8124727e-5 -> 9.4007779e-5`|`9.4777192e-3 -> 2.9097240e-3`|`1.0632218e-3 -> 1.0716820e-3` ↑|`3.7894074e-5 -> 3.7803568e-5`|
|10|`5.3766814e-3 -> 2.1946956e-3`|`5.6697398e-4 -> 4.7683181e-4`|`7.7962351e-4 -> 1.4639807e-4`|`6.1275932e-4 -> 1.3835371e-4`|`1.1333948e-4 -> 1.0896498e-4`|`1.0191129e-2 -> 3.6725292e-3`|`1.9069448e-3 -> 1.9125157e-3` ↑|`4.4560094e-5 -> 4.4470351e-5`|
|11|`5.5237595e-3 -> 2.2490505e-3`|`3.2907810e-4 -> 1.6857730e-4`|`7.2147951e-4 -> 1.7648230e-4`|`5.2606252e-4 -> 1.4630600e-4`|`1.2907990e-4 -> 1.2436859e-4`|`1.0583131e-2 -> 4.0645194e-3`|`1.3894164e-3 -> 1.3983156e-3` ↑|`5.1274783e-5 -> 5.1184722e-5`|
|12|`5.8576453e-3 -> 2.6296170e-3`|`4.4751948e-4 -> 3.1094751e-4`|`6.5175000e-4 -> 1.9078862e-4`|`6.4203604e-4 -> 2.0019701e-4`|`1.4538162e-4 -> 1.4023239e-4`|`1.1206308e-2 -> 4.7757310e-3`|`1.5706436e-3 -> 1.5794164e-3` ↑|`5.7949744e-5 -> 5.7848977e-5`|
|13|`6.1939779e-3 -> 2.9122616e-3`|`3.6816032e-4 -> 1.9006820e-4`|`6.1710509e-4 -> 2.1092032e-4`|`6.0952469e-4 -> 2.3392760e-4`|`1.6183663e-4 -> 1.5628610e-4`|`1.1831587e-2 -> 5.2803586e-3`|`1.7612261e-3 -> 1.7700445e-3` ↑|`6.4612089e-5 -> 6.4521034e-5`|
|14|`6.6538144e-3 -> 3.3223021e-3`|`4.8433484e-4 -> 3.4537842e-4`|`6.5351035e-4 -> 1.8732315e-4`|`6.0557985e-4 -> 1.8343706e-4`|`1.7871279e-4 -> 1.7258453e-4`|`1.2698125e-2 -> 6.0490862e-3`|`1.9544135e-3 -> 1.9580193e-3` ↑|`7.1263501e-5 -> 7.1174796e-5`|
|15|`7.0775840e-3 -> 3.8105895e-3`|`5.9607542e-4 -> 4.8268215e-4`|`7.3535106e-4 -> 3.3582770e-4`|`5.8532823e-4 -> 1.9006878e-4`|`1.9550509e-4 -> 1.8926980e-4`|`1.3379794e-2 -> 6.7509256e-3`|`2.6559763e-3 -> 2.6635545e-3` ↑|`7.7854025e-5 -> 7.7762095e-5`|
|16|`7.4062796e-3 -> 4.0263805e-3`|`4.2547239e-4 -> 2.1733923e-4`|`7.3954176e-4 -> 2.4853454e-4`|`7.2899664e-4 -> 2.7578058e-4`|`2.1249326e-4 -> 2.0622816e-4`|`1.4099474e-2 -> 7.3373722e-3`|`2.3482437e-3 -> 2.3573445e-3` ↑|`8.4358231e-5 -> 8.4273233e-5`|
|17|`7.8467567e-3 -> 4.5378595e-3`|`6.4322295e-4 -> 4.8294906e-4`|`6.7796758e-4 -> 2.4154658e-4`|`6.6593221e-4 -> 2.6680642e-4`|`2.2949119e-4 -> 2.2309160e-4`|`1.4821866e-2 -> 8.1399036e-3`|`2.9924432e-3 -> 2.9597650e-3`|`9.0807942e-5 -> 9.0710876e-5`|
|18|`8.2630035e-3 -> 5.0675465e-3`|`7.4513876e-4 -> 6.2753679e-4`|`6.6649540e-4 -> 2.4791496e-4`|`6.3913250e-4 -> 2.4551978e-4`|`2.4678851e-4 -> 2.4026675e-4`|`1.5542393e-2 -> 9.0464160e-3`|`3.4011142e-3 -> 3.3998030e-3`|`9.7177156e-5 -> 9.7080486e-5`|
|19|`8.6518839e-3 -> 5.5046695e-3`|`6.6225640e-4 -> 5.1262023e-4`|`6.9555465e-4 -> 3.1398220e-4`|`7.0499534e-4 -> 3.6868879e-4`|`2.6389810e-4 -> 2.5754624e-4`|`1.6325891e-2 -> 9.9715340e-3`|`3.3652110e-3 -> 3.3684017e-3` ↑|`1.0352787e-4 -> 1.0342776e-4`|
|20|`9.0497568e-3 -> 6.0871945e-3`|`7.1017369e-4 -> 6.0778652e-4`|`6.4668032e-4 -> 3.2708149e-4`|`5.9426950e-4 -> 2.8381591e-4`|`2.8111143e-4 -> 2.7474244e-4`|`1.7082037e-2 -> 1.1117886e-2`|`3.4990037e-3 -> 3.5016353e-3` ↑|`1.0985402e-4 -> 1.0975996e-4`|
|21|`1.0037981e-2 -> 7.3646566e-3`|`1.3993342e-3 -> 1.3408810e-3`|`6.0946308e-4 -> 2.8923116e-4`|`5.8562488e-4 -> 3.1219826e-4`|`2.9845423e-4 -> 2.9211797e-4`|`1.8232280e-2 -> 1.2566375e-2`|`6.0277863e-3 -> 6.0305324e-3` ↑|`1.1612836e-4 -> 1.1603023e-4`|
|22|`1.0249266e-2 -> 7.5496962e-3`|`1.1540169e-3 -> 1.0779112e-3`|`5.7940424e-4 -> 3.2754063e-4`|`7.2896941e-4 -> 3.6599248e-4`|`3.1594862e-4 -> 3.0956409e-4`|`1.8864550e-2 -> 1.3253821e-2`|`5.5161029e-3 -> 5.5176770e-3` ↑|`1.2241864e-4 -> 1.2231805e-4`|
|23|`1.1078524e-2 -> 8.6137095e-3`|`1.6087938e-3 -> 1.5513770e-3`|`6.2116391e-4 -> 4.1214931e-4`|`7.2079030e-4 -> 4.8691503e-4`|`3.3332504e-4 -> 3.2700266e-4`|`1.9921569e-2 -> 1.4656619e-2`|`7.1194153e-3 -> 7.1214440e-3` ↑|`1.2864470e-4 -> 1.2855009e-4`|
|24|`1.0740602e-2 -> 8.0552048e-3`|`6.7042472e-4 -> 5.0341878e-4`|`6.9627061e-4 -> 4.0082704e-4`|`7.1959353e-4 -> 4.9286504e-4`|`3.5065794e-4 -> 3.4430764e-4`|`2.0226325e-2 -> 1.4825388e-2`|`4.3331818e-3 -> 4.3363033e-3` ↑|`1.3465300e-4 -> 1.3457354e-4`|
|25|`1.1248938e-2 -> 8.6430890e-3`|`8.4048752e-4 -> 6.8670042e-4`|`6.8432358e-4 -> 4.1349730e-4`|`7.8538541e-4 -> 4.9823188e-4`|`3.6834810e-4 -> 3.6200596e-4`|`2.1113623e-2 -> 1.5856749e-2`|`4.7920806e-3 -> 4.7891617e-3`|`1.4062357e-4 -> 1.4055471e-4`|
|26|`1.1584119e-2 -> 9.0741554e-3`|`7.9428197e-4 -> 6.4164771e-4`|`5.6025635e-4 -> 4.3065052e-4`|`8.2144482e-4 -> 5.1046903e-4`|`3.8569855e-4 -> 3.7953526e-4`|`2.1785705e-2 -> 1.6726347e-2`|`4.7808017e-3 -> 4.8144094e-3` ↑|`1.4666947e-4 -> 1.4660730e-4`|
|27|`1.2958941e-2 -> 1.0721591e-2`|`2.0165314e-3 -> 1.9505992e-3`|`6.1020724e-4 -> 5.3054111e-4`|`9.6774890e-4 -> 5.7130207e-4`|`4.0284867e-4 -> 3.9709203e-4`|`2.3015156e-2 -> 1.8188943e-2`|`8.9452004e-3 -> 8.9531857e-3` ↑|`1.5254885e-4 -> 1.5248666e-4`|
|28|`1.3055058e-2 -> 1.0883244e-2`|`1.7552845e-3 -> 1.6791115e-3`|`6.1903459e-4 -> 5.3475238e-4`|`9.2640907e-4 -> 5.9245192e-4`|`4.1980837e-4 -> 4.1455081e-4`|`2.3587663e-2 -> 1.9012568e-2`|`8.1345640e-3 -> 8.1315797e-3`|`1.5875933e-4 -> 1.5869750e-4`|
|29|`1.3134519e-2 -> 1.1024960e-2`|`1.5306551e-3 -> 1.4593243e-3`|`6.0381225e-4 -> 5.2293769e-4`|`8.6481669e-4 -> 6.2044580e-4`|`4.3653767e-4 -> 4.3173913e-4`|`2.3979749e-2 -> 1.9595142e-2`|`7.5823130e-3 -> 7.5789840e-3`|`1.6484493e-4 -> 1.6478075e-4`|
|30|`1.4241019e-2 -> 1.2397568e-2`|`2.2344510e-3 -> 2.1952963e-3`|`5.9619957e-4 -> 5.3021171e-4`|`8.3329943e-4 -> 5.6331494e-4`|`4.5333981e-4 -> 4.4895391e-4`|`2.5079362e-2 -> 2.1092449e-2`|`1.0256964e-2 -> 1.0254313e-2`|`1.7094137e-4 -> 1.7087804e-4`|

Direction summary: T/S/u/v/ssh and 0–100 m T improve on all 30 days;
100–1000 m T improves on 6 and worsens on 24; >1000 m T improves on 29
and worsens on day 1. The worst day-gap regression is +1.90% (day 2,
100–1000 m); all month-level headline fields improve.

## 7. Rule 12 per card

| card | resolved behavior | disposition |
|---|---|---|
| GYRE-zco | RK3 `now2` + face `avm` + live QCO metric | measured in §§4–6; operator bit-exact with NEMO inputs; no AT-BAR loss |
| LOCK_EXCHANGE-zco | constant vertical mixing; no TKE | 0 changed rows by construction |
| OVERFLOW-zps | constant vertical mixing; no TKE | 0 changed rows by construction |
| DINO NEMO MLF | already `nemo_face_native/nemo_face/live_face`; leap-frog arm remains strict | 0 changed rows: new branch is statically `now2` only |
| DINO Kamm | squared-centered/tpoint | face-metric branch not reached; 0 changed rows |
| legacy `nemo_v1` | no `now2` or live-face selector | 0 changed rows |
| ORCA2 | optional `now2`, no measured default card | **UNMEASURED-with-spec** |

## 8. ASKED / UNASKED

| choice | status |
|---|---|
| Select NEMO face-native shear, face `avm`, and live QCO metric on GYRE | **ASKED** (decision 36, user “Yes”) |
| Represent RK3's `Nbb,Nbb` call as explicit `nemo_face_native_now2` | **UNASKED — implementation mechanism**, offered for revert only with an equally fail-closed static representation |
| Key live Kbb eta on the variant, preserving a strict leap-frog arm | mechanical consequence of the compiled call; no physics choice |
| Add the two-step red/green reconciliation test and citation mappings | verification only |
| Shift older citation-gate anchors after the card's inserted lines | mechanical maintenance only; no scientific claim changes |

## 9. Open questions

- The upstream kt2 velocity gap leaves every model-selected sh2 cell bitwise
  unequal despite a `1.52e-18` maximum; it remains the first operand owner.
- P3's miss shows that the corrected closure statement is not the sole
  30-day T owner. The 100–1000 m regressions should be attributed before a
  month-scale follow-up, but they do not reverse an AT-BAR verdict here.
- ORCA2's opt-in `now2` selection remains UNMEASURED-with-spec.
