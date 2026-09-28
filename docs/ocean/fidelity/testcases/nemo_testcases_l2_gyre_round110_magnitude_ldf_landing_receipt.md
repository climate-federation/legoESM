# NEMO-testcases L2 GYRE round 110 receipt: magnitude-ranked stage-3 LDF landing

Date: 2026-09-18

Incoming tip: `51a4d088cf85ce1ea49b51b2ecf0ec294338e647`

Candidate commit: `5004884fb49445af4149ad822f2cba3f826d5d0c`

Round status: **LANDED — the compiled stage-3 lateral-diffusion content route
reduces day-30 temperature RMS by 179.914933 times and passes all five
Decision-43 conditions**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round110/`

## Outcome first

The magnitude-ranked candidate lands.  It changes no card, coefficient,
default, stabilizer, carried state, or restart schema.  It removes the private
diagnostic selector and makes the shared WS-RK3 implementation add the already
computed GM/Redi temperature and salinity rates to the stage-3 source tuple
whenever that card already selects GM/Redi.

The day-30 temperature RMS falls from
`1.2397011296352737e-2` K to `6.890484901489568e-5` K, a factor of
`179.9149330357401`.  The kt3 T maximum falls from
`1.627497246303733e-4` K to `8.600419718618468e-7` K.  First-over-bar remains
kt2 U/V, and no kt1 AT-BAR row leaves the bar.  The normal Decision-43 gate
prints

```text
STATUS PASS: moved_rows=53 day30_T=1.23970112963527369e-02->6.89048490148956762e-05
```

The `day30-no-improvement` plant prints

```text
STATUS PLANT-FIRED: day30-no-improvement
```

and exits 1.  Artifacts are `candidate/decision43_gate.json` and
`candidate/decision43_gate_plant.json`.

The complete candidate artifacts
`round110/candidate/{ladder.json,day_gap.json}` are now the immutable before
arm for round 111.  The 954-row extended artifact is
`round110/candidate/ladder_full.json`.

## Preregistration and prediction ledger

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round110.md` at commit
`587de6a8c9648f0e9df603252201cef57c545615`.  It predates every round-110
measurement and numerical edit.

| prediction | disposition | evidence |
|---|---|---|
| P1, owners-instrument calibration | **CONFIRMED scientifically; literal whole-document byte forecast REFUTED** | `--self-check` passed.  Step 1 is rounding-scale and the gap after step 2 is `4.2266e-6` K RMS.  The baseline day-30 scientific payload is identical to the immutable arm, including all five frozen RMS values.  The emitted JSON is not literally byte-for-byte because its mandatory worktree stamp names the new clean preregistration commit and root.  That provenance-only difference is retained here; it is not a state-field discrepancy. |
| P2, tip-local source route | **CONFIRMED for the registered numerical rows** | The clean pre-edit private `ldf_only` arm and post-edit native production arm have identical complete `content_vs_oracle` and `kt3_vs_oracle` dictionaries for T and S under production JIT.  Production eager and JIT dictionaries are identical.  Both source-injection rows are BIT.  The historical round-69 wrapper still says `REFUTED` because its frozen pre-round-85 targets are stale; that status is not relabeled. |
| P3, ladder forecast | **PARTLY CONFIRMED, numerically REFUTED** | The exact named 70-row before arm has the predicted 53 moved endpoint rows, unchanged kt2 headline maxima, and kt3 T exactly as predicted.  The 954-row instrument has 82, not 53, moved rows; four kt1 diagnostic rows move without a class change.  Chained kt3 S is `6.979441735666114e-8`, not the forecast `6.97944244620885e-8`.  First-over-bar and all row classes are unchanged. |
| P4, month forecast | **CONFIRMED** | `6.890484901489568e-5` K lies in the preregistered `1e-5`--`1e-3` K band and is strictly below the immutable before value. |
| P5, shared-card risk | **CONFIRMED AFTER ROUND-113 AMENDMENT** | Recipe-derived resolution finds two executing cards: GYRE-zco and the generic `build_nemo_gyre_recipe()` card.  Round 113 measures both; neither DINO card nor LOCK_EXCHANGE/OVERFLOW executes the statement.  The original Round-110 table omitted the generic card and is superseded below. |

P1's phrase “byte-for-byte” was over-constrained because the same document is
also required to carry a fresh fail-closed worktree stamp.  The mismatch is
therefore recorded as a failed literal prediction rather than hidden.  The
calibration quantities that controlled whether the owners instrument was
scientifically usable all reproduced exactly, so the magnitude measurement
continued.

## Compiled source and landed statement

The record-producing compiled program first clears tracer `Krhs` and
accumulates advection and surface sources at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868`.
For stage 3 it evaluates the shortwave source, calls `tra_ldf`, snapshots the
result, and only then calls `tra_zdf`, in that compiled order, at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:917-965`.

The resolved GYRE record selects the compiled isoneutral-Laplacian operator.
That operator reads the Kbb tracers at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:183-192`,
uses the Kmm metrics in its flux program at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:230-246`,
and adds the signed divergence into the same `Krhs` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:287-305`.
The following vertical solve forms its content RHS from the Kbb content plus
`p2dt` times the Kmm thickness times that accumulated `Krhs` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`.

The first magnitude-owning non-NEMO statement was therefore the missing
route, not a new diffusion formula: legoESM computed `dT_gm`/`dS_gm` but its
ordinary stage-3 WS source tuple omitted them before its implicit vertical
solve.  Commit `5004884fb` changes that one shared route to add the existing
rates when `_cfg_b.gm_redi is not None`; the downstream ZDF geometry and solve
are untouched.  The source file's line count is unchanged, so existing
compiled-source citation anchors did not move.

## Magnitude ranking

The owners instrument first reproduced round 62: kt2 entry after one step has
T RMS `2.0293e-15` K, while kt3 entry after two steps has T RMS
`4.226596281050612e-6` K.  At day 30, 77.1869% of squared T error is above
100 m and 50.9824% is in the west third.  The absolute peak is separately at
`k=11`, 208.504 m, 44.5031 N; it is not in the upper-100-m band.

The intervention gives the following one-table ranking.  “Carries” here is
the measured before-minus-after intervention effect, not an assumption that
RMS components add orthogonally.

| rank / owner | kt3 T carried or remaining | day-30 T RMS carried or remaining | where born | evidence row |
|---|---:|---:|---|---|
| 1. Missing stage-3 LDF content route | `1.6188968265851145e-4` K removed | `1.232810644733784e-2` K removed | First macroscopic T gap after step 2; baseline concentrates west/upper ocean | kt3 max `1.627497246303733e-4` -> `8.600419718618468e-7`; day 30 `1.2397011296352737e-2` -> `6.890484901489568e-5` |
| 2. Residual, operator not yet assigned | `8.600419718618468e-7` K remains | `6.890484901489568e-5` K remains | Still born after step 2: `8.80218929304543e-8` K RMS; by day 30, 54.0502% lies at 100--1000 m, 54.9790% in the west third, and 83.9340% in the wind band | candidate `step60/step_gap.json`, `decompose_day30.json`; peak `k=7`, 91.420 m, 26.9779 N |

The spatial cuts overlap and are localization evidence, not separate causal
owners.  The next-largest measured same-stage discriminator is the preserved
round-70 paired FCT/advection-content arm: locally it reduces kt3 T maximum
from `8.600420500215478e-7` to `5.760972143775689e-8`.  It is not yet a
production/source-exact candidate and does not land in this round.

## Local production proof

The pre-edit evidence is `local_before/private_pair_jit.json`, stamped at the
clean preregistration commit.  Its `ldf_only` T/S content maxima are
`5.743498263655056e-5` / `7.387909136014059e-6`; its kt3 T/S maxima are
`8.600420500215478e-7` / `6.979443156751586e-8`; both selected source rows
are BIT.

The post-edit evidence is `local_after/native_jit.json` and
`local_after/native_eager.json`, both stamped at clean candidate commit
`5004884fb49445af4149ad822f2cba3f826d5d0c`.  For each tracer, the entire
post-edit `content_vs_oracle` and `kt3_vs_oracle` dictionaries equal the
same-tip pre-edit private `ldf_only` dictionaries.  JIT and eager agree on
those dictionaries.  T's separate host-reconstruction diagnostic remains
three words at `2.842170943040401e-14`; that already-retracted host
association is neither used nor claimed as exact.

The local wrapper's top-level historical verdict remains `REFUTED` because it
compares against obsolete round-69 absolute metrics.  This receipt claims
only the preregistered same-tip dictionary equality above.  Its built-in
one-ULP content scorer changes exactly one cell.  The actual landing admission
has the independent nonzero-exit plant quoted in the outcome section.

## Decision-43 trajectory admission

The landing's own-base arm is
`phase3/round110/before_same_tip_51a4d088c/{ladder.json,day_gap.json}`.  The
operator measured it at the exact incoming commit and found it bit-identical
to `phase3/merge_main_2026-09-17/after2/` in all 50 ladder rows, 210 arrays,
and all 30 daily states.  Thus the 179.914933 factor is a one-variable
before/after result, not a cross-tip inference.  The exact named 70-row
comparison registers 53 moved endpoint rows.  The extended 954-row comparison
against the matching preserved
extended before instrument registers 82 moved rows, 59 with at least one
greater-than-two-row-ULP worsening.  That old Rule-12 comparison therefore
fails, as Decision 43 explicitly permits for kt >= 2; this receipt does not
mislabel it as a Rule-12 pass.

| required headline | immutable before | candidate after | Decision-43 result |
|---|---:|---:|---|
| kt2 T max | `1.4210854715202004e-14` | `1.4210854715202004e-14` | unchanged, AT-BAR |
| kt2 S max | `2.1316282072803006e-14` | `2.1316282072803006e-14` | unchanged, AT-BAR |
| kt2 U max | `2.7377110452773967e-12` | `2.7377110452773967e-12` | unchanged, DEBT |
| kt2 V max | `3.2849219221489645e-12` | `3.2849219221489645e-12` | unchanged, DEBT |
| kt3 T max | `1.627497246303733e-4` | `8.600419718618468e-7` | improved 189.235 times |
| kt3 S max | `6.327735185607253e-6` | `6.979441735666114e-8` | improved 90.6625 times |
| day-30 T RMS | `1.2397011296352737e-2` | `6.890484901489568e-5` | improved 179.914933 times |

The 30-day after values for the other required state fields are: S
`1.1781247458573714e-5`, u `5.579572380229224e-6`, v
`4.620276844589294e-6`, and ssh `6.802771093693658e-6` in their native units.
The daily T series is not monotone: it reaches `5.2839e-4` K on day 23 before
returning to the day-30 value.  The operator's field/level audit identifies
this as a physics trajectory, not an instrument artifact: T, S, u, and v
co-spike in the upper 0--1000 m while ssh remains on trend.  The spike remains
registered open behavior and is not hidden by the endpoint.

All 82 extended moved rows follow.  “Improved” and “worsened” are cell counts
against the common oracle; `>2-ULP` is the count that violates the superseded
Rule-12 per-cell limit.  The four kt1 rows are first; both AT-BAR rows remain
AT-BAR.

+Error while loading conda entry point: conda-anaconda-tos (cannot import name 'validate_prefix_exists' from 'conda.cli.install' (/home/dbalwada/miniconda3/lib/python3.13/site-packages/conda/cli/install.py))
| row | before max / unequal / class | after max / unequal / class | max model move | improved / worsened / >2-ULP |
|---|---:|---:|---:|---:|
| `GYRE-zco.kt1.stage3.u` | `2.7377110452773967e-12` / 17,398 / DEBT | `2.7377110452773967e-12` / 17,399 / DEBT | `6.2450045135165055e-17` | 1,261 / 1,237 / 0 |
| `GYRE-zco.kt1.stage3.v` | `3.284922138989399e-12` / 17,100 / DEBT | `3.2849219221489645e-12` / 17,100 / DEBT | `9.0205620750793969e-17` | 1,076 / 1,046 / 0 |
| `GYRE-zco.kt1.zdf_entry.avm` | `2.7755575615628914e-17` / 4,924 / AT-BAR | `2.7755575615628914e-17` / 4,783 / AT-BAR | `2.7755575615628914e-17` | 229 / 124 / 0 |
| `GYRE-zco.kt1.zdf_entry.avt` | `1.7347234759768071e-18` / 4,905 / AT-BAR | `2.6020852139652106e-18` / 4,764 / AT-BAR | `1.7347234759768071e-18` | 229 / 122 / 0 |
| `GYRE-zco.kt10.after.uu_b` | `8.7336337465434427e-08` / 580 / DEBT | `8.9948433665797134e-08` / 580 / DEBT | `3.8616865760798661e-08` | 345 / 235 / 235 |
| `GYRE-zco.kt10.after.vv_b` | `1.9001900318678377e-07` / 570 / DEBT | `1.8475276181931927e-07` / 570 / DEBT | `5.3474128593964656e-08` | 251 / 319 / 319 |
| `GYRE-zco.kt10.before.S` | `0.00026118831114274599` / 17,895 / DEBT | `1.2955953181403856e-06` / 17,775 / DEBT | `0.00026109280796049461` | 15,028 / 2,781 / 2,518 |
| `GYRE-zco.kt10.before.T` | `0.0077397267823400284` / 18,000 / DEBT | `8.9813185475406954e-06` / 17,999 / DEBT | `0.007736948175733005` | 15,287 / 2,713 / 2,713 |
| `GYRE-zco.kt10.before.ssh` | `1.8935128029820558e-05` / 600 / DEBT | `6.6429194487890864e-07` / 600 / DEBT | `1.8987219013343742e-05` | 437 / 163 / 163 |
| `GYRE-zco.kt10.before.u` | `0.0015136670604975304` / 17,400 / DEBT | `2.8633490999089607e-05` / 17,400 / DEBT | `0.0015064928225924056` | 12,663 / 4,737 / 4,737 |
| `GYRE-zco.kt10.before.v` | `0.00058258010362488045` / 17,100 / DEBT | `3.2166251382306603e-05` / 17,100 / DEBT | `0.00059033626370017141` | 11,978 / 5,122 / 5,122 |
| `GYRE-zco.kt2.after.uu_b` | `5.0398350619879594e-08` / 580 / DEBT | `5.0398350609254412e-08` / 580 / DEBT | `1.8973538018496328e-17` | 309 / 270 / 0 |
| `GYRE-zco.kt2.after.vv_b` | `4.8006349786253524e-08` / 570 / DEBT | `4.8006349783109338e-08` / 570 / DEBT | `1.588356182691264e-17` | 293 / 276 / 0 |
| `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.S` | `1.0800249583553523e-12` / 14,641 / DEBT | `1.0800249583553523e-12` / 14,630 / DEBT | `2.1316282072803006e-14` | 88 / 92 / 2 |
| `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.T` | `8.2174267390655586e-12` / 17,061 / DEBT | `8.2174267390655586e-12` / 17,060 / DEBT | `1.0658141036401503e-14` | 107 / 98 / 2 |
| `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.u` | `3.501291214201346e-08` / 17,400 / DEBT | `3.501291214201346e-08` / 17,400 / DEBT | `9.7144514654701197e-17` | 1,276 / 1,442 / 0 |
| `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.v` | `2.9515900715128707e-08` / 17,100 / DEBT | `2.9515900715128707e-08` / 17,100 / DEBT | `6.2450045135165055e-17` | 1,223 / 1,107 / 0 |
| `GYRE-zco.kt2.arm.omit_freshwater_forcing.S` | `0.0018528642353103919` / 17,790 / DEBT | `0.0018528642353103919` / 17,790 / DEBT | `2.1316282072803006e-14` | 99 / 74 / 0 |
| `GYRE-zco.kt2.arm.omit_freshwater_forcing.T` | `0.0011744889249065693` / 17,995 / DEBT | `0.0011744889249065693` / 17,995 / DEBT | `1.0658141036401503e-14` | 101 / 84 / 0 |
| `GYRE-zco.kt2.arm.omit_freshwater_forcing.u` | `9.7241693073554791e-06` / 17,400 / DEBT | `9.7241693073554791e-06` / 17,400 / DEBT | `9.0205620750793969e-17` | 1,359 / 1,174 / 0 |
| `GYRE-zco.kt2.arm.omit_freshwater_forcing.v` | `5.1184108874613443e-06` / 17,100 / DEBT | `5.1184108874613443e-06` / 17,100 / DEBT | `9.0205620750793969e-17` | 1,205 / 1,049 / 0 |
| `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.S` | `2.1316282072803006e-14` / 6,457 / AT-BAR | `2.1316282072803006e-14` / 6,438 / AT-BAR | `1.4210854715202004e-14` | 97 / 65 / 0 |
| `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.T` | `1.4210854715202004e-14` / 6,858 / AT-BAR | `1.4210854715202004e-14` / 6,831 / AT-BAR | `1.0658141036401503e-14` | 100 / 64 / 0 |
| `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.u` | `2.7377110452773967e-12` / 17,398 / DEBT | `2.7377110452773967e-12` / 17,399 / DEBT | `6.2450045135165055e-17` | 1,261 / 1,237 / 0 |
| `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.v` | `3.284922138989399e-12` / 17,100 / DEBT | `3.2849219221489645e-12` / 17,100 / DEBT | `9.0205620750793969e-17` | 1,076 / 1,046 / 0 |
| `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.S` | `0.10368337147600215` / 17,942 / DEBT | `0.10368337147600215` / 17,942 / DEBT | `2.1316282072803006e-14` | 89 / 87 / 0 |
| `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.T` | `0.06572405954187488` / 17,999 / DEBT | `0.06572405954187488` / 17,999 / DEBT | `1.0658141036401503e-14` | 98 / 85 / 3 |
| `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.u` | `0.0011939011497228095` / 17,400 / DEBT | `0.0011939011497228095` / 17,400 / DEBT | `6.2450045135165055e-17` | 237 / 249 / 0 |
| `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.v` | `0.0011670177562268253` / 17,100 / DEBT | `0.0011670177562268253` / 17,100 / DEBT | `9.7144514654701197e-17` | 218 / 245 / 0 |
| `GYRE-zco.kt2.before.S` | `2.1316282072803006e-14` / 6,457 / AT-BAR | `2.1316282072803006e-14` / 6,438 / AT-BAR | `1.4210854715202004e-14` | 97 / 65 / 0 |
| `GYRE-zco.kt2.before.T` | `1.4210854715202004e-14` / 6,858 / AT-BAR | `1.4210854715202004e-14` / 6,831 / AT-BAR | `1.0658141036401503e-14` | 100 / 64 / 0 |
| `GYRE-zco.kt2.before.u` | `2.7377110452773967e-12` / 17,398 / DEBT | `2.7377110452773967e-12` / 17,399 / DEBT | `6.2450045135165055e-17` | 1,261 / 1,237 / 0 |
| `GYRE-zco.kt2.before.v` | `3.284922138989399e-12` / 17,100 / DEBT | `3.2849219221489645e-12` / 17,100 / DEBT | `9.0205620750793969e-17` | 1,076 / 1,046 / 0 |
| `GYRE-zco.kt3.after.uu_b` | `1.178525769766816e-07` / 580 / DEBT | `1.1804872147933975e-07` / 580 / DEBT | `5.2205882737459131e-10` | 269 / 311 / 311 |
| `GYRE-zco.kt3.after.vv_b` | `7.089790817762457e-08` / 570 / DEBT | `7.0736399770065204e-08` / 570 / DEBT | `5.3702325778556131e-10` | 250 / 320 / 320 |
| `GYRE-zco.kt3.before.S` | `6.3277351856072528e-06` / 17,378 / DEBT | `6.9794417356661143e-08` / 17,380 / DEBT | `6.3265926044664411e-06` | 7,812 / 6,233 / 5,824 |
| `GYRE-zco.kt3.before.T` | `0.00016274972463037329` / 18,000 / DEBT | `8.6004197186184683e-07` / 18,000 / DEBT | `0.0001626617136771813` | 9,665 / 8,320 / 8,240 |
| `GYRE-zco.kt3.before.ssh` | `7.0725589820016393e-07` / 600 / DEBT | `7.072558983222451e-07` / 600 / DEBT | `1.6147239795261115e-14` | 368 / 232 / 14 |
| `GYRE-zco.kt3.before.u` | `1.2295527572661613e-05` / 17,400 / DEBT | `1.2295527576644538e-05` / 17,400 / DEBT | `7.5564554613549717e-15` | 8,881 / 8,418 / 212 |
| `GYRE-zco.kt3.before.v` | `2.3465017240615477e-05` / 17,100 / DEBT | `2.3465017237225827e-05` / 17,100 / DEBT | `8.7291285311152933e-15` | 8,165 / 8,708 / 211 |
| `GYRE-zco.kt4.after.uu_b` | `1.4040031106644923e-07` / 580 / DEBT | `1.4078950131722265e-07` / 580 / DEBT | `1.1519671930769471e-09` | 256 / 324 / 324 |
| `GYRE-zco.kt4.after.vv_b` | `9.4137505911892454e-08` / 570 / DEBT | `9.3848314050896074e-08` / 570 / DEBT | `1.2803273753692057e-09` | 299 / 271 / 271 |
| `GYRE-zco.kt4.before.S` | `1.48574231531029e-05` / 17,669 / DEBT | `4.7252278534415382e-07` / 17,607 / DEBT | `1.4809267824489325e-05` | 12,029 / 5,129 / 4,452 |
| `GYRE-zco.kt4.before.T` | `0.00037880763850139942` / 17,999 / DEBT | `5.8269268166100119e-06` / 17,999 / DEBT | `0.00037764030215114985` | 12,746 / 5,253 / 5,238 |
| `GYRE-zco.kt4.before.ssh` | `4.577542667915345e-07` / 600 / DEBT | `4.6933513034698796e-07` / 600 / DEBT | `1.0319142198459558e-07` | 327 / 273 / 273 |
| `GYRE-zco.kt4.before.u` | `2.0557060249691561e-05` / 17,400 / DEBT | `2.0552956758734808e-05` / 17,400 / DEBT | `3.0702527805352015e-06` | 8,951 / 8,449 / 8,449 |
| `GYRE-zco.kt4.before.v` | `1.8319409220590721e-05` / 17,100 / DEBT | `1.7604221222392025e-05` / 17,100 / DEBT | `4.5753190741903982e-06` | 8,755 / 8,345 / 8,345 |
| `GYRE-zco.kt5.after.uu_b` | `2.9752690469843487e-07` / 580 / DEBT | `1.492776845594436e-07` / 580 / DEBT | `2.9280480927345437e-07` | 300 / 280 / 280 |
| `GYRE-zco.kt5.after.vv_b` | `2.1959943558032106e-07` / 570 / DEBT | `7.9384012524521402e-08` / 570 / DEBT | `2.1048770321438817e-07` | 264 / 306 / 306 |
| `GYRE-zco.kt5.before.S` | `0.0025487148408984694` / 17,748 / DEBT | `4.2859174698151037e-07` / 17,702 / DEBT | `0.002548712412668408` | 12,682 / 4,778 / 4,185 |
| `GYRE-zco.kt5.before.T` | `0.057537968257086902` / 18,000 / DEBT | `7.5037487796691948e-06` / 18,000 / DEBT | `0.057537701686197096` | 13,177 / 4,823 / 4,821 |
| `GYRE-zco.kt5.before.ssh` | `7.7419854865911145e-07` / 600 / DEBT | `3.688991887561624e-07` / 600 / DEBT | `8.1852641625377275e-07` | 312 / 288 / 288 |
| `GYRE-zco.kt5.before.u` | `0.0091535810057503421` / 17,400 / DEBT | `3.7099124175329163e-05` / 17,400 / DEBT | `0.0091640821478057924` | 9,220 / 8,180 / 8,180 |
| `GYRE-zco.kt5.before.v` | `0.038977704462847096` / 17,100 / DEBT | `1.8740728250601912e-05` / 17,100 / DEBT | `0.03897198280478506` | 9,078 / 8,022 / 8,022 |
| `GYRE-zco.kt6.after.uu_b` | `2.1650266523456311e-07` / 580 / DEBT | `1.2779222405279178e-07` / 580 / DEBT | `2.0327607879883091e-07` | 317 / 263 / 263 |
| `GYRE-zco.kt6.after.vv_b` | `1.3081666796051173e-07` / 570 / DEBT | `1.0840287174656259e-07` / 570 / DEBT | `1.2811341528661197e-07` | 283 / 287 / 287 |
| `GYRE-zco.kt6.before.S` | `0.00037495788822639042` / 17,789 / DEBT | `9.2775731985739185e-07` / 17,749 / DEBT | `0.00037496991571828175` | 13,303 / 4,314 / 3,863 |
| `GYRE-zco.kt6.before.T` | `0.010066457774108528` / 18,000 / DEBT | `1.1337056005089607e-05` / 18,000 / DEBT | `0.010066541800341611` | 13,734 / 4,266 / 4,265 |
| `GYRE-zco.kt6.before.ssh` | `3.2036937495085946e-06` / 600 / DEBT | `5.5872756314677766e-07` / 600 / DEBT | `3.0925782928086626e-06` | 409 / 191 / 191 |
| `GYRE-zco.kt6.before.u` | `0.0035817005120260779` / 17,400 / DEBT | `4.1657379143527403e-05` / 17,400 / DEBT | `0.0035722317368089701` | 10,361 / 7,039 / 7,039 |
| `GYRE-zco.kt6.before.v` | `0.012604558083863959` / 17,100 / DEBT | `3.5678782491713711e-05` / 17,100 / DEBT | `0.012620542226240622` | 9,621 / 7,479 / 7,479 |
| `GYRE-zco.kt7.after.uu_b` | `1.9830559538162851e-07` / 580 / DEBT | `1.069594848758168e-07` / 580 / DEBT | `1.7944533941546348e-07` | 253 / 327 / 327 |
| `GYRE-zco.kt7.after.vv_b` | `1.4977735369276976e-07` / 570 / DEBT | `1.4349374571551166e-07` / 570 / DEBT | `9.5211222473517466e-08` | 281 / 289 / 289 |
| `GYRE-zco.kt7.before.S` | `0.0013330682220242807` / 17,829 / DEBT | `7.537604886920235e-07` / 17,772 / DEBT | `0.0013332084358879115` | 13,860 / 3,822 / 3,424 |
| `GYRE-zco.kt7.before.T` | `0.031855998174336264` / 17,999 / DEBT | `1.1442120872118267e-05` / 17,999 / DEBT | `0.031859709106203837` | 14,189 / 3,811 / 3,810 |
| `GYRE-zco.kt7.before.ssh` | `6.7956931858242572e-06` / 600 / DEBT | `8.827015061667505e-07` / 600 / DEBT | `6.876574823058354e-06` | 406 / 194 / 194 |
| `GYRE-zco.kt7.before.u` | `0.0078155829836336378` / 17,400 / DEBT | `3.0582514300339647e-05` / 17,400 / DEBT | `0.0078144640847852935` | 11,926 / 5,474 / 5,474 |
| `GYRE-zco.kt7.before.v` | `0.0042142807016476158` / 17,100 / DEBT | `2.1825902969774529e-05` / 17,100 / DEBT | `0.0042210284481761923` | 11,529 / 5,571 / 5,571 |
| `GYRE-zco.kt8.after.uu_b` | `1.6195036391092711e-07` / 580 / DEBT | `1.336788745710674e-07` / 580 / DEBT | `1.1690827206474724e-07` | 316 / 264 / 264 |
| `GYRE-zco.kt8.after.vv_b` | `1.4934223134142541e-07` / 570 / DEBT | `1.4235700907189522e-07` / 570 / DEBT | `8.4215915249893566e-08` | 282 / 288 / 288 |
| `GYRE-zco.kt8.before.S` | `0.00026539122494284584` / 17,883 / DEBT | `1.0231654883341434e-06` / 17,775 / DEBT | `0.00026543671366852095` | 14,487 / 3,251 / 2,895 |
| `GYRE-zco.kt8.before.T` | `0.0073109445962309394` / 18,000 / DEBT | `9.4578708313974857e-06` / 18,000 / DEBT | `0.0073124282598300283` | 14,883 / 3,117 / 3,116 |
| `GYRE-zco.kt8.before.ssh` | `1.0411528457225996e-05` / 600 / DEBT | `6.3984443173653593e-07` / 600 / DEBT | `1.1051372888962532e-05` | 417 / 183 / 183 |
| `GYRE-zco.kt8.before.u` | `0.0037893641214670377` / 17,400 / DEBT | `4.1569950919390086e-05` / 17,400 / DEBT | `0.0037936121303709554` | 12,738 / 4,662 / 4,662 |
| `GYRE-zco.kt8.before.v` | `0.0024124980342377228` / 17,100 / DEBT | `3.5962302026254327e-05` / 17,100 / DEBT | `0.0024299388598170157` | 12,176 / 4,924 / 4,924 |
| `GYRE-zco.kt9.after.uu_b` | `1.1302896615634096e-07` / 580 / DEBT | `9.8408898053028993e-08` / 580 / DEBT | `6.282286972090329e-08` | 332 / 248 / 248 |
| `GYRE-zco.kt9.after.vv_b` | `2.324981445752522e-07` / 570 / DEBT | `2.252500010148633e-07` / 570 / DEBT | `6.6733981252684385e-08` | 251 / 319 / 319 |
| `GYRE-zco.kt9.before.S` | `0.00028746065442675217` / 17,883 / DEBT | `8.6036124713473328e-07` / 17,784 / DEBT | `0.00028740233192081632` | 14,634 / 3,148 / 2,869 |
| `GYRE-zco.kt9.before.T` | `0.0085224492466018376` / 18,000 / DEBT | `7.0207171880554142e-06` / 18,000 / DEBT | `0.0085207644815312733` | 14,947 / 3,053 / 3,053 |
| `GYRE-zco.kt9.before.ssh` | `1.4383345478020948e-05` / 600 / DEBT | `9.0001235190267145e-07` / 600 / DEBT | `1.4320103249965827e-05` | 441 / 159 / 159 |
| `GYRE-zco.kt9.before.u` | `0.0020269816331771329` / 17,400 / DEBT | `2.9207515902711454e-05` / 17,400 / DEBT | `0.0020189026907110434` | 12,899 / 4,501 / 4,501 |
| `GYRE-zco.kt9.before.v` | `0.0015707397392859292` / 17,100 / DEBT | `3.3940538973480962e-05` / 17,100 / DEBT | `0.0015751977745332279` | 11,600 / 5,500 / 5,500 |

## Other cards and scope — Round-113 amendment

The original Round-110 table hard-coded five cards and omitted the certified
generic NEMO-GYRE recipe.  It is superseded by the recipe-derived census from
the amended Decision-43 gate:

| card | tracer integrator / GM-Redi condition | executes this route | measured consequence |
|---|---|---:|---|
| GYRE-zco | `rk3_ws`, GM/Redi configured | yes | full ladder and 30 days above |
| LOCK_EXCHANGE-zco | `rk3_ws`, no GM/Redi | no | condition false; no statement bit can move |
| OVERFLOW-zps | `rk3_ws`, no GM/Redi | no | condition false; no statement bit can move |
| `build_nemo_gyre_recipe()` | `rk3_ws`, GM/Redi configured | yes | measured at exact base `51a4d088c` and descendant `0e7c8c9e4`; all certified assertions pass on both arms; exact moves registered below |
| DINO `nemo_dino_kamm` | Euler tracer lane, GM/Redi configured | no | route is outside executed lane |
| DINO `nemo_dino_kamm_mlf` | Euler tracer lane, GM/Redi configured | no | route is outside executed lane |

The generic card uses its existing certified deterministic three-step forced
loop.  Its base and descendant artifacts are
`phase3/round113/generic_card/{before_51a4d088c,after_0e7c8c9e4}.{json,npz}`;
`comparison.json` registers all 15 field/step rows.  Both arms pass the same
five assertions: all fields finite, bounded u/eta, and non-vacuous thermal and
wind forcing.  Fourteen rows move:

| row | unequal cells | maximum absolute move |
|---|---:|---:|
| step1 S | 12,370 | `1.22690245518697338e-7` |
| step1 T | 17,831 | `1.29111745792442889e-5` |
| step1 eta | 0 | `0` |
| step1 u | 360 | `2.16840434497100887e-19` |
| step1 v | 311 | `2.16840434497100887e-19` |
| step2 S | 14,591 | `1.80391687365499820e-7` |
| step2 T | 17,957 | `2.46658336102711928e-5` |
| step2 eta | 600 | `4.87545617588840130e-8` |
| step2 u | 17,400 | `1.32275851105756459e-8` |
| step2 v | 17,100 | `2.98704303375307845e-8` |
| step3 S | 16,439 | `7.84909794049326592e-7` |
| step3 T | 17,998 | `3.82807321734901507e-5` |
| step3 eta | 600 | `1.52569185909025122e-7` |
| step3 u | 17,400 | `8.49594800067521305e-8` |
| step3 v | 17,100 | `7.47595064742867521e-8` |

The Round-113 preregistration correctly predicted the step-1 tracer movement
and step-1 eta identity, but its prediction that step-1 u/v would be unchanged
is **REFUTED** by 360/311 last-bit cells at `2.1684e-19`.  No certified bound
worsens, so no re-baselining decision is needed.  Re-running Decision 43 with
this card named as measured and with the explicit 53-row registry passes; a
missing-registry-row plant exits 1 and prints `STATUS PLANT-FIRED`.

The real-card control now observes the routed source through two full
production steps (zero versus `0.125` GM/Redi sentinel), rather than inspecting
source text.  Disabling the production guard makes that test fail with zero
routed cells.  Replacing the exact registry predicate by the old
`len(moved)>0` predicate likewise makes the missing-row test fail.  The mutant
logs are under `phase3/round113/nonvacuity/`.

Thus DINO does not share the exact landed statement and Decision 43 does not
require a before/after DINO numerical run.  The recipe-derived test still
fails closed if either DINO card changes to the executing WS+GM/Redi lane;
merely setting `dino_measurement_required=true` cannot satisfy admission.

ORCA2 is **UNMEASURED-WITH-SPEC**.  Resolve its tracer time integrator and
GM/Redi selection, then, from identical fp64 inputs, compare stage-3
post-QSR/LDF `Krhs`, pre/post-ZDF T/S, and the next consumed state before and
after this route.  No ORCA2 fidelity verdict is claimed.

## Review, citations, and tests

The separate read-only review command was attempted after the candidate and
gate commits.  It produced no scientific verdict.  Its complete verdict is
quoted verbatim:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Therefore: **independent review unavailable in-sandbox**.  It did not mark the
diff `DO NOT SHIP`.

Citation-gate, shifted-citation-plant, focused-test, and full-tree summaries
are recorded below after their final runs:

The compiled-source citation gate finds six citations, zero unmapped
citations, zero failures, and zero failing map entries: `status=PASS`.  Its
rigid `+2` shifted plant on the compiled 827--868 range reports
`SYMBOL-NOT-AT-LINE` and exits 1.  The first unplanted run also caught and
refused an off-by-one endpoint; the corrected map pins the repeated loop
opener to occurrence 1.  Removing the private selector invalidated one
historical current-tree source anchor, so the landed source keeps that old
text only in an explicit `is ... retired` comment; the field itself remains
absent.

The focused final command reports exactly:

```text
204 passed, 9 warnings in 128.51s (0:02:08)
```

It covers the Decision-43 gate, the complete phase-3 gate tests, the round-67
LDF local gate, the receipt citation gate, DINO experiment tests, and the
NEMO-testcase recipe.  The earlier candidate-focused source test reports
`9 passed in 0.98s`.  Decision-gate development runs reported
`3 passed in 8.09s`, then `4 passed in 8.39s` after the shared-DINO fail-closed
case was added.

The required monolithic twelve-worker command over both complete trees was
run once.  Its terminal summary must be quoted rather than presented as a
valid regression verdict:

```text
210 failed, 5769 passed, 112 skipped, 2 xfailed, 55 warnings, 25 errors in 861.01s (0:14:21)
```

It exited 3 at 74% after five JAX compiler aborts, worker replacement, and a
controller `MemoryError`.  The aborted tests include unrelated advection,
MPAS-QCO, leapfrog, partial-cell, and RK3-WS files.  This is the repository's
documented large-suite per-process compiler limit, not a usable 210-failure
scientific result.

Fresh-process splits and isolated reruns were therefore used, as required by
the house rule for this failure mode.  Every completed suite summary is kept
here; interrupted suites are named explicitly and have no invented summary:

| suite | literal terminal summary / disposition |
|---|---|
| first 50 fidelity files | `604 passed, 5 skipped in 177.22s (0:02:57)` |
| next fidelity files 51--75 | `215 passed in 16.63s` |
| fidelity files 76--80 | `17 passed in 2.54s` |
| fidelity files 81--85 | `43 passed in 13.43s` |
| fidelity files 86--90 | `33 passed in 30.33s` |
| fidelity files 91--95 | `50 passed in 243.84s (0:04:03)` |
| original 50-file fidelity chunk 2 | interrupted at 85% after prolonged accumulated compiler state; superseded for files 51--95 by the rows above |
| fidelity files 96--100 | interrupted in a redundant slow phase-3 integration subgroup; those files had already run before the monolithic controller failed |
| first 50 unit files | `11 failed, 766 passed, 1 skipped, 2 xfailed, 5 warnings in 615.29s (0:10:15)` |
| second 50 unit files | agent-interrupted immediately after 75 passes so exact new-ID reruns could take priority; no summary claimed |
| unexpected periodic-FCT ID, fresh process | `1 passed in 4.20s` |
| four unexpected CATKE IDs, fresh process | `4 passed in 5.41s` |
| candidate-sensitive GM/Redi, recipe and RK3-WS files | `2 failed, 185 passed, 1 warning in 282.10s (0:04:42)` |
| same two MPAS failures at incoming `51a4d088c` | `2 failed in 2.50s` |
| initial new-gate stamp audit | `1 failed in 1.61s`; correctly exposed the new unstamped emitter |
| final Decision-43 plus stamp audit | `1 failed, 4 passed in 9.90s`; the sole failure is the known nine-offender aggregate |
| incoming-tip stamp audit | `1 failed in 1.62s`; its nine-offender dictionary is byte-identical to the final candidate's |

In the usable first-unit-chunk report, six failures are in the frozen 87-ID
list.  The five IDs outside it were one periodic-FCT test and four CATKE tests;
all five pass in fresh processes.  The candidate-sensitive run's two failures
are MPAS barotropic scan float64/float32 carry mismatches.  Both fail with the
same traceback at incoming tip `51a4d088c`, and MPAS cannot execute the
changed lat-lon statement.  The lat-lon GM/Redi tests, recipe resolution,
round-67 route proof, and all RK3-WS tests pass.

The worktree-stamp guard initially found ten offenders because the new gate
omitted its own stamp.  Commit `88806088e` adds the fail-closed stamp.  The
final candidate and incoming-tip assertion lines are byte-identical and list
the same nine known offenders, so this round adds none.

The complete-tree controller limitation means this receipt does **not** claim
an all-green 8,167-test run.  It does claim, with isolated evidence, no new
failing ID on the changed execution path and no growth in the known stamp
offender set.  Logs are `full_ocean_tests.log`, `full_ocean_split*.log`,
`full_unit_split_tests.log`, `rerun_new_*.log`,
`candidate_sensitive_unit_tests.log`, and `decision43_stamp_tests.log`.

## ASKED / UNASKED

ASKED and completed: magnitude ranking with the round-62 calibration; compiled
source walk; same-tip eager and production-JIT local proof; complete kt1--10
ladder; days 1--30 score; every moved-row registry; shared-card resolution;
Decision-43 admission and failing plant; citation control; review attempt; and
the full ocean fidelity/unit suites.

UNASKED and not done: no NEMO source was modified; `makenemo` and `mpirun`
were not run; no oracle record was acquired; no public configuration or
carried state changed; the year harness, reconciliation gate, freshwater pair,
and #1484 guard were not touched; no held patch from another stage was
bundled.

## OPEN for round 111

1. Use the new immutable before arm
   `phase3/round110/candidate/{ladder.json,day_gap.json}` and rerun the residual
   magnitude ranking before selecting a candidate.
2. Re-prove the same-stage FCT/advection-content discriminator at the new tip,
   isolate its compiled production statement, and promote it only if it is
   source-exact under production JIT.  Its preserved local kt3 T endpoint is
   `5.760972143775689e-8`; it has no month verdict yet.
3. Localize the candidate's day-23 T spike before treating day 30 as a smooth
   growth law.
4. Keep ORCA2 UNMEASURED until the explicit spec above is run.  The remaining
   GYRE residual is not zero and no complete-identity claim is made.
