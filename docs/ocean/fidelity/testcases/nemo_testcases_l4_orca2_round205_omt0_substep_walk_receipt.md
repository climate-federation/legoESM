# ORCA2 round 205 — OMT-0 split-explicit statement walk

Date: 2026-10-09. Base: `c46995c4c`. Measurement tip: `9bc7892d6`.
Status: **HELD**. All numbers in this receipt are labelled **independent
OMT-0**. The shipped rung-10 ORCA2 card and its sea-ice selectors and
`unmeasured_features` are unchanged.

## Verdict

The admitted round-203 barotropic streams are sufficient: both the 65-row
`BTSUB_2` stream and the two-row `BTORD_2` stream parse to EOF and twins A/B
are exact on every defined payload byte. The pure traced replay is bit-exact
with the untraced replay in terminal SSH, barotropic U/V, and both transport
means. All five independent plants fire.

The literal first non-bit statement is the continuity forcing sign. NEMO
forms `sshe_rhs = r1_rho0 * emp` and preserves positive zero in the compiled
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stp2d.f90:278-281`, then copies it to
`ssh_frc` at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:286-289`. legoESM's literal continuity path negates its
positive-convergence operand and produces negative zero. Values are equal,
but all 13,320 recorded cells differ in their zero sign. This is AT-BAR, not a
numerical owner.

The first value above the `2e-10` floor is earlier than the substep loop:
NEMO's completed V slow forcing at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:289,320-324`. It differs
only on 35 complete-recorded northern-fold cells, with maximum
`1.6557659420864476e-06 m s-2` at `[147,54]`; the active-domain values are
bit-exact. Substituting the recorded V forcing is itself bit-exact but is
trajectory-null: the missing post-update association overwrites the same
fold values. NEMO performs that seven-array association at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:712-741`.

The source-ordered arm census identifies the already-registered fold
cancelling unit:

| arm | first above-floor boundary | substep-2 `zhV` max error | substep-2 SSH max error | result |
|---|---|---:|---:|---|
| baseline | substep 1 `slow_v` | `70585.8134447541` | `0.0016553017522018282 m` | debt enters at the fold |
| recorded `slow_v` only | substep 1 `v_exit` | unchanged | unchanged | exact operand is overwritten; terminal arrays unchanged bit-for-bit |
| association only | substep 1 `slow_v` | `70585.8134447541` | `0.0014337083011086806 m` | closes the large substep-1 V exit, not its input |
| `slow_v` + association | substep 2 `transport_metric_v` | `70585.8134447541` | `0.0014337083011086806 m` | substep 1 is within the floor |
| plus NEMO's unmasked V transport | substep 2 `transport_metric_v` | `9933.931454588455` | `0.00022552707003865863 m` | large improvement, but the 65-substep arm grows to `11.812976037119444 m` SSH error |
| complete prior fold unit: plus raw `hu_0/hv_0` and materialised `zhV` | substep 2 `transport_metric_v` | `9.313225746154785e-10` | `6.938893903907228e-18 m` | first two substeps nearly close; later partner remains and terminal arrays move |

The V metric transport is the literal product
`zhV = e1v * va_e * zhvp2_e` at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:532,535`; NEMO has no extra
compact `v_mask` factor. The reference face depth is formed from raw `hv_0`
at `ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:512-519`, and the association uses V sign `-1` while depth
and reciprocal depth use sign `+1` at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:721-730`. The complete
arm is therefore one cited cancelling unit, not four independently landable
fixes. It is **HELD**: it does not put the OMT-0 solve at the bar, and its
unmasked partial arm becomes unstable within the same 65-substep window. No
package or card line lands this round.

This places all three registered structures at **OMT-0**, before momentum or
tracer advection, either lateral diffusion, or bottom drag is enabled: the
halo association, V-transport masking, and fold-HPG/slow-forcing partner are
not owned by OMT rungs 1-5.

## Baseline 65-substep table

These are complete-recorded-domain maximum absolute errors. `SSH`, `U`, and
`V` are the post-association arrays written by the oracle instrument. The
first-non-bit column includes zero-sign differences; the first-over-floor
column is the first numerical debt in source order.

| substep | first non-bit | first over floor | SSH max m | U max m/s | V max m/s |
|---:|---|---|---:|---:|---:|
| 1 | pgf_u | slow_v | 0.0 | 2.168404344971009e-19 | 0.0002751118796082097 |
| 2 | u_entry | v_entry | 0.0016553017522018282 | 0.000020464557812709203 | 0.0005441362201412241 |
| 3 | eta_entry | eta_entry | 0.005464962169529501 | 0.00006643556717676031 | 0.000802883175690743 |
| 4 | eta_entry | eta_entry | 0.010363202696355697 | 0.0001592686014082311 | 0.0010446399881243159 |
| 5 | eta_entry | eta_entry | 0.01578366701199451 | 0.0003030107874744184 | 0.0014464600163672874 |
| 6 | eta_entry | eta_entry | 0.022484817918277396 | 0.0004938684570456489 | 0.001854461156988874 |
| 7 | eta_entry | eta_entry | 0.029620481908031775 | 0.0007218101909844811 | 0.0022130727325347407 |
| 8 | eta_entry | eta_entry | 0.03637839048300172 | 0.0009726130052128701 | 0.0024966351986666266 |
| 9 | eta_entry | eta_entry | 0.04216964619760305 | 0.0012304791920406204 | 0.002762132074021052 |
| 10 | eta_entry | eta_entry | 0.04654652505152395 | 0.0014807803476345016 | 0.0029813562955916207 |
| 11 | eta_entry | eta_entry | 0.0494907634492357 | 0.0017124132761544193 | 0.0031634578888283782 |
| 12 | eta_entry | eta_entry | 0.052913269808826546 | 0.001920579233505549 | 0.0034738971596148553 |
| 13 | eta_entry | eta_entry | 0.056054489422626017 | 0.0022752635713341966 | 0.003834552852845704 |
| 14 | eta_entry | eta_entry | 0.06164128236748449 | 0.002701299667018855 | 0.004232759619707335 |
| 15 | eta_entry | eta_entry | 0.06764469203893306 | 0.0031496476396973127 | 0.004660821028122909 |
| 16 | eta_entry | eta_entry | 0.07282186830782054 | 0.0036142687680164164 | 0.005097171754845082 |
| 17 | eta_entry | eta_entry | 0.07707794109473266 | 0.004085976864488786 | 0.005517325575484573 |
| 18 | eta_entry | eta_entry | 0.0807633573161231 | 0.004550824525476048 | 0.0059048999229987355 |
| 19 | eta_entry | eta_entry | 0.08518497480886528 | 0.004989440925514168 | 0.006257068130849442 |
| 20 | eta_entry | eta_entry | 0.08807242905990792 | 0.005377929862472031 | 0.006582597308975934 |
| 21 | eta_entry | eta_entry | 0.09051815507455575 | 0.006190732281597807 | 0.006943381972701454 |
| 22 | eta_entry | eta_entry | 0.09382383934064377 | 0.007263776823187934 | 0.007351961354149569 |
| 23 | eta_entry | eta_entry | 0.09673815118472012 | 0.00843133165913545 | 0.007746627969130722 |
| 24 | eta_entry | eta_entry | 0.09893484445950737 | 0.009683392063893442 | 0.008130741924209704 |
| 25 | eta_entry | eta_entry | 0.10025290960386282 | 0.011005825835751899 | 0.008507825863377233 |
| 26 | eta_entry | eta_entry | 0.10474845311125719 | 0.012380685253594672 | 0.008879003697917049 |
| 27 | eta_entry | eta_entry | 0.11014118358723454 | 0.013786855073926286 | 0.009241676532910624 |
| 28 | eta_entry | eta_entry | 0.1145883407303662 | 0.015200970122856095 | 0.009589900329653877 |
| 29 | eta_entry | eta_entry | 0.11804691936768116 | 0.01659850464413019 | 0.00991613367114482 |
| 30 | eta_entry | eta_entry | 0.12053251601188475 | 0.017954925838228566 | 0.010213498234437004 |
| 31 | eta_entry | eta_entry | 0.1221155527876013 | 0.019246813533363726 | 0.010477617806865085 |
| 32 | eta_entry | eta_entry | 0.12291393950162285 | 0.020452869949369547 | 0.010707420586157194 |
| 33 | eta_entry | eta_entry | 0.12308364279456802 | 0.02155476983407846 | 0.010904794201884966 |
| 34 | eta_entry | eta_entry | 0.1228083459684553 | 0.022537824736750412 | 0.011073433076343904 |
| 35 | eta_entry | eta_entry | 0.12228892664741392 | 0.023391451524955975 | 0.011217449381250619 |
| 36 | eta_entry | eta_entry | 0.12173304500094834 | 0.024109443582251058 | 0.01134029067659602 |
| 37 | eta_entry | eta_entry | 0.12301504571369976 | 0.024690045551748207 | 0.011444285577573644 |
| 38 | eta_entry | eta_entry | 0.1288094362136167 | 0.025135832831298378 | 0.011530845365337635 |
| 39 | eta_entry | eta_entry | 0.13414474779836 | 0.02545339919809549 | 0.011601104336219402 |
| 40 | eta_entry | eta_entry | 0.13883390145425029 | 0.025652862544996526 | 0.011656661657239702 |
| 41 | eta_entry | eta_entry | 0.14273686752362372 | 0.025747210219093573 | 0.01170010960565985 |
| 42 | eta_entry | eta_entry | 0.1457854393214391 | 0.026479377446835234 | 0.011735161343946733 |
| 43 | eta_entry | eta_entry | 0.14799907777932622 | 0.027420685330858376 | 0.012039103664101153 |
| 44 | eta_entry | eta_entry | 0.15546286937162407 | 0.028274263079385922 | 0.012775160022858527 |
| 45 | eta_entry | eta_entry | 0.16282490591213986 | 0.029047659865910205 | 0.013534039387317988 |
| 46 | eta_entry | eta_entry | 0.169927610376064 | 0.02975073403619954 | 0.014313698924381335 |
| 47 | eta_entry | eta_entry | 0.17674369544791202 | 0.030986836832308635 | 0.015111543651250359 |
| 48 | eta_entry | eta_entry | 0.18325052449133877 | 0.03310542793101173 | 0.015924388517300058 |
| 49 | eta_entry | eta_entry | 0.18942992024023925 | 0.0352334546164596 | 0.016748440127712066 |
| 50 | eta_entry | eta_entry | 0.19526796551221168 | 0.03735919111247428 | 0.017579303105796924 |
| 51 | eta_entry | eta_entry | 0.20075483814910203 | 0.039470795703398516 | 0.018412014624334037 |
| 52 | eta_entry | eta_entry | 0.20588470711991436 | 0.04155642330266369 | 0.019241108713893465 |
| 53 | eta_entry | eta_entry | 0.21065569823679284 | 0.043604337510350415 | 0.020060709860498394 |
| 54 | eta_entry | eta_entry | 0.21506991935851213 | 0.04560302286680376 | 0.020864653419123528 |
| 55 | eta_entry | eta_entry | 0.21913351930148503 | 0.04754129764273045 | 0.021646628721453707 |
| 56 | eta_entry | eta_entry | 0.22285674426481136 | 0.049408426932286593 | 0.02240033958367021 |
| 57 | eta_entry | eta_entry | 0.22625395162455558 | 0.051194235143103214 | 0.023119676259647233 |
| 58 | eta_entry | eta_entry | 0.2293435434572626 | 0.05288921632069624 | 0.023798892685673635 |
| 59 | eta_entry | eta_entry | 0.23214779003373548 | 0.054484640204730844 | 0.02443278301381629 |
| 60 | eta_entry | eta_entry | 0.23469252498906562 | 0.055972651562591465 | 0.025016851795992848 |
| 61 | eta_entry | eta_entry | 0.2370067068667871 | 0.05734636021698458 | 0.02554747262951572 |
| 62 | eta_entry | eta_entry | 0.2391218543410023 | 0.0585999192787485 | 0.026022030506707913 |
| 63 | eta_entry | eta_entry | 0.2410713731567352 | 0.05972858938463238 | 0.026439043468678713 |
| 64 | eta_entry | eta_entry | 0.24288980077243366 | 0.06072878717441538 | 0.026798259436195567 |
| 65 | eta_entry | eta_entry | 0.2446119994740612 | 0.061598116765667045 | 0.027100724312191973 |

At substep 65 the baseline errors are therefore `0.2446119994740612 m`
SSH, `0.061598116765667045 m/s` U, and `0.027100724312191973 m/s` V. This
is internal kt=1 evidence; round 204 remains the authoritative kt=10 state
score (`0.0288100436468264 / 0.5086605459790218 m` SSH RMS/max at kt10
stage 3).

## Frozen predictions

| ID | disposition |
|---|---|
| R205-P1 | **CONFIRMED**: both inherited streams parse, reach EOF, and are exact between twins on defined bytes. |
| R205-P2 | **CONFIRMED**: all five untraced/traced terminal arrays are bit-exact; the one-ULP terminal plant fires. |
| R205-P3 | **REFUTED and retained**: active entry and histories are exact, but complete-recorded continuity zero signs and 35 slow-V fold cells differ before the loop. |
| R205-P4 | **REFUTED and retained**: the first numerical debt is the entry slow-V forcing, not an internal substep-1 statement. |
| R205-P5 | **CONFIRMED as a cancelling unit**: exact slow V alone is null, association closes substep 1, and mask/depth/materialisation are jointly required to reduce substep 2 to one-ULP scale. The unit does not close all 65 substeps. |
| R205-P6 | **CONFIRMED**: no `packages/` or card file changed. |
| R205-P7 | **CONFIRMED**: header, twin-ULP, source-order, passivity, and terminal-ULP plants each refuse at their named predicate. |

## Validation and review

- Round-205 measurement: `PASS_R205_OMT0_SUBSTEP_WALK` on CPU fp64/libm.
- Focused test: 7 passed.
- Plants: all five fire, with their exact refusal strings retained in
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round205/plants.log`.
- The measurement is repository-read-only with respect to model code; the
  only committed implementation is the fail-closed gate and its tests.
- Independent review unavailable in-sandbox. `codex exec --sandbox
  read-only` exited 1 before reading the diff with `failed to initialize
  in-process app-server client: Read-only file system (os error 30)`; this is
  retained in `round205/independent_review.log`, not represented as PASS.

## OPEN

1. Treat the positive-zero continuity construction as the literal first
   identity statement. A later landing must transcribe NEMO's
   `ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stp2d.f90:278-281` sign without changing nonzero freshwater semantics and
   pass the complete shared gates; no sign-only change lands here.
2. Score the complete fold unit as one private candidate over the OMT-0
   kt=1..10 ladder. Land only if Decision 96 passes; otherwise retain it as
   registered OMT-0 cancellation debt and name the next post-substep-2
   partner. Do not land any half.
3. OMT-1 (+ linear implicit bottom drag) remains next only after this OMT-0
   unit is dispositioned. Its month protocol starts there.

No configuration choice is pending. No additional NEMO stream is needed for
the next OMT-0 arm census.
