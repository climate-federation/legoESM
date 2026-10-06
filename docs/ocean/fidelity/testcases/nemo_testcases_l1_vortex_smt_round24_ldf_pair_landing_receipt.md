# VORTEX_SMT round 24 (lane round 236) — the LDF pair is HELD by the ratchet

**ROUND STATUS: HELD. No production physics or configuration remains changed.**

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round24_ldf_pair_landing.md`.
Base and immutable before arm: `9af53c118`, with the certified SMT-3
registry at `phase3/round226/smt3_ladder.json`. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round236/`.

The source-exact pair passes the local production-JIT proof and improves 30
of 50 SMT-3 aggregate rows, but the certified cellwise two-ULP comparison is
red. That is the preregistered falsifier, so production was restored and the
candidate is preserved only as
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l1_vortex_smt_round236_ldf_pair_held.patch`.

## 1. Compiled statements and exact scope

All citations below are from the compiled ppsrc of the record-producing build,
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/`.

* `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259`
  forms the horizontal tensor with the W masks at `jk` and `jk+1`. At the
  deepest tracer level NEMO reads the closed `jpk` W level; legoESM's held
  statement prevents a periodic vertical roll from wrapping the surface W
  mask onto that pair.
* The regular and deepest tracer RHS statements divide by the live
  `e3t_3d*(1+r3t(Kmm)*tmask)` at
  `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`
  and
  `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331`.
* The compiled stage program labels stage 3's middle slot `Kmm=N+1/2` at
  `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:225` and
  passes that slot to `tra_ldf` at
  `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:531-532`.
  Therefore the divisor takes the materialised stage-2-output SSH. The
  once-per-step slope and existing face-thickness operands keep their own
  already cited time levels.

No scheme, coefficient, threshold, timestep, stabiliser, carried state, card
selection, or restart layout was changed. The held implementation reuses the
shared `nemo_iso_lap` operator and the existing literal QCO T-thickness helper;
it adds no duplicate operator or helper.

## 2. Local production-JIT proof

The unmodified production replay first reproduced the frozen boundary:

| row | baseline | pair / switched production |
|---|---:|---:|
| max tracer-LDF RHS residual [K s-1] | 7.2621621143171395e-11 | 4.9526264855300654e-12 |
| fraction removed | — | 0.9318023144131400 |
| live `e3t(Kmm)` divisor unequal cells | 36,494 in the initially misrouted attempt | **0** after routing stage-2 SSH |
| closed-mask `hmsku` / `hmskv` unequal cells | non-bit baseline | **0 / 0** |

After the final routing correction, the ordinary production-JIT arm and the
explicit pair arm both report exactly
`4.9526264855300654e-12 K s-1`; their divisor arrays are bit-identical to the
record. R24-P1 and R24-P2 are therefore **CONFIRMED**.

The first switched replay used the step-entry SSH and produced a divisor
difference of 36,494 cells, maximum `1.4735542815174085e-03 m`. A post-hoc
hypothesis blamed the card's analytic bathymetry. Direct measurement refuted
it: the card depth equals the source-ordered reference-column sum exactly,
while the step-entry SSH differs from the compiled stage-3 Kmm SSH by
`1.4681564638190814e-02 m`. The compiled stage citation above then named the
actual routing error. Both failed intermediate claims remain in
`after_pair_reproof.json` and `after_pair_reproof_retry.json`; only
`after_pair_reproof_stage_kmm.json` is the accepted local proof.

The fail-closed one-ULP divisor plant uses the same production step, compares
against `pair_divisor_clean.json`, prints **`STATUS PLANT-FIRED`**, and exits
`1`. Two earlier invocations also exited nonzero but were correctly rejected
as controls because they lacked, respectively, `--clean-report` and a clean
report containing `--divisor-arm`; neither is counted as a fired plant.

## 3. Certified SMT-3 trajectory — R24-P3 REFUTED

The candidate keeps all kt=1 rows at the bar and first-over-bar stays kt=2
with T/u/v/ssh. No aggregate row changes AT-BAR/DEBT status. Nevertheless the
cellwise oracle-relative gate reports **FAIL**: 40 row violations, with maximum
worsening `431802197.9165039` row-scale oracle ULP against the two-ULP bar.
That is sufficient to hold the candidate.

All 34 changed aggregate rows are registered below; 30 move toward NEMO and
four move away.

| row | before | candidate | direction |
|---|---:|---:|---|
| kt2 T | 1.020298116571876e-08 | 6.957537701781045e-10 | toward |
| kt3 T | 2.120981422817502e-08 | 2.839158383429883e-09 | toward |
| kt3 ssh | 1.527741377849168e-09 | 6.231736793260723e-10 | toward |
| kt3 u | 1.042009542540079e-08 | 1.013289665369177e-08 | toward |
| kt3 v | 7.401099691439761e-09 | 1.654683991821754e-09 | toward |
| kt4 T | 3.437562734586925e-08 | 2.417274950323488e-09 | toward |
| kt4 ssh | 4.082109539282897e-09 | 8.099967918617779e-10 | toward |
| kt4 u | 2.874344674774765e-08 | 2.285250661274163e-08 | toward |
| kt4 v | 2.081734762002241e-08 | 3.529006063502510e-09 | toward |
| kt5 T | 4.882805569776860e-08 | 4.965180625967719e-09 | toward |
| kt5 ssh | 4.542263676299285e-09 | 1.104131674978248e-09 | toward |
| kt5 u | 4.402378664739026e-08 | 3.626384267874094e-08 | toward |
| kt5 v | 3.752869002365822e-08 | 6.223943748405458e-09 | toward |
| kt6 T | 6.403584661271743e-08 | 8.243512792624066e-09 | toward |
| kt6 ssh | 5.315172857400796e-09 | 1.297312923753680e-09 | toward |
| kt6 u | 8.483425171373904e-08 | 5.425727961361204e-08 | toward |
| kt6 v | 7.029845694450643e-08 | 1.154071884172832e-08 | toward |
| kt7 S | 1.624097681737370e-15 | 1.421085471520199e-15 | toward |
| kt7 T | 8.262627111845391e-08 | 1.235688425850942e-08 | toward |
| kt7 ssh | 6.501331917263542e-09 | 3.422397134786315e-09 | toward |
| kt7 u | 1.466380783222976e-07 | 1.512125080566085e-07 | **away** |
| kt7 v | 1.089517367991455e-07 | 2.217514574689875e-08 | toward |
| kt8 T | 9.520935155668749e-08 | 1.747103006144312e-08 | toward |
| kt8 ssh | 1.199041754773589e-08 | 7.703695974539682e-09 | toward |
| kt8 u | 3.721325309569146e-07 | 3.833577468595473e-07 | **away** |
| kt8 v | 1.511664352860098e-07 | 4.287059453614295e-08 | toward |
| kt9 T | 1.327717592185642e-07 | 2.455230544376252e-08 | toward |
| kt9 ssh | 2.543102683993936e-08 | 1.050440961948215e-08 | toward |
| kt9 u | 6.562476215130461e-07 | 6.684648320137095e-07 | **away** |
| kt9 v | 2.345347185104863e-07 | 1.127359968592872e-07 | toward |
| kt10 T | 1.959980453527762e-07 | 3.498342973430311e-08 | toward |
| kt10 ssh | 3.049630914464530e-08 | 1.316862729278112e-08 | toward |
| kt10 u | 8.195995781950300e-07 | 8.402824479748999e-07 | **away** |
| kt10 v | 3.290032220972971e-07 | 2.181972887413521e-07 | toward |

The largest aggregate benefit is the kt2 T row (14.67 times smaller); it does
not license the cellwise violations. This round does not infer a cancellation
mechanism from those violations.

## 4. Landing verdict and blast radius

R24-P3's explicit falsifier fired, so the round stopped the landing sequence
before R24-P4. The GYRE ladder/year, generic recipe, remaining VORTEX/SMT
registries, tanks, DINO month gate, and ORCA2 ladder were **NOT MEASURED on
the refused arm**. Calling them passed would be false. After restoring
production, the SMT-3 registry reproduces the immutable before arm with the
oracle-relative comparison PASS; therefore no certified baseline or pin is
changed by this round.

ORCA2 pointer: its rung-0 isoneutral lateral-diffusion path shares both cited
statements. The held pair must not be folded into ORCA2 as a landing; its lane
may use the manifest only as a measurement arm and must apply its own
cellwise ratchet.

## 5. Review, citations, and tests

Independent read-only Codex review: **PENDING FINAL PASS**.

Citation gate: **PENDING FINAL PASS**. The shifted-citation plant must exit
nonzero.

Focused tests: **PENDING FINAL PASS**.

## 6. OPEN — next round

1. **SMT-4 begins now.** Per Decision 93 and the lane budget order, build the
   lateral-momentum-diffusion rung (`ln_dynldf_lap + ln_dynldf_lev`,
   `nn_ahm_ijk_t=20`, with the stated ORCA2 rung-0 stand-in), print the deck
   diff and every explicit card value, acquire kt=1..10 plus 100-day records,
   then name the first owner in compiled order over partial cells.
2. The SMT-3 mask/divisor pair remains HELD at the manifest above. Do not
   retry it as a single landing unless a later same-stage statement removes
   the measured ratchet violations; the complete before/after registry is
   `round236/smt3_compare.json`.
3. The downstream `2.597863951157186e-14 K s-1` arithmetic residue remains a
   closed bit walk. Round 236 does not reopen it.

**DECISION_NEEDED: NONE. ACQUISITION_NEEDED: NONE.**
