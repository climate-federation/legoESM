# NEMO-testcases L2 GYRE round-65 Krhs receipt

Date: 2026-09-12

## Verdict

No transcription is eligible to land.  Both preregistered candidates are
REFUTED.  Production is restored byte-for-byte to parent `7069668c9bac`; the
temporary native-transport diagnostic is removed at the final tip.
Every NEMO citation below refers to the actually compiled branch
`cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo`.

## Admission calibration

The round-64 admission is PASS: 43/63 inherited records were exact, 20 changed,
and all 132 consumed fields were admitted.  Independent calibration reports
zero unequal for `content_T`, `content_S`, `e3t_Kbb`, `e3t_Kmm`, TKE
`strat_product`, `diss_product`, and `rhs_statement`.  Both record schemas are
fp64 `(32,22,30)`, with 23 Krhs and 11 TKE fields.  Truncation, one-ULP, and
wrong-commit plants all exited nonzero.

## Model-path Krhs boundary

The model starts at NEMO's kt=2 entry, then uses its compiled production step
and live stage-3 operand trace.  The compiled program zeroes Krhs at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-828` and calls
advection at `:860`.

| boundary | T unequal/max abs | S unequal/max abs |
|---|---:|---:|
| zero | 0 / 0 | 0 / 0 |
| FCT upstream first guess | 18,000 / 6.785710994e-11 | 18,000 / 5.884990366e-12 |
| complete limited FCT | 18,000 / 6.196948294e-11 | 18,000 / 5.135360970e-12 |
| SBC | UNMEASURED | UNMEASURED |
| QSR | UNMEASURED | UNMEASURED |
| LDF | UNMEASURED | UNMEASURED |
| content | prior clean-stamped model-path row: 18,000 / 1.679392692e-3 | prior clean-stamped model-path row: 18,000 / 6.967976674e-5 |

The first unequal written boundary is therefore `fct_up1_2stp`'s Krhs writer
at compiled
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:602-611`, observed
at `:607`.  It is not an
operator-owner claim: before that writer, the live Kmm inputs already differ:
T 17,994/18,000 (8.369461071e-7 K), S 16,769/18,000
(6.794565621e-8), and `e3t(Kmm)` 18,000/18,000
(2.472597771e-8 m).  Kbb T/S and `e3t(Kbb)` are exact.

The statement ladder follows compiled upstream faces `:503-510`, midpoint
divergence/update `:532-540`, averaged upstream faces `:570-580`, Krhs/guess
writers `:602-611`, centred faces `:197-201,264-269`, active memory-optimised
nonosc bounds/min/max/budgets/limiters `:798-933`, and final correction
`:318-329`.  Replacing the live `zFu/zFv` association does not clear the first
guess and worsens complete FCT (T max 2.231849600e-7; S 8.221876777e-9).
Replacing all attempted recorded Kmm/thickness/`ww` operands also remains
non-bit.  Therefore the preregistered `:503-506` transcription candidate is
REFUTED; the record lacks the face/subresult stream required to continue this
ladder without reconstructing an oracle value NEMO never wrote.

## TKE

Compiled
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/zdftke.f90:428-431` adds shear,
`-p_avt*rn2`, and the dissipation subterm `+zfact3*dissl*en` at `:430`.
Direct shared-statement scoring with
all admitted operands gives 0/18,000 unequal for stratification product,
dissipation product, and complete RHS under both the current and explicitly
barriered compiled associations.  The one-ULP plant exits 1.  Thus the known
live-path 238/17,400, max 1.665334537e-16 residual is an upstream-operand
residual, not a line-430 transcription; no TKE patch lands.

## Rule 12 and trajectories

With no eligible code change, GYRE/LOCK_EXCHANGE/OVERFLOW/DINO/ORCA2 Rule-12
change cards are NOT APPLICABLE.  Resolved LOCK_EXCHANGE and OVERFLOW do run
FCT2 (`ln_traadv_fct=T`, `nn_fct_h=nn_fct_v=2`), but no moved row exists to
measure.  GYRE kt=3 remains T 1.627511418e-4 K and S 6.327755180e-6 max; day-30
remains T RMS 1.239756827e-2 K and S RMS 2.195296278e-3.  First-over-bar is not
earlier because production is unchanged.  DINO shared-statement risk is zero;
ORCA2 remains UNMEASURED with the existing exact specification: run its
production gate on the same final commit if a shared FCT/TKE statement moves.

## ASKED / UNASKED

| state | item | result |
|---|---|---|
| ASKED | configuration choice | none encountered |
| UNASKED | change configuration or carried state | not performed |

Open measurement: acquire WRITE-only stage-3 FCT `pU/pV/pW`, first-step
faces, `zta_up1`, bounds, budgets, limiters, and corrected faces at kt=2.  That
is the minimal oracle needed to name a first non-bit FCT statement.
