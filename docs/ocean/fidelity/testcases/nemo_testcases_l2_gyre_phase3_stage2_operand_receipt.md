# NEMO testcase lane 2 — GYRE RK stage-2 operand receipt

Date: 2026-09-01

Session: `ea650f83-28b8-4c68-b0cc-809a9fd417de`

Preregistration commit: `0a83eb45d47fb5e488bca917b58f18018ba1087b`

Parent drag landing: `fe8af4f79b7dfeb5245ab637ee28d60e9a6aecfc`

## Verdict

**DEBT, localized to the stage-2 baroclinic RHS before the physical Kaa
composition.**  Stage-1 Kmm U/V and the external-mode target remain AT-BAR.
The invariant stage-2 RHS differs by `1.4419237554233738e-11` U and
`3.430804646662383e-11` V.  Multiplication by the resolved half-step
`7200 s` gives `1.0381851039048291e-7` and `2.470179345596916e-7`, matching
the observed raw and corrected stage-2 residuals to roundoff.  This is a
scaling localization, not yet an HPG, vorticity, or advection owner label.

## Source and alias adjudication

`stprk3_stg.F90:178-182` selects `rn_Dt/2`; `:335-351` accumulates HPG,
vorticity, and vector-invariant advection; `:377-386` advances from Kbb; and
`:450-462` replaces the raw depth mean with `uu_b/vv_b(Kaa)`.  The stage-2
call at `stprk3.F90:219-225` resolves `Kbb=1`, `Kmm=3`, `Krhs=2`, `Kaa=2`:
Krhs and Kaa alias.  Therefore a Krhs read at the later correction boundary is
already the raw updated Kaa state.  The gate no longer prints that aliased slot
as a pre-update RHS; it reconstructs the source RHS from
`(raw_Kaa-Kbb)/7200`.

NEMO carries the full depth mean through raw integration and removes it with
`zub/zvb`; the collapsed legoESM identity removes the mean before integration
and restores the external target afterward.  Raw mean and literal correction
therefore use different but algebraically equivalent intermediate gauges.
Those two rows are explicitly UNINFORMATIVE, with reasons.  The gate scores
the gauge-invariant baroclinic RHS, raw baroclinic Kaa, external target, and
corrected Kaa.  It does not demand a Frankenstein match of internal storage.

## Operand table

All rows use fp64 native owned masks and the exact normalized bar `1e-15`.
Scaling precedes localization.

| row | U absolute | V absolute | status |
|---|---:|---:|---|
| Kbb | `0` | `0` | AT-BAR |
| corrected stage-1 Kmm | `2.710505431213761e-19` | `2.710505431213761e-19` | AT-BAR |
| baroclinic RHS | `1.4419237554233738e-11` | `3.430804646662383e-11` | DEBT |
| raw baroclinic Kaa | `1.0381851039048275e-7` | `2.470179345596902e-7` | DEBT |
| external barotropic target | `2.710505431213761e-19` | `2.710505431213761e-19` | AT-BAR |
| corrected Kaa | `1.0381851039053744e-7` | `2.4701793455947666e-7` | DEBT |
| raw mean / literal correction gauge | — | — | UNINFORMATIVE |

The baroclinic RHS oracle magnitudes equal the errors to roundoff, so the
candidate term is near zero at this boundary.  That fact is not an operator
exoneration: the source-ordered component split is still UNMEASURED.

## Instrument and controls

The separate `NEMO_L2_RKSTG_1` stream records Kbb, corrected stage-1 Kmm, the
aliased Krhs/Kaa slot, raw Kaa, external target, literal correction, and
corrected Kaa.  It never assigns model state.  The reader encodes the actual
layout: U/V state and external target have full halos, while `zub/zvb` are
native A2D temporaries.  Corrupt header/payload tests fail closed.  The private
raw-stage exposure substitutes U/V only after the ordinary step completes;
T/S are bit-identical to the uninstrumented result.

All previously accepted oracle step-entry, stage, RHS, barotropic-frame,
substep, ENE, and drag artifacts retain their registered hashes bit-for-bit.
The planted `+1` RHS control fired at `1.0000000000021816` and VERIFIED.  The
selected fidelity, WS tracer-stage, recipe, drag, and barotropic regressions
pass: **85 passed**.

## Review and next boundary

This new localization is **UNREVIEWED** by independent Claude/GLM reviewers.
No dual-review claim is made.  The next committed walk splits the source RHS
as HPG -> live ENE vorticity -> vector-invariant advection and separately
measures stage-1 T/S, because NEMO's stage-2 HPG consumes Kmm tracers while the
current legoESM momentum substage reuses frozen step-entry density.  That is a
registered hypothesis, not yet an owner finding.  Stage-2/3 T/S and the
kt=2-10 trajectory remain DEBT/UNMEASURED; no trajectory is called matched.

Run root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10_stage2_walk`.
