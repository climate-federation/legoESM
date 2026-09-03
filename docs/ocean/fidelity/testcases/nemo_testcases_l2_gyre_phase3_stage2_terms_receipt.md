# NEMO testcase lane 2 — GYRE stage-2 term split receipt

Date: 2026-09-01

Session: `ea650f83-28b8-4c68-b0cc-809a9fd417de`

Preregistration commit: `14fc05fe99906684b5f6c773b93223e61b442ac2`

Parent localization: `e60d0bec8ab0268b630f52573dc9d1cb085a65ff`

## Verdict

**DEBT, with HPG the first source-ordered stage-2 term and the T/S-only causal
arm refuted as an owner.**  The full baroclinic RHS residual is carried by HPG
at `1.441923755423408e-11` U and `3.430804646663045e-11` V.  Vorticity and
advection are AT-BAR in absolute terms (`<=1.99e-23` and `<=4.59e-26`) and are
NEAR-NULL at this state; that gives them no discriminating power and is not an
exoneration for later states.  Injecting oracle stage-1 T/S into stage-2
EOS/HPG moves only `8.8301%` of the corrected-stage residual and does not clear
the HPG or Kaa rows.  Its honest label is
**NEAR_NULL_NO_DISCRIMINATING_POWER**.

## Source-order evidence

The WRITE-only `NEMO_L2_RKTRM_1` stream captures Krhs immediately before HPG,
after `dyn_hpg`, after `dyn_vor`, and after `dyn_adv`, following
`stprk3_stg.F90:338-351` literally.  Successive differences are reduced to
their fixed-depth baroclinic component before scoring, so the later zub/zvb
gauge cannot leak into this term walk.  The candidate diagnostic seam returns
the already computed `-dp/rho0`, live ENE vorticity, and KEG+ZAD arrays; its
static false default preserves the production return.

| source term | U absolute | V absolute | oracle max | disposition |
|---|---:|---:|---:|---|
| HPG, frozen step-entry T/S | `1.441923755423408e-11` | `3.430804646663045e-11` | same as error | DEBT |
| live ENE vorticity | `1.9852334701272664e-23` | `1.9852334701272664e-23` | `1.3234889800848443e-23` | AT-BAR, NEAR-NULL |
| vector-invariant advection | `1.2761797294212918e-26` | `4.5878572525843605e-26` | `<=4.5880150247654047e-26` | AT-BAR, NEAR-NULL |

The HPG residual equals the previously reconstructed combined RHS residual to
roundoff.  Scaling therefore localizes the next operand to the stage-2 HPG
input bundle before any owner label.

## Tracer-stage measurement and causal arm

The existing oracle stage artifacts always contained T/S; the gate now exposes
the candidate Kaa tracer stages only after the ordinary step completes and
scores them directly.  Stage-1 T and S are already DEBT by
`0.3241418107674683` and `1.8542789845810148e-6`.  Stage-2 T and S are DEBT by
`0.32414315819211126` and `2.781416519326285e-6`.  This is measured debt, not
the prior UNMEASURED placeholder.

The preregistered T/S-only arm changes no velocity, external target, transport,
vorticity, advection, or returned tracer state.  It leaves direct HPG errors of
`1.6559728275028788e-11` U and `3.242480961563407e-11` V and corrected-stage
errors of `1.1922972343819893e-7` U and `2.3345858285246356e-7` V.  Its maximum
movement is `2.181197751862057e-8` against a `2.4701793455947666e-7` faithful
residual, ratio `0.0883011897800834`.  That is below the preregistered one-tenth
threshold and refutes T/S alone as a discriminating owner.

## Controls, review, and continuation

All prior oracle artifacts remain bit-identical.  The source-order reader has
fail-closed header/payload and permutation tests.  The planted post-HPG control
fired at `1.0000000000021816` and VERIFIED.  The selected fidelity,
WS-tracer-stage, and momentum-diagnostic regressions pass: **30 passed**.

This round is **UNREVIEWED** by independent Claude/GLM reviewers.  The next
operand is the remainder of the stage-1 Kmm thermodynamic geometry.  In
particular, NEMO's stage-2 HPG consumes stage-1 SSH/thickness as well as T/S,
whereas the current legoESM momentum substage reuses the step-entry geometry
bundle.  That geometry extension is UNMEASURED until its own committed
preregistration and causal arm.  No kt=2 trajectory match is claimed.

Run root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10_stage2_terms`.
