# Decision 36 preregistration — the GYRE card takes NEMO's shear-production statement

Date: 2026-09-12. Frozen base: `24ed85bf6e9f` (round 61). CPU, production JIT,
fp64/scalar-libm only. Written BEFORE the after-arm is measured; the
before-arm runs at the frozen base in `/tmp/codex-gyre`.

## The decision (ASKED, user 2026-09-12: "Yes")

Round 61 scored every operand the model carries into the kt=2 TKE closure
against the clean R59 record: the shear production `sh2` differs in all
17,400 wet interfaces (max `4.103e-8`) while every other operand agrees to
`<= 3e-17`. The GYRE card resolved legoESM's legacy statement
(`tke_shear_production="squared_centered"`, `tke_shear_avm_weighting="tpoint"`,
`tke_shear_metric_source="tpoint_jacobian"`); NEMO's compiled statement
(`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:83-114`) forms `sh2`
at the uw/vw faces from `(avm(i+1)+avm(i)) * du(Kmm) * du(Kbb) /
(e3w_1d*(1+r3u(Kmm)) * e3w_1d*(1+r3u(Kbb))) * wumask`, and
`stprk3.f90:168` calls `zdf_phy(kstp, Nbb, Nbb, Nrhs)`, so both levels are
the step-entry state — no new carried state. The three NEMO selections
already exist (the DINO NEMO card runs them). The user chose to select them
on the GYRE card.

## What changes

`nemo_testcase_recipe.py` (GYRE branch only): `tke_shear_production=
"nemo_face_native_now2"` (the face-native statement with both factors at
the step-entry level — the model refuses `nemo_face_native` outside the
leap-frog family because that variant reads a carried before-state; under
RK3 NEMO itself passes Nbb for both), `tke_shear_avm_weighting="nemo_face"`,
`tke_shear_metric_source="nemo_qco_live_face"`. No shared code changes; the
card constructs (checked before this file was committed).

## Predictions (falsifiers stated)

- P1 (mechanism): re-scoring the kt=2 carried `sh2` against the R59 record
  with the round-61 tool, the max abs difference drops from `4.103e-8` to
  below `1e-15` (the residual is set by the `2.7e-12` kt2 velocity gap).
  FALSIFIER: max abs stays above `1e-12`.
- P2 (ladder): `kt3.before.T` max drops from `3.7223442430e-4` normalised
  (`8.74e-3` K absolute) by at least 10x; the round-55 prediction for a
  NEMO-exact coefficient arm (`1.6275031290e-4` K / `6.3278533133e-6` g/kg
  absolute) is the target if the shear was the sole coefficient owner.
  FALSIFIER: kt3 T stays above `1e-3` K absolute. First-over-bar stays
  `kt2 u/v` (`2.7478404751243857e-12` / `3.305560306813421e-12`); kt2 rows
  unchanged bit-for-bit (the closure output is first consumed at kt=2's
  stage 3, after the kt2 rows are taken). FALSIFIER: any kt2 row changes,
  or first-over-bar moves earlier.
- P3 (month): the day-30 3-D T rms gap drops from `1.4241019e-02` K to
  below `5e-3` K. FALSIFIER: unchanged to three significant digits.
- Rows at kt>=3 are expected to move (improve); every moved row is
  registered, and any row that WORSENS is listed with its size.

Rule 12 per card: GYRE measured; LOCK_EXCHANGE/OVERFLOW run constant vertical
mixing (`ln_zdfcst=.true.`), the TKE statement does not execute; DINO's NEMO
card already selects these three forms on its own branch; ORCA2
UNMEASURED-with-spec.
