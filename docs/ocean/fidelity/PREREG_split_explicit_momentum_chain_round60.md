# Preregistration: row-8.3 residual attribution, round 60

Date: 2026-08-30. Frozen before the round-60 offline measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 59's held slow-forcing production replay reduced row 8.3 from 170 to
104 red U columns and the maximum normalized column error from
`5.9792156327e-15` to `1.4948039082e-15`, but did not meet the unchanged
`1e-15` pointwise bar. Do not promote it.

Use only the retained 68-row NEMO `substep_dump.bin`, `mesh_mask.nc`, and
`spg_dump_un_adv_final.bin`. Reconstruct each source operand at
`dynspg_ts.F90:645-704`: AB3 mid-step U and SSH, live QCO U-face depth, raw
`zhU=(e2u*ua_e)*zhup2_e`, and stored-form reciprocal `1/e2u`. Evaluate the
four arithmetic arms formed by:

- raw `wgtbtp2` recurrence plus one final `/r1_wgt2s`, versus a
  pre-normalized secondary weight; and
- materialized `zhU` followed by `*r1_e2u`, versus the cancelled
  `ua_e*zhup2_e` form.

Every arm is accumulated substep-by-substep in source order. Compare the
result with NEMO's retained final `un_adv` over the real surface U-mask. The
registered metric is maximum absolute error divided by wet-face oracle RMS;
the unchanged pointwise bar is `1e-15`.

CONFIRM local round-59 arithmetic if and only if the raw+metric arm is at bar,
the reconstruction self-check is finite, and at least one nonliteral arm is
red. If raw+metric is red, the round-59 implementation remains unresolved:
the next measurement must capture production `Hu_avg` and separate upstream
substep-operand error from the final accumulator. Identity, one-wet-point,
substep-roll, and cancelled-form plants must fire. No NEMO build or run is
authorized or needed.

Frozen inputs:

- round-59 held receipt SHA-256
  `85ea27cce4804d98f281940fe472e798d9fa64c741c23bb55e3fca40ee9ca677`;
- `substep_dump.bin`
  `39a2b3f6464758233d75eea1112aebb73b0f21383761e43d2e2d765a94b42e32`;
- `mesh_mask.nc`
  `3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622`;
- `spg_dump_un_adv_final.bin`
  `4d8e7a6445ba465c8229954b443c467862381409805163f2045f5854017917ab`.
