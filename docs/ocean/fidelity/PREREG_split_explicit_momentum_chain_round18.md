# Preregistration amendment: row-1.2 BEFORE thickness, round 18

Date: 2026-08-29. Frozen after round 17 and before this measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 17 used `sshn` to reconstruct the live face stretch and returned
`INPUT_OR_TIME_LEVEL_OPEN`: literal left accumulation remained above the
POINTWISE `1e-15` maximum-error bar. Source inspection then found the missing
time level. In an MLF restart, `restart.F90:331-352` reads `ub/vb` but no 2-D
barotropic restart. `istate.F90:149-155` reconstructs `uu_b/vv_b(Kbb)` by
zero-initialized, surface-to-bottom accumulation of
`e3u/e3v(Kbb)*ub/vb*mask`, followed by `r1_hu/r1_hv(Kbb)`. The live Kbb
thickness is controlled by BEFORE SSH `sshb`, not NOW SSH `sshn`.

The sole new authoritative arm replaces `sshn` with restart `sshb` in the
otherwise unchanged committed round-17 scorer. The round-17 artifact is
SHA-bound as the NOW-level control. Same inputs, populations, CPU/fp64 gate,
source-ordered multiplication and left accumulation, controls, and POINTWISE
`1e-15` RMS/maximum bars apply.

- If both U and V literal BEFORE arms are at bar and the bound NOW arm is not,
  disposition is `OWNED_BY_BEFORE_THICKNESS_TIME_LEVEL_AND_LEFT_ACCUMULATION`.
- If the BEFORE arm is above bar, disposition remains
  `INPUT_OR_TIME_LEVEL_OPEN` and a SLOT writer must snapshot actual
  `e3u/e3v(Kbb)` and the final partial sums in `istate`.
- If BEFORE and NOW both pass, the time-level contrast is not discriminating
  and row 1.2 remains open.

No production change or later-row promotion is authorized unless both
components pass. If they do, the production fix must preserve the NEMO
source-ordered seed association and use the BEFORE eta already supplied to the
MLF barotropic call; rows 1.2--1.4 are then replayed in order.

