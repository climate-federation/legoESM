# Preregistration: row-6 target × association factorial, round 48

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 47 (`60a4b7d8f45b8cc9a611deb4d4e641e5952c67960f6edff2ddade51771a3e4aa`)
leaves only a few-ULP pointwise row-6 residual. This round runs a 2×2 local
factorial from existing dumps, under the unchanged round-46 `zu_frc/zv_frc`
hold:

* T0 is the production primary barotropic target captured at the unique
  `_apply_after_level_reconcile` call; T1 substitutes full-halo
  `spg_dump_puu_b_final.bin` / `spg_dump_pvv_b_final.bin`.
* A0 is production `after_level_column_mean_reconcile`; A1 evaluates
  `stpmlf.F90:709-765` literally: source-left vertical accumulation of raw
  reference `e3u_0/e3v_0 * velocity * mask`, separately associated
  `r1_hu_0/r1_hv_0`, then `before - mean + target` in source order.

All four arms start from `baro_dump_{u,v}_before_kt00005761.bin` and score
against the corresponding after dumps. The same POINTWISE `1e-15` normalized-
RMS and maximum/RMS bars apply. T1A1 must be at bar or the transcription is
invalid. Identity, wet-point and roll plants, nonzero correction, full-halo
sizes, 9758/9868 surface wet populations, unique hook/solver calls, null, and
restoration remain required.

Ownership uses squared-error removal relative to T0A0 plus the exact arm
table. At least 95% removal by one single substitution with the other below
5% localizes that operand; two material main effects or a negative single-arm
removal with joint closure is `TARGET_X_ASSOCIATION_COMPOSITION`. If the
literal T1A1 arm misses the bar, row 6 remains `OPEN_UNRESOLVED`. A production
fix is built only after this disposition. The free-surface and later chains
remain ordered-blocked.
