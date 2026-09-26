# Split-explicit momentum chain — round 92 result

Date: 2026-08-30. Session: `01a04e34-d1fb-73e0-b25a-177641f0a246`.

The faithful DINO cards now carry the post-stage Naa `wslpi/wslpj` pair into
the Redi vertical tensor while preserving the certified Kmm `uslp/vslp` and
the wholly Naa bolus tuple. Generic cards retain `redi_tuple`. The selector is
validated and threaded through every DINO lat-lon configuration constructor.

The physical owner is confirmed in production. Relative to round 87, skew
moves from `9462/9920`, `7.735372278997023e-8` to `86/9920`,
`3.319416367691509e-15`. A33 moves to `488/9920`,
`4.8788494395565624e-15`; the full temperature vertical flux is
`1001/9920`, `1.3361435463646085e-14`. Temperature zfu/zfv and rows 8.3-8.10
remain at bar. Receipt:
`/tmp/dino_split_explicit_momentum_chain_round92_association.json`, SHA-256
`5d04eea660a097ae68e14581f516c5ceb9d15d3d1b2573ad1b5cb8480466515d`.
Its legacy round-89 classifier correctly
prints `REDI_ZFW_T_SKEW_POSTFIX_REGRESSION`: the strict bar is not met.

The existing row-30 walk was rerun on CPU/fp64. For the j W-slope, `prd`, both
`zgrv` slots, `zaj`, `zbw`, `zbj`, and `zfk` are bit-exact; raw `zww` first
shows a `1.377e-15` maximum normalized error, and final `wslpj` is below bar.
Final `wslpi` retains one column at `1.179e-15`. Transcribing and fencing the
written two-arm `ldfslp.F90:320-328` expression is numerically inert, so it is
not claimed as the final-bit owner. The ordered stop is now the coupled exact
W-slope/Shapiro and A33 association floor. Salinity and the ZDF/Asselin tail
remain ordered-blocked; no physical-scale Redi residual survives.

Focused tests: three selected DINO/Redi tests pass. The broader DINO file has
an unrelated pre-existing `tke_etau_htau` latitude-profile failure at its
sixth test; this change does not touch that path.
