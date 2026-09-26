# Preregistration: row-5 upstream-exact certification, round 46

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 45 shows that the literal `dynspg_ts.F90:1170-1174` Kmm rewrite removes
the structural call-2 defect but leaves Q/H/W residuals of order `2e-6` to
`4e-6`. The Kaa residual is unchanged by retaining raw pre-projection SSH.
The row-1.3/1.4 and round-32 standards were measured with NEMO's already-owned
`zu_frc/zv_frc` substituted at the split-explicit solver boundary. This round
reuses that exact hold from `split_explicit_momentum_chain_round6.py`:
`spg_dump_zu_frc.bin` and `spg_dump_zv_frc.bin`, mapped once to legoESM's
redundant U/V layouts. No SSH, seed, QCO, transport, hdiv, or W field is held.

The held forcing must be byte-exact on 9758/9868 wet faces and the production
seed must remain unmodified. The unique production call-2 capture must carry
raw Kaa SSH plus the model's own `Hu_avg/Hv_avg`; Q, H, and W are scored in
that order against the same retained streams. Each requires normalized RMS
and maximum error over NEMO RMS no larger than `1e-12`. Identity and roll
controls, hook/solver restoration, post-dyn_zdf W carry, finite arrays, and
cited-interior shapes remain mandatory. The unheld round-45 Q/H/W failures
are red controls.

Disposition is `ROW5_WZV_CALL2_AT_BAR_UPSTREAM_EXACT` only if every bar and
control passes. That promotes row 5 as a local operator conditional on the
already-recorded row-1.1 upstream debt; it does not relabel the unheld
production chain exact. Otherwise the first Q/H/W failure stays open. Row 6
remains ordered-blocked until this receipt.
