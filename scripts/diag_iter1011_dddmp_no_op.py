"""Iter-1011: confirm dddmp_prod is silently no-op for our W5 case.

In `cdgrid_momentum_tendencies` and `fv3_sw_tendencies`, the
adaptive Smagorinsky damp is `max(d2_bg, min(0.20, dddmp*|div|))`.

For iter-1009 calibration:
  div_damp = 10 * _div_damp_cube(36) = 2.67e8
  da_min_c ≈ 3.85e10
  d2_bg = div_damp / da_min_c ≈ 0.0069

For dddmp*|div| > 0.0069, |div| must exceed 0.0069/dddmp.  W2 and
W5 typical divergence is much smaller, so adaptive term is always
< d2_bg and the max picks d2_bg.  dddmp has no effect.

This explains why iter-1011 sweep of dddmp_prod=0.05..0.50
produced IDENTICAL W2 and W5 metrics across the entire range.

Conclusion: adaptive Smagorinsky cannot help W5 long-term in this
regime.  Only div_damp_factor matters for the static damping
strength, and it's already at the W2/W5 trade-off optimum (10*).
"""
print(__doc__)
