# Adversarial review packet — MPAS q_v smoothing (round 4, doc-precision close)

Independent adversarial reviewer. In round 3 you returned "no substantive
implementation defect remains" and one precision/documentation correction. That
correction is now applied VERBATIM. Confirm it is resolved and that NO other
substantive finding exists. This round changed ONLY comments/docstrings — no code.

## Applied since round 3 (documentation only)

1. "exactly conserved" / "strict no-op" now QUALIFIED as exact-arithmetic / to
   floating-point roundoff (~1e-7 rel in fp32) in all three places:
   - operator docstring `scalar_del2_cell_3d`: "``sum_c A_c q_c`` is conserved
     exactly in exact arithmetic ...; in fp32 to floating-point roundoff (~1e-7
     relative)."
   - `_mpas_qv_smooth_step` docstring: "the floor is then a no-op and the
     per-level ``sum_c A_c q_c`` integral is conserved (in exact arithmetic; to
     floating-point roundoff — ~1e-7 relative in fp32).  The floor only
     sanitises finite negatives ..., NOT pre-existing NaN/Inf (max(NaN,0)=NaN)."
   - driver setup guard comment: "keeps the q>=0 floor a no-op, so the per-level
     sum_c A_c q_c integral is conserved — exact arithmetic, to fp roundoff
     ~1e-7 in fp32."

2. Stale `dp_cell_3d=None` wording removed from `scalar_del2_cell_cfl_factor`'s
   docstring (that argument no longer exists): now just
   "``q + nu*dt*scalar_del2_cell_3d(q, mesh)`` is a convex combination ...".

3. The misleading "pre-weight ``q`` yourself" line in `scalar_del2_cell_3d`'s
   docstring is corrected: "if you need it, use a separate weighted-flux
   implementation ``div(dp_edge*grad q)/dp`` AFTER asserting dp > 0 on your
   coordinate — it cannot be obtained by pre-weighting ``q``."

No source/logic lines changed. `17 passed` for
`tests/unit/test_mpas_qv_smoothing.py` + CLI round-trip still green.

## Ask

Confirm the round-3 documentation correction is fully addressed and there are no
remaining substantive findings (implementation, conservation, sign,
differentiability, dtype, config/CLI, test vacuity). If clean, state "no
substantive findings" explicitly.
