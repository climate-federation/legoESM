# GYRE round 62 post-code self-review

Date: 2026-09-12. Production parent: `b6027cebae17`.

## Scope and result

- Production packages are bit-identical to the parent. Both preregistered
  content-association candidates were refuted and explicitly reverted.
- The retained changes are fail-closed diagnostics, tests, preregistrations,
  citation mappings, this review, and the receipt correction.
- The expanded tracer walk uses the model's own step to capture the shared ZDF
  call, substitutes one recorded operand/row at a time, and includes an
  operand-ULP plant. A first implementation exhausted LLVM memory because it
  compiled the full step once per arm; the retained implementation captures
  once and re-enters only the shared dispatcher.
- The TKE walk captures the actual production solver inputs. Virtual surface
  and unread deepest-upper storage are excluded from comparisons.

## Claims deliberately not made

- Full recorded content identifies an upstream boundary, not an individual
  stage-3 tracer operator. The existing record does not split Krhs.
- The all-recorded coefficient materialization is not the live kt=3 owner and
  has not passed Rule 12.
- The 238-cell TKE RHS residual names the complete RHS statement only; the
  source-order reconstruction does not split it further.
- No physics fix, GYRE before/after candidate, DINO result, or ORCA2 result is
  claimed.

## Review availability

Independent GLM review could not start because `ZAI_API_KEY` is absent.
Independent Codex review could not start because network access is denied.
Under Rules 9 and 12, no physics candidate is eligible to ship. The diagnostic
evidence is marked UNREVIEWED.

At clean stamp `06b00873b`, the receipt citation gate passes and its shifted
line plant exits nonzero. The operand-ULP production plant detects one changed
cell and exits nonzero. All 50 focused tracer, TKE, citation, and provenance
tests pass.

## Retraction audit

The two failed content candidates are absent from production. The decision-36
receipt's false “all five kt2 rows bit-identical” sentence is replaced with
the measured status: T/S AT-BAR but non-bit, u/v DEBT, ssh exact.
