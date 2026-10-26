# Preregistration: literal Kmm tracer-entry production fix, round 56

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 55 reproduces the old row-8.3 signature after all momentum-loop fixes:
`9758/9758` columns diverge, correlation is `0.9846254752`, RMS ratio is
`1.0294306361`, and the largest normalized column error is `2.6794216264`.
This cannot be the few-ULP production-transport qualification from round 53.
Source inspection identifies the missing write: legoESM applies the literal
Kmm execute/undo cycle only to WZV call 2 and the later Asselin filter, while
its tracer mass flux still reads the generic `state_new` velocity. NEMO writes
the corrected Kmm velocity at `dynspg_ts.F90:1170-1174`, then `traadv.F90:
301-304` points `zptu/zptv` at exactly that state.

The production change routes the execute half of the existing
`nemo_qco_kmm_velocity_cycle` into the tracer mass-flux block when the already
two-card-scoped `wzv_call2_evaluation="nemo_literal"` selector is active.
The prescribed-flow arm and generic arm remain separate and unchanged. Re-run
round 55's full ordered 8.3--8.10 ladder against the same held streams and
unchanged bars. Row 8 is promoted only if every subrow passes and all prior
plants fire. A stale-Kmm unit plant must differ, selector defaults must remain
literal only on `nemo_dino_kamm` and `nemo_dino_kamm_mlf`, and focused generic
path tests must remain green. Otherwise stop at the first red subrow.
