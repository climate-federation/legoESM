# Preregistration: production Kmm-cycle certification, round 53

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 52 localizes the momentum tail to the centered Kmm execute-and-undo
cycle. The production change reuses one source-ordered operand builder for
both call-2 WZV and the post-`mlf_baro_corr` NOW value entering
`dyn_atf_qco`. The raw `Hu_avg/Hv_avg` transports accompany the existing raw
Kaa SSH transiently out of `_step_impl`; none enters prognostic or restart
state.

One actual `_nemo_mlf_step` is run on CPU/fp64 with the established
`zu_frc/zv_frc` hold. At row 6 only, retained pre-correction U/V and retained
oracle primary targets are supplied so the momentum-tail measurement starts
from the already-certified exact local boundary. No Kmm, QCO geometry,
transport, filter input, or filtered result is held. Score the production
cycle's restored Kmm U/V against `atf_dump_{uu,vv}_before.bin`, then score the
returned `u_before/v_before` filtered carry against the corresponding AFTER
dumps.

All four outputs retain the POINTWISE `1e-15` normalized-RMS and maximum/RMS
bars. The skipped-cycle metrics from round 52 must remain outside the bar;
identity passes, point/roll plants fire, the correction is nonzero, hooks run
once and restore, and every retained stream/source/receipt is hash-bound.
Only `MOMENTUM_TAIL_AT_BAR_UPSTREAM_EXACT` releases the tracer tail.
