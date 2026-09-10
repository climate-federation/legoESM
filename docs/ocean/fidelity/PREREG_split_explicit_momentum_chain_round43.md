# Preregistration: coupled production WZV call-2 implementation, round 43

Date: 2026-08-30. Frozen before implementation measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 42 (`d9dec7def95b9dbe4389452f988ea2ea05087b4830b4af7dd1b90d3a8a441d29`)
proves that call-2 W is a coupled second-`hdiv` x barotropic-Kaa-`r3t`
composition. Each half alone worsens call-delta squared error by roughly
48--50 times; the joint source-literal arm is bit-exact.

## Implementation

Add `wzv_call2_evaluation={generic,nemo_literal}`. `generic` is the historical
path and remains the default everywhere except the two DINO fidelity cards;
those select `nemo_literal`. Literal mode requires the already coupled
`zad_qco_evaluation=nemo_literal` path and is rejected otherwise.

At the production tracer-flux boundary, literal mode uses as one inseparable
operation:

1. the post-barotropic Kmm velocity/transport state;
2. live Kmm QCO `e3u/e3v` and source-ordered second `div_hor`; and
3. the actual barotropic Kaa SSH/r3t already returned by the split-explicit
   solve, not a new first-guess continuity prediction.

The bottom-up recurrence remains `sshwzv.F90:218-227`. The result replaces
only the call-2/tracer W. The call-1 W and face thickness used by dynzad stay
the already certified round-40 operands.

## Gates

The matched day-180 production call-2 capture is scored against the unchanged
`wzv_dump_ww_call2.bin` under the accumulating `1e-12` bar. Target is
`ROW5_WZV_CALL2_AT_BAR`; otherwise the first literal suboperand is the ordered
stop. Two independent CPU captures must be byte-identical and each must be the
unique literal invocation carrying an explicit Kaa override. Identity must be
AT BAR; sign, roll, and `2e-12*RMS` point plants must fail.

Red-capability also binds round 42: H-only and Q-only must remain worse than
the H0Q0 call-1 baseline. Unit tests cover the literal recurrence, Kaa override,
unknown selector, dependency rejection, scoped defaults, JIT, and gradient.
Non-DINO recipes are byte-pinned by comparing generic/default configurations.

Only an AT-BAR production receipt promotes row 5 and releases row 6. No
climate or later-chain conclusion is read before that gate.
