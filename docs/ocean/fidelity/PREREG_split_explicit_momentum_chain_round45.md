# Preregistration: literal call-2 Kaa/Kmm operands, round 45

Date: 2026-08-30. Frozen before implementation and measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 44 (`b2b38638bbd4e1e0aaf3f30e67ccec3998addb9af6057f9e2e12b53e76c2b515`)
localizes the first call-2 failure to Kaa `r3t` and also bounds the following
Kmm `hdiv` as DEBT. The production amendment is one source-ordered operation:

1. retain the raw boxcar Kaa SSH returned by the split-explicit solver before
   legoESM's model-only global eta-drift projection, matching
   `dynspg_ts.F90:991,1003`;
2. reconstruct the live Kmm velocity as
   `u(Kmm)+un_adv*r1_hu(Kmm)-puu_b(Kmm)` (V analog), with QCO live face
   thickness, separately associated reciprocal depth, and source-left
   vertical transport reduction, matching `dynspg_ts.F90:1170-1174`;
3. feed that Kmm velocity and the raw Kaa SSH jointly to the already literal
   `div_hor`/WZV recurrence (`stpmlf.F90:350-412`,
   `sshwzv.F90:218-227`).

The selector remains `wzv_call2_evaluation="nemo_literal"`, faithful by
default only on the two DINO cards and generic elsewhere. The implementation
must be JIT/autodiff-safe. The old post-projection SSH and old after/tracer
velocity are planted violations and must each fail their direct operand bar.
The corrected Q and H operands must each have normalized RMS and maximum error
over NEMO RMS no larger than `1e-12`; the resulting row-5 W field uses the
same accumulating `1e-12` bar. Identity, roll, unique-capture, hook-restoration,
finite, and full cited-interior shape controls remain mandatory.

Disposition is `ROW5_WZV_CALL2_AT_BAR` only if Q, H, and W all meet those
bars and both old-operand plants fire. Otherwise it is the first failure in
Q, H, W order. Row 6 remains ordered-blocked until that joint receipt passes.
