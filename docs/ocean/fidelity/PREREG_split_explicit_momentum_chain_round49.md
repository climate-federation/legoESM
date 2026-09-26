# Preregistration: executed-live row-6 association, round 49

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 48 (`8055e1138c7784942167ae06212e2801d78f6690d3646351b6040d557a70c06b`)
shows that the oracle primary target removes 82.2% of row-6 squared error but
the algebraically cancelled reference-ladder association is inert. This
round fixes T1 and compares two source associations offline from existing
dumps:

* C0: `e3*_0` and `r1_h*_0` after algebraic QCO cancellation (round 48 A1);
* C1: executed `stpmlf.F90:744-755`: form
  `e3*(Kaa)=e3*_0*(1+r3*(Kaa)*mask)`, source-left accumulate its transport,
  independently form `r1_h*(Kaa)=r1_h*_0/(1+r3*(Kaa))`, then apply
  `before - transport*r1_h(Kaa) + target` in literal order.

Kaa `r3u/r3v` come from existing full-halo
`seq_dump_r3{u,v}_aaa_kt00005761.bin`; target, before, after, mesh, and
reference geometry remain the round-48 bindings. The POINTWISE `1e-15` RMS
and maximum/RMS bars are unchanged. Identity, roll, wet-point, nonzero-live-
factor, shape, finite, and round-48 C0 reproduction controls must pass.

If C1 is at bar for U and V, disposition is
`ROW6_LOCALIZED_TO_LIVE_QCO_ASSOCIATION_GIVEN_ORACLE_TARGET`; otherwise row 6
remains `OPEN_UNRESOLVED`. A production change is permitted only on the former
receipt and must preserve generic/off cards byte-for-byte with red tests.
