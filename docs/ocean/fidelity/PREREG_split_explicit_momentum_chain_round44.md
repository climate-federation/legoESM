# Preregistration: row-5 literal call-2 operand capture, round 44

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 43 (`7109a7c0eae5684b778452cc5b07e4e1397a4b88849f37d448c515c451494eec`)
shows that the literal recurrence with actual Kaa override remains at
normalized RMS `0.1712884891`. This round captures the exact arguments of the
unique production call-2 helper and scores its two materialized inputs against
existing NEMO streams:

1. **Q**: `eta_after_override * r1_ht_0` versus
   `seq_dump_r3t_aaa_kt00005761.bin` (`domqco.F90:159-161`);
2. **H**: source-ordered live-Kmm transport divergence divided by live Kmm
   `e3t` versus `seq_dump_hdiv_nnn_kt00005761.bin`
   (`divhor.F90:179-184`).

Both use the raw mesh metrics and masks already bound by rounds 40--42. The
input capture must be unique, hooks restored, shapes cited, and all fields
finite. Identity is the zero-error control and a one-cell roll must fail.
The direct operand bar is normalized RMS `1e-12` with maximum normalized by
oracle RMS also no larger than `1e-12`.

Disposition is the first failing source operand in Q then H order:
`CALL2_KAA_R3T_DIVERGED`, `CALL2_HDIV_DIVERGED`, or
`CALL2_OPERANDS_AT_BAR_RECURRENCE_DIVERGED`. If both are at bar, only recurrence
association remains. No fix is built from this capture until its disposition
is known.
