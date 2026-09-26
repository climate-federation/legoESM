# Preregistration: literal `un_adv/vn_adv` accumulator, round 59

Date: 2026-08-30. Frozen design; no round-59 measurement has run. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Implement a dedicated `generic` / `nemo_literal` barotropic transport-
accumulation selector. Default `generic` everywhere; select `nemo_literal`
only on `nemo_dino_kamm` and `nemo_dino_kamm_mlf`. The literal path must:

1. retain raw `wgtbtp2` secondary weights and their raw `r1_wgt2s` sum rather
   than pre-normalizing each substep;
2. assemble `zhU=e2u*ua_e*zhup2_e` and V analog in the existing literal metric
   helper's source order;
3. accumulate `(za2*zhU)*r1_e2u` and V analog at every substep; and
4. divide each completed accumulator once by `r1_wgt2s` after the loop.

The contract applies identically to JAX scan, fori-loop, and wide-halo chunked
execution. Generic cards must remain byte-identical. Tests must cover raw
weight/sum identity, source-association arithmetic with a planted cancelled-
metric violation, scan/fori equality, chunked/wide-halo equality, autodiff/JIT,
two-card selector scope, and an unknown-selector red case.

Then rerun the corrected-population unheld and slow-forcing row-8 ladders. The
unchanged target is all 8.3--8.10 subrows at their registered bars; the old
170-column held receipt must remain a red-capable control. Only an all-green
held ladder releases Redi T/S and the complete tracer tail. Existing retained
streams suffice; no NEMO build/run block is registered.
