# Preregistration: direct Kmm tracer-entry score, round 63

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 62 proves the prior 104-column row-8.3 residual is an instrument-created
proxy error, not model debt:

- production `Hu_avg` is at bar (maximum normalized error
  `2.3895446809e-16`);
- the substituted transport consumed by the first Kmm cycle is bit-exact; and
- that cycle's corrected 3-D U is bit-exact against
  `fct_entry_dump_un.bin`.

The old scorer instead multiplied this exact velocity by live face thickness
to form mass flux and divided by thickness again to reconstruct velocity. That
algebraic round trip created 104 red columns / `1.4948039082e-15`.

Amend the scorer to take row 8.3 directly from the captured first
`nemo_qco_kmm_velocity_cycle` output. Retain the old mass-flux/division proxy
as a planted violation that must reproduce the admitted 104-column red row.
Then score rows 8.3--8.10 sequentially at their unchanged bars. Direct row
8.3 must be bit-exact; otherwise stop. Stop at the first later failure. If all
pass, release Redi temperature. All round-61/62 transport, hook-count,
population, identity, point, roll, and sign controls remain required.

Frozen round-62 receipt SHA-256:
`290caa5bb3c3187d5ea13fe62276f299bd47953b556615dfab3f4443d1597a86`.
