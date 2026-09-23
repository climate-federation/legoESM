# Split-explicit momentum chain: round 27 result

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Verdict

`PRODUCTION_LITERAL_EEN_BUILDER_REQUIRED`; the ordered walk stops at row 1.3.
The user's admission and the round-25/26 receipts establish the NEMO source
composition and bottom/update identities at bar, but they do not replace the
production replay required by the registered gate. The branch still reaches
the generic AL81 association at
`barotropic_latlon_cgrid.py:821-843`, whereas NEMO materializes the eight
coefficients at `dynspg_ts.F90:1517-1565`.

The first CPU/fp64 replay, from commit `1bd55bc67f9`, used the unchanged
production card. Artifact
`/tmp/dino_split_explicit_momentum_chain_round27_recurrence.json`, SHA-256
`2b6685f986492f3b22b4b2c813dd2da2a2c5cc2b990ce0afe6e7f34daed3125f`,
classified substep-1 U/V DEBT at normalized RMS
`4.182607878559972e-7`/`1.8478649957504146e-7`.

A stopped selector-only implementation replay at commit `87e87a809cd` used
NEMO F-point latitude but retained the generic association. Artifact
`/tmp/dino_split_explicit_momentum_chain_round27_postfix_recurrence.json`,
SHA-256
`74f4bb05827190a9d5bffad0a885374f175d270413a967937e98e95e80bd2b07`,
improved U/V to `1.9256402731006323e-7`/`1.0000704249865956e-7` but remained
DEBT by five orders of magnitude. This confirms that face latitude is only
one component of the exact arm. The partial card change is retracted; no
physics default changes survive this round.

Row-1.4 values printed by both stopped replays are downstream diagnostics and
are inadmissible because row 1.3 failed first. Rows 2--6 (`div_hor`,
`dom_qco_r3c`, `dyn_zdf`, `wzv`, `mlf_baro_corr`) and the free-surface-filter,
momentum-RHS, and tracer-tail chains were not scored. Round 28 preregisters
the complete production literal builder. This is a production-change stop,
not a missing-operand stop: existing streams are sufficient and no SLOT block
is issued.

Both runs used the existing day-180 dumps, checkout-local imports, CPU/fp64,
and tracked-only cleanliness. No NEMO execution, GPU, MPI process, or new
instrumentation occurred.
