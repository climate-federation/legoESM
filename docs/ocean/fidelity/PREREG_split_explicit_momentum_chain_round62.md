# Preregistration: Kmm-cycle input capture, round 62

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 61 replaced the completed solver transport once but left row 8.3
unchanged at 104 red columns / `1.4948039082e-15`. Instrument the actual first
production `nemo_qco_kmm_velocity_cycle` call during the same held-forcing,
oracle-transport arm. Capture:

1. the production `Hu_avg/Hv_avg` returned by the solver before replacement;
2. the `un_adv/vn_adv` arguments received by the first Kmm cycle after
   replacement; and
3. that call's corrected U/V outputs.

Compare production and consumed U transport with retained NEMO `un_adv` over
the real wet surface-face population, and corrected U with retained 3-D
`fct_entry_dump_un.bin` over the registered 9,758-column umask. The unchanged
pointwise bar is `1e-15`.

If production Hu is at bar and identical to the consumed oracle arm while the
corrected output remains red, exonerate the accumulator fully and move the
owner into another operand of the literal Kmm composition. If production Hu
is red but consumed Hu is at bar and the corrected output remains unchanged,
the substitution plumbing is invalid and the chain stops. If consumed Hu is
not at bar, the layout/substitution is invalid. Require exactly one solver
replacement, at least one captured cycle, hook restoration, and the admitted
round-61 null receipt. No NEMO build/run is needed.

Frozen round-61 receipt SHA-256:
`b8f7a376a0a13fd384cb4195cf27db8ff6f82c71e576bb8d449451903d3bd07c`.
