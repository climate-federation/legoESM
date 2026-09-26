# Preregistration: split-explicit chain round 12 — divergence exactness tie-break

Date: 2026-08-29. Status: **FROZEN BEFORE MEASUREMENT**.

Round 11 classified 9.4 and 9.5 `LITERAL_OWNED`, each with zero error for
NEMO's left-associated metric transport and DEBT for both alternatives. It
stopped at 9.6 because two bit-distinct divergence classes were AT BAR:
NEMO's literal `(du+dv)*r1_e1e2t` and `(du+dv)/e1e2t`. The frozen ambiguity
verdict stands.

This round adds one not-yet-measured discriminator: exact uint64 mismatch
counts against NEMO's dumped `zhdiv` over all 9,920 wet T cells. It rebuilds
both candidates from the same SHA-bound `zhU`, `zhV`, mesh, and restart.

The literal source arm is **EXACT-OWNED** iff it has zero bit mismatches, every
bit-distinct rival that passed the round-11 accumulating bar has at least one
mismatch, and a one-bit oracle plant makes the literal mismatch count nonzero.
If more than one bit-distinct arm has zero mismatches the result is
`AMBIGUOUS`; if the literal has any mismatch it is `OPEN_UNRESOLVED`.

Only `EXACT-OWNED`, together with round 11's two `LITERAL_OWNED` products,
authorizes the minimal NEMO-order production fix registered in round 11. The
unchanged production scorer must still put 9.1--9.7 AT BAR before row 1.4 is
released. No NEMO run/build, new dump, GPU, `mpirun`, or push is authorized.
