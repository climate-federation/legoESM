# Round 10 codex-internal source-fidelity review

Reviewer identity: `codex-internal/source-fidelity`
Reviewed commit: `6b33449fc758e0afa19fdff79813cee9e5857973`
Verdict: **SHIP**

## Checks

- Unregistered runtimes now receive
  `friction_association_probe.bit_identical = null` and
  `bit_identity_status = WITHHELD_RUNTIME`; registered-runtime claims remain
  `VALID`.
- The friction plant still fails non-vacuously through the legacy
  association/missing-mask arm.
- The complete Stage-1 census remains 271,560 comparisons: 271,542
  bit-identical, 18 `exp`-owned residuals, and zero over-bar.
- Literal `icesbc.F90:337-338` friction grouping and mask remain intact.
- External artifact SHA-256 matches the receipt:
  `a05b4b90b0b2346995430183cf5162319d94b72fe53346d551b445235f03d4d4`.
- Focused gate suite: `13 passed in 21.75s`.

UNVERIFIED: The reviewer did not rerun the full combined suite or rebuild the
oracles.
