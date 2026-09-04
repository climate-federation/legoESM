# Round 10 codex-internal source-fidelity review

Reviewer identity: `codex-internal/source-fidelity`
Reviewed commit: `b57abb598e9f9f6486172541a7505bfc5d715ce6`
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
- The post-review delta contains documentation and the folded-constant replay
  patch only.  The canonical and original patches produce byte-identical
  Fortran: `sbc_phy.F90`
  `dd531af858fff69275493db1e3a90a1298a2ec71afa047ecfb3091bb9c27275a`
  and `sbcblk.F90`
  `af906e92757acf4a0d2075e7d5d3caac92580383176ae8b01db25c7d93dbc382`.
- The canonical patch applies to the shipped sources, its SHA-256 is the
  receipt's
  `bd2fe0b51e7516ac1df7736e374e793eed47e1ccc04541a0b0a3b598dd66635e`,
  and `git diff --check` is clean.

UNVERIFIED: The reviewer did not rerun tests or rebuild the oracles for the
provenance-only post-review delta.
