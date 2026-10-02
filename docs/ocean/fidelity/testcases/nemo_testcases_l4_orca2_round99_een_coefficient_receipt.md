# ORCA2 round 99 — rung-0 frozen EEN coefficient discriminator

Date: 2026-10-02. Base `0b9e9d07b3`; measurement tip `8eac28bad`.
Scope is ocean only. Every model number below is labelled **independent**:
hierarchy rung 0 starts from NEMO's own from-rest state.

## Verdict

**HELD.** The operator's completed round-98 record is sound. Its checker had
incorrectly required each coefficient group's dimensions to equal the separate
haloed local-domain dimensions. The repaired checker reads the groups' own
self-describing dimensions and verifies that they equal the owned bounds. Both
90x148 rank slabs admit, cover the 148x180 global domain exactly once, and all
20 terminal ocean restart shards are byte-identical to round 96. No NEMO rerun
is required.

The admitted record resolves the round-98 discriminator. The first non-bit
boundary is coefficient construction, not the subsequent four-product
application: all eight legoESM coefficients differ from the compiled NEMO
coefficients. The first field, `ffu_nw`, differs at 3,515 cells solely by the
sign of zero. Replacing only coefficient zero signs from the NEMO record makes
both substep-1 U and V Coriolis outputs bit-exact. The exact accumulation or
final-scale operation producing those signs is not yet isolated, so no model
statement lands.

The northern-fold magnitude difference is separate: only `ffv_nw` and
`ffv_ne` differ in magnitude, at 66 and 67 cells respectively, all on the last
row. Replacing the fold row alone moves neither of the first two application
rows. The later substep-2 U residual remains 68 cells with maximum
`2.9617669311254642e-8` at `(j=147,i=134)` and is not attributed here.

No `packages/` file, card field, configuration choice, stabilizer, threshold,
carried-state convention, sea-ice selector, or `unmeasured_features` entry
changes. The GYRE trajectory is unchanged by construction.

## Record admission

The compiled solver allocates the eight coefficient arrays over the owned
`Nis0:Nie0,Njs0:Nje0` bounds, not over the haloed `jpi,jpj` domain, in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:126-127`. The writer
therefore correctly declared 90x148 groups inside files whose local-domain
header is 94x152. The checker now validates that header relationship instead
of replacing the self-described group contract with the domain shape.

| Admission item | Independent result |
|---|---:|
| rank records | 2 |
| bytes per record | 852,816 |
| fields per rank | 8 |
| group shape | 90x148 |
| rank coverage | exactly once over 148x180 |
| rank-0 SHA-256 | `d79f10bc2e55de5a99288b2950ea5524f1e028be0c409bd69883fcf43f39e28f` |
| rank-1 SHA-256 | `6c78b2c49d2d7726615742c31429b57f492c42fc0b99c5bbf1c11b831929e096` |
| terminal restarts | 20/20 byte-identical |

The original producer manifest remains immutable. The repaired launcher pins
the recorded old launcher and checker digests explicitly, while continuing to
content-verify the unchanged writer, patch, preregistration, and pre-patch
compiled source. All eight record/restart plants and the producer-content plant
fire; `--admit-existing` ends with
`ORCA2_ROUND98_EEN_COEFF_ACQUISITION_PASS`.

## Source-ordered statement

NEMO initializes, accumulates, and scales the eight EEN coefficients in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1213-1265`, then consumes
them in its written four-U/four-V product and pairwise-sum order at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1369-1392`.

Round 98's implication that the strict application itself might create the
795 substep-1 signed-zero differences is **RETRACTED**. With the recorded NEMO
coefficients and the same bit-exact midpoint velocities, that application is
bit-exact for both faces. The mismatch is already present in the frozen
coefficients before the application runs.

| Coefficient | Bit-unequal | Signed-zero only | Magnitude-unequal | Magnitude support |
|---|---:|---:|---:|---|
| `ffu_nw` | 3,515 | 3,515 | 0 | none |
| `ffu_ne` | 3,514 | 3,514 | 0 | none |
| `ffu_sw` | 3,577 | 3,577 | 0 | none |
| `ffu_se` | 3,574 | 3,574 | 0 | none |
| `ffv_sw` | 3,584 | 3,584 | 0 | none |
| `ffv_se` | 3,585 | 3,585 | 0 | none |
| `ffv_nw` | 3,570 | 3,504 | 66 | northern-fold row only |
| `ffv_ne` | 3,572 | 3,505 | 67 | northern-fold row only |

| Coefficient arm | Substep-1 U | Substep-1 V | Substep-2 U | Substep-2 V |
|---|---:|---:|---:|---:|
| legoESM source builder | 398 signed-zero | 397 signed-zero | 68, max `2.962e-8` | bit |
| NEMO zero signs only | bit | bit | 68, max `2.962e-8` | bit |
| NEMO fold row only | 398 signed-zero | 397 signed-zero | 68, max `2.962e-8` | bit |
| all NEMO coefficients | bit | bit | 68, max `2.962e-8` | bit |

This confirms R99-P1 through R99-P3. R99-P4 selects
`COEFFICIENT_CONSTRUCTION`; it does not yet select one arithmetic operation
inside the vertical accumulation and final scale. The measured artifact is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round99/een_coefficient_discriminator_final.json`.

## Controls, tests, and review

The dimension plant mutates the first group's self-declared shape and is
rejected against the owned bounds. Header, name, truncation, missing-field,
all-zero-payload, swapped-rank, and restart-byte plants also fire during
admission. The one-ULP coefficient and application plants fire on the measured
path.

Focused parser, assembly, application, launcher, control, and citation tests
pass 42/42.
The single `tests/ocean/fidelity -n 12` battery reached 99%, reproduced the
same six established failures as rounds 97-98, and lost its pytest process
without a terminal summary. It is recorded as **incomplete**, not PASS.

The required separate read-only Codex review did not reach the diff:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. Walk the source-ordered zero sign from the positive-zero initialization
   through the per-level accumulation and final coefficient scale; land only
   a source-derived operation that makes all eight coefficients bit-exact.
2. Independently reproduce the northern-fold `ffv_nw`/`ffv_ne` halo operands;
   their 66/67 magnitude differences do not own the first two application
   rows but remain coefficient debt.
3. After the coefficient row is exact, walk the separate 68-cell substep-2 U
   association at the fold, then advance to drag and the velocity update.
4. The rung-0 given-entry ladder, independent ladder, and month remain the
   subsequent hierarchy deliverables.

## UNVERIFIED

- The exact coefficient accumulation or final-scale operation that creates
  the zero-sign mismatch is not yet named.
- The later 68-cell substep-2 U residual remains unowned.
- No model statement or candidate ten-step ladder is proposed this round.
- No unasked scientific or configuration choice was made.
