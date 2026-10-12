# ORCA2 round 261 — stage-2 V-transport operand split

Date: 2026-10-12. Base: `0aee6bedb`. Preregistration commit:
`83cbb8ed8`. Measurement-gate commits: `ab5b250bb`, `c603dbb7b`,
`356a79943`, and `c217e3041`. Private-arm removal commit: `1dd4c425b`.
Disposition: **HELD**.

This round changes no production model, card, deck, selector, carried-state
contract, stabiliser, sea-ice declaration, or `unmeasured_features` entry at
its tip. It measures OMT-4 separately as **independent** and **given NEMO's
entry**, on CPU fp64/libm production JIT. No NEMO acquisition was needed.

## Compiled statements

At stage 1 NEMO computes the V-point free-surface ratio in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:213-217` and then
associates that V scalar across the northern fold in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:168-170`. At stage
2 it forms the barotropic correction in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279` and writes
the V transport, in source order, as `e1v`, live `e3v(Kmm)`, `vv(Kmm)`,
`zvb`, and `vmask` in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`.

The source-exact oracle replay uses the admitted stage-1 SSH, D114's raw
NEMO `e3v_0`, `hv_0`, metrics and masks, the compiled `domqco` arithmetic,
the compiled V-fold copy, and the admitted barotropic-average streams. It
reproduces NEMO's recorded stage-2 `zFv` bit-for-bit. The candidate replay
also reproduces its passive executable output bit-for-bit. These two controls
prevent an inferred or wrong-time-level operand from owning the split.

## Frozen predictions and verdicts

R261-P1 and R261-P2 are **CONFIRMED** under both labels. The round reproduces
round 260's `zFv` boundary exactly: 849/8,100 fold-band slots differ, maximum
`0.7981690085143782` m3/s at rank-local `[j=147,i=41,k=25]`. Candidate and
oracle source replays each have 0 unequal slots, and the passive live trace is
array-identical to the ordinary completed state for every carried field.

R261-P3 is **REFUTED**. The first valued operand is not `zvb`; it is live
`e3v`:

| source-ordered operand | unequal/support | maximum absolute error |
|---|---:|---:|
| `e1v` | 0/100 | 0.0 m |
| live `e3v(Kmm)` | 668/1,977 | 0.007467786710066093 m |
| associated `vv(Kmm)` | 0/1,977 | 0.0 m/s |
| `zvb` | 35/100 | 9.30045701726806e-7 m/s |
| resolved `vmask` | 0/8,100 | 0.0 |

The `e3v` maximum is at `[j=147,i=39,k=19]`, on the fold pivot row. The
`zvb` split names its first internal difference as live `r1_hv`: 35/100
faces, maximum `4.002598942174848e-6` 1/m; `vn_adv` and `vv_b` are exact.
Thus the compiled stage-1 V-fold association of `r3va` is the shared upstream
statement: it feeds both stage-2 live thickness and the reciprocal used by
`zvb`.

R261-P4 reports a **CANCELLING UNIT**, not a single-operand landing. Replacing
`e3v` alone leaves 849 unequal slots and makes the maximum transport error
worse, from `0.7981690085143782` to `5.656858628935879` m3/s. Replacing the
source-ordered prefix through associated `vv` leaves 668 unequal slots at the
same `5.656858628935879` m3/s. Adding the recorded-source `zvb` closes the
transport exactly: 0/8,100 unequal and maximum 0.0 under both labels.
`vmask` is already exact. A partial `e3v` transcription would therefore split
a measured cancelling pair and is prohibited.

R261-P5 and R261-P6 are **NOT REACHED**. The upstream V-ratio fold statement
was named, but its complete production integration and Decision-96 trajectory
census belong to the next round; the batch explicitly ends after round 261.
Nothing lands here. R261-P7 is **CONFIRMED**: all nine planted violations
refuse (label coverage, source order, record identity, candidate replay,
oracle replay, round-260 reproduction, first owner, substitution closure, and
one-bit sensitivity).

## Evidence and validation

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round261/`.
The principal SHA-256 values are:

- independent operand split:
  `b67afd600e4ffc1f113dc45645b094fff0e23e71925a641d6ee183aa1a98f5b2`;
- given-NEMO-entry operand split:
  `bddb3a31c95d70b5ae24d94c97687fb939b1e748e7199a8309feb922a28bca64`;
- two-label classification:
  `8417f69668e93fb2a3df36a1bea38bf5290ac1452753f92e72fb75292854cf8e`.

The first two measurement attempts are retained as instrument retractions:
one rejected a valid rank-local 90-column slab, and the next supplied a
candidate-derived mask/time level to the oracle replay. Neither result is
cited. The final source-exact oracle replay has 0 unequal slots, and every
number above comes from the corrected committed instrument.

The package candidate and its temporary citation re-anchor were removed
before final validation. The final model file and cumulative citation files
are byte-identical to the base, so this HELD round cannot move OMT, rung 0,
rung 10, GYRE, DINO, or tank trajectories.

The requested separate `codex exec --sandbox read-only` review failed before
review because its in-process app-server could not initialize on the
read-only filesystem. **Independent review unavailable in-sandbox.** Its log
SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

Integrate NEMO's stage-1 V-point fold association of `r3va` at the compiled
call site so that the stage-2 live `e3v` and `r1_hv`/`zvb` move together.
Re-run the source replay first, then the complete private unit through OMT-4
kt=8..10 and the full Decision-96 gates. Do not land either half separately.

ASKED choices: none. Unasked configuration choices made: none. Round
disposition: **HELD**.
