# Preregistration — ORCA2 round 101 GYRE year merge-owner bisect

Date: 2026-10-02. Base: `3c8f2095d47d05da77b3988246607b37bf3c5519`.
Scope is diagnosis of round 100's red GYRE year predicate only. No model,
card, deck, sea-ice selector, carried state, threshold, stabilizer, or
`unmeasured_features` entry changes in this round.

The controlled endpoints are the live GYRE source tip
`d7de69d51f6392e77085aabd879ab9e0da23ebf5` and round 100's combined tree
`23a9911ae48f80ef4c40e4639373b06d5bd42ad8`. The committed year harness and
phase-3 gate are byte-identical between those endpoints. Round 100 measured
daily snapshots identical through day 16 and different from day 17 onward.
The discriminator therefore runs the same CPU/fp64/libm seed-0 protocol for
17 days with daily snapshots and changes only selected ocean-package files
between the two endpoint versions.

Every candidate records both endpoint SHAs, the exact selected file list,
and SHA-256 for every overlaid file. `SOURCE` means the selected files come
from the live GYRE tip while all unselected files stay at the combined tree.
**Pre-measurement correction (instrument control):** the first synthetic
signed-zero control refuted `np.array_equal` as a sufficient bit predicate:
NumPy deliberately treats `+0.0 == -0.0`. The frozen predicate is therefore
`np.array_equal` on values **and** `np.array_equal` on the arrays viewed as
same-width unsigned integers, over every array in every day-1..17 snapshot
against the certified live-source member. Printed norms are never the
predicate. This correction was committed before any candidate model run.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R101-P1 | The round-100 movement is owned by ocean-package content, not the unchanged year harness or run protocol. | The no-overlay combined control reproduces round 100: days 1..16 exact and day 17 non-bit. Replacing all 15 differing ocean files by the live-source versions makes all day-1..17 snapshots array-identical to the certified source member. | Either endpoint control misses its registered answer: instrument **REFUTED**; stop without a bisect claim. |
| R101-P2 | One coherent file group contains the first owner. | A half-split produces one SOURCE overlay that is exact through day 17 and its complement that still differs; recurse only into the exact-restoring half. | Both halves restore, neither restores, or a candidate does not construct: interaction/ABI confound; report the smallest unresolved group, no single-owner claim. |
| R101-P3 | The final file contains a source-visible executable statement introduced on the ORCA side after the common ancestor. | One-file SOURCE replacement restores all 17 snapshots; its combined-versus-source hunks map by `git blame` to one or more ORCA commits, and the earliest restoring hunk/commit is named with its existing receipt claim. | One file restores but no one-hunk/commit discriminator is constructible: name the file owner only and keep the statement **UNMEASURED**. |
| R101-P4 | The ten-step ladder blindness is reproducible rather than assumed. | The named owner's combined and source endpoints retain round 100's identical 70-row residual archive while the daily full-state predicate diverges at day 17. | The ladder moves under the controlled endpoint comparison: reconcile round 100 before recording the owner. |

## Landing bar

This is a measurement-only round. It is HELD even if an owner is named: the
merge still requires a NEMO-cited repair plus the rung-7 and rung-0 gates from
round 100. The receipt must retain failed predictions, quote the first
non-bit day/field/cell for the final discriminator, pass the citation gate
with a firing plant, run focused tests, and include a separate read-only Codex
review verdict. No follow-on source-ordered barotropic statement lands here.
