# NEMO testcase Lane 4 — ORCA2 card round 25 component-isolation receipt

Date: 2026-09-26

Parent: `cae4be7af0c7cf9becbd6f6bac87cdb560af5e2c`

Status: **HELD — LIVE LDF THICKNESS OWNS THE ROUND-24 REFUSAL.**  Of the
three Decision-54 component arms, only the live-thickness arm reproduces the
non-positive raw-`e3w` refusal while entering kt=4.  The single-mask arm is
trajectory-vacuous through kt=10.  The native-F-metric arm moves the
trajectory but remains finite through kt=10.  Every experimental model change
was reverted; the final `packages/` tree is identical to the parent.

All trajectory numbers are **independent with Decision-52 SSH**.  No
given-NEMO-entry operator number is mixed into the tables below.  The six
sea-ice selectors and the card's `unmeasured_features` tuple remain unchanged.

Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round25/`.

## Compiled statements and controlled arms

The admitted build reads its three-dimensional T- and F-point coefficients at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:348-353` and
masks them exactly once at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:387-393`.
The executing level operator consumes the already-masked F coefficient, live
F thickness, F-cell area reciprocal and circulation edges, live T/U/V
thicknesses, and Kmm face divisors at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123-140`.

The existing round-1 ladder gate was reused; no second trajectory instrument
was built.  Each experimental commit starts from the same restored parent
implementation, changes one input boundary, carries a clean worktree stamp,
and is followed by a committed revert before the next arm.  No resolved card
leaf changes in any arm.

| arm | clean commit | isolated input boundary |
|---|---|---|
| single mask | `0dd42f88878c2ea2fcc117687cf35cf2bdf4abdb` | file-read `ahmf` is not multiplied by legoESM's second binary vertex mask |
| live thickness | `66dcc9cac48c9f8d12c581dfa27e38afbb7d8354` | LDF receives the recorded T/U/V/F mesh and live Kbb/Kmm thicknesses; baseline mask and LDF metrics remain |
| native F metrics | `4824ec563fdc46f7a5a532531bc6b8eeddead6b7` | LDF receives stored F-area and native edge reciprocals; baseline mask and thicknesses remain |

## Independent trajectory result

The fresh parent baseline and both finite arms contain 40 checkpoints
(kt=1..10).  The bounded thickness artifact contains the 12 checkpoints
through kt=3, followed by a separate captured run that fails closed while
entering kt=4 with `raw-mesh e3w_int must contain only finite values > 0`.

| arm | checkpoints | moved field rows | first moved row | direction by max error | outcome |
|---|---:|---:|---|---:|---|
| single mask | 40 | **0 / 200** | none | 0 toward / 0 away | `LADDER_MEASURED` through kt=10 |
| live thickness | 12 | **45 / 60 observable** | kt=1 stage-2 U | 15 toward / 30 away | exact step-4 raw-`e3w` refusal |
| native F metrics | 40 | **175 / 200** | kt=2 stage-1 T | 75 toward / 100 away | `LADDER_MEASURED` through kt=10 |

No arm moves a formerly bit-identical row off the bar.  The first non-bit
NEMO statement remains kt=1 stage-1 temperature in every arm.  The
live-thickness arm reproduces round 24's first two moved rows exactly:

| row | parent max error | live-thickness max error | direction |
|---|---:|---:|---|
| kt=1 stage-2 U | `0.06470386947382581` m/s | `0.10080009966621735` m/s | away |
| kt=1 stage-2 V | `0.034012848056840184` m/s | `0.11456680400114852` m/s | away |

The isolation is therefore sharp: the mask change has no trajectory effect
over ten steps; the metric change begins one whole step later and remains
finite; the thickness change alone reproduces the first movement and the
failure boundary.  Round 24's OPEN wording anticipated a cancelling component
inside this three-part attribution.  **RETRACTION:** no such component is
needed to reproduce the refusal.  Live thickness is sufficient; any
compensating error that made the parent finite lies inside the baseline
thickness construction or outside these other two Decision-54 components.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R25-P1 | **REFUTED** | The single-mask arm is trajectory-vacuous: 0 of 200 rows move. |
| R25-P2 | **REFUTED** | Thickness first moves at kt=1 stage-2 U, metrics first move at kt=2 stage-1 T, and the mask arm never moves. |
| R25-P3 | **CONFIRMED** | Thickness completes kt=1..3 and hits the exact registered refusal entering kt=4. |
| R25-P4 | **CONFIRMED** | Single-mask and native-F-metric arms each produce 40 checkpoints and `LADDER_MEASURED`. |
| R25-P5 | **CONFIRMED** | No formerly exact row leaves the bar; every first non-bit statement is unchanged. |
| R25-P6 | **CONFIRMED** | The outcome gate's one-ULP summary plant is refused; the citation plant also fires. |

No prediction was rewritten after measurement.

## Gate, review, and tests

The round-25 outcome gate exits 2 with `HELD`, registers every moved row,
requires the exact step-4 refusal, and names `live_thickness` as the isolated
owner.  Its plant changes an otherwise exact kt=1 entry-T row by one
representable value and exits 1 with `formerly AT-BAR rows left`.

The required `codex exec --sandbox read-only` review was attempted on the
committed round diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

The receipt citation gate passes all three compiled citations with zero
failures and zero unmapped citations.  Its rigid +2-line plant on the rendered
file-read citation exits 1 with `SYMBOL-NOT-AT-LINE`; all nine self-tests fire.
The focused battery reports **17 passed**.

The one required `tests/ocean/fidelity -n 12` battery collected 1,866 tests
and reached the inherited final-tail hang after **1,827 passed, 7 skipped,
5 failed, and 27 unfinished** were visibly emitted.  The idle wrapper was
interrupted after a bounded wait.  Serial reruns reproduce the same five
inherited failures as round 24: the round-129 stepping-gate stamp, the stale
round-51 live-trace suffix assertion, SI3 scalar-math source provenance,
three unstamped legacy report emitters, and the missing `hires_lane_surface`
case-board row.  No round-25 path fails.

No GYRE/DINO/lock-exchange/overflow landing gate is claimed: this round lands
no model statement, and the final `packages/` tree is byte-identical to the
already-certified parent.

## Choices

ASKED: Decision 54 and round 24's OPEN item authorize separate measurement of
the three already-cited inputs.

UNASKED: none.  No configuration value, default, carried state, stabilizer,
scoring rule, sea-ice selector, or NEMO source changed.

## OPEN

1. Decision 54 remains held.  Split the live-thickness bundle next in compiled
   order: F thickness in the curl term; T/U/V Kbb thicknesses in divergence;
   and U/V Kmm outer divisors.  Keep the mask and metric paths at the parent.
2. The compensating error that keeps the parent trajectory finite is not the
   second mask and not the native-F metric path; do not retry the whole landing
   until the thickness subcomponents identify it.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. The independent ORCA2 year still depends on its scheduled independent
   initial-state completion.
6. The seven inherited duplicate citation-map literal keys remain open.
