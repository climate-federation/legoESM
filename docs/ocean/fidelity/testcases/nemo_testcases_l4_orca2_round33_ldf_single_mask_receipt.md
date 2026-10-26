# NEMO testcase Lane 4 — ORCA2 card round 33 LDF single-mask landing

Date: 2026-09-26

Parent: `94522f39a4f909eb963340ee1c72a242aa525be0`

Landing commit: `97a0b12bc`

Status: **LANDED — THE FILE-READ F-VISCOSITY COEFFICIENT IS MASKED ONCE.**
The admitted ORCA2 coefficient already contains NEMO's mask, so the production
path no longer applies legoESM's generic vertex mask a second time.  The
given-entry lateral-diffusion replay improves by four orders of magnitude but
is still non-bit; its next owner remains OPEN.  The independent ORCA2 ladder
completes all 40 checkpoints without losing an exact/AT-BAR row, and GYRE is
byte-identical through ten steps and 30 days.

Direct-operator results below are **given NEMO's entry (kt=2 recorded
state)**.  Trajectory results are **independent with Decision-52 SSH**.  The
two labels are not mixed in one table.  The six sea-ice selectors and the
card's `unmeasured_features` tuple are unchanged.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round33/`.

## Compiled statement and landed change

The executing NEMO build reads `ahmt_3d` and `ahmf_3d` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:348-353`, then
masks each stored coefficient exactly once at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:387-393`.
The executing level operator consumes that already-masked `ahmf` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123`.

The legoESM card's carried coefficient is bit-identical to the admitted
record: 0 / 809,070 cells differ.  Applying the old second binary mask changes
48,505 / 809,070 cells, with maximum 80,000 m2/s.  The landing therefore skips
only the generic coefficient-mask multiplication when the coefficient source
is `nemo_ahm_3d_file`; formula-generated coefficients retain the multiplication.
The unit control proves both arms independently.

## Given-NEMO-entry operand walk

The parent replay and landed replay are:

| field | parent unequal / scored | parent max, m/s2 | landed unequal / scored | landed max, m/s2 |
|---|---:|---:|---:|---:|
| U | 411,736 / 411,736 | `2.2089745065738572e-05` | 410,556 / 411,736 | `3.6514518328784234e-09` |
| V | 412,485 / 412,537 | `1.7978718510645183e-05` | 411,162 / 412,537 | `3.4245703200749253e-09` |

This parent result reproduces round 32's final six-reciprocal reconstruction,
not that receipt's separately named canonical legacy probe.  The instruments
have different V score details; no result is silently substituted for the
other.  R33-P2 is **REFUTED** because the landed replay is not bit-exact.

One-variable substitutions for the Kbb face thickness, Kmm face thickness,
and both together are exactly tendency-inert: all three reproduce the landed
row above.  They are therefore not the next owner in this replay.  Substituting
NEMO's carried `hf_0` into the F-column ratio reduces, but does not close, the
remaining row:

| field | unequal / scored | max, m/s2 | L2 ratio |
|---|---:|---:|---:|
| U | 410,460 / 411,736 | `3.181628207426175e-09` | `1.1072530302454793e-04` |
| V | 407,570 / 412,537 | `2.9702048395431957e-09` | `8.952453804424378e-05` |

That is the registered secondary owner, measured but not landed.  The
operator plant changes the carried coefficient and exits nonzero.

## Independent ORCA2 ladder

The parent and landed runs each complete kt=1..10, 40 checkpoints.  The first
non-bit statement remains kt=1 stage-1 temperature.  Of 200 scored rows, 175
move: 119 maxima move toward NEMO and 56 move away; no exact/AT-BAR row leaves
its bar.  The first movement is kt=2 stage-1 temperature.  Every moved row is
registered in `orca_outcome.json`.

| kt=10 stage-3 field | parent max | landed max | direction |
|---|---:|---:|---|
| U, m/s | `0.4230544199344075` | `0.33860877536933787` | toward |
| V, m/s | `0.6838675521949865` | `0.5353439568885162` | toward |

Thus R33-P3 is **REFUTED**: the repair is not trajectory-vacuous.  Its planted
exact-row loss identifies kt=1 entry temperature and exits nonzero.

## Independent GYRE safety gates

The base `94522f39a4` and tip `6965a047a` ten-step ladders compare exactly:
all 70 certified rows have zero field movement and zero oracle-residual
worsening, and all 210 residual arrays are `np.array_equal`.  Both residual
archives have SHA-256
`43f37831256832c31f9a983352d8949e9947040faf01d5f4f39751dc1a70935c`.

The separately run base and tip 30-day members have 0 / 30 differing daily
snapshots.  Their day-30 snapshot SHA-256 is
`a66143733bcc9e4efaa22fe2b3a831629e4d57ffd701ddd827d5553c3e0b007a`.
This confirms R33-P4 by measurement.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R33-P1 | **CONFIRMED** | Carried `ahmf` is 0 / 809,070 unequal; the second mask moves 48,505 cells and is the first differing live operand. |
| R33-P2 | **REFUTED** | The corrected replay retains `3.65e-09` / `3.42e-09` m/s2 maxima. |
| R33-P3 | **REFUTED** | 175 / 200 trajectory rows move, although no exact/AT-BAR row leaves its bar. |
| R33-P4 | **CONFIRMED** | GYRE has 70 / 70 unchanged rows, 210 / 210 exact residual arrays, and 30 / 30 exact snapshots. |
| R33-P5 | **CONFIRMED** | The shared-card, focused, citation and outcome controls pass; each planted violation fires. |

Failed predictions R33-P2 and R33-P3 are retained exactly as preregistered.

## Gates, review, and tests

The exact shared-card inventory is green: 160 DINO/rule-12/lock/overflow
tests pass with nine warnings, and the remaining tank round-34 file passes
10/10, for **170/170**.  The focused round-33 unit, operand, outcome and
citation suites pass **38/38 in 25.53 s**.

The required separate `codex exec --sandbox read-only` review was attempted
against the committed diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.  The failure log is
preserved as `codex_readonly_review.log`.

The receipt citation gate passes all three compiled citations with zero
failures, zero unmapped citations and zero map-audit failures.  Its rigid
line-shift plant exits nonzero with `SYMBOL-NOT-AT-LINE`.

The required single `tests/ocean/fidelity -n 12` battery collected 1,891
tests.  It reached the inherited final-tail stall at 98% and was interrupted
after a bounded idle wait, having emitted **1,868 passed, 7 skipped, 5 failed
and 11 unfinished**.  The five failures were rerun by exact ID in isolation
and reproduce the inherited SI3 MY_SRC provenance mismatch, round-129
worktree-stamp mismatch, stale round-51 trace suffix, three legacy report
emitters without stamps, and missing `hires_lane_surface` case-board row.
No round-33 test fails.

The final ORCA2 push battery runs the citation gate, TKE terms, recipe,
freshwater closure and parallel-receipt gate against the committed receipt:
**127/127 pass in 371.12 s**; its post-closing-commit repeat is also
**127/127 in 422.59 s**.

## Choices

ASKED: Decision 54 and round 32's OPEN section authorize walking and landing
the first NEMO-cited live lateral-diffusion operand.

UNASKED: none.  No configuration value, stabilizer, NEMO source, carried
state, sea-ice selector, score or acquisition changed.

## OPEN

1. The given-NEMO-entry lateral-diffusion replay still has
   `3.6514518328784234e-09` / `3.4245703200749253e-09` m/s2 maxima.  Continue
   the compiled-order one-variable walk from the remaining live r3f/metric
   operands; do not reuse the refuted bit-exact prediction.
2. NEMO's carried `hf_0` is a registered secondary owner which reduces but
   does not close that replay.  The Kbb/Kmm face-thickness substitutions are
   directly measured as tendency-inert.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. Decision 52's independent ORCA2 initial-state transcription and year
   comparison remain owed.
6. The inherited duplicate citation-map literal keys remain open.
