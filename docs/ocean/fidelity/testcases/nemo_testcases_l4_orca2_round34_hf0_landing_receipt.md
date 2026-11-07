# NEMO testcase Lane 4 — ORCA2 card round 34 carried F-column landing

Date: 2026-09-26

Parent: `7de81830940c672714cd51e091ffac74cfb3ca23`

Implementation commit: `4ae4b9700ff68f624b5763918a01375c5aa1bb70`

Final tested implementation/citation tip: `13c2927a844570fd0f4892a9ee8ed6946d6c67bb`

Status: **LANDED — USE NEMO'S CARRIED F-COLUMN DEPTH IN `r3f`.**  The card's
carried `hf_0` is bit-identical to a literal transcription of NEMO's compiled
mesh construction in all 26,640 cells.  The given-entry lateral-diffusion
replay remains non-bit, so the next compiled-order owner remains OPEN.  The
independent ORCA2 ladder completes kt=1..10 without losing an exact/AT-BAR
row.  GYRE is byte-identical through ten steps and 30 days.

Direct-operator results below are **given NEMO's entry (kt=2 recorded
state)**.  Trajectory results are **independent with Decision-52 SSH**.  The
two labels are not mixed in one table.  The six sea-ice selectors and the
card's `unmeasured_features` tuple are unchanged.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round34/`.

## Compiled statement and landed change

The executing NEMO build constructs `hf_0` from mesh `e3f_3d` and the two
adjacent V masks, then performs its F-point halo exchange, at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domain.f90:203-206`.
The executing QCO routine consumes the stored reciprocal in the four-cell
area-weighted `r3f` statement at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`.
The lateral-diffusion consumer applies that live `r3f` to the mesh F
thickness at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123`.

Before this round, legoESM reconstructed the F-column depth by summing the
distinct vorticity-reference thickness `e3f_0vor` under `fe3mask`.  The
landing selects the card's already-carried `hf_0` when present and preserves
that reconstruction as the fallback for cards without the operand.  It
changes no card, selector, forcing, score, or NEMO source.

## Given-NEMO-entry operand and replay results

The card's carried depth is 0 / 26,640 unequal to an independent literal
construction from the admitted mesh operands.  The displaced reconstruction
is 14,324 / 26,640 unequal, with maximum absolute difference
`651.239260895305` m.  Thus R34-P1 is **CONFIRMED**.

The production replay reproduces round 33's registered secondary-owner
result exactly and remains debt:

| field | unequal / scored | maximum, m/s2 | L2 ratio |
|---|---:|---:|---:|
| U | 410,460 / 411,736 | `3.181628207426175e-09` | `1.1072530302454793e-04` |
| V | 407,570 / 412,537 | `2.9702048395431957e-09` | `8.952453804424378e-05` |

R34-P2 is **CONFIRMED**: neither field is bit-exact, the frozen maxima and
score sets are reproduced, and the operand plant exits nonzero.

## Independent ORCA2 ladder

The parent and landed runs each complete kt=1..10, 40 checkpoints.  The first
non-bit statement remains kt=1 stage-1 temperature.  Of 200 scored rows, 185
move: 77 maxima move toward NEMO, 107 move away, and one keeps the same
maximum; no exact/AT-BAR row leaves its bar.  The first movement is kt=1
stage-2 U.  Every moved row is registered in `orca_outcome.json`.

| kt=10 stage-3 field | parent maximum | landed maximum | direction |
|---|---:|---:|---|
| U, m/s | `0.33860877536933787` | `0.33900405135003436` | away |
| V, m/s | `0.5353439568885162` | `0.5353447182593362` | away |

R34-P3 is **CONFIRMED**.  The faithful operand repair is not presented as a
trajectory improvement: the maximum errors above move slightly away.  Its
planted exact-row loss exits nonzero.

## Independent GYRE safety gates

The exact base `7de81830940c672714cd51e091ffac74cfb3ca23` and final tested tip
`13c2927a844570fd0f4892a9ee8ed6946d6c67bb` ten-step ladders compare
exactly.  All 70 certified rows have zero field movement and zero
oracle-residual worsening; all 210 residual arrays are `np.array_equal`.
Both residual archives have SHA-256
`43f37831256832c31f9a983352d8949e9947040faf01d5f4f39751dc1a70935c`.
The three-ULP worsening plant exits nonzero.

Separately run base and tip 30-day members have 0 / 30 differing daily
snapshots.  Their day-30 snapshot SHA-256 is
`a66143733bcc9e4efaa22fe2b3a831629e4d57ffd701ddd827d5553c3e0b007a`.
Their manifests name the exact base and implementation commits above.  This
confirms R34-P4 by measurement.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R34-P1 | **CONFIRMED** | Carried `hf_0` is 0 / 26,640 unequal to the compiled construction; reconstruction is 14,324 unequal with 651.239260895305 m maximum. |
| R34-P2 | **CONFIRMED** | The production replay retains the frozen `3.181628207426175e-09` / `2.9702048395431957e-09` m/s2 maxima and remains non-bit. |
| R34-P3 | **CONFIRMED** | 40/40 checkpoints complete, 185/200 rows move, no exact/AT-BAR row leaves its bar, and the first statement is unchanged. |
| R34-P4 | **CONFIRMED** | GYRE has 70/70 unchanged rows, 210/210 exact residual arrays, and 30/30 byte-identical snapshots. |
| R34-P5 | **CONFIRMED** | The shared-card, focused, citation and outcome controls pass; every planted violation fires. |

The preregistration's two pre-measurement corrections are retained: `hf_0`
is constructed from admitted mesh inputs rather than dumped as a stream, and
GYRE carries a constant-depth operand which equals its reconstruction.

## Gates, review, and tests

The exact shared-card inventory is green at the final tested tip: 160
DINO/rule-12/lock/overflow tests pass with nine warnings in 334.22 s, and the
remaining tank round-34 file passes 10/10 in 7.49 s, for **170/170**.  The
pre-commit focused round-31/34 reference and operand tests pass **12/12**;
after the citation-preserving helper extraction, the focused reference,
operand, and citation files pass **28/28 in 6.08 s**.

The required separate `codex exec --sandbox read-only` review was attempted
against the committed diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.  The complete failure
log is preserved as `codex_readonly_review.log`.

The receipt citation gate passes all three compiled citations with zero
failures, zero unmapped citations, and zero map-audit failures.  Its rigid
line-shift plant exits nonzero with `SYMBOL-NOT-AT-LINE`.  The implementation
helper leaves the historical 99-line vorticity-function citation unchanged;
the two later `vertical.py` citations and their source anchors are shifted by
the same ten lines.

The required single `tests/ocean/fidelity -n 12` battery collected 1,893
tests.  It reached the inherited final-tail stall at 98% and was interrupted
after a bounded silent wait.  Its progress stream emitted five failures, all
at the same positions as round 33.  Their exact node IDs were rerun in
isolation and reproduce the inherited SI3 MY_SRC provenance mismatch, stale
GYRE member/gate stamp, round-51 trace suffix, three unstamped legacy report
emitters, and missing `hires_lane_surface` case-board row.  No round-34 test
fails.

The final ORCA2 push battery runs the citation gate, TKE terms, recipe,
freshwater closure, and parallel-receipt gate against the committed receipt:
**127/127 pass in 372.23 s**.

## Choices

ASKED: round 33's OPEN section authorizes the compiled-order `r3f` operand
walk, and the card already carries the selected NEMO `hf_0` operand.

UNASKED: none.  No configuration value, stabilizer, NEMO source, carried
state, sea-ice selector, score, or acquisition changed.

## OPEN

1. The given-NEMO-entry lateral-diffusion replay still has
   `3.181628207426175e-09` / `2.9702048395431957e-09` m/s2 maxima.  Continue
   the compiled-order one-variable walk at the remaining `r3f`/metric
   operands; do not call this faithful operand repair a tendency closure.
2. Round 20's ranked slow-forcing producer walk remains open.
3. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
4. Decision 52's independent ORCA2 initial-state transcription and year
   comparison remain owed.
5. The inherited duplicate citation-map literal keys remain open.
