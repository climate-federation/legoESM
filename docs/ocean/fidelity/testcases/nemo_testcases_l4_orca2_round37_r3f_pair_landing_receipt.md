# NEMO testcase Lane 4 — ORCA2 card round 37 `r3f` pair landing receipt

Date: 2026-09-26

Parent: `53dab7544`

Implementation commit: `b9522dafa`

Final tested implementation/citation tip: `407cd6890`

Status: **LANDED — NATIVE F AREA PLUS NEMO'S STORED RECIPROCAL CLOSE THE
GIVEN-ENTRY LATERAL-DIFFUSION REPLAY BIT-FOR-BIT.**  The two halves measured
separately in rounds 35 and 36 are a real cancelling pair at the bit boundary:
neither isolated arm meets the zero-cell bar, while their compiled combination
has zero unequal U and V cells.  ORCA2 completes kt=1..10 without losing an
exact/AT-BAR row.  GYRE is byte-identical through ten steps and 30 days.

Direct-operator results below are **given NEMO's entry (kt=2 recorded
state)**.  Trajectory results are **independent with Decision-52 SSH**.  The
labels are not mixed in one table.  The six sea-ice selectors and the card's
`unmeasured_features` tuple are unchanged.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round37/`.

## Compiled pair and landed change

The executing build materializes native `e1f*e2f` and stores `r1_e1e2f` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domhgr.f90:155-157`.
The executing QCO statement multiplies its `r3f` numerator by that reciprocal
at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`.
The lateral-diffusion consumer applies the resulting live F thickness at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123`.

Production now selects the card-carried native F metrics when raw NEMO mesh
operands exist, materializes their product and reciprocal, and preserves the
compiled multiplication boundary.  Cards without raw NEMO F metrics retain
the existing geometric fallback.  No card value or configuration field was
added.

## Given-NEMO-entry pair gate

The gate reruns both isolated controls before accepting the combination:

| arm | U unequal / scored | U max, m/s2 | V unequal / scored | V max, m/s2 |
|---|---:|---:|---:|---:|
| parent shifted area + division | 410,460 / 411,736 | `3.181628207426175e-09` | 407,570 / 412,537 | `2.9702048395431957e-09` |
| shifted area + stored reciprocal | 410,460 / 411,736 | `3.181628207426175e-09` | 407,570 / 412,537 | `2.9702048395431957e-09` |
| native area + division | 24 / 411,736 | `1.0587911840678754e-22` | 24 / 412,537 | `1.0587911840678754e-22` |
| **native area + stored reciprocal** | **0 / 411,736** | **0.0** | **0 / 412,537** | **0.0** |

R37-P1 is **CONFIRMED**.  This is an exact cancellation result, not a tolerance
argument.  The planted U value exits nonzero.

## Independent ORCA2 ladder

The base artifact is round 34's final certified ladder at `13c2927a8`.  A
tree comparison proves its `packages/` tree is identical to round 37's parent
`dfe2710a3`; the interrupted round-36 attempt is not used.  The final landed
arm at `407cd6890` completes all 40 checkpoints.  The helper extraction and
fallback repair made after the first run were each followed by a complete
re-run; only the final-tip artifact is authoritative.

Of 200 scored rows, **185 move**: 111 maxima move toward NEMO and 74 move away.
No exact/AT-BAR row leaves its bar, every moved row is registered, and the
first non-bit statement remains kt=1 stage-1 temperature.  The first movement
is kt=1 stage-2 U.

| kt=10 stage-3 field | parent maximum | landed maximum | direction |
|---|---:|---:|---|
| T, degC | `0.9838706913744293` | `0.9838385157064167` | toward |
| S, psu | `0.22099878628421266` | `0.22099092579694712` | toward |
| U, m/s | `0.33900405135003436` | `0.33841376678661433` | toward |
| V, m/s | `0.5353447182593362` | `0.5353620444727542` | away |
| SSH, m | `0.5971345413221235` | `0.5964525198077717` | toward |

R37-P2 is **CONFIRMED**.  The planted exact-row loss exits nonzero.

## Independent GYRE safety gates

The round-34 final ladder is also the certified GYRE base because its package
tree equals the round-37 parent.  The offline oracle-relative comparison finds
**0 differing certified rows out of 70**, zero residual worsening, and no
first-over-bar movement.  All **210 / 210** residual arrays are
`np.array_equal`; both archives have SHA-256
`43f37831256832c31f9a983352d8949e9947040faf01d5f4f39751dc1a70935c`.
The three-ULP worsening plant exits nonzero.

The separately run landed 30-day member has **30 / 30 byte-identical daily
snapshots** against round 34's package-identical base.  Both day-30 files have
SHA-256
`a66143733bcc9e4efaa22fe2b3a831629e4d57ffd701ddd827d5553c3e0b007a`.
R37-P3 is **CONFIRMED**.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R37-P1 | **CONFIRMED** | Both isolated controls reproduce; the pair has zero unequal U and V cells. |
| R37-P2 | **CONFIRMED** | ORCA2 completes 40/40 checkpoints; 185 rows move, no AT-BAR row leaves, and the first statement is unchanged. |
| R37-P3 | **CONFIRMED** | GYRE has 70/70 unchanged rows, 210/210 exact residual arrays, and 30/30 byte-identical daily snapshots. |
| R37-P4 | **CONFIRMED** | The 170-test shared-card battery passes and all measured scientific controls fire. |

No prediction was rewritten after measurement.

## Gates, review, and tests

The final-tip shared-card battery passes **170 / 170**: 160
DINO/Rule-12/lock/overflow tests pass with nine warnings in 344.46 s, followed
by 10 / 10 tank tests in 7.43 s.  The pre-commit round-32/35/36/37 focused
battery passes **12 / 12**.

The required separate `codex exec --sandbox read-only` review was attempted
at the committed diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

The first citation-map run caught three displaced `vertical.py` anchors.  A
naive rigid shift then failed because this round had expanded the cited
function.  The area arithmetic was extracted into a helper so the historical
function kept its exact 99-line extent; all three anchors and their map entries
then moved by one rigid +12-line shift.  The final citation gate passes all
three compiled citations with zero failures, zero unmapped citations and zero
map-audit failures.  Its rigid source-shift plant exits nonzero and all nine
self-tests fire.

The first wide battery exposed two real new failures: synthetic partial-card
fixtures carry `hf_0/e3f_0` but no `e1f/e2f`, and the helper treated them as
metric-complete.  The final implementation requires both raw metrics before
selecting the native path and otherwise retains the geometric fallback.  Both
tests pass after that repair, all ORCA2/GYRE/card gates above were rerun at the
repair commit, and the rigid citation positions did not move.

The closing `tests/ocean/fidelity -n 12` battery reached 99% and the inherited
silent tail, then was interrupted after a bounded wait.  It emits only the
five known inherited failures: SI3 MY_SRC provenance, the stale GYRE
member/gate stamp, round-51 trace suffix, three unstamped legacy report
emitters, and missing `hires_lane_surface` case-board row.  An isolated rerun
reports **5 failed, 2 passed in 10.60 s**; the two passes are the repaired
round-31 partial-card tests.  No round-37 path remains red.

The closing focused battery passes **39 / 39 in 6.60 s**.  The final push
battery passes **130 / 130 in 381.32 s**, including the citation-gate tests,
TKE terms, recipe and freshwater-closure units, the ORCA2 parallel receipt
gate, and this round's three planted-control tests.

## Choices

ASKED: round 36's OPEN section authorized this measured cancelling pair after
both halves were isolated.  Both landed operations are cited compiled NEMO
statements.

UNASKED: none.  No configuration value, stabilizer, carried state, NEMO
source, sea-ice selector, score, or acquisition changed.

## OPEN

1. The given-entry lateral-diffusion replay is now closed.  Return to the
   actual first non-bit statement, kt=1 stage-1 temperature, and resume round
   20's ranked slow-forcing producer walk in compiled order.
2. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
3. Decision 52's independent ORCA2 initial-state transcription and year
   comparison remain owed.
4. The inherited duplicate citation-map literal keys remain open.
