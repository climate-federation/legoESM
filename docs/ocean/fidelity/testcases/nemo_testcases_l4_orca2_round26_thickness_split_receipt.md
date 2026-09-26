# NEMO testcase Lane 4 — ORCA2 card round 26 thickness-split receipt

Date: 2026-09-26

Parent: `7f382b7ccd33aa8c8785d52493192057c5d60dcc`

Status: **HELD — F-CURL THICKNESS ALONE OWNS THE KT=4 REFUSAL.**
NEMO's carried live F thickness, isolated at the compiled curl statement,
reproduces the non-positive raw-`e3w` refusal while entering kt=4.  The
T/U/V Kbb-divergence arm and the U/V Kmm-divisor arm both remain finite
through kt=10 and produce identical trajectory score documents.  Every
experimental model change was reverted; the final `packages/` tree is
identical to the parent.

All trajectory numbers are **independent with Decision-52 SSH**.  No
given-NEMO-entry operator result is mixed into the table.  The six sea-ice
selectors and the card's `unmeasured_features` tuple remain unchanged.

Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round26/`.

## Compiled statements and controlled arms

The admitted build forms the F curl with live
`e3f_3d*(1+r3f*fe3mask)` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:121-125`.
It forms the divergence with live T/U/V Kbb thicknesses at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:127-129`,
then divides the U and V curl increments by their live Kmm face thicknesses
at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:132-140`.

Each arm started from the same restored implementation and a clean committed
worktree.  The instantiated card object is equal to the baseline in every
arm.  Mask, metric, carried-state, forcing, and sea-ice paths are unchanged.

| arm | clean commit | isolated compiled position |
|---|---|---|
| F curl | `bd99f62abb81256c43713c1a152051d95885dff4` | only the carried F reference thickness, frozen F-thickness mask, and F-column depth used to build live `e3f` at lines 123--125; parent area/edge metrics retained |
| Kbb divergence | `a829392da2db186a1aad795da1de5bf509ba2205` | only U/V Kbb face thicknesses inside lines 127--129; parent F and Kmm operands retained |
| Kmm divisor | `61d80d32ab5bebb238572f3ea75c2a9fd0b3edc1` | only U/V Kmm outer divisors at lines 135/139; parent F and Kbb operands retained |

## Independent trajectory result

The fresh parent, Kbb, and Kmm artifacts contain 40 checkpoints (kt=1..10).
The F-curl artifact contains the 12 checkpoints through kt=3, followed by a
separate captured run that fails closed while entering kt=4 with
`raw-mesh e3w_int must contain only finite values > 0`.

| arm | checkpoints | moved field rows | first moved row | direction by maximum error | outcome |
|---|---:|---:|---|---:|---|
| F curl | 12 | **45 / 60 observable** | kt=1 stage-2 U | 15 toward / 30 away | exact kt=4 refusal |
| Kbb divergence | 40 | **185 / 200** | kt=1 stage-2 U | 82 toward / 98 away / 5 same maximum | `LADDER_MEASURED` through kt=10 |
| Kmm divisor | 40 | **185 / 200** | kt=1 stage-2 U | 82 toward / 98 away / 5 same maximum | `LADDER_MEASURED` through kt=10 |

No arm moves a formerly bit-identical row off the bar.  The first non-bit
NEMO statement remains kt=1 stage-1 temperature in every arm.

The first movement separates the owner sharply:

| row | parent max error | F curl | Kbb | Kmm |
|---|---:|---:|---:|---:|
| kt=1 stage-2 U | `0.06470386947382581` | `0.1007577986233476` | `0.06470519697104807` | `0.06470519697104807` |
| kt=1 stage-2 V | `0.034012848056840184` | `0.11456531655396814` | `0.03400745468409555` | `0.03400745468409555` |

By kt=3 stage 3, the F-curl U and V errors are
`348.14849703616125` and `158.39341920935638` m/s.  Kbb and Kmm finish kt=10;
their entire `candidate_trajectory` score objects are equal, not merely their
first rows.  Their kt=10 stage-3 U/V maximum errors are
`15.36545117195394` / `42.67107849176734` m/s versus parent
`15.365503106245665` / `42.669598831454074` m/s.

The round-25 combined live-thickness artifact is reproduced by the outcome
gate at its published first movement: U `0.10080009966621735` and V
`0.11456680400114852` m/s.  The isolated F values differ from that combined
arm, so the other thickness positions still interact with the owner; they do
not independently cause the refusal.

**RETRACTION:** round 25's indivisible `live_thickness` owner is now narrowed.
The F-curl thickness is sufficient to reproduce the refusal.  Kbb divergence
and Kmm divisors are not sufficient.  The preregistered round-26 hypothesis
that Kbb alone would refuse is also retracted by its complete ten-step run.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R26-P1 | **CONFIRMED** | Each clean commit changes its named operand position; all instantiated cards equal the parent. |
| R26-P2 | **CONFIRMED** | Every arm first moves at kt=1 stage-2 U and keeps the first NEMO mismatch unchanged. |
| R26-P3 | **REFUTED** | F curl, not Kbb, refuses at kt=4; Kbb and Kmm reach kt=10. |
| R26-P4 | **CONFIRMED** | No formerly exact row leaves the bar. |
| R26-P5 | **CONFIRMED** | The admitted combined artifact reproduces the two frozen round-25 first-movement scores exactly. |
| R26-P6 | **CONFIRMED** | The exact-row plant is refused; the citation plant also fires. |

No failed prediction was rewritten.

## Gate, review, and tests

The round-26 outcome gate exits 2 with `HELD`, registers every moved row,
requires the exact F-curl refusal, requires complete Kbb/Kmm ladders, compares
the instantiated cards, and records the identical Kbb/Kmm score documents.
Its one-ULP exact-row plant exits 1 with `formerly AT-BAR rows left`.

The required separate `codex exec --sandbox read-only` review was attempted
on the committed round diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

Citation and pytest results are recorded in the final commit after their
single-battery executions.

No GYRE/DINO/lock-exchange/overflow landing gate is claimed: this round lands
no model statement, and the final `packages/` tree is byte-identical to the
already-certified parent.

## Choices

ASKED: Decision 54 and round 25's OPEN item authorize this compiled-order
thickness split.

UNASKED: none.  No configuration value, default, carried state, stabilizer,
scoring rule, sea-ice selector, or NEMO source changed.

## OPEN

1. Decision 54 remains held.  The faithful F-curl operand exposes the
   trajectory failure; next walk the kt=1 stage-1 LDF curl contribution and
   its consumers against the admitted operator replay to find the compensating
   statement before retrying a landing.  Keep the raw-`e3w` refusal unchanged.
2. Kbb and Kmm produce identical score documents despite occupying different
   compiled positions.  A direct tendency-component discriminator must decide
   whether this is exact cancellation at the measured entry or a score-level
   collision before either is called equivalent.
3. Round 20's ranked slow-forcing producer walk remains open.
4. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
5. The independent ORCA2 year still depends on its scheduled independent
   initial-state completion.
6. The seven inherited duplicate citation-map literal keys remain open.
