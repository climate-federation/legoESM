# ORCA2 round 42 — EOS80 `rhd` landing

Date: 2026-09-27  
Card: `orca2_vector_een_c2`  
Claim label: **given NEMO's entry** (Decision 52)

## Verdict

**LANDED.**  The first non-bit input to the round-41 stage-1 HPG row was not
geometry: it was the EOS80 density-anomaly ratio `rhd`.  legoESM formed the
algebraically equivalent `(rho-rho0)/rho0`; NEMO stores
`rho/rho0-1`, then consumes that dimensionless value directly.  The landing
wires the existing source-associated EOS80 ratio and carries it to the literal
HPG consumer without the intervening `*rho0/rho0` round trip.  No selector,
sea-ice field, or stabiliser changed.

The 64 disputed rank-1 periodic-east U faces close completely:

| registered boundary | before unequal / scored | before max | after |
|---|---:|---:|---:|
| `rhd` | 3,497 / 3,508 | `2.6281060661048627e-16` | 0 / 3,508 |
| `e3w` | 0 / 3,508 | 0 | 0 / 3,508 |
| `gdept_z0` | 0 / 3,508 | 0 | 0 / 3,508 |
| `r1_e1u` | 0 / 64 | 0 | 0 / 64 |
| `zhpi_u` | 1,753 / 1,754 | `1.4862887125471208e-17` | 0 / 1,754 |
| `zuap_u` | 1,753 / 1,754 | `3.88774887899923e-22` | 0 / 1,754 |
| `sum_u` | 1,753 / 1,754 | `1.4862887125471208e-17` | 0 / 1,754 |

The record self-replay is independently exact for all 413,030 owned U cells.
The recorded-replay ULP plant and first-boundary selector plant both fire.

## Preregistered predictions

| prediction | outcome |
|---|---|
| P1 acquisition admission and passive parents remain exact | CONFIRMED |
| P2 round-41 result reproduces before the arm | CONFIRMED |
| P3 recorded literal HPG self-replays exactly | CONFIRMED |
| P4 geometry is exact and `zhpi` is the first non-bit input | **REFUTED**: geometry is exact, but `rhd` is already non-bit |
| P5 land only if one recorded operand closes the row and every gate passes | CONFIRMED by the source-ordered `rhd` statement |

The failed P4 prediction remains in the preregistration and is not rewritten.

## Ten-step ORCA2 ladder

The production CPU/fp64/libm ladder completes kt=1..10 (40 checkpoints, 200
field rows).  Against the last certified ladder, 195 rows move: 65 maxima move
toward NEMO, 113 away, and 17 retain the same maximum.  No bit-identical row
leaves the bar, and the first non-bit statement remains kt=1 stage-1
temperature.  Every moved row is serialized in `orca2_ladder_tip.json`; the
AT-BAR-loss plant exits nonzero.

This landing therefore closes the named stage-1 HPG statement without claiming
that the whole ORCA2 step is bit-identical.  The first non-bit whole-step owner
remains the pre-existing, unattributed stage-1 temperature row.

## GYRE non-regression

The base (`368345a08`) and tip (`a82146d4a`) ten-step gates compare 70/70
certified rows with zero field move, zero residual worsening, unchanged first
over bar at kt=3, and identical residual archive SHA-256
`43f37831256832c31f9a983352d8949e9947040faf01d5f4f39751dc1a70935c`.
The three-ULP plant fails as required.

The required 30-day members also have 30/30 byte-identical daily snapshots.
Day 30 remains SHA-256
`a66143733bcc9e4efaa22fe2b3a831629e4d57ffd701ddd827d5553c3e0b007a`.

## Controls, tests, and review

- Focused round-42, precomputed-density parity, and EOS tests: 6 passed.
- `tests/ocean/fidelity -n 12`: all 1,925 items were dispatched; the xdist
  controller did not terminate after reaching the final progress row and was
  interrupted after its workers exited.  Eight red markers appeared during
  the run; no final node-id report was materialized.  This is recorded as an
  incomplete battery, not called green.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).

## Evidence

All durable artifacts are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round42/`.
The decisive reports are `stage1_hpg_walk_diagnostic2.json`,
`stage1_hpg_walk_landing.json`, `orca2_ladder_tip.json`,
`gyre_ladder_compare.json`, and the two 30-day member directories.

## OPEN

1. The full fidelity walk returns to the unchanged first non-bit whole-step
   statement: kt=1 stage-1 temperature (`0.0014770192519700243 K` maximum on
   the landed ladder), still `UNATTRIBUTED` by the ladder.
2. The full fidelity battery must be rerun in smaller shards if a clean suite
   summary, beyond the focused green tests, is required.
3. Sea ice remains exactly the card's six-item `unmeasured_features` tuple.

## Compiled-source citations

The executing EOS80 branch normalizes its operands, evaluates the Roquet
polynomial, and stores the masked density anomaly in the source order
`zn * r1_rho0 - 1` at
`ORCA2_ORCA1ICE_OMIP_L4_R41HPG1/BLD/ppsrc/nemo/eosbn2.f90:810-844`.
The executing HPG branch consumes `rhd` directly in the surface and interior
recurrences and writes `zhpi + zuap` at
`ORCA2_ORCA1ICE_OMIP_L4_R41HPG1/BLD/ppsrc/nemo/dynhpg.f90:403-458`.
