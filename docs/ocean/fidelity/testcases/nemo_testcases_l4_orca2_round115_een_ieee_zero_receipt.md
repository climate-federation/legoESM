# ORCA2 round 115 — EEN recurrence signed-zero walk

**Status: HELD.** Given NEMO's recorded rung-0 entry and operands, rounds
111--114 already made the stored northwest-U product bit-exact. The first
non-bit source statement is now the carried addition in the EEN vertical
recurrence: 1,618 `acc_before` and 5,197 `acc_after` values differ only by the
sign of exact zero. Restoring IEEE zero-addition semantics closes both rows at
zero unequal bits, but no model statement lands because the other seven EEN
coefficient recurrences are not recorded at this boundary.

The compiled program initializes the eight accumulators to positive zero and
enters the U loop through `mbku`
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1238-1245`). It then
evaluates the northwest product, records the accumulator before, performs
`ffu_nw = ffu_nw + product`, records it after, and continues to the northeast,
southwest, and southeast recurrences
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1263-1273`).

## Frozen prediction disposition

All measurements below are **given NEMO's recorded entry** and use the
admitted two-rank round-108 per-level record under production JIT on CPU with
fp64/x64/libm.

| prediction | disposition |
|---|---|
| R115-P1: stored product remains two signed-zero bits unequal | **REFUTED**; it is already 0 unequal after the upstream quotient closure |
| R115-P2: one-statement product materialization closes those two bits | **REFUTED AS NULL**; baseline and arm are already bit-identical |
| R115-P3: accumulator-before is next | **CONFIRMED**, 1,618 signed-zero-only bits, first `(7,42,1)` |
| R115-P5/P6: peeling the first recurrence iteration closes the carry | **REFUTED**, zero rows move |
| R115-P7: IEEE signed-zero addition closes accumulator-before without nonzero movement | **CONFIRMED**, 0 unequal; all 1,618 movements are zero-sign-only |
| R115-P8: the same rule closes accumulator-after | **CONFIRMED**, 5,197 to 0 unequal; all movements are zero-sign-only |
| R115-P4: ORCA2/VORTEX/VORTEX_VEC/two DINO recipes execute; GYRE/tanks do not | **CONFIRMED** by the closed card census |

At the first mismatch, NEMO records `acc_before=+0`, `term=-0`, and the
following `acc_after=+0`; the JIT baseline carries `-0`. The admitted record's
own independent recurrence check reports zero unequal bits for host IEEE
`acc_before + term`. Thus the stored product is exonerated and the addition's
zero-sign semantics own the boundary. This is an arithmetic identity finding,
not a stabilizer, tolerance, configuration choice, or physical tendency.

## Instrument controls

The gate reads the existing self-describing rank shards, requires exactly-once
148 x 180 coverage, applies `mbku` before comparing, and prints every compared
shape and dtype. Both shards retain their admitted SHA-256 values
`7387b11c90a540e24e8947dd53af06d0752d2b7d5c227cddae5e4f0029b639a9` and
`6a41ab2d5acbc903c32ca44ac3fc71cbf99ff7a8aad04d6b193ce98d52908c56`.
The oracle-bit, candidate-bit, and resolved-route plants each exit 2.

The first gate invocation refused the short commit pin before reading data;
the second correctly refused the frozen P1 baseline and exposed
`acc_before`. Both refusals are instrument outcomes, not ocean results. The
committed addenda preserve the failed product and peeled-loop predictions.

## Scope and landing decision

No `packages/` file changes in this round. Therefore the rung-0/rung-7 ORCA2,
GYRE, DINO, tank, and generic-card trajectories do not move and are not
relabelled as rerun results. A production landing would also change the
unmasked face-thickness and literal loop arm held since round 107; it must be
judged on all eight coefficients, not on northwest U alone. This round stops
at the exact first statement and carries the candidate forward without a
partial landing.

Separate `codex exec --sandbox read-only` review was attempted. Verdict:
**independent review unavailable in-sandbox** (`failed to initialize
in-process app-server client: Read-only file system`).

The focused round-107/109/114/115 battery passes 10/10. The one required
`tests/ocean/fidelity -n 12` invocation collected 2,342 tests, reached 99%,
and reproduced the lane's registered xdist-controller stall after 60 seconds
without output. Before interruption it emitted exactly the six pre-existing
failures from round 114: SI3 scalar-math provenance, GYRE round-51 private
operands, round-35 escape scope, worktree stamping, the recipe case-board row,
and the GYRE round-129 record stamp. Both new round-115 tests passed.

## OPEN

1. Record the per-level operands, products, and before/after accumulators for
   `ffu_ne`, `ffu_sw`, and `ffu_se`; the admitted stream contains only
   northwest U. Walk them in compiled order before applying the IEEE-zero arm
   to production.
2. Then walk the northern V cancelling pair and the separate later substep-2
   68-cell U residual.
3. Land the shared arithmetic only when every executing coefficient path is
   bit-exact and the full ORCA2/GYRE/DINO/tank/generic-card gates pass.
4. The package-exposed rung-0 card and independent 240-step month remain open
   hierarchy deliverables.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
