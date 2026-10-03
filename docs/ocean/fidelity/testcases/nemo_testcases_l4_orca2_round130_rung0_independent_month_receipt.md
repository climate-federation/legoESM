# ORCA2 round 130 — rung-0 independent month

Date: 2026-10-03. Base `59ca6ad42`; measurement tip `68b88da2d`. Every
scientific number is **independent**: legoESM starts from the rung-0 card's own
climatological T/S, zero velocity, and zero sea surface. Decision 52's recorded
entry bridge is not used. No configuration, forcing, carried state,
stabilizer, sea-ice selector, or `unmeasured_features` entry changed. No
`packages/` file changed.

## Verdict

**HELD.** The committed scorer re-admits NEMO's finite kt=240 rung-0 restart,
but legoESM does not reach it. Two identical production-JIT CPU/fp64/x64/libm
runs first become non-finite at step 16, field T, index
`(j,i,k)=(1,49,0)`. Both emit the exact refusal:

```text
STATUS REFUSE: first non-finite step=16 field={'field': 'T', 'index': [1, 49, 0], 'value': nan}
```

The first run took 155.7 s through step 10; the independent repeat took
165.5 s. No terminal field is scored, so no RMS/max magnitude or ranking is
reported. NEMO's admitted from-rest run reaches kt=240 with finite T, S, u, v,
and sea surface on both ranks. Its RK3 program calls stages 1, 2, and 3 in
that order (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:211-227`). This is a
legoESM rung-0 debt, not permission to add a stabilizer or alter the deck.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R130-P1: 240 finite steps | **REFUTED**: deterministic first non-finite T at step 16, `(1,49,0)`. |
| R130-P2: round-83 terminal record unchanged and complete | **CONFIRMED** before stepping: admission SHA-256, binary identity, month deck, both terminal shard digests, kt=240, fp64/finiteness, coverage, and card orientation pass. |
| R130-P3: five non-bit terminal rows; frozen RMS/max orders | **UNMEASURED**: no terminal candidate exists. The prediction is retained, not post-hoc replaced. |
| R130-P4: terminal ULP/non-finite plants bind | **CONFIRMED at the committed classification boundary**: both dedicated tests raise the gate refusal. A measured-report plant cannot run because the baseline has no terminal report. |
| R130-P5: measurement-only diff | **CONFIRMED**: no model file or resolved card field moved. |

The first attempted run produced no model number: it stopped during record
admission because the inherited generic ledger parser rejected the duplicate
ten-step-twin basenames in the cumulative SHA ledger. The repaired committed
parser selects the two month shards by full path, refuses missing/duplicate
month rows, and passes a direct regression test containing a colliding twin
basename. Only the two later runs support the step-16 finding.

## Instrument, controls, and tests

The scorer checks its clean-worktree commit stamp, CPU backend, production JIT,
fp64/x64/libm policy, rung-0 card selectors, round-83 admission digest and
binary identity, month deck controls, both restart shard hashes and axes, and
every field for finiteness after every step. Its terminal path, when reached,
reports strict bits, unequal count, RMS, maximum absolute difference, argmax,
and candidate/oracle values for all five fields.

Focused round-71/82/103/130 tests pass **30/30**. The named
`terminal-ulp` and `terminal-nonfinite` controls are visible separately in the
test log and both pass only because the classifier raises. The citation gate
and full ocean-fidelity battery are recorded in the final validation section
below.

Separate read-only Codex review was attempted before measurement and returned
**independent review unavailable in-sandbox**:
`failed to initialize in-process app-server client: Read-only file system`.

## Final validation

PENDING.

## OPEN

1. Walk the independent rung-0 trajectory from the last finite step through
   step 16 and name the first non-finite statement, preserving the frozen
   deck, zero forcing, and card-owned initial state. The first evidence target
   is T at `(1,49,0)`; its producing statement is **UNVERIFIED**.
2. Do not reland round 129's held split-explicit boundary/transport chain. Its
   month predicate cannot be evaluated until the baseline itself is finite.
3. After the rung-0 baseline reaches kt=240, run this committed scorer and
   report the preregistered five-field RMS/max rankings; only then declare
   rung 0 and proceed to the parked hierarchy merges.

## UNVERIFIED

- The step-16 statement that first creates the non-finite temperature.
- Every rung-0 terminal RMS/max value and ranking.
- The independent month effect of round 129's held source unit.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
