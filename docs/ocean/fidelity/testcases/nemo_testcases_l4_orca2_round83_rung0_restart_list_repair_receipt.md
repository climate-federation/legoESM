# ORCA2 round 83 — rung-0 restart-list acquisition repair

Base: `1088d13195`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_round83.md`.  Claim label:
**independent**.  No package file, rung-0 scientific assignment, shipped
ORCA2 card, NEMO source, CPP key, carried state, threshold, or sea-ice
selector changed.

## Verdict

**STOPPED_FOR_RECORD.**  The operator's round-82 run never entered time
stepping: NEMO rejected periodic `nn_stock=1` because the unchanged surface
update cadence is `nn_fsbc=2`.  Round 82's claim that this schedule would
produce ten per-step restarts is **RETRACTED** in its own receipt and in the
tool.  Round 83 switches only the two ten-step twins to NEMO's compiled
explicit restart-list run control, renders and validates the actual decks,
and writes only fresh targets.  The repaired operator acquisition is
preflight-ready; no rung-0 NEMO trajectory exists yet, so no card or ladder
measurement lands.

## Source-cited repair

The compiled surface initialization checks experiment length against
`nn_fsbc`, but applies the `nn_stock` divisibility requirement only when
explicit restart-list mode is false
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-346`).  The compiled restart
writer takes its first target from `nn_stocklist` and opens the named target
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-119`), then advances to the next
list entry after writing
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:188-202`).  The repaired
ten-step decks therefore set:

```text
nn_itend=10
nn_stock=1
ln_rst_list=.true.
nn_stocklist=1,2,3,4,5,6,7,8,9,10
```

The month remains `nn_itend=240`, `nn_stock=240`, with no restart-list
override.  The scientific canonical deck SHA-256 remains
`b627f4e2d94e4619dbfa27f39b811732497f032b73be86adcc57ddca2dee0e91`.
The rendered ten-step run deck is
`d25c69958aeb7d4dffeeab6b08c89f6b314dd7c6d130ed94acfee7cb90643c2c`;
its only additional assignments are the two output-control keys above.

The record gate now refuses any other assignment inventory, a false
restart-list selector, a missing/reordered list step, or restart-list mode on
the month.  Admission also requires the resolved NEMO log to show all ten
restart opens; existence of filenames alone is insufficient.  Existing
payload, rank, step, non-finite, hidden-month-deck, binary, CPP, input-manifest,
and twin-identity checks remain binding.

## Fresh acquisition

The launcher targets only round-83 paths:

| target | restart mode | required output |
|---|---|---|
| `orca2_rung0_restart_list_10step_a_np2` | explicit list 1..10 | 20 rank-step shards |
| `orca2_rung0_restart_list_10step_b_np2` | explicit list 1..10 | array-equal twin |
| `orca2_rung0_restart_list_repair_240step_np2` | periodic terminal 240 | two finite terminal shards |

The incomplete round-82 directory remains untouched.  Preflight prints
`ORCA2_ROUND83_RUNG0_PREFLIGHT_READY` and mechanically reports the twin mode
as `list` with steps 1..10 and the month mode as `periodic` at step 240.  The
operator must run the committed round-83 launcher; in-sandbox MPI execution is
forbidden by the standing PMIx note.

## Prediction ledger

| frozen prediction | result |
|---|---|
| only ten-step output scheduling changes | **CONFIRMED at preflight** |
| twins resolve list 1..10; month stays periodic | **CONFIRMED at preflight** |
| twenty shards per twin and finite month | **UNMEASURED — operator record absent** |
| five record plants fire | **UNMEASURED — they require the record** |
| card/ladders/month stay unmeasured without record | **CONFIRMED** |

## Gates, review, and tests

Focused rung-0 tests pass **13/13**; the new tests bind exact list rendering,
missing-list-step refusal, and month-mode refusal.  Both shell launchers pass
`bash -n`.  The campaign-default citation gate passes with zero failures and
zero unmapped citations; shifting the new `sbcmod` citation exits 1 with
`SYMBOL-NOT-AT-LINE`.

The prescribed `tests/ocean/fidelity -n 12` battery selected 2,207 tests,
reached 99%, and repeated the standing no-summary stall; it was stopped after
the visible SI3 scalar-math and round-129 record failures.  The six established
failure files were rerun together in isolation: **6 failed / 37 passed**,
exactly the known round-129 certification, round-35 stamp scope, worktree-stamp
emitter, missing case-board row, SI3 scalar-math provenance, and round-51
private-trace registry reds.  No round-83 test failed.

The required read-only Codex review failed before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

No `packages/` file changed.  GYRE, DINO, tank cards, the shipped ORCA2 card,
and sea ice are byte-identical by construction; no trajectory claim is made.

## OPEN

1. The operator runs the round-83 launcher and returns its log.
2. Admit all three fresh records and show every payload/control plant firing.
3. Only after admission, build the explicit rung-0 legoESM card and run its
   given-entry and independent ten-step ladders plus independent month.
4. Name the first non-bit statement and score every row; higher rungs remain
   untouched until rung 0 lands.

ASKED: repair and reissue the failed rung-0 acquisition without changing its
physics.

UNASKED: changing `nn_fsbc`, any physical selector/coefficient, NEMO source,
the shipped card or its ice tuple, thresholds, carried state, or stabilizers.
