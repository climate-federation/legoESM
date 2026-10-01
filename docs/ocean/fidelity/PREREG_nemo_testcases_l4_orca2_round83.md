# ORCA2 round 83 preregistration — rung-0 restart-list repair

Frozen before repairing or rerunning the rung-0 acquisition.  Base:
`1088d13195`.  Claim label: **independent**.  This round responds to the
operator's failed round-82 run; it does not change the rung-0 scientific deck,
the shipped ORCA2 card, any package file, NEMO source, CPP key, carried state,
threshold, or sea-ice selector.

## Admitted failure and compiled-source reading

The round-82 first twin stopped during initialization with exit 123.  Both
rank logs identify one error: `nn_stock=1` is not a multiple of the shipped
`nn_fsbc=2`.  The compiled guard applies that restriction only to periodic
restart output; it explicitly bypasses it when the restart-list mode is used
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-350`).  The compiled restart
writer initializes its next output from `nn_stocklist`, opens/writes the named
step, then advances to the next list entry
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-119,188-202`).

## Frozen repair and predictions

1. The two fresh ten-step twins use the compiled run-control interface
   `ln_rst_list=.true.` and `nn_stocklist=1,2,...,10`; `nn_stock=1` remains
   present but is ignored for scheduling.  The 240-step month retains
   `ln_rst_list=.false.`, `nn_stock=240`.  No scientific namelist assignment
   differs between the three runs.  Any other run-control or deck delta
   **REFUTES** the repair.
2. The ten-step resolved logs print restart-list mode and all ten list entries;
   each twin produces steps 1..10 on ranks 0 and 1.  A missing/wrong step or
   rank, a periodic-restart cadence refusal, or a non-finite payload
   **REFUTES** the repair.
3. The month completes step 240 and remains finite on both ranks.  It must not
   select restart-list mode.  A hidden month-deck delta **REFUTES** admission.
4. Twin A and B are array-equal for fp64 T/S/u/v/ssh at every step and rank.
   The existing one-ULP, missing-rank, wrong-step, non-finite, and hidden-deck
   plants must all fire.
5. The rung-0 card, ladders, month scores, and first non-bit statement remain
   **UNMEASURED** until this repaired record is admitted.  Without the operator
   record this round is `STOPPED_FOR_RECORD` and reports the fresh run path.

## Scope and falsifiers

The incomplete round-82 target is read-only evidence and is never reused or
deleted.  Round 83 writes only fresh target names under its own evidence
directory.  The repair extends the existing deck/record gates rather than
duplicating them.  Admission must pin the same scalar-math binary, CPP card,
input manifests, exact-zero surface file construction, and producer commit.

ASKED: repair the failed rung-0 acquisition without changing the rung physics,
then admit and score it only if the complete record exists.

UNASKED: changing `nn_fsbc`, changing any physical selector or coefficient,
editing NEMO source, reusing the partial target, changing the shipped ORCA2
card or its sea-ice debt tuple, adding a stabilizer, or relaxing any gate.
