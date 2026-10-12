# NEMO SWG level-1 round 8 — shipped multi-year rung

Date: 2026-10-11. Base: `d250ce95a`. Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round8`.

## Preregistration

This section is frozen before implementing the round-8 scorer or running any
long legoESM integration. Round 7b has content-pinned the admitted ten-step and
thirty-day records; its clean citation gate passes and its shifted-citation
plant exits nonzero.

The oracle is the operator-acquired shipped reference run at
`round0_probe/year10`: 172800 one-hour steps, `nn_write=480`, 360 stored
twenty-day means, and the terminal step-172800 restart. The compiled engine
reads the run controls at
`SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:332-333`, copies `nn_itend` into the
executed limit at `SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:374-377`, and executes
through that limit at `SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:147-160`.

Frozen predictions and falsifiers:

1. **Immutable oracle.** Before reading values, the scorer will require the
   recorded SHA-256 of the manifest, T/U/V/W files, final restart, mesh,
   initial-state output, and resolved `ocean.output`. It will also require the
   manifest's binary and namelist hashes. Any missing or changed byte
   **REFUTES** admission; a one-byte file plant must refuse.
2. **Independent shipped run.** Starting only from the SWG card's shipped rest
   state and analytic shipped wind, normal production `step()` will execute
   all 172800 one-hour steps on CPU under the explicit fp64/libm certification
   policy. No oracle state will enter the model. The run will use 480-step
   blocks and retain only the block diagnostics needed for scoring and
   statistics. Feeding an oracle state, changing the deck, or using a replay
   path **REFUTES** independence.
3. **Exact output semantics.** NEMO passes the just-computed stage-3 state to
   diagnostics after the time-level swap at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:308-321`. The compiled diagnostic is
   called every step and declares averaged output at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:123-126,156-160`; it writes SSH and
   native U/V from that state at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:457` and
   `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:536-538`. The scorer will reproduce
   IOIPSL's source-associated running mean, including its first-sample branch,
   from compiled support source `ext/IOIPSL/src/mathelp.f90:3086-3119`, then
   cast only at the stored output boundary. NEMO declares these three outputs
   as 32-bit records at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:258-259` and
   `SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:363-374`. An instantaneous snapshot,
   a differently associated average, or an fp64-versus-fp32 comparison
   **REFUTES** the instrument.
4. **Bit prediction.** All 1080 stored output rows (360 times times SSH/U/V) are
   predicted bit-identical on their native wet/staggered masks after the
   compiled output cast. All ten fp64 terminal restart fields are predicted
   bit-identical. Therefore the predicted first over-bar output and restart
   rows are both `None`, and every maximum absolute difference is zero. Any
   unequal stored bit is **DEBT**, even if its relative score is at the 1e-15
   numerical bar.
5. **Frontier rule.** The scorer will emit every twenty-day row and the growth
   curve. If a row differs, it will report the first output index, ending step,
   field, unequal-cell count, first unequal wet cell, and maximum absolute
   difference. A twenty-day mean cannot identify an arithmetic statement, so
   no source attribution will be made. Instead, a new daily-cadence acquisition
   around that first window becomes `ACQUISITION_NEEDED`. Attribution without
   that narrower record **REFUTES** the receipt.
6. **Final restart.** NEMO schedules the terminal restart through the compiled
   `nn_stock`/`nitend` logic at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/restart.f90:103-129`, after the just-computed
   time level is selected at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:308-321`. The existing ten-field
   restart bridge, staggered masks, fp64 checks, and bit-exact gate will be
   reused unchanged. Omitting a restart field **REFUTES** coverage.
7. **Statistics after exactness only.** Only if the full rung is bit-identical,
   statistics will be computed from both matched stored series with shared
   diagnostics already in the repository: temporal-mean SSH, the shared
   barotropic transport streamfunction, shared cell-centred volume-integrated
   KE and zonal-anomaly EKE, and the shared isotropic KE spectrum on the 80 by
   80 wet basin. The two sides must be bit-identical inputs and yield identical
   reported statistics. New or duplicated diagnostic numerics, alternate
   weighting, or statistics before fidelity closure **REFUTE** this arm.
8. **Performance.** Wall time will exclude the initial JIT warm-up and include
   all 172800 production steps; peak resident memory will come from the process
   high-water mark. Throughput will be reported per 360-day SWG model year.
   **Correction to the operator shorthand:** the deck's `nn_leapy=30` selects a
   360-day calendar at
   `SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:416-426`, so the shipped run is exactly
   20 model years (19.71 365-day-year equivalents), not 19.7 model years under
   its own calendar.
9. **Controls and Rule 12.** The running-mean unit test will use nonzero values
   for which changing the source association changes a stored bit. The
   existing recorded-input one-ULP production control must remain DEBT. Given,
   chained, independent, carry, and month rows may not leave the bar, and all
   certified card hashes remain unchanged. The shifted-citation plant must
   exit nonzero. A vacuous plant, earlier frontier, changed certified hash, or
   dirty measurement tree **REFUTES** landing.

No physical, configuration, cadence, weighting, or carried-state choice is
introduced. The shipped deck and requested statistics are fixed by the user;
the statistical implementations reuse the shared diagnostics rather than
creating lane-specific definitions.

## Independent review

Independent review run by the operator.

## Measurement

UNMEASURED. This receipt is committed before the round-8 scorer exists or the
172800-step legoESM run starts.

## OPEN

1. Implement and plant-test the content-pinned shipped-run scorer.
2. Run the 172800-step production integration and classify every row.
3. If exact, compute the requested shared statistics and close the SWG rung;
   otherwise write the daily acquisition for the first twenty-day window.
