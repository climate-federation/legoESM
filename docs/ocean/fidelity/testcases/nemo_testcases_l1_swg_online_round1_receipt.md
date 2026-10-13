# SWG online learning — round 1 receipt

Date: 2026-10-13

Status: **PREREGISTERED; MEASUREMENTS NOT YET RUN**.

Incoming branch tip: `8f8551c87`. Worktree:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-288230979`.
All round evidence is written below
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round1/`.

## Frozen round-1 measurements

1. The dg50 ten-step and 30-day NEMO ladders produce 10 stage/restart rows and
   30 daily rows at `nn_SWG=2`, `rn_Dt=400 s`, with 216 steps/day. **CONFIRM**:
   every scored row is bit-identical and the one-ULP controls fail.
   **REFUTE**: any unequal cell, missing row, wrong clock, or non-binding plant.
2. The Gaussian filter has length 50 km (`sigma=8` fine cells), mask
   normalization, six-sigma kernel support, native T/U/V staggering, and
   overlap-area averaging. **CONFIRM**: constants preserved to `2e-15`, plane
   wave attenuation within `2e-5` of `exp(-sigma^2 k^2/2)`, and each stagger's
   wet mean preserved to `5e-16`; the unnormalised-wall plant must differ by
   more than `0.1`. **REFUTE**: any failed bound or non-binding plant.
3. Filter all 1095 existing fine daily snapshot and daily-mean frames. Every
   input/output file is content-pinned. **CONFIRM**: 1095 finite 84x84 T/U/V
   frames in each sampling directory and complete manifests. **REFUTE**: a
   missing/non-finite frame, digest mismatch, or deck mismatch.
4. Run dg50 without closure for 7300 days from rest at the selected physical
   parameters. **NUMERICAL HEALTH CONFIRM**: complete finite output, positive
   500 m + SSH, and saved native velocity Courant numbers below one. **REFUTE**:
   any non-finite state, nonpositive thickness, `Cu>=1` or `Cv>=1`. Climate
   numbers are measured over matched days 1-1095 against filtered snapshots;
   no direction is predicted.
5. Fine predictability uses the day-1095 snapshot, unperturbed and with seed-0
   wet-T SSH Gaussian noise of amplitude `1e-6 m`. CPU step cost is measured
   first; the run length is the largest integer model-day count fitting a
   one-hour execution estimate. The reported predictability time is the first
   log-interpolated crossing of `e` times the exact realized initial SSH RMS;
   if it does not cross, the result is explicitly unresolved. No saturation
   verdict is inferred from a partial CPU twin; a 120-day GPU pair remains an
   acquisition.

The fine and shipped card constants are not edited. Their established digests
and ladders must remain unchanged. No NEMO build, MPI run, source edit, GPU run,
stabilizer, viscosity change, or closure implementation is part of round 1.

## OPEN

- Fill all measurement rows, artifact hashes, test counts, card digests,
  acquisition commands, and adversarial-review verdicts after the committed
  instruments run.
- The operator's announced 7300-day fine truth directory is not present at
  preregistration time. Round 1 therefore filters the complete 1095-day run and
  uses its matched window; the 20-year filtered truth remains open.
