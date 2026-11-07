# SWG double-gyre round 3 preregistration

Date: 2026-10-12. Incoming tip:
`129fa0f9c6456e192510de5c040536b266d12ae3`. No Round 3 performance,
scan-parity, or double-gyre ladder result has been produced or opened before
this registration. Logs will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round3/`.

## Daily-clock repair

The compiled daily writer evaluates
`NINT(rday / rn_Dt)` in `stprk3.F90`. The shared acquisition preflight will
compute that same positive NINT result and accept a deck only when
`86400/rn_Dt` is exactly that integer. It will print the refusal and the NINT
cadence when the quotient is fractional.

CONFIRM if the shipped 3,600 s deck resolves to 24 steps/day, the Maddison
200 s month resolves to 432 steps/day, all four no-run preflights pass, and a
planted `rn_Dt = 50.5` is refused while naming NINT cadence 1,711. REFUTE if
the plant is accepted, either valid deck is refused, or a different cadence is
certified.

The three source-exact patch files retain their bytes. Their intentional
trailing whitespace will be declared as path-specific Git whitespace
exemptions; `git diff --check` over the campaign range must then pass.

## Non-certified GPU daily scan

The certified CPU mode retains the per-step Python production dispatch. The
non-certified GPU mode will close the steady forcing over one JIT-compiled
`lax.scan` program of exactly one model day (1,728 steps on the run deck),
called once per daily output. No physics, timestep, or output cadence changes.

On physical GPU 1 under `PrecisionPolicy.fp64(transcendentals="native")`, a
warm 100-step wall-clock profile with a final `block_until_ready` will measure
both the retained Python loop and a same-length scan from the same seeded
state. The existing operator observation is about 45 ms/step for the Python
loop. CONFIRM the performance target if the scan is below 5 ms/step; REFUTE it
otherwise. Compilation time is excluded and recorded separately.

A two-model-day comparison uses the full `nn_SWG=16`, 640 by 640 wet-cell
deck and the production daily scan twice versus 3,456 production Python
steps, both from the same fp64 state and steady forcing. CONFIRM parity only if
all `ssh`, native `u`, and native `v` values are finite and each field's
maximum absolute difference is at most `1e-12`. Last-bit differences are
expected; any larger difference REFUTES parity. A tiny-grid direct test applies
the same bound and also requires nonzero wind response so the comparison is
not a rest-state identity.

## Conditional double-gyre ladders

The preregistered Round 1 bars and controls are unchanged. If both immutable
admission JSON files exist after the tool and driver work, run the ten-step
6.25 km given-entry and chained stage ladders and the thirty-day 25 km daily
ladder under the certified CPU policy. Every row must be bit-identical and
each one-ULP plant must exit nonzero. If either admission is absent, record
`ACQUISITION_NEEDED` and leave all double-gyre ladder predictions unmeasured.

## Choices

ASKED: exact daily-clock refusal, path-specific whitespace exemption, one
compiled non-certified GPU program per day, 100-step profile, two-day parity
check, and conditional ladders.

UNASKED: none.
