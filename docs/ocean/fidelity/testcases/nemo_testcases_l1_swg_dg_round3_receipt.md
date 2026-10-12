# SWG double-gyre round 3 receipt

Date: 2026-10-12

Status: **HELD — GPU PROFILE AND DOUBLE-GYRE LADDERS UNMEASURED**. The
daily-clock preflight is repaired and the non-certified daily scan is
implemented and directly tested. Physical GPU 1 is not exposed in this
sandbox, and the 30-day NEMO admission was absent when the conditional ladder
item was reached. No NEMO build, NEMO integration, or GPU integration was run
by the agent.

## Commits and scope

Incoming tip: `129fa0f9ca173a3ff284a8265041b6ae57564d9e`.

Implementation:

- `03a531574` — frozen Round 3 preregistration;
- `a65f21f5c33eb368383a1ff43bc3469395dcab15` — exact daily-clock refusal,
  path-specific whitespace exemptions, daily scan, profile, and parity driver;
- `1abb6ce36e0a3cfc91b8a6e9aa773b8f44372f6` — binding cadence mutation
  control; and
- `3393127dddb5520584c161ec6c4e2199a38a551f` — passive carry-seed control.

There is no change under `packages/`, no physics or parameter change, and no
change to the card, SWE stage program, forcing formula, or shipped ladder.
The parameterised card still takes resolution, nominal spacing, basin size,
wind amplitude, gravity, viscosity pair, layer drag, beta, f0, timestep, and
run length from its retained resolved namelist. The previously certified
shipped rows are unchanged; they were not rerun in this round.

## Exact daily clock and acquisition preflight

The old shell guard truncated `rn_Dt` before division. A `50.5` s deck could
therefore be certified as `1,728` steps/day even though the compiled writer's
NINT cadence is `1,711`. The shared deck resolver now evaluates the same
positive NINT result as the writer and refuses unless `86400/rn_Dt` equals that
integer exactly. Its refusal prints the resolved quotient and NINT cadence.

The binding mutation control supplies the complete false cadence: 51,840
steps, stock 1,728, and write 1,728. With the fix reverted the test fails
because the deck is accepted and prints `STEPS_PER_DAY=1728`; with the fix
present it passes by refusing with NINT cadence 1,711. Logs:

- `round3/cadence_plant_nonvacuity_v2.log` — expected test failure under the
  reverted guard;
- `round3/cadence_fixed_v2.log` — passing fixed control; and
- `round3/cadence_plant_nonvacuity.log` — retracted first mutation attempt,
  which was not binding because its later 30-day-length check refused.

All four committed no-run preflights pass in
`round3/acquisition_preflights.log`: shipped ten-step, shipped 30-day,
Maddison ten-step, and Maddison 30-day. They print 24 and 432 steps/day for
the two month decks and create every writer target from shipped `SWG` before
the deck source overlay. No `makenemo` or `mpirun` was called.

The three patch files retain their exact bytes. `.gitattributes` exempts only
their embedded source-context lines from `blank-at-eol`; `git diff --check`
therefore passes without changing the patches.

## Non-certified daily scan

| Field | Certified CPU mode | Non-certified GPU mode |
|---|---|---|
| precision | fp64/libm | fp64/native |
| dispatch | one production call per step | one JIT-compiled scan per day |
| forcing | steady explicit argument | same steady forcing closed over |
| daily cadence | 1,728 steps | 1,728 steps |
| certification use | ladder comparison | long-run preview only |

The GPU mode seeds the previous-stress carry with the same steady stress that
the raw first production step selects when the history is absent. A direct
control proves raw and seeded first-step SSH/U/V are bit-identical. The
realistic-timestep tiny-grid control then compares 3,456 Python production
steps with two 1,728-step compiled daily scans under fp64/libm, requires
nonzero wind response, and passes the preregistered `1e-12` maximum-absolute
bound for SSH/U/V. An earlier four-step fixture with two oversized steps/day
failed that bound and is retracted as a parity protocol; it was replaced by
the requested 50 s, two-model-day clock before the final test log.

The current-tip card, ladder-unit, geometry, acquisition, and driver battery
passes **79/79** at `3393127dddb5520584c161ec6c4e2199a38a551f`; the command,
commit, and pytest terminal line are archived in `round3/direct_tests_final.log`.

The committed GPU-1 command refused before integration with
`CUDA_ERROR_NO_DEVICE`, then confirmed that the backend was CPU. The refusal
is archived in `round3/gpu_profile_2d.log`; no before/after ms-per-step value
or full-resolution two-day parity result is claimed. The `<5 ms/step` target
remains unmeasured.

## Conditional ladders

At the conditional decision point,
`oracle_swg_dg_r1/swg_dg_round1_stage_admission.json` existed, but
`oracle_swg_dg_r1_30d/swg_dg_round1_30d_admission.json` did not. The
preregistered rule required both, so neither new-deck ladder nor its one-ULP
plant was run. The AT-BAR prediction remains unmeasured and moves to Round 4.

## Review and evidence corrections

The Round 1 `170/170` and Round 2 `174/174` test-count claims had no archived
logs and are retracted in their receipts. The supported current-tip count is
the logged 79-test battery above.

Independent review is **UNVERIFIED**. Claude's tool-enabled review stalled
behind workspace trust; a supplied-diff, no-tools fallback also stalled.
Codex review could not initialize its app-server on the read-only home mount.
The attempted-review logs are under `round3/`; none produced a verdict, so no
clean-review claim is made.

## Acquisition needed

The operator's 30-day acquisition command was already in progress when this
round began. Await its admission at:

```text
bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/run_30d.sh --deck maddison --run
  -> /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/oracle_swg_dg_r1_30d/swg_dg_round1_30d_admission.json
```

Run the committed GPU profile and full-resolution parity check on physical
GPU 1 in an environment that exposes CUDA:

```text
bash scripts/run/run_swg_double_gyre_gpu.sh --days 2 --profile-steps 100 --compare-days 2 --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round3/gpu_profile_2d_operator
  -> /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round3/gpu_profile_2d_operator/manifest.json
```

## Oracle citations

The instrumented compiled writer applies the cadence as
`NINT(rday / rn_Dt)` at
`SWG_DG_OMIP_L1_P3D/BLD/ppsrc/nemo/stprk3.f90:146-161`. NEMO defines
`rday` as 24 times 60 times 60 seconds at `phycst.F90:29`.

## Choices and unverified items

ASKED: exact fractional-timestep refusal, source-patch whitespace exemption,
one compiled non-certified program per day, 100-step profile, two-day parity
control, and conditional ladder execution.

UNASKED: none.

UNVERIFIED: GPU throughput, full-resolution GPU scan parity, double-gyre
ladders, one-ULP double-gyre plants, and independent review verdicts.
