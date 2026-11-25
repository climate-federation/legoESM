# ORCA2 round 155 — V-transport production retraction

Date: 2026-10-05. Base `361ff8bda`; preregistration `42b030131`;
measurement tip `30445eeae`. Verdict: **HELD**.

Every measured ORCA2 number below is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity, and zero sea surface. The
given-NEMO-entry rung-7 candidate was not measured after the rung-0 terminal
failure and is explicitly UNMEASURED. No configuration, initial state,
forcing, carried-state form, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Result

The complete four-statement V-transport chain is exact at the admitted
substep-2 boundary, but it is **not safe to land**. NEMO reads raw `hu_0/hv_0`
when it constructs the face depths at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545`, completes
and stores `zhV` at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:568-570`, consumes it
in the separate continuity loop at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:584-591`, and
associates the seven external-mode fields at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`.

The initial production transcription reproduced the proved chain exactly at
kt=1, substep 2:

| boundary | unequal cells | maximum absolute difference |
|---|---:|---:|
| completed V metric transport | 0 | 0 |
| north-minus-south V difference | 0 | 0 |
| complete divergence | 0 | 0 |
| sea surface | 0 | 0 |

The explicitly enabled four-arm control was a no-op against that production
path. T/S/u/v/eta/uu_b/vv_b were also bit-identical. The first remaining
measured boundary was the U trend: 15,789 wet cells differ, maximum
`2.1780653503519722e-08`. The causal artifact is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round155/production_boundary.json`,
SHA-256 `99bf18b4b8b421041a2f1ab653a86a9152bd5907e259f540efc121bd0d32cf8a`.

The independent rung-0 ladder then refused at **kt=8 stage-1 T** because the
candidate was non-finite. A second run of the complete private arm after the
production code was restored refused at the same boundary. This is the round's
first non-bit terminal statement. The candidate production activation was
therefore removed as required by the frozen terminal rule.

## Round-154 retraction

Round 154's assertion that the complete four-prerequisite candidate completed
both ladders with 0/400 moved rows is **RETRACTED**. The private
`barotropic_external_mode_association` flag was routed only inside the
substep-trace-model branch. Ordinary ladder models recorded the flag but did
not execute its association. That comparison therefore exercised only three
of the four prerequisites.

This round moved the private association and T-pivot hook routing outside the
trace-only conditional. With the corrected instrument, the full private arm
now refuses at kt=8 stage-1 T, reproducing the production candidate's failure.
The failed run is retained at `rung0_complete_arm.log`; it is not evidence for
a landing.

After restoration, the independent rung-0 gate again passes kt=1..10. Its
frozen-base comparison has 0/200 moved rows, no status change, first debt
unchanged at kt=1 stage-1 T (the held round-94 `stp2d` boundary), and unchanged
kt=10 stage-3 salinity maximum `0.4156673855360964`. The restored ladder and
comparison SHA-256 values are respectively
`f7a471adaf0445ac7703321e6e2f6d019adfeb4ace1771c3925fc2a4fde5eea6` and
`1f99c8c5057e68d267b83a03aea9bb24aaad6d7f4ed530091ed666c58786a625`.
The given-NEMO-entry rung-7 candidate is
**UNMEASURED_TERMINAL_R155_RUNG0**; no rung-7 propagation claim is made.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R155-P1 production equals the proved arm at the named substep boundary | **CONFIRMED_AT_KT1**: all four named boundaries and all seven exposed state arrays are bit-identical. |
| R155-P2 neither ladder loses an exact row or advances first debt | **REFUTED**: the independent rung-0 candidate refuses non-finite at kt=8 stage-1 T. Rung 7 was not run after that terminal failure. |
| R155-P3 landing is salinity-safe | **UNMEASURED_TERMINAL_R155_RUNG0**: the candidate never reaches kt=10; the restored production maximum remains `0.4156673855360964`. |
| R155-P4 shared-card gates admit production | **UNMEASURED_TERMINAL_R155_RUNG0**: no production landing remains. |
| R155-P5 independent month advances beyond step 36 | **UNMEASURED_TERMINAL_R155_RUNG0**. |

## Retained implementation and shared path

The retained code is measurement-only: a helper maps the bridge-carried raw
NEMO face depths into the compact grid, and corrected private hook plumbing
makes the full arm executable by an ordinary ladder model. Neither is selected
by any card or normal production call.

The required GYRE default-path checks are exact. Its 70-row ten-step comparison
has 0 moved rows, maximum worsening 0 ULP, and unchanged first debt kt=3.
All 30 daily day-1..30 snapshot archives are byte-identical to round 154; the
tip snapshot-set SHA-256 manifest is
`9ca5110d40c5d512eb2754b6519974807964e9445c6190b66fbc04084c3677aa`.
The comparison artifact SHA-256 is
`bbd1ea6d857ebe8f5ac743f0cb0c381330eb0efc1457777817f850f0ba76151d`.

The separate `codex exec --sandbox read-only` review returned **independent
review unavailable in-sandbox**: `failed to initialize in-process app-server
client: Read-only file system`. No independent verdict is claimed.

No DINO, VORTEX, tank, generic-card, rung-7 candidate, or production landing
gate was run after the rung-0 terminal refusal. The preregistered terminal rule
requires the change to remain unlanded; running downstream landing gates could
not change that verdict. ASKED choices: none. UNASKED choices: empty.

## Controls

The materialisation plant now changes exactly one active V-transport cell and
requires the scored boundary to report exactly one difference. It exits
nonzero with `STATUS PLANT-FIRED transport-v-materialization`; log SHA-256
`b2cd9583e61456fa7dc7c4ed6bf0d28d45bf4a33766651d134bf704451ce7169`.
The raw-depth replay also now proves which of the legacy reconstruction and raw
NEMO face depth is the live source instead of assuming the source name.

## OPEN

1. Keep production unchanged. Starting from the corrected complete private
   arm, locate the first finite divergence before kt=8 by scoring each step and
   stage; do not add a stabiliser or relax the non-finite refusal.
2. At that first boundary, split the associated seven fields and their first
   consumers in compiled source order, one recorded operand at a time, to name
   the exact growth owner. The complete association remains the source unit;
   partial substitutions are measurement arms only.
3. Re-run rung 0 before measuring rung 7 or the independent month. A future
   production landing still owes the full GYRE, DINO, VORTEX, tank,
   generic-card, citation, and push-gate predicate.

## Verification

Verification is CPU-only in fp64/x64. The canonical citation map was
mechanically re-anchored with `difflib.SequenceMatcher` when the model helper
was inserted; the default receipt and this receipt both report zero unmapped
citations, zero failures, and zero failing map entries. The planted two-line
shift exits nonzero with the named span at `SYMBOL-NOT-AT-LINE`. Focused unit,
instrument, ladder, and receipt tests pass **46/46**.

The one required `tests/ocean/fidelity -n 12` invocation is **INCOMPLETE**, not
PASS. It collected 2,617 items and reached 99%, showing seven skips and exactly
four failures before its remaining workers stopped reporting. Those four are
the registered pre-existing SI3 scalar-math provenance, GYRE round-129
spread-record stamp, round-35 escape-scope, and worktree-stamp failures already
recorded by round 154. After prolonged quiescence at 99% the invocation was
interrupted once; it produced no terminal summary. No round-155 test failed,
and the battery was not rerun.
