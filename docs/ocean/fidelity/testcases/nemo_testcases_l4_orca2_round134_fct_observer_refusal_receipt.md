# ORCA2 round 134 — rung-0 FCT observer refusal

Date: 2026-10-04. Base `6b08fe8c3`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round134`.
Every trajectory statement is **independent**: legoESM starts from the rung-0
card's own climatological T/S, zero velocity, and zero sea surface. Decision
52's recorded-entry bridge is not used.

## Verdict

**HELD.** Round 133's first measured non-finite boundary remains the
stage-3 FCT call, but this round does not name an arithmetic statement inside
it. Both preregistered observers changed the ordinary returned step-36 bits
and refused themselves. Every scientific array obtained through those
observers is inadmissible and is not reported here.

No configuration, forcing, carried state, stabilizer, sea-ice selector,
`unmeasured_features` entry, or `packages/` file changed. The shipped ORCA2
card and its ice tuple remain untouched. No NEMO acquisition is needed.

## Compiled source and exact missing operand

The resolved ORCA2 dispatcher takes FCT at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:529,531-532`. Within that compiled
routine the required source order is:

1. the two-step upstream predictor and low-order fluxes at
   `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:495-609`;
2. the horizontal and vertical antidiffusive fluxes at
   `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:193-197,260-266`;
3. the limiter call and implementation at
   `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:316,743-938`;
4. the corrected divergence and final RHS at
   `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:320-329`.

The first attempt enabled the existing same-JIT FCT trace. The second called
ordinary FCT and copied its ten exact inputs through an ordered callback,
then evaluated the trace in an isolated JIT. Both returned
`STATUS REFUSE: write-only FCT observer moved the ordinary step` after the
production CPU/fp64/libm trajectory had reached step 35. The logs are
`step36_fct.log` and `step36_fct_passive.log` in the evidence root.

The existing narrow stage-transport hook is not an exact substitute. On this
card stage 3 partitions the vertical transport into explicit `ww` and
implicit `wi`; the hook exposes only the post-partition explicit operand,
while the compiled first predictor consumes both. Inferring `wi` from `ww`
would reconstruct a source operand and could change the association order.
The walk therefore stops before its first unmeasured statement.

## Prediction ledger

| ID | Result | Evidence |
|---|---|---|
| R134-P1 | **UNMEASURED** | both observers failed before a baseline result could be admitted |
| R134-P2 | **UNMEASURED** | no admissible internal FCT trace |
| R134-P3 | **UNMEASURED** | no admissible target-cell trace |
| R134-P4 | **UNMEASURED** | downstream propagation was not scored |
| R134-P5 | **REFUTED** | same-JIT trace moved the ordinary returned state |
| R134-P6 | **CONFIRMED** | measurement-only diff; no model/card/selector change |
| R134-P7 | **REFUTED** | ordered input callback also moved the ordinary returned state |

The two refutations are retained in the preregistration. No post-hoc boundary
is promoted to a campaign claim.

## Controls, tests, and review

The focused round-134 gate battery passes **7/7**, including source-order,
passivity, and support plants. Both real measurement attempts independently
exercise the runtime passivity refusal.

The single prescribed `tests/ocean/fidelity -n 12` battery reached 98% with
six failure marks, then entered the established silent tail and was
interrupted. It is **incomplete, not PASS**; no second broad battery was run.
The round-134 focused file had already passed separately.

The required separate `codex exec --sandbox read-only` review did not reach
the diff: `failed to initialize in-process app-server client: Read-only file
system`. Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. Add one private post-step exposure for the exact stage-3 implicit tracer
   transport (and only the other already-materialized FCT inputs that cannot
   be recovered bitwise), prove every diagnostic run passive against the
   ordinary step-36 state, and re-run this source-ordered walk. Because that
   seam touches `packages/`, it must carry the full shared GYRE gate even
   though no public card selects it.
2. The first non-finite arithmetic statement inside rung-0 stage-3 FCT
   remains **UNMEASURED**. Do not skip to the limiter or final RHS.
3. The independent rung-0 month remains unmeasured at step 240. Do not merge
   the parked hierarchy decks or climb to rung 1 until rung 0 is finite.
4. Round 129's barotropic association arm remains held; its salinity exposure
   is not part of this tree.
