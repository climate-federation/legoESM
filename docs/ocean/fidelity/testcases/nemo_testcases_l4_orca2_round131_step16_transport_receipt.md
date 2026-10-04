# ORCA2 round 131 — rung-0 step-16 transport walk

Date: 2026-10-03. Base `4a217f925`; measurement tip `1a39813f2`. Every
scientific number is **independent**: legoESM starts from the hierarchy rung-0
card's climatological T/S, zero velocity, and zero sea surface. Decision 52's
recorded-entry bridge is not used. No configuration, forcing, carried state,
stabilizer, sea-ice selector, `unmeasured_features` entry, or `packages/` file
changed.

## Verdict

**HELD.** Round 130's failure repeats exactly: the trajectory is finite through
step 15 and step 16 first returns non-finite T at `(j,i,k)=(1,49,0)`. The
source-ordered walk moves the first measured non-finite boundary upstream from
the returned tracer to the stage-1 transport triplet. `zFu` has 803,640
non-finite values, while `zFv` and `zFw` each have 799,200. The immediately
following after-advection accumulator is wholly non-finite in both T and S
(799,200 each), so later source, stage, and vertical-solve boundaries only
propagate the failure.

The first named executed statements are NEMO's stage-1 horizontal transport
products, which multiply the live face thickness by the barotropically
corrected stage velocity in
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:282-283`. NEMO runs stage 1 before
stages 2 and 3 (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:211-227`). The next
walk must split the transport operands already exposed by the production code:
stage thickness, corrected velocity, and transport average. No stabilizer or
configuration change is authorized.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R131-P1: finite through step 15; step-16 T first fails at `(1,49,0)` | **CONFIRMED** exactly, twice within the final gate and in both earlier attempts. |
| R131-P2: stages 1/2 finite; stage-3 advection first non-finite | **REFUTED**: stage-1 transport is already non-finite. |
| R131-P3: stage-3 advection creates the failure and the solve propagates it | **REFUTED**: every measured stage-3/later boundary is non-finite, but the owner precedes them. |
| R131-P4: broad live-stage observer is passive on rung 0 | **UNMEASURED**: that observer refuses because its return contract requires TKE internals absent from constant-mixing rung 0. Existing post-step write-only tracer observers supplied the narrower boundaries. |
| R131-P5: measurement-only | **CONFIRMED**: no model or card file changed. |

The preregistered stage-3 ownership claim remains retracted. The first failed
attempt and its `live WS-RK3 operand trace is incomplete` refusal remain in
the evidence directory; they support no model number.

## Measurements and controls

At the finite step-15 entry the maxima are T `5.211733572435753e39 K`, S
`8.979011597097724e38 g/kg`, u `4.51613258606195e41 m/s`, v
`1.0942965966925958e47 m/s`, and sea surface `25103.45463857757 m`. These are
finite but already catastrophically amplified; NEMO's admitted rung-0 month
remains finite through kt=240.

The final boundary census is:

- step-15 entry: 0 non-finite values in all five fields;
- stage-1 transport: `zFu=803640`, `zFv=799200`, `zFw=799200`;
- after stage-1 advection and surface-source accumulation: T/S
  `799200/799200` at both boundaries;
- stage-1 and stage-2 tracer states: T/S `430552/430552`;
- stage-3 advection and complete pre-implicit content: T/S
  `799200/799200`;
- pre-implicit concentration: T `492990`, S `442228`;
- returned state: T `492990`, S `442228`, u/v `799200/799200`, and sea
  surface `26640`.

The ordinary step-16 repeat is bit-identical in T, S, u, v, and sea surface,
including NaN payload bits. The stage-1 census and repeat-passivity plants both
exit 2 with `STATUS PLANT-FIRED`. The broad-observer refusal drove the committed
narrowing before any boundary result was accepted.

Separate read-only Codex review returned **independent review unavailable
in-sandbox**: `failed to initialize in-process app-server client: Read-only
file system`.

## Validation

The focused round-131 tests, citation gate and its shifted-line plant, and the
prescribed `tests/ocean/fidelity -n 12` battery are recorded in the round-131
evidence directory. No model file changed, so the ORCA2/GYRE/DINO/tank
trajectory gates are unchanged by construction rather than re-scored.

## OPEN

1. At the same independent step-16 entry, expose stage-1 thickness, corrected
   velocity, and transport-average operands one at a time. Name the first
   non-finite operand before changing physics.
2. If the corrected velocity is first, walk the external-mode result and the
   complete round-129 barotropic boundary/transport chain before relanding any
   part of that held unit.
3. The rung-0 month RMS/max ranking remains unmeasured until the baseline
   reaches kt=240. Do not climb to rung 1 or perform the parked hierarchy
   merges first.

## UNVERIFIED

- Which operand first makes the stage-1 `zFu/zFv` products non-finite.
- The exact earlier finite step at which the catastrophic amplification first
  leaves its normal magnitude range.
- Every rung-0 terminal month RMS/max value and ranking.
- The independent-month effect of round 129's held source unit.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
