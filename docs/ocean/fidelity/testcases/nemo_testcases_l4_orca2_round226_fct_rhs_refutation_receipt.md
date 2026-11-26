# ORCA2 round 226 — final FCT RHS association, HELD

Date: 2026-10-10. Base: `35b5f6bff`. Measurement tip: `82e789430`.
Scope: OMT-4, ocean only. All numbers below are **independent** (the OMT-4
card starts from its own corrected climatological state). No selector,
carried state, stabiliser, sea-ice setting, or `unmeasured_features` entry
changed. OMT-5 remains blocked.

## Result

The next source-ordered non-bit statement is NEMO's two-write final FCT
tracer-RHS association. NEMO first writes the averaged-upstream divergence,
already divided by live `Kmm` thickness, at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:598-609`; after
`nonosc` it differences the limited anti-fluxes and adds a second divided
rate at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:318-329`. The active stage program zeroes `Krhs`,
calls advection, and hands that completed rate to its stage-3 implicit tracer
solve at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:600-649` and
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:700-749`.

The committed offline replay uses only admitted passive completed states and
the rank-0 `NEMO_L2_RKTR3_1` record; no executable observer is present. Its
rank ownership check selected the southern half, matched NEMO `Kmm` on every
owned cell, and rejected the northern half on 276,774 values.

| quantity | temperature | salinity |
|---|---:|---:|
| literal-vs-generic RHS unequal cells | 386,574 / 430,552 | 402,500 / 430,552 |
| literal-vs-generic RHS max abs | 1.0333801956502464e-19 | 1.4738373282224826e-19 |
| generic-vs-NEMO RHS rms | 4.0322039396813195e-07 | 2.303114051692021e-06 |
| literal-vs-NEMO RHS rms | 4.032203939681319e-07 | 2.303114051692021e-06 |
| stage output unequal active cells | 704 | 411 |
| stage output max abs | 7.105427357601002e-15 | 1.4210854715202004e-14 |

Thus R226-P1 and R226-P2 are **CONFIRMED**: source association is real and
the consumer is live; restoring generic content association restores the
baseline on 0 unequal cells. The tiny local improvement does not establish
trajectory sufficiency.

## Sufficiency falsifier

The private candidate combined round 224's donor/centred fold association,
round 225's literal `HUGE`, north-neighbour stencil and V-face limiter
association, and this round's divided-RHS/stage association. It completed
steps 1--7 and hit the registered kt=8 `raw-mesh e3w_int must contain only
finite values > 0` refusal. The complete candidate log has SHA256
`973bbd48ba821edff92502b57124cf1d6d560d6dd4acccef930d646bc29f90af`,
identical byte-for-byte to both the round-223 baseline and round-225
candidate. R226-P3 is therefore **REFUTED**. The experimental model commit
`53f5b9c99` was retracted by `989a3fa70`; there is no net `packages/` diff.
No Decision-96 census is manufactured, so R226-P4 is **NOT REACHED**.

## Mechanical controls and validation

The gate at
`nemo_testcase_l4_orca2_round226_fct_rhs_walk.py:2-367` owns the ORCA2 record
schema, rank-half cross-check, RHS and stage associations, sufficiency log
hash, prediction classification, and fail-closed plants. All four planted
violations fired:

- `record-owner`: rank ownership refused;
- `source-association`: zero moved RHS values refused;
- `stage-live`: zero moved stage values refused;
- `sufficiency`: unequal boundary hashes refused.

The focused test module
`test_nemo_testcase_l4_orca2_round226_fct_rhs_walk.py:1-50` covers the clean
classification, measured refutation, and every plant. The round's final
evidence is under `orca2_rounds/round226/`; `fct_rhs_walk_final.json` is the
machine-readable verdict and `omt4_atomic_candidate.log` is the trajectory
falsifier.

Independent review unavailable in-sandbox. The required separate
`codex exec --sandbox read-only` invocation exited 1 before review with
`failed to initialize in-process app-server client: Read-only file system`;
the complete output and exit status are retained as
`independent_review.log` and `independent_review.exit`.

The focused round-226 battery passes 6/6. The prescribed parallel
tests/ocean/fidelity battery collected 3,100 tests and reached 97% plus 56
more completions before its last worker ceased producing output; the idle
wrapper was interrupted and the run is not called PASS. Its four failures
were then rerun together in isolation: 27 passed and exactly the four
registered pre-existing reds remained — the GYRE round-129 spread-floor
record (certified year harness moved), allow-dirty scope ratchet (13 existing
drivers), worktree-stamp ratchet (13 existing report emitters), and SI3
scalar-math provenance gate (`A MY_SRC is not verbatim`). The interrupted
parallel log SHA256 is
`b27d566fc64ff0fd8b42ca1637b0fc287386b56b0538abaaf63cc983ce0a38ae`;
the isolated known-red log SHA256 is
`5e6ddc5e298392a3ff670cbeed7255734855ba464e1ae1100a8a5f770e804854`.

## Prediction disposition

| prediction | disposition |
|---|---|
| R226-P1 final RHS is next non-bit boundary | **CONFIRMED** |
| R226-P2 stage association is live | **CONFIRMED** — 1,115 T+S active values move |
| R226-P3 complete statement is sufficient | **REFUTED** — byte-identical kt=8 refusal |
| R226-P4 landing predicate | **NOT REACHED** |
| R226-P5 controls fire | **CONFIRMED** |

## OPEN

OMT-4 remains HELD and OMT-5 remains blocked. Continue in compiled source
order at the stage-3 implicit tracer solve
(`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:700-749`): replay
the input `Krhs`, vertical-diffusion coefficients and tridiagonal
forward/back substitutions from passive completed states, and name the first
non-bit statement. The known `t3d` off-diagonal/diagonal thickness association
is already exonerated and must not be re-walked. No acquisition is needed for
the present boundary; request one only if the admitted record lacks an operand
required by that split.
