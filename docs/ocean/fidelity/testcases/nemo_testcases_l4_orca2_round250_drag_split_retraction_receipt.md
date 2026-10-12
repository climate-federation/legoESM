# ORCA2 round 250 — literal drag replay retracts the round-249 owner

Date: 2026-10-12. Frozen base: `cbc70ff20`. Preregistration commit:
`97b4a160a`. Measurement-arm commit: `b3ba6a4b0`. Retraction commit:
`f6114050f`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round250/`.

This round changes no effective model, deck, selector, carried state,
stabiliser, sea-ice field, or `unmeasured_features` entry. The exact private
stage-mask arm used by round 249 was restored for measurement and then
removed. The final package tree is identical to the frozen base. Results are
separately labelled **independent** and **given NEMO's entry**; every measured
operand row is identical between the two labels. No acquisition was needed.

## Compiled statement and method

The executing R230 build constructs the signed T-point drag rate over its
one-cell halo at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/zdfdrg.f90:215-249`. It then forms
the V-face coefficient, the NOW bottom-minus-barotropic residual, and the
live-depth-scaled V correction at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1404-1463`.

The committed round-250 probe replays those scalar associations offline from
the admitted slow-forcing record and the passive completed stage-entry state.
The oracle statement replays bit-exactly. The candidate statement's own
operands replay to the oracle result, not to the private arm's field labelled
`post_drag_v`. That failed self-replay is the decisive finding: the prior
35-face field was supplied by the private slow-forcing override after the
literal correction, so it cannot attribute the executing drag statement.

## Source-ordered split

Both labels produce this identical table on the 35 northern-fold faces.

| operand/output | unequal | maximum absolute | verdict |
|---|---:|---:|---|
| stored V reciprocal | 0/35 | 0 | exact |
| live V reciprocal | 13/35 | 0.0029717310957946257 | first unequal operand |
| signed V-face drag coefficient | 0/35 | 0 | exact |
| bottom V velocity | 0/35 | 0 | exact |
| carried barotropic V velocity | 0/35 | 0 | exact |
| bottom residual | 0/35 | 0 | exact |
| literal drag increment | 0/35 | 0 | exact |
| private arm `post_drag_v` | 35/35 | 1.6557659420864476e-6 | non-source-exact |

The 13 reciprocal differences are non-load-bearing here: multiplying by the
recorded residual still gives an exact drag increment on all 35 faces. The
coefficient prediction is therefore refuted twice: an earlier reciprocal
operand differs, while the predicted coefficient itself is bit-exact. More
importantly, the complete literal output closes exactly, so neither difference
owns the private arm's 35-face slow-V mismatch.

## Frozen predictions

1. **R250-P1 — REFUTED.** Records, passivity, and oracle self-replay pass, but
   the candidate operands do not replay the private arm's recorded
   `post_drag_v`. The preregistration says this invalidates the split.
2. **R250-P2 — REFUTED.** The live reciprocal is unequal on 13 faces before
   the coefficient; the coefficient is exact; and the literal drag increment
   is exact. No coefficient arm exists to land.
3. **R250-P3 — NOT REACHED.** The atomic stability score is forbidden after
   P1 fails. Neither outcome (a) nor outcome (b) is claimed.
4. **R250-P4 — CONFIRMED.** The private package arm is removed and no physics
   lands.
5. **R250-P5 — CONFIRMED.** All seven implemented classification plants fire;
   focused tests pass.

Evidence hashes: `independent.json`
`b6cca02f4467351b58b236d6ba4447f5bbde937663f6b26db54191504948a9cf`;
`given.json`
`c1bc59a5ea512fecf85075affd7412b98983912373abafc99ac5c83160ce830d`;
`classification.json`
`151fc59d91344b55ebe96d2999430931af51cd273e7800aa50d1a2de577b5d27`;
`round249_retraction.json`
`c585a7fc6cb0f9f2fa1c2ecd5bfecebdf8cd4d637f602f0e7dcf5175b6284543`.

## Retraction and landing disposition

Round 249's statement that the baroclinic bottom-drag correction is the first
non-bit slow-V output is **RETRACTED**. The round-249 classifier now emits
`RETRACTED_R249_DRAG_OWNER_BY_R250` rather than the old owner verdict. This is
a tool-level retraction, not only prose.

Nothing lands. Because the final package tree equals the frozen base, GYRE,
DINO, tanks, OMT ladders, rung 0, rung 10, and month trajectories are
unchanged by construction. No trajectory result is inferred from the private
measurement arm.

## Controls, tests, citations, and review

The round-249 and round-250 focused tests pass 18/18. All seven round-250
plants exit 2 with `PLANT-FIRED`. The citation gate, full fidelity battery,
and independent diff review are recorded below after their final runs.

## OPEN

The source walk returns to the last source-exact boundary: the literal
post-drag V forcing is exact. The next round must remove the non-source-exact
slow-forcing override from the diagnostic and walk the executing chain from
post-drag through wind, the two-dimensional Coriolis subtraction, and the
final V surface-mask association. It must name the first non-bit output of
that direct chain before re-scoring the private fold/transport unit. Round
244's pending Decision 116 remains separate and is not re-emitted.
