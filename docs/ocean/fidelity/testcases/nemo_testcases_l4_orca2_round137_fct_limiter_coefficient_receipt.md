# ORCA2 round 137 — level-3 FCT limiter-coefficient walk

Date: 2026-10-04. Base `76c6f95e9`; admitted measurement commit
`8a14c34f6`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round137`.
All step-36 values are **independent**: rung 0 starts from its own
climatological T/S, zero velocity, and zero sea surface. No given-entry
population is mixed into the walk.

## Verdict

**HELD.** The first non-finite stage-3 FCT value at the round-136 upstream
target `(j,i,k)=(86,159,3)` is the north incident V-face limiter coefficient,
not an upstream flux or antidiffusive flux. The source-aligned statement is
NEMO's V-flux `zcoef = MERGE(...)` and multiplication at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:910-913`.

Every earlier target operand exposed by the passive trace is finite. The
coefficient is `1.0` on the south face and `nan` on the north face
`(87,159,3)`. Its input antidiffusive V flux there is still finite at
`6.012356680224643e301`; multiplication produces `nan`, the corrected
divergence and final RHS become `nan`, caller advection content first becomes
non-finite at `(86,159,3)`, and round 136's literal ZDF solve propagates that
debt to surface `(86,159,0)`.

No physics, production package, configuration, deck, card, selector, carried
state, stabilizer, threshold, sea-ice field, or `unmeasured_features` entry
changes. In particular, the rung-7 one-category SI3 debt remains exactly as
carried. No acquisition is needed.

## Frozen predictions

| ID | Result | Measurement |
|---|---|---|
| R137-P1 | **CONFIRMED** | all 49 ordinary state leaves and the independent T/S/u/v/ssh repeat are bit-identical; four plants fire |
| R137-P2 | **CONFIRMED** | first target non-finite field is `coef_v`, after finite low-order, midpoint, averaged-upstream, upstream-RHS, and antidiffusive rows |
| R137-P3 | **CONFIRMED** | `coef_v` → limited V flux → corrected divergence → final RHS → caller content → pre-ZDF level 3 is non-finite in source order |
| R137-P4 | **CONFIRMED** | base and tip `packages/` tree IDs are both `a369750741385921838cc500f378c491f7125a26`; shared trajectories cannot move |

The first runtime attempt refused before classification because the new
checker asked the admitted round-136 JSON for `content_T`; its self-described
row name is `pre_zdf_content`. A second artifact exposed a stale display-only
`target_jik=[1,49,0]` inherited from round 131 even though its actual target
census used `(86,159,3)`. Both checker defects were fixed and committed before
the final clean run; neither refused artifact is cited for a scientific value.

## Source-ordered target census

The compiled limiter builds the neighbourhood extrema and beta ratios at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:853-878`, then selects the V-face
coefficient from adjacent `zbetup/zbetdo` values at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:910-913`.

| Row | Target incident values | Non-finite? |
|---|---|---:|
| first U/V/W fluxes | finite; largest shown `8.352069232165382e150` | 0 |
| midpoint | `-9.21252654556632e143` | 0 |
| averaged U/V/W fluxes | finite; V north `-6.012356680224643e301` | 0 |
| upstream divergence | `6.28529854808396e291` | 0 |
| RHS after upstream | `1.2541836824526768e291` | 0 |
| antidiffusive V flux | `[0.0, 6.012356680224643e301]` | 0 |
| U limiter coefficient | `[1.0, 0.0]` | 0 |
| **V limiter coefficient** | **`[1.0, nan]`** | **1** |
| W limiter coefficient | `[0.0, 1.0]` | 0 |
| limited V flux | `[0.0, nan]` | 1 |
| corrected divergence | `nan` | 1 |
| divisor | `5.011465733466133` | 0 |
| final RHS | `nan` | 1 |
| caller advection content | first T non-finite `(86,159,3)`; 90 active T cells | 1 |
| pre-ZDF content, round 136 | first T non-finite `(86,159,3)`; 90 active T cells | 1 |

The final `step36_fct_content.json` is stamped to clean commit `8a14c34f6`,
CPU production JIT, fp64/libm, and 35 completed finite steps. The
source-order, passivity, support-census, and round-136-link plants all print
`STATUS PLANT-FIRED`.

## Gates, tests, and review

The production `packages/` tree is object-identical to base, so the certified
ORCA2 rung-0/rung-7, GYRE, LOCK_EXCHANGE, OVERFLOW, and DINO trajectories have
zero possible round-induced movement. This is stronger than rerunning the
same unchanged model tree; it does not claim that their pre-existing debt is
closed.

The focused FCT/ZDF battery passes 52/52. The mandated
`tests/ocean/fidelity -n 12` run reached 97%, emitted exactly four failures,
and entered the campaign's known silent xdist tail; it was interrupted and is
**incomplete, not PASS**. Isolation reproduces exactly the established red
set with 27 passes: round-35 allow-dirty escape scope, the grow-only worktree
stamp ratchet, the GYRE round-129 retained-record provenance stamp, and SI3
scalar-math provenance. No round-137 test fails.

The required `codex exec --sandbox read-only` review could not initialize:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. At V face `(87,159,3)`, expose the two adjacent cells' `zup`, `zdo`,
   `zpos`, `zneg`, `zbt`, `zbetup`, and `zbetdo` in the exact compiled order
   at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:853-878`; name the first
   finite-to-non-finite beta operand before changing production code.
2. Keep the separate global averaged-upstream-flux overflow at `(87,160,5)`
   distinct. It may feed the beta stencil, but that ownership is unmeasured
   until the adjacent-beta trace joins them mechanically.
3. Rung 0 remains incomplete. Do not merge the hierarchy-decks lane or climb
   to rung 1 until the independent month is finite through step 240.
