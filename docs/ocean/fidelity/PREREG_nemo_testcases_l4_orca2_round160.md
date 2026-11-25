# Preregistration — ORCA2 round 160 constant-background EVD composition

Date: 2026-10-05. Frozen base: `427639e609`. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round160/`.

This round qualifies the already-committed GYRE round-236 merge. It changes
only how the implicit vertical solve keeps the constant closure coefficient
separate from NEMO enhanced vertical diffusion before composing them. The
round-159 halo arm remains OFF and HELD. No configuration, initial state,
forcing, carried state, stabiliser, sea-ice selector, or
`unmeasured_features` entry may change.

## Compiled statement and existing refusal

The resolved rung-0 deck selects constant mixing plus enhanced diffusion:
`round83/.../namelist_cfg:408-421` has `ln_zdfcst=.true.`,
`ln_zdfevd=.true.`, `nn_evdm=0`, `rn_evd=100`, `rn_avm0=1.2e-4`, and
`rn_avt0=1.2e-5`.

The compiled rung-0 program constructs the constant closure fields once at
`ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfphy.f90:205-229`, copies those
separate closure fields into the live tracer and momentum coefficients at
`:347-351`, and only then calls enhanced diffusion at `:359`. The called
statement replaces tracer diffusivity by `rn_evd*wmask` on fired interfaces
at `zdfevd.f90:107-110`; because `nn_evdm=0`, the momentum replacement arm at
`:121-135` does not execute.

The merged legoESM fast path receives a single already-summed coefficient
from the physics pipeline. It correctly refuses to treat that sum as the EVD
field, because doing so would discard the constant closure on stable
interfaces. The isolated repair routes this stated combination through the
existing `compute_vertical_K_profiles` path, which still owns separate
constant-closure and EVD fields and already implements the cited replacement.
No new coefficient formula or selector is introduced.

## Frozen protocol

1. Add a direct regression that instantiates constant vertical mixing plus
   `evd_composition="nemo_replace"`, proves one step completes, and proves the
   fired tracer coefficient is replacement rather than addition while the
   `nn_evdm=0` momentum coefficient retains the constant closure. The planted
   old fast-path routing must raise or produce the old sum.
2. Run the independent rung-0 kt=1..10 ladder first. Register every moved row
   against round 159's production control; no bit-exact row may leave the bar
   and the first non-bit boundary may not move earlier.
3. Only after rung 0 passes, run the given-NEMO-entry rung-7 kt=1..10 ladder.
   Keep claim labels separate. Apply the same exact-row and first-boundary
   predicates and register every moved row.
4. Run GYRE's certified ladder and year under its current registered pins,
   DINO, VORTEX/SMT, LOCK_EXCHANGE, OVERFLOW, generic-card, citation, focused,
   and push gates. Run the full ocean-fidelity battery once, one battery at a
   time. Any model-file citation is mechanically re-anchored.
5. Request the prescribed separate read-only review after the diff. A review
   unavailable in the sandbox is recorded as unavailable, never as PASS.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R160-P1 | The repair preserves the two coefficient owners and implements the compiled call order. | The direct constant+EVD regression completes; stable tracer and all momentum interfaces retain the constant coefficient, while fired tracer interfaces equal `rn_evd` alone. | Construction still refuses, a stable interface loses the closure, a fired tracer interface contains the sum, or momentum is replaced with `nn_evdm=0`. |
| R160-P2 | The merged independent rung-0 ladder becomes executable without an earlier boundary or exact-row loss. | All 200 rows complete; no exact row leaves the bar; first non-bit remains kt=1 stage-1 T or moves later. | Refusal, non-finite value, exact-row loss, or earlier first debt. |
| R160-P3 | The merged given-entry rung-7 ladder remains landing-eligible. | All 200 rows complete under the same exact-row and first-boundary predicates. | Refusal, non-finite value, exact-row loss, or earlier first debt. |
| R160-P4 | Cards that do not execute constant+NEMO-replacement composition remain within their registered gates. | GYRE ladder/year, DINO, VORTEX/SMT, tanks, and generic cards pass, with every movement registered. | Any unregistered move, certified-row loss, earlier debt, or registered tolerance failure. |
| R160-P5 | The round-159 halo candidate remains absent. | The production external-mode association selector remains unchanged and the rung-0 card census reports the same held arm OFF. | Any halo production hunk, selector, or card field changes. |

## Terminal rule

Landing requires R160-P1 through R160-P5 plus the citation and repository
gates. A red rung-0 gate ends the round HELD before rung 7. A red shared-card
gate leaves the merge scientifically unqualified. No stabiliser, tolerance
relaxation, per-card routing choice, or configuration change is permitted.

ASKED choices: none. UNASKED choices: empty.
