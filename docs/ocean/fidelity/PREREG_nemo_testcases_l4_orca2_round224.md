# Preregistration: ORCA2 round 224 — OMT-4 tracer fold walk

Date: 2026-10-10. Frozen base: `f969dce99`. Scope: walk the OMT-4
tracer-advection interaction from passive stage states, in compiled source
order, and name the first statement that the complete round-217 vector unit
exposes. This file is committed before inspecting any OMT-4 frame payload or
running any legoESM trajectory in this round.

No NEMO run, configuration choice, carried-state change, stabiliser, sea-ice
selector, threshold, or `unmeasured_features` change is authorised. The
complete fold/association/V-transport unit remains atomic and private unless a
cited statement passes the full Decision-96 landing gate. Replays are CPU,
JIT, fp64/libm. No in-executable observer is permitted: only the admitted
completed entry/stage states and pure operator calls may supply numbers.

## Source-ordered candidate

The resolved FCT dispatcher runs centred advection at stages 1--2 and FCT only
at stage 3 (`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:497-540`).
NEMO's centred stages form the V-face tracer from the current cell and its
north halo at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_cen.f90:149-160`.
legoESM's literal centred-stage helper already calls the grid-aware
`interp_to_v_points`, so the T-pivot fold is represented there.

At stage 3, NEMO's first FCT statement forms the donor-cell V flux from the
before tracer and the exchanged north halo at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:495-539`, especially
`:501-504`; its second donor-cell pass reuses the exchanged midpoint halo at
`:562-573`. The high-order V flux follows at `traadv_fct.f90:191-199`.
legoESM's `fct_tracer_advection` calls `upwind_to_v_points` twice and
`centred2_to_v_points` once without passing the grid; those helpers therefore
apply their regular-wall north boundary instead of ORCA2's T-pivot fold.

## Frozen predictions and falsifiers

1. **R224-P1 — centred stages are exonerated.** CONFIRM: on the admitted
   OMT-4 stage-1 and stage-2 states, the literal centred helper's north V-face
   tracer equals the T-pivot source association, while a planted wall/identity
   north row changes at least one active fold face. REFUTE: the live helper
   differs from the source association or the control is vacuous.
2. **R224-P2 — first stage-3 difference is donor-cell V.** CONFIRM: for both T
   and S, the first donor-cell V-face reconstruction differs between the
   current no-grid call and the grid-aware T-pivot call on at least one of the
   complete unit's nonzero northern-fold transports; U and vertical donor-cell
   statements before it remain unchanged. REFUTE: no active fold value moves,
   an earlier U/vertical statement moves, or the difference is outside the
   northern fold.
3. **R224-P3 — both donor-cell passes share the defect.** CONFIRM: substituting
   the grid-aware V reconstruction in the pure stage-3 FCT replay changes the
   first and midpoint upstream V fluxes in NEMO's source order and moves the
   replay toward the fold-aware literal result. REFUTE: only a later high-order
   or limiter statement moves, or the substitution has no effect.
4. **R224-P4 — statement sufficiency.** CONFIRM: a one-variable private arm for
   the cited V donor-cell association removes or delays OMT-4's kt=8
   live-W-thickness refusal without changing OMT-3, and the complete atomic
   unit then qualifies under Decision 96. REFUTE: the refusal remains at the
   same boundary or any certified row is lost. A refutation keeps the arm
   private and names the next source-ordered partner (the centred high-order V
   flux, then nonosc) rather than landing a partial fix.
5. **R224-P5 — controls.** Plants for wall north association, source order,
   active-fold support, and statement sufficiency must each refuse. A plant
   that stays green invalidates the corresponding claim.

## Stop conditions

If the existing frames cannot support a passive offline replay, status is
`STOPPED_FOR_RECORD` and a copied, self-describing acquisition is requested;
no inferred operand is called measured. If the first statement is named but a
production arm cannot complete the full gate in this round, status is `HELD`
with the statement and exact next measurement. Failed predictions remain in
the receipt as `REFUTED`.
