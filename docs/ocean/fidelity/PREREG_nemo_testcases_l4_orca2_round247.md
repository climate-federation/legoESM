# Preregistration — ORCA2 round 247 resolved stage V-mask unit

Date: 2026-10-11. Frozen base: `89c1bb679`. Scope: route the WS-RK3 stage
transport's three-dimensional face masks through the already-landed D114
resolved-mesh operand builder, then re-score the complete round-217 unit as one
atomic source unit. Results remain separately labelled **independent** and
**given NEMO's entry**. No deck, selector, carried state, stabiliser, sea-ice
field, or `unmeasured_features` entry may change.

## Cited source unit

The executing record build forms the stage correction and V transport at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-285`:
`zvb = vn_adv*r1_hv(Kmm)-vv_b(Kmm)` followed by
`zFv = e1v*e3v(Kmm)*(vv(Kmm)+zvb*vmask)`. NEMO reads the reference face
thicknesses from the domain file at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domzgr.f90:184-188`, associates the
three-dimensional face masks at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:206-249`, and exposes no
selector between those operands and the stage transport statement.

The production candidate reuses `nemo_qco_resolved_mesh_operands`; it does not
construct another mask or geometry route. Its atomic landing unit is the
round-217 external-mode unit plus the round-245 raw surface V-mask/T-pivot pair
and this resolved stage-mask routing. Partial members remain diagnostic arms
and cannot land.

## Frozen predictions and falsifiers

1. **R247-P1 — stage operand closure.** Under both labels, the five-operand
   replay changes the three-dimensional V-mask and completed `zFv` rows from
   668 unequal to 0, while `e1v`, live `e3v`, `vv`, `zvb`, `vn_adv`, `r1_hv`,
   and `vv_b` remain 0 unequal. Any nonzero row **REFUTES** the routing claim.
2. **R247-P2 — direct cancelling-pair closure.** Midpoint V, midpoint live V
   depth, and `zhV` remain 0 unequal at substeps 3--6 for both labels. Any
   nonzero row **INVALIDATES** the candidate before trajectory scoring.
3. **R247-P3 — OMT-4 survival.** The complete atomic unit removes the former
   `raw-mesh e3w_int must contain only finite values > 0` refusal and completes
   OMT-4 kt=8--10 under both labels. A refusal at kt=8 or earlier **REFUTES**
   the complete-partner claim; retain the first boundary and do not land.
4. **R247-P4 — Decision 96 landing.** Only P1--P3 green activates the full
   OMT-1..4, independent rung-0, given-entry rung-10, GYRE year, DINO, tank,
   moved-row, citation, review, and test gates. The unit lands only if the
   first-over-bar row is toward or unchanged, a majority of RMS-moved rows are
   toward NEMO, no exact row is lost beyond the floor, SSH maximum is not
   worse, and every shared gate stays inside its standing bar.
5. **R247-P5 — controls and blast radius.** The V-mask, `zFv`, kt=8 survival,
   label-coverage, moved-row, citation, and route plants must each refuse.
   GYRE's analytic-mesh route must remain byte-identical; generic cards without
   raw NEMO operands keep the reconstructed masks they already use.

## Stop conditions

No configuration choice or NEMO acquisition is authorised. A failed
prediction is retained as a refutation; it is never repaired by a stabiliser,
a relaxed bar, or a partial landing. Decision 116 remains pending separately
and is not re-emitted.
