# PREREG — ORCA2 round 311: split G8's stage-1 order from G1b's KE-gradient guard; D122 row for the stage-1 order

Frozen before any code change or measurement (Claude round; codex review: paused (quota)).
Base `b725669d1` (round 310 landed).  CPU, fp64, libm, production JIT.

Defect (read off the code, round 310): the model hands ONE operator-order
argument to the tendency routine.  With `rk3_stage1_ldf_source_order` it
carries the stage-1 tuple instead of G1b's flag, and the stage-1 block then
re-adds the KE gradient as a plain barrier-wrapped sum, replacing G1b's
guarded dyn_keg subtraction (`_nemo_keg_accumulate`).  NEMO runs both in
one stage-1 accumulator: stp_2D calls dyn_hpg, dyn_ldf, dyn_vor
(ORCA2_OMIP_L4 stp2d.f90:141/:144/:147; G1b's build ORCA2G1_OMIP_L1_P5
:138/:141/:144; G8's build ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1 :141/:144/:147),
then dyn_keg (:164 / :161 / :164) whose subtraction is dynkeg.f90:129-130
in all three.

Fix: the stage-1 order gets its own argument; G1b's flag keeps its own.
With both selected, the stage-1 block applies G1b's guarded subtraction.
Stage-1 order + vector-form momentum advection without G1b's order raises
(NEMO's vector stage 1 always runs dyn_keg's statement).  Stage-1 order with
flux form (G8: dynadv OFF) keeps the plain add, as certified.

| ID | Prediction | Falsifier |
|---|---|---|
| R311-P1 | Every card with the stage-1 order off (all shipped ORCA2 cards, ORCA2G1-flat, G1-G7, G1b, V1-V3, GYRE, DINO, VORTEX x4, TSUNAMI) has a step jaxpr identical to base. | any jaxpr line differs. |
| R311-P2 | G8 (stage-1 order on, flux form, G1b order off) has a step jaxpr identical to base; G1b and G8 certification tests pass on the patched gridladder tree. | jaxpr differs, or a cert test fails. |
| R311-P3 | Shipped-card controls on the fixed tree reproduce round 310's `final/` D122 rows row for row on all seven pairs. | any row differs. |
| R311-P4 | ORDER+guard (stage-1 order with G1b's guard kept) moves at least one row on rung 0, OMT-3, OMT-4, rung 10 (live), and differs from round 310's ORDER arm on OMT-1/OMT-2 (the guard is now applied). | 0 rows moved / identical to round 310's ORDER arm. |
| R311-P5 | ORDER+guard passes D122 (c)+(d) on rung 0, OMT-3, OMT-4, rung 10 both labels: 0 exact rows lost, first over-bar unchanged. | an exact row lost, first over-bar `away`, or a refusal. |

Landing: ORDER+guard on rung 0, OMT-3, OMT-4, rung 10 if R311-P5 holds
(D122 (b): those decks run dyn_ldf; OMT-0..2 run ln_dynldf_OFF, so (b) fails
there and it stays off whatever (c)/(d) say).
