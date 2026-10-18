# PREREG — ORCA2 round 310: rung G8's lateral-momentum statements under Decision 122

Frozen before any measurement (Claude round; codex review: paused (quota)).
Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round310_g8_d122_gate.py`.
CPU, fp64, libm, production JIT.  Records as round 309 (seven D96 pairs).

Resolved before measuring (resolved configs + each card's own deck):
- Every card: operator NEMO div-curl laplacian, iso-level, e3-weighted,
  coefficient read from eddy_viscosity_3D.nc (nn_ahm_ijk_t = -30), free slip,
  no biharmonic/Smagorinsky/Leith.  This is ORCA2_OMIP_L4/EXP00's
  &namdyn_ldf (lap, lev, -30).
- Rung 0, OMT-3, OMT-4, rung 10: A_h = 1e5 (selector only), diffusion on;
  their decks set ln_dynldf_lap = T.
- OMT-0, OMT-1, OMT-2: A_h = 0, diffusion off; their decks set
  ln_dynldf_off = T (the hierarchy's reduced decks).  Card = deck.
- So no card differs from its NEMO deck: no DECISION on settings.

| ID | Prediction | Falsifier |
|---|---|---|
| R310-P1 | AREA, FOLDR, R3F move 0 rows on OMT-1 and OMT-2 (diffusion off). | any row moves. |
| R310-P2 | Each of AREA, FOLDR, ORDER, R3F, and G8 together, passes D122 (c)+(d) on rung 0, OMT-3, OMT-4 and rung 10 (both labels). | an exact row lost, or first over-bar `away`, or a refusal. |
| R310-P3 | AREA and FOLDR each move at least one row on rung 0, OMT-3, OMT-4 and rung 10 (the statements are live there). | 0 rows moved. |
| R310-P4 | The default-off import leaves GYRE (ten-step residuals, day 30), DINO and VORTEX step jaxprs identical. | any difference. |
