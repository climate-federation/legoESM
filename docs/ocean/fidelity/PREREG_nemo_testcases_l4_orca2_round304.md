# PREREG — ORCA2 round 304: Decision-122 import of G5's certified statements

Frozen before any measurement (2026-10-16). Codex review: paused (quota).

Statements (G5 receipts nemo_testcases_gridladder_g5_round3/round4, gridladder branch):
- A. Implicit ice-ocean top drag (rCdU_top=rCdU_ice; dyn_drg_init residual before wind, stp2d.f90:219 before :228-231; substep, dyn_zdf, barotropic consumers).
- B. Stage-1/2 tra_sbc_RK3 Kmm divisor + EMP dilution (trasbc.f90:284-286) with stage-1 r3t(Kaa) (stprk3_stg.f90:177): option `rk3_stage2_tracer_nemo_stored_thickness`.
- C. Stage-3 salt flux sfx*z1_rho0_e3t (trasbc.f90:308): option `rk3_stage3_salt_nemo_written`, operand `OceanSurfaceForcing.salt_flux_sfx` (raw recorded sfx).

Read off the code before measuring:
- A is NOT importable: ORCA2 resolves ln_lin=T (namelist_cfg:251) with ln_drgimp=T, so rCdU_bot != 0 and pCdU = .5*(bottom pair + top pair); the card runs bottom drag in the matrix and in the substeps, and G5's top-drag implementation (certified with ln_drg_OFF) refuses that composition by construction. A stays off; not ported.
- B and C are ported from gridladder commits 6c1cebfd7 + 62060a46c (package hunks only), default False, selected on the shipped ORCA2 card. C needs the recorded sfx, so it is scoped OFF on rung 0 and OMT-1..4 (no surface forcing; same scoping as round 285's stage-3 heat unit). B stays live on every card.

Predictions (D122 (c)+(d), seven card/label pairs: rung 0, OMT-1..4 independent; rung 10 independent and given-NEMO-entry):
- R304-P1: rung 0 and OMT-1..4 have zero RMS-moving rows (no emp/rnf/sfx there). Falsifier: any moving row.
- R304-P2: rung 10 (both labels): zero exact-row losses and first over-bar row equal or toward. Falsifier: any exact loss or away first over-bar row.
- R304-P3: rung 10 kt-1 S rows move only at the last digits (relative change <= 1e-12), i.e. the card's stage-3 salt source holds no term other than sfx and runoff. Falsifier: a kt-1 S max change > 1e-12 relative.
- R304-P4: GYRE ten-step residuals and 30 daily snapshots byte-identical (both options default False).
