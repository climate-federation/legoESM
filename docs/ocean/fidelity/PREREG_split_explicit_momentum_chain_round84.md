# Preregistration: separate Redi and through-FCT bolus slope stages, round 84

Date: 2026-08-30. Frozen before replay. Round-83 invalid receipt SHA-256 is
`f759087db5680d8ee13f8053000812e7b0a89471f8efc28196f2ead0006f8bb3`.

NEMO constructs the GM bolus transport before dynamics and applies `tra_ldf`
after dynamics (`stpmlf.F90:214-233,528,548`). The dispatcher will retain the
same-stage Naa native slopes for the exported through-FCT bolus while the Redi
explicit tensor alone consumes Kmm geometry. Rows 8.3--8.10 must return to
their unchanged bars; final uslp and live e3u must remain AT-BAR. If zfu is
then AT-BAR, disposition is `REDI_ZFU_T_BOLUS_STAGE_SPLIT_AT_BAR`. If only the
exact-ahtu substitution clears a residual below `1e-12`, disposition is
`REDI_ZFU_T_BOLUS_STAGE_SPLIT_FIXED_AHTU_RESIDUAL`. Any certified-row
regression is invalid and blocks all later Redi flux rows.
