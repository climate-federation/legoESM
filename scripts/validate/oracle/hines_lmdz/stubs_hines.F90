! Minimal stand-ins for the two LMDZ modules hines_gwd.f90 USEs besides
! yoegwd_mod_h (compiled verbatim): dimphy (grid sizes) and yomcst_mod_h
! (physical constants).  Values are set by drv_hines.F90 from the input file,
! so the constants are legoESM's and are perturbed by the F4 spread harness.
MODULE dimphy
  IMPLICIT NONE
  INTEGER, SAVE :: klon, klev, kidia, kfdia
END MODULE dimphy

MODULE yomcst_mod_h
  IMPLICIT NONE
  REAL, SAVE :: rd, rg, rcpd, rpi
END MODULE yomcst_mod_h
