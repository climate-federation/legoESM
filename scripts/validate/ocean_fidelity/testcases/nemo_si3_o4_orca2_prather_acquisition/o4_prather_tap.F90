MODULE o4_prather_tap
   !! Scalar hand-off only; deliberately independent of NEMO modules so that the
   !! instrumented ice routine and the write-only observer remain acyclic.
   USE par_kind, ONLY : wp
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: o4_capture_cfl, o4_zcfl_delayed, o4_icycle
   REAL(wp), SAVE :: o4_zcfl_delayed=0.0_wp
   INTEGER, SAVE :: o4_icycle=0
CONTAINS
   SUBROUTINE o4_capture_cfl(zcfl_delayed,icycle)
      REAL(wp), INTENT(in) :: zcfl_delayed
      INTEGER, INTENT(in) :: icycle
      o4_zcfl_delayed=zcfl_delayed
      o4_icycle=icycle
   END SUBROUTINE o4_capture_cfl
END MODULE o4_prather_tap
