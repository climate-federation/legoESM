MODULE o7_passivity_fixture
   USE, INTRINSIC :: iso_fortran_env, ONLY: real64
   IMPLICIT NONE
CONTAINS
   SUBROUTINE observer_sink(x)
      REAL(real64), INTENT(in) :: x
      IF (x < -HUGE(x)) ERROR STOP
   END SUBROUTINE observer_sink

   SUBROUTINE blk_ice_1_probe(x,y,z,n)
      INTEGER, INTENT(in) :: n
      REAL(real64), INTENT(in) :: x(n),y(n)
      REAL(real64), INTENT(out) :: z
      INTEGER :: i
      z=0._real64
      DO i=1,n; z=z+x(i)*y(i); END DO
#ifdef O7_INJECT_FP
      z=z+1._real64
#endif
#ifdef O7_OBSERVED
      CALL observer_sink(z)
#endif
   END SUBROUTINE blk_ice_1_probe

   SUBROUTINE blk_ice_2_probe(x,y,z,n)
      INTEGER, INTENT(in) :: n
      REAL(real64), INTENT(in) :: x(n),y(n)
      REAL(real64), INTENT(out) :: z
      INTEGER :: i
      z=1._real64
      DO i=1,n; z=z-x(i)/y(i); END DO
#ifdef O7_INJECT_FP
      z=z*1.5_real64
#endif
#ifdef O7_OBSERVED
      CALL observer_sink(z)
#endif
   END SUBROUTINE blk_ice_2_probe

   SUBROUTINE ice_alb_probe(x,y,z,n)
      INTEGER, INTENT(in) :: n
      REAL(real64), INTENT(in) :: x(n),y(n)
      REAL(real64), INTENT(out) :: z
      INTEGER :: i
      z=0._real64
      DO i=1,n; z=z+(x(i)-y(i))*(x(i)+y(i)); END DO
#ifdef O7_INJECT_FP
      z=z-2._real64
#endif
#ifdef O7_OBSERVED
      CALL observer_sink(z)
#endif
   END SUBROUTINE ice_alb_probe

   SUBROUTINE ice_flx_other_probe(x,y,z,n)
      INTEGER, INTENT(in) :: n
      REAL(real64), INTENT(in) :: x(n),y(n)
      REAL(real64), INTENT(out) :: z
      INTEGER :: i
      z=0._real64
      DO i=1,n; z=z+SQRT(ABS(x(i)*y(i))); END DO
#ifdef O7_INJECT_FP
      z=z/2._real64
#endif
#ifdef O7_OBSERVED
      CALL observer_sink(z)
#endif
   END SUBROUTINE ice_flx_other_probe
END MODULE o7_passivity_fixture
