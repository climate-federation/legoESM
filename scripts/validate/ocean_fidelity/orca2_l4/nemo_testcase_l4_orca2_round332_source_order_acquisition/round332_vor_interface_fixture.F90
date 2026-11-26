PROGRAM round332_vor_interface_fixture
   USE, INTRINSIC :: iso_fortran_env, ONLY : real64
   USE, INTRINSIC :: ieee_arithmetic, ONLY : ieee_is_negative
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp=real64, jpi=94, jpj=152, jpk=31
   INTEGER, PARAMETER :: inner_i=90, inner_j=148
   CHARACTER(LEN=16) :: mode
   REAL(wp), ALLOCATABLE :: pu(:,:,:),pv(:,:,:),pu_rhs(:,:,:),pv_rhs(:,:,:)

   CALL get_command_argument(1,mode)
   IF(TRIM(mode) == 'inner') THEN
      ! Deliberate reproduction of the rejected 90x148x31 scratch allocation.
      ALLOCATE(pu(inner_i,inner_j,jpk),pv(inner_i,inner_j,jpk), &
         & pu_rhs(inner_i,inner_j,jpk),pv_rhs(inner_i,inner_j,jpk))
   ELSE
      ALLOCATE(pu(jpi,jpj,jpk),pv(jpi,jpj,jpk), &
         & pu_rhs(jpi,jpj,jpk),pv_rhs(jpi,jpj,jpk))
   ENDIF
   pu=0._wp; pv=0._wp
   pu_rhs=-0._wp; pv_rhs=-0._wp
   CALL checked_vor_een_call(1,1,1,pu,pv,pu_rhs,pv_rhs)
   IF(.NOT.ieee_is_negative(pv_rhs(1,1,1))) &
      & ERROR STOP 'negative-zero replay add was not preserved'
   WRITE(*,'(A,3(1X,I0),1X,L1)') 'R332_VOR_INTERFACE_PASS', &
      & SIZE(pv_rhs,1),SIZE(pv_rhs,2),SIZE(pv_rhs,3),ieee_is_negative(pv_rhs(1,1,1))

CONTAINS

   SUBROUTINE checked_vor_een_call(kt, Kmm, kvor, pu, pv, pu_rhs, pv_rhs)
      INTEGER, INTENT(in) :: kt,Kmm,kvor
      REAL(wp), DIMENSION(:,:,:), INTENT(inout) :: pu,pv,pu_rhs,pv_rhs
      INTEGER, PARAMETER :: declared_shape(3)=(/jpi,jpj,jpk/)
      IF(ANY(SHAPE(pu) /= declared_shape) .OR. &
         & ANY(SHAPE(pv) /= declared_shape) .OR. &
         & ANY(SHAPE(pu_rhs) /= declared_shape) .OR. &
         & ANY(SHAPE(pv_rhs) /= declared_shape)) &
         & ERROR STOP 'vor_een explicit-shape extent mismatch'
      CALL vor_een_interface_replica(kt,Kmm,kvor,pu,pv,pu_rhs,pv_rhs)
   END SUBROUTINE checked_vor_een_call

   SUBROUTINE vor_een_interface_replica(kt, Kmm, kvor, pu, pv, pu_rhs, pv_rhs)
      INTEGER                         , INTENT(in   ) ::   kt
      INTEGER                         , INTENT(in   ) ::   Kmm
      INTEGER                         , INTENT(in   ) ::   kvor
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(inout) ::   pu, pv
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(inout) ::   pu_rhs, pv_rhs
      INTEGER :: ji,jj,jk
      REAL(wp) :: zua,zva
      zua=-0._wp; zva=-0._wp
      DO jk=1,jpk
         DO jj=1,jpj
            DO ji=1,jpi
               pu_rhs(ji,jj,jk)=pu_rhs(ji,jj,jk)+zua
               pv_rhs(ji,jj,jk)=pv_rhs(ji,jj,jk)+zva
            END DO
         END DO
      END DO
      IF(kt+Kmm+kvor == -1) pu(1,1,1)=pv(1,1,1)
   END SUBROUTINE vor_een_interface_replica

END PROGRAM round332_vor_interface_fixture
