MODULE dom_oce
   USE, INTRINSIC :: iso_fortran_env, ONLY : real64
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp=real64, jpi=5, jpj=4, jpk=5, jpkm1=4
   INTEGER :: ntsi=1,ntei=3,ntsj=1,ntej=2,narea=1
END MODULE dom_oce

MODULE in_out_manager
   IMPLICIT NONE
   INTEGER :: nit000=1
END MODULE in_out_manager

MODULE dynzdf_round328_writer
   USE dom_oce, ONLY : wp
   USE, INTRINSIC :: iso_fortran_env, ONLY : int64
   IMPLICIT NONE
CONTAINS
   FUNCTION checksum2(value) RESULT(checksum)
      REAL(wp), INTENT(in) :: value(:,:)
      INTEGER(int64) :: checksum,word
      INTEGER :: ji,jj
      checksum=0_int64
      DO jj=1,SIZE(value,2); DO ji=1,SIZE(value,1)
         word=TRANSFER(value(ji,jj),word); checksum=IEOR(checksum,word)
      END DO; END DO
   END FUNCTION checksum2

   FUNCTION checksum3(value) RESULT(checksum)
      REAL(wp), INTENT(in) :: value(:,:,:)
      INTEGER(int64) :: checksum,word
      INTEGER :: ji,jj,jk
      checksum=0_int64
      DO jk=1,SIZE(value,3); DO jj=1,SIZE(value,2); DO ji=1,SIZE(value,1)
         word=TRANSFER(value(ji,jj,jk),word); checksum=IEOR(checksum,word)
      END DO; END DO; END DO
   END FUNCTION checksum3
END MODULE dynzdf_round328_writer

MODULE lib_mpp
   IMPLICIT NONE
CONTAINS
   SUBROUTINE ctl_stop(message)
      CHARACTER(LEN=*), INTENT(in) :: message
      WRITE(*,*) TRIM(message)
      ERROR STOP 1
   END SUBROUTINE ctl_stop
END MODULE lib_mpp
