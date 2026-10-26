PROGRAM round336_writer_fixture
   USE, INTRINSIC :: iso_fortran_env, ONLY : int64
   USE par_kind, ONLY : wp
   USE par_oce
   USE oce, ONLY : wi
   USE dom_oce
   USE zdf_oce
   USE zdfmfc
   USE ldfslp
   USE round336_trazdf_writer, ONLY : r336_zdfphy_capture
   USE trazdf, ONLY : tra_zdf_observed => tra_zdf
   USE trazdf_reference, ONLY : tra_zdf_reference_run => tra_zdf
   IMPLICIT NONE
   REAL(wp), ALLOCATABLE :: pt_observed(:,:,:,:,:),pt_reference(:,:,:,:,:)
   INTEGER :: i,j,k,n,rank
   CHARACTER(LEN=16) :: arg

   CALL GET_COMMAND_ARGUMENT(1,arg); READ(arg,*) rank; narea=rank+1
   ALLOCATE(e3t_3d(jpi,jpj,jpk),e3w_1d(jpk),r3t(jpi,jpj,jpt),tmask(jpi,jpj,jpk))
   ! Match zdf_oce.F90's A2D(0) declaration for avt/avs exactly.  Bounds
   ! checking must exercise the real interior-only contract.
   ALLOCATE(avt(ntsi:ntei,ntsj:ntej,jpk),avs(ntsi:ntei,ntsj:ntej,jpk))
   ALLOCATE(akz(jpi,jpj,jpk),ah_wslp2(jpi,jpj,jpk))
   ALLOCATE(wi(jpi,jpj,jpk),edmfm(jpi,jpj,jpk),edmftra(jpi,jpj,jpk,jpts))
   ALLOCATE(pt_observed(jpi,jpj,jpk,jpts,jpt),pt_reference(jpi,jpj,jpk,jpts,jpt))
   DO k=1,jpk
      e3w_1d(k)=10._wp+REAL(k,wp)
      DO j=1,jpj; DO i=1,jpi
         e3t_3d(i,j,k)=8._wp+REAL(k,wp)+0.01_wp*REAL(i+j+rank,wp)
         tmask(i,j,k)=1._wp
         akz(i,j,k)=2.e-6_wp*REAL(1+k,wp)
         ah_wslp2(i,j,k)=0._wp
         wi(i,j,k)=0._wp; edmfm(i,j,k)=0._wp
      END DO; END DO
   END DO
   DO k=1,jpk; DO j=ntsj,ntej; DO i=ntsi,ntei
      avt(i,j,k)=1.e-5_wp*REAL(1+i+j+k+rank,wp)
      avs(i,j,k)=1.5e-5_wp*REAL(1+i+j+k+rank,wp)
   END DO; END DO; END DO
   edmftra=0._wp
   tmask(4,3,4:5)=0._wp
   avt(ntsi,ntsj,1)=-0._wp; avs(ntsi,ntsj,1)=-0._wp; akz(1,1,1)=-0._wp
   DO n=1,jpt; DO j=1,jpj; DO i=1,jpi
      r3t(i,j,n)=1.e-5_wp*REAL(i-j+n+rank,wp)
   END DO; END DO; END DO
   pt_observed=-0._wp
   DO n=1,jpts; DO k=1,jpk; DO j=1,jpj; DO i=1,jpi
      pt_observed(i,j,k,n,3)=2._wp+0.1_wp*REAL(i+j+k+n+rank,wp)
      pt_observed(i,j,k,n,1)=1.e-6_wp*REAL(i-j+k+n+rank,wp)
   END DO; END DO; END DO; END DO

   ! This live wet cell differs under ((p2dt*e3t)*stretch)*Krhs.
   e3t_3d(3,3,2)=10800._wp
   r3t(3,3,2)=-1.e-5_wp
   DO n=1,jpts
      pt_observed(3,3,2,n,3)=0._wp
      pt_observed(3,3,2,n,1)=8._wp
   END DO
   pt_reference=pt_observed

   DO n=1,5
      CALL r336_zdfphy_capture(2,n,avt(3:5,3:4,2:4), &
         & avt(3:5,3:4,2:4),avs(3:5,3:4,2:4))
   END DO

   CALL tra_zdf_reference_run(2,3,2,1,pt_reference,1)
   CALL tra_zdf_observed(2,3,2,1,pt_observed,1)
   CALL require_bit_equal(pt_observed,pt_reference)
   WRITE(*,*) 'R336_PATCHED_TRAZDF_FIXTURE_PASS',rank

CONTAINS

   SUBROUTINE require_bit_equal(left,right)
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(in) :: left,right
      INTEGER :: ii,jj,kk,nn,ll
      INTEGER(int64) :: left_word,right_word
      DO ll=1,SIZE(left,5); DO nn=1,SIZE(left,4); DO kk=1,SIZE(left,3)
         DO jj=1,SIZE(left,2); DO ii=1,SIZE(left,1)
            left_word=TRANSFER(left(ii,jj,kk,nn,ll),left_word)
            right_word=TRANSFER(right(ii,jj,kk,nn,ll),right_word)
            IF(left_word/=right_word) ERROR STOP 'patched trazdf differs from unpatched source'
         END DO; END DO
      END DO; END DO; END DO
   END SUBROUTINE require_bit_equal
END PROGRAM round336_writer_fixture
