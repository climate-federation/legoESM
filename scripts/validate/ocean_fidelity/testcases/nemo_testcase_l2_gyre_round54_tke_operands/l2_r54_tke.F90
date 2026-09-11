MODULE l2_r54_tke
   !! Round-54 WRITE-only kt=2 TKE operand recorder.  Captured values are
   !! never read back by NEMO and the inactive arm is a no-op.
   USE dom_oce
   USE zdf_oce,        ONLY : en, rn2, rn2b
   USE in_out_manager, ONLY : lwp, nit000
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r54_tke_begin, r54_tke_matrix_row, r54_tke_avn_row, r54_tke_finish

   CHARACTER(LEN=16), PARAMETER :: r54_magic = 'NEMO_L2_R54TKE1'
   INTEGER, SAVE :: r54_unit = -1
   LOGICAL, SAVE :: r54_active = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r54_zdiag, r54_zup, r54_zlow, r54_rhs
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r54_mxlm, r54_mxld, r54_pdlr

#  include "domzgr_substitute.h90"

CONTAINS

   SUBROUTINE put0(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(r54_unit) field ; WRITE(r54_unit) 0, 1, 1, 1
      WRITE(r54_unit) value
   END SUBROUTINE put0

   SUBROUTINE put3(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(r54_unit) field ; WRITE(r54_unit) 3, SIZE(value,1), SIZE(value,2), SIZE(value,3)
      WRITE(r54_unit) value
   END SUBROUTINE put3

   SUBROUTINE put_live(name, family, level)
      CHARACTER(LEN=*), INTENT(in) :: name
      INTEGER, INTENT(in) :: family, level
      INTEGER :: ji, jj, jk
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: z
      ALLOCATE(z(jpi,jpj,jpk)); z(:,:,:) = 0._wp
      DO jk=1,jpk ; DO jj=1,jpj ; DO ji=1,jpi
         SELECT CASE(family)
         CASE(1) ; z(ji,jj,jk)=e3t(ji,jj,jk,level)
         CASE(2) ; z(ji,jj,jk)=e3w(ji,jj,jk,level)
         CASE DEFAULT ; CALL ctl_stop('round54: invalid e3 family')
         END SELECT
      END DO ; END DO ; END DO
      CALL put3(name,z); DEALLOCATE(z)
   END SUBROUTINE put_live

   SUBROUTINE r54_tke_begin(kt,Kbb,Kmm,p_sh2,p_avm,p_avt,p_dissl)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: p_sh2,p_avm,p_avt,p_dissl
      INTEGER :: ios
      r54_active = lwp .AND. kt == nit000+1
      IF(.NOT.r54_active) RETURN
      IF(r54_unit /= -1) CALL ctl_stop('round54: nested TKE record')
      IF(l_istiled) CALL ctl_stop('round54: whole-array writer refuses tiling')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round54: writer requires 64-bit wp')
      OPEN(NEWUNIT=r54_unit,FILE='oracle_tke_operands_kt00000002.bin', &
         & ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round54: cannot open TKE record')
      WRITE(r54_unit) r54_magic
      WRITE(r54_unit) 1,kt,Kbb,Kmm,jpi,jpj,jpk,jpkm1,ntsi,ntei,ntsj,ntej,STORAGE_SIZE(1._wp)
      CALL put0('rn_Dt           ',rn_Dt)
      CALL put3('en_entry        ',en)
      CALL put3('avm_entry       ',p_avm)
      CALL put3('avt_entry       ',p_avt)
      CALL put3('dissl_entry     ',p_dissl)
      CALL put3('rn2             ',rn2)
      CALL put3('rn2b            ',rn2b)
      CALL put3('sh2             ',p_sh2)
      CALL put_live('e3t_Kmm        ',1,Kmm)
      CALL put_live('e3w_Kmm        ',2,Kmm)
      ALLOCATE(r54_zdiag(jpi,jpj,jpk),r54_zup(jpi,jpj,jpk), &
         & r54_zlow(jpi,jpj,jpk),r54_rhs(jpi,jpj,jpk), &
         & r54_mxlm(jpi,jpj,jpk),r54_mxld(jpi,jpj,jpk), &
         & r54_pdlr(jpi,jpj,jpk))
      r54_zdiag=0._wp; r54_zup=0._wp; r54_zlow=0._wp; r54_rhs=0._wp
      r54_mxlm=0._wp; r54_mxld=0._wp; r54_pdlr=0._wp
   END SUBROUTINE r54_tke_begin

   SUBROUTINE r54_tke_matrix_row(jj,zdiag,zup,zlow,p_rhs,p_pdlr)
      INTEGER, INTENT(in) :: jj
      REAL(wp), DIMENSION(:,:), INTENT(in) :: zdiag,zup,zlow
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: p_rhs,p_pdlr
      IF(.NOT.r54_active) RETURN
      IF(SIZE(zdiag,1) /= jpi .OR. SIZE(zdiag,2) /= jpk) &
         & CALL ctl_stop('round54: unexpected TKE row shape')
      ! jpk is not part of the solve; p_pdlr is defined only on 2:jpkm1.
      ! Leave every non-consumed slot at the explicit zero fill from begin.
      r54_zdiag(:,jj,1:jpkm1)=zdiag(:,1:jpkm1)
      r54_zup(:,jj,1:jpkm1)=zup(:,1:jpkm1)
      r54_zlow(:,jj,1:jpkm1)=zlow(:,1:jpkm1)
      r54_rhs(:,jj,1:jpkm1)=p_rhs(:,jj,1:jpkm1)
      r54_pdlr(:,jj,2:jpkm1)=p_pdlr(:,jj,2:jpkm1)
   END SUBROUTINE r54_tke_matrix_row

   SUBROUTINE r54_tke_avn_row(jj,zmxlm,zmxld)
      INTEGER, INTENT(in) :: jj
      REAL(wp), DIMENSION(:,:), INTENT(in) :: zmxlm,zmxld
      IF(.NOT.r54_active) RETURN
      IF(SIZE(zmxlm,1) /= jpi .OR. SIZE(zmxlm,2) /= jpk) &
         & CALL ctl_stop('round54: unexpected mixing-length row shape')
      r54_mxlm(:,jj,1:jpkm1)=zmxlm(:,1:jpkm1)
      r54_mxld(:,jj,1:jpkm1)=zmxld(:,1:jpkm1)
   END SUBROUTINE r54_tke_avn_row

   SUBROUTINE r54_tke_finish(p_avm,p_avt,p_dissl)
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: p_avm,p_avt,p_dissl
      IF(.NOT.r54_active) RETURN
      CALL put3('matrix_diag     ',r54_zdiag)
      CALL put3('matrix_upper    ',r54_zup)
      CALL put3('matrix_lower    ',r54_zlow)
      CALL put3('rhs_pre_sweep   ',r54_rhs)
      CALL put3('en_post_sweep   ',en)
      CALL put3('mxl_momentum    ',r54_mxlm)
      CALL put3('mxl_dissipation ',r54_mxld)
      CALL put3('pdlr            ',r54_pdlr)
      CALL put3('avm_output      ',p_avm)
      CALL put3('avt_output      ',p_avt)
      CALL put3('dissl_output    ',p_dissl)
      CLOSE(r54_unit); r54_unit=-1; r54_active=.FALSE.
      DEALLOCATE(r54_zdiag,r54_zup,r54_zlow,r54_rhs,r54_mxlm,r54_mxld,r54_pdlr)
   END SUBROUTINE r54_tke_finish
END MODULE l2_r54_tke
