MODULE tke_round336_writer
   !! Passive kt=2 TKE statement recorder.  NEMO arithmetic writes only to
   !! the public copy-out arrays; all procedure calls are outside NEMO loops.
   USE dom_oce
   USE in_out_manager, ONLY : nit000
   USE dynzdf_round328_writer, ONLY : checksum2, checksum3
   USE lib_mpp, ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r336_tke_begin, r336_tke_after_tke, r336_tke_finish
   PUBLIC :: r336_after_bc, r336_after_langmuir, r336_after_rhs
   PUBLIC :: r336_matrix_diag, r336_matrix_upper, r336_matrix_lower
   PUBLIC :: r336_after_solve, r336_pre_sweep_mxl, r336_final_mxl
   PUBLIC :: r336_final_mxld, r336_pre_pdlr_zav, r336_pre_pdlr_avt

   INTEGER, SAVE :: nkt=-1,nkbb=-1,nkmm=-1
   LOGICAL, SAVE :: lactive=.FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: r336_after_bc(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_after_langmuir(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_after_rhs(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_matrix_diag(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_matrix_upper(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_matrix_lower(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_after_solve(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_pre_sweep_mxl(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_final_mxl(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_final_mxld(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_pre_pdlr_zav(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_pre_pdlr_avt(:,:,:)

   REAL(wp), ALLOCATABLE, SAVE :: entry_en(:,:,:),entry_avt(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_avm(:,:,:),entry_sh2(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_dissl(:,:,:),entry_rn2(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_rn2b(:,:,:),entry_wmask(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_drag_bottom(:,:),entry_drag_top(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_nmln(:,:),entry_hmlp(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_fr_i(:,:),entry_hm_i(:,:),entry_taum(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: after_tke_en(:,:,:),after_tke_pdlr(:,:,:)

#  include "do_loop_substitute.h90"

CONTAINS

   SUBROUTINE fill_negative_zero(value)
      REAL(wp), INTENT(out) :: value(:,:,:)
      value=-0._wp
   END SUBROUTINE fill_negative_zero

   SUBROUTINE r336_tke_begin(kt,Kbb,Kmm,pen,pavt,pavm,psh2,pdissl,prn2,prn2b, &
      & pwmask,pbottom,ptop,pnmln,phmlp,pfr_i,phm_i,ptaum)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: pen,pavt,psh2,pdissl
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: pavm,prn2,prn2b,pwmask
      REAL(wp), DIMENSION(A2D(0)), INTENT(in) :: pbottom,ptop,phmlp,pfr_i,phm_i,ptaum
      INTEGER, DIMENSION(A2D(0)), INTENT(in) :: pnmln
      IF(ALLOCATED(r336_after_bc)) CALL ctl_stop('round336 TKE: nested record')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round336 TKE: requires fp64')
      IF(kt == nit000 .AND. (Kbb /= 1 .OR. Kmm /= 1)) &
         & CALL ctl_stop('round336 TKE: kt1 levels moved')
      IF(kt == nit000+1 .AND. (Kbb /= 3 .OR. Kmm /= 3)) &
         & CALL ctl_stop('round336 TKE: kt2 levels moved')
      nkt=kt; nkbb=Kbb; nkmm=Kmm; lactive=.TRUE.
      ALLOCATE(r336_after_bc(A2D(0),jpk),r336_after_langmuir(A2D(0),jpk), &
         & r336_after_rhs(A2D(0),jpk),r336_matrix_diag(A2D(0),jpk), &
         & r336_matrix_upper(A2D(0),jpk),r336_matrix_lower(A2D(0),jpk), &
         & r336_after_solve(A2D(0),jpk),r336_pre_sweep_mxl(A2D(0),jpk), &
         & r336_final_mxl(A2D(0),jpk),r336_final_mxld(A2D(0),jpk), &
         & r336_pre_pdlr_zav(A2D(0),jpk),r336_pre_pdlr_avt(A2D(0),jpk))
      ALLOCATE(entry_en(A2D(0),jpk),entry_avt(A2D(0),jpk), &
         & entry_avm(jpi,jpj,jpk),entry_sh2(A2D(0),jpk), &
         & entry_dissl(A2D(0),jpk),entry_rn2(jpi,jpj,jpk), &
         & entry_rn2b(jpi,jpj,jpk),entry_wmask(jpi,jpj,jpk), &
         & entry_drag_bottom(A2D(0)),entry_drag_top(A2D(0)), &
         & entry_nmln(A2D(0)),entry_hmlp(A2D(0)),entry_fr_i(A2D(0)), &
         & entry_hm_i(A2D(0)),entry_taum(A2D(0)), &
         & after_tke_en(A2D(0),jpk),after_tke_pdlr(A2D(0),jpk))
      entry_en=pen; entry_avt=pavt; entry_avm=pavm; entry_sh2=psh2
      entry_dissl=pdissl; entry_rn2=prn2; entry_rn2b=prn2b
      entry_wmask=pwmask; entry_drag_bottom=pbottom; entry_drag_top=ptop
      entry_nmln=REAL(pnmln,wp); entry_hmlp=phmlp
      entry_fr_i=pfr_i; entry_hm_i=phm_i; entry_taum=ptaum
      CALL fill_negative_zero(r336_after_bc)
      CALL fill_negative_zero(r336_after_langmuir)
      CALL fill_negative_zero(r336_after_rhs)
      CALL fill_negative_zero(r336_matrix_diag)
      CALL fill_negative_zero(r336_matrix_upper)
      CALL fill_negative_zero(r336_matrix_lower)
      CALL fill_negative_zero(r336_after_solve)
      CALL fill_negative_zero(r336_pre_sweep_mxl)
      CALL fill_negative_zero(r336_final_mxl)
      CALL fill_negative_zero(r336_final_mxld)
      CALL fill_negative_zero(r336_pre_pdlr_zav)
      CALL fill_negative_zero(r336_pre_pdlr_avt)
      CALL fill_negative_zero(after_tke_en)
      CALL fill_negative_zero(after_tke_pdlr)
   END SUBROUTINE r336_tke_begin

   SUBROUTINE r336_tke_after_tke(pen,ppdlr)
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: pen,ppdlr
      IF(.NOT.lactive) CALL ctl_stop('round336 TKE: missing begin')
      after_tke_en=pen
      after_tke_pdlr(:,:,2:jpkm1)=ppdlr(:,:,2:jpkm1)
   END SUBROUTINE r336_tke_after_tke

   SUBROUTINE put2(unit,name,value,i0,j0)
      INTEGER, INTENT(in) :: unit,i0,j0
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value(:,:)
      CHARACTER(LEN=24) :: field
      field=name
      WRITE(unit) field
      WRITE(unit) 2,SIZE(value,1),SIZE(value,2),1,i0,j0,1
      WRITE(unit) checksum2(value)
      WRITE(unit) value
   END SUBROUTINE put2

   SUBROUTINE put3(unit,name,value,i0,j0,k0)
      INTEGER, INTENT(in) :: unit,i0,j0,k0
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value(:,:,:)
      CHARACTER(LEN=24) :: field
      field=name
      WRITE(unit) field
      WRITE(unit) 3,SIZE(value,1),SIZE(value,2),SIZE(value,3),i0,j0,k0
      WRITE(unit) checksum3(value)
      WRITE(unit) value
   END SUBROUTINE put3

   SUBROUTINE r336_tke_finish(pavm,pavt)
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: pavm
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: pavt
      INTEGER :: unit,ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=96) :: filename
      IF(.NOT.lactive) CALL ctl_stop('round336 TKE: missing active record')
      IF(nkt == nit000+1) THEN
         WRITE(filename,'("oracle_r336_tke_kt",I8.8,"_r",I4.4,".bin")') nkt,narea-1
         OPEN(NEWUNIT=unit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
            & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
         IF(ios /= 0) CALL ctl_stop('round336 TKE: cannot open record')
         magic='NEMO_L4_R336TKE'
         WRITE(unit) magic,2,nkt,nkbb,nkmm,narea-1,jpi,jpj,jpk,ntsi,ntei,ntsj,ntej, &
            & STORAGE_SIZE(1._wp),31
         CALL put3(unit,'entry_en',entry_en,ntsi,ntsj,1)
         CALL put3(unit,'entry_avt',entry_avt,ntsi,ntsj,1)
         CALL put3(unit,'entry_avm',entry_avm,1,1,1)
         CALL put3(unit,'entry_sh2',entry_sh2,ntsi,ntsj,1)
         CALL put3(unit,'entry_dissl',entry_dissl,ntsi,ntsj,1)
         CALL put3(unit,'entry_rn2',entry_rn2,1,1,1)
         CALL put3(unit,'entry_rn2b',entry_rn2b,1,1,1)
         CALL put2(unit,'entry_drag_bottom',entry_drag_bottom,ntsi,ntsj)
         CALL put2(unit,'entry_drag_top',entry_drag_top,ntsi,ntsj)
         CALL put2(unit,'entry_nmln',entry_nmln,ntsi,ntsj)
         CALL put2(unit,'entry_hmlp',entry_hmlp,ntsi,ntsj)
         CALL put2(unit,'entry_fr_i',entry_fr_i,ntsi,ntsj)
         CALL put2(unit,'entry_hm_i',entry_hm_i,ntsi,ntsj)
         CALL put2(unit,'entry_taum',entry_taum,ntsi,ntsj)
         CALL put3(unit,'after_bc_en',r336_after_bc,ntsi,ntsj,1)
         CALL put3(unit,'post_langmuir_en',r336_after_langmuir,ntsi,ntsj,1)
         CALL put3(unit,'matrix_rhs_en',r336_after_rhs,ntsi,ntsj,1)
         CALL put3(unit,'matrix_diag',r336_matrix_diag,ntsi,ntsj,1)
         CALL put3(unit,'matrix_upper',r336_matrix_upper,ntsi,ntsj,1)
         CALL put3(unit,'matrix_lower',r336_matrix_lower,ntsi,ntsj,1)
         CALL put3(unit,'after_solve_en',r336_after_solve,ntsi,ntsj,1)
         CALL put3(unit,'after_tke_en',after_tke_en,ntsi,ntsj,1)
         CALL put3(unit,'tke_pdlr',after_tke_pdlr,ntsi,ntsj,1)
         CALL put3(unit,'pre_sweep_mxl',r336_pre_sweep_mxl,ntsi,ntsj,1)
         CALL put3(unit,'final_mxl',r336_final_mxl,ntsi,ntsj,1)
         CALL put3(unit,'final_mxld',r336_final_mxld,ntsi,ntsj,1)
         CALL put3(unit,'pre_pdlr_zav',r336_pre_pdlr_zav,ntsi,ntsj,1)
         CALL put3(unit,'pre_pdlr_avt',r336_pre_pdlr_avt,ntsi,ntsj,1)
         CALL put3(unit,'final_avm',pavm,1,1,1)
         CALL put3(unit,'final_avt',pavt,ntsi,ntsj,1)
         CALL put3(unit,'tke_wmask',entry_wmask,1,1,1)
         CLOSE(unit)
      ENDIF
      DEALLOCATE(r336_after_bc,r336_after_langmuir,r336_after_rhs, &
         & r336_matrix_diag,r336_matrix_upper,r336_matrix_lower,r336_after_solve, &
         & r336_pre_sweep_mxl,r336_final_mxl,r336_final_mxld,r336_pre_pdlr_zav, &
         & r336_pre_pdlr_avt,entry_en,entry_avt,entry_avm,entry_sh2,entry_dissl, &
         & entry_rn2,entry_rn2b,entry_wmask,entry_drag_bottom,entry_drag_top, &
         & entry_nmln,entry_hmlp,entry_fr_i,entry_hm_i,entry_taum, &
         & after_tke_en,after_tke_pdlr)
      lactive=.FALSE.
   END SUBROUTINE r336_tke_finish

END MODULE tke_round336_writer
