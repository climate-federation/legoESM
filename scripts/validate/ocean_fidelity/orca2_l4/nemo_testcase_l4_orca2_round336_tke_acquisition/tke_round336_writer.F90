MODULE tke_round336_writer
   !! Passive kt=2 TKE statement recorder.  NEMO arithmetic writes only to
   !! the public copy-out arrays; all procedure calls are outside NEMO loops.
   USE, INTRINSIC :: iso_fortran_env, ONLY : int64
   USE dom_oce
   USE in_out_manager, ONLY : nit000
   USE lib_mpp, ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r336_tke_begin, r336_tke_after_tke, r336_tke_finish
   PUBLIC :: r336_after_bc, r336_after_langmuir, r336_after_rhs
   PUBLIC :: r336_matrix_diag, r336_matrix_upper, r336_matrix_lower
   PUBLIC :: r336_solve_diag, r336_solve_rhs, r336_solve_bottom, r336_solve_back
   PUBLIC :: r336_after_solve, r336_pre_sweep_mxl, r336_after_lup, r336_after_ldown
   PUBLIC :: r336_final_mxl, r336_final_mxld, r336_pre_pdlr_avt

   INTEGER, SAVE :: nkt=-1,nkbb=-1,nkmm=-1
   LOGICAL, SAVE :: lactive=.FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: r336_after_bc(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_after_langmuir(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_after_rhs(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_matrix_diag(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_matrix_upper(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_matrix_lower(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_solve_diag(:,:,:),r336_solve_rhs(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_solve_bottom(:,:,:),r336_solve_back(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_after_solve(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_pre_sweep_mxl(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_after_lup(:,:,:),r336_after_ldown(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_final_mxl(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_final_mxld(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_pre_pdlr_avt(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_entry_gdepw_kmm(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_entry_e3t_kmm(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r336_entry_e3w_kmm(:,:,:)

   REAL(wp), ALLOCATABLE, SAVE :: entry_en(:,:,:),entry_avt(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_avm_k(:,:,:),entry_sh2(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_dissl(:,:,:),entry_rn2(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_rn2b(:,:,:),entry_wmask(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_drag_bottom(:,:),entry_drag_top(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_nmln(:,:),entry_hmlp(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_fr_i(:,:),entry_hm_i(:,:),entry_taum(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_htau(:,:),entry_ssmask(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_mbkt(:,:),entry_mikt(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_tmask(:,:,:),entry_umask(:,:,:),entry_vmask(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_uu_kbb(:,:,:),entry_vv_kbb(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: entry_avtb_2d(:,:),entry_avtb(:),entry_avmb(:)
   REAL(wp), ALLOCATABLE, SAVE :: after_tke_en(:,:,:),after_tke_pdlr(:,:,:)

#  include "do_loop_substitute.h90"

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

   SUBROUTINE fill_negative_zero(value)
      REAL(wp), INTENT(out) :: value(:,:,:)
      value=-0._wp
   END SUBROUTINE fill_negative_zero

   SUBROUTINE r336_tke_begin(kt,Kbb,Kmm,pen,pavt,pavm,psh2,pdissl,prn2,prn2b, &
      & pwmask,ptmask,pumask,pvmask,pssmask,pmbkt,pmikt,pbottom,ptop,pnmln, &
      & phmlp,pfr_i,phm_i,ptaum,phtau,puu_kbb,pvv_kbb,pavtb_2d,pavtb,pavmb, &
      & pgdepw_1d,pe3w_1d,pe3t_3d,pr3t_kmm)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: pen,pavt,psh2,pdissl
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: pavm,prn2,prn2b,pwmask
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: ptmask,pumask,pvmask
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) :: pssmask,pbottom,ptop,pfr_i
      REAL(wp), DIMENSION(A2D(2)), INTENT(in) :: phmlp
      REAL(wp), DIMENSION(A2D(0)), INTENT(in) :: phm_i,ptaum,phtau,pavtb_2d
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: puu_kbb,pvv_kbb
      REAL(wp), DIMENSION(jpk), INTENT(in) :: pavtb,pavmb,pgdepw_1d,pe3w_1d
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: pe3t_3d
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) :: pr3t_kmm
      INTEGER, DIMENSION(A2D(2)), INTENT(in) :: pnmln
      INTEGER, DIMENSION(jpi,jpj), INTENT(in) :: pmbkt,pmikt
      INTEGER :: ji,jj,jk
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
         & r336_solve_diag(A2D(0),jpk),r336_solve_rhs(A2D(0),jpk), &
         & r336_solve_bottom(A2D(0),jpk),r336_solve_back(A2D(0),jpk), &
         & r336_after_solve(A2D(0),jpk),r336_pre_sweep_mxl(A2D(0),jpk), &
         & r336_after_lup(A2D(0),jpk),r336_after_ldown(A2D(0),jpk), &
         & r336_final_mxl(A2D(0),jpk),r336_final_mxld(A2D(0),jpk), &
         & r336_pre_pdlr_avt(A2D(0),jpk), &
         & r336_entry_gdepw_kmm(A2D(0),jpk),r336_entry_e3t_kmm(A2D(0),jpk), &
         & r336_entry_e3w_kmm(A2D(0),jpk))
      ALLOCATE(entry_en(A2D(0),jpk),entry_avt(A2D(0),jpk), &
         & entry_avm_k(jpi,jpj,jpk),entry_sh2(A2D(0),jpk), &
         & entry_dissl(A2D(0),jpk),entry_rn2(jpi,jpj,jpk), &
         & entry_rn2b(jpi,jpj,jpk),entry_wmask(jpi,jpj,jpk), &
         & entry_tmask(jpi,jpj,jpk),entry_umask(jpi,jpj,jpk),entry_vmask(jpi,jpj,jpk), &
         & entry_drag_bottom(jpi,jpj),entry_drag_top(jpi,jpj), &
         & entry_nmln(A2D(2)),entry_hmlp(A2D(2)),entry_fr_i(jpi,jpj), &
         & entry_hm_i(A2D(0)),entry_taum(A2D(0)), &
         & entry_htau(A2D(0)),entry_ssmask(jpi,jpj), &
         & entry_mbkt(jpi,jpj),entry_mikt(jpi,jpj), &
         & entry_uu_kbb(jpi,jpj,jpk),entry_vv_kbb(jpi,jpj,jpk), &
         & entry_avtb_2d(A2D(0)),entry_avtb(jpk),entry_avmb(jpk), &
         & after_tke_en(A2D(0),jpk),after_tke_pdlr(A2D(0),jpk))
      entry_en=pen; entry_avt=pavt; entry_avm_k=pavm; entry_sh2=psh2
      entry_dissl=pdissl; entry_rn2=prn2; entry_rn2b=prn2b
      entry_wmask=pwmask; entry_drag_bottom=pbottom; entry_drag_top=ptop
      entry_nmln=REAL(pnmln,wp); entry_hmlp=phmlp
      entry_fr_i=pfr_i; entry_hm_i=phm_i; entry_taum=ptaum
      entry_htau=phtau; entry_ssmask=pssmask
      entry_mbkt=REAL(pmbkt,wp); entry_mikt=REAL(pmikt,wp)
      entry_tmask=ptmask; entry_umask=pumask; entry_vmask=pvmask
      entry_uu_kbb=puu_kbb; entry_vv_kbb=pvv_kbb
      entry_avtb_2d=pavtb_2d; entry_avtb=pavtb; entry_avmb=pavmb
      CALL fill_negative_zero(r336_after_bc)
      CALL fill_negative_zero(r336_after_langmuir)
      CALL fill_negative_zero(r336_after_rhs)
      CALL fill_negative_zero(r336_matrix_diag)
      CALL fill_negative_zero(r336_matrix_upper)
      CALL fill_negative_zero(r336_matrix_lower)
      CALL fill_negative_zero(r336_solve_diag)
      CALL fill_negative_zero(r336_solve_rhs)
      CALL fill_negative_zero(r336_solve_bottom)
      CALL fill_negative_zero(r336_solve_back)
      CALL fill_negative_zero(r336_after_solve)
      CALL fill_negative_zero(r336_pre_sweep_mxl)
      CALL fill_negative_zero(r336_after_lup)
      CALL fill_negative_zero(r336_after_ldown)
      CALL fill_negative_zero(r336_final_mxl)
      CALL fill_negative_zero(r336_final_mxld)
      CALL fill_negative_zero(r336_pre_pdlr_avt)
      CALL fill_negative_zero(r336_entry_gdepw_kmm)
      CALL fill_negative_zero(r336_entry_e3t_kmm)
      CALL fill_negative_zero(r336_entry_e3w_kmm)
      DO jk=1,jpk
         DO jj=njs0,nje0
            DO ji=nis0,nie0
               r336_entry_gdepw_kmm(ji,jj,jk)=pgdepw_1d(jk)*(1._wp+pr3t_kmm(ji,jj))
               r336_entry_e3w_kmm(ji,jj,jk)=pe3w_1d(jk)*(1._wp+pr3t_kmm(ji,jj))
               r336_entry_e3t_kmm(ji,jj,jk)=pe3t_3d(ji,jj,jk) * &
                  & (1._wp+pr3t_kmm(ji,jj)*ptmask(ji,jj,jk))
            END DO
         END DO
      END DO
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

   SUBROUTINE put1(unit,name,value,k0)
      INTEGER, INTENT(in) :: unit,k0
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value(:)
      CHARACTER(LEN=24) :: field
      INTEGER(int64) :: checksum
      INTEGER :: jk
      field=name; checksum=0_int64
      DO jk=1,SIZE(value); checksum=IEOR(checksum,TRANSFER(value(jk),checksum)); END DO
      WRITE(unit) field
      WRITE(unit) 1,SIZE(value),1,1,1,1,k0
      WRITE(unit) checksum
      WRITE(unit) value
   END SUBROUTINE put1

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
            & STORAGE_SIZE(1._wp),51
         CALL put3(unit,'entry_en',entry_en,ntsi,ntsj,1)
         CALL put3(unit,'entry_avt',entry_avt,ntsi,ntsj,1)
         CALL put3(unit,'entry_avm_k',entry_avm_k,1,1,1)
         CALL put3(unit,'entry_sh2',entry_sh2,ntsi,ntsj,1)
         CALL put3(unit,'entry_dissl',entry_dissl,ntsi,ntsj,1)
         CALL put3(unit,'entry_rn2',entry_rn2,1,1,1)
         CALL put3(unit,'entry_rn2b',entry_rn2b,1,1,1)
         CALL put2(unit,'entry_drag_bottom',entry_drag_bottom,1,1)
         CALL put2(unit,'entry_drag_top',entry_drag_top,1,1)
         CALL put2(unit,'entry_nmln',entry_nmln,1,1)
         CALL put2(unit,'entry_hmlp',entry_hmlp,1,1)
         CALL put2(unit,'entry_fr_i',entry_fr_i,1,1)
         CALL put2(unit,'entry_hm_i',entry_hm_i,ntsi,ntsj)
         CALL put2(unit,'entry_taum',entry_taum,ntsi,ntsj)
         CALL put2(unit,'entry_htau',entry_htau,ntsi,ntsj)
         CALL put3(unit,'entry_uu_kbb',entry_uu_kbb,1,1,1)
         CALL put3(unit,'entry_vv_kbb',entry_vv_kbb,1,1,1)
         CALL put3(unit,'entry_tmask',entry_tmask,1,1,1)
         CALL put3(unit,'entry_umask',entry_umask,1,1,1)
         CALL put3(unit,'entry_vmask',entry_vmask,1,1,1)
         CALL put2(unit,'entry_ssmask',entry_ssmask,1,1)
         CALL put2(unit,'entry_mbkt',entry_mbkt,1,1)
         CALL put2(unit,'entry_mikt',entry_mikt,1,1)
         CALL put3(unit,'entry_gdepw_kmm',r336_entry_gdepw_kmm,ntsi,ntsj,1)
         CALL put3(unit,'entry_e3t_kmm',r336_entry_e3t_kmm,ntsi,ntsj,1)
         CALL put3(unit,'entry_e3w_kmm',r336_entry_e3w_kmm,ntsi,ntsj,1)
         CALL put2(unit,'entry_avtb_2d',entry_avtb_2d,ntsi,ntsj)
         CALL put1(unit,'entry_avtb',entry_avtb,1)
         CALL put1(unit,'entry_avmb',entry_avmb,1)
         CALL put3(unit,'after_bc_en',r336_after_bc,ntsi,ntsj,1)
         CALL put3(unit,'post_langmuir_en',r336_after_langmuir,ntsi,ntsj,1)
         CALL put3(unit,'matrix_rhs_en',r336_after_rhs,ntsi,ntsj,1)
         CALL put3(unit,'matrix_diag',r336_matrix_diag,ntsi,ntsj,1)
         CALL put3(unit,'matrix_upper',r336_matrix_upper,ntsi,ntsj,1)
         CALL put3(unit,'matrix_lower',r336_matrix_lower,ntsi,ntsj,1)
         CALL put3(unit,'solve_diag',r336_solve_diag,ntsi,ntsj,1)
         CALL put3(unit,'solve_rhs',r336_solve_rhs,ntsi,ntsj,1)
         CALL put3(unit,'solve_bottom',r336_solve_bottom,ntsi,ntsj,1)
         CALL put3(unit,'solve_back',r336_solve_back,ntsi,ntsj,1)
         CALL put3(unit,'after_solve_en',r336_after_solve,ntsi,ntsj,1)
         CALL put3(unit,'after_tke_en',after_tke_en,ntsi,ntsj,1)
         CALL put3(unit,'tke_pdlr',after_tke_pdlr,ntsi,ntsj,1)
         CALL put3(unit,'pre_sweep_mxl',r336_pre_sweep_mxl,ntsi,ntsj,1)
         CALL put3(unit,'after_lup_mxld',r336_after_lup,ntsi,ntsj,1)
         CALL put3(unit,'after_ldown_mxl',r336_after_ldown,ntsi,ntsj,1)
         CALL put3(unit,'final_mxl',r336_final_mxl,ntsi,ntsj,1)
         CALL put3(unit,'final_mxld',r336_final_mxld,ntsi,ntsj,1)
         CALL put3(unit,'pre_pdlr_avt',r336_pre_pdlr_avt,ntsi,ntsj,1)
         CALL put3(unit,'final_avm',pavm,1,1,1)
         CALL put3(unit,'final_avt',pavt,ntsi,ntsj,1)
         CALL put3(unit,'tke_wmask',entry_wmask,1,1,1)
         CLOSE(unit)
      ENDIF
      DEALLOCATE(r336_after_bc,r336_after_langmuir,r336_after_rhs, &
         & r336_matrix_diag,r336_matrix_upper,r336_matrix_lower, &
         & r336_solve_diag,r336_solve_rhs,r336_solve_bottom,r336_solve_back, &
         & r336_after_solve,r336_pre_sweep_mxl,r336_after_lup,r336_after_ldown, &
         & r336_final_mxl,r336_final_mxld,r336_pre_pdlr_avt, &
         & entry_en,entry_avt,entry_avm_k,entry_sh2,entry_dissl, &
         & r336_entry_gdepw_kmm,r336_entry_e3t_kmm,r336_entry_e3w_kmm, &
         & entry_rn2,entry_rn2b,entry_wmask,entry_tmask,entry_umask,entry_vmask, &
         & entry_drag_bottom,entry_drag_top,entry_ssmask,entry_mbkt,entry_mikt, &
         & entry_nmln,entry_hmlp,entry_fr_i,entry_hm_i,entry_taum, &
         & entry_htau,entry_uu_kbb,entry_vv_kbb,entry_avtb_2d,entry_avtb,entry_avmb, &
         & after_tke_en,after_tke_pdlr)
      lactive=.FALSE.
   END SUBROUTINE r336_tke_finish

END MODULE tke_round336_writer
