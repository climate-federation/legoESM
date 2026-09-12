MODULE l2_r63_krhs
   !! Round-63 WRITE-only kt=2 tracer-Krhs and TKE-RHS recorder.
   !! No captured value is ever read back into NEMO state.
   USE dom_oce
   USE in_out_manager, ONLY : lwp, nit000
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r63_begin, r63_fct_first, r63_snapshot, r63_content_finish, &
      & r63_tke_begin, r63_tke_before_row, r63_tke_after_row, r63_tke_finish

   INTEGER, SAVE :: r63_unit = -1, r63_tke_unit = -1
   LOGICAL, SAVE :: r63_active = .FALSE., r63_tke_active = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: tke_entry, tke_shear
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: tke_strat, tke_diss, tke_post
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: tke_avt, tke_rn2, tke_dissl
   REAL(wp), SAVE :: tke_zfact3 = 0._wp

#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"

CONTAINS

   SUBROUTINE put0(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field=name
      WRITE(unit) field ; WRITE(unit) 0,1,1,1 ; WRITE(unit) value
   END SUBROUTINE put0

   SUBROUTINE put2(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      ! dom_oce/do_loop_substitute.h90: T2D(0)=ntsi:ntei,ntsj:ntej.
      REAL(wp), DIMENSION(T2D(0)), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field=name
      WRITE(unit) field
      WRITE(unit) 2,ntei-ntsi+1,ntej-ntsj+1,1
      WRITE(unit) value
   END SUBROUTINE put2

   SUBROUTINE put3(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      ! Explicit reduced-domain bounds prevent assumed-shape rebasing.
      REAL(wp), DIMENSION(T2D(0),jpkm1), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field=name
      WRITE(unit) field
      WRITE(unit) 3,ntei-ntsi+1,ntej-ntsj+1,jpkm1
      WRITE(unit) value
   END SUBROUTINE put3

   SUBROUTINE r63_begin(kt,kstg,Kbb,Kmm,Krhs,p2dt,pts)
      INTEGER, INTENT(in) :: kt,kstg,Kbb,Kmm,Krhs
      REAL(wp), INTENT(in) :: p2dt
      ! stprk3_stg/MY_SRC declares the active tracer store at this full shape.
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt), INTENT(in) :: pts
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      r63_active=lwp .AND. kt==nit000+1 .AND. kstg==3
      IF(.NOT.r63_active) RETURN
      IF(r63_unit/=-1) CALL ctl_stop('round63: nested tracer record')
      IF(l_istiled) CALL ctl_stop('round63: tracer writer refuses tiling')
      IF(STORAGE_SIZE(1._wp)/=64) CALL ctl_stop('round63: tracer writer requires fp64')
      IF(jpts/=2) CALL ctl_stop('round63: tracer writer requires T/S')
      OPEN(NEWUNIT=r63_unit,FILE='oracle_krhs_split_kt00000002.bin', &
         & ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios/=0) CALL ctl_stop('round63: cannot open tracer record')
      magic='NEMO_L2_R63KRS1'
      WRITE(r63_unit) magic
      WRITE(r63_unit) 1,kt,kstg,Kbb,Kmm,Krhs,ntei-ntsi+1,ntej-ntsj+1, &
         & jpkm1,ntsi,ntsj,STORAGE_SIZE(1._wp),23
      CALL put0(r63_unit,'p2dt',p2dt)
      CALL put3(r63_unit,'krhs_zero_T',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_tem,Krhs))
      CALL put3(r63_unit,'krhs_zero_S',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_sal,Krhs))
   END SUBROUTINE r63_begin

   SUBROUTINE r63_fct_first(jn,rhs)
      INTEGER, INTENT(in) :: jn
      ! traadv_fct.F90:85 declares pt(jpi,jpj,jpk,kjpt,jpt).
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: rhs
      IF(.NOT.r63_active) RETURN
      SELECT CASE(jn)
      CASE(jp_tem)
         CALL put3(r63_unit,'adv_up1_T',rhs(ntsi:ntei,ntsj:ntej,1:jpkm1))
      CASE(jp_sal)
         CALL put3(r63_unit,'adv_up1_S',rhs(ntsi:ntei,ntsj:ntej,1:jpkm1))
      CASE DEFAULT
         CALL ctl_stop('round63: unexpected active tracer index')
      END SELECT
   END SUBROUTINE r63_fct_first

   SUBROUTINE r63_snapshot(label,pts,Krhs)
      CHARACTER(LEN=*), INTENT(in) :: label
      INTEGER, INTENT(in) :: Krhs
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt), INTENT(in) :: pts
      IF(.NOT.r63_active) RETURN
      CALL put3(r63_unit,TRIM(label)//'_T', &
         & pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_tem,Krhs))
      CALL put3(r63_unit,TRIM(label)//'_S', &
         & pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_sal,Krhs))
   END SUBROUTINE r63_snapshot

   SUBROUTINE r63_content_finish(Kbb,Kmm,Krhs,pts,actual_rhs)
      INTEGER, INTENT(in) :: Kbb,Kmm,Krhs
      ! trazdf.F90:72 and its round-35 zl2_rhs declaration at :56.
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt), INTENT(in) :: pts
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in) :: actual_rhs
      INTEGER :: ji,jj,jk
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: z
      IF(.NOT.r63_active) RETURN
      ALLOCATE(z(ntsi:ntei,ntsj:ntej,jpkm1))
      CALL put3(r63_unit,'T_Kbb',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_tem,Kbb))
      CALL put3(r63_unit,'S_Kbb',pts(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_sal,Kbb))
      DO jk=1,jpkm1 ; DO jj=ntsj,ntej ; DO ji=ntsi,ntei
         z(ji,jj,jk)=e3t(ji,jj,jk,Kbb)
      END DO ; END DO ; END DO
      CALL put3(r63_unit,'e3t_Kbb',z)
      DO jk=1,jpkm1 ; DO jj=ntsj,ntej ; DO ji=ntsi,ntei
         z(ji,jj,jk)=e3t(ji,jj,jk,Kmm)
      END DO ; END DO ; END DO
      CALL put3(r63_unit,'e3t_Kmm',z)
      CALL put2(r63_unit,'r3t_Kbb',r3t(ntsi:ntei,ntsj:ntej,Kbb))
      CALL put2(r63_unit,'r3t_Kmm',r3t(ntsi:ntei,ntsj:ntej,Kmm))
      CALL put3(r63_unit,'e3t_3d',e3t_3d(ntsi:ntei,ntsj:ntej,1:jpkm1))
      CALL put3(r63_unit,'tmask',tmask(ntsi:ntei,ntsj:ntej,1:jpkm1))
      CALL put3(r63_unit,'content_T',actual_rhs(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_tem))
      CALL put3(r63_unit,'content_S',actual_rhs(ntsi:ntei,ntsj:ntej,1:jpkm1,jp_sal))
      DEALLOCATE(z)
      CLOSE(r63_unit)
      r63_unit=-1
      r63_active=.FALSE.
   END SUBROUTINE r63_content_finish

   SUBROUTINE r63_tke_begin(kt)
      INTEGER, INTENT(in) :: kt
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      r63_tke_active=lwp .AND. kt==nit000+1
      IF(.NOT.r63_tke_active) RETURN
      IF(r63_tke_unit/=-1) CALL ctl_stop('round63: nested TKE record')
      IF(l_istiled) CALL ctl_stop('round63: TKE writer refuses tiling')
      OPEN(NEWUNIT=r63_tke_unit,FILE='oracle_tke_rhs_split_kt00000002.bin', &
         & ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios/=0) CALL ctl_stop('round63: cannot open TKE record')
      magic='NEMO_L2_R63TKR1'
      WRITE(r63_tke_unit) magic
      WRITE(r63_tke_unit) 1,kt,ntei-ntsi+1,ntej-ntsj+1,jpkm1,ntsi,ntsj, &
         & STORAGE_SIZE(1._wp),11
      ALLOCATE(tke_entry(T2D(0),jpkm1),tke_shear(T2D(0),jpkm1), &
         & tke_strat(T2D(0),jpkm1),tke_diss(T2D(0),jpkm1), &
         & tke_post(T2D(0),jpkm1),tke_avt(T2D(0),jpkm1), &
         & tke_rn2(T2D(0),jpkm1),tke_dissl(T2D(0),jpkm1))
      tke_entry=0._wp ; tke_shear=0._wp ; tke_strat=0._wp ; tke_diss=0._wp
      tke_post=0._wp ; tke_avt=0._wp ; tke_rn2=0._wp ; tke_dissl=0._wp
   END SUBROUTINE r63_tke_begin

   SUBROUTINE r63_tke_before_row(jj,p_sh2,p_avt,p_rn2,p_dissl,p_en,zfact3)
      INTEGER, INTENT(in) :: jj
      ! zdftke.F90:215,217 use A2D(0); oce rn2 is full jpi,jpj,jpk.
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: p_sh2,p_avt,p_dissl,p_en
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: p_rn2
      REAL(wp), INTENT(in) :: zfact3
      INTEGER :: ji,jk
      IF(.NOT.r63_tke_active) RETURN
      tke_zfact3=zfact3
      DO jk=2,jpkm1 ; DO ji=ntsi,ntei
         tke_entry(ji,jj,jk)=p_en(ji,jj,jk)
         tke_shear(ji,jj,jk)=p_sh2(ji,jj,jk)
         tke_avt(ji,jj,jk)=p_avt(ji,jj,jk)
         tke_rn2(ji,jj,jk)=p_rn2(ji,jj,jk)
         tke_dissl(ji,jj,jk)=p_dissl(ji,jj,jk)
         tke_strat(ji,jj,jk)=p_avt(ji,jj,jk)*p_rn2(ji,jj,jk)
         tke_diss(ji,jj,jk)=zfact3*p_dissl(ji,jj,jk)*p_en(ji,jj,jk)
      END DO ; END DO
   END SUBROUTINE r63_tke_before_row

   SUBROUTINE r63_tke_after_row(jj,p_en)
      INTEGER, INTENT(in) :: jj
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: p_en
      IF(.NOT.r63_tke_active) RETURN
      tke_post(:,jj,2:jpkm1)=p_en(ntsi:ntei,jj,2:jpkm1)
   END SUBROUTINE r63_tke_after_row

   SUBROUTINE r63_tke_finish()
      IF(.NOT.r63_tke_active) RETURN
      CALL put0(r63_tke_unit,'rn_Dt',rn_Dt)
      CALL put0(r63_tke_unit,'zfact3',tke_zfact3)
      CALL put3(r63_tke_unit,'en_rhs_entry',tke_entry)
      CALL put3(r63_tke_unit,'shear',tke_shear)
      CALL put3(r63_tke_unit,'avt',tke_avt)
      CALL put3(r63_tke_unit,'rn2',tke_rn2)
      CALL put3(r63_tke_unit,'dissl',tke_dissl)
      CALL put3(r63_tke_unit,'strat_product',tke_strat)
      CALL put3(r63_tke_unit,'diss_product',tke_diss)
      CALL put3(r63_tke_unit,'wmask',wmask(ntsi:ntei,ntsj:ntej,1:jpkm1))
      CALL put3(r63_tke_unit,'en_rhs_post',tke_post)
      CLOSE(r63_tke_unit)
      r63_tke_unit=-1
      r63_tke_active=.FALSE.
      DEALLOCATE(tke_entry,tke_shear,tke_strat,tke_diss,tke_post, &
         & tke_avt,tke_rn2,tke_dissl)
   END SUBROUTINE r63_tke_finish
END MODULE l2_r63_krhs
