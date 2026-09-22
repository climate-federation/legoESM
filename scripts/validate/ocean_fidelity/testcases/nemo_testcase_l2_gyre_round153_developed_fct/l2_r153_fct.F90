MODULE l2_r153_fct
   !! Round-153 WRITE-only developed-state stage-3 FCT recorder.
   !! Captured values are never read back into NEMO state.
   USE dom_oce
   USE in_out_manager, ONLY : lwp
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r153_begin, r153_first_flux, r153_midpoint, &
      & r153_average_flux, r153_upstream, r153_limiter_begin, &
      & r153_coef_u, r153_coef_v, r153_coef_w, r153_limiter_end, &
      & r153_final, lr153_active

   INTEGER, SAVE :: r153_unit = -1, r153_jn = 0, r153_fields = 0
   LOGICAL, SAVE :: lr153_active = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r153_cu, r153_cv, r153_cw

CONTAINS

   SUBROUTINE put0(name,value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field=name
      WRITE(r153_unit) field
      WRITE(r153_unit) 0,1,1,1,0,0,0
      WRITE(r153_unit) value
      r153_fields=r153_fields+1
   END SUBROUTINE put0

   SUBROUTINE put2(name,value,i0,j0)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      INTEGER, INTENT(in) :: i0,j0
      CHARACTER(LEN=16) :: field
      field=name
      WRITE(r153_unit) field
      WRITE(r153_unit) 2,SIZE(value,1),SIZE(value,2),1,i0,j0,1
      WRITE(r153_unit) value
      r153_fields=r153_fields+1
   END SUBROUTINE put2

   SUBROUTINE put3(name,value,i0,j0,k0)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      INTEGER, INTENT(in) :: i0,j0,k0
      CHARACTER(LEN=16) :: field
      field=name
      WRITE(r153_unit) field
      WRITE(r153_unit) 3,SIZE(value,1),SIZE(value,2),SIZE(value,3),i0,j0,k0
      WRITE(r153_unit) value
      r153_fields=r153_fields+1
   END SUBROUTINE put3

   SUBROUTINE tracer_put3(stem,value,i0,j0,k0)
      CHARACTER(LEN=*), INTENT(in) :: stem
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      INTEGER, INTENT(in) :: i0,j0,k0
      CHARACTER(LEN=16) :: field
      SELECT CASE(r153_jn)
      CASE(jp_tem)
         field=TRIM(stem)//'_T'
      CASE(jp_sal)
         field=TRIM(stem)//'_S'
      CASE DEFAULT
         CALL ctl_stop('round153: unexpected active tracer index')
      END SELECT
      CALL put3(field,value,i0,j0,k0)
   END SUBROUTINE tracer_put3

   SUBROUTINE r153_begin(kt,Kbb,Kmm,Kaa,Krhs,jn,p2dt,pt_b,pt_n,pU,pV,pW,pt_rhs)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm,Kaa,Krhs,jn
      REAL(wp), INTENT(in) :: p2dt
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pt_b,pt_n,pt_rhs
      REAL(wp), DIMENSION(ntsi-nn_hls:ntei+nn_hls, &
         & ntsj-nn_hls:ntej+nn_hls,jpk), INTENT(in) :: pU,pV,pW
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic

      IF(.NOT.(lwp .AND. kt==1081 .AND. Kbb==1 .AND. Kmm==2 .AND. &
         & Kaa==3 .AND. Krhs==3)) THEN
         lr153_active=.FALSE.
         RETURN
      ENDIF
      lr153_active=.TRUE.
      r153_jn=jn
      IF(jn==jp_tem) THEN
         IF(r153_unit/=-1) CALL ctl_stop('round153: nested FCT record')
         IF(l_istiled) CALL ctl_stop('round153: FCT writer refuses tiling')
         IF(STORAGE_SIZE(1._wp)/=64) CALL ctl_stop('round153: FCT writer requires fp64')
         OPEN(NEWUNIT=r153_unit,FILE='oracle_developed_fct_kt00001081.bin', &
            & ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE', &
            & ACTION='WRITE',IOSTAT=ios)
         IF(ios/=0) CALL ctl_stop('round153: cannot open FCT record')
         magic='NEMO_L2_R153FCT'
         WRITE(r153_unit) magic
         WRITE(r153_unit) 1,kt,Kbb,Kmm,Kaa,Krhs,jpi,jpj,jpk, &
            & STORAGE_SIZE(1._wp),61,ntsi,ntsj,nn_hls
         r153_fields=0
         CALL put0('p2dt',p2dt)
         CALL put3('transport_u',pU(ntsi-2:ntei+1,ntsj-2:ntej+1,1:jpkm1), &
            & ntsi-2,ntsj-2,1)
         CALL put3('transport_v',pV(ntsi-2:ntei+1,ntsj-2:ntej+1,1:jpkm1), &
            & ntsi-2,ntsj-2,1)
         CALL put3('transport_w',pW(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpkm1), &
            & ntsi-1,ntsj-1,1)
         CALL put3('e3t_3d',e3t_3d(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpkm1), &
            & ntsi-1,ntsj-1,1)
         CALL put2('r3t_Kbb',r3t(ntsi-1:ntei+1,ntsj-1:ntej+1,Kbb), &
            & ntsi-1,ntsj-1)
         CALL put2('r3t_Kmm',r3t(ntsi-1:ntei+1,ntsj-1:ntej+1,Kmm), &
            & ntsi-1,ntsj-1)
         CALL put2('r3t_Kaa',r3t(ntsi-1:ntei+1,ntsj-1:ntej+1,Kaa), &
            & ntsi-1,ntsj-1)
         CALL put3('tmask',tmask(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpkm1), &
            & ntsi-1,ntsj-1,1)
         CALL put3('wmask',wmask(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpkm1), &
            & ntsi-1,ntsj-1,1)
         CALL put2('r1_e1e2t',r1_e1e2t(ntsi-1:ntei+1,ntsj-1:ntej+1), &
            & ntsi-1,ntsj-1)
      ELSE IF(jn/=jp_sal .OR. r153_unit==-1) THEN
         CALL ctl_stop('round153: FCT tracer order is not T then S')
      ENDIF
      CALL tracer_put3('base',pt_b(1:jpi,1:jpj,1:jpkm1),1,1,1)
      CALL tracer_put3('now',pt_n(1:jpi,1:jpj,1:jpkm1),1,1,1)
      CALL tracer_put3('rhs_entry',pt_rhs(ntsi:ntei,ntsj:ntej,1:jpkm1), &
         & ntsi,ntsj,1)
   END SUBROUTINE r153_begin

   SUBROUTINE r153_first_flux(ptFu,ptFv,ptFw)
      REAL(wp), DIMENSION(ntsi-nn_hls:ntei+nn_hls, &
         & ntsj-nn_hls:ntej+nn_hls,jpk), INTENT(in) :: ptFu,ptFv,ptFw
      IF(.NOT.lr153_active) RETURN
      CALL tracer_put3('first_u',ptFu(ntsi-2:ntei+1,ntsj-2:ntej+1,1:jpkm1), &
         & ntsi-2,ntsj-2,1)
      CALL tracer_put3('first_v',ptFv(ntsi-2:ntei+1,ntsj-2:ntej+1,1:jpkm1), &
         & ntsi-2,ntsj-2,1)
      CALL tracer_put3('first_w',ptFw(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpk), &
         & ntsi-1,ntsj-1,1)
   END SUBROUTINE r153_first_flux

   SUBROUTINE r153_midpoint(divergence,pt_up1)
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: divergence
      REAL(wp), DIMENSION(ntsi-nn_hls:ntei+nn_hls, &
         & ntsj-nn_hls:ntej+nn_hls,jpk), INTENT(in) :: pt_up1
      IF(.NOT.lr153_active) RETURN
      CALL tracer_put3('first_div',divergence(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('midpoint',pt_up1(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
   END SUBROUTINE r153_midpoint

   SUBROUTINE r153_average_flux(ptFu,ptFv,ptFw)
      REAL(wp), DIMENSION(ntsi-nn_hls:ntei+nn_hls, &
         & ntsj-nn_hls:ntej+nn_hls,jpk), INTENT(in) :: ptFu,ptFv,ptFw
      IF(.NOT.lr153_active) RETURN
      CALL tracer_put3('average_u',ptFu(ntsi-1:ntei,ntsj-1:ntej,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('average_v',ptFv(ntsi-1:ntei,ntsj-1:ntej,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('average_w',ptFw(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpk), &
         & ntsi-1,ntsj-1,1)
   END SUBROUTINE r153_average_flux

   SUBROUTINE r153_upstream(divergence,pt_rhs)
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: divergence
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pt_rhs
      IF(.NOT.lr153_active) RETURN
      CALL tracer_put3('upstream_div',divergence(ntsi:ntei,ntsj:ntej,1:jpkm1), &
         & ntsi,ntsj,1)
      CALL tracer_put3('rhs_after_up',pt_rhs(ntsi:ntei,ntsj:ntej,1:jpkm1), &
         & ntsi,ntsj,1)
   END SUBROUTINE r153_upstream

   SUBROUTINE r153_limiter_begin(paa,pbb,pcc)
      REAL(wp), DIMENSION(ntsi-nn_hls:ntei+nn_hls, &
         & ntsj-nn_hls:ntej+nn_hls,jpk), INTENT(in) :: paa,pbb,pcc
      IF(.NOT.lr153_active) RETURN
      IF(ALLOCATED(r153_cu)) CALL ctl_stop('round153: nested limiter capture')
      ALLOCATE(r153_cu(jpi,jpj,jpk),r153_cv(jpi,jpj,jpk),r153_cw(jpi,jpj,jpk))
      r153_cu=1._wp ; r153_cv=1._wp ; r153_cw=1._wp
      CALL tracer_put3('anti_pre_u',paa(ntsi-1:ntei,ntsj-1:ntej,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('anti_pre_v',pbb(ntsi-1:ntei,ntsj-1:ntej,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('anti_pre_w',pcc(ntsi:ntei,ntsj:ntej,1:jpk), &
         & ntsi,ntsj,1)
   END SUBROUTINE r153_limiter_begin

   SUBROUTINE r153_coef_u(ji,jj,jk,zcoef)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: zcoef
      IF(lr153_active) r153_cu(ji,jj,jk)=zcoef
   END SUBROUTINE r153_coef_u

   SUBROUTINE r153_coef_v(ji,jj,jk,zcoef)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: zcoef
      IF(lr153_active) r153_cv(ji,jj,jk)=zcoef
   END SUBROUTINE r153_coef_v

   SUBROUTINE r153_coef_w(ji,jj,jk,zcoef)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: zcoef
      IF(lr153_active) r153_cw(ji,jj,jk)=zcoef
   END SUBROUTINE r153_coef_w

   SUBROUTINE r153_limiter_end(paa,pbb,pcc)
      REAL(wp), DIMENSION(ntsi-nn_hls:ntei+nn_hls, &
         & ntsj-nn_hls:ntej+nn_hls,jpk), INTENT(in) :: paa,pbb,pcc
      IF(.NOT.lr153_active) RETURN
      CALL tracer_put3('coef_u',r153_cu(ntsi-1:ntei,ntsj-1:ntej,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('coef_v',r153_cv(ntsi-1:ntei,ntsj-1:ntej,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('coef_w',r153_cw(ntsi:ntei,ntsj:ntej,1:jpk), &
         & ntsi,ntsj,1)
      CALL tracer_put3('anti_post_u',paa(ntsi-1:ntei,ntsj-1:ntej,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('anti_post_v',pbb(ntsi-1:ntei,ntsj-1:ntej,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('anti_post_w',pcc(ntsi:ntei,ntsj:ntej,1:jpk), &
         & ntsi,ntsj,1)
      DEALLOCATE(r153_cu,r153_cv,r153_cw)
   END SUBROUTINE r153_limiter_end

   SUBROUTINE r153_final(divergence,divisor,pt_rhs)
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: divergence,divisor
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pt_rhs
      IF(.NOT.lr153_active) RETURN
      CALL tracer_put3('final_div',divergence(ntsi:ntei,ntsj:ntej,1:jpkm1), &
         & ntsi,ntsj,1)
      CALL tracer_put3('divisor',divisor(ntsi:ntei,ntsj:ntej,1:jpkm1), &
         & ntsi,ntsj,1)
      CALL tracer_put3('rhs_final',pt_rhs(ntsi:ntei,ntsj:ntej,1:jpkm1), &
         & ntsi,ntsj,1)
      IF(r153_jn==jp_sal) THEN
         IF(r153_fields/=61) CALL ctl_stop('round153: wrong FCT field count')
         CLOSE(r153_unit)
         r153_unit=-1
         lr153_active=.FALSE.
      ENDIF
   END SUBROUTINE r153_final

END MODULE l2_r153_fct
