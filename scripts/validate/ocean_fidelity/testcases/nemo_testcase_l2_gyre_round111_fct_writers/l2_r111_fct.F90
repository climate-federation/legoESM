MODULE l2_r111_fct
   !! Round-111 WRITE-only kt=2 stage-3 FCT upstream-writer recorder.
   !! Captured values are never read back into NEMO state.
   USE dom_oce
   USE in_out_manager, ONLY : lwp, nit000
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r111_begin, r111_first_flux, r111_midpoint, &
      & r111_average_flux, r111_upstream

   INTEGER, SAVE :: r111_unit = -1, r111_jn = 0, r111_fields = 0
   LOGICAL, SAVE :: r111_active = .FALSE.

CONTAINS

   SUBROUTINE put0(name,value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field=name
      WRITE(r111_unit) field
      WRITE(r111_unit) 0,1,1,1,0,0,0
      WRITE(r111_unit) value
      r111_fields=r111_fields+1
   END SUBROUTINE put0

   SUBROUTINE put2(name,value,i0,j0)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      INTEGER, INTENT(in) :: i0,j0
      CHARACTER(LEN=16) :: field
      field=name
      WRITE(r111_unit) field
      WRITE(r111_unit) 2,SIZE(value,1),SIZE(value,2),1,i0,j0,1
      WRITE(r111_unit) value
      r111_fields=r111_fields+1
   END SUBROUTINE put2

   SUBROUTINE put3(name,value,i0,j0,k0)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      INTEGER, INTENT(in) :: i0,j0,k0
      CHARACTER(LEN=16) :: field
      field=name
      WRITE(r111_unit) field
      WRITE(r111_unit) 3,SIZE(value,1),SIZE(value,2),SIZE(value,3),i0,j0,k0
      WRITE(r111_unit) value
      r111_fields=r111_fields+1
   END SUBROUTINE put3

   SUBROUTINE tracer_put3(stem,value,i0,j0,k0)
      CHARACTER(LEN=*), INTENT(in) :: stem
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      INTEGER, INTENT(in) :: i0,j0,k0
      CHARACTER(LEN=16) :: field
      SELECT CASE(r111_jn)
      CASE(jp_tem)
         field=TRIM(stem)//'_T'
      CASE(jp_sal)
         field=TRIM(stem)//'_S'
      CASE DEFAULT
         CALL ctl_stop('round111: unexpected active tracer index')
      END SELECT
      CALL put3(field,value,i0,j0,k0)
   END SUBROUTINE tracer_put3

   SUBROUTINE r111_begin(kt,Kbb,Kmm,Kaa,Krhs,jn,p2dt,pt_b,pU,pV,pW,pt_rhs)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm,Kaa,Krhs,jn
      REAL(wp), INTENT(in) :: p2dt
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pt_b,pt_rhs
      REAL(wp), DIMENSION(ntsi-nn_hls:ntei+nn_hls, &
         & ntsj-nn_hls:ntej+nn_hls,jpk), INTENT(in) :: pU,pV,pW
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic

      IF(.NOT.(lwp .AND. kt==nit000+1)) THEN
         r111_active=.FALSE.
         RETURN
      ENDIF
      r111_active=.TRUE.
      r111_jn=jn
      IF(jn==jp_tem) THEN
         IF(r111_unit/=-1) CALL ctl_stop('round111: nested FCT record')
         IF(l_istiled) CALL ctl_stop('round111: FCT writer refuses tiling')
         IF(STORAGE_SIZE(1._wp)/=64) CALL ctl_stop('round111: FCT writer requires fp64')
         OPEN(NEWUNIT=r111_unit,FILE='oracle_fct_writers_kt00000002_s3.bin', &
            & ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE', &
            & ACTION='WRITE',IOSTAT=ios)
         IF(ios/=0) CALL ctl_stop('round111: cannot open FCT writer record')
         magic='NEMO_L2_R111F1'
         WRITE(r111_unit) magic
         WRITE(r111_unit) 1,kt,Kbb,Kmm,Kaa,Krhs,jpi,jpj,jpk, &
            & STORAGE_SIZE(1._wp),34,ntsi,ntsj,nn_hls
         r111_fields=0
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
         CALL put3('tmask',tmask(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpkm1), &
            & ntsi-1,ntsj-1,1)
         CALL put3('wmask',wmask(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpkm1), &
            & ntsi-1,ntsj-1,1)
         CALL put2('r1_e1e2t',r1_e1e2t(ntsi-1:ntei+1,ntsj-1:ntej+1), &
            & ntsi-1,ntsj-1)
      ELSE IF(jn/=jp_sal .OR. r111_unit==-1) THEN
         CALL ctl_stop('round111: FCT tracer order is not T then S')
      ENDIF
      CALL tracer_put3('base',pt_b(1:jpi,1:jpj,1:jpkm1),1,1,1)
      CALL tracer_put3('rhs_entry',pt_rhs(ntsi:ntei,ntsj:ntej,1:jpkm1), &
         & ntsi,ntsj,1)
   END SUBROUTINE r111_begin

   SUBROUTINE r111_first_flux(ptFu,ptFv,ptFw)
      REAL(wp), DIMENSION(ntsi-nn_hls:ntei+nn_hls, &
         & ntsj-nn_hls:ntej+nn_hls,jpk), INTENT(in) :: ptFu,ptFv,ptFw
      IF(.NOT.r111_active) RETURN
      CALL tracer_put3('first_u',ptFu(ntsi-2:ntei+1,ntsj-2:ntej+1,1:jpkm1), &
         & ntsi-2,ntsj-2,1)
      CALL tracer_put3('first_v',ptFv(ntsi-2:ntei+1,ntsj-2:ntej+1,1:jpkm1), &
         & ntsi-2,ntsj-2,1)
      CALL tracer_put3('first_w',ptFw(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpk), &
         & ntsi-1,ntsj-1,1)
   END SUBROUTINE r111_first_flux

   SUBROUTINE r111_midpoint(divergence,pt_up1)
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: divergence
      REAL(wp), DIMENSION(ntsi-nn_hls:ntei+nn_hls, &
         & ntsj-nn_hls:ntej+nn_hls,jpk), INTENT(in) :: pt_up1
      IF(.NOT.r111_active) RETURN
      CALL tracer_put3('first_div', &
         & divergence(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('midpoint', &
         & pt_up1(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
   END SUBROUTINE r111_midpoint

   SUBROUTINE r111_average_flux(ptFu,ptFv,ptFw)
      REAL(wp), DIMENSION(ntsi-nn_hls:ntei+nn_hls, &
         & ntsj-nn_hls:ntej+nn_hls,jpk), INTENT(in) :: ptFu,ptFv,ptFw
      IF(.NOT.r111_active) RETURN
      CALL tracer_put3('average_u',ptFu(ntsi-1:ntei,ntsj-1:ntej,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('average_v',ptFv(ntsi-1:ntei,ntsj-1:ntej,1:jpkm1), &
         & ntsi-1,ntsj-1,1)
      CALL tracer_put3('average_w',ptFw(ntsi-1:ntei+1,ntsj-1:ntej+1,1:jpk), &
         & ntsi-1,ntsj-1,1)
   END SUBROUTINE r111_average_flux

   SUBROUTINE r111_upstream(divergence,pt_rhs)
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: divergence
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pt_rhs
      IF(.NOT.r111_active) RETURN
      CALL tracer_put3('final_div', &
         & divergence(ntsi:ntei,ntsj:ntej,1:jpkm1),ntsi,ntsj,1)
      CALL tracer_put3('rhs_after', &
         & pt_rhs(ntsi:ntei,ntsj:ntej,1:jpkm1),ntsi,ntsj,1)
      IF(r111_jn==jp_sal) THEN
         IF(r111_fields/=34) CALL ctl_stop('round111: wrong FCT field count')
         CLOSE(r111_unit)
         r111_unit=-1
         r111_active=.FALSE.
      ENDIF
   END SUBROUTINE r111_upstream

END MODULE l2_r111_fct
