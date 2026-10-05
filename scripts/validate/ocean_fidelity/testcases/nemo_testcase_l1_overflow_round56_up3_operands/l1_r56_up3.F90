MODULE l1_r56_up3
   !! Round-56 WRITE-only kt=3 stage-2 UP3 source-order recorder.
   !! Recorded values are never read back into NEMO state.
   USE dom_oce
   USE in_out_manager, ONLY : lwp, nit000
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r56_up3_begin, r56_up3_curv, r56_up3_select_t, &
      & r56_up3_select_f, r56_up3_hflux, r56_up3_hrhs_before, &
      & r56_up3_hrhs_after, r56_up3_vrhs_before, r56_up3_vrhs_after, &
      & r56_up3_bottom_before, r56_up3_finish

   CHARACTER(LEN=16), PARAMETER :: r56_magic = 'NEMO_L1_R56UP301'
   INTEGER, PARAMETER :: r56_field_count = 47
   INTEGER, SAVE :: r56_kt=-1, r56_kbb=-1, r56_kmm=-1, r56_krhs=-1
   INTEGER, SAVE :: r56_curv_count=0, r56_t_count=0, r56_f_count=0
   INTEGER, SAVE :: r56_hflux_count=0, r56_hbefore_count=0, r56_hafter_count=0
   INTEGER, SAVE :: r56_vbefore_count=0, r56_vafter_count=0
   INTEGER, SAVE :: r56_bottom_before_count=0
   LOGICAL, SAVE :: r56_active=.FALSE.

   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: transport_u, transport_v, transport_w
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: kbb_u, kbb_v, kmm_u, kmm_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: rhs_entry_u, rhs_entry_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: curv_uu, curv_vv, curv_uv, curv_vu
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: pair_u_t, pair_v_t, selected_u_t, selected_v_t
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: pair_u_f, pair_v_f, selected_u_f, selected_v_f
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: flux_u_t, flux_v_t, flux_u_f, flux_v_f
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: hscale_u, hscale_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: h_rhs_before_u, h_rhs_before_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: h_rhs_after_u, h_rhs_after_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: v_curv_u, v_curv_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: v_transport_u, v_transport_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: v_selected_u, v_selected_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: v_flux_prev_u, v_flux_prev_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: v_flux_new_u, v_flux_new_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: v_scale_u, v_scale_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: v_rhs_before_u, v_rhs_before_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: v_rhs_after_u, v_rhs_after_v

CONTAINS

   SUBROUTINE allocate_fields
      ALLOCATE(transport_u(jpi,jpj,jpkm1),transport_v(jpi,jpj,jpkm1),transport_w(jpi,jpj,jpkm1), &
         & kbb_u(jpi,jpj,jpkm1),kbb_v(jpi,jpj,jpkm1),kmm_u(jpi,jpj,jpkm1),kmm_v(jpi,jpj,jpkm1), &
         & rhs_entry_u(jpi,jpj,jpkm1),rhs_entry_v(jpi,jpj,jpkm1), &
         & curv_uu(jpi,jpj,jpkm1),curv_vv(jpi,jpj,jpkm1),curv_uv(jpi,jpj,jpkm1),curv_vu(jpi,jpj,jpkm1), &
         & pair_u_t(jpi,jpj,jpkm1),pair_v_t(jpi,jpj,jpkm1),selected_u_t(jpi,jpj,jpkm1),selected_v_t(jpi,jpj,jpkm1), &
         & pair_u_f(jpi,jpj,jpkm1),pair_v_f(jpi,jpj,jpkm1),selected_u_f(jpi,jpj,jpkm1),selected_v_f(jpi,jpj,jpkm1), &
         & flux_u_t(jpi,jpj,jpkm1),flux_v_t(jpi,jpj,jpkm1),flux_u_f(jpi,jpj,jpkm1),flux_v_f(jpi,jpj,jpkm1), &
         & hscale_u(jpi,jpj,jpkm1),hscale_v(jpi,jpj,jpkm1), &
         & h_rhs_before_u(jpi,jpj,jpkm1),h_rhs_before_v(jpi,jpj,jpkm1), &
         & h_rhs_after_u(jpi,jpj,jpkm1),h_rhs_after_v(jpi,jpj,jpkm1), &
         & v_curv_u(jpi,jpj,jpkm1),v_curv_v(jpi,jpj,jpkm1),v_transport_u(jpi,jpj,jpkm1),v_transport_v(jpi,jpj,jpkm1), &
         & v_selected_u(jpi,jpj,jpkm1),v_selected_v(jpi,jpj,jpkm1),v_flux_prev_u(jpi,jpj,jpkm1),v_flux_prev_v(jpi,jpj,jpkm1), &
         & v_flux_new_u(jpi,jpj,jpkm1),v_flux_new_v(jpi,jpj,jpkm1),v_scale_u(jpi,jpj,jpkm1),v_scale_v(jpi,jpj,jpkm1), &
         & v_rhs_before_u(jpi,jpj,jpkm1),v_rhs_before_v(jpi,jpj,jpkm1),v_rhs_after_u(jpi,jpj,jpkm1),v_rhs_after_v(jpi,jpj,jpkm1))
      transport_u=0._wp ; transport_v=0._wp ; transport_w=0._wp
      kbb_u=0._wp ; kbb_v=0._wp ; kmm_u=0._wp ; kmm_v=0._wp
      rhs_entry_u=0._wp ; rhs_entry_v=0._wp
      curv_uu=0._wp ; curv_vv=0._wp ; curv_uv=0._wp ; curv_vu=0._wp
      pair_u_t=0._wp ; pair_v_t=0._wp ; selected_u_t=0._wp ; selected_v_t=0._wp
      pair_u_f=0._wp ; pair_v_f=0._wp ; selected_u_f=0._wp ; selected_v_f=0._wp
      flux_u_t=0._wp ; flux_v_t=0._wp ; flux_u_f=0._wp ; flux_v_f=0._wp
      hscale_u=0._wp ; hscale_v=0._wp
      h_rhs_before_u=0._wp ; h_rhs_before_v=0._wp ; h_rhs_after_u=0._wp ; h_rhs_after_v=0._wp
      v_curv_u=0._wp ; v_curv_v=0._wp ; v_transport_u=0._wp ; v_transport_v=0._wp
      v_selected_u=0._wp ; v_selected_v=0._wp ; v_flux_prev_u=0._wp ; v_flux_prev_v=0._wp
      v_flux_new_u=0._wp ; v_flux_new_v=0._wp ; v_scale_u=0._wp ; v_scale_v=0._wp
      v_rhs_before_u=0._wp ; v_rhs_before_v=0._wp ; v_rhs_after_u=0._wp ; v_rhs_after_v=0._wp
   END SUBROUTINE allocate_fields

   SUBROUTINE r56_up3_begin(kt,Kbb,Kmm,Krhs,pu_b,pv_b,pu_m,pv_m,pu_rhs,pv_rhs,pFu,pFv,pFw)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm,Krhs
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: pu_b,pv_b,pu_m,pv_m,pu_rhs,pv_rhs,pFu,pFv,pFw
      r56_active=lwp .AND. kt==nit000+2 .AND. Kbb==3 .AND. Kmm==3 .AND. Krhs==2
      IF(.NOT.r56_active) RETURN
      IF(ALLOCATED(transport_u)) CALL ctl_stop('round56: nested UP3 record')
      IF(STORAGE_SIZE(1._wp)/=64) CALL ctl_stop('round56: writer requires fp64')
      r56_kt=kt ; r56_kbb=Kbb ; r56_kmm=Kmm ; r56_krhs=Krhs
      CALL allocate_fields
      transport_u=pFu(:,:,1:jpkm1) ; transport_v=pFv(:,:,1:jpkm1) ; transport_w=pFw(:,:,1:jpkm1)
      kbb_u=pu_b(:,:,1:jpkm1) ; kbb_v=pv_b(:,:,1:jpkm1)
      kmm_u=pu_m(:,:,1:jpkm1) ; kmm_v=pv_m(:,:,1:jpkm1)
      rhs_entry_u=pu_rhs(:,:,1:jpkm1) ; rhs_entry_v=pv_rhs(:,:,1:jpkm1)
      r56_curv_count=0 ; r56_t_count=0 ; r56_f_count=0 ; r56_hflux_count=0
      r56_hbefore_count=0 ; r56_hafter_count=0 ; r56_vbefore_count=0 ; r56_vafter_count=0
      r56_bottom_before_count=0
   END SUBROUTINE r56_up3_begin

   SUBROUTINE r56_up3_curv(ji,jj,jk,a,b,c,d)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: a,b,c,d
      IF(.NOT.r56_active) RETURN
      IF(ji<1.OR.ji>jpi.OR.jj<1.OR.jj>jpj) RETURN
      curv_uu(ji,jj,jk)=a ; curv_vv(ji,jj,jk)=b ; curv_uv(ji,jj,jk)=c ; curv_vu(ji,jj,jk)=d
      r56_curv_count=r56_curv_count+1
   END SUBROUTINE r56_up3_curv

   SUBROUTINE r56_up3_select_t(ji,jj,jk,upair,vpair,usel,vsel)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: upair,vpair,usel,vsel
      IF(.NOT.r56_active) RETURN
      IF(ji<1.OR.ji>jpi.OR.jj<1.OR.jj>jpj) RETURN
      pair_u_t(ji,jj,jk)=upair ; pair_v_t(ji,jj,jk)=vpair
      selected_u_t(ji,jj,jk)=usel ; selected_v_t(ji,jj,jk)=vsel
      r56_t_count=r56_t_count+1
   END SUBROUTINE r56_up3_select_t

   SUBROUTINE r56_up3_select_f(ji,jj,jk,upair,vpair,usel,vsel)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: upair,vpair,usel,vsel
      IF(.NOT.r56_active) RETURN
      IF(ji<1.OR.ji>jpi.OR.jj<1.OR.jj>jpj) RETURN
      pair_u_f(ji,jj,jk)=upair ; pair_v_f(ji,jj,jk)=vpair
      selected_u_f(ji,jj,jk)=usel ; selected_v_f(ji,jj,jk)=vsel
      r56_f_count=r56_f_count+1
   END SUBROUTINE r56_up3_select_f

   SUBROUTINE r56_up3_hflux(ji,jj,jk,fut,fvt,fuf,fvf)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: fut,fvt,fuf,fvf
      IF(.NOT.r56_active) RETURN
      IF(ji<1.OR.ji>jpi.OR.jj<1.OR.jj>jpj) RETURN
      flux_u_t(ji,jj,jk)=fut ; flux_v_t(ji,jj,jk)=fvt
      flux_u_f(ji,jj,jk)=fuf ; flux_v_f(ji,jj,jk)=fvf
      r56_hflux_count=r56_hflux_count+1
   END SUBROUTINE r56_up3_hflux

   SUBROUTINE r56_up3_hrhs_before(ji,jj,jk,su,sv,ru,rv)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: su,sv,ru,rv
      IF(.NOT.r56_active) RETURN
      hscale_u(ji,jj,jk)=su ; hscale_v(ji,jj,jk)=sv
      h_rhs_before_u(ji,jj,jk)=ru ; h_rhs_before_v(ji,jj,jk)=rv
      r56_hbefore_count=r56_hbefore_count+1
   END SUBROUTINE r56_up3_hrhs_before

   SUBROUTINE r56_up3_hrhs_after(ji,jj,jk,ru,rv)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: ru,rv
      IF(.NOT.r56_active) RETURN
      h_rhs_after_u(ji,jj,jk)=ru ; h_rhs_after_v(ji,jj,jk)=rv
      r56_hafter_count=r56_hafter_count+1
   END SUBROUTINE r56_up3_hrhs_after

   SUBROUTINE r56_up3_vrhs_before(ji,jj,jk,cu,cv,tu,tv,su,sv,pu,pv,nu,nv,xu,xv,ru,rv)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: cu,cv,tu,tv,su,sv,pu,pv,nu,nv,xu,xv,ru,rv
      IF(.NOT.r56_active) RETURN
      v_curv_u(ji,jj,jk)=cu ; v_curv_v(ji,jj,jk)=cv
      v_transport_u(ji,jj,jk)=tu ; v_transport_v(ji,jj,jk)=tv
      v_selected_u(ji,jj,jk)=su ; v_selected_v(ji,jj,jk)=sv
      v_flux_prev_u(ji,jj,jk)=pu ; v_flux_prev_v(ji,jj,jk)=pv
      v_flux_new_u(ji,jj,jk)=nu ; v_flux_new_v(ji,jj,jk)=nv
      v_scale_u(ji,jj,jk)=xu ; v_scale_v(ji,jj,jk)=xv
      v_rhs_before_u(ji,jj,jk)=ru ; v_rhs_before_v(ji,jj,jk)=rv
      r56_vbefore_count=r56_vbefore_count+1
   END SUBROUTINE r56_up3_vrhs_before

   SUBROUTINE r56_up3_vrhs_after(ji,jj,jk,ru,rv)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: ru,rv
      IF(.NOT.r56_active) RETURN
      v_rhs_after_u(ji,jj,jk)=ru ; v_rhs_after_v(ji,jj,jk)=rv
      r56_vafter_count=r56_vafter_count+1
   END SUBROUTINE r56_up3_vrhs_after

   SUBROUTINE r56_up3_bottom_before(ji,jj,jk,pu,pv,xu,xv,ru,rv)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: pu,pv,xu,xv,ru,rv
      IF(.NOT.r56_active) RETURN
      v_flux_prev_u(ji,jj,jk)=pu ; v_flux_prev_v(ji,jj,jk)=pv
      v_scale_u(ji,jj,jk)=xu ; v_scale_v(ji,jj,jk)=xv
      v_rhs_before_u(ji,jj,jk)=ru ; v_rhs_before_v(ji,jj,jk)=rv
      r56_bottom_before_count=r56_bottom_before_count+1
   END SUBROUTINE r56_up3_bottom_before

   SUBROUTINE put3(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field=name
      WRITE(unit) field
      WRITE(unit) 3,SIZE(value,1),SIZE(value,2),SIZE(value,3)
      WRITE(unit) value
   END SUBROUTINE put3

   SUBROUTINE r56_up3_finish
      INTEGER :: unit,ios
      IF(.NOT.r56_active) RETURN
      IF(r56_curv_count<=0.OR.r56_t_count<=0.OR.r56_f_count<=0.OR.r56_hflux_count<=0) &
         & CALL ctl_stop('round56: incomplete UP3 face record')
      IF(r56_hbefore_count<=0.OR.r56_hbefore_count/=r56_hafter_count) &
         & CALL ctl_stop('round56: incomplete horizontal RHS record')
      IF(r56_vbefore_count<=0.OR.r56_vbefore_count/=r56_vafter_count-r56_bottom_before_count) &
         & CALL ctl_stop('round56: incomplete vertical RHS record')
      IF(r56_bottom_before_count<=0) CALL ctl_stop('round56: missing bottom RHS record')
      OPEN(NEWUNIT=unit,FILE='oracle_r56_up3_kt00000003_s2.bin',ACCESS='STREAM', &
         & FORM='UNFORMATTED',STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios/=0) CALL ctl_stop('round56: cannot open UP3 record')
      WRITE(unit) r56_magic
      WRITE(unit) 1,r56_kt,2,r56_kbb,r56_kmm,r56_krhs,jpi,jpj,jpkm1,ntsi,ntsj,STORAGE_SIZE(1._wp),r56_field_count
      CALL put3(unit,'transport_u',transport_u) ; CALL put3(unit,'transport_v',transport_v)
      CALL put3(unit,'transport_w',transport_w) ; CALL put3(unit,'kbb_u',kbb_u)
      CALL put3(unit,'kbb_v',kbb_v) ; CALL put3(unit,'kmm_u',kmm_u) ; CALL put3(unit,'kmm_v',kmm_v)
      CALL put3(unit,'rhs_entry_u',rhs_entry_u) ; CALL put3(unit,'rhs_entry_v',rhs_entry_v)
      CALL put3(unit,'curv_uu',curv_uu) ; CALL put3(unit,'curv_vv',curv_vv)
      CALL put3(unit,'curv_uv',curv_uv) ; CALL put3(unit,'curv_vu',curv_vu)
      CALL put3(unit,'pair_u_t',pair_u_t) ; CALL put3(unit,'pair_v_t',pair_v_t)
      CALL put3(unit,'selected_u_t',selected_u_t) ; CALL put3(unit,'selected_v_t',selected_v_t)
      CALL put3(unit,'pair_u_f',pair_u_f) ; CALL put3(unit,'pair_v_f',pair_v_f)
      CALL put3(unit,'selected_u_f',selected_u_f) ; CALL put3(unit,'selected_v_f',selected_v_f)
      CALL put3(unit,'flux_u_t',flux_u_t) ; CALL put3(unit,'flux_v_t',flux_v_t)
      CALL put3(unit,'flux_u_f',flux_u_f) ; CALL put3(unit,'flux_v_f',flux_v_f)
      CALL put3(unit,'hscale_u',hscale_u) ; CALL put3(unit,'hscale_v',hscale_v)
      CALL put3(unit,'h_rhs_before_u',h_rhs_before_u) ; CALL put3(unit,'h_rhs_before_v',h_rhs_before_v)
      CALL put3(unit,'h_rhs_after_u',h_rhs_after_u) ; CALL put3(unit,'h_rhs_after_v',h_rhs_after_v)
      CALL put3(unit,'v_curv_u',v_curv_u) ; CALL put3(unit,'v_curv_v',v_curv_v)
      CALL put3(unit,'v_transport_u',v_transport_u) ; CALL put3(unit,'v_transport_v',v_transport_v)
      CALL put3(unit,'v_selected_u',v_selected_u) ; CALL put3(unit,'v_selected_v',v_selected_v)
      CALL put3(unit,'v_flux_prev_u',v_flux_prev_u) ; CALL put3(unit,'v_flux_prev_v',v_flux_prev_v)
      CALL put3(unit,'v_flux_new_u',v_flux_new_u) ; CALL put3(unit,'v_flux_new_v',v_flux_new_v)
      CALL put3(unit,'v_scale_u',v_scale_u) ; CALL put3(unit,'v_scale_v',v_scale_v)
      CALL put3(unit,'v_rhs_before_u',v_rhs_before_u) ; CALL put3(unit,'v_rhs_before_v',v_rhs_before_v)
      CALL put3(unit,'v_rhs_after_u',v_rhs_after_u) ; CALL put3(unit,'v_rhs_after_v',v_rhs_after_v)
      CLOSE(unit)
      DEALLOCATE(transport_u,transport_v,transport_w,kbb_u,kbb_v,kmm_u,kmm_v,rhs_entry_u,rhs_entry_v, &
         & curv_uu,curv_vv,curv_uv,curv_vu,pair_u_t,pair_v_t,selected_u_t,selected_v_t,pair_u_f,pair_v_f,selected_u_f,selected_v_f, &
         & flux_u_t,flux_v_t,flux_u_f,flux_v_f,hscale_u,hscale_v,h_rhs_before_u,h_rhs_before_v,h_rhs_after_u,h_rhs_after_v, &
         & v_curv_u,v_curv_v,v_transport_u,v_transport_v,v_selected_u,v_selected_v,v_flux_prev_u,v_flux_prev_v, &
         & v_flux_new_u,v_flux_new_v,v_scale_u,v_scale_v,v_rhs_before_u,v_rhs_before_v,v_rhs_after_u,v_rhs_after_v)
      r56_active=.FALSE.
   END SUBROUTINE r56_up3_finish

END MODULE l1_r56_up3
