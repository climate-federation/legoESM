MODULE o17_master_writer
   !! O17 extension of the O8--O16 master observer.  Every call is outside
   !! NEMO's computational loops and writes only copies of live module arrays.
   USE, INTRINSIC :: iso_fortran_env, ONLY: int32, real64
   USE par_oce, ONLY: wp, jpi, jpj
   USE par_ice, ONLY: jpl, nlay_i, nlay_s
   USE ice
   USE dom_oce, ONLY: tmask, umask, vmask, e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f, &
      & e1e2t, e1e2u, e1e2v, e1e2f, r1_e1u, r1_e2v, r1_e1e2t, ff_t
   USE oce, ONLY: uu, vv
   USE sbc_oce, ONLY: sss_m, sst_m, ssh_m, ssu_m, ssv_m, frq_m, e3t_m, &
      & qsr, qns, emp, sfx, sprecip, tprecip, cloud_fra, theta_air_zt, q_air_zt, rhoa, &
      & utau, vtau, taum
   USE sbc_ice, ONLY: snwice_mass, snwice_mass_b, snwice_fmass, qsr_oce, qns_oce, &
      & qemp_oce, qemp_ice, emp_oce, emp_ice, qsr_ice, qns_ice, dqns_ice, qtr_ice_top, &
      & qevap_ice, evap_ice, qprec_ice, utau_ice, vtau_ice, rCdU_ice
   USE lib_mpp, ONLY: nbdelay, c_delaylist, todelay
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: o17_set_context, o17_state, o17_prather, o17_raw_blk1, o17_raw_blk2, o17_dyn_masks
   INTEGER :: current_kt=-1, current_rank=-1, current_kbb=-1
   REAL(wp),ALLOCATABLE,DIMENSION(:,:)::cached_fast_t,cached_fast_u,cached_fast_v
   REAL(wp),ALLOCATABLE,DIMENSION(:,:)::cached_icb_t,cached_icb_u,cached_icb_v
   INTERFACE put
      MODULE PROCEDURE put0, put1, put2, put3, put4
   END INTERFACE put

CONTAINS

   SUBROUTINE put_header(u,name,rank,dims)
      INTEGER,INTENT(in)::u,rank
      CHARACTER(LEN=*),INTENT(in)::name
      INTEGER(int32),DIMENSION(4),INTENT(in)::dims
      CHARACTER(LEN=32)::field
      field=name
      WRITE(u) field,INT(rank,int32),dims
   END SUBROUTINE put_header

   SUBROUTINE put0(u,name,value)
      INTEGER,INTENT(in)::u
      CHARACTER(LEN=*),INTENT(in)::name
      REAL(wp),INTENT(in)::value
      CALL put_header(u,name,0,[0_int32,0_int32,0_int32,0_int32])
      WRITE(u) REAL(value,real64)
   END SUBROUTINE put0

   SUBROUTINE put1(u,name,value)
      INTEGER,INTENT(in)::u
      CHARACTER(LEN=*),INTENT(in)::name
      REAL(wp),DIMENSION(:),INTENT(in)::value
      CALL put_header(u,name,1,[INT(SIZE(value,1),int32),0_int32,0_int32,0_int32])
      WRITE(u) REAL(value,real64)
   END SUBROUTINE put1

   SUBROUTINE put2(u,name,value)
      INTEGER,INTENT(in)::u
      CHARACTER(LEN=*),INTENT(in)::name
      REAL(wp),DIMENSION(:,:),INTENT(in)::value
      CALL put_header(u,name,2,[INT(SIZE(value,1),int32),INT(SIZE(value,2),int32),0_int32,0_int32])
      WRITE(u) REAL(value,real64)
   END SUBROUTINE put2

   SUBROUTINE put3(u,name,value)
      INTEGER,INTENT(in)::u
      CHARACTER(LEN=*),INTENT(in)::name
      REAL(wp),DIMENSION(:,:,:),INTENT(in)::value
      CALL put_header(u,name,3,[INT(SIZE(value,1),int32),INT(SIZE(value,2),int32), &
         & INT(SIZE(value,3),int32),0_int32])
      WRITE(u) REAL(value,real64)
   END SUBROUTINE put3

   SUBROUTINE put4(u,name,value)
      INTEGER,INTENT(in)::u
      CHARACTER(LEN=*),INTENT(in)::name
      REAL(wp),DIMENSION(:,:,:,:),INTENT(in)::value
      CALL put_header(u,name,4,[INT(SIZE(value,1),int32),INT(SIZE(value,2),int32), &
         & INT(SIZE(value,3),int32),INT(SIZE(value,4),int32)])
      WRITE(u) REAL(value,real64)
   END SUBROUTINE put4

   SUBROUTINE finish_record(u)
      INTEGER,INTENT(in)::u
      CALL put_header(u,'__END__',-1,[0_int32,0_int32,0_int32,0_int32])
   END SUBROUTINE finish_record

   SUBROUTINE o17_set_context(kt,rank,kbb)
      INTEGER,INTENT(in)::kt,rank,kbb
      current_kt=kt ; current_rank=rank ; current_kbb=kbb
   END SUBROUTINE o17_set_context

   SUBROUTINE o17_dyn_masks(fast_t,fast_u,fast_v,icb_t,icb_u,icb_v)
      REAL(wp),DIMENSION(:,:),INTENT(in)::fast_t,fast_u,fast_v,icb_t,icb_u,icb_v
      IF(.NOT.ALLOCATED(cached_fast_t)) THEN
         ALLOCATE(cached_fast_t(SIZE(fast_t,1),SIZE(fast_t,2)),cached_fast_u(SIZE(fast_u,1),SIZE(fast_u,2)))
         ALLOCATE(cached_fast_v(SIZE(fast_v,1),SIZE(fast_v,2)),cached_icb_t(SIZE(icb_t,1),SIZE(icb_t,2)))
         ALLOCATE(cached_icb_u(SIZE(icb_u,1),SIZE(icb_u,2)),cached_icb_v(SIZE(icb_v,1),SIZE(icb_v,2)))
      ENDIF
      cached_fast_t=fast_t; cached_fast_u=fast_u; cached_fast_v=fast_v
      cached_icb_t=icb_t; cached_icb_u=icb_u; cached_icb_v=icb_v
   END SUBROUTINE o17_dyn_masks

   SUBROUTINE open_o17(kind,label,u)
      INTEGER,INTENT(in)::kind
      CHARACTER(LEN=*),INTENT(in)::label
      INTEGER,INTENT(out)::u
      CHARACTER(LEN=256)::path
      CHARACTER(LEN=8),PARAMETER::magic='O17S0001'
      INTEGER(int32)::meta(8)
      WRITE(path,'("o17_master_records/o17_kind",I2.2,"_",A,"_step",I4.4,"_rank",I4.4,".bin")') &
         kind,TRIM(label),current_kt,current_rank
      OPEN(NEWUNIT=u,FILE=TRIM(path),ACCESS='stream',FORM='unformatted',STATUS='replace',ACTION='write')
      meta=[kind,current_kt,current_rank,jpi,jpj,jpl,nlay_i,nlay_s]
      WRITE(u) magic ; WRITE(u) INT(1,int32) ; WRITE(u) meta
   END SUBROUTINE open_o17

   SUBROUTINE o17_state(kind)
      INTEGER,INTENT(in)::kind
      INTEGER::u,jd
      REAL(real64)::cflice_delayed
      CHARACTER(LEN=32)::label
      SELECT CASE(kind)
      CASE(22); label='step_entry'
      CASE(23); label='after_rheology'
      CASE(24); label='after_advection'
      CASE(25); label='after_rdgrft'
      CASE(26); label='after_dyn_cor'
      CASE(27); label='after_thermo'
      CASE(28); label='step_exit'
      CASE(31); label='after_flx_other'
      CASE DEFAULT; ERROR STOP 'O17 unknown state kind'
      END SELECT
      cflice_delayed=0._real64
      DO jd=1,nbdelay
         IF(TRIM(c_delaylist(jd)) == 'cflice') THEN
            IF(ALLOCATED(todelay(jd)%zdpbuffout)) THEN
               IF(SIZE(todelay(jd)%zdpbuffout)>0) cflice_delayed=todelay(jd)%zdpbuffout(1)
            ENDIF
         ENDIF
      ENDDO
      CALL open_o17(kind,label,u)
      ! Full-halo prognostic state (icestp.F90:151,388-426).
      CALL put(u,'v_i',v_i); CALL put(u,'v_s',v_s); CALL put(u,'sv_i',sv_i); CALL put(u,'a_i',a_i)
      CALL put(u,'t_su',t_su); CALL put(u,'u_ice',u_ice); CALL put(u,'v_ice',v_ice); CALL put(u,'oa_i',oa_i)
      CALL put(u,'a_ip',a_ip); CALL put(u,'v_ip',v_ip); CALL put(u,'v_il',v_il)
      CALL put(u,'e_s',e_s); CALL put(u,'e_i',e_i); CALL put(u,'szv_i',szv_i)
      CALL put(u,'snwice_mass',snwice_mass); CALL put(u,'snwice_mass_b',snwice_mass_b)
      CALL put(u,'stress1_i',stress1_i); CALL put(u,'stress2_i',stress2_i); CALL put(u,'stress12_i',stress12_i)
      CALL put(u,'t_si',t_si); CALL put(u,'a_i_b',a_i_b); CALL put(u,'at_i_b',at_i_b)
      CALL put(u,'u_ice_b',u_ice_b); CALL put(u,'v_ice_b',v_ice_b)
      CALL put(u,'cflice_delayed',cflice_delayed); CALL put(u,'hi_max',hi_max)
      CALL put(u,'hi_mean',hi_mean); CALL put(u,'rn_amax_2d',rn_amax_2d)
      ! Every ice_stp forcing/read operand available at the boundary.
      CALL put(u,'sss_m',sss_m); CALL put(u,'sst_m',sst_m); CALL put(u,'ssh_m',ssh_m)
      CALL put(u,'ssu_m',ssu_m); CALL put(u,'ssv_m',ssv_m); CALL put(u,'frq_m',frq_m)
      CALL put(u,'e3t_m',e3t_m); CALL put(u,'t_bo',t_bo)
      CALL put(u,'tmask',tmask(:,:,1)); CALL put(u,'umask',umask(:,:,1)); CALL put(u,'vmask',vmask(:,:,1))
      CALL put(u,'e1t',e1t); CALL put(u,'e2t',e2t); CALL put(u,'e1u',e1u); CALL put(u,'e2u',e2u)
      CALL put(u,'e1v',e1v); CALL put(u,'e2v',e2v); CALL put(u,'e1f',e1f); CALL put(u,'e2f',e2f)
      CALL put(u,'e1e2t',e1e2t); CALL put(u,'e1e2u',e1e2u); CALL put(u,'e1e2v',e1e2v)
      CALL put(u,'e1e2f',e1e2f); CALL put(u,'r1_e1u',r1_e1u); CALL put(u,'r1_e2v',r1_e2v)
      CALL put(u,'r1_e1e2t',r1_e1e2t); CALL put(u,'ff_t',ff_t)
      CALL put(u,'sprecip',sprecip); CALL put(u,'tprecip',tprecip)
      CALL put(u,'cloud_fra',cloud_fra); CALL put(u,'theta_air_zt',theta_air_zt)
      CALL put(u,'q_air_zt',q_air_zt); CALL put(u,'rhoa',rhoa)
      CALL put(u,'utau_ice',utau_ice); CALL put(u,'vtau_ice',vtau_ice); CALL put(u,'rCdU_ice',rCdU_ice)
      CALL put(u,'u_oce',u_oce); CALL put(u,'v_oce',v_oce); CALL put(u,'drag_io',drag_io)
      CALL put(u,'utau',utau); CALL put(u,'vtau',vtau); CALL put(u,'taum',taum)
      CALL put(u,'uu_kbb',uu(:,:,1,current_kbb)); CALL put(u,'vv_kbb',vv(:,:,1,current_kbb))
      IF(ALLOCATED(cached_fast_t)) THEN
         CALL put(u,'fast_tmask',cached_fast_t); CALL put(u,'fast_umask',cached_fast_u)
         CALL put(u,'fast_vmask',cached_fast_v); CALL put(u,'icb_tmask',cached_icb_t)
         CALL put(u,'icb_umask',cached_icb_u); CALL put(u,'icb_vmask',cached_icb_v)
      ENDIF
      ! O16 atmosphere/ocean exchange fields and carried stateful fluxes.
      CALL put(u,'qsr',qsr); CALL put(u,'qns',qns); CALL put(u,'emp',emp); CALL put(u,'sfx',sfx)
      CALL put(u,'qsr_oce',qsr_oce); CALL put(u,'qns_oce',qns_oce); CALL put(u,'qemp_oce',qemp_oce)
      CALL put(u,'qemp_ice',qemp_ice); CALL put(u,'emp_oce',emp_oce); CALL put(u,'emp_ice',emp_ice)
      CALL put(u,'qsr_ice',qsr_ice); CALL put(u,'qns_ice',qns_ice); CALL put(u,'dqns_ice',dqns_ice)
      CALL put(u,'qtr_ice_top',qtr_ice_top); CALL put(u,'qevap_ice',qevap_ice)
      CALL put(u,'evap_ice',evap_ice); CALL put(u,'qprec_ice',qprec_ice); CALL put(u,'snwice_fmass',snwice_fmass)
      CALL put(u,'wfx_snw',wfx_snw); CALL put(u,'wfx_snw_sni',wfx_snw_sni)
      CALL put(u,'wfx_snw_sum',wfx_snw_sum); CALL put(u,'wfx_pnd',wfx_pnd); CALL put(u,'wfx_spr',wfx_spr)
      CALL put(u,'wfx_sub',wfx_sub); CALL put(u,'wfx_snw_sub',wfx_snw_sub); CALL put(u,'wfx_ice_sub',wfx_ice_sub)
      CALL put(u,'wfx_ice',wfx_ice); CALL put(u,'wfx_sni',wfx_sni); CALL put(u,'wfx_opw',wfx_opw)
      CALL put(u,'wfx_bog',wfx_bog); CALL put(u,'wfx_dyn',wfx_dyn); CALL put(u,'wfx_bom',wfx_bom)
      CALL put(u,'wfx_sum',wfx_sum); CALL put(u,'wfx_lam',wfx_lam); CALL put(u,'wfx_res',wfx_res)
      CALL put(u,'wfx_err_sub',wfx_err_sub); CALL put(u,'sfx_bog',sfx_bog); CALL put(u,'sfx_bom',sfx_bom)
      CALL put(u,'sfx_lam',sfx_lam); CALL put(u,'sfx_sum',sfx_sum); CALL put(u,'sfx_sni',sfx_sni)
      CALL put(u,'sfx_opw',sfx_opw); CALL put(u,'sfx_bri',sfx_bri); CALL put(u,'sfx_dyn',sfx_dyn)
      CALL put(u,'sfx_res',sfx_res); CALL put(u,'sfx_sub',sfx_sub); CALL put(u,'hfx_bog',hfx_bog)
      CALL put(u,'hfx_bom',hfx_bom); CALL put(u,'hfx_sum',hfx_sum); CALL put(u,'hfx_opw',hfx_opw)
      CALL put(u,'hfx_dif',hfx_dif); CALL put(u,'hfx_snw',hfx_snw); CALL put(u,'hfx_err_dif',hfx_err_dif)
      CALL put(u,'hfx_sub',hfx_sub); CALL put(u,'hfx_spr',hfx_spr); CALL put(u,'hfx_thd',hfx_thd)
      CALL put(u,'hfx_dyn',hfx_dyn); CALL put(u,'hfx_res',hfx_res)
      CALL put(u,'qlead',qlead); CALL put(u,'qsb_ice_bot',qsb_ice_bot); CALL put(u,'fhld',fhld)
      CALL finish_record(u)
      CLOSE(u)
   END SUBROUTINE o17_state

   SUBROUTINE o17_raw_blk1(wndi,wndj,tair,qair,slp,tm_su,rhoa_in,smask)
      REAL(wp),DIMENSION(:,:),INTENT(in)::wndi,wndj,tair,qair,slp,tm_su,rhoa_in,smask
      INTEGER::u
      CALL open_o17(32,'raw_blk1',u)
      CALL put(u,'wndi',wndi); CALL put(u,'wndj',wndj); CALL put(u,'tair',tair); CALL put(u,'qair',qair)
      CALL put(u,'slp',slp); CALL put(u,'tm_su',tm_su); CALL put(u,'rhoa',rhoa_in); CALL put(u,'smask',smask)
      CALL finish_record(u)
      CLOSE(u)
   END SUBROUTINE o17_raw_blk1

   SUBROUTINE o17_raw_blk2(tair,qair,slp,qlw,precip,snow,qsr_in,emp_in)
      REAL(wp),DIMENSION(:,:),INTENT(in)::tair,qair,slp,qlw,precip,snow,qsr_in,emp_in
      INTEGER::u
      CALL open_o17(33,'raw_blk2',u)
      CALL put(u,'tair',tair); CALL put(u,'qair',qair); CALL put(u,'slp',slp); CALL put(u,'qlw',qlw)
      CALL put(u,'precip',precip); CALL put(u,'snow',snow); CALL put(u,'qsr',qsr_in); CALL put(u,'emp',emp_in)
      CALL finish_record(u)
      CLOSE(u)
   END SUBROUTINE o17_raw_blk2

   SUBROUTINE o17_prather(kind,sxice,syice,sxxice,syyice,sxyice,sxsn,sysn,sxxsn,syysn,sxysn, &
      & sxa,sya,sxxa,syya,sxya,sxage,syage,sxxage,syyage,sxyage,sxsal,sysal,sxxsal,syysal,sxysal, &
      & sxap,syap,sxxap,syyap,sxyap,sxvp,syvp,sxxvp,syyvp,sxyvp,sxvl,syvl,sxxvl,syyvl,sxyvl, &
      & sxc0,syc0,sxxc0,syyc0,sxyc0,sxe,sye,sxxe,syye,sxye,sxsi,sysi,sxxsi,syysi,sxysi)
      INTEGER,INTENT(in)::kind
      REAL(wp),DIMENSION(:,:,:),INTENT(in)::sxice,syice,sxxice,syyice,sxyice,sxsn,sysn,sxxsn,syysn,sxysn
      REAL(wp),DIMENSION(:,:,:),INTENT(in)::sxa,sya,sxxa,syya,sxya,sxage,syage,sxxage,syyage,sxyage
      REAL(wp),DIMENSION(:,:,:),INTENT(in)::sxsal,sysal,sxxsal,syysal,sxysal,sxap,syap,sxxap,syyap,sxyap
      REAL(wp),DIMENSION(:,:,:),INTENT(in)::sxvp,syvp,sxxvp,syyvp,sxyvp,sxvl,syvl,sxxvl,syyvl,sxyvl
      REAL(wp),DIMENSION(:,:,:,:),INTENT(in)::sxc0,syc0,sxxc0,syyc0,sxyc0,sxe,sye,sxxe,syye,sxye
      REAL(wp),DIMENSION(:,:,:,:),INTENT(in)::sxsi,sysi,sxxsi,syysi,sxysi
      INTEGER::u
      CALL open_o17(kind,MERGE('prather_entry','prather_exit ',kind==29),u)
      CALL put(u,'sxice',sxice); CALL put(u,'syice',syice); CALL put(u,'sxxice',sxxice)
      CALL put(u,'syyice',syyice); CALL put(u,'sxyice',sxyice); CALL put(u,'sxsn',sxsn)
      CALL put(u,'sysn',sysn); CALL put(u,'sxxsn',sxxsn); CALL put(u,'syysn',syysn); CALL put(u,'sxysn',sxysn)
      CALL put(u,'sxa',sxa); CALL put(u,'sya',sya); CALL put(u,'sxxa',sxxa); CALL put(u,'syya',syya)
      CALL put(u,'sxya',sxya); CALL put(u,'sxage',sxage); CALL put(u,'syage',syage)
      CALL put(u,'sxxage',sxxage); CALL put(u,'syyage',syyage); CALL put(u,'sxyage',sxyage)
      CALL put(u,'sxsal',sxsal); CALL put(u,'sysal',sysal); CALL put(u,'sxxsal',sxxsal)
      CALL put(u,'syysal',syysal); CALL put(u,'sxysal',sxysal); CALL put(u,'sxap',sxap)
      CALL put(u,'syap',syap); CALL put(u,'sxxap',sxxap); CALL put(u,'syyap',syyap); CALL put(u,'sxyap',sxyap)
      CALL put(u,'sxvp',sxvp); CALL put(u,'syvp',syvp); CALL put(u,'sxxvp',sxxvp)
      CALL put(u,'syyvp',syyvp); CALL put(u,'sxyvp',sxyvp); CALL put(u,'sxvl',sxvl)
      CALL put(u,'syvl',syvl); CALL put(u,'sxxvl',sxxvl); CALL put(u,'syyvl',syyvl); CALL put(u,'sxyvl',sxyvl)
      CALL put(u,'sxc0',sxc0); CALL put(u,'syc0',syc0); CALL put(u,'sxxc0',sxxc0)
      CALL put(u,'syyc0',syyc0); CALL put(u,'sxyc0',sxyc0); CALL put(u,'sxe',sxe)
      CALL put(u,'sye',sye); CALL put(u,'sxxe',sxxe); CALL put(u,'syye',syye); CALL put(u,'sxye',sxye)
      CALL put(u,'sxsi',sxsi); CALL put(u,'sysi',sysi); CALL put(u,'sxxsi',sxxsi)
      CALL put(u,'syysi',syysi); CALL put(u,'sxysi',sxysi)
      CALL finish_record(u)
      CLOSE(u)
   END SUBROUTINE o17_prather

END MODULE o17_master_writer
