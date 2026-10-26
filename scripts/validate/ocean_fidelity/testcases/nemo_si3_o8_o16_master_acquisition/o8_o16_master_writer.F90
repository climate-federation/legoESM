MODULE o8_o16_master_writer
   USE, INTRINSIC :: iso_fortran_env, ONLY: int32, real64
   USE par_oce, ONLY: wp, jpi, jpj, nn_hls, ntsi, ntei, ntsj, ntej
   USE par_ice, ONLY: jpl, nlay_i, nlay_s, rn_kappa_i, rn_kappa_s, rn_snwblow, rn_sinew, &
      &              nn_icesal, r1_nlay_i, r1_nlay_s, rn_simin, rn_himin, nn_snwfra
   USE phycst, ONLY: rcnd_s
   USE dom_oce, ONLY: tmask, smask0, umask, vmask
   USE ice
   USE ice1D
   USE sbc_oce, ONLY: qsr, qns, emp, sfx, fr_i, fwfice, qns_tot, qsr_tot, frq_m, &
      &               sss_m, sst_m, cloud_fra, sprecip, tprecip, utau, vtau, taum
   USE sbc_ice, ONLY: qsr_oce, qns_oce, qemp_oce, qemp_ice, emp_oce, emp_ice, qsr_ice, qns_ice, qevap_ice, &
      &               qcn_ice, qml_ice, qtr_ice_top, tn_ice, alb_ice, &
      &               snwice_mass, snwice_mass_b, snwice_fmass, rCdU_ice, utau_ice, vtau_ice
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: o8o16_set_context, o8o16_frazil, o8o16_category, o8o16_picard
   PUBLIC :: o8o16_global, o8o16_update, o8o16_tau
   PUBLIC :: o8o16_salinity_config, o8o16_pond_config
   PUBLIC :: o8o16_da_config, o8o16_albedo_config

   INTEGER :: current_kt=-1, current_kmm=-1, current_nit000=-1, current_rank=-1, current_jl=-1
   INTEGER :: current_jpni=-1, current_jpnj=-1
   REAL(wp) :: current_rdt=0._wp, current_r1dt=0._wp
   REAL(wp) :: current_beta=0._wp, current_dmin=0._wp, current_hinew=0._wp
   REAL(wp) :: current_picard=-1._wp
   REAL(wp) :: current_sal_himin=0._wp, current_flushrate=0._wp, current_vbrc=0._wp
   REAL(wp) :: current_alpha_rjw=0._wp, current_rc_rjw=0._wp
   REAL(wp) :: current_apnd_min=0._wp, current_apnd_max=0._wp, current_pnd_flush=0._wp
   REAL(wp) :: current_icesal=0._wp, current_sal_gd=0._wp, current_time_gd=0._wp
   REAL(wp) :: current_sal_fl=0._wp, current_time_fl=0._wp
   REAL(wp) :: current_alpha_gn=0._wp, current_rc_gn=0._wp, current_alpha_cw=0._wp
   REAL(wp) :: current_alb_sdry=0._wp, current_alb_smlt=0._wp, current_alb_idry=0._wp
   REAL(wp) :: current_alb_imlt=0._wp, current_alb_dpnd=0._wp, current_alb_lpnd=0._wp
   REAL(wp) :: current_alb_hpiv=0._wp
   REAL(wp) :: current_apnd=0._wp, current_hpnd=0._wp
   REAL(wp) :: current_pnd_hl_min=0._wp, current_pnd_hl_max=0._wp
   INTEGER :: current_ndrainage=-1, current_nflushing=-1, current_sal_scheme=-1, current_nice_pnd=-1
   INTEGER :: current_liquidus=-1
   LOGICAL :: current_drainage=.FALSE., current_flushing=.FALSE.
   LOGICAL :: current_pnd_lids=.FALSE., current_pnd_rain=.FALSE.
   LOGICAL :: current_sal_chk=.FALSE., current_pnd=.FALSE., current_pnd_lev=.FALSE.
   LOGICAL :: current_pnd_cst=.FALSE., current_pnd_topo=.FALSE., current_pnd_alb=.FALSE.

CONTAINS

   SUBROUTINE o8o16_set_context(kt,kmm,nit000,rank,jpni,jpnj,rdt,r1dt)
      INTEGER, INTENT(in) :: kt,kmm,nit000,rank,jpni,jpnj
      REAL(wp), INTENT(in) :: rdt,r1dt
      current_kt=kt ; current_kmm=kmm ; current_nit000=nit000 ; current_rank=rank
      current_jpni=jpni ; current_jpnj=jpnj ; current_rdt=rdt ; current_r1dt=r1dt
      current_picard=-1._wp
   END SUBROUTINE o8o16_set_context

   SUBROUTINE o8o16_da_config(beta,dmin)
      REAL(wp),INTENT(in)::beta,dmin
      current_beta=beta ; current_dmin=dmin
   END SUBROUTINE o8o16_da_config

   SUBROUTINE o8o16_albedo_config(alb_sdry,alb_smlt,alb_idry,alb_imlt,alb_dpnd,alb_lpnd,alb_hpiv)
      REAL(wp),INTENT(in)::alb_sdry,alb_smlt,alb_idry,alb_imlt,alb_dpnd,alb_lpnd,alb_hpiv
      current_alb_sdry=alb_sdry ; current_alb_smlt=alb_smlt ; current_alb_idry=alb_idry
      current_alb_imlt=alb_imlt ; current_alb_dpnd=alb_dpnd ; current_alb_lpnd=alb_lpnd
      current_alb_hpiv=alb_hpiv
   END SUBROUTINE o8o16_albedo_config

   SUBROUTINE o8o16_salinity_config(ldrain,lflush,ndrain,nflush,sal_himin,flushrate,vbrc,sal_scheme,alpha_rjw,rc_rjw, &
      &                             icesal,sal_gd,time_gd,sal_fl,time_fl,liquidus,alpha_gn,rc_gn,alpha_cw,lsal_chk)
      LOGICAL,INTENT(in)::ldrain,lflush,lsal_chk
      INTEGER,INTENT(in)::ndrain,nflush,sal_scheme,liquidus
      REAL(wp),INTENT(in)::sal_himin,flushrate,vbrc,alpha_rjw,rc_rjw,icesal,sal_gd,time_gd,sal_fl,time_fl
      REAL(wp),INTENT(in)::alpha_gn,rc_gn,alpha_cw
      current_drainage=ldrain ; current_flushing=lflush
      current_ndrainage=ndrain ; current_nflushing=nflush ; current_sal_scheme=sal_scheme
      current_sal_himin=sal_himin ; current_flushrate=flushrate ; current_vbrc=vbrc
      current_alpha_rjw=alpha_rjw ; current_rc_rjw=rc_rjw
      current_icesal=icesal ; current_sal_gd=sal_gd ; current_time_gd=time_gd
      current_sal_fl=sal_fl ; current_time_fl=time_fl ; current_liquidus=liquidus
      current_alpha_gn=alpha_gn ; current_rc_gn=rc_gn ; current_alpha_cw=alpha_cw
      current_sal_chk=lsal_chk
   END SUBROUTINE o8o16_salinity_config

   SUBROUTINE o8o16_pond_config(nice_pnd,apnd_min,apnd_max,pnd_flush,lpnd_lids,lpnd_rain, &
      &                        lpnd,lpnd_lev,lpnd_cst,lpnd_topo,apnd,hpnd,lpnd_alb,pnd_hl_min,pnd_hl_max)
      INTEGER,INTENT(in)::nice_pnd
      REAL(wp),INTENT(in)::apnd_min,apnd_max,pnd_flush,apnd,hpnd,pnd_hl_min,pnd_hl_max
      LOGICAL,INTENT(in)::lpnd_lids,lpnd_rain,lpnd,lpnd_lev,lpnd_cst,lpnd_topo,lpnd_alb
      current_nice_pnd=nice_pnd ; current_apnd_min=apnd_min ; current_apnd_max=apnd_max
      current_pnd_flush=pnd_flush ; current_pnd_lids=lpnd_lids ; current_pnd_rain=lpnd_rain
      current_pnd=lpnd ; current_pnd_lev=lpnd_lev ; current_pnd_cst=lpnd_cst
      current_pnd_topo=lpnd_topo ; current_apnd=apnd ; current_hpnd=hpnd
      current_pnd_alb=lpnd_alb ; current_pnd_hl_min=pnd_hl_min ; current_pnd_hl_max=pnd_hl_max
   END SUBROUTINE o8o16_pond_config

   SUBROUTINE open_record(kind,jl,npti_local,nfields,u)
      INTEGER, INTENT(in) :: kind,jl,npti_local,nfields
      INTEGER, INTENT(out) :: u
      CHARACTER(LEN=8), PARAMETER :: magic='O816M001'
      CHARACTER(LEN=192) :: path,label,suffix
      INTEGER(int32), DIMENSION(20) :: meta
      REAL(real64), DIMENSION(59) :: scalars
      SELECT CASE(kind)
      CASE(1); label='frazil_entry'
      CASE(2); label='frazil_exit'
      CASE(3); label='category_entry'
      CASE(4); label='zdf_exit'
      CASE(5); label='dh_exit'
      CASE(6); label='temp1_exit_sal_entry'
      CASE(7); label='sal_exit'
      CASE(8); label='temp2_exit_da_entry'
      CASE(9); label='da_exit'
      CASE(10); label='pond_entry_category_done'
      CASE(11); label='pond_exit_itd_entry'
      CASE(12); label='itd_exit_do_entry'
      CASE(13); label='do_exit_cor_entry'
      CASE(14); label='cor_exit'
      CASE(15); label='update_flx_entry'
      CASE(16); label='update_flx_exit_ocean_boundary'
      CASE(17); label='picard_final'
      CASE(18); label='pre_aggregation'
      CASE(19); label='tau_entry'
      CASE(20); label='tau_exit'
      CASE(21); label='pre_category_state'
      CASE DEFAULT; ERROR STOP 'O8-O16 unknown record kind'
      END SELECT
      IF(jl > 0) THEN
         WRITE(suffix,'("_cat",I2.2)') jl
      ELSE
         suffix=''
      ENDIF
      WRITE(path,'("o8_o16_master_records/o8o16_kind",I2.2,"_",A,A,"_step",I4.4,"_rank",I4.4,".bin")') &
         kind,TRIM(label),TRIM(suffix),current_kt,current_rank
      OPEN(NEWUNIT=u,FILE=TRIM(path),ACCESS='stream',FORM='unformatted',STATUS='replace',ACTION='write')
      meta=[kind,current_kt,jl,current_rank,jpi,jpj,jpl,nlay_i,nlay_s,ntsi,ntei,ntsj,ntej, &
            npti_local,nfields,1,current_nit000,current_kmm,1,1]
      scalars=[current_rdt,current_r1dt,current_beta,current_dmin,current_hinew,current_picard, &
         rcnd_s,rn_kappa_i,rn_kappa_s,rn_snwblow,rn_sinew,REAL(nn_icesal,wp), &
         r1_nlay_i,r1_nlay_s,rn_simin,rn_himin, &
         MERGE(1._wp,0._wp,current_drainage),MERGE(1._wp,0._wp,current_flushing), &
         REAL(current_ndrainage,wp),REAL(current_nflushing,wp),current_sal_himin,current_flushrate, &
         current_vbrc,REAL(current_sal_scheme,wp),current_alpha_rjw,current_rc_rjw, &
         REAL(current_nice_pnd,wp),current_apnd_min,current_apnd_max,current_pnd_flush, &
         MERGE(1._wp,0._wp,current_pnd_lids),MERGE(1._wp,0._wp,current_pnd_rain), &
         current_icesal,current_sal_gd,current_time_gd,current_sal_fl,current_time_fl, &
         REAL(current_liquidus,wp),current_alpha_gn,current_rc_gn,current_alpha_cw, &
         MERGE(1._wp,0._wp,current_sal_chk),MERGE(1._wp,0._wp,current_pnd), &
         MERGE(1._wp,0._wp,current_pnd_lev),MERGE(1._wp,0._wp,current_pnd_cst), &
         MERGE(1._wp,0._wp,current_pnd_topo),current_apnd,current_hpnd, &
         MERGE(1._wp,0._wp,current_pnd_alb),current_pnd_hl_min,current_pnd_hl_max, &
         current_alb_sdry,current_alb_smlt,current_alb_idry,current_alb_imlt, &
         current_alb_dpnd,current_alb_lpnd,current_alb_hpiv,REAL(nn_snwfra,wp)]
      WRITE(u) magic ; WRITE(u) INT(1,int32) ; WRITE(u) meta ; WRITE(u) scalars
   END SUBROUTINE open_record

   SUBROUTINE put_i2(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:),INTENT(in)::a
      INTEGER::ni,nj
      ni=ntei-ntsi+1 ; nj=ntej-ntsj+1
      IF(SIZE(a,1)==ni .AND. SIZE(a,2)==nj) THEN
         WRITE(u) a
      ELSEIF(SIZE(a,1)==jpi .AND. SIZE(a,2)==jpj) THEN
         WRITE(u) a(ntsi:ntei,ntsj:ntej)
      ELSE
         ERROR STOP 'O8-O16 I2 extent mismatch'
      ENDIF
   END SUBROUTINE put_i2

   SUBROUTINE put_f2(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:),INTENT(in)::a
      IF(SIZE(a,1)/=jpi .OR. SIZE(a,2)/=jpj) ERROR STOP 'O8-O16 F2 extent mismatch'
      WRITE(u) a
   END SUBROUTINE put_f2

   SUBROUTINE put_h1(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:),INTENT(in)::a
      IF(SIZE(a,1)/=jpi-2*nn_hls+2 .OR. SIZE(a,2)/=jpj-2*nn_hls+2) &
         ERROR STOP 'O8-O16 H1 extent mismatch'
      WRITE(u) a
   END SUBROUTINE put_h1

   SUBROUTINE put_i3(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:,:),INTENT(in)::a
      IF(SIZE(a,3)/=jpl) ERROR STOP 'O8-O16 I3 category mismatch'
      IF(SIZE(a,1)==jpi .AND. SIZE(a,2)==jpj) THEN
         WRITE(u) a(ntsi:ntei,ntsj:ntej,1:jpl)
      ELSEIF(SIZE(a,1)==ntei-ntsi+1 .AND. SIZE(a,2)==ntej-ntsj+1) THEN
         WRITE(u) a
      ELSE
         ERROR STOP 'O8-O16 I3 extent mismatch'
      ENDIF
   END SUBROUTINE put_i3

   SUBROUTINE put_ii4(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:,:,:),INTENT(in)::a
      IF(SIZE(a,1)/=jpi .OR. SIZE(a,2)/=jpj .OR. SIZE(a,3)/=nlay_i .OR. SIZE(a,4)/=jpl) &
         ERROR STOP 'O8-O16 II4 extent mismatch'
      WRITE(u) a(ntsi:ntei,ntsj:ntej,1:nlay_i,1:jpl)
   END SUBROUTINE put_ii4

   SUBROUTINE put_is4(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:,:,:),INTENT(in)::a
      IF(SIZE(a,1)/=jpi .OR. SIZE(a,2)/=jpj .OR. SIZE(a,3)/=nlay_s .OR. SIZE(a,4)/=jpl) &
         ERROR STOP 'O8-O16 IS4 extent mismatch'
      WRITE(u) a(ntsi:ntei,ntsj:ntej,1:nlay_s,1:jpl)
   END SUBROUTINE put_is4

   SUBROUTINE put_g1(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:),INTENT(in)::a
      IF(SIZE(a)<npti) ERROR STOP 'O8-O16 G1 extent mismatch'
      WRITE(u) a(1:npti)
   END SUBROUTINE put_g1

   SUBROUTINE put_gi(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:),INTENT(in)::a
      IF(SIZE(a,1)<npti .OR. SIZE(a,2)/=nlay_i) ERROR STOP 'O8-O16 GI extent mismatch'
      WRITE(u) a(1:npti,1:nlay_i)
   END SUBROUTINE put_gi

   SUBROUTINE put_gs(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:),INTENT(in)::a
      IF(SIZE(a,1)<npti .OR. SIZE(a,2)/=nlay_s) ERROR STOP 'O8-O16 GS extent mismatch'
      WRITE(u) a(1:npti,1:nlay_s)
   END SUBROUTINE put_gs

   SUBROUTINE put_gi0(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,0:),INTENT(in)::a
      IF(SIZE(a,1)<npti .OR. SIZE(a,2)/=nlay_i+1) ERROR STOP 'O8-O16 GI0 extent mismatch'
      WRITE(u) a(1:npti,0:nlay_i)
   END SUBROUTINE put_gi0

   SUBROUTINE put_gs0(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,0:),INTENT(in)::a
      IF(SIZE(a,1)<npti .OR. SIZE(a,2)/=nlay_s+1) ERROR STOP 'O8-O16 GS0 extent mismatch'
      WRITE(u) a(1:npti,0:nlay_s)
   END SUBROUTINE put_gs0

   SUBROUTINE put_c0(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(0:),INTENT(in)::a
      IF(SIZE(a)/=jpl+1) ERROR STOP 'O8-O16 C0 extent mismatch'
      WRITE(u) a(0:jpl)
   END SUBROUTINE put_c0

   SUBROUTINE put_c1(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:),INTENT(in)::a
      IF(SIZE(a)/=jpl) ERROR STOP 'O8-O16 C1 extent mismatch'
      WRITE(u) a(1:jpl)
   END SUBROUTINE put_c1

   SUBROUTINE put_idx(u)
      INTEGER,INTENT(in)::u
      WRITE(u) INT(nptidx(1:npti),int32)
   END SUBROUTINE put_idx

   SUBROUTINE o8o16_frazil(kind,rn_hinew)
      INTEGER,INTENT(in)::kind
      REAL(wp),INTENT(in)::rn_hinew
      INTEGER::u
      current_hinew=rn_hinew
      IF(kind==1) THEN
         CALL open_record(kind,0,0,10,u)
         CALL put_i2(u,qlead); CALL put_i2(u,smask0); CALL put_f2(u,at_i)
         CALL put_f2(u,u_ice); CALL put_f2(u,v_ice)
         CALL put_f2(u,umask(:,:,1)); CALL put_f2(u,vmask(:,:,1))
         CALL put_f2(u,utau_ice); CALL put_f2(u,vtau_ice); CALL put_c0(u,hi_max)
      ELSEIF(kind==2) THEN
         CALL open_record(kind,0,0,2,u)
         CALL put_i2(u,ht_i_new); CALL put_i2(u,fraz_frac)
      ELSE
         ERROR STOP 'O8 frazil kind'
      ENDIF
      CLOSE(u)
   END SUBROUTINE o8o16_frazil

   SUBROUTINE put_cat_base(u)
      INTEGER,INTENT(in)::u
      CALL put_g1(u,a_i_1d); CALL put_g1(u,at_i_1d); CALL put_g1(u,h_i_1d); CALL put_g1(u,h_s_1d)
      CALL put_g1(u,t_su_1d); CALL put_g1(u,s_i_1d); CALL put_g1(u,t_bo_1d); CALL put_g1(u,sss_1d)
      CALL put_g1(u,sst_1d); CALL put_g1(u,qns_ice_1d); CALL put_g1(u,qsr_ice_1d)
      CALL put_g1(u,dqns_ice_1d); CALL put_g1(u,qtr_ice_top_1d); CALL put_g1(u,qsb_ice_bot_1d)
      CALL put_g1(u,fhld_1d); CALL put_g1(u,frq_m_1d); CALL put_g1(u,sprecip_1d)
      CALL put_g1(u,qprec_ice_1d); CALL put_g1(u,evap_ice_1d); CALL put_g1(u,hfx_err_dif_1d)
      CALL put_g1(u,hfx_dif_1d); CALL put_g1(u,qml_ice_1d); CALL put_g1(u,qcn_ice_1d); CALL put_g1(u,sfx_bri_1d)
      CALL put_g1(u,sfx_lam_1d); CALL put_g1(u,wfx_lam_1d); CALL put_g1(u,o_i_1d)
      CALL put_gi(u,t_i_1d); CALL put_gi(u,e_i_1d)
      CALL put_gi(u,sz_i_1d); CALL put_gs(u,t_s_1d); CALL put_gs(u,e_s_1d)
   END SUBROUTINE put_cat_base

   SUBROUTINE put_dh(u)
      INTEGER,INTENT(in)::u
      CALL put_g1(u,dh_s_tot); CALL put_g1(u,dh_i_sum); CALL put_g1(u,dh_i_itm)
      CALL put_g1(u,dh_i_bom); CALL put_g1(u,dh_i_bog); CALL put_g1(u,dh_i_sub)
      CALL put_g1(u,dh_s_sum); CALL put_g1(u,dh_s_itm); CALL put_g1(u,dh_snowice)
   END SUBROUTINE put_dh

   SUBROUTINE put_dh_fluxes(u)
      INTEGER,INTENT(in)::u
      CALL put_g1(u,hfx_res_1d); CALL put_g1(u,wfx_snw_sum_1d); CALL put_g1(u,hfx_spr_1d)
      CALL put_g1(u,wfx_spr_1d); CALL put_g1(u,hfx_sub_1d); CALL put_g1(u,wfx_snw_sub_1d)
      CALL put_g1(u,hfx_snw_1d); CALL put_g1(u,wfx_res_1d); CALL put_g1(u,sfx_res_1d)
      CALL put_g1(u,hfx_thd_1d); CALL put_g1(u,hfx_sum_1d); CALL put_g1(u,wfx_sum_1d)
      CALL put_g1(u,sfx_sum_1d); CALL put_g1(u,wfx_ice_sub_1d); CALL put_g1(u,sfx_sub_1d)
      CALL put_g1(u,wfx_err_sub_1d); CALL put_g1(u,hfx_bog_1d); CALL put_g1(u,wfx_bog_1d)
      CALL put_g1(u,sfx_bog_1d); CALL put_g1(u,hfx_bom_1d); CALL put_g1(u,wfx_bom_1d)
      CALL put_g1(u,sfx_bom_1d); CALL put_g1(u,sfx_sni_1d); CALL put_g1(u,wfx_sni_1d)
      CALL put_g1(u,wfx_snw_sni_1d)
   END SUBROUTINE put_dh_fluxes

   SUBROUTINE o8o16_category(kind,jl,rn_beta,rn_dmin)
      INTEGER,INTENT(in)::kind,jl
      REAL(wp),INTENT(in),OPTIONAL::rn_beta,rn_dmin
      INTEGER::u, record_jl
      record_jl=jl
      IF(jl > 0) current_jl=jl
      IF(kind == 3) current_picard=-1._wp
      IF(jl == 0) record_jl=current_jl
      IF(record_jl < 1 .OR. record_jl > jpl) ERROR STOP 'O9-O11 category unavailable'
      IF(PRESENT(rn_beta)) current_beta=rn_beta
      IF(PRESENT(rn_dmin)) current_dmin=rn_dmin
      SELECT CASE(kind)
      CASE(3)
         CALL open_record(kind,record_jl,npti,70,u); CALL put_idx(u); CALL put_cat_base(u); CALL put_dh(u); CALL put_dh_fluxes(u)
         CALL put_g1(u,qcn_ice_top_1d); CALL put_g1(u,qcn_ice_bot_1d); CALL put_g1(u,qtr_ice_bot_1d)
      CASE(5,6,7,8,9)
         CALL open_record(kind,record_jl,npti,67,u); CALL put_idx(u); CALL put_cat_base(u); CALL put_dh(u); CALL put_dh_fluxes(u)
      CASE(4)
         CALL open_record(kind,record_jl,npti,16,u); CALL put_idx(u)
         CALL put_g1(u,t_su_1d); CALL put_g1(u,qns_ice_1d); CALL put_g1(u,qcn_ice_top_1d)
         CALL put_g1(u,qcn_ice_bot_1d); CALL put_g1(u,qtr_ice_bot_1d); CALL put_g1(u,hfx_err_dif_1d)
         CALL put_g1(u,hfx_dif_1d); CALL put_g1(u,cnd_ice_1d); CALL put_g1(u,t1_ice_1d)
         CALL put_g1(u,t_si_1d); CALL put_gi(u,t_i_1d); CALL put_gi(u,e_i_1d)
         CALL put_gs(u,t_s_1d); CALL put_gs(u,e_s_1d); CALL put_g1(u,o_i_1d)
      CASE DEFAULT
         ERROR STOP 'O9-O11 category kind'
      END SELECT
      CLOSE(u)
   END SUBROUTINE o8o16_category

   SUBROUTINE o8o16_picard(iconv,lconv,zkappa_i,zkappa_s,zradab_i,zradab_s,zradtr_i,zradtr_s)
      INTEGER,INTENT(in)::iconv
      LOGICAL,DIMENSION(:),INTENT(in)::lconv
      REAL(wp),DIMENSION(:,0:),INTENT(in)::zkappa_i,zkappa_s,zradab_i,zradab_s,zradtr_i,zradtr_s
      INTEGER::u
      INTEGER(int32),ALLOCATABLE::conv(:)
      IF(current_kt/=1 .AND. current_kt/=2 .AND. current_kt/=100) RETURN
      current_picard=REAL(iconv,wp)
      ALLOCATE(conv(npti)); conv=MERGE(1_int32,0_int32,lconv(1:npti))
      IF(current_jl < 1 .OR. current_jl > jpl) ERROR STOP 'O9 Picard category unavailable'
      CALL open_record(17,current_jl,npti,12,u); CALL put_idx(u); WRITE(u) conv
      CALL put_g1(u,t_su_1d); CALL put_g1(u,qns_ice_1d)
      CALL put_gi(u,t_i_1d); CALL put_gs(u,t_s_1d)
      CALL put_gi0(u,zkappa_i); CALL put_gi(u,zradab_i(:,1:nlay_i)); CALL put_gi0(u,zradtr_i)
      CALL put_gs0(u,zkappa_s); CALL put_gs(u,zradab_s(:,1:nlay_s)); CALL put_gs0(u,zradtr_s)
      CLOSE(u); DEALLOCATE(conv)
   END SUBROUTINE o8o16_picard

   SUBROUTINE put_state(u)
      INTEGER,INTENT(in)::u
      CALL put_i3(u,a_i); CALL put_i3(u,v_i); CALL put_i3(u,v_s); CALL put_i3(u,h_i)
      CALL put_i3(u,h_s); CALL put_i3(u,s_i); CALL put_i3(u,sv_i); CALL put_i3(u,o_i)
      CALL put_i3(u,oa_i); CALL put_i3(u,t_su); CALL put_i3(u,a_ip); CALL put_i3(u,v_ip)
      CALL put_i3(u,h_ip); CALL put_i3(u,v_il); CALL put_i3(u,h_il); CALL put_i3(u,a_ip_frac)
      CALL put_i3(u,a_ip_eff); CALL put_ii4(u,t_i); CALL put_ii4(u,e_i); CALL put_ii4(u,sz_i)
      CALL put_ii4(u,szv_i); CALL put_is4(u,t_s); CALL put_is4(u,e_s)
   END SUBROUTINE put_state

   SUBROUTINE put_totals(u)
      INTEGER,INTENT(in)::u
      CALL put_i2(u,at_i); CALL put_i2(u,ato_i); CALL put_i2(u,vt_i); CALL put_i2(u,vt_s)
      CALL put_i2(u,st_i); CALL put_i2(u,et_i); CALL put_i2(u,et_s); CALL put_i2(u,tm_i)
      CALL put_i2(u,tm_s); CALL put_i2(u,sm_i); CALL put_i2(u,tm_su); CALL put_i2(u,hm_i)
      CALL put_i2(u,hm_s); CALL put_i2(u,om_i); CALL put_i2(u,at_ip); CALL put_i2(u,at_ip_eff)
      CALL put_i2(u,hm_ip); CALL put_i2(u,vt_ip); CALL put_i2(u,hm_il); CALL put_i2(u,vt_il)
      CALL put_i2(u,tm_si)
   END SUBROUTINE put_totals

   SUBROUTINE put_pre_agg_totals(u)
      INTEGER,INTENT(in)::u
      CALL put_i2(u,at_i); CALL put_i2(u,ato_i); CALL put_i2(u,vt_i); CALL put_i2(u,vt_s)
      CALL put_i2(u,st_i); CALL put_i2(u,et_i); CALL put_i2(u,et_s); CALL put_i2(u,tm_su)
      CALL put_i2(u,at_ip); CALL put_i2(u,at_ip_eff); CALL put_i2(u,vt_ip); CALL put_i2(u,vt_il)
   END SUBROUTINE put_pre_agg_totals

   SUBROUTINE put_pond(u)
      INTEGER,INTENT(in)::u
      CALL put_i3(u,dh_i_sum_2d); CALL put_i3(u,dh_s_sum_2d)
   END SUBROUTINE put_pond

   SUBROUTINE put_global_aux(u)
      INTEGER,INTENT(in)::u
      CALL put_i3(u,h_i_b); CALL put_i3(u,a_i_b)
      CALL put_i2(u,t_bo); CALL put_i2(u,ht_i_new); CALL put_i2(u,fraz_frac)
      CALL put_i2(u,sss_m); CALL put_i2(u,sst_m); CALL put_i2(u,rn_amax_2d)
      CALL put_f2(u,tmask(:,:,1)); CALL put_i2(u,smask0)
      CALL put_c0(u,hi_max); CALL put_c1(u,hi_mean)
   END SUBROUTINE put_global_aux

   SUBROUTINE put_pond_forcing(u)
      INTEGER,INTENT(in)::u
      CALL put_i2(u,sprecip); CALL put_i2(u,tprecip)
      CALL put_i3(u,qsr_ice); CALL put_i3(u,qns_ice)
   END SUBROUTINE put_pond_forcing

   SUBROUTINE put_flux(u)
      INTEGER,INTENT(in)::u
      CALL put_i2(u,wfx_snw); CALL put_i2(u,wfx_snw_sni); CALL put_i2(u,wfx_snw_sum)
      CALL put_i2(u,wfx_pnd); CALL put_i2(u,wfx_spr); CALL put_i2(u,wfx_sub)
      CALL put_i2(u,wfx_snw_sub); CALL put_i2(u,wfx_ice_sub); CALL put_i2(u,wfx_ice)
      CALL put_i2(u,wfx_sni); CALL put_i2(u,wfx_opw); CALL put_i2(u,wfx_bog)
      CALL put_i2(u,wfx_dyn); CALL put_i2(u,wfx_bom); CALL put_i2(u,wfx_sum)
      CALL put_i2(u,wfx_lam); CALL put_i2(u,wfx_res); CALL put_i2(u,wfx_err_sub)
      CALL put_i2(u,sfx_bog); CALL put_i2(u,sfx_bom); CALL put_i2(u,sfx_lam)
      CALL put_i2(u,sfx_sum); CALL put_i2(u,sfx_sni); CALL put_i2(u,sfx_opw)
      CALL put_i2(u,sfx_bri); CALL put_i2(u,sfx_dyn); CALL put_i2(u,sfx_res)
      CALL put_i2(u,sfx_sub); CALL put_i2(u,hfx_bog); CALL put_i2(u,hfx_bom)
      CALL put_i2(u,hfx_sum); CALL put_i2(u,hfx_opw); CALL put_i2(u,hfx_dif)
      CALL put_i2(u,hfx_snw); CALL put_i2(u,hfx_err_dif); CALL put_i2(u,hfx_sub)
      CALL put_i2(u,hfx_spr); CALL put_i2(u,hfx_thd); CALL put_i2(u,hfx_dyn)
      CALL put_i2(u,hfx_res); CALL put_i2(u,qlead); CALL put_i2(u,qsb_ice_bot)
      CALL put_i2(u,fhld); CALL put_i2(u,qt_atm_oi); CALL put_i2(u,qt_oce_ai)
   END SUBROUTINE put_flux

   SUBROUTINE o8o16_global(kind)
      INTEGER,INTENT(in)::kind
      INTEGER::u
      IF(kind==21) THEN
         CALL open_record(kind,0,0,23,u); CALL put_state(u)
      ELSEIF(kind>=10 .AND. kind<=14) THEN
         CALL open_record(kind,0,0,98,u); CALL put_state(u); CALL put_pre_agg_totals(u); CALL put_pond(u); CALL put_flux(u)
         CALL put_global_aux(u); CALL put_pond_forcing(u)
      ELSE
         ERROR STOP 'O9/O12-O15 global kind'
      ENDIF
      CLOSE(u)
   END SUBROUTINE o8o16_global

   SUBROUTINE o8o16_update(kind)
      INTEGER,INTENT(in)::kind
      INTEGER::u
      IF(kind==18) THEN
         CALL open_record(kind,0,0,26,u); CALL put_state(u)
         CALL put_i3(u,t_si); CALL put_i2(u,sss_m); CALL put_c0(u,hi_max)
      ELSEIF(kind==15) THEN
         CALL open_record(kind,0,0,118,u); CALL put_state(u); CALL put_totals(u); CALL put_flux(u)
         CALL put_i2(u,qsr_oce); CALL put_i2(u,qns_oce); CALL put_i2(u,qemp_oce); CALL put_i2(u,qemp_ice)
         CALL put_i2(u,emp_oce); CALL put_i2(u,emp_ice); CALL put_i2(u,qns_tot)
         CALL put_i2(u,qsr_tot); CALL put_i2(u,frq_m); CALL put_i3(u,a_i_b)
         CALL put_i3(u,qsr_ice); CALL put_i3(u,qns_ice); CALL put_i3(u,qtr_ice_bot)
         CALL put_i3(u,qevap_ice); CALL put_i3(u,qcn_ice); CALL put_i3(u,qcn_ice_top); CALL put_i3(u,qcn_ice_bot)
         CALL put_i3(u,qml_ice)
         CALL put_i3(u,qtr_ice_top); CALL put_i3(u,alb_ice); CALL put_i2(u,at_i_b); CALL put_i2(u,wfx_snw_dyn)
         CALL put_i2(u,qsr); CALL put_i2(u,sfx)
         CALL put_i2(u,cloud_fra); CALL put_f2(u,tmask(:,:,1)); CALL put_i2(u,smask0)
         CALL put_f2(u,snwice_mass); CALL put_f2(u,snwice_mass_b)
      ELSEIF(kind==16) THEN
         CALL open_record(kind,0,0,16,u)
         CALL put_i2(u,qsr); CALL put_i2(u,qns); CALL put_i2(u,emp); CALL put_i2(u,sfx)
         CALL put_i2(u,fwfice); CALL put_i2(u,qt_atm_oi); CALL put_i2(u,qt_oce_ai)
         CALL put_i2(u,wfx_ice); CALL put_i2(u,wfx_snw); CALL put_i2(u,wfx_pnd)
         CALL put_f2(u,fr_i)
         CALL put_i3(u,tn_ice); CALL put_i3(u,alb_ice)
         CALL put_f2(u,snwice_mass); CALL put_f2(u,snwice_mass_b); CALL put_f2(u,snwice_fmass)
      ELSE
         ERROR STOP 'O16 update kind'
      ENDIF
      CLOSE(u)
   END SUBROUTINE o8o16_update

   SUBROUTINE o8o16_tau(kind,pu_oce,pv_oce,ptmod_io,putau_oce,pvtau_oce)
      INTEGER,INTENT(in)::kind
      REAL(wp),DIMENSION(:,:),INTENT(in)::pu_oce,pv_oce,ptmod_io,putau_oce,pvtau_oce
      INTEGER::u
      IF(kind==19) THEN
         CALL open_record(kind,0,0,16,u)
         CALL put_f2(u,pu_oce); CALL put_f2(u,pv_oce)
         CALL put_f2(u,umask(:,:,1)); CALL put_f2(u,vmask(:,:,1))
         CALL put_f2(u,u_ice); CALL put_f2(u,v_ice); CALL put_f2(u,u_oce); CALL put_f2(u,v_oce)
         CALL put_f2(u,drag_io); CALL put_f2(u,at_i); CALL put_f2(u,ptmod_io)
         CALL put_f2(u,utau); CALL put_f2(u,vtau); CALL put_i2(u,taum)
         CALL put_i2(u,putau_oce); CALL put_i2(u,pvtau_oce)
      ELSEIF(kind==20) THEN
         CALL open_record(kind,0,0,4,u)
         CALL put_f2(u,utau); CALL put_f2(u,vtau); CALL put_i2(u,taum); CALL put_h1(u,rCdU_ice)
      ELSE
         ERROR STOP 'O16 tau kind'
      ENDIF
      CLOSE(u)
   END SUBROUTINE o8o16_tau

END MODULE o8_o16_master_writer
