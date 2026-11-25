PROGRAM real_bounds_driver
   USE, INTRINSIC :: ieee_arithmetic, ONLY: ieee_value, ieee_quiet_nan
   USE par_oce
   USE par_ice
   USE par_kind, ONLY: wp
   USE phycst, ONLY: rcnd_s
   USE dom_oce, ONLY: tmask, smask0, umask, vmask
   USE ice
   USE ice1D
   USE sbc_oce
   USE sbc_ice
   USE icetab, ONLY: tab_2d_1d
   USE icethd_do, ONLY: ice_thd_frazil
   USE o8_o16_master_writer
   IMPLICIT NONE
   INTEGER :: ji, jj, ierr, kind
   LOGICAL, ALLOCATABLE :: converged(:)
   REAL(wp), ALLOCATABLE :: pu(:,:), pv(:,:)
   REAL(wp), ALLOCATABLE :: zkappa_i(:,:), zradab_i(:,:), zradtr_i(:,:)
   REAL(wp), ALLOCATABLE :: zkappa_s(:,:), zradab_s(:,:), zradtr_s(:,:)

   INCLUDE 'generated_bounds_allocations.inc'
   rDt_ice=10800._wp ; r1_Dt_ice=1._wp/rDt_ice
   r1_nlay_i=1._wp/REAL(nlay_i,wp) ; r1_nlay_s=1._wp/REAL(nlay_s,wp)
   rn_kappa_i=1._wp ; rn_kappa_s=10._wp ; rn_snwblow=0.66_wp
   rn_sinew=0.75_wp ; nn_icesal=4 ; nn_snwfra=1 ; rn_simin=0.1_wp ; rn_himin=0.1_wp
   ln_zdf_chkcvg=.FALSE. ; ln_sal_chk=.FALSE. ; rcnd_s=0.31_wp

   ierr=ice_alloc() ; ierr=ierr+ice1D_alloc()
   IF(ierr/=0) ERROR STOP 'real bounds allocation'

   tmask=701._wp ; umask=702._wp ; vmask=703._wp ; smask0=704._wp
   qsr=705._wp ; qns=706._wp ; emp=707._wp ; sfx=708._wp ; fr_i=709._wp ; fwfice=710._wp
   qns_tot=711._wp ; qsr_tot=712._wp ; frq_m=713._wp ; sss_m=714._wp ; sst_m=714.5_wp
   cloud_fra=715._wp ; sprecip=716._wp ; tprecip=717._wp ; utau=718._wp ; vtau=719._wp ; taum=720._wp
   qsr_oce=721._wp ; qns_oce=722._wp ; qemp_oce=723._wp ; qemp_ice=724._wp
   emp_oce=725._wp ; emp_ice=726._wp ; qsr_ice=727._wp ; qns_ice=728._wp
   qevap_ice=729._wp ; qcn_ice=730._wp ; qcn_ice_top=730.25_wp ; qcn_ice_bot=730.5_wp
   qml_ice=731._wp ; qtr_ice_top=732._wp
   tn_ice=733._wp ; alb_ice=734._wp ; snwice_mass=735._wp ; snwice_mass_b=736._wp ; snwice_fmass=737._wp
   utau_ice=737.25_wp ; vtau_ice=737.5_wp ; rCdU_ice=738._wp

   a_i=801._wp ; v_i=802._wp ; v_s=803._wp ; h_i=804._wp ; h_s=805._wp ; s_i=806._wp
   sv_i=807._wp ; o_i=808._wp ; oa_i=809._wp ; t_su=810._wp ; a_ip=811._wp
   v_ip=812._wp ; h_ip=813._wp ; v_il=814._wp ; h_il=815._wp ; a_ip_frac=816._wp
   a_ip_eff=817._wp ; t_i=818._wp ; e_i=819._wp ; sz_i=820._wp ; szv_i=821._wp
   t_s=822._wp ; e_s=823._wp ; at_i=824._wp ; ato_i=825._wp ; vt_i=826._wp
   vt_s=827._wp ; st_i=828._wp ; et_i=829._wp ; et_s=830._wp ; tm_i=831._wp
   tm_s=832._wp ; sm_i=833._wp ; tm_su=834._wp ; hm_i=835._wp ; hm_s=836._wp
   om_i=837._wp ; at_ip=838._wp ; at_ip_eff=839._wp ; hm_ip=840._wp ; vt_ip=841._wp
   hm_il=842._wp ; vt_il=843._wp ; tm_si=845._wp
   dh_i_sum_2d=846._wp ; dh_s_sum_2d=847._wp ; h_i_b=848._wp ; a_i_b=849._wp
   t_bo=850._wp ; t_si=850.5_wp ; rn_amax_2d=853._wp
   hi_max=854._wp ; hi_mean=855._wp ; u_ice=856._wp ; v_ice=857._wp
   u_oce=858._wp ; v_oce=859._wp ; drag_io=860._wp

   wfx_snw=901._wp ; wfx_snw_sni=902._wp ; wfx_snw_sum=903._wp ; wfx_pnd=904._wp
   wfx_spr=905._wp ; wfx_sub=906._wp ; wfx_snw_sub=907._wp ; wfx_ice_sub=908._wp
   wfx_ice=909._wp ; wfx_sni=910._wp ; wfx_opw=911._wp ; wfx_bog=912._wp
   wfx_dyn=913._wp ; wfx_bom=914._wp ; wfx_sum=915._wp ; wfx_lam=916._wp
   wfx_res=917._wp ; wfx_err_sub=918._wp ; sfx_bog=919._wp ; sfx_bom=920._wp
   sfx_lam=921._wp ; sfx_sum=922._wp ; sfx_sni=923._wp ; sfx_opw=924._wp
   sfx_bri=925._wp ; sfx_dyn=926._wp ; sfx_res=927._wp ; sfx_sub=928._wp
   hfx_bog=929._wp ; hfx_bom=930._wp ; hfx_sum=931._wp ; hfx_opw=932._wp
   hfx_dif=933._wp ; hfx_snw=934._wp ; hfx_err_dif=935._wp ; hfx_sub=936._wp
   hfx_spr=937._wp ; hfx_thd=938._wp ; hfx_dyn=939._wp ; hfx_res=940._wp
   qlead=941._wp ; qsb_ice_bot=942._wp ; fhld=943._wp ; qt_atm_oi=944._wp
   qt_oce_ai=945._wp ; qtr_ice_bot=946._wp ; wfx_snw_dyn=947._wp

   npti=0
   DO jj=Njs0,Nje0 ; DO ji=Nis0,Nie0
      npti=npti+1 ; nptidx(npti)=(jj-1)*jpi+ji
   END DO ; END DO
   a_i_1d=1001._wp ; at_i_1d=1004._wp ; h_i_1d=1005._wp ; h_s_1d=1006._wp
   t_su_1d=1007._wp ; s_i_1d=1008._wp
   t_bo_1d=1009._wp ; sss_1d=1010._wp ; sst_1d=1011._wp ; qns_ice_1d=1012._wp
   qsr_ice_1d=1013._wp ; dqns_ice_1d=1014._wp ; qtr_ice_top_1d=1015._wp
   qsb_ice_bot_1d=1016._wp ; fhld_1d=1017._wp ; frq_m_1d=1018._wp
   sprecip_1d=1019._wp ; qprec_ice_1d=1020._wp ; evap_ice_1d=1021._wp
   hfx_err_dif_1d=1022._wp ; hfx_dif_1d=1023._wp ; qml_ice_1d=1024._wp
   sfx_bri_1d=1025._wp ; sfx_lam_1d=1026._wp ; wfx_lam_1d=1027._wp
   t_i_1d=1028._wp ; e_i_1d=1029._wp ; sz_i_1d=1030._wp ; t_s_1d=1031._wp ; e_s_1d=1032._wp
   dh_s_tot=1033._wp ; dh_i_sum=1034._wp ; dh_i_itm=1035._wp ; dh_i_bom=1036._wp
   dh_i_bog=1037._wp ; dh_i_sub=1038._wp ; dh_s_sum=1039._wp ; dh_s_itm=1040._wp
   dh_snowice=1041._wp ; hfx_res_1d=1042._wp ; wfx_snw_sum_1d=1043._wp
   hfx_spr_1d=1044._wp ; wfx_spr_1d=1045._wp ; hfx_sub_1d=1046._wp
   wfx_snw_sub_1d=1047._wp ; hfx_snw_1d=1048._wp ; wfx_res_1d=1049._wp
   sfx_res_1d=1050._wp ; hfx_thd_1d=1051._wp ; hfx_sum_1d=1052._wp
   wfx_sum_1d=1053._wp ; sfx_sum_1d=1054._wp ; wfx_ice_sub_1d=1055._wp
   sfx_sub_1d=1056._wp ; wfx_err_sub_1d=1057._wp ; hfx_bog_1d=1058._wp
   wfx_bog_1d=1059._wp ; sfx_bog_1d=1060._wp ; hfx_bom_1d=1061._wp
   wfx_bom_1d=1062._wp ; sfx_bom_1d=1063._wp ; sfx_sni_1d=1064._wp
   wfx_sni_1d=1065._wp ; wfx_snw_sni_1d=1066._wp

   qlead=-0._wp
   DO jj=Njs0,Nje0 ; DO ji=Nis0,Nie0
      qlead(ji,jj)=-REAL(100*jj+ji+7000,wp)
   END DO ; END DO
   qlead(Nis0,Njs0)=-0._wp
   CALL o8o16_set_context(1,2,1,7,1,1,rDt_ice,r1_Dt_ice)
   CALL o8o16_da_config(0.9_wp,8._wp)
   CALL o8o16_albedo_config(0.85_wp,0.75_wp,0.60_wp,0.50_wp,0.36_wp,0.40_wp,1._wp)
   CALL ice_thd_frazil
   CALL o8o16_global(21)
   o_i_1d=ieee_value(0._wp,ieee_quiet_nan)
   CALL tab_2d_1d(npti,nptidx(1:npti),o_i_1d(1:npti),o_i(:,:,1))
   dh_s_tot=0._wp ; dh_i_sum=0._wp ; dh_i_itm=0._wp ; dh_i_bom=0._wp
   dh_i_bog=0._wp ; dh_i_sub=0._wp ; dh_s_sum=0._wp ; dh_s_itm=0._wp ; dh_snowice=0._wp
   CALL o8o16_category(3,1)
   dh_s_tot=1033._wp ; dh_i_sum=1034._wp ; dh_i_itm=1035._wp ; dh_i_bom=1036._wp
   dh_i_bog=1037._wp ; dh_i_sub=1038._wp ; dh_s_sum=1039._wp ; dh_s_itm=1040._wp ; dh_snowice=1041._wp
   DO kind=4,9 ; CALL o8o16_category(kind,1) ; END DO
   converged=.TRUE. ; zkappa_i=1101._wp ; zkappa_s=1102._wp
   zradtr_i=1103._wp ; zradtr_s=1104._wp ; zradab_i=1105._wp ; zradab_s=1106._wp
   zradab_i(:,0)=ieee_value(0._wp,ieee_quiet_nan)
   zradab_s(:,0)=ieee_value(0._wp,ieee_quiet_nan)
   CALL o8o16_picard(4,converged,zkappa_i,zkappa_s,zradab_i,zradab_s,zradtr_i,zradtr_s)
   DO kind=10,14 ; CALL o8o16_global(kind) ; END DO
   CALL o8o16_update(18) ; CALL o8o16_update(15) ; CALL o8o16_update(16)
   pu=1201._wp ; pv=1202._wp
   CALL o8o16_tau(19,pu,pv,pu,pu,pv) ; CALL o8o16_tau(20,pu,pv,pu,pu,pv)
   PRINT '(A)', 'PASS_REAL_PATCHED_BOUNDS_FIXTURE'
END PROGRAM real_bounds_driver
