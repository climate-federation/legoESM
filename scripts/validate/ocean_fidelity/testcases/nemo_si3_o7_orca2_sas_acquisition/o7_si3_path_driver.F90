PROGRAM o7_si3_path_driver
   USE par_oce
   USE lib_mpp, ONLY: narea
   USE dom_oce, ONLY: o7_init_dom
   USE oce, ONLY: o7_init_oce
   USE ice, ONLY: o7_init_ice
   USE sbc_oce, ONLY: o7_init_sbc_oce
   USE sbc_ice, ONLY: o7_init_sbc_ice
   USE sbcblk, ONLY: sbc_blk_init,sf,precip,jp_wndi,jp_wndj,jp_tair,jp_humi,jp_qsr,jp_qlw,jp_prec,jp_snow,jp_slp
   USE icesbc, ONLY: ice_sbc_init
   USE icestp, ONLY: ice_stp
   USE in_out_manager, ONLY: numnam_ref,numnam_cfg,numnam_ice_ref,numnam_ice_cfg
   IMPLICIT NONE
   INTEGER :: rank
   CALL o7_init_dom
   CALL o7_init_oce
   numnam_ref='&namsbc_blk ln_NCAR=T,ln_COARE_3p0=F,ln_COARE_3p6=F,ln_ECMWF=F,ln_ANDREAS=F,ln_MFS=F,' // &
      'rn_zqt=2.,rn_zu=10.,nn_iter_algo=1,ln_skin_cs=F,ln_skin_wl=F,rn_pfac=1.,rn_efac=1.,' // &
      'ln_crt_fbk=F,rn_stau_a=0.,rn_stau_b=0.,ln_humi_sph=T,ln_humi_dpt=F,ln_humi_rlh=F,' // &
      'ln_tair_pot=T,ln_prec_met=F,ln_Cx_ice_cst=T,rn_Cd_ia=0.0014,rn_Ce_ia=0.0012,rn_Ch_ia=0.0012,' // &
      'ln_Cx_ice_frm=F,nn_frm=1,rn_Cs_io=0.001,rn_Cs_ia=0.001,rn_Cr_ia=0.,rn_Cr_io=0.,rn_Cf_ia=0.,rn_Cf_io=0.,' // &
      'ln_Cx_ice_AN05=F,ln_Cx_ice_LU12=F,ln_Cx_ice_LG15=F,cn_dir="./" /'
   numnam_cfg=numnam_ref
   numnam_ice_ref='&namsbc rn_Cd_io=0.005,nn_snwfra=0,rn_snwblow=1.,nn_flxdist=-1,ln_cndflx=F,ln_cndemulate=F,nn_qtrice=0 /'
   numnam_ice_cfg=numnam_ice_ref
   CALL sbc_blk_init
   CALL ice_sbc_init
   DO rank=0,1
      narea=rank+1
      CALL o7_init_ice(rank)
      CALL o7_init_sbc_oce(rank)
      CALL o7_init_sbc_ice(rank)
      sf(jp_wndi)%fnow=4._wp+rank; sf(jp_wndj)%fnow=-3._wp-rank
      sf(jp_tair)%fnow=270._wp+rank; sf(jp_humi)%fnow=0.002_wp+rank*1.e-4_wp
      sf(jp_qsr)%fnow=100._wp+rank; sf(jp_qlw)%fnow=-40._wp-rank
      sf(jp_prec)%fnow=1.e-5_wp; sf(jp_snow)%fnow=-0._wp; sf(jp_slp)%fnow=101325._wp+rank
      precip=sf(jp_prec)%fnow(:,:,1)
      CALL ice_stp(rank+1,1,2,2)
   END DO
END PROGRAM o7_si3_path_driver
