PROGRAM o4_bounds_driver
   USE par_kind, ONLY: wp
   USE par_ice
   USE ice
   USE sbc_oce, ONLY: sss_m,sst_m
   USE sbc_ice, ONLY: snwice_mass
   USE lib_mpp, ONLY: narea
   USE icedyn_adv_pra, ONLY: adv_pra_init,ice_dyn_adv_pra
   USE o4_prather_writer, ONLY: o4_capture_adv,o4_capture_stage
   IMPLICIT NONE
   e1u=10000._wp;e2v=10000._wp;r1_e1u=1._wp/e1u;r1_e2v=1._wp/e2v
   e2u=9500._wp;e1v=10500._wp;e1e2t=1.e8_wp;r1_e1e2t=1._wp/e1e2t
   tmask=1._wp;umask=1._wp;vmask=1._wp
   wfx_res=-0._wp;hfx_res=-0._wp;sfx_res=-0._wp
   diag_adv_mass=-0._wp;diag_adv_salt=-0._wp;diag_adv_heat=-0._wp
   diag_trp_vi=-0._wp;diag_trp_vs=-0._wp;diag_trp_ei=-0._wp;diag_trp_es=-0._wp;diag_trp_sv=-0._wp
   u_ice=0._wp;v_ice=0._wp;ato_i=1._wp;h_i=1._wp;h_s=0._wp;h_ip=0._wp;h_il=0._wp
   v_i=0._wp;v_s=0._wp;sv_i=0._wp;oa_i=0._wp;a_i=0._wp;a_ip=0._wp;v_ip=0._wp;v_il=0._wp
   t_su=0._wp;e_s=0._wp;e_i=0._wp;szv_i=0._wp;sz_i=0._wp;t_i=0._wp;t_s=0._wp
   at_i=0._wp;vt_i=0._wp;snwice_mass=0._wp;sss_m=0._wp;sst_m=-273.15_wp
   CALL adv_pra_init
   CALL o4_capture_adv(2,1)
   CALL ice_dyn_adv_pra(1,u_ice,v_ice,h_i,h_s,h_ip,ato_i,v_i,v_s,sv_i,oa_i,a_i,a_ip,v_ip,v_il,e_s,e_i,szv_i)
   CALL o4_capture_adv(5,1); CALL o4_capture_stage(7,1); CALL o4_capture_stage(8,1); CALL o4_capture_stage(9,1)
   narea=2
   ato_i=2._wp
   CALL o4_capture_adv(2,2)
   CALL ice_dyn_adv_pra(2,u_ice,v_ice,h_i,h_s,h_ip,ato_i,v_i,v_s,sv_i,oa_i,a_i,a_ip,v_ip,v_il,e_s,e_i,szv_i)
   CALL o4_capture_adv(5,2); CALL o4_capture_stage(7,2); CALL o4_capture_stage(8,2); CALL o4_capture_stage(9,2)
   PRINT '(A)', 'PASS_O4_PATCHED_BOUNDS_FIXTURE'
END PROGRAM o4_bounds_driver
