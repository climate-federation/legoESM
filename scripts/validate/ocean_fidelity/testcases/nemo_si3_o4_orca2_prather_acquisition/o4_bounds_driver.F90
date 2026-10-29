PROGRAM o4_bounds_driver
   USE par_kind, ONLY: wp
   USE par_ice
   USE ice
   USE lib_mpp, ONLY: narea
   USE icedyn_adv_pra, ONLY: adv_pra_init,ice_dyn_adv_pra
   IMPLICIT NONE
   REAL(wp) :: u(jpi,jpj),v(jpi,jpj),ato(jpi,jpj)
   REAL(wp) :: hi(jpi,jpj,jpl),hs(jpi,jpj,jpl),hip(jpi,jpj,jpl)
   REAL(wp) :: vi(jpi,jpj,jpl),vs(jpi,jpj,jpl),svi(jpi,jpj,jpl),oai(jpi,jpj,jpl)
   REAL(wp) :: ai(jpi,jpj,jpl),aip(jpi,jpj,jpl),vip(jpi,jpj,jpl),vil(jpi,jpj,jpl)
   REAL(wp) :: es(jpi,jpj,nlay_s,jpl),ei(jpi,jpj,nlay_i,jpl),szvi(jpi,jpj,nlay_i,jpl)
   e1u=10000._wp;e2v=10000._wp;r1_e1u=1._wp/e1u;r1_e2v=1._wp/e2v
   e2u=9500._wp;e1v=10500._wp;e1e2t=1.e8_wp;r1_e1e2t=1._wp/e1e2t
   tmask=1._wp;umask=1._wp;vmask=1._wp
   wfx_res=-0._wp;hfx_res=-0._wp;sfx_res=-0._wp
   diag_adv_mass=-0._wp;diag_adv_salt=-0._wp;diag_adv_heat=-0._wp
   u=0._wp;v=0._wp;ato=1._wp;hi=1._wp;hs=0._wp;hip=0._wp
   vi=0._wp;vs=0._wp;svi=0._wp;oai=0._wp;ai=0._wp;aip=0._wp;vip=0._wp;vil=0._wp
   es=0._wp;ei=0._wp;szvi=0._wp
   CALL adv_pra_init
   CALL ice_dyn_adv_pra(1,u,v,hi,hs,hip,ato,vi,vs,svi,oai,ai,aip,vip,vil,es,ei,szvi)
   narea=2
   ato=2._wp
   CALL ice_dyn_adv_pra(2,u,v,hi,hs,hip,ato,vi,vs,svi,oai,ai,aip,vip,vil,es,ei,szvi)
   PRINT '(A)', 'PASS_O4_PATCHED_BOUNDS_FIXTURE'
END PROGRAM o4_bounds_driver
