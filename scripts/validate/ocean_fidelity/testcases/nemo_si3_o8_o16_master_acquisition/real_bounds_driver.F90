PROGRAM real_bounds_driver
   USE par_oce
   USE par_ice
   USE par_kind, ONLY: wp
   USE phycst, ONLY: rcnd_s
   USE ice, ONLY: qlead, ht_i_new, fraz_frac
   USE icethd_do, ONLY: ice_thd_frazil
   USE o8_o16_master_writer, ONLY: o8o16_set_context
   IMPLICIT NONE
   INTEGER :: ji, jj
   jpi=6 ; jpj=6 ; jpk=2 ; jpkm1=1 ; nn_hls=1
   Nis0=2 ; Nie0=5 ; Njs0=2 ; Nje0=4
   ntsi=Nis0 ; ntei=Nie0 ; ntsj=Njs0 ; ntej=Nje0
   jpij=(Nie0-Nis0+1)*(Nje0-Njs0+1) ; jpni=1 ; jpnj=1
   jpl=5 ; nlay_i=10 ; nlay_s=5
   rDt_ice=10800._wp ; r1_Dt_ice=1._wp/rDt_ice
   r1_nlay_i=1._wp/REAL(nlay_i,wp) ; r1_nlay_s=1._wp/REAL(nlay_s,wp)
   rn_kappa_i=1._wp ; rn_kappa_s=10._wp ; rn_snwblow=0.66_wp
   rn_sinew=0.75_wp ; nn_icesal=4 ; rn_simin=0.1_wp ; rn_himin=0.1_wp
   rcnd_s=0.31_wp
   ALLOCATE(qlead(Nis0:Nie0,Njs0:Nje0),ht_i_new(Nis0:Nie0,Njs0:Nje0), &
      &     fraz_frac(Nis0:Nie0,Njs0:Nje0))
   DO jj=Njs0,Nje0 ; DO ji=Nis0,Nie0
      qlead(ji,jj)=-REAL(100*jj+ji,wp)
      ht_i_new(ji,jj)=REAL(10*jj+ji,wp)
      fraz_frac(ji,jj)=REAL(ji-jj,wp)
   END DO ; END DO
   qlead(Nis0,Njs0)=-0._wp ; ht_i_new(Nis0,Njs0)=-0._wp ; fraz_frac(Nis0,Njs0)=-0._wp
   CALL o8o16_set_context(1,2,1,0,1,1,rDt_ice,r1_Dt_ice)
   CALL ice_thd_frazil
   IF(ANY(ht_i_new /= 0._wp) .OR. ANY(fraz_frac /= 0._wp)) ERROR STOP 'real bounds frazil result'
   PRINT '(A)', 'PASS_REAL_PATCHED_BOUNDS_FIXTURE'
END PROGRAM real_bounds_driver
