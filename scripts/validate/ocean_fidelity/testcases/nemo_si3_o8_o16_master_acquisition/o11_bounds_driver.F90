PROGRAM o11_bounds_driver
   USE par_kind, ONLY: wp
   USE ice1D
   USE icethd_da, ONLY: ice_thd_da_init, ice_thd_da
   IMPLICIT NONE
   INTEGER :: ji
   CALL ice_thd_da_init
   DO ji=1,npti
      at_i_1d(ji)=0.8_wp+REAL(ji,wp)*0.01_wp
      a_i_1d(ji)=0.1_wp*REAL(ji,wp)
      h_i_1d(ji)=REAL(ji,wp); h_s_1d(ji)=0.1_wp
      s_i_1d(ji)=5._wp; sst_1d(ji)=275._wp; t_bo_1d(ji)=271._wp
   END DO
   a_i_1d(1)=0._wp
   sz_i_1d=5._wp; e_i_1d=-3.e8_wp; e_s_1d=-1.e8_wp
   sfx_lam_1d=-0._wp; hfx_thd_1d=-0._wp; wfx_lam_1d=-0._wp
   CALL ice_thd_da
   IF(ANY(.NOT.(a_i_1d <= [0._wp,0.2_wp,0.3_wp]))) ERROR STOP 'bounds fixture DA result'
   PRINT '(A)', 'PASS_PATCHED_O11_BOUNDS_FIXTURE'
END PROGRAM o11_bounds_driver
