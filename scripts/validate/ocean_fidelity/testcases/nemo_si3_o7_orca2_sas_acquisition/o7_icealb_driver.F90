PROGRAM o7_icealb_driver
   USE par_oce
   USE par_ice
   USE icealb
   USE o7_si3_writer, ONLY: o7_set_context, o7_set_coeffs
   IMPLICIT NONE
   REAL(wp), DIMENSION(Nis0-nn_hls:Nie0+nn_hls,Njs0-nn_hls:Nje0+nn_hls,jpl) :: ts, hi, hs, hp, hl
   REAL(wp), DIMENSION(Nis0:Nie0,Njs0:Nje0,jpl) :: pond, alb
   REAL(wp), DIMENSION(Nis0:Nie0,Njs0:Nje0) :: cloud

   ts=270._wp; hi=0.5_wp; hs=0.1_wp; hp=0.02_wp; hl=0.01_wp
   pond=0.2_wp; cloud=0.5_wp
   hs(Nis0,Njs0,1)=-0._wp
   CALL ice_alb_init
   CALL o7_set_context(1,2,1,0,1,1,5400._wp,1._wp/5400._wp,1026._wp,3991.86795711963_wp)
   CALL o7_set_coeffs(1.4e-3_wp,1.4e-3_wp,1.4e-3_wp)
   CALL ice_alb(.TRUE.,ts,hi,hs,pond,hp,hl,cloud,alb)
END PROGRAM o7_icealb_driver
