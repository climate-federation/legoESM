PROGRAM driver
   USE par_kind, ONLY : wp
   USE dom_oce
   USE l2_r156_stage2
   IMPLICIT NONE
   REAL(wp) :: a3(6,5,4), b3(6,5,4), a2(6,5), b2(6,5)
   REAL(wp) :: zu(4,3), zv(4,3)
   INTEGER :: i, j, k
   DO k = 1, 4 ; DO j = 1, 5 ; DO i = 1, 6
      a3(i,j,k) = REAL(i + 10*j + 100*k, wp)
      b3(i,j,k) = -a3(i,j,k)
   END DO ; END DO ; END DO
   DO j = 1, 5 ; DO i = 1, 6
      a2(i,j) = REAL(i + 10*j, wp) ; b2(i,j) = -a2(i,j)
   END DO ; END DO
   DO j = 1, 3 ; DO i = 1, 4
      zu(i,j) = REAL(1000 + i + 10*j, wp) ; zv(i,j) = -zu(i,j)
   END DO ; END DO
   CALL r156_stage2_open( 1081, 2, 1, 2, 3, 3 )
   CALL r156_stage2_pair3( 'rhs_entry       ', a3, b3 )
   CALL r156_stage2_pair2( 'r3u_r3v_Kbb     ', a2, b2 )
   CALL r156_stage2_scal ( 'rDt_r1_Dt       ', 7200.0_wp, 1.0_wp/7200.0_wp )
   CALL r156_stage2_pair2( 'zub_zvb         ', zu, zv )
   CALL r156_stage2_close()
END PROGRAM driver
