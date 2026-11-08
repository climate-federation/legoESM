! Dump the KF_LUTAB tables (TTAB, QSTAB, THE0K, ALU) for the constants on stdin:
!   SVP1 SVP2 SVP3 SVPT0
PROGRAM lut_dump
  USE module_cu_kfeta
  IMPLICIT NONE
  REAL :: s1, s2, s3, s0, tt(250,220), qs(250,220), th(220), al(200)
  INTEGER :: i, j
  READ(*,*) s1, s2, s3, s0
  CALL kf_lutab(s1, s2, s3, s0)
  CALL kf_dump_lut(tt, qs, th, al)
  DO j = 1, 220
    DO i = 1, 250
      WRITE(*,'(2(1X,ES25.17E3))') tt(i,j), qs(i,j)
    END DO
  END DO
  DO j = 1, 220
    WRITE(*,'(ES25.17E3)') th(j)
  END DO
  DO i = 1, 200
    WRITE(*,'(ES25.17E3)') al(i)
  END DO
END PROGRAM lut_dump
