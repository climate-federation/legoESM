MODULE l2_r154_transport
   USE par_kind, ONLY : wp
   USE dom_oce,  ONLY : jpi, jpj, jpk, ntsi, ntei, ntsj, ntej
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r154_transport_dump
   LOGICAL, SAVE :: ldumped = .FALSE.
CONTAINS
   SUBROUTINE r154_transport_dump( kt, kstg, Kbb, Kmm, Kaa, e2u, e1v, &
      & e3u_0, e3v_0, r3u, r3v, umask, vmask, uu, vv, zub, zvb, zFu, zFv, &
      & un_adv, vn_adv, r1_hu_0, r1_hv_0, uu_b, vv_b )
      INTEGER, INTENT(in) :: kt, kstg, Kbb, Kmm, Kaa
      REAL(wp), DIMENSION(:,:), INTENT(in) :: e2u, e1v, r3u, r3v
      REAL(wp), DIMENSION(:,:), INTENT(in) :: zub, zvb, un_adv, vn_adv
      REAL(wp), DIMENSION(:,:), INTENT(in) :: r1_hu_0, r1_hv_0, uu_b, vv_b
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: e3u_0, e3v_0, umask, vmask
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: uu, vv, zFu, zFv
      INTEGER :: unit, ios
      CHARACTER(LEN=16) :: magic
      IF( ldumped ) RETURN
      magic = 'NEMO_L2_R154TRP'
      OPEN( NEWUNIT=unit, FILE='oracle_developed_transport_kt00001081.bin', &
         & ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE', &
         & ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 ) ERROR STOP 'R154_TRANSPORT_OPEN_FAILED'
      WRITE(unit) magic
      WRITE(unit) 1, kt, kstg, Kbb, Kmm, Kaa, jpi, jpj, jpk, &
         & STORAGE_SIZE(1._wp), 20, ntsi, ntei, ntsj, ntej
      WRITE(unit) e2u, e1v, r3u, r3v, zub, zvb, un_adv, vn_adv, &
         & r1_hu_0, r1_hv_0, uu_b, vv_b
      WRITE(unit) e3u_0, e3v_0, umask, vmask, uu, vv, zFu, zFv
      CLOSE(unit)
      ldumped = .TRUE.
   END SUBROUTINE r154_transport_dump
END MODULE l2_r154_transport
