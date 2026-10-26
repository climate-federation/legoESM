MODULE l2_r48_bt_memory
   !! Round-48 WRITE-only kt=1 -> kt=2 barotropic-memory recorder.
   !! Every model array is INTENT(IN); no captured value is read back by NEMO.
   USE dom_oce,        ONLY : wp, jpi, jpj, ntsi, ntei, ntsj, ntej, l_istiled
   USE in_out_manager, ONLY : lwp, numout, nit000
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r48_bt_memory

   CHARACTER(LEN=16), PARAMETER :: r48_magic = 'NEMO_L2_R48BTM1'

CONTAINS

   SUBROUTINE put2(unit, name, value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(unit) field
      WRITE(unit) 2, SIZE(value,1), SIZE(value,2), 1
      WRITE(unit) value
   END SUBROUTINE put2

   SUBROUTINE r48_bt_memory(phase, kt, Kbb, Kmm, Kaa, puu_b, pvv_b, pssh, &
      &                     pun_e, pvn_e, psshn_e, pun_adv, pvn_adv, &
      &                     pubb_e, pub_e, pvbb_e, pvb_e, psshbb_e, psshb_e)
      INTEGER, INTENT(in) :: phase, kt, Kbb, Kmm, Kaa
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: puu_b, pvv_b, pssh
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pun_e, pvn_e, psshn_e
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pun_adv, pvn_adv
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pubb_e, pub_e, pvbb_e, pvb_e
      REAL(wp), DIMENSION(:,:), INTENT(in) :: psshbb_e, psshb_e
      INTEGER :: unit, ios
      CHARACTER(LEN=96) :: filename
      LOGICAL :: active

      active = lwp .AND. ((phase == 1 .AND. kt == nit000) .OR. &
         &                (phase == 2 .AND. kt == nit000 + 1))
      IF(.NOT.active) RETURN
      IF(l_istiled) CALL ctl_stop('round48: whole-array writer refuses tiling')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round48: writer requires 64-bit wp')
      IF(phase == 1) THEN
         WRITE(filename,'("oracle_bt_memory_kt",I8.8,"_end.bin")') kt
      ELSEIF(phase == 2) THEN
         WRITE(filename,'("oracle_bt_memory_kt",I8.8,"_start.bin")') kt
      ELSE
         CALL ctl_stop('round48: invalid memory phase')
      ENDIF
      OPEN(NEWUNIT=unit, FILE=TRIM(filename), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round48: cannot open memory record')
      WRITE(unit) r48_magic
      WRITE(unit) 1, kt, phase, Kbb, Kmm, Kaa, jpi, jpj, ntsi, ntei, ntsj, ntej, &
         & STORAGE_SIZE(1._wp)
      CALL put2(unit, 'ubb_e           ', pubb_e)
      CALL put2(unit, 'ub_e            ', pub_e)
      CALL put2(unit, 'vbb_e           ', pvbb_e)
      CALL put2(unit, 'vb_e            ', pvb_e)
      CALL put2(unit, 'sshbb_e         ', psshbb_e)
      CALL put2(unit, 'sshb_e          ', psshb_e)
      CALL put2(unit, 'un_e            ', pun_e)
      CALL put2(unit, 'vn_e            ', pvn_e)
      CALL put2(unit, 'sshn_e          ', psshn_e)
      CALL put2(unit, 'un_adv          ', pun_adv)
      CALL put2(unit, 'vn_adv          ', pvn_adv)
      CALL put2(unit, 'ubar_Kmm        ', puu_b(:,:,Kmm))
      CALL put2(unit, 'vbar_Kmm        ', pvv_b(:,:,Kmm))
      CALL put2(unit, 'ssh_Kmm         ', pssh(:,:,Kmm))
      CALL put2(unit, 'ubar_Kaa        ', puu_b(:,:,Kaa))
      CALL put2(unit, 'vbar_Kaa        ', pvv_b(:,:,Kaa))
      CALL put2(unit, 'ssh_Kaa         ', pssh(:,:,Kaa))
      CLOSE(unit)
      WRITE(numout,*) 'ROUND48_BT_MEMORY ', kt, phase, TRIM(filename)
   END SUBROUTINE r48_bt_memory

END MODULE l2_r48_bt_memory
