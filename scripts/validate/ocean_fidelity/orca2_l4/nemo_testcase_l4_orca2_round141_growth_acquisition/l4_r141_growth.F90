MODULE l4_r141_growth
   !! Rank-complete, self-describing, write-only step-30..36 growth record.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE in_out_manager, ONLY : numout
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r141_growth_entry, r141_growth_baro, r141_growth_stage1

   INTEGER, SAVE :: record_unit = -1
   INTEGER, SAVE :: current_kt = -1
   INTEGER, SAVE :: phase = 0

CONTAINS
   LOGICAL FUNCTION selected(kt)
      INTEGER, INTENT(in) :: kt
      selected = kt >= 30 .AND. kt <= 36
   END FUNCTION selected

   SUBROUTINE r141_growth_entry(kt, ssh_entry, r3t_entry, uub_entry, vvb_entry)
      INTEGER, INTENT(in) :: kt
      REAL(wp), DIMENSION(:,:), INTENT(in) :: ssh_entry, r3t_entry, uub_entry, vvb_entry
      INTEGER :: ios, length, status
      CHARACTER(LEN=512) :: output_dir, filename
      CHARACTER(LEN=16) :: magic
      IF(.NOT.selected(kt)) RETURN
      IF(record_unit /= -1 .OR. phase /= 0) CALL ctl_stop('round141: entry while record open')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round141: growth record requires fp64')
      output_dir = ''
      CALL GET_ENVIRONMENT_VARIABLE('ORCA2_R141_GROWTH_DIR', output_dir, &
         & LENGTH=length, STATUS=status)
      IF(status /= 0 .OR. length < 1 .OR. length > LEN(output_dir)) &
         & CALL ctl_stop('round141: missing growth output directory')
      IF(output_dir(1:1) /= '/') CALL ctl_stop('round141: growth output directory is not absolute')
      WRITE(filename,'(A,"/oracle_r141_growth_rank",I4.4,"_kt",I8.8,".bin")') &
         & TRIM(output_dir), mpprank, kt
      OPEN(NEWUNIT=record_unit, FILE=TRIM(filename), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round141: cannot initialize growth record')
      magic = 'NEMO_L4_R141G1'
      WRITE(record_unit) magic
      WRITE(record_unit) 1, kt, mpprank, jpi, jpj, jpk, nimpp, njmpp, &
         & ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 14
      CALL write_2d('ssh_entry', ssh_entry)
      CALL write_2d('r3t_entry', r3t_entry)
      CALL write_2d('uub_entry', uub_entry)
      CALL write_2d('vvb_entry', vvb_entry)
      current_kt = kt
      phase = 1
      WRITE(numout,*) 'ORCA2_R141_GROWTH_ENTRY ', kt, mpprank, TRIM(filename)
   END SUBROUTINE r141_growth_entry

   SUBROUTINE r141_growth_baro(kt, ssh_after, r3t_after, uub_after, vvb_after, un_adv, vn_adv)
      INTEGER, INTENT(in) :: kt
      REAL(wp), DIMENSION(:,:), INTENT(in) :: ssh_after, r3t_after
      REAL(wp), DIMENSION(:,:), INTENT(in) :: uub_after, vvb_after, un_adv, vn_adv
      IF(.NOT.selected(kt)) RETURN
      IF(current_kt /= kt .OR. phase /= 1) CALL ctl_stop('round141: barotropic phase out of order')
      CALL write_2d('ssh_after', ssh_after)
      CALL write_2d('r3t_after', r3t_after)
      CALL write_2d('uub_after', uub_after)
      CALL write_2d('vvb_after', vvb_after)
      CALL write_2d('un_adv', un_adv)
      CALL write_2d('vn_adv', vn_adv)
      phase = 2
   END SUBROUTINE r141_growth_baro

   SUBROUTINE r141_growth_stage1(kt, kstg, r3t_stage1, zfu, zfv, zfw)
      INTEGER, INTENT(in) :: kt, kstg
      REAL(wp), DIMENSION(:,:), INTENT(in) :: r3t_stage1
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: zfu, zfv, zfw
      INTEGER :: ios
      IF(.NOT.selected(kt) .OR. kstg /= 1) RETURN
      IF(current_kt /= kt .OR. phase /= 2) CALL ctl_stop('round141: stage-1 phase out of order')
      CALL write_2d('r3t_stage1', r3t_stage1)
      CALL write_3d('zFu_stage1', zfu)
      CALL write_3d('zFv_stage1', zfv)
      CALL write_3d('zFw_stage1', zfw)
      CLOSE(record_unit, IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round141: cannot close growth record')
      WRITE(numout,*) 'ORCA2_R141_GROWTH_DUMP ', kt, mpprank
      record_unit = -1
      current_kt = -1
      phase = 0
   END SUBROUTINE r141_growth_stage1

   SUBROUTINE write_2d(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(record_unit) field
      WRITE(record_unit) mpprank, 2, SIZE(value,1), SIZE(value,2), 1
      WRITE(record_unit) value
   END SUBROUTINE write_2d

   SUBROUTINE write_3d(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(record_unit) field
      WRITE(record_unit) mpprank, 3, SIZE(value,1), SIZE(value,2), SIZE(value,3)
      WRITE(record_unit) value
   END SUBROUTINE write_3d
END MODULE l4_r141_growth
