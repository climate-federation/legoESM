MODULE l4_r107_een_step
   !! Rank-complete, self-describing, write-only ffu_nw recurrence operands.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE in_out_manager, ONLY : nit000, numout
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r107_een_step_init, r107_een_step_dump

   INTEGER, SAVE :: record_unit = -1
   LOGICAL, SAVE :: initialized = .FALSE.
   LOGICAL, SAVE :: dumped = .FALSE.

CONTAINS
   SUBROUTINE r107_een_step_init
      INTEGER :: ios, length, status
      CHARACTER(LEN=512) :: output_dir, filename
      IF(initialized) CALL ctl_stop('round107: EEN step recorder initialized twice')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round107: EEN step record requires fp64')
      output_dir = ''
      CALL GET_ENVIRONMENT_VARIABLE('ORCA2_R107_EEN_STEP_DIR', output_dir, &
         & LENGTH=length, STATUS=status)
      IF(status /= 0 .OR. length < 1 .OR. length > LEN(output_dir)) &
         & CALL ctl_stop('round107: missing EEN step output directory')
      IF(output_dir(1:1) /= '/') CALL ctl_stop('round107: EEN step output directory is not absolute')
      WRITE(filename,'(A,"/oracle_r107_een_step_rank",I4.4,"_kt",I8.8,".bin")') &
         & TRIM(output_dir), mpprank, nit000
      OPEN(NEWUNIT=record_unit, FILE=TRIM(filename), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round107: cannot initialize EEN step record')
      initialized = .TRUE.
      WRITE(numout,*) 'ORCA2_R107_EEN_STEP_INIT ', nit000, mpprank, TRIM(filename)
   END SUBROUTINE r107_een_step_init

   SUBROUTINE r107_een_step_dump(zpvo, e3u_live, e3v_live, neighbor_mask, &
      & term, acc_before, acc_after, bottom_index)
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: zpvo, e3u_live, e3v_live
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: neighbor_mask, term
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: acc_before, acc_after
      INTEGER, DIMENSION(:,:), INTENT(in) :: bottom_index
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      IF(.NOT.initialized) CALL ctl_stop('round107: EEN step dump before initialization')
      IF(dumped) RETURN
      magic = 'NEMO_L4_R107ES1'
      WRITE(record_unit) magic
      WRITE(record_unit) 1, nit000, mpprank, jpi, jpj, jpk, nimpp, njmpp, &
         & ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 8
      CALL write_3d(record_unit, 'zpvo_nw', zpvo)
      CALL write_3d(record_unit, 'e3u_live', e3u_live)
      CALL write_3d(record_unit, 'e3v_live', e3v_live)
      CALL write_3d(record_unit, 'neighbor_mask', neighbor_mask)
      CALL write_3d(record_unit, 'term_nw', term)
      CALL write_3d(record_unit, 'acc_before', acc_before)
      CALL write_3d(record_unit, 'acc_after', acc_after)
      CALL write_2d_int_as_real(record_unit, 'mbku', bottom_index)
      CLOSE(record_unit, IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round107: cannot close EEN step record')
      dumped = .TRUE.
      WRITE(numout,*) 'ORCA2_R107_EEN_STEP_DUMP ', nit000, mpprank
   END SUBROUTINE r107_een_step_dump

   SUBROUTINE write_3d(unit, name, value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(unit) field
      WRITE(unit) 3, ntei-ntsi+1, ntej-ntsj+1, jpk
      WRITE(unit) value(ntsi:ntei,ntsj:ntej,:)
   END SUBROUTINE write_3d

   SUBROUTINE write_2d_int_as_real(unit, name, value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      INTEGER, DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(unit) field
      WRITE(unit) 2, ntei-ntsi+1, ntej-ntsj+1, 1
      WRITE(unit) REAL(value(ntsi:ntei,ntsj:ntej), wp)
   END SUBROUTINE write_2d_int_as_real
END MODULE l4_r107_een_step
