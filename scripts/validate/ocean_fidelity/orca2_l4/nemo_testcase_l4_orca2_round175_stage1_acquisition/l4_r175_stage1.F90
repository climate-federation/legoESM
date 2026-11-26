MODULE l4_r175_stage1_writer
   !! Write-only, rank-complete kt=1 stage-1 boundary recorder.
   USE par_kind, ONLY : wp
   USE par_oce,  ONLY : jpi, jpj, jpk
   USE dom_oce,  ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE lib_mpp,  ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r175_start, r175_put2, r175_put3, r175_finish

   INTEGER, PARAMETER :: target_kt = 1, target_stage = 1, expected_fields = 23
   INTEGER, SAVE :: record_unit = -1, field_count = 0

CONTAINS

   LOGICAL FUNCTION selected(kt, stage)
      INTEGER, INTENT(in) :: kt, stage
      selected = kt == target_kt .AND. stage == target_stage
   END FUNCTION selected

   SUBROUTINE r175_start(kt, stage, Kbb, Kmm, Krhs, Kaa)
      INTEGER, INTENT(in) :: kt, stage, Kbb, Kmm, Krhs, Kaa
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=112) :: filename
      IF(.NOT.selected(kt, stage)) RETURN
      IF(record_unit /= -1) CALL ctl_stop('round175: stage-1 record already open')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round175: stage-1 record requires fp64')
      WRITE(filename,'("oracle_r175_stage1_rank",I4.4,"_kt",I8.8,".bin")') mpprank, kt
      OPEN(NEWUNIT=record_unit, FILE=TRIM(filename), ACCESS='STREAM', &
         & FORM='UNFORMATTED', STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round175: cannot open rank stage-1 record')
      magic = 'NEMO_L4_R175S1'
      WRITE(record_unit) magic
      WRITE(record_unit) 1, kt, stage, Kbb, Kmm, Krhs, Kaa, mpprank, &
         & jpi, jpj, jpk, nimpp, njmpp, ntsi, ntsj, ntei, ntej, &
         & STORAGE_SIZE(1._wp), expected_fields
      field_count = 0
   END SUBROUTINE r175_start

   SUBROUTINE r175_put2(kt, stage, name, value)
      INTEGER, INTENT(in) :: kt, stage
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      IF(.NOT.selected(kt, stage)) RETURN
      IF(record_unit == -1) CALL ctl_stop('round175: stage-1 record is not open')
      field = name
      WRITE(record_unit) field
      WRITE(record_unit) 2, SIZE(value,1), SIZE(value,2), 1
      WRITE(record_unit) value
      field_count = field_count + 1
   END SUBROUTINE r175_put2

   SUBROUTINE r175_put3(kt, stage, name, value)
      INTEGER, INTENT(in) :: kt, stage
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      IF(.NOT.selected(kt, stage)) RETURN
      IF(record_unit == -1) CALL ctl_stop('round175: stage-1 record is not open')
      field = name
      WRITE(record_unit) field
      WRITE(record_unit) 3, SIZE(value,1), SIZE(value,2), SIZE(value,3)
      WRITE(record_unit) value
      field_count = field_count + 1
   END SUBROUTINE r175_put3

   SUBROUTINE r175_finish(kt, stage)
      INTEGER, INTENT(in) :: kt, stage
      IF(.NOT.selected(kt, stage)) RETURN
      IF(record_unit == -1) CALL ctl_stop('round175: stage-1 record is not open')
      IF(field_count /= expected_fields) CALL ctl_stop('round175: stage-1 field count moved')
      CLOSE(record_unit)
      record_unit = -1
      field_count = 0
   END SUBROUTINE r175_finish

END MODULE l4_r175_stage1_writer
