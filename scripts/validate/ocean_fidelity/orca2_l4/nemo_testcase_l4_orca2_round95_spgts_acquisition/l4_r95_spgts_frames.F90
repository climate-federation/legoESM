MODULE l4_r95_spgts_frames
   !! Rank-complete, write-only rung-0 split-explicit substep recorder.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE in_out_manager, ONLY : nit000, numout
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r95_spg_open, r95_spg_substep, r95_spg_w2, r95_spg_w1, r95_spg_close

   INTEGER, SAVE :: r95_unit = -1
   INTEGER, SAVE :: r95_substep = 0

CONTAINS
   SUBROUTINE r95_spg_open(kt, Kbb, Kmm, Kaa, Krhs, Kcycle)
      INTEGER, INTENT(in) :: kt, Kbb, Kmm, Kaa, Krhs, Kcycle
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=112) :: filename
      IF(kt /= nit000) RETURN
      IF(r95_unit /= -1) CALL ctl_stop('round95: spg record already open')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round95: spg record requires fp64')
      WRITE(filename,'("oracle_r95_spg_rank",I4.4,"_kt",I8.8,".bin")') mpprank, kt
      OPEN(NEWUNIT=r95_unit, FILE=TRIM(filename), ACCESS='STREAM', &
         & FORM='UNFORMATTED', STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round95: cannot open spg record')
      magic = 'NEMO_L4_R95SPG1'
      WRITE(r95_unit) magic
      WRITE(r95_unit) 1, kt, Kbb, Kmm, Kaa, Krhs, mpprank, jpi, jpj, jpk, &
         & Kcycle, nimpp, njmpp, ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp)
      r95_substep = 0
   END SUBROUTINE r95_spg_open

   SUBROUTINE r95_spg_substep(kn)
      INTEGER, INTENT(in) :: kn
      IF(r95_unit == -1) RETURN
      r95_substep = kn
   END SUBROUTINE r95_spg_substep

   SUBROUTINE r95_spg_w2(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      IF(r95_unit == -1) RETURN
      field = frame_name(name)
      WRITE(r95_unit) field
      WRITE(r95_unit) 2, SIZE(value,1), SIZE(value,2), 1
      WRITE(r95_unit) value
   END SUBROUTINE r95_spg_w2

   SUBROUTINE r95_spg_w1(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      IF(r95_unit == -1) RETURN
      field = frame_name(name)
      WRITE(r95_unit) field
      WRITE(r95_unit) 1, SIZE(value,1), 1, 1
      WRITE(r95_unit) value
   END SUBROUTINE r95_spg_w1

   SUBROUTINE r95_spg_close(kt, Kcycle)
      INTEGER, INTENT(in) :: kt, Kcycle
      IF(r95_unit == -1) RETURN
      CLOSE(r95_unit)
      WRITE(numout,*) 'ORCA2_R95_SPGTS_DUMP ', kt, Kcycle, mpprank
      r95_unit = -1
      r95_substep = 0
   END SUBROUTINE r95_spg_close

   CHARACTER(LEN=16) FUNCTION frame_name(name)
      CHARACTER(LEN=*), INTENT(in) :: name
      IF(r95_substep == 0) THEN
         WRITE(frame_name,'("i000_",A)') TRIM(name)
      ELSE IF(r95_substep < 0) THEN
         WRITE(frame_name,'("o000_",A)') TRIM(name)
      ELSE
         WRITE(frame_name,'("j",I3.3,"_",A)') r95_substep, TRIM(name)
      ENDIF
   END FUNCTION frame_name
END MODULE l4_r95_spgts_frames
