MODULE l4_r118_een_v_recurrence
   !! Write-only, rank-complete recorder for all four V EEN recurrences.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE in_out_manager, ONLY : nit000, numout
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r118_een_v_init, r118_een_v_before, r118_een_v_after, r118_een_v_dump

   INTEGER, SAVE :: record_unit = -1
   LOGICAL, SAVE :: initialized = .FALSE.
   LOGICAL, SAVE :: dumped = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: zpvo(:,:,:,:), e3v_live(:,:,:,:), e3u_live(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: neighbor_mask(:,:,:,:), term(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: acc_before(:,:,:,:), acc_after(:,:,:,:)

CONTAINS
   SUBROUTINE r118_een_v_init
      INTEGER :: ios, length, status
      CHARACTER(LEN=512) :: output_dir, filename
      IF(initialized) CALL ctl_stop('round118: EEN V recorder initialized twice')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round118: EEN V record requires fp64')
      output_dir = ''
      CALL GET_ENVIRONMENT_VARIABLE('ORCA2_R118_EEN_V_DIR', output_dir, &
         & LENGTH=length, STATUS=status)
      IF(status /= 0 .OR. length < 1 .OR. length > LEN(output_dir)) &
         & CALL ctl_stop('round118: missing EEN V output directory')
      IF(output_dir(1:1) /= '/') CALL ctl_stop('round118: EEN V output directory is not absolute')
      WRITE(filename,'(A,"/oracle_r118_een_v_rank",I4.4,"_kt",I8.8,".bin")') &
         & TRIM(output_dir), mpprank, nit000
      OPEN(NEWUNIT=record_unit, FILE=TRIM(filename), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round118: cannot initialize EEN V record')
      ALLOCATE(zpvo(jpi,jpj,jpk,4), e3v_live(jpi,jpj,jpk,4), &
         & e3u_live(jpi,jpj,jpk,4), neighbor_mask(jpi,jpj,jpk,4), &
         & term(jpi,jpj,jpk,4), acc_before(jpi,jpj,jpk,4), &
         & acc_after(jpi,jpj,jpk,4), STAT=ios)
      IF(ios /= 0) CALL ctl_stop('round118: cannot allocate EEN V record')
      zpvo = 0._wp ; e3v_live = 0._wp ; e3u_live = 0._wp
      neighbor_mask = 0._wp ; term = 0._wp
      acc_before = 0._wp ; acc_after = 0._wp
      initialized = .TRUE.
      WRITE(numout,*) 'ORCA2_R118_EEN_V_INIT ', nit000, mpprank, TRIM(filename)
   END SUBROUTINE r118_een_v_init

   SUBROUTINE r118_een_v_before(slot, ji, jj, jk, z, e3v, e3u, mask, accumulator)
      INTEGER, INTENT(in) :: slot, ji, jj, jk
      REAL(wp), INTENT(in) :: z, e3v, e3u, mask, accumulator
      IF(.NOT.initialized) CALL ctl_stop('round118: EEN V store before initialization')
      IF(slot < 1 .OR. slot > 4) CALL ctl_stop('round118: invalid EEN V recurrence slot')
      zpvo(ji,jj,jk,slot) = z
      e3v_live(ji,jj,jk,slot) = e3v
      e3u_live(ji,jj,jk,slot) = e3u
      neighbor_mask(ji,jj,jk,slot) = mask
      term(ji,jj,jk,slot) = e3v * e3u * mask * z
      acc_before(ji,jj,jk,slot) = accumulator
   END SUBROUTINE r118_een_v_before

   SUBROUTINE r118_een_v_after(slot, ji, jj, jk, accumulator)
      INTEGER, INTENT(in) :: slot, ji, jj, jk
      REAL(wp), INTENT(in) :: accumulator
      IF(.NOT.initialized) CALL ctl_stop('round118: EEN V after-call before initialization')
      IF(slot < 1 .OR. slot > 4) CALL ctl_stop('round118: invalid EEN V recurrence slot')
      acc_after(ji,jj,jk,slot) = accumulator
   END SUBROUTINE r118_een_v_after

   SUBROUTINE r118_een_v_dump(bottom_index)
      INTEGER, DIMENSION(:,:), INTENT(in) :: bottom_index
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      IF(.NOT.initialized) CALL ctl_stop('round118: EEN V dump before initialization')
      IF(dumped) RETURN
      magic = 'NEMO_L4_R118EV1'
      WRITE(record_unit) magic
      WRITE(record_unit) 1, nit000, mpprank, jpi, jpj, jpk, nimpp, njmpp, &
         & ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 29
      CALL write_recurrence(record_unit, 'nw', 1)
      CALL write_recurrence(record_unit, 'ne', 2)
      CALL write_recurrence(record_unit, 'sw', 3)
      CALL write_recurrence(record_unit, 'se', 4)
      CALL write_2d_int_as_real(record_unit, 'mbkv', bottom_index)
      CLOSE(record_unit, IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round118: cannot close EEN V record')
      dumped = .TRUE.
      WRITE(numout,*) 'ORCA2_R118_EEN_V_DUMP ', nit000, mpprank
   END SUBROUTINE r118_een_v_dump

   SUBROUTINE write_recurrence(unit, suffix, slot)
      INTEGER, INTENT(in) :: unit, slot
      CHARACTER(LEN=*), INTENT(in) :: suffix
      CALL write_3d(unit, 'zpvo_'//suffix, zpvo(:,:,:,slot))
      CALL write_3d(unit, 'e3v_'//suffix, e3v_live(:,:,:,slot))
      CALL write_3d(unit, 'e3u_'//suffix, e3u_live(:,:,:,slot))
      CALL write_3d(unit, 'mask_'//suffix, neighbor_mask(:,:,:,slot))
      CALL write_3d(unit, 'term_'//suffix, term(:,:,:,slot))
      CALL write_3d(unit, 'before_'//suffix, acc_before(:,:,:,slot))
      CALL write_3d(unit, 'after_'//suffix, acc_after(:,:,:,slot))
   END SUBROUTINE write_recurrence

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
END MODULE l4_r118_een_v_recurrence
