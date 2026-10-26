MODULE l4_r110_een_fraction
   !! Write-only, rank-complete recorder for the three zpvo_nw fractions.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE in_out_manager, ONLY : nit000, numout
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r110_een_fraction_init, r110_een_fraction_store, r110_een_fraction_dump

   INTEGER, SAVE :: record_unit = -1
   LOGICAL, SAVE :: initialized = .FALSE.
   LOGICAL, SAVE :: dumped = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: west_ff(:,:,:), west_e3f0(:,:,:), west_r3f(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: west_mask(:,:,:), west_denom(:,:,:), frac_west(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: center_ff(:,:,:), center_e3f0(:,:,:), center_r3f(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: center_mask(:,:,:), center_denom(:,:,:), frac_center(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: south_ff(:,:,:), south_e3f0(:,:,:), south_r3f(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: south_mask(:,:,:), south_denom(:,:,:), frac_south(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: sum_west_center(:,:,:), sum_all(:,:,:)

CONTAINS
   SUBROUTINE r110_een_fraction_init
      INTEGER :: ios, length, status
      CHARACTER(LEN=512) :: output_dir, filename
      IF(initialized) CALL ctl_stop('round110: EEN fraction recorder initialized twice')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round110: EEN fraction record requires fp64')
      output_dir = ''
      CALL GET_ENVIRONMENT_VARIABLE('ORCA2_R110_EEN_FRACTION_DIR', output_dir, &
         & LENGTH=length, STATUS=status)
      IF(status /= 0 .OR. length < 1 .OR. length > LEN(output_dir)) &
         & CALL ctl_stop('round110: missing EEN fraction output directory')
      IF(output_dir(1:1) /= '/') CALL ctl_stop('round110: EEN fraction output directory is not absolute')
      WRITE(filename,'(A,"/oracle_r110_een_fraction_rank",I4.4,"_kt",I8.8,".bin")') &
         & TRIM(output_dir), mpprank, nit000
      OPEN(NEWUNIT=record_unit, FILE=TRIM(filename), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round110: cannot initialize EEN fraction record')
      ALLOCATE(west_ff(jpi,jpj,jpk), west_e3f0(jpi,jpj,jpk), west_r3f(jpi,jpj,jpk), &
         & west_mask(jpi,jpj,jpk), west_denom(jpi,jpj,jpk), frac_west(jpi,jpj,jpk), &
         & center_ff(jpi,jpj,jpk), center_e3f0(jpi,jpj,jpk), center_r3f(jpi,jpj,jpk), &
         & center_mask(jpi,jpj,jpk), center_denom(jpi,jpj,jpk), frac_center(jpi,jpj,jpk), &
         & south_ff(jpi,jpj,jpk), south_e3f0(jpi,jpj,jpk), south_r3f(jpi,jpj,jpk), &
         & south_mask(jpi,jpj,jpk), south_denom(jpi,jpj,jpk), frac_south(jpi,jpj,jpk), &
         & sum_west_center(jpi,jpj,jpk), sum_all(jpi,jpj,jpk), STAT=ios)
      IF(ios /= 0) CALL ctl_stop('round110: cannot allocate EEN fraction record')
      west_ff = 0._wp ; west_e3f0 = 0._wp ; west_r3f = 0._wp
      west_mask = 0._wp ; west_denom = 0._wp ; frac_west = 0._wp
      center_ff = 0._wp ; center_e3f0 = 0._wp ; center_r3f = 0._wp
      center_mask = 0._wp ; center_denom = 0._wp ; frac_center = 0._wp
      south_ff = 0._wp ; south_e3f0 = 0._wp ; south_r3f = 0._wp
      south_mask = 0._wp ; south_denom = 0._wp ; frac_south = 0._wp
      sum_west_center = 0._wp ; sum_all = 0._wp
      initialized = .TRUE.
      WRITE(numout,*) 'ORCA2_R110_EEN_FRACTION_INIT ', nit000, mpprank, TRIM(filename)
   END SUBROUTINE r110_een_fraction_init

   SUBROUTINE r110_een_fraction_store(ji, jj, jk, wff, we3f0, wr3f, wmask, &
      & cff, ce3f0, cr3f, cmask, sff, se3f0, sr3f, smask)
      INTEGER, INTENT(in) :: ji, jj, jk
      REAL(wp), INTENT(in) :: wff, we3f0, wr3f, wmask
      REAL(wp), INTENT(in) :: cff, ce3f0, cr3f, cmask
      REAL(wp), INTENT(in) :: sff, se3f0, sr3f, smask
      IF(.NOT.initialized) CALL ctl_stop('round110: EEN fraction store before initialization')
      west_ff(ji,jj,jk) = wff
      west_e3f0(ji,jj,jk) = we3f0
      west_r3f(ji,jj,jk) = wr3f
      west_mask(ji,jj,jk) = wmask
      west_denom(ji,jj,jk) = we3f0 * (1._wp + wr3f * wmask)
      frac_west(ji,jj,jk) = wff / west_denom(ji,jj,jk)
      center_ff(ji,jj,jk) = cff
      center_e3f0(ji,jj,jk) = ce3f0
      center_r3f(ji,jj,jk) = cr3f
      center_mask(ji,jj,jk) = cmask
      center_denom(ji,jj,jk) = ce3f0 * (1._wp + cr3f * cmask)
      frac_center(ji,jj,jk) = cff / center_denom(ji,jj,jk)
      south_ff(ji,jj,jk) = sff
      south_e3f0(ji,jj,jk) = se3f0
      south_r3f(ji,jj,jk) = sr3f
      south_mask(ji,jj,jk) = smask
      south_denom(ji,jj,jk) = se3f0 * (1._wp + sr3f * smask)
      frac_south(ji,jj,jk) = sff / south_denom(ji,jj,jk)
      sum_west_center(ji,jj,jk) = frac_west(ji,jj,jk) + frac_center(ji,jj,jk)
      sum_all(ji,jj,jk) = sum_west_center(ji,jj,jk) + frac_south(ji,jj,jk)
   END SUBROUTINE r110_een_fraction_store

   SUBROUTINE r110_een_fraction_dump(bottom_index)
      INTEGER, DIMENSION(:,:), INTENT(in) :: bottom_index
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      IF(.NOT.initialized) CALL ctl_stop('round110: EEN fraction dump before initialization')
      IF(dumped) RETURN
      magic = 'NEMO_L4_R110EF1'
      WRITE(record_unit) magic
      WRITE(record_unit) 1, nit000, mpprank, jpi, jpj, jpk, nimpp, njmpp, &
         & ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 21
      CALL write_3d(record_unit, 'west_ff', west_ff)
      CALL write_3d(record_unit, 'west_e3f0', west_e3f0)
      CALL write_3d(record_unit, 'west_r3f', west_r3f)
      CALL write_3d(record_unit, 'west_mask', west_mask)
      CALL write_3d(record_unit, 'west_denom', west_denom)
      CALL write_3d(record_unit, 'frac_west', frac_west)
      CALL write_3d(record_unit, 'center_ff', center_ff)
      CALL write_3d(record_unit, 'center_e3f0', center_e3f0)
      CALL write_3d(record_unit, 'center_r3f', center_r3f)
      CALL write_3d(record_unit, 'center_mask', center_mask)
      CALL write_3d(record_unit, 'center_denom', center_denom)
      CALL write_3d(record_unit, 'frac_center', frac_center)
      CALL write_3d(record_unit, 'south_ff', south_ff)
      CALL write_3d(record_unit, 'south_e3f0', south_e3f0)
      CALL write_3d(record_unit, 'south_r3f', south_r3f)
      CALL write_3d(record_unit, 'south_mask', south_mask)
      CALL write_3d(record_unit, 'south_denom', south_denom)
      CALL write_3d(record_unit, 'frac_south', frac_south)
      CALL write_3d(record_unit, 'sum_west_center', sum_west_center)
      CALL write_3d(record_unit, 'sum_all', sum_all)
      CALL write_2d_int_as_real(record_unit, 'mbku', bottom_index)
      CLOSE(record_unit, IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round110: cannot close EEN fraction record')
      dumped = .TRUE.
      WRITE(numout,*) 'ORCA2_R110_EEN_FRACTION_DUMP ', nit000, mpprank
   END SUBROUTINE r110_een_fraction_dump

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
END MODULE l4_r110_een_fraction
