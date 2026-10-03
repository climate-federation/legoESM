MODULE l4_r120_een_v_fraction
   !! Write-only, rank-complete recorder for northern V zpvo fractions.
   USE par_kind,       ONLY : wp
   USE par_oce,        ONLY : jpi, jpj, jpk
   USE dom_oce,        ONLY : nimpp, njmpp, ntsi, ntsj, ntei, ntej
   USE in_out_manager, ONLY : nit000, numout
   USE lib_mpp,        ONLY : mpprank, ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r120_een_v_fraction_init, r120_een_v_fraction_store, &
      & r120_een_v_fraction_dump

   INTEGER, SAVE :: record_unit = -1
   LOGICAL, SAVE :: initialized = .FALSE.
   LOGICAL, SAVE :: dumped = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: ff(:,:,:,:,:), e3f0(:,:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r3(:,:,:,:,:), mask(:,:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: denom(:,:,:,:,:), frac(:,:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: partial(:,:,:,:), sum_all(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: nemo_sum(:,:,:,:)

CONTAINS
   SUBROUTINE r120_een_v_fraction_init
      INTEGER :: ios, length, status
      CHARACTER(LEN=512) :: output_dir, filename
      IF(initialized) CALL ctl_stop('round120: EEN V fraction recorder initialized twice')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round120: EEN V fraction record requires fp64')
      output_dir = ''
      CALL GET_ENVIRONMENT_VARIABLE('ORCA2_R120_EEN_V_FRACTION_DIR', output_dir, &
         & LENGTH=length, STATUS=status)
      IF(status /= 0 .OR. length < 1 .OR. length > LEN(output_dir)) &
         & CALL ctl_stop('round120: missing EEN V fraction output directory')
      IF(output_dir(1:1) /= '/') CALL ctl_stop('round120: EEN V fraction output directory is not absolute')
      WRITE(filename,'(A,"/oracle_r120_een_v_fraction_rank",I4.4,"_kt",I8.8,".bin")') &
         & TRIM(output_dir), mpprank, nit000
      OPEN(NEWUNIT=record_unit, FILE=TRIM(filename), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='NEW', ACTION='WRITE', IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round120: cannot initialize EEN V fraction record')
      ALLOCATE(ff(2,3,jpi,jpj,jpk), e3f0(2,3,jpi,jpj,jpk), &
         & r3(2,3,jpi,jpj,jpk), mask(2,3,jpi,jpj,jpk), &
         & denom(2,3,jpi,jpj,jpk), frac(2,3,jpi,jpj,jpk), &
         & partial(2,jpi,jpj,jpk), sum_all(2,jpi,jpj,jpk), &
         & nemo_sum(2,jpi,jpj,jpk), STAT=ios)
      IF(ios /= 0) CALL ctl_stop('round120: cannot allocate EEN V fraction record')
      ff = 0._wp ; e3f0 = 0._wp ; r3 = 0._wp ; mask = 0._wp
      denom = 0._wp ; frac = 0._wp ; partial = 0._wp
      sum_all = 0._wp ; nemo_sum = 0._wp
      initialized = .TRUE.
      WRITE(numout,*) 'ORCA2_R120_EEN_V_FRACTION_INIT ', nit000, mpprank, TRIM(filename)
   END SUBROUTINE r120_een_v_fraction_init

   SUBROUTINE r120_een_v_fraction_store(ji, jj, jk, &
      & ne1f, ne1e, ne1r, ne1m, ne2f, ne2e, ne2r, ne2m, &
      & ne3f, ne3e, ne3r, ne3m, nesum, &
      & nw1f, nw1e, nw1r, nw1m, nw2f, nw2e, nw2r, nw2m, &
      & nw3f, nw3e, nw3r, nw3m, nwsum)
      INTEGER, INTENT(in) :: ji, jj, jk
      REAL(wp), INTENT(in) :: ne1f, ne1e, ne1r, ne1m
      REAL(wp), INTENT(in) :: ne2f, ne2e, ne2r, ne2m
      REAL(wp), INTENT(in) :: ne3f, ne3e, ne3r, ne3m, nesum
      REAL(wp), INTENT(in) :: nw1f, nw1e, nw1r, nw1m
      REAL(wp), INTENT(in) :: nw2f, nw2e, nw2r, nw2m
      REAL(wp), INTENT(in) :: nw3f, nw3e, nw3r, nw3m, nwsum
      IF(.NOT.initialized) CALL ctl_stop('round120: EEN V fraction store before initialization')
      ff(1,:,ji,jj,jk) = (/ ne1f, ne2f, ne3f /)
      e3f0(1,:,ji,jj,jk) = (/ ne1e, ne2e, ne3e /)
      r3(1,:,ji,jj,jk) = (/ ne1r, ne2r, ne3r /)
      mask(1,:,ji,jj,jk) = (/ ne1m, ne2m, ne3m /)
      nemo_sum(1,ji,jj,jk) = nesum
      ff(2,:,ji,jj,jk) = (/ nw1f, nw2f, nw3f /)
      e3f0(2,:,ji,jj,jk) = (/ nw1e, nw2e, nw3e /)
      r3(2,:,ji,jj,jk) = (/ nw1r, nw2r, nw3r /)
      mask(2,:,ji,jj,jk) = (/ nw1m, nw2m, nw3m /)
      nemo_sum(2,ji,jj,jk) = nwsum
      denom(:,:,ji,jj,jk) = e3f0(:,:,ji,jj,jk) * &
         & (1._wp + r3(:,:,ji,jj,jk) * mask(:,:,ji,jj,jk))
      frac(:,:,ji,jj,jk) = ff(:,:,ji,jj,jk) / denom(:,:,ji,jj,jk)
      partial(:,ji,jj,jk) = frac(:,1,ji,jj,jk) + frac(:,2,ji,jj,jk)
      sum_all(:,ji,jj,jk) = partial(:,ji,jj,jk) + frac(:,3,ji,jj,jk)
   END SUBROUTINE r120_een_v_fraction_store

   SUBROUTINE r120_een_v_fraction_dump(bottom_index)
      INTEGER, DIMENSION(:,:), INTENT(in) :: bottom_index
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      IF(.NOT.initialized) CALL ctl_stop('round120: EEN V fraction dump before initialization')
      IF(dumped) RETURN
      magic = 'NEMO_L4_R120VF1'
      WRITE(record_unit) magic
      WRITE(record_unit) 1, nit000, mpprank, jpi, jpj, jpk, nimpp, njmpp, &
         & ntsi, ntsj, ntei, ntej, STORAGE_SIZE(1._wp), 43
      CALL write_path(record_unit, 'ne', 1)
      CALL write_path(record_unit, 'nw', 2)
      CALL write_2d_int_as_real(record_unit, 'mbkv', bottom_index)
      CLOSE(record_unit, IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round120: cannot close EEN V fraction record')
      dumped = .TRUE.
      WRITE(numout,*) 'ORCA2_R120_EEN_V_FRACTION_DUMP ', nit000, mpprank
   END SUBROUTINE r120_een_v_fraction_dump

   SUBROUTINE write_path(unit, prefix, path_index)
      INTEGER, INTENT(in) :: unit, path_index
      CHARACTER(LEN=*), INTENT(in) :: prefix
      INTEGER :: component
      CHARACTER(LEN=16) :: field
      CHARACTER(LEN=1), PARAMETER :: labels(3) = (/ '1', '2', '3' /)
      DO component = 1, 3
         field = prefix//'_'//labels(component)//'_ff'
         CALL write_3d(unit, field, ff(path_index,component,:,:,:))
         field = prefix//'_'//labels(component)//'_e3f0'
         CALL write_3d(unit, field, e3f0(path_index,component,:,:,:))
         field = prefix//'_'//labels(component)//'_r3f'
         CALL write_3d(unit, field, r3(path_index,component,:,:,:))
         field = prefix//'_'//labels(component)//'_mask'
         CALL write_3d(unit, field, mask(path_index,component,:,:,:))
         field = prefix//'_'//labels(component)//'_denom'
         CALL write_3d(unit, field, denom(path_index,component,:,:,:))
         field = prefix//'_'//labels(component)//'_frac'
         CALL write_3d(unit, field, frac(path_index,component,:,:,:))
      END DO
      CALL write_3d(unit, prefix//'_partial', partial(path_index,:,:,:))
      CALL write_3d(unit, prefix//'_sum', sum_all(path_index,:,:,:))
      CALL write_3d(unit, prefix//'_nemo_sum', nemo_sum(path_index,:,:,:))
   END SUBROUTINE write_path

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
END MODULE l4_r120_een_v_fraction
