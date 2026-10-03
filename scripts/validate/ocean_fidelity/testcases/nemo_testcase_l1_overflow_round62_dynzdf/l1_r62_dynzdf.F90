MODULE l1_r62_dynzdf
   !! Round-62 WRITE-only kt=3 stage-3 dyn_zdf boundary recorder.
   !! Captured values are never read back into NEMO state.
   USE dom_oce
   USE in_out_manager, ONLY : lwp, nit000
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r62_zdf_begin, r62_zdf_row, r62_zdf_u_solve, &
      & r62_zdf_v_solve, r62_zdf_finish

   CHARACTER(LEN=16), PARAMETER :: r62_magic = 'NEMO_L1_R62ZDF01'
   INTEGER, PARAMETER :: r62_field_count = 8
   INTEGER, SAVE :: r62_unit = -1
   INTEGER, SAVE :: r62_kt = -1, r62_kbb = -1, r62_kmm = -1
   INTEGER, SAVE :: r62_krhs = -1, r62_kaa = -1
   INTEGER, SAVE, DIMENSION(5) :: r62_counts = 0
   LOGICAL, SAVE :: r62_active = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r62_explicit_u
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r62_explicit_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r62_subtract_u
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r62_subtract_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r62_drag_u
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r62_drag_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r62_solve_u
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r62_solve_v

CONTAINS

   SUBROUTINE put3(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(r62_unit) field
      WRITE(r62_unit) 3, SIZE(value,1), SIZE(value,2), SIZE(value,3)
      WRITE(r62_unit) value
   END SUBROUTINE put3

   SUBROUTINE r62_zdf_begin(kt, Kbb, Kmm, Krhs, Kaa)
      INTEGER, INTENT(in) :: kt, Kbb, Kmm, Krhs, Kaa
      INTEGER :: nx, ny
      r62_active = lwp .AND. kt == nit000+2 .AND. Kbb == 1 .AND. &
         & Kmm == 2 .AND. Krhs == 3 .AND. Kaa == 3
      IF(.NOT.r62_active) RETURN
      IF(r62_unit /= -1) CALL ctl_stop('round62: nested dyn_zdf record')
      IF(STORAGE_SIZE(1._wp) /= 64) &
         & CALL ctl_stop('round62: writer requires fp64')
      r62_kt=kt ; r62_kbb=Kbb ; r62_kmm=Kmm ; r62_krhs=Krhs ; r62_kaa=Kaa
      nx=ntei-ntsi+1 ; ny=ntej-ntsj+1
      ALLOCATE(r62_explicit_u(nx,ny,jpkm1),r62_explicit_v(nx,ny,jpkm1), &
         & r62_subtract_u(nx,ny,jpkm1),r62_subtract_v(nx,ny,jpkm1), &
         & r62_drag_u(nx,ny,jpkm1),r62_drag_v(nx,ny,jpkm1), &
         & r62_solve_u(nx,ny,jpkm1),r62_solve_v(nx,ny,jpkm1))
      r62_explicit_u=0._wp ; r62_explicit_v=0._wp
      r62_subtract_u=0._wp ; r62_subtract_v=0._wp
      r62_drag_u=0._wp ; r62_drag_v=0._wp
      r62_solve_u=0._wp ; r62_solve_v=0._wp ; r62_counts=0
   END SUBROUTINE r62_zdf_begin

   SUBROUTINE r62_zdf_row(phase, jj, urow, vrow)
      INTEGER, INTENT(in) :: phase, jj
      REAL(wp), DIMENSION(:,:), INTENT(in) :: urow, vrow
      INTEGER :: jlocal
      IF(.NOT.r62_active) RETURN
      jlocal=jj-ntsj+1
      IF(jlocal < 1 .OR. jlocal > ntej-ntsj+1) &
         & CALL ctl_stop('round62: row outside owned domain')
      IF(SIZE(urow,1) /= ntei-ntsi+1 .OR. SIZE(urow,2) /= jpkm1 .OR. &
         & ANY(SHAPE(urow) /= SHAPE(vrow))) &
         & CALL ctl_stop('round62: wrong dyn_zdf row shape')
      SELECT CASE(phase)
      CASE(1)
         r62_explicit_u(:,jlocal,:)=urow ; r62_explicit_v(:,jlocal,:)=vrow
      CASE(2)
         r62_subtract_u(:,jlocal,:)=urow ; r62_subtract_v(:,jlocal,:)=vrow
      CASE(3)
         r62_drag_u(:,jlocal,:)=urow ; r62_drag_v(:,jlocal,:)=vrow
      CASE DEFAULT
         CALL ctl_stop('round62: invalid dyn_zdf phase')
      END SELECT
      r62_counts(phase)=r62_counts(phase)+1
   END SUBROUTINE r62_zdf_row

   SUBROUTINE r62_zdf_u_solve(jj, urow)
      INTEGER, INTENT(in) :: jj
      REAL(wp), DIMENSION(:,:), INTENT(in) :: urow
      IF(.NOT.r62_active) RETURN
      r62_solve_u(:,jj-ntsj+1,:)=urow ; r62_counts(4)=r62_counts(4)+1
   END SUBROUTINE r62_zdf_u_solve

   SUBROUTINE r62_zdf_v_solve(jj, vrow)
      INTEGER, INTENT(in) :: jj
      REAL(wp), DIMENSION(:,:), INTENT(in) :: vrow
      IF(.NOT.r62_active) RETURN
      r62_solve_v(:,jj-ntsj+1,:)=vrow ; r62_counts(5)=r62_counts(5)+1
   END SUBROUTINE r62_zdf_v_solve

   SUBROUTINE r62_zdf_finish
      INTEGER :: ios, expected
      IF(.NOT.r62_active) RETURN
      expected=ntej-ntsj+1
      IF(ANY(r62_counts /= expected)) &
         & CALL ctl_stop('round62: incomplete dyn_zdf record')
      OPEN(NEWUNIT=r62_unit,FILE='oracle_r62_dynzdf_kt00000003_s3.bin', &
         & ACCESS='STREAM',FORM='UNFORMATTED',STATUS='NEW', &
         & ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round62: cannot open dyn_zdf record')
      WRITE(r62_unit) r62_magic
      WRITE(r62_unit) 1,r62_kt,3,r62_kbb,r62_kmm,r62_krhs,r62_kaa, &
         & ntei-ntsi+1,ntej-ntsj+1,jpkm1,ntsi,ntsj, &
         & STORAGE_SIZE(1._wp),r62_field_count
      CALL put3('explicit_u',r62_explicit_u)
      CALL put3('explicit_v',r62_explicit_v)
      CALL put3('baro_subtract_u',r62_subtract_u)
      CALL put3('baro_subtract_v',r62_subtract_v)
      CALL put3('baro_drag_u',r62_drag_u)
      CALL put3('baro_drag_v',r62_drag_v)
      CALL put3('implicit_solve_u',r62_solve_u)
      CALL put3('implicit_solve_v',r62_solve_v)
      CLOSE(r62_unit) ; r62_unit=-1 ; r62_active=.FALSE.
      DEALLOCATE(r62_explicit_u,r62_explicit_v,r62_subtract_u, &
         & r62_subtract_v,r62_drag_u,r62_drag_v,r62_solve_u,r62_solve_v)
   END SUBROUTINE r62_zdf_finish

END MODULE l1_r62_dynzdf
