! ORACLE STAGE-STATE driver: one REAL fv_dynamics step (the C48/npz=5
! hydro parity deck, dt_atmos=1920) run twice from the same cold-start
! IC --
!
!   arm A: the REAL linked fv_dynamics_mod::fv_dynamics (no hooks);
!   arm B: the STAGED verbatim copies (gen_dyncore_stage_copy.py:
!          fv_dynamics_staged_mod -> dyn_core_staged_mod) with
!          per-stage dump hooks live on acoustic substep 1.
!
! CERTIFICATION: after both arms the driver records
! max|armA - armB| for every prognostic + auxiliary field as CERT lines
! in the manifest.  The staged copy is NOT trusted by construction --
! the consuming comparator requires every CERT to be exactly 0.0 before
! any stage dump is read.  A staged copy that drifts is a broken
! instrument, not a finding.
!
! EXTERNAL CONTROL: the driver runs the same deck as
! run_hydro_1step_gfs, so arm A's final state must equal that run's
! RESTART netCDF (checked by the comparator) -- proving this driver's
! init path reproduces the production solo driver.
!
! Init sequence replicates atmos_drivers/solo/atmos_model.F90 +
! driver/solo/atmosphere.F90 (adiabatic lane): main_nml read,
! NO_CALENDAR, diag_manager_init/get_base_date, fv_control_init,
! fv_restart (cold start -> test_case IC), fv_diag_init, zvir = 0.
!
! Build: scripts/cluster/fv3_native/dyncore_stage_oracle.sbatch
! (GFS-constants build: $PIN/build_hydro_gfsconst + fms build_gfs).
program fv3_dyncore_stage_driver
  use fms_mod,            only: fms_init, fms_end, error_mesg, FATAL, &
                                check_nml_error, stdlog, set_domain, &
                                nullify_domain, file_exist
  use fms_io_mod,         only: fms_io_exit
  use mpp_mod,            only: mpp_pe, mpp_root_pe, mpp_npes, mpp_error, &
                                mpp_sync, input_nml_file
  use mpp_domains_mod,    only: mpp_get_tile_id
  use time_manager_mod,   only: time_type, set_time, get_time, &
                                set_calendar_type, NO_CALENDAR, operator(+)
  use diag_manager_mod,   only: diag_manager_init, get_base_date
  use field_manager_mod,  only: MODEL_ATMOS
  use tracer_manager_mod, only: register_tracers
  use constants_mod,      only: kappa, cp_air, rvgas, rdgas, SECONDS_PER_DAY
  use fv_arrays_mod,      only: fv_atmos_type
  use fv_control_mod,     only: fv_control_init, ngrids
  use fv_restart_mod,     only: fv_restart
  use fv_diagnostics_mod, only: fv_diag_init, fv_time
  use fv_dynamics_mod,    only: fv_dynamics
  use fv_dynamics_staged_mod, only: fv_dynamics_staged => fv_dynamics
  use stage_dump_mod,     only: stage_dump_open, stage_dump_close, &
                                stage_dump2, stage_dump3, stage_mf_note
  implicit none

  type(fv_atmos_type), allocatable, target :: Atm(:)
  logical, allocatable :: grids_on_this_pe(:)
  integer :: this_grid, p_split, mytile, n
  type(time_type) :: Time, Time_init, Time_step_atmos
  integer :: date(6), date_init(6)
  integer :: seconds, days, sec_fv, days_fv
  integer :: axes(4)
  integer :: ierr, io
  integer :: tile_id(1), tile
  integer :: isd, ied, jsd, jed
  logical :: cold_start
  real :: zvir, time_total
  character(len=64) :: fname_dat, fname_mf
  character(len=256) :: mfline

  ! main_nml (mirrors atmos_drivers/solo/atmos_model.F90)
  character(len=17) :: calendar = 'no_calendar      '
  integer, dimension(4) :: current_time = (/0, 0, 0, 0/)
  integer :: years = 0, months = 0, hours = 0, minutes = 0
  integer :: dt_atmos = 0
  integer :: memuse_interval = 72
  integer :: atmos_nthreads = 1
  logical :: use_hyper_thread = .false.
  namelist /main_nml/ calendar, current_time, dt_atmos, years, months, &
      days, hours, minutes, seconds, memuse_interval, atmos_nthreads, &
      use_hyper_thread

  ! saved IC (allocatable assignment auto-shapes)
  real, allocatable, dimension(:, :, :) :: s_u, s_v, s_w, s_pt, s_delp, &
      s_pkz, s_omga, s_ua, s_va, s_uc, s_vc, s_mfx, s_mfy, s_cx, s_cy, &
      s_pe, s_peln, s_pk, s_qcon, s_delz
  real, allocatable :: s_q(:, :, :, :)
  real, allocatable :: s_ps(:, :), s_phis(:, :)
  ! arm A finals
  real, allocatable, dimension(:, :, :) :: a_u, a_v, a_pt, a_delp, &
      a_pkz, a_omga, a_ua, a_va, a_uc, a_vc, a_mfx, a_mfy, a_cx, a_cy, &
      a_pe, a_peln, a_pk, a_w, a_delz, a_qcon, a_ze0
  real, allocatable :: a_q(:, :, :, :)
  real, allocatable :: a_ps(:, :), a_phis(:, :)
  real, allocatable :: s_ze0(:, :, :)
  integer :: iq

  integer :: ntrace, ntprog, ntdiag, ntfamily

  call fms_init()

  ! how many tracers have been registered (field_table read)
  call register_tracers(MODEL_ATMOS, ntrace, ntprog, ntdiag, ntfamily)

  ! days/seconds locals shadow the nml names used by atmos_model; keep
  ! the nml read faithful by using the same names via a block read.
  days = 0; seconds = 0
  read (input_nml_file, nml=main_nml, iostat=io)
  ierr = check_nml_error(io, 'main_nml')
  if (dt_atmos == 0) call error_mesg('stage_driver', &
      'dt_atmos has not been specified', FATAL)
  if (trim(calendar) /= 'no_calendar') call error_mesg('stage_driver', &
      'this driver replicates the NO_CALENDAR deck only', FATAL)
  call set_calendar_type(NO_CALENDAR)

  if (file_exist('INPUT/atmos_model.res')) call error_mesg('stage_driver', &
      'INPUT/atmos_model.res exists -- this driver is cold-start only', &
      FATAL)
  date(1:2) = 0
  date(3:6) = current_time

  call diag_manager_init
  call get_base_date(date_init(1), date_init(2), date_init(3), &
                     date_init(4), date_init(5), date_init(6))
  Time_init = set_time(date_init(4)*3600 + date_init(5)*60 + date_init(6), &
                       date_init(3))
  Time = set_time(date(4)*3600 + date(5)*60 + date(6), date(3))
  Time_step_atmos = set_time(dt_atmos, 0)

  ! ----- atmosphere_init (adiabatic solo lane) -----
  cold_start = (.not. file_exist('INPUT/fv_core.res.nc') .and. &
                .not. file_exist('INPUT/fv_core.res.tile1.nc'))
  if (.not. cold_start) call error_mesg('stage_driver', &
      'warm start detected; the parity deck is cold-start', FATAL)

  p_split = 1
  call fv_control_init(Atm, real(dt_atmos), this_grid, grids_on_this_pe, &
                       p_split)

  if (mpp_npes() /= 6) call mpp_error(FATAL, &
      'stage driver requires exactly 6 PEs (one per tile)')
  mytile = 1
  do n = 1, ngrids
    if (grids_on_this_pe(n)) mytile = n
  end do
  n = mytile

  call fv_restart(Atm(1)%domain, Atm, real(dt_atmos), seconds, days, &
                  cold_start, Atm(1)%flagstruct%grid_type, mytile)

  fv_time = Time
  Atm(n)%flagstruct%moist_phys = .false.
  call fv_diag_init(Atm(n:n), axes, Time, Atm(n)%npx, Atm(n)%npy, &
                    Atm(n)%npz, Atm(n)%flagstruct%p_ref)

  if (.not. Atm(n)%flagstruct%adiabatic) call mpp_error(FATAL, &
      'stage driver replicates the ADIABATIC deck (zvir = 0) only')
  if (.not. Atm(n)%flagstruct%hydrostatic) call mpp_error(FATAL, &
      'stage driver replicates the HYDROSTATIC lane only')
  if (Atm(n)%flagstruct%k_split /= 1) call mpp_error(FATAL, &
      'stage driver assumes k_split = 1 (one dyn_core call per step)')
  if (.not. Atm(n)%flagstruct%duogrid) call mpp_error(FATAL, &
      'deck must set duogrid = .true.')
  if (.not. Atm(n)%gridstruct%dg%is_initialized) call mpp_error(FATAL, &
      'duogrid not initialized')
  zvir = 0.

  tile_id = mpp_get_tile_id(Atm(n)%domain)
  tile = tile_id(1)
  isd = Atm(n)%bd%isd; ied = Atm(n)%bd%ied
  jsd = Atm(n)%bd%jsd; jed = Atm(n)%bd%jed

  ! ----- open dumps; record coords + resolved flags + the IC -----
  write (fname_dat, '(A,I1,A)') 'dyncore_stage_t', tile, '.dat'
  write (fname_mf, '(A,I1,A)') 'dyncore_stage_t', tile, '.mf'
  call stage_dump_open(fname_dat, fname_mf)

  write (mfline, '(A,7(1X,I6))') 'NOTE bounds is ie js je isd ied ng =', &
      Atm(n)%bd%is, Atm(n)%bd%ie, Atm(n)%bd%js, Atm(n)%bd%je, &
      isd, ied, Atm(n)%bd%ng
  call stage_mf_note(mfline)
  write (mfline, '(A,6(1X,I6))') 'NOTE npx npz tile nsplit ksplit nord =', &
      Atm(n)%flagstruct%npx, Atm(n)%flagstruct%npz, tile, &
      Atm(n)%flagstruct%n_split, Atm(n)%flagstruct%k_split, &
      Atm(n)%flagstruct%nord
  call stage_mf_note(mfline)
  write (mfline, '(A,4(1X,ES24.16E3))') &
      'NOTE dt_atmos consv_te d4_bg vtdm4 =', real(dt_atmos), &
      Atm(n)%flagstruct%consv_te, Atm(n)%flagstruct%d4_bg, &
      Atm(n)%flagstruct%vtdm4
  call stage_mf_note(mfline)
  write (mfline, '(A,3(1X,ES24.16E3))') 'NOTE kappa cp_air ptop =', &
      kappa, cp_air, Atm(n)%ptop
  call stage_mf_note(mfline)

  call stage_dump2('AG_LON', &
      real(Atm(n)%gridstruct%agrid(isd:ied, jsd:jed, 1)))
  call stage_dump2('AG_LAT', &
      real(Atm(n)%gridstruct%agrid(isd:ied, jsd:jed, 2)))
  call stage_dump2('GR_LON', &
      real(Atm(n)%gridstruct%grid(isd:ied + 1, jsd:jed + 1, 1)))
  call stage_dump2('GR_LAT', &
      real(Atm(n)%gridstruct%grid(isd:ied + 1, jsd:jed + 1, 2)))

  ! the runtime Coriolis f0 (test_cases.F90:787-800: analytic +
  ! ext_scalar + fill_corners YDir) -- the vort = wk + f0 input whose
  ! corner-diagonal region d_sw5's fv_tp_2d consumes.
  call stage_dump2('IC_f0', Atm(n)%gridstruct%f0(isd:ied, jsd:jed))
  call stage_dump3('IC_u', Atm(n)%u)
  call stage_dump3('IC_v', Atm(n)%v)
  call stage_dump3('IC_pt', Atm(n)%pt)
  call stage_dump3('IC_delp', Atm(n)%delp)
  call stage_dump2('IC_ps', Atm(n)%ps)
  call stage_dump2('IC_phis', Atm(n)%phis)

  ! ----- save the IC (everything fv_dynamics may mutate) -----
  s_u = Atm(n)%u;      s_v = Atm(n)%v;       s_w = Atm(n)%w
  s_pt = Atm(n)%pt;    s_delp = Atm(n)%delp; s_q = Atm(n)%q
  s_ps = Atm(n)%ps;    s_pe = Atm(n)%pe;     s_pk = Atm(n)%pk
  s_peln = Atm(n)%peln; s_pkz = Atm(n)%pkz
  s_phis = Atm(n)%phis; s_omga = Atm(n)%omga
  s_ua = Atm(n)%ua;    s_va = Atm(n)%va
  s_uc = Atm(n)%uc;    s_vc = Atm(n)%vc
  s_mfx = Atm(n)%mfx;  s_mfy = Atm(n)%mfy
  s_cx = Atm(n)%cx;    s_cy = Atm(n)%cy
  s_qcon = Atm(n)%q_con
  s_delz = Atm(n)%delz
  if (allocated(Atm(n)%ze0)) s_ze0 = Atm(n)%ze0

  ! ----- time bookkeeping exactly as atmosphere(Time) -----
  fv_time = Time + Time_step_atmos
  call get_time(fv_time, sec_fv, days_fv)
  time_total = days_fv*SECONDS_PER_DAY + sec_fv
  call set_domain(Atm(n)%domain)

  ! =================== arm A: the REAL fv_dynamics ===================
  call run_arm(.false.)

  a_u = Atm(n)%u;      a_v = Atm(n)%v
  a_pt = Atm(n)%pt;    a_delp = Atm(n)%delp; a_q = Atm(n)%q
  a_ps = Atm(n)%ps;    a_pe = Atm(n)%pe;     a_pk = Atm(n)%pk
  a_peln = Atm(n)%peln; a_pkz = Atm(n)%pkz
  a_omga = Atm(n)%omga
  a_ua = Atm(n)%ua;    a_va = Atm(n)%va
  a_uc = Atm(n)%uc;    a_vc = Atm(n)%vc
  a_mfx = Atm(n)%mfx;  a_mfy = Atm(n)%mfy
  a_cx = Atm(n)%cx;    a_cy = Atm(n)%cy
  a_w = Atm(n)%w;      a_delz = Atm(n)%delz
  a_qcon = Atm(n)%q_con
  a_phis = Atm(n)%phis
  if (allocated(Atm(n)%ze0)) a_ze0 = Atm(n)%ze0

  call stage_dump3('FINA_u', a_u)
  call stage_dump3('FINA_v', a_v)
  call stage_dump3('FINA_pt', a_pt)
  call stage_dump3('FINA_delp', a_delp)
  call stage_dump2('FINA_ps', a_ps)
  call stage_dump2('FINA_phis', a_phis)
  do iq = 1, min(Atm(n)%ncnst, 9)
    write (fname_dat, '(A,I1)') 'FINA_q', iq
    call stage_dump3(trim(fname_dat), a_q(:, :, :, iq))
  end do

  ! ----- restore the IC -----
  Atm(n)%u = s_u;      Atm(n)%v = s_v;       Atm(n)%w = s_w
  Atm(n)%pt = s_pt;    Atm(n)%delp = s_delp; Atm(n)%q = s_q
  Atm(n)%ps = s_ps;    Atm(n)%pe = s_pe;     Atm(n)%pk = s_pk
  Atm(n)%peln = s_peln; Atm(n)%pkz = s_pkz
  Atm(n)%phis = s_phis; Atm(n)%omga = s_omga
  Atm(n)%ua = s_ua;    Atm(n)%va = s_va
  Atm(n)%uc = s_uc;    Atm(n)%vc = s_vc
  Atm(n)%mfx = s_mfx;  Atm(n)%mfy = s_mfy
  Atm(n)%cx = s_cx;    Atm(n)%cy = s_cy
  Atm(n)%q_con = s_qcon
  Atm(n)%delz = s_delz
  if (allocated(Atm(n)%ze0)) Atm(n)%ze0 = s_ze0
  call mpp_sync()

  ! =================== arm B: the STAGED copies ======================
  call run_arm(.true.)

  call stage_dump3('FINB_u', Atm(n)%u)
  call stage_dump3('FINB_v', Atm(n)%v)
  call stage_dump3('FINB_pt', Atm(n)%pt)
  call stage_dump3('FINB_delp', Atm(n)%delp)
  call stage_dump2('FINB_ps', Atm(n)%ps)

  ! ----- CERT: staged must equal real bitwise ------------------------
  call cert3('u', a_u, Atm(n)%u)
  call cert3('v', a_v, Atm(n)%v)
  call cert3('pt', a_pt, Atm(n)%pt)
  call cert3('delp', a_delp, Atm(n)%delp)
  call cert3('pe', a_pe, Atm(n)%pe)
  call cert3('pk', a_pk, Atm(n)%pk)
  call cert3('peln', a_peln, Atm(n)%peln)
  call cert3('pkz', a_pkz, Atm(n)%pkz)
  call cert3('omga', a_omga, Atm(n)%omga)
  call cert3('ua', a_ua, Atm(n)%ua)
  call cert3('va', a_va, Atm(n)%va)
  call cert3('uc', a_uc, Atm(n)%uc)
  call cert3('vc', a_vc, Atm(n)%vc)
  call cert3('mfx', a_mfx, Atm(n)%mfx)
  call cert3('mfy', a_mfy, Atm(n)%mfy)
  call cert3('cx', a_cx, Atm(n)%cx)
  call cert3('cy', a_cy, Atm(n)%cy)
  call cert4('q', a_q, Atm(n)%q)
  call cert2('ps', a_ps, Atm(n)%ps)
  ! codex r1 #3: every intent(inout) actual of fv_dynamics, including
  ! the ones expected inactive on this lane -- an inactive field that
  ! moved is exactly the kind of surprise this rung exists to catch.
  call cert3('w', a_w, Atm(n)%w)
  call cert3('delz', a_delz, Atm(n)%delz)
  call cert3('q_con', a_qcon, Atm(n)%q_con)
  call cert2('phis', a_phis, Atm(n)%phis)
  if (allocated(Atm(n)%ze0)) call cert3('ze0', a_ze0, Atm(n)%ze0)

  call stage_dump_close()
  call nullify_domain()
  call mpp_sync()
  if (mpp_pe() == mpp_root_pe()) write (*, '(A)') 'STAGE_DRIVER_DONE'
  call fms_io_exit
  call fms_end()

contains

  subroutine run_arm(staged)
    logical, intent(in) :: staged
    ! Mirrors driver/solo/atmosphere.F90:444-456 exactly (adiabatic:
    ! zvir = 0; consv_te, n_split, q_split from the resolved namelist).
    if (staged) then
      call fv_dynamics_staged(Atm(n)%npx, Atm(n)%npy, Atm(n)%npz, &
           Atm(n)%ncnst, Atm(n)%ng, real(dt_atmos), &
           Atm(n)%flagstruct%consv_te, Atm(n)%flagstruct%fill, &
           Atm(n)%flagstruct%reproduce_sum, kappa, cp_air, zvir, &
           Atm(n)%ptop, Atm(n)%ks, Atm(n)%ncnst, &
           Atm(n)%flagstruct%n_split, Atm(n)%flagstruct%q_split, &
           Atm(n)%u, Atm(n)%v, Atm(n)%w, Atm(n)%delz, &
           Atm(n)%flagstruct%hydrostatic, Atm(n)%flagstruct%duogrid, &
           Atm(n)%pt, Atm(n)%delp, Atm(n)%q, Atm(n)%ps, &
           Atm(n)%pe, Atm(n)%pk, Atm(n)%peln, Atm(n)%pkz, &
           Atm(n)%phis, Atm(n)%q_con, Atm(n)%omga, Atm(n)%ua, &
           Atm(n)%va, Atm(n)%uc, Atm(n)%vc, Atm(n)%ak, Atm(n)%bk, &
           Atm(n)%mfx, Atm(n)%mfy, Atm(n)%cx, Atm(n)%cy, Atm(n)%ze0, &
           Atm(n)%flagstruct%hybrid_z, Atm(n)%gridstruct, &
           Atm(n)%flagstruct, Atm(n)%neststruct, Atm(n)%idiag, &
           Atm(n)%bd, Atm(n)%parent_grid, Atm(n)%domain, &
           Atm(n)%inline_mp, time_total=time_total)
    else
      call fv_dynamics(Atm(n)%npx, Atm(n)%npy, Atm(n)%npz, &
           Atm(n)%ncnst, Atm(n)%ng, real(dt_atmos), &
           Atm(n)%flagstruct%consv_te, Atm(n)%flagstruct%fill, &
           Atm(n)%flagstruct%reproduce_sum, kappa, cp_air, zvir, &
           Atm(n)%ptop, Atm(n)%ks, Atm(n)%ncnst, &
           Atm(n)%flagstruct%n_split, Atm(n)%flagstruct%q_split, &
           Atm(n)%u, Atm(n)%v, Atm(n)%w, Atm(n)%delz, &
           Atm(n)%flagstruct%hydrostatic, Atm(n)%flagstruct%duogrid, &
           Atm(n)%pt, Atm(n)%delp, Atm(n)%q, Atm(n)%ps, &
           Atm(n)%pe, Atm(n)%pk, Atm(n)%peln, Atm(n)%pkz, &
           Atm(n)%phis, Atm(n)%q_con, Atm(n)%omga, Atm(n)%ua, &
           Atm(n)%va, Atm(n)%uc, Atm(n)%vc, Atm(n)%ak, Atm(n)%bk, &
           Atm(n)%mfx, Atm(n)%mfy, Atm(n)%cx, Atm(n)%cy, Atm(n)%ze0, &
           Atm(n)%flagstruct%hybrid_z, Atm(n)%gridstruct, &
           Atm(n)%flagstruct, Atm(n)%neststruct, Atm(n)%idiag, &
           Atm(n)%bd, Atm(n)%parent_grid, Atm(n)%domain, &
           Atm(n)%inline_mp, time_total=time_total)
    end if
  end subroutine run_arm

  subroutine cert2(name, a, b)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:, :), b(:, :)
    character(len=256) :: line
    write (line, '(A,A,1X,ES24.16E3)') 'CERT ', trim(name), &
        maxval(abs(a - b))
    call stage_mf_note(line)
  end subroutine cert2

  subroutine cert3(name, a, b)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:, :, :), b(:, :, :)
    character(len=256) :: line
    write (line, '(A,A,1X,ES24.16E3)') 'CERT ', trim(name), &
        maxval(abs(a - b))
    call stage_mf_note(line)
  end subroutine cert3

  subroutine cert4(name, a, b)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:, :, :, :), b(:, :, :, :)
    character(len=256) :: line
    write (line, '(A,A,1X,ES24.16E3)') 'CERT ', trim(name), &
        maxval(abs(a - b))
    call stage_mf_note(line)
  end subroutine cert4

end program fv3_dyncore_stage_driver
