! Minimal FMS-infrastructure stubs for running Isca v1.0 qe_moist_convection /
! betts_miller and GFDL atmos_phys 2021.02 moist_conv standalone.
! Precision: compiled with -fdefault-real-8 -fdefault-double-8 (fp64).
! CONSTANTS: -DSTOCK selects the Isca/FMS constants_mod values; default = legoESM
!            constants.py (g 9.80616, R_d 287.05, c_pd 1004.64, L_v 2.501e6, R_v 461.51).
! SATURATION (runtime, sat_mode):
!   1 = legoesm.thermo exactly: Tetens 611.2*exp(17.67 Tc/(Tc+243.5)), Tc=max(T,150)-273.15
!       (the e_s the oracle's escomp/lookup_es_des return), and for compute_qs (moist_conv only)
!       q = w/(1+w) with w = 1 - softplus(20(1-w0))/20, w0 = eps e/(softplus(p-e-1)+1)
!   2 = stock FMS: Goff-Gratch over water/ice with -20..0 C blend (sat_vapor_pres_k::compute_es_k,
!       evaluated exactly instead of via its lookup table), q = eps e/(p-(1-eps)e)
module constants_mod
  implicit none
#ifdef STOCK
  real, parameter :: GRAV = 9.80, RDGAS = 287.04, KAPPA = 2./7., CP_AIR = RDGAS/KAPPA
  real, parameter :: RVGAS = 461.50, HLV = 2.500e6, HLF = 3.34e5, HLS = HLV + HLF
  real, parameter :: TFREEZE = 273.16, ES0 = 1.0, CP_VAPOR = 4.0*RVGAS
#else
  real, parameter :: GRAV = 9.80616, RDGAS = 287.05, CP_AIR = 1004.64, KAPPA = RDGAS/CP_AIR
  real, parameter :: RVGAS = 461.51, HLV = 2.501e6, HLF = 3.337e5, HLS = HLV + HLF
  real, parameter :: TFREEZE = 273.15, ES0 = 1.0, CP_VAPOR = 1846.0
#endif
end module constants_mod

module mpp_mod
  implicit none
  character(len=64), dimension(1) :: input_nml_file = (/ '&moist_conv_nml /' /)
contains
  integer function mpp_pe(); mpp_pe = 0; end function mpp_pe
  integer function mpp_root_pe(); mpp_root_pe = 0; end function mpp_root_pe
  integer function stdlog(); stdlog = 6; end function stdlog
end module mpp_mod

module fms_mod
  use mpp_mod, only: mpp_pe, mpp_root_pe, stdlog
  implicit none
  integer, parameter :: NOTE = 0, WARNING = 1, FATAL = 2
  integer :: n_notes = 0, n_warnings = 0
contains
  logical function file_exist(name)
    character(len=*), intent(in) :: name
    inquire(file=trim(name), exist=file_exist)
  end function file_exist
  subroutine error_mesg(routine, msg, level)
    character(len=*), intent(in) :: routine, msg
    integer, intent(in) :: level
    if (level == FATAL) then
      write(*,*) 'FATAL ', trim(routine), ': ', trim(msg); stop 3
    else if (level == WARNING) then
      n_warnings = n_warnings + 1; write(*,*) 'WARNING ', trim(routine), ': ', trim(msg)
    else
      n_notes = n_notes + 1
    end if
  end subroutine error_mesg
  integer function open_file(file, action)
    character(len=*), intent(in) :: file
    character(len=*), intent(in), optional :: action
    if (trim(file) == 'logfile.out') then
      open(newunit=open_file, file='logfile.out', position='append', status='unknown')
    else
      open(newunit=open_file, file=trim(file), status='old', action='read')
    end if
  end function open_file
  integer function open_namelist_file()
    open(newunit=open_namelist_file, file='input.nml', status='old', action='read')
  end function open_namelist_file
  integer function check_nml_error(io, name)
    integer, intent(in) :: io
    character(len=*), intent(in) :: name
    check_nml_error = io
    if (io > 0) then
      write(*,*) 'namelist read error ', trim(name), io; stop 4
    end if
  end function check_nml_error
  subroutine close_file(unit)
    integer, intent(in) :: unit
    close(unit)
  end subroutine close_file
  subroutine write_version_number(v, t)
    character(len=*), intent(in) :: v, t
  end subroutine write_version_number
end module fms_mod

module time_manager_mod
  implicit none
  type time_type
    integer :: seconds = 0
  end type time_type
end module time_manager_mod

module diag_manager_mod
  implicit none
  interface send_data
    module procedure send_data_2d, send_data_3d
  end interface
contains
  integer function register_diag_field(mod_name, name, axes, time, lname, units, missing_value)
    use time_manager_mod, only: time_type
    character(len=*), intent(in) :: mod_name, name, lname, units
    integer, intent(in) :: axes(:)
    type(time_type), intent(in) :: time
    real, intent(in), optional :: missing_value
    register_diag_field = 0
  end function register_diag_field
  logical function send_data_2d(id, field, time, is, js)
    use time_manager_mod, only: time_type
    integer, intent(in) :: id
    real, intent(in) :: field(:,:)
    type(time_type), intent(in) :: time
    integer, intent(in), optional :: is, js
    send_data_2d = .false.
  end function send_data_2d
  logical function send_data_3d(id, field, time, is, js, ks, rmask)
    use time_manager_mod, only: time_type
    integer, intent(in) :: id
    real, intent(in) :: field(:,:,:)
    type(time_type), intent(in) :: time
    integer, intent(in), optional :: is, js, ks
    real, intent(in), optional :: rmask(:,:,:)
    send_data_3d = .false.
  end function send_data_3d
end module diag_manager_mod

module field_manager_mod
  implicit none
  integer, parameter :: MODEL_ATMOS = 1
end module field_manager_mod

module tracer_manager_mod
  implicit none
  integer, parameter :: NO_TRACER = -1
contains
  subroutine get_number_tracers(model, num_tracers, num_prog)
    integer, intent(in) :: model
    integer, intent(out), optional :: num_tracers, num_prog
    if (present(num_tracers)) num_tracers = 1
    if (present(num_prog)) num_prog = 1
  end subroutine get_number_tracers
  integer function get_tracer_index(model, name)
    integer, intent(in) :: model
    character(len=*), intent(in) :: name
    get_tracer_index = NO_TRACER
    if (trim(name) == 'sphum') get_tracer_index = 1
  end function get_tracer_index
  subroutine get_tracer_names(model, n, name, units)
    integer, intent(in) :: model, n
    character(len=*), intent(out), optional :: name, units
    if (present(name)) name = 'sphum'
    if (present(units)) units = 'kg/kg'
  end subroutine get_tracer_names
  subroutine get_tracer_indices(model)
    integer, intent(in) :: model
  end subroutine get_tracer_indices
  logical function query_method(a)
    character(len=*), intent(in) :: a
    query_method = .false.
  end function query_method
end module tracer_manager_mod

module sat_vapor_pres_mod
  use iso_c_binding, only: c_double
  use constants_mod, only: TFREEZE, RDGAS, RVGAS
  implicit none
  integer :: sat_mode = 1
  real, parameter :: TFRZ_LEGO = 273.15
  interface
    pure real(c_double) function c_log1p(x) bind(c, name="log1p")
      import :: c_double
      real(c_double), value :: x
    end function c_log1p
  end interface
  interface escomp
    module procedure escomp_0d
  end interface
  interface descomp
    module procedure descomp_0d
  end interface
  interface lookup_es_des
    module procedure lookup_es_des_0d, lookup_es_des_3d
  end interface
  interface compute_qs
    module procedure compute_qs_0d, compute_qs_3d
  end interface
contains
  elemental real function softplus(x)
    real, intent(in) :: x
    softplus = max(x, 0.) + c_log1p(exp(-abs(x)))
  end function softplus
  elemental real function sigm(x)
    real, intent(in) :: x
    sigm = 1./(1. + exp(-x))
  end function sigm
  elemental subroutine es_des(t, es, des)
    real, intent(in) :: t
    real, intent(out) :: es, des
    real :: tc, x, xi, xw, esi, esw, desi, desw, tbw, tbi, ln10
    if (sat_mode == 1) then
      tc = max(t, 150.) - TFRZ_LEGO
      es = 611.2*exp(17.67*tc/(tc + 243.5))
      if (t > 150.) then
        des = es*17.67*243.5/(tc + 243.5)**2
      else
        des = 0.
      end if
    else
      ! FMS compute_es_k (Goff-Gratch, ice below TFREEZE, blend -20..0 C); derivative analytic
      ln10 = log(10.)
      tbw = TFREEZE + 100.; tbi = TFREEZE
      esi = 0.; desi = 0.; esw = 0.; desw = 0.
      if (t < tbi) then
        xi = -9.09718*(tbi/t - 1.) - 3.56654*log10(tbi/t) + 0.876793*(1. - t/tbi) + log10(610.71)
        esi = 10.**xi
        desi = esi*ln10*(9.09718*tbi/t**2 + 3.56654/(t*ln10) - 0.876793/tbi)
      end if
      if (t > -20. + tbi) then
        xw = -7.90298*(tbw/t - 1.) + 5.02808*log10(tbw/t) &
             - 1.3816e-07*(10.**((1. - t/tbw)*11.344) - 1.) &
             + 8.1328e-03*(10.**((tbw/t - 1.)*(-3.49149)) - 1.) + log10(101324.60)
        esw = 10.**xw
        desw = esw*ln10*(7.90298*tbw/t**2 - 5.02808/(t*ln10) &
             + 1.3816e-07*10.**((1. - t/tbw)*11.344)*ln10*11.344/tbw &
             + 8.1328e-03*10.**((tbw/t - 1.)*(-3.49149))*ln10*3.49149*tbw/t**2)
      end if
      if (t <= -20. + tbi) then
        es = esi; des = desi
      else if (t >= tbi) then
        es = esw; des = desw
      else
        es = 0.05*((tbi - t)*esi + (t - tbi + 20.)*esw)
        des = 0.05*(-esi + (tbi - t)*desi + esw + (t - tbi + 20.)*desw)
      end if
    end if
  end subroutine es_des
  subroutine escomp_0d(t, es)
    real, intent(in) :: t
    real, intent(out) :: es
    real :: d
    call es_des(t, es, d)
  end subroutine escomp_0d
  subroutine descomp_0d(t, des)
    real, intent(in) :: t
    real, intent(out) :: des
    real :: e
    call es_des(t, e, des)
  end subroutine descomp_0d
  subroutine lookup_es_des_0d(t, es, des)
    real, intent(in) :: t
    real, intent(out) :: es, des
    call es_des(t, es, des)
  end subroutine lookup_es_des_0d
  subroutine lookup_es_des_3d(t, es, des)
    real, intent(in) :: t(:,:,:)
    real, intent(out) :: es(:,:,:), des(:,:,:)
    call es_des(t, es, des)
  end subroutine lookup_es_des_3d
  elemental subroutine qs_point(t, p, hc, qs, dqs, es)
    real, intent(in) :: t, p, hc
    real, intent(out) :: qs, dqs, es
    real :: des, eps, d, w0, w, dw0de, dwdw0
    eps = RDGAS/RVGAS
    call es_des(t, es, des)
    es = es*hc; des = des*hc
    if (sat_mode == 1) then
      d = softplus(p - es - 1.) + 1.
      w0 = eps*es/d
      dw0de = eps/d + eps*es*sigm(p - es - 1.)/d**2
      w = 1. - softplus(20.*(1. - w0))/20.
      dwdw0 = sigm(20.*(1. - w0))
      qs = w/(1. + w)
      dqs = dwdw0*dw0de*des/(1. + w)**2
    else
      d = p - (1. - eps)*es
      qs = eps*es/d
      dqs = eps*p*des/d**2
    end if
  end subroutine qs_point
  subroutine compute_qs_0d(t, p, qs, dqsdT, hc, esat)
    real, intent(in) :: t, p
    real, intent(out) :: qs
    real, intent(out), optional :: dqsdT, esat
    real, intent(in), optional :: hc
    real :: h, d, e
    h = 1.; if (present(hc)) h = hc
    call qs_point(t, p, h, qs, d, e)
    if (present(dqsdT)) dqsdT = d
    if (present(esat)) esat = e
  end subroutine compute_qs_0d
  subroutine compute_qs_3d(t, p, qs, dqsdT, hc, esat)
    real, intent(in) :: t(:,:,:), p(:,:,:)
    real, intent(out) :: qs(:,:,:)
    real, intent(out), optional :: dqsdT(:,:,:), esat(:,:,:)
    real, intent(in), optional :: hc
    real :: h
    real, dimension(size(t,1),size(t,2),size(t,3)) :: d, e
    h = 1.; if (present(hc)) h = hc
    call qs_point(t, p, h, qs, d, e)
    if (present(dqsdT)) dqsdT = d
    if (present(esat)) esat = e
  end subroutine compute_qs_3d
end module sat_vapor_pres_mod
