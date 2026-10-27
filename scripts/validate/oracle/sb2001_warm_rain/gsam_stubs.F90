! STUBS for compiling gSAM 1.8.8 SRC/MICRO_M2005/module_mp_graupel.f90 standalone.
! params: legoESM constants (packages/core/legoesm/constants.py @37915678bb):
!   cp = c_pd = 1004.64, lcond = L_v = 2.501e6, lsub = L_v+L_f = 2.501e6+3.337e5,
!   rgas = R_d = 287.05, rv = R_v = 461.51, fac_cond = lcond/cp.
!   (gSAM consts.f90 values: cp 1004.64, lcond 2.501e6, lsub 2.834e6, lfus 0.337e6,
!    rgas 287.04, rv 461.5, ggr 9.79764 -- replaced per harness rule; mismatch reported.)
! grid: dx, dy, ady, mu only referenced by the gridarea fudge factor
!   (do_scale_dependence_of_autoconv = .false. => never read).
! lego_sat: legoESM saturation SPECIFIC humidity (thermo.py), identical to the
!   tier-1 stub verified rel 0.0 against legoesm.thermo (morrison_ice_phase).
module params
  implicit none
  real(8), parameter :: cp = 1004.64d0, lcond = 2.501d6, lsub = 2.501d6 + 3.337d5
  real(8), parameter :: rgas = 287.05d0, rv = 461.51d0
  real(8), parameter :: fac_cond = lcond/cp
end module params

module grid
  implicit none
  real :: dx = 1., dy = 1.
  real :: ady(1) = 1., mu(1) = 1.
  integer :: rank = 0
end module grid

module lego_sat
  implicit none
  real, parameter, private :: rd = 287.05, rv_ = 461.51, tf = 273.15
  real, parameter, private :: ls = 2.501e6 + 3.337e5, eps = rd/rv_
  interface
     pure function c_log1p(x) bind(C, name='log1p')
       use iso_c_binding, only: c_double
       real(c_double), value :: x
       real(c_double) :: c_log1p
     end function c_log1p
  end interface
contains
  elemental real function softplus(x)
    real, intent(in) :: x
    softplus = max(x, 0.) + c_log1p(exp(-abs(x)))
  end function softplus
  elemental real function qs_from_e(e, p)
    real, intent(in) :: e, p
    real :: w
    w = eps*e/(softplus(p - e - 1.) + 1.)
    w = 1. - softplus(20.*(1. - w))/20.
    qs_from_e = w/(1. + w)
  end function qs_from_e
  elemental real function qsl_lego(t, p)
    real, intent(in) :: t, p
    real :: tc
    tc = max(t, 150.) - tf
    qsl_lego = qs_from_e(611.2*exp(17.67*tc/(tc + 243.5)), p)
  end function qsl_lego
  elemental real function qsi_lego(t, p)
    real, intent(in) :: t, p
    qsi_lego = qs_from_e(611.2*exp(ls/rv_*(1./tf - 1./t)), p)
  end function qsi_lego
end module lego_sat

subroutine task_abort()
  stop 9
end subroutine task_abort
