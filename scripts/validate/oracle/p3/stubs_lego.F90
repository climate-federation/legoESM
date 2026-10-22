! legoESM saturation SPECIFIC humidity stub fed to the WRF v4.8.0 P3 oracle (qv_sat is
! patched to call it, see patch_p3.py).  Reproduces legoesm.thermo
! saturation_specific_humidity[_ice] (Tetens over liquid with the T >= 150 K floor,
! Clausius-Clapeyron with L_s = L_v + L_f over ice, softplus caps); verified against
! legoesm.thermo by make_fixture.py on every fixture build.  Same code as the gSAM P3
! and tier-2 warm_rain_satadj oracle stubs.  Compiled with -fdefault-real-8.
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
