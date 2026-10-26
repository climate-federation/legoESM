! legoESM saturation thermodynamics transcribed for the HARMONISED oracle arm.
! Source: packages/core/legoesm/thermo.py @ 37915678bb (unchanged through the port base)
!   saturation_vapor_pressure (Tetens, T clipped at 150 K),
!   _mixing_ratio_from_esat (softplus floor on p-e, LogSumExp cap at 1),
!   saturation_specific_humidity q = r/(1+r), its exact dT derivative,
!   latent_heat_vaporization L_v - (c_pw - c_pv)(T - T_freeze).
! Same function names/signatures as the oracle thermo_functions3.f90.
module lego_c
 use iso_c_binding
 interface
  real(c_double) function c_log1p(x) bind(C, name='log1p')
   import c_double
   real(c_double), value :: x
  end function
 end interface
contains
 real function softplus(x)
  real, intent(in) :: x
  softplus = max(x, 0.0) + c_log1p(exp(-abs(x)))
 end function
 real function sigm(x)
  real, intent(in) :: x
  sigm = 1.0/(1.0 + exp(-x))
 end function
end module lego_c

 FUNCTION esat(t)
 IMPLICIT NONE
 REAL :: esat
 REAL, INTENT(IN) :: t
 REAL :: tc
 tc = max(t, 150.0) - 273.15
 esat = 611.2*exp(17.67*tc/(tc + 243.5))
 END FUNCTION esat

 FUNCTION qsat(p,t)
 use lego_c
 IMPLICIT NONE
 REAL :: qsat, esat, e, d, r0, r, eps
 REAL, INTENT(IN) :: p,t
 eps = 287.05/461.51
 e = esat(t)
 d = softplus(p - e - 1.0) + 1.0
 r0 = eps*e/d
 r = 1.0 - softplus(20.0*(1.0 - r0))/20.0
 qsat = r/(1.0 + r)
 END FUNCTION qsat

 FUNCTION dqsat(p,t)
 use lego_c
 IMPLICIT NONE
 REAL :: dqsat, esat, e, de, d, r0, r, dr0, dr, eps, tc
 REAL, INTENT(IN) :: p,t
 eps = 287.05/461.51
 e = esat(t)
 tc = max(t, 150.0) - 273.15
 de = e*17.67*243.5/(tc + 243.5)**2
 if (t < 150.0) de = 0.0
 d = softplus(p - e - 1.0) + 1.0
 r0 = eps*e/d
 r = 1.0 - softplus(20.0*(1.0 - r0))/20.0
 dr0 = eps*de*(d + e*sigm(p - e - 1.0))/d**2
 dr = sigm(20.0*(1.0 - r0))*dr0
 dqsat = dr/(1.0 + r)**2
 END FUNCTION dqsat

 FUNCTION Lh(t,lsub)
 IMPLICIT NONE
 REAL :: Lh
 REAL, INTENT(IN) :: t
 LOGICAL, INTENT(IN) :: lsub
 Lh = 2.501E6 - (4218.0 - 1846.0)*(t - 273.15)
 if (lsub) Lh = Lh   ! sublimation branch disabled in the oracle Kuo path (lsub=.false.)
 END FUNCTION Lh
