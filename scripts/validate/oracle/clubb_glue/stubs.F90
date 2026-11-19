! Minimal CAM infrastructure stubs for the clubb_intr / cldfrc2m / tropopause oracle.
! Physical constants = legoESM values (packages/core/legoesm/constants.py) per the
! harness rule; CAM/CESM2.1 shr_const values are listed in README for the mismatch finding.
module shr_kind_mod
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
end module shr_kind_mod

module ppgrid
  integer, parameter :: pcols = 8, pver = 32, pverp = pver + 1
end module ppgrid

module physconst
  use shr_kind_mod, only: r8 => shr_kind_r8
  real(r8), parameter :: gravit = 9.80616_r8          ! legoESM constants.g (CAM 9.80616)
  real(r8), parameter :: rair   = 287.05_r8           ! legoESM R_d (CAM 287.0423)
  real(r8), parameter :: rh2o   = 461.51_r8           ! legoESM R_v (CAM 461.5046)
  real(r8), parameter :: cpair  = 1004.64_r8          ! legoESM c_pd (CAM 1004.64)
  real(r8), parameter :: latvap = 2.501e6_r8          ! legoESM L_v (CAM 2.501e6)
  real(r8), parameter :: latice = 3.337e5_r8          ! legoESM L_f (CAM 3.337e5)
  real(r8), parameter :: zvir   = rh2o/rair - 1._r8   ! legoESM 1/eps - 1 (CAM 0.607793)
  real(r8), parameter :: epsilo = rair/rh2o           ! legoESM epsilon (CAM mwh2o/mwdry)
  real(r8), parameter :: tmelt  = 273.15_r8
  real(r8), parameter :: h2otrip = 273.16_r8
  real(r8), parameter :: tboil  = 373.16_r8           ! CAM wv_saturation tboil
  real(r8), parameter :: ttrice = 20.0_r8
  real(r8), parameter :: cappa  = rair/cpair
  real(r8), parameter :: pi     = 3.14159265358979323846_r8
end module physconst

module wv_saturation
  ! CAM wv_saturation.F90 wrappers (qsat_water -> wv_sat_qsat_water; svp_* -> wv_sat_svp_*)
  use shr_kind_mod, only: r8 => shr_kind_r8
  use wv_sat_methods, only: wv_sat_qsat_water, wv_sat_svp_water, wv_sat_svp_ice
  implicit none
contains
  elemental subroutine qsat_water(t, p, es, qs)
    real(r8), intent(in) :: t, p
    real(r8), intent(out) :: es, qs
    call wv_sat_qsat_water(t, p, es, qs)
  end subroutine qsat_water
  elemental function svp_water(t) result(es)
    real(r8), intent(in) :: t
    real(r8) :: es
    es = wv_sat_svp_water(t)
  end function svp_water
  elemental function svp_ice(t) result(es)
    real(r8), intent(in) :: t
    real(r8) :: es
    es = wv_sat_svp_ice(t)
  end function svp_ice
end module wv_saturation

module cldfrc2m_x
  ! cldfrc2m module state needed by aist_vector (iceopt=5, CAM6 clubb namelist)
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols
  use physconst, only: rair
  use wv_saturation, only: qsat_water, svp_water, svp_ice
  implicit none
  real(r8) :: rhmini_const = 0.80_r8, rhmaxi_const = 1.0_r8
  real(r8) :: rhminis_const = 1.0_r8, rhmaxis_const = 1.0_r8
  logical  :: cldfrc2m_do_subgrid_growth = .false.
  include 'inc_qist.inc'
  real(r8) :: rhminl_const = 0.9_r8, rhminl_adj_land_const = 0.1_r8, rhminh_const = 0.8_r8
  real(r8) :: premit = 75000._r8, premib = 75000._r8, icecrit = 0.95_r8
  integer  :: iceopt = 5
contains
  include 'inc_aist_vector.inc'
end module cldfrc2m_x

module tropo_x
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols, pver, pverp
  use physconst, only: cappa, gravit, rair
  implicit none
  integer, parameter :: NOTFOUND = -1
  real(r8) :: cnst_kap, cnst_faktor, cnst_ka1
  type physics_state
     integer :: lchnk = 1, ncol
     real(r8) :: lat(pcols)
     real(r8) :: t(pcols,pver), pmid(pcols,pver), pdel(pcols,pver), pint(pcols,pverp)
     real(r8) :: zm(pcols,pver), zi(pcols,pverp), s(pcols,pver), u(pcols,pver), v(pcols,pver)
     real(r8) :: omega(pcols,pver), phis(pcols), q(pcols,pver,4)
  end type physics_state
contains
  subroutine tropo_init()
    ! tropopause.F90:141-143
    cnst_kap    = cappa
    cnst_faktor = -gravit/rair
    cnst_ka1    = cnst_kap - 1._r8
  end subroutine tropo_init
  real(r8) function tropopause_interpolateT(pstate, icol, tropLev, tropP)
    type(physics_state), intent(in) :: pstate
    integer, intent(in) :: icol, tropLev
    real(r8), intent(in) :: tropP
    tropopause_interpolateT = -1._r8   ! stub: optional output never requested
  end function
  real(r8) function tropopause_interpolateZ(pstate, icol, tropLev, tropP)
    type(physics_state), intent(in) :: pstate
    integer, intent(in) :: icol, tropLev
    real(r8), intent(in) :: tropP
    tropopause_interpolateZ = -1._r8
  end function
  include 'inc_tropopause_analytic.inc'
  include 'inc_twmo.inc'
  include 'inc_tropopause_twmo.inc'
end module tropo_x
