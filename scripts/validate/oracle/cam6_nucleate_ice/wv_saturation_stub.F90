! Minimal stand-in for CAM's wv_saturation module as nucleate_ice.F90 uses it: svp_water and svp_ice are
! wv_saturation.F90's elemental wrappers of wv_sat_methods (default scheme Goff-Gratch), here without the
! lookup tables nucleate_ice.F90 never calls.
module wv_saturation
  use wv_sat_methods, only: wv_sat_svp_water, wv_sat_svp_ice
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  private
  public :: svp_water, svp_ice
contains
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
