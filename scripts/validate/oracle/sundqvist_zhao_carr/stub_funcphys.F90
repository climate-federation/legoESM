! STUB of ccpp-physics funcphys::fpvs (GFS lookup-table saturation vapour
! pressure, mixed liquid/ice). Replaced by the legoESM liquid Tetens curve
! (thermo.saturation_vapor_pressure: 611.2*exp(17.67*Tc/(Tc+243.5)), Tc=T-273.15,
! T clipped at 150 K) so that the oracle's RH equals ours and the comparison
! isolates the precipitation kernel. The curve difference is reported separately.
module funcphys
  use machine, only: kind_phys
  implicit none
contains
  elemental real(kind_phys) function fpvs(t)
    real(kind_phys), intent(in) :: t
    real(kind_phys) :: tc
    tc = max(t, 150.0_kind_phys) - 273.15_kind_phys
    fpvs = 611.2_kind_phys*exp(17.67_kind_phys*tc/(tc + 243.5_kind_phys))
  end function
end module funcphys
