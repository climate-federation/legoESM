"""Make wv_sat_methods_patched.F90 from CAM cam_cesm2_1_rel_60 wv_sat_methods.F90.

Patch W1 (curve substitution only, like the V1 qsat stub): scheme index 91
selects legoESM's shared saturation curves inside CAM's own svp functions --
over water the Tetens curve of legoesm.thermo.saturation_vapor_pressure (with
its 150 K AD floor), over ice the Clausius-Clapeyron curve of
legoesm.thermo.saturation_vapor_pressure_ice -- so the unmodified
wv_sat_svp_trans (the CAM qsat liquid->ice blend, ttrice = 20 K) runs on the
curves the port uses.  lego_tfrz / lego_ls / lego_rv are set by the stub.
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
n = 0

decl = "integer :: default_idx = initial_default_idx\n"
assert s.count(decl) == 1
s = s.replace(decl, decl + (
    "! [patch W1] legoESM curves (index 91)\n"
    "integer, parameter, public :: Lego_idx = 91\n"
    "real(r8), public :: lego_tfrz = 273.15_r8, lego_ls = 2.8347e6_r8, lego_rv = 461.51_r8\n"))
n += 1

for fn, curve in (("wv_sat_svp_water", "lego_svp_water(t)"), ("wv_sat_svp_ice", "lego_svp_ice(t)")):
    head = f"elemental function {fn}(t, idx) result(es)"
    i = s.index(head)
    j = s.index("  end select", i)
    s = s[:j] + f"  case(Lego_idx)   ! [patch W1]\n     es = {curve}\n" + s[j:]
    n += 1

tail = "end module wv_sat_methods"
assert s.count(tail) == 1
s = s.replace(tail, """! [patch W1] legoesm.thermo.saturation_vapor_pressure (Tetens, 150 K floor)
elemental function lego_svp_water(t) result(es)
  real(r8), intent(in) :: t
  real(r8) :: es, tc
  tc = max(t, 150._r8) - lego_tfrz
  es = 611.2_r8*exp(17.67_r8*tc/(tc + 243.5_r8))
end function lego_svp_water

! [patch W1] legoesm.thermo.saturation_vapor_pressure_ice (Clausius-Clapeyron)
elemental function lego_svp_ice(t) result(es)
  real(r8), intent(in) :: t
  real(r8) :: es
  es = 611.2_r8*exp(lego_ls/lego_rv*(1._r8/lego_tfrz - 1._r8/t))
end function lego_svp_ice

""" + tail)
n += 1
assert n == 4, n
open(dst, "w").write(s)
print("patched wv_sat_methods", n)
