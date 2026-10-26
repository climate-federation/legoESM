"""Diagnostic-only patch of Isca v1.0 qe_moist_convection.F90 (P1): expose the
parcel calculation through a public wrapper.  No numerics changed."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding='latin-1').read()
a = "public  qe_moist_convection, qe_moist_convection_init, qe_moist_convection_end"
assert s.count(a) == 1
s = s.replace(a, a + "\n  public  qe_diag_parcel")
b = "end module qe_moist_convection_mod"
assert s.count(b) == 1
wrap = """  subroutine qe_diag_parcel(p_full, p_half, Tin, rin, kLZB, Tp, rp, cape, cin)
    ! DIAGNOSTIC WRAPPER (audit patch P1): the parcel routine the scheme calls.
    real, intent(in), dimension(:) :: p_full, p_half, Tin, rin
    integer, intent(out) :: kLZB
    real, intent(out), dimension(:) :: Tp, rp
    real, intent(out) :: cape, cin
    call CAPE_calculation(size(Tin), p_full, p_half, Tin, rin, kLZB, Tp, rp, &
         cape, cin, val_min, val_max, lcl_temp_table)
  end subroutine qe_diag_parcel

"""
s = s.replace(b, wrap + b)
open(dst, 'w', encoding='latin-1').write(s)
