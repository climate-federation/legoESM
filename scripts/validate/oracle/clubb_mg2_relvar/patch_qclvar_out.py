"""Expose CLUBB's CAM output ``qclvar`` from the clubb_release f2py build.

Applied by build_oracle.sh to a clubb_release checkout (SHA pinned there) that
already carries ../clubb_core/constants_legoesm.patch.  Only OUTPUT routing
changes; no arithmetic is touched:

* pdf_closure_module: the ``#ifndef CLUBB_CAM`` stats gate around ``rcp2``
  is removed, i.e. ``rcp2`` is computed unconditionally as under CLUBB_CAM;
* advance_clubb_core_module: next to CAM's ``qclvar = rcp2_zt`` block
  (CLUBB_CAM) the same ``rcp2_zt`` is copied to a public module array, so
  the advance_clubb_core signature (and every other caller) is unchanged;
* the f2py wrapper, signature file and driver return it as one more trailing
  output of ``advance_clubb_core``.

Usage: python3 patch_qclvar_out.py <clubb_release dir>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def edit(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    if n != count:
        raise SystemExit(f"{rel}: expected {count} x {old!r}, found {n}")
    p.write_text(s.replace(old, new))


core = "src/CLUBB_core/"
edit(core + "pdf_closure_module.F90", "#ifndef CLUBB_CAM\n",
     "#if 0 /* rcp2 computed as under CLUBB_CAM */\n", 4)
edit(core + "advance_clubb_core_module.F90",
     "  implicit none\n\n  public ::  &\n    advance_clubb_core\n",
     "  use clubb_precision, only: qv_rknd => core_rknd\n\n  implicit none\n\n"
     "  real(qv_rknd), allocatable, public :: qclvar_out_last(:,:)\n\n"
     "  public ::  &\n    advance_clubb_core\n")
edit(core + "advance_clubb_core_module.F90",
     "#ifdef CLUBB_CAM\n    !$acc parallel loop gang vector collapse(2) default(present)\n"
     "    do k = 1, nzt\n      do i = 1, ngrdcol\n        qclvar(i,k) = rcp2_zt(i,k)\n",
     "    qclvar_out_last = rcp2_zt   ! = CAM's qclvar (CLUBB_CAM block below)\n"
     "#ifdef CLUBB_CAM\n    !$acc parallel loop gang vector collapse(2) default(present)\n"
     "    do k = 1, nzt\n      do i = 1, ngrdcol\n        qclvar(i,k) = rcp2_zt(i,k)\n")
w = "clubb_python_api/f2py_fortran_wrappers/advance_clubb_core_module_wrapper.F90"
edit(w, "    Kh_zt, Kh_zm, thlprcp, Lscale)\n", "    Kh_zt, Kh_zm, thlprcp, Lscale, qclvar)\n")
edit(w, "  use advance_clubb_core_module, only: advance_clubb_core\n",
     "  use advance_clubb_core_module, only: advance_clubb_core, qclvar_out_last\n")
edit(w, "    cloudy_downdraft_frac, Kh_zt, Lscale\n",
     "    cloudy_downdraft_frac, Kh_zt, Lscale, qclvar\n")
edit(w, "    Lscale)\n\nend subroutine f2py_advance_clubb_core",
     "    Lscale)\n\n  qclvar = qclvar_out_last\n\nend subroutine f2py_advance_clubb_core")
pyf = "clubb_python_api/clubb_f2py.pyf"
edit(pyf, "        kh_zt, kh_zm, thlprcp, lscale)\n", "        kh_zt, kh_zm, thlprcp, lscale, qclvar)\n")
edit(pyf, "        cloudy_downdraft_frac, kh_zt, lscale\n",
     "        cloudy_downdraft_frac, kh_zt, lscale, qclvar\n")
edit("clubb_python_driver/advance_clubb_to_end.py",
     "        state['Lscale'],\n    ) = clubb_api.advance_clubb_core(",
     "        state['Lscale'],\n        state['qclvar'],\n    ) = clubb_api.advance_clubb_core(")
print("PATCH_QCLVAR_OK")
