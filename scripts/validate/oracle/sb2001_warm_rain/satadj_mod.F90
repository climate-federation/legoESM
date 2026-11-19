! Wrapper module for the VERBATIM gSAM 1.8.8 satadj_liquid body
! (SRC/MICRO_M2005/microphysics.f90:1638-1720), included twice by
! py/patch_gsam.py: _stock (Flatau polysvp mixing ratio, as shipped) and _lego
! (legoESM qsl, per harness rule).  Only edits (see satadj_*.inc.diff):
! routine rename, run-time tolerance (thresh_mult, maxiter_run; stock = 1, 20),
! iteration-limit write -> counter.
module satadj_mod
  use params, only: cp, lcond, rv, fac_cond
  use module_mp_GRAUPEL, only: polysvp
  use lego_sat, only: qsl_lego
  implicit none
  real :: thresh_mult = 1.
  integer :: maxiter_run = 20
  integer :: nhit_maxiter = 0
contains
#include "satadj_stock.inc"
#include "satadj_lego.inc"
end module satadj_mod
