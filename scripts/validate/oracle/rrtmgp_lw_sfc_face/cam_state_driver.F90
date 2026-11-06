! Runs CAM cam_cesm2_1_rel_60 rrtmg_state_init + rrtmg_state_create (the CAM6
! RRTMG host's level temperatures tlev, incl. the extra layer above the model
! top when pref_edge(1) > 1 Pa) on columns read from <dir>/cam_in.bin
! (Fortran order, float64): t(ncol,pver), pmid(ncol,pver), pint(ncol,pverp),
! tsfc(ncol); model top first.  lwup = stebol*tsfc**4 (a unit-emissivity
! surface), lnpint/lnpmid = log(pint/pmid) as the CAM dycore coupling sets
! them, pref_edge = pint of column 1.  Writes <dir>/cam_out.bin:
! num_rrtmg_levs (float64), tlev(ncol, num_rrtmg_levs+1).
program cam_state_driver
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols, pver, pverp
  use physconst, only: stebol
  use ref_pres, only: pref_edge
  use physics_types, only: physics_state
  use camsrfexch, only: cam_in_t
  use rrtmg_state, only: rrtmg_state_t, rrtmg_state_init, rrtmg_state_create, num_rrtmg_levs
  implicit none
  character(len=512) :: d
  type(physics_state) :: ps
  type(cam_in_t) :: ci
  type(rrtmg_state_t) :: rs
  real(r8) :: tsfc(pcols)
  integer :: u

  call get_command_argument(1, d)
  ps%lchnk = 1
  ps%ncol = pcols
  open(newunit=u, file=trim(d)//'/cam_in.bin', access='stream', form='unformatted', status='old')
  read(u) ps%t, ps%pmid, ps%pint, tsfc
  close(u)
  ps%lnpmid = log(ps%pmid)
  ps%lnpint = log(ps%pint)
  ci%lwup = stebol * tsfc**4
  pref_edge = ps%pint(1,:)
  call rrtmg_state_init()
  rs = rrtmg_state_create(ps, ci)
  open(newunit=u, file=trim(d)//'/cam_out.bin', access='stream', form='unformatted', status='replace')
  write(u) real(num_rrtmg_levs, r8), rs%tlev(:pcols, :num_rrtmg_levs+1)
  close(u)
  print *, 'CAM_STATE_DONE num_rrtmg_levs', num_rrtmg_levs, 'pverp', pverp
end program cam_state_driver
