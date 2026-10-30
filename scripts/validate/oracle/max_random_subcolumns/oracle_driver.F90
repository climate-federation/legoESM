! CAM cam_cesm2_1_rel_60 maximum-random subcolumn oracle (rrtmg/aer_src/mcica_subcol_gen_sw.f90).
! mcica_subcol_gen_sw.f90 is compiled from a copy whose ONLY change is one line inserted just
! before the case(2) recursion (build.sh):
!     if (use_inject) CDF(:,:,:) = cdf_inject_sw(:nsubcol,:ncol,:nlay)
! so CAM's own recursion, cloudy test and path assignment run on harness-supplied uniforms.
! usage: oracle_driver <dir>
!  <dir>/dims.txt : ncol nlay
!  <dir>/in.bin   : play(ncol,nlay)[hPa] cld(ncol,nlay) clwp(ncol,nlay) ciwp(ncol,nlay)
!                   cdf(ngptsw,ncol,nlay)          (Fortran order, little-endian float64)
!  <dir>/out.bin  : cldf_sw, clwp_sw, ciwp_sw      (ngptsw,ncol,nlay each)
program oracle_driver
  use shr_kind_mod,        only: r8 => shr_kind_r8
  use mcica_inject_data,   only: use_inject, cdf_inject_sw
  use mcica_subcol_gen_sw, only: mcica_subcol_sw
  use parrrsw,             only: ngptsw, nbndsw
  use rrsw_wvn,            only: ngb_sw => ngb
  implicit none
  character(len=512) :: d
  integer :: ncol, nlay, u, ib, ig, k
  integer, parameter :: ngc_sw(14) = (/6,12,8,8,10,10,2,10,8,6,6,8,6,12/)   ! rrtmg_sw_init
  real(r8), allocatable :: play(:,:), cld(:,:), clwp(:,:), ciwp(:,:), rei(:,:), rel(:,:)
  real(r8), allocatable :: tauc(:,:,:), ssac(:,:,:), asmc(:,:,:), fsfc(:,:,:)
  real(r8), allocatable :: cldf_sw(:,:,:), ciwp_sw(:,:,:), clwp_sw(:,:,:), tau_sw(:,:,:), &
                           ssa_sw(:,:,:), asm_sw(:,:,:), fsf_sw(:,:,:), reic(:,:), relq(:,:)

  call get_command_argument(1, d)
  use_inject = .true.
  open(newunit=u, file=trim(d)//'/dims.txt', status='old')
  read(u,*) ncol, nlay
  close(u)
  allocate(play(ncol,nlay), cld(ncol,nlay), clwp(ncol,nlay), ciwp(ncol,nlay), rei(ncol,nlay), rel(ncol,nlay))
  allocate(cdf_inject_sw(ngptsw,ncol,nlay))
  open(newunit=u, file=trim(d)//'/in.bin', access='stream', form='unformatted', status='old')
  read(u) play, cld, clwp, ciwp, cdf_inject_sw
  close(u)
  rei = 25._r8; rel = 10._r8
  k = 0
  do ib = 1, 14
    do ig = 1, ngc_sw(ib)
      k = k + 1; ngb_sw(k) = 15 + ib
    end do
  end do
  allocate(tauc(nbndsw,ncol,nlay), ssac(nbndsw,ncol,nlay), asmc(nbndsw,ncol,nlay), fsfc(nbndsw,ncol,nlay))
  tauc = 1._r8; ssac = 0.9_r8; asmc = 0.8_r8; fsfc = 0.64_r8
  allocate(cldf_sw(ngptsw,ncol,nlay), ciwp_sw(ngptsw,ncol,nlay), clwp_sw(ngptsw,ncol,nlay), &
           tau_sw(ngptsw,ncol,nlay), ssa_sw(ngptsw,ncol,nlay), asm_sw(ngptsw,ncol,nlay), &
           fsf_sw(ngptsw,ncol,nlay), reic(ncol,nlay), relq(ncol,nlay))
  ! icld = 2 (maximum-random), permuteseed 1 as CAM radsw.F90
  call mcica_subcol_sw(0, ncol, nlay, 2, 1, play, cld, ciwp, clwp, rei, rel, tauc, ssac, asmc, fsfc, &
                       cldf_sw, ciwp_sw, clwp_sw, reic, relq, tau_sw, ssa_sw, asm_sw, fsf_sw)
  open(newunit=u, file=trim(d)//'/out.bin', access='stream', form='unformatted', status='replace')
  write(u) cldf_sw, clwp_sw, ciwp_sw
  close(u)
  print *, 'ORACLE_DONE ', ngptsw
end program oracle_driver
