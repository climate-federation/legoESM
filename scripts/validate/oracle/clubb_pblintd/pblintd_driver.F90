! CAM6 PBL height as clubb_intr.F90:2594-2617 (cam_cesm2_1_rel_60,
! a03b84b7c4e34f965b115686f22a043b85739e56) computes it after the CLUBB
! update: th = t*exner, thv = th*(1+zvir*q), calc_ustar, calc_obklen,
! kbfs(-0) -> 0, hb_diff::pblintd.  The statements below are copied from
! clubb_intr.F90; only the state1/cam_in members are plain arrays read from
! the input file.  init_hb_diff as clubb_intr.F90:828-831 (ntop_eddy = 1,
! nbot_eddy = pver).  cloud_frac is any value >= 0 (pblintd only tests >= 0).
! usage: pblintd_driver <inputs.txt> <outputs.txt>
program pblintd_driver
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid,       only: pver, pverp, pcols
  use pbl_utils,    only: pbl_utils_init, calc_ustar, calc_obklen
  use hb_diff,      only: init_hb_diff, pblintd
  implicit none
  character(len=512) :: fin, fout
  integer :: ncol, nlev, i, k
  real(r8) :: gravit, karman, cpair, rair, zvir
  real(r8) :: cin(5), buf(4)
  real(r8) :: pref_mid(pver)
  real(r8), dimension(pcols,pver)  :: t, q, u, v, pmid, zm, exner, th, thv, cloud_frac
  real(r8), dimension(pcols,pverp) :: zi
  real(r8), dimension(pcols) :: wsx, wsy, shf, cflx, landfrac
  real(r8), dimension(pcols) :: rrho, ustar2, kinheat, kinwat, kbfs, obklen, pblh, &
                                dummy2, dummy3

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old')
  read(10,*) ncol, nlev
  if (nlev /= pver .or. ncol > pcols) stop 'dimension mismatch'
  read(10,*) cin
  gravit = cin(1); karman = cin(2); cpair = cin(3); rair = cin(4); zvir = cin(5)
  read(10,*) pref_mid
  do i = 1, ncol
     read(10,*) t(i,:);  read(10,*) q(i,:);  read(10,*) u(i,:);  read(10,*) v(i,:)
     read(10,*) pmid(i,:); read(10,*) zm(i,:); read(10,*) zi(i,:); read(10,*) exner(i,:)
     read(10,*) buf
     wsx(i) = buf(1); wsy(i) = buf(2); shf(i) = buf(3); cflx(i) = buf(4)
  end do
  close(10)
  cloud_frac = 0._r8
  landfrac = 0._r8

  call pbl_utils_init(gravit, karman, cpair, rair, zvir)
  call init_hb_diff(gravit, cpair, 1, pver, pref_mid, karman, 'CLUBB_SGS       ')

  ! ---- clubb_intr.F90:2594-2617 ----
   do i=1,ncol
      do k=1,pver
         th(i,k) = t(i,k)*exner(i,k)
         thv(i,k) = th(i,k)*(1.0_r8+zvir*q(i,k))
      enddo
   enddo

   call calc_ustar( ncol, t(1:ncol,pver), pmid(1:ncol,pver), wsx(1:ncol), wsy(1:ncol), &
                    rrho(1:ncol), ustar2(1:ncol))
   call calc_obklen( ncol, th(1:ncol,pver), thv(1:ncol,pver), cflx(1:ncol), shf(1:ncol), &
                     rrho(1:ncol), ustar2(1:ncol), kinheat(1:ncol), kinwat(1:ncol), kbfs(1:ncol), &
                     obklen(1:ncol))

   dummy2(:) = 0._r8
   dummy3(:) = 0._r8

   where (kbfs(:ncol) .eq. -0.0_r8) kbfs(:ncol) = 0.0_r8

   call pblintd(ncol, thv, zm, u, v, &
                ustar2, obklen, kbfs, pblh, dummy2, &
                zi, cloud_frac(:,1:pver), 1._r8-landfrac, dummy3)
  ! ---- end clubb_intr.F90 ----

  open(20, file=trim(fout), status='replace')
  do i = 1, ncol
     write(20,'(5es26.17)') ustar2(i), obklen(i), kbfs(i), pblh(i), dummy2(i)
  end do
  close(20)
  write(*,*) 'ORACLE_OK ', ncol
end program pblintd_driver
