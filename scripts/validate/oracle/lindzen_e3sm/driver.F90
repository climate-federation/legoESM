! Copied unmodified from the tier-2 oracle audit (oracle_audit/simple_gwd_hines_lindzen_rayleigh_spectral/fortran_e3sm/driver_e3sm.F90).
! Standalone driver for the E3SM v3.0.1 gw_tend OROGRAPHIC-ONLY branch
! (gw_drag.F90:293 orographic_only, :341 alpha=1e-6, :722 gw_prof,
!  :884-895 gw_oro_src + gw_drag_prof, :901-914 landfrac + discrete-KE heating).
! Reads the same column file as the CAM driver; pmid is read as an extra
! record set (E3SM uses state%pmid, not the interface average).
program gw_oro_e3sm_driver
  use gw_utils, only: r8
  use gw_common, only: gw_common_init, gw_prof, gw_drag_prof, orographic_only
  use gw_oro, only: gw_oro_init, gw_oro_src
  implicit none
  integer :: ncol, pver, i, k, iu, ou, lndscl, vdiff, alpha_mode, kbotbg
  real(r8) :: gravit, rair, cpair, dt, effgw_oro, fcrit2, prndl, south_fac, alpha1
  real(r8), allocatable :: pint(:,:), pmid(:,:), t(:,:), u(:,:), v(:,:), zm(:,:), qv(:,:)
  real(r8), allocatable :: lat(:), sgh(:), lfrac(:), pref_edge(:), alpha(:)
  real(r8), allocatable :: rhoi(:,:), ti(:,:), nm(:,:), ni(:,:), piln(:,:), dpm(:,:), rdpm(:,:)
  real(r8), allocatable :: dse(:,:), q(:,:,:), kvtt(:,:), tau(:,:,:), ubm(:,:), ubi(:,:)
  real(r8), allocatable :: xv(:), yv(:), c(:,:), utgw(:,:), vtgw(:,:), ttgw(:,:), qtgw(:,:,:)
  real(r8), allocatable :: taucd(:,:,:), egwdffi(:,:), gwut(:,:,:), dttdf(:,:), dttke(:,:), tt(:,:)
  integer, allocatable :: src_level(:), tend_level(:)
  real(r8) :: cref(0:0)
  character(len=512) :: errstring, fin, fout
  call get_command_argument(1, fin); call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), status='old', action='read')
  read(iu,*) ncol, pver
  read(iu,*) gravit, rair, cpair
  read(iu,*) dt, effgw_oro, fcrit2, prndl, south_fac
  read(iu,*) lndscl, vdiff, alpha_mode
  allocate(pref_edge(pver+1), alpha(0:pver))
  read(iu,*) pref_edge
  allocate(pint(ncol,0:pver), pmid(ncol,pver), t(ncol,pver), u(ncol,pver), v(ncol,pver), &
       zm(ncol,pver), qv(ncol,pver), lat(ncol), sgh(ncol), lfrac(ncol))
  do i = 1, ncol
     read(iu,*) lat(i), sgh(i), lfrac(i)
     read(iu,*) pint(i,:)
     read(iu,*) t(i,:)
     read(iu,*) u(i,:)
     read(iu,*) v(i,:)
     read(iu,*) zm(i,:)
     read(iu,*) qv(i,:)
  end do
  do i = 1, ncol
     read(iu,*) pmid(i,:)
  end do
  read(iu,*) alpha1
  close(iu)
  orographic_only = .true.
  alpha = alpha1                       ! gw_drag.F90:341 (orographic_only: alpha = 1.e-6)
  kbotbg = maxloc(pref_edge, 1, (pref_edge < 50000._r8)) - 1
  cref = 0._r8
  call gw_common_init(pver, 0, 2.5_r8, cref, .false., .false., 0, 0, kbotbg, &
       fcrit2, 2._r8*acos(-1._r8)/1.e5_r8, gravit, rair, alpha, errstring)
  if (trim(errstring) /= "") stop 'gw_common_init'
  call gw_oro_init(errstring)
  allocate(rhoi(ncol,0:pver), ti(ncol,0:pver), nm(ncol,pver), ni(ncol,0:pver), piln(ncol,0:pver), &
       dpm(ncol,pver), rdpm(ncol,pver), dse(ncol,pver), q(ncol,pver,1), kvtt(ncol,0:pver), &
       tau(ncol,0:0,0:pver), ubm(ncol,pver), ubi(ncol,0:pver), xv(ncol), yv(ncol), c(ncol,0:0), &
       utgw(ncol,pver), vtgw(ncol,pver), ttgw(ncol,pver), qtgw(ncol,pver,1), taucd(ncol,0:pver,4), &
       egwdffi(ncol,0:pver), gwut(ncol,pver,0:0), dttdf(ncol,pver), dttke(ncol,pver), tt(ncol,pver), &
       src_level(ncol), tend_level(ncol))
  piln = log(pint)
  dpm = pint(:,1:pver) - pint(:,0:pver-1)
  rdpm = 1._r8/dpm
  kvtt = 0._r8
  dse = cpair*t + gravit*zm
  q(:,:,1) = qv
  call gw_prof(ncol, cpair, t, pmid, pint, rhoi, ti, nm, ni)
  call gw_oro_src(ncol, u, v, t, sgh, pmid, pint, dpm, zm, nm, &
       src_level, tend_level, tau, ubm, ubi, xv, yv, c)
  call gw_drag_prof(ncol, 0, src_level, tend_level, .false., dt, &
       lat, t, ti, pmid, pint, dpm, rdpm, piln, rhoi, nm, ni, ubm, ubi, xv, yv, &
       effgw_oro, c, kvtt, q, dse, tau, utgw, vtgw, ttgw, qtgw, taucd, egwdffi, gwut, dttdf, dttke)
  do k = 1, pver
     if (lndscl == 1) then
        utgw(:,k) = utgw(:,k)*lfrac
        vtgw(:,k) = vtgw(:,k)*lfrac
     end if
     tt(:,k) = (ttgw(:,k) - (utgw(:,k)*(u(:,k) + utgw(:,k)*0.5_r8*dt) &
          + vtgw(:,k)*(v(:,k) + vtgw(:,k)*0.5_r8*dt)))/cpair
  end do
  open(newunit=ou, file=trim(fout), status='replace', action='write')
  write(ou,'(a,2i6)') '# ncol pver', ncol, pver
  write(ou,'(a,40es25.16e3)') 'alpha', alpha
  do i = 1, ncol
     write(ou,'(a,i4,2i6,4es25.16e3)') 'col', i, src_level(i), tend_level(i), xv(i), yv(i), effgw_oro, sgh(i)
     write(ou,'(a,40es25.16e3)') 'tau', tau(i,0,:)
     write(ou,'(a,40es25.16e3)') 'utgw', utgw(i,:)
     write(ou,'(a,40es25.16e3)') 'vtgw', vtgw(i,:)
     write(ou,'(a,40es25.16e3)') 'ttgw', tt(i,:)
  end do
  close(ou)
  write(*,*) 'E3SM_ORACLE_OK', ncol, pver
end program gw_oro_e3sm_driver
