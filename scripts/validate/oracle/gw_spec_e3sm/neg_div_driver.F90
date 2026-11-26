! #2567 oracle probe (neg_div_probe.py): call UNMODIFIED E3SM v3.0.1 gw_drag_prof directly on a single
! constructed column (one wave, spectral path), with a caller-set launched
! stress profile.  Input (text): pver dt effgw c src tend lat xv yv kwv fcrit2
! gravit rair cpair / pint(0:pver) / pmid / t / ubm / ubi(0:pver) / tau(0:pver)
! / alpha(0:pver).
program gw_neg_div
  use gw_utils, only: r8
  use gw_common, only: gw_common_init, gw_prof, gw_drag_prof, orographic_only
  implicit none
  integer, parameter :: ncol = 1
  integer :: pver, src, tend, iu, ou, k
  real(r8) :: dt, effgw, c0, lat0, xv0, yv0, kwv, fcrit2, gravit, rair, cpair
  real(r8), allocatable :: pint(:,:), pmid(:,:), t(:,:), ubm(:,:), ubi(:,:), tau(:,:,:)
  real(r8), allocatable :: alpha(:), piln(:,:), dpm(:,:), rdpm(:,:), rhoi(:,:), ti(:,:)
  real(r8), allocatable :: nm(:,:), ni(:,:), q(:,:,:), dse(:,:), kvtt(:,:), utgw(:,:), vtgw(:,:)
  real(r8), allocatable :: ttgw(:,:), qtgw(:,:,:), taucd(:,:,:), egwdffi(:,:), gwut(:,:,:)
  real(r8), allocatable :: dttdf(:,:), dttke(:,:)
  real(r8) :: c(ncol,0:0), cref(0:0), lat(ncol), xv(ncol), yv(ncol)
  integer :: src_level(ncol), tend_level(ncol)
  character(len=512) :: fin, fout, errstring

  call get_command_argument(1, fin); call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), status='old', action='read')
  read(iu,*) pver, dt, effgw, c0, src, tend, lat0, xv0, yv0, kwv, fcrit2, gravit, rair, cpair
  allocate(pint(ncol,0:pver), pmid(ncol,pver), t(ncol,pver), ubm(ncol,pver), ubi(ncol,0:pver), &
       tau(ncol,0:0,0:pver), alpha(0:pver), piln(ncol,0:pver), dpm(ncol,pver), rdpm(ncol,pver), &
       rhoi(ncol,0:pver), ti(ncol,0:pver), nm(ncol,pver), ni(ncol,0:pver), q(ncol,pver,1), &
       dse(ncol,pver), kvtt(ncol,0:pver), utgw(ncol,pver), vtgw(ncol,pver), ttgw(ncol,pver), &
       qtgw(ncol,pver,1), taucd(ncol,0:pver,4), egwdffi(ncol,0:pver), gwut(ncol,pver,0:0), &
       dttdf(ncol,pver), dttke(ncol,pver))
  read(iu,*) pint(1,:); read(iu,*) pmid(1,:); read(iu,*) t(1,:)
  read(iu,*) ubm(1,:); read(iu,*) ubi(1,:); read(iu,*) tau(1,0,:); read(iu,*) alpha
  close(iu)
  c = c0; cref = c0; lat = lat0; xv = xv0; yv = yv0
  src_level = src; tend_level = tend
  orographic_only = .false.
  call gw_common_init(pver, 0, 0._r8, cref, .false., .false., 0, 0, pver, &
       fcrit2, kwv, gravit, rair, alpha, errstring)
  if (trim(errstring) /= "") stop 'gw_common_init'
  call gw_prof(ncol, cpair, t, pmid, pint, rhoi, ti, nm, ni)
  piln = log(pint)
  dpm(:,:) = pint(:,1:pver) - pint(:,0:pver-1)
  rdpm = 1._r8/dpm
  q = 0._r8; dse = cpair*t; kvtt = 0._r8
  call gw_drag_prof(ncol, 0, src_level, tend_level, .false., dt, &
       lat, t, ti, pmid, pint, dpm, rdpm, &
       piln, rhoi, nm, ni, ubm, ubi, xv, yv, &
       effgw, c, kvtt, q, dse, tau, utgw, vtgw, &
       ttgw, qtgw, taucd, egwdffi, gwut, dttdf, dttke)
  open(newunit=ou, file=trim(fout), status='replace', action='write')
  write(ou,'(a,1x,1000es25.16e3)') 'rhoi', rhoi(1,:)
  write(ou,'(a,1x,1000es25.16e3)') 'ti', ti(1,:)
  write(ou,'(a,1x,1000es25.16e3)') 'nm', nm(1,:)
  write(ou,'(a,1x,1000es25.16e3)') 'ni', ni(1,:)
  write(ou,'(a,1x,1000es25.16e3)') 'tau', tau(1,0,:)
  write(ou,'(a,1x,1000es25.16e3)') 'utgw', utgw(1,:)
  write(ou,'(a,1x,1000es25.16e3)') 'vtgw', vtgw(1,:)
  write(ou,'(a,1x,1000es25.16e3)') 'gwut', gwut(1,:,0)
  close(ou)
  write(*,*) 'E3SM_DIRECT_OK'
end program gw_neg_div
