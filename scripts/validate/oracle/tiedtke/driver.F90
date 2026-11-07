! Standalone driver for WRF v4.8.0 cu_tiedtke (module_cu_tiedtke.F), one call per
! column (i-dim = 1, j = 1).  WRF arrays are BOTTOM-UP (k=1 lowest); cu_tiedtke
! flips to the scheme's TOP-DOWN index.  Dumped internals (module tdbg) are in the
! scheme's top-down index (k=1 = model top).  itimestep=2 so qvften/qvpblten are used.
! stdin:
!   ncol nlev dt
!   per column: xland qfx
!               nlev lines (bottom-up): u v t qv(mixing ratio) qc qi pi rho qvften qvpblten dz8w pcps znu
!               nlev+1 lines (bottom-up interfaces): p8w w
program tdriver
  use module_cu_tiedtke
  use tdbg
  implicit none
  integer :: ncol, nlev, i, k
  real :: dt
  real, allocatable, dimension(:,:,:) :: u3d, v3d, w, t3d, qv3d, qc3d, qi3d, pi3d, rho3d, &
       qvften, qvpblten, dz8w, pcps, p8w, rthcuten, rqvcuten, rqccuten, rqicuten, rucuten, rvcuten
  real :: raincv(1,1), pratec(1,1), qfx(1,1), xland(1,1)
  real, allocatable :: znu(:)
  logical :: cu_act_flag(1,1)

  read(*,*) ncol, nlev, dt
  allocate(u3d(1,nlev+1,1), v3d(1,nlev+1,1), w(1,nlev+1,1), t3d(1,nlev+1,1), qv3d(1,nlev+1,1), &
       qc3d(1,nlev+1,1), qi3d(1,nlev+1,1), pi3d(1,nlev+1,1), rho3d(1,nlev+1,1), qvften(1,nlev+1,1), &
       qvpblten(1,nlev+1,1), dz8w(1,nlev+1,1), pcps(1,nlev+1,1), p8w(1,nlev+1,1), rthcuten(1,nlev+1,1), &
       rqvcuten(1,nlev+1,1), rqccuten(1,nlev+1,1), rqicuten(1,nlev+1,1), rucuten(1,nlev+1,1), &
       rvcuten(1,nlev+1,1), znu(nlev+1))
  write(*,'(A,I0,1X,I0)') 'NCOL ', ncol, nlev
  do i = 1, ncol
    u3d=0.; v3d=0.; w=0.; t3d=0.; qv3d=0.; qc3d=0.; qi3d=0.; pi3d=1.; rho3d=1.; qvften=0.; qvpblten=0.
    dz8w=0.; pcps=0.; p8w=0.; znu=0.
    rthcuten=0.; rqvcuten=0.; rqccuten=0.; rqicuten=0.; rucuten=0.; rvcuten=0.; raincv=0.; pratec=0.
    read(*,*) xland(1,1), qfx(1,1)
    do k = 1, nlev
      read(*,*) u3d(1,k,1), v3d(1,k,1), t3d(1,k,1), qv3d(1,k,1), qc3d(1,k,1), qi3d(1,k,1), &
           pi3d(1,k,1), rho3d(1,k,1), qvften(1,k,1), qvpblten(1,k,1), dz8w(1,k,1), pcps(1,k,1), znu(k)
    end do
    do k = 1, nlev + 1
      read(*,*) p8w(1,k,1), w(1,k,1)
    end do
    call cu_tiedtke(dt, 2, 1, raincv, pratec, qfx, znu, &
         u3d, v3d, w, t3d, qv3d, qc3d, qi3d, pi3d, rho3d, qvften, qvpblten, &
         dz8w, pcps, p8w, xland, cu_act_flag, &
         1, 2, 1, 2, 1, nlev+1, &
         1, 1, 1, 1, 1, nlev+1, &
         1, 1, 1, 1, 1, nlev, &
         rthcuten, rqvcuten, rqccuten, rqicuten, rucuten, rvcuten, &
         .true., .true., .false., .true., .false.)
    write(*,'(A,I0)') 'COL ', i
    call s('icum', real(d_icum))
    call s('ktype', real(d_ktype(1))); call s('ktype0', real(d_ktype0(1)))
    call s('ldcum', real(d_ldcum(1))); call s('ldcum0', real(d_ldcum0(1)))
    call s('kcbot', real(d_kcbot(1))); call s('kctop', real(d_kctop(1))); call s('kcbot0', real(d_kcbot0(1)))
    call s('idtop', real(d_idtop(1))); call s('loddraf', real(d_lodd(1)))
    call s('mfub', d_mfub(1)); call s('mfub0', d_mfub0(1)); call s('zdqpbl', d_zdqpbl(1))
    call s('zdqcv', d_zdqcv(1)); call s('zqumqe', d_zqumqe(1)); call s('zcape', d_zcape(1))
    call s('zheat', d_zheat(1)); call s('zrelh', d_zrelh(1)); call s('prec', d_prec(1))
    call s('raincv', raincv(1,1)); call s('pratec', pratec(1,1))
    call a('dtdt', d_ptte(1,1:nlev)); call a('dqdt', d_qte(1,1:nlev)); call a('dcdt', d_cte(1,1:nlev))
    call a('dudt', d_vom(1,1:nlev)); call a('dvdt', d_vol(1,1:nlev))
    call a('mfu', d_mfu(1,1:nlev)); call a('mfd', d_mfd(1,1:nlev)); call a('tu', d_tu(1,1:nlev))
    call a('qu', d_qu(1,1:nlev)); call a('lu', d_lu(1,1:nlev)); call a('lude', d_lude(1,1:nlev))
    call a('dmfup', d_dmfup(1,1:nlev)); call a('uu', d_uu(1,1:nlev)); call a('vu', d_vu(1,1:nlev))
    call a('ud', d_ud(1,1:nlev)); call a('vd', d_vd(1,1:nlev))
    call a('rthcuten', rthcuten(1,1:nlev,1)); call a('rqvcuten', rqvcuten(1,1:nlev,1))
    call a('rqccuten', rqccuten(1,1:nlev,1)); call a('rqicuten', rqicuten(1,1:nlev,1))
    call a('rucuten', rucuten(1,1:nlev,1))
  end do
contains
  subroutine s(key, val)
    character(*), intent(in) :: key
    real, intent(in) :: val
    write(*,'(A,1X,ES25.16E3)') 'S '//key, val
  end subroutine s
  subroutine a(key, arr)
    character(*), intent(in) :: key
    real, intent(in) :: arr(:)
    integer :: n
    write(*,'(A)', advance='no') 'A '//key
    do n = 1, size(arr)
      write(*,'(1X,ES25.16E3)', advance='no') arr(n)
    end do
    write(*,*)
  end subroutine a
end program tdriver
