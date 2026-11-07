! Oracle driver for WRF v4.8.0 MYJ PBL (phys/module_bl_myjpbl.F::MYJPBL,
! WRF 06d4240ae989cc3e50af412bb472df3d9048783c), one column per call
! (ITS=ITE=JTS=JTE=1, LOWLYR=1), QV/QCW/QCI present (NSPEC=4).
! The MYJSFC surface-layer fields (USTAR, AKHS, AKMS, THZ0, QZ0, UZ0, VZ0,
! CT, CHKLOWQ, ELFLX, TSK, QSFC, ZNT) are inputs; MYJSFC is not called.
!
! Input (stdin, list-directed): ncol nlev nstep, then per column
!   dt xland ht ust akhs akms thz0 qz0 uz0 vz0 ct chklowq elflx tsk qsfc snow sice znt
!   then bottom-up level arrays: dz u v th exner pmid qv qcw qci, then pint (nlev+1)
! QV/QCW/QCI are mixing ratios (WRF moist arrays).  Each column is integrated
! nstep steps: step 1 starts from TKE_MYJ = EPSQ2 (MYJPBLINIT, no restart);
! later steps carry TKE_MYJ, CT, THZ0, QZ0, QSFC and advance u, v, th, qv, qcw,
! qci by the scheme's own tendencies (t = th*exner).
! Output (stdout): for every column and step, the inputs the step saw and
! every output, as "name n v1 .. vn".
program myj_driver
  use module_model_constants, only: epsq2
  use module_bl_myjpbl, only: myjpbl
  implicit none
  integer :: ncol, nlev, nstep, ic, is, n1
  real :: dt
  real, dimension(1,1) :: xland, ht, ust, akhs, akms, thz0, qz0, uz0, vz0, ct, chklowq, &
       elflx, tsk, qsfc, snow, sice, znt, pblh, mixht
  real, allocatable, dimension(:,:,:) :: dz, u, v, th, ex, pmid, qv, qcw, qci, pint, t, rho, &
       qcs, qcr, qcg, tke, el, exch, rub, rvb, rthb, rqvb, rqcb, rqib, rqsb, rqrb, rqgb
  integer, dimension(1,1) :: kpbl, lowlyr
  read(*,*) ncol, nlev, nstep
  n1 = nlev + 1
  allocate(dz(1,n1,1), u(1,n1,1), v(1,n1,1), th(1,n1,1), ex(1,n1,1), pmid(1,n1,1), &
       qv(1,n1,1), qcw(1,n1,1), qci(1,n1,1), pint(1,n1,1), t(1,n1,1), rho(1,n1,1), &
       qcs(1,n1,1), qcr(1,n1,1), qcg(1,n1,1), tke(1,n1,1), el(1,n1,1), exch(1,n1,1), &
       rub(1,n1,1), rvb(1,n1,1), rthb(1,n1,1), rqvb(1,n1,1), rqcb(1,n1,1), rqib(1,n1,1), &
       rqsb(1,n1,1), rqrb(1,n1,1), rqgb(1,n1,1))
  write(*,'(a,3i6)') 'dims ', ncol, nlev, nstep
  lowlyr = 1
  do ic = 1, ncol
     read(*,*) dt, xland, ht, ust, akhs, akms, thz0, qz0, uz0, vz0, ct, chklowq, &
          elflx, tsk, qsfc, snow, sice, znt
     dz = 0.; u = 0.; v = 0.; th = 0.; ex = 1.; pmid = 0.; qv = 0.; qcw = 0.; qci = 0.
     read(*,*) dz(1,1:nlev,1)
     read(*,*) u(1,1:nlev,1)
     read(*,*) v(1,1:nlev,1)
     read(*,*) th(1,1:nlev,1)
     read(*,*) ex(1,1:nlev,1)
     read(*,*) pmid(1,1:nlev,1)
     read(*,*) qv(1,1:nlev,1)
     read(*,*) qcw(1,1:nlev,1)
     read(*,*) qci(1,1:nlev,1)
     read(*,*) pint(1,1:n1,1)
     ! RHO is an unused MYJPBL argument (the scheme recomputes density)
     rho = 0.; qcs = 0.; qcr = 0.; qcg = 0.
     tke = epsq2; el = 0.; exch = 0.
     ! MYJPBL writes tendencies on 1..nlev only; the nlev+1 slot stays 0
     rub = 0.; rvb = 0.; rthb = 0.; rqvb = 0.; rqcb = 0.; rqib = 0.; rqsb = 0.; rqrb = 0.; rqgb = 0.
     do is = 1, nstep
        t = th*ex
        write(*,'(a,2i6)') 'step ', ic, is
        call put0('in_dt', dt); call put0('in_xland', xland(1,1)); call put0('in_ht', ht(1,1))
        call put0('in_ust', ust(1,1)); call put0('in_akhs', akhs(1,1)); call put0('in_akms', akms(1,1))
        call put0('in_thz0', thz0(1,1)); call put0('in_qz0', qz0(1,1)); call put0('in_uz0', uz0(1,1))
        call put0('in_vz0', vz0(1,1)); call put0('in_ct', ct(1,1)); call put0('in_chklowq', chklowq(1,1))
        call put0('in_elflx', elflx(1,1)); call put0('in_tsk', tsk(1,1)); call put0('in_qsfc', qsfc(1,1))
        call put0('in_snow', snow(1,1)); call put0('in_sice', sice(1,1))
        call put1('in_dz', dz(1,1:nlev,1)); call put1('in_u', u(1,1:nlev,1)); call put1('in_v', v(1,1:nlev,1))
        call put1('in_th', th(1,1:nlev,1)); call put1('in_t', t(1,1:nlev,1)); call put1('in_exner', ex(1,1:nlev,1))
        call put1('in_pmid', pmid(1,1:nlev,1)); call put1('in_qv', qv(1,1:nlev,1))
        call put1('in_qcw', qcw(1,1:nlev,1)); call put1('in_qci', qci(1,1:nlev,1)); call put1('in_pint', pint(1,:,1))
        call put1('in_tke', tke(1,1:nlev,1))
        call myjpbl(dt, 1, ht, dz, pmid, pint, th, t, ex, qv, qcw, qci, qcs, qcr, qcg, &
             u, v, rho, tsk, qsfc, chklowq, thz0, qz0, uz0, vz0, &
             lowlyr, xland, sice, snow, &
             tke, exch, ust, znt, el, pblh, kpbl, ct, &
             akhs, akms, elflx, mixht, &
             rub, rvb, rthb, rqvb, rqcb, rqib, rqsb, rqrb, rqgb, &
             1, 2, 1, 2, 1, n1, 1, 1, 1, 1, 1, n1, 1, 1, 1, 1, 1, nlev)
        call put1('out_rublten', rub(1,1:nlev,1)); call put1('out_rvblten', rvb(1,1:nlev,1))
        call put1('out_rthblten', rthb(1,1:nlev,1)); call put1('out_rqvblten', rqvb(1,1:nlev,1))
        call put1('out_rqcblten', rqcb(1,1:nlev,1)); call put1('out_rqiblten', rqib(1,1:nlev,1))
        call put1('out_tke', tke(1,1:nlev,1)); call put1('out_el', el(1,1:nlev,1))
        call put1('out_exch_h', exch(1,1:nlev,1))
        call put0('out_pblh', pblh(1,1)); call put0('out_kpbl', real(kpbl(1,1))); call put0('out_mixht', mixht(1,1))
        call put0('out_ct', ct(1,1)); call put0('out_thz0', thz0(1,1)); call put0('out_qz0', qz0(1,1))
        call put0('out_qsfc', qsfc(1,1))
        u = u + dt*rub; v = v + dt*rvb; th = th + dt*rthb
        qv = qv + dt*rqvb; qcw = qcw + dt*rqcb; qci = qci + dt*rqib
     end do
  end do
  write(*,'(a)') 'END_OK'
contains
  subroutine put0(name, x)
    character(len=*), intent(in) :: name
    real, intent(in) :: x
    write(*,'(a,1x,i6,1x,es26.17e3)') name, 1, x
  end subroutine put0
  subroutine put1(name, x)
    character(len=*), intent(in) :: name
    real, intent(in) :: x(:)
    write(*,'(a,1x,i6,*(1x,es26.17e3))') name, size(x), x
  end subroutine put1
end program myj_driver
