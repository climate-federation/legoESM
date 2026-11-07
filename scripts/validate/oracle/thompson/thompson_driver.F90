! Column driver for WRF v4.8.0 phys/module_mp_thompson.F (mp_physics=8:
! not aerosol-aware, not hail-aware), calling the scheme's own entry points
! thompson_init and mp_gt_driver exactly as WRF does.
!
! usage: thompson_driver <infile> <outfile> [<table_dump_dir>]
! infile:  line 1 "ncol nz dt"; then ncol*nz lines, level 1 = BOTTOM:
!          T p dz w qv qc qi qr qs qg ni nr
!          (T K, p Pa, dz m, w m/s, q mixing ratios kg/kg, ni nr per kg)
! outfile: per column "PPT rainncv snowncv graupelncv" then nz lines
!          "k T qv qc qi qr qs qg ni nr" (final state) and, when the dump
!          build is used, "TEN k tten qvten qcten qiten qrten qsten qgten
!          niten nrten ncten" (mp_thompson's tendency arrays just before the
!          final update; write-only copies, see patch_dump.py).
! th is passed as T with pii = 1 so t1d = th*pii = T exactly.
program thompson_driver
  use module_mp_thompson, only: thompson_init, mp_gt_driver, &
       tcg_racg, tmr_racg, tcr_gacr, tnr_racg, tnr_gacr, &
       tcs_racs1, tmr_racs1, tcs_racs2, tmr_racs2, tcr_sacr1, tms_sacr1, &
       tcr_sacr2, tms_sacr2, tnr_racs1, tnr_racs2, tnr_sacr1, tnr_sacr2, &
       tpi_qrfz, tpg_qrfz, tni_qrfz, tnr_qrfz, tpi_qcfz, tni_qcfz, &
       tps_iaus, tni_iaus, tpi_ide, t_Efrw, t_Efsw
#ifdef THOM_DUMP
  use thom_dump
#endif
  implicit none
  character(len=512) :: fin, fout, tdir
  integer :: ncol, nz, i, k, nargs
  real :: dt
  real, allocatable, dimension(:,:,:) :: qv, qc, qr, qi, qs, qg, ni, nr, th, pii, p, w, dz, &
       refl, rec, rei, res, hgt
  real, allocatable, dimension(:,:) :: rainnc, rainncv, snownc, snowncv, grnc, grncv, sr

  nargs = command_argument_count()
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=fin, status='old')
  read(10, *) ncol, nz, dt
  allocate(qv(ncol,nz,1), qc(ncol,nz,1), qr(ncol,nz,1), qi(ncol,nz,1), qs(ncol,nz,1), &
       qg(ncol,nz,1), ni(ncol,nz,1), nr(ncol,nz,1), th(ncol,nz,1), pii(ncol,nz,1), &
       p(ncol,nz,1), w(ncol,nz,1), dz(ncol,nz,1), refl(ncol,nz,1), rec(ncol,nz,1), &
       rei(ncol,nz,1), res(ncol,nz,1))
  allocate(rainnc(ncol,1), rainncv(ncol,1), snownc(ncol,1), snowncv(ncol,1), &
       grnc(ncol,1), grncv(ncol,1), sr(ncol,1))
  do i = 1, ncol
    do k = 1, nz
      read(10, *) th(i,k,1), p(i,k,1), dz(i,k,1), w(i,k,1), qv(i,k,1), qc(i,k,1), &
           qi(i,k,1), qr(i,k,1), qs(i,k,1), qg(i,k,1), ni(i,k,1), nr(i,k,1)
    end do
  end do
  close(10)
  pii = 1.0
  refl = 0.; rec = 0.; rei = 0.; res = 0.
  rainnc = 0.; rainncv = 0.; snownc = 0.; snowncv = 0.; grnc = 0.; grncv = 0.; sr = 0.

  ! thompson_init only reads hgt in a debug print at (its+1, k, jts+1); give
  ! it room so that read stays in bounds.
  allocate(hgt(ncol+1, nz, 2))
  hgt = 0.
  call thompson_init(hgt, ids=1, ide=ncol+2, jds=1, jde=3, kds=1, kde=nz, &
       ims=1, ime=ncol+1, jms=1, jme=2, kms=1, kme=nz, &
       its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz)

  if (nargs >= 3) then
    call get_command_argument(3, tdir)
    call dump5('tcg_racg', tcg_racg); call dump5('tmr_racg', tmr_racg)
    call dump5('tcr_gacr', tcr_gacr); call dump5('tnr_racg', tnr_racg)
    call dump5('tnr_gacr', tnr_gacr)
    call dump4('tcs_racs1', tcs_racs1); call dump4('tmr_racs1', tmr_racs1)
    call dump4('tcs_racs2', tcs_racs2); call dump4('tmr_racs2', tmr_racs2)
    call dump4('tcr_sacr1', tcr_sacr1); call dump4('tms_sacr1', tms_sacr1)
    call dump4('tcr_sacr2', tcr_sacr2); call dump4('tms_sacr2', tms_sacr2)
    call dump4('tnr_racs1', tnr_racs1); call dump4('tnr_racs2', tnr_racs2)
    call dump4('tnr_sacr1', tnr_sacr1); call dump4('tnr_sacr2', tnr_sacr2)
    call dump4('tpi_qrfz', tpi_qrfz); call dump4('tpg_qrfz', tpg_qrfz)
    call dump4('tni_qrfz', tni_qrfz); call dump4('tnr_qrfz', tnr_qrfz)
    call dump4('tpi_qcfz', tpi_qcfz); call dump4('tni_qcfz', tni_qcfz)
    call dump2('tps_iaus', tps_iaus); call dump2('tni_iaus', tni_iaus)
    call dump2('tpi_ide', tpi_ide)
    call dump2('t_Efrw', t_Efrw); call dump2('t_Efsw', t_Efsw)
  end if

#ifdef THOM_DUMP
  allocate(dmp(ncol, nz, ndmp))
  dmp = 0.
#endif
  call mp_gt_driver(qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, ni=ni, nr=nr, &
       th=th, pii=pii, p=p, w=w, dz=dz, dt_in=dt, itimestep=1, &
       RAINNC=rainnc, RAINNCV=rainncv, SNOWNC=snownc, SNOWNCV=snowncv, &
       GRAUPELNC=grnc, GRAUPELNCV=grncv, SR=sr, refl_10cm=refl, &
       re_cloud=rec, re_ice=rei, re_snow=res, has_reqc=0, has_reqi=0, has_reqs=0, &
       ids=1, ide=ncol+1, jds=1, jde=2, kds=1, kde=nz, &
       ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz, &
       its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz)

  open(20, file=fout, status='replace')
  write(20, '(a,2i6,es25.16e3)') 'HDR', ncol, nz, dt
  do i = 1, ncol
    write(20, '(a,3es25.16e3)') 'PPT', rainncv(i,1), snowncv(i,1), grncv(i,1)
    do k = 1, nz
      write(20, '(i4,9es25.16e3)') k, th(i,k,1), qv(i,k,1), qc(i,k,1), qi(i,k,1), &
           qr(i,k,1), qs(i,k,1), qg(i,k,1), ni(i,k,1), nr(i,k,1)
#ifdef THOM_DUMP
      write(20, '(a,i4,10es25.16e3)') 'TEN', k, dmp(i,k,1:10)
#endif
    end do
  end do
  close(20)

contains

  subroutine opent(name, u)
    character(len=*), intent(in) :: name
    integer, intent(out) :: u
    open(newunit=u, file=trim(tdir)//'/'//name//'.bin', access='stream', &
         form='unformatted', status='replace')
  end subroutine opent

  subroutine dump2(name, a)
    character(len=*), intent(in) :: name
    real(kind=8), intent(in) :: a(:,:)
    integer :: u
    call opent(name, u); write(u) 2, shape(a), a; close(u)
  end subroutine dump2

  subroutine dump4(name, a)
    character(len=*), intent(in) :: name
    real(kind=8), intent(in) :: a(:,:,:,:)
    integer :: u
    call opent(name, u); write(u) 4, shape(a), a; close(u)
  end subroutine dump4

  subroutine dump5(name, a)
    character(len=*), intent(in) :: name
    real(kind=8), intent(in) :: a(:,:,:,:,:)
    integer :: u
    call opent(name, u); write(u) 5, shape(a), a; close(u)
  end subroutine dump5

end program thompson_driver
