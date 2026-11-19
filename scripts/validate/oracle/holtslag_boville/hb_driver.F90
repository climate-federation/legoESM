! Holtslag-Boville oracle driver: E3SM hb_diff::compute_hb_diff (eddy_scheme
! 'HB') then CAM diffusion_solver::compute_vdiff set up exactly as
! vertical_diffusion.F90::vertical_diffusion_tend does for HB in a CAM4 run
! (do_iss=.false. [namelist_defaults_cam.xml], no TMS, no Beljaars, no
! molecular diffusion; fieldlist u, v, s, q; kvq = kvh; qmincg(Q) = 1e-12
! [physpkg.F90 cnst_add('Q', ..., qmin=1.e-12)]).  phis = 0, so s = cp T + g zm.
! usage: hb_driver <inputs.txt> <outputs.txt>
program hb_driver
  use shr_kind_mod,     only: r8 => shr_kind_r8
  use ppgrid,           only: pver, pverp, pcols
  use physconst,        only: rairv
  use pbl_utils,        only: pbl_utils_init
  use hb_diff,          only: init_hb_diff, compute_hb_diff
  use coords_1d,        only: Coords1D
  use diffusion_solver, only: init_vdiff, compute_vdiff, new_fieldlist_vdiff, vdiff_select, &
                              vdiff_selector
  implicit none
  character(len=512) :: fin, fout
  integer :: ncol, nlev, i, k
  real(r8) :: g, vk, cpair, rair, zvir, dt
  real(r8) :: cin(6), buf(4)
  real(r8) :: pref_mid(pver)
  real(r8), dimension(pcols,pver)  :: t, q, u, v, pmid, zm, th, cldn
  real(r8), dimension(pcols,pverp) :: pint, zi
  real(r8), dimension(pcols) :: shflx, qflx, taux, tauy, ocnfrac
  real(r8), dimension(pcols,pverp) :: kvm, kvh, kvq, cgh, cgs, tke
  real(r8), dimension(pcols) :: obklen, ustar, pblh, tpert, qpert
  real(r8) :: ri(pcols,pver)
  type(Coords1D) :: p
  type(vdiff_selector) :: fieldlist, fieldlistm
  character(128) :: errstring
  real(r8), dimension(pcols,pverp) :: tint, rhoi
  real(r8), dimension(pcols,pver)  :: uo, vo, so, dtk, cpairv, dragblj
  real(r8) :: qo(pcols,pver,1), cflx(pcols,1), qmincg(1)
  real(r8), dimension(pcols) :: ksrftms, tautmsx, tautmsy, topflx, tauresx, tauresy, dse_top

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old')
  read(10,*) ncol, nlev
  if (nlev /= pver .or. ncol > pcols) stop 'dimension mismatch'
  read(10,*) cin
  g = cin(1); vk = cin(2); cpair = cin(3); rair = cin(4); zvir = cin(5); dt = cin(6)
  read(10,*) pref_mid
  do i = 1, ncol
     read(10,*) t(i,:);  read(10,*) q(i,:);  read(10,*) u(i,:);  read(10,*) v(i,:)
     read(10,*) pmid(i,:); read(10,*) pint(i,:); read(10,*) zm(i,:); read(10,*) zi(i,:)
     read(10,*) th(i,:)
     read(10,*) buf
     shflx(i) = buf(1); qflx(i) = buf(2); taux(i) = buf(3); tauy(i) = buf(4)
  end do
  close(10)

  cldn = 0._r8          ! any value >= 0 -> the unconditional zi(pver)+50 floor
  ocnfrac = 0._r8       ! read by nothing in compute_hb_diff
  call pbl_utils_init(g, vk, cpair, rair, zvir)
  call init_hb_diff(g, cpair, 1, pver, pref_mid, vk, 'HB              ')
  call compute_hb_diff(1, ncol, th, t, q, zm, zi, pmid, u, v, taux, tauy, &
       shflx, qflx, obklen, ustar, pblh, kvm, kvh, kvq, cgh, cgs, &
       tpert, qpert, cldn, ocnfrac, tke, ri, 'HB              ')

  call init_vdiff(r8, 6, rair, cpair, g, .false., .false., errstring)
  if (len_trim(errstring) > 0) stop 'init_vdiff failed'
  fieldlist  = new_fieldlist_vdiff(1)
  fieldlistm = new_fieldlist_vdiff(1)
  errstring = vdiff_select(fieldlist, 'u')
  errstring = vdiff_select(fieldlist, 'v')
  errstring = vdiff_select(fieldlist, 's')
  errstring = vdiff_select(fieldlist, 'q', 1)
  do k = 2, pver
     tint(:ncol,k) = 0.5_r8*(t(:ncol,k) + t(:ncol,k-1))
  end do
  tint(:ncol,pver+1) = t(:ncol,pver)
  tint(:ncol,1) = t(:ncol,1)
  p = Coords1D(pint(:ncol,:))
  do k = 1, pver+1
     rhoi(:ncol,k) = p%ifc(:,k)/(rair*tint(:ncol,k))
  end do
  rairv = rair
  cpairv = cpair
  ksrftms = 0._r8; dragblj = 0._r8; dse_top = 0._r8
  tauresx = 0._r8; tauresy = 0._r8
  qmincg(1) = 1.e-12_r8
  cflx(:,1) = qflx
  uo = u; vo = v; qo(:,:,1) = q
  so(:ncol,:) = cpair*t(:ncol,:) + g*zm(:ncol,:)
  call compute_vdiff(1, pcols, pver, 1, ncol, tint, &
       p, t, rhoi, dt, taux, &
       tauy, shflx, cflx, &
       kvh, kvm, kvq, cgs, cgh, &
       zi, ksrftms, dragblj, &
       qmincg, fieldlist, fieldlistm, &
       uo, vo, qo, so, &
       tautmsx, tautmsy, dtk, topflx, errstring, &
       tauresx, tauresy, 1, cpairv, dse_top, &
       .false., .false.)
  if (len_trim(errstring) > 0) then
     write(0,*) trim(errstring); stop 'compute_vdiff failed'
  end if

  open(20, file=trim(fout), status='replace')
  do i = 1, ncol
     write(20,'(a,i4)') 'COL', i
     write(20,'(5es26.17)') ustar(i), obklen(i), pblh(i), tpert(i), qpert(i)
     write(20,'(*(es26.17))') kvm(i,:)
     write(20,'(*(es26.17))') kvh(i,:)
     write(20,'(*(es26.17))') kvq(i,:)
     write(20,'(*(es26.17))') cgh(i,:)
     write(20,'(*(es26.17))') cgs(i,:)
     write(20,'(*(es26.17))') tke(i,:)
     write(20,'(*(es26.17))') ri(i,1:pver-1)
     write(20,'(*(es26.17))') uo(i,:)
     write(20,'(*(es26.17))') vo(i,:)
     write(20,'(*(es26.17))') so(i,:)
     write(20,'(*(es26.17))') qo(i,:,1)
     write(20,'(*(es26.17))') dtk(i,:)
  end do
  close(20)
  write(*,*) 'ORACLE_OK ', ncol
end program hb_driver
