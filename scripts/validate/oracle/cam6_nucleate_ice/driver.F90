! Oracle driver for CAM6 cirrus ice nucleation (cam_cesm2_1_rel_60).
! Calls the UNMODIFIED nucleate_ice.F90 (nucleati_init, nucleati) and transcribes, line for line, the
! per-level loop of nucleate_ice_cam.F90::nucleate_ice_cam_calc for the modal aerosol without a separate
! dust mode (:597-778): relhum from qsat_water (:603-611), icldm (:613), the aerosol numbers (:623-690),
! soot_num = 0 (:694), the nucleati call (:696-703) and the stratospheric sulfate term (:718-728).
! CAM6 namelist: use_preexisting_ice, use_hetfrz_classnuc true, nucleate_ice_incloud false, subgrid 1.2
! everywhere, nucleate_ice_strat 1, nucleate_ice_use_troplev true, mincld 1e-4.
! Input (text): ncol nlev top_lev(1-based) / tmelt rair epsilo h2otrip tboil pi / then per column: troplev
! (1-based) and 13 rows of nlev values: t qn pmid qc qi ni ast wsubi num_accum num_aitken num_coarse
! coarse_dust coarse_nacl coarse_so4. Output: naai (1/kg) per column row, then nihf and oso4 rows.
program nucleate_ice_driver
  use wv_sat_methods, only: wv_sat_methods_init, wv_sat_qsat_water
  use nucleate_ice,   only: nucleati_init, nucleati
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  character(len=512) :: fin, fout, errstring
  integer :: ncol, nlev, top_lev, i, k, iu, ou
  real(r8) :: tmelt, rair, epsilo, h2otrip, tboil, pi
  real(r8), parameter :: mincld = 0.0001_r8, subgrid = 1.2_r8, nucleate_ice_strat = 1.0_r8
  real(r8), allocatable :: t(:), qn(:), pmid(:), qc(:), qi(:), ni(:), ast(:), wsubi(:)
  real(r8), allocatable :: num_accum(:), num_aitken(:), num_coarse(:), coarse_dust(:), coarse_nacl(:), coarse_so4(:)
  real(r8), allocatable :: naai(:), nihf(:), oso4o(:)
  integer :: troplev
  real(r8) :: rho, es, qs, relhum, icldm, so4_num, soot_num, dst_num, dmc, ssmc, so4mc, wght, so4_num_cr
  real(r8) :: so4_num_ac, dso4_num, oniimm, onidep, onimey, wice, weff, fhom, regm, oso4_num, odst_num, osoot_num
  real(r8) :: onihf

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=iu, file=trim(fin), status='old', action='read')
  read(iu,*) ncol, nlev, top_lev
  read(iu,*) tmelt, rair, epsilo, h2otrip, tboil, pi
  call wv_sat_methods_init(r8, tmelt, h2otrip, tboil, 20._r8, epsilo, errstring)
  call nucleati_init(.true., .true., .false., 6, pi, mincld)
  allocate(t(nlev), qn(nlev), pmid(nlev), qc(nlev), qi(nlev), ni(nlev), ast(nlev), wsubi(nlev), &
       num_accum(nlev), num_aitken(nlev), num_coarse(nlev), coarse_dust(nlev), coarse_nacl(nlev), &
       coarse_so4(nlev), naai(nlev), nihf(nlev), oso4o(nlev))
  open(newunit=ou, file=trim(fout), status='replace', action='write')
  do i = 1, ncol
     read(iu,*) troplev
     read(iu,*) t;  read(iu,*) qn; read(iu,*) pmid; read(iu,*) qc; read(iu,*) qi; read(iu,*) ni
     read(iu,*) ast; read(iu,*) wsubi; read(iu,*) num_accum; read(iu,*) num_aitken; read(iu,*) num_coarse
     read(iu,*) coarse_dust; read(iu,*) coarse_nacl; read(iu,*) coarse_so4
     naai = 0._r8; nihf = 0._r8; oso4o = 0._r8
     do k = top_lev, nlev
        rho = pmid(k)/(rair*t(k))
        call wv_sat_qsat_water(t(k), pmid(k), es, qs)
        relhum = qn(k)/qs
        icldm = max(ast(k), mincld)
        if (t(k) < tmelt - 5._r8) then
           soot_num = num_accum(k)*rho*1.0e-6_r8
           dmc  = coarse_dust(k)*rho
           ssmc = coarse_nacl(k)*rho
           if (dmc > 0._r8) then
              so4mc = coarse_so4(k)*rho
              wght = dmc/(ssmc + dmc + so4mc)
              dst_num = wght * num_coarse(k)*rho*1.0e-6_r8
           else
              dst_num = 0.0_r8
           end if
           so4mc = coarse_so4(k)*rho
           if (so4mc > 0._r8) then
              wght = so4mc/(ssmc + dmc + so4mc)
              so4_num_cr = wght * num_coarse(k)*rho*1.0e-6_r8
           else
              so4_num_cr = 0.0_r8
           end if
           so4_num = num_aitken(k)*rho*1.0e-6_r8
           soot_num = 0.0_r8
           call nucleati(wsubi(k), t(k), pmid(k), relhum, icldm, qc(k), qi(k), ni(k), rho, &
                so4_num, dst_num, soot_num, subgrid, naai(k), onihf, oniimm, onidep, onimey, &
                wice, weff, fhom, regm, oso4_num, odst_num, osoot_num)
           nihf(k) = onihf
           oso4o(k) = oso4_num
           if ((k < troplev) .and. (nucleate_ice_strat > 0._r8)) then
              if (oso4_num > 0._r8) then
                 so4_num_ac = num_accum(k)*rho*1.0e-6_r8
                 dso4_num = max(0._r8, (nucleate_ice_strat * (so4_num_cr + so4_num_ac)) - oso4_num) * 1e6_r8 / rho
                 naai(k) = naai(k) + dso4_num
                 nihf(k) = nihf(k) + dso4_num
              end if
           end if
        end if
     end do
     write(ou,'(*(ES26.17E3,1X))') naai
     write(ou,'(*(ES26.17E3,1X))') nihf
     write(ou,'(*(ES26.17E3,1X))') oso4o
  end do
  close(iu); close(ou)
end program nucleate_ice_driver
