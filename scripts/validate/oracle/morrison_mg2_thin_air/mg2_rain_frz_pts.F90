! Point oracle for the CAM6 MG2 post-sedimentation rain freezing at -40 C
! (micro_mg2_0.F90:2571-2610, verbatim, one level per point): rain dumr, dumnr
! (the grid pools of :2511-2512, :2527) freezes below rainfrze where
! t + tlat/cpp*deltat says so, heat-limited, into snow or, where the mean drop
! 1/lamr of size_dist_param_basic(mg_rain_props, dumr, dumnr) is below Dcs,
! into cloud ice.  Calls the UNMODIFIED micro_mg_utils.F90 size_dist_param_basic
! (cam_cesm2_1_rel_60); the qrtend/nrtend/qstend/nstend/qitend/nitend updates are
! kept as rates, frzrdttot is left out.
!
! Input (first line: npts deltat, then one line per point):
!   t tlat dumr dumnr   (K, W/kg, kg/kg, #/kg)
! Output (one line per point): qrfrz nrfrz qsfrz nsfrz qifrz nifrz lamr tlat
!   (the rain loss and the snow and ice gains [kg/kg/s, #/kg/s], the rain
!   slope [1/m], tlat after the freezing [W/kg])
! usage: mg2_rain_frz_pts pts_in.txt pts_out.txt dcs
program mg2_rain_frz_pts
  use micro_mg_utils, only: r8, mg_rain_props, size_dist_param_basic
  use micro_mg2_0, only: micro_mg_init
  use wv_sat_methods, only: wv_sat_methods_init
  implicit none
  real(r8), parameter :: g = 9.80616_r8, rair = 287.05_r8, rh2o = 461.51_r8
  real(r8), parameter :: cpair = 1004.64_r8, tmelt = 273.15_r8
  real(r8), parameter :: latvap = 2.501e6_r8, latice = 3.337e5_r8
  integer :: npts, i, k
  character(len=512) :: fin, fout, adcs
  character(128) :: err
  character(len=16) :: pfm = 'in_cloud'
  real(r8) :: Dcs, deltat, cpp, xlf, rainfrze, dum, dum1
  real(r8), allocatable, dimension(:,:) :: t, tlat, dumr, dumnr, lamr, qrtend, nrtend, &
       qstend, nstend, qitend, nitend
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, adcs); read(adcs, *) Dcs
  call wv_sat_methods_init(r8, tmelt, 273.16_r8, 373.16_r8, 20._r8, rair/rh2o, err)
  call micro_mg_init(kind=r8, gravit=g, rair=rair, rh2o=rh2o, cpair=cpair, &
       tmelt_in=tmelt, latvap=latvap, latice=latice, rhmini_in=0.8_r8, &
       micro_mg_dcs=Dcs, microp_uniform_in=.false., do_cldice_in=.true., &
       use_hetfrz_classnuc_in=.true., micro_mg_precip_frac_method_in=pfm, &
       micro_mg_berg_eff_factor_in=1._r8, allow_sed_supersat_in=.false., &
       do_sb_physics_in=.false., nccons_in=.false., nicons_in=.false., &
       ncnst_in=1.e8_r8, ninst_in=0._r8, errstring=err)
  ! micro_mg2_0.F90:300, :316, :331 (cpp = cpair, xlf = latice, rainfrze = tmelt - 40)
  cpp = cpair
  xlf = latice
  rainfrze = tmelt - 40._r8
  open(10, file=trim(fin), status='old'); read(10, *) npts, deltat
  allocate(t(npts,1), tlat(npts,1), dumr(npts,1), dumnr(npts,1), lamr(npts,1), &
       qrtend(npts,1), nrtend(npts,1), qstend(npts,1), nstend(npts,1), qitend(npts,1), &
       nitend(npts,1))
  do i = 1, npts
     read(10, *) t(i,1), tlat(i,1), dumr(i,1), dumnr(i,1)
  end do
  close(10)
  qrtend = 0._r8; nrtend = 0._r8; qstend = 0._r8; nstend = 0._r8
  qitend = 0._r8; nitend = 0._r8; lamr = 0._r8
  k = 1
  do i = 1, npts
        ! verbatim micro_mg2_0.F90:2574-2608 (frzrdttot left out)
        if (t(i,k)+tlat(i,k)/cpp*deltat < rainfrze) then

           if (dumr(i,k) > 0._r8) then

              ! make sure freezing rain doesn't increase temperature above threshold
              dum = xlf/cpp*dumr(i,k)
              if (t(i,k)+tlat(i,k)/cpp*deltat+dum.gt.rainfrze) then
                 dum = -(t(i,k)+tlat(i,k)/cpp*deltat-rainfrze)*cpp/xlf
                 dum = dum/dumr(i,k)
                 dum = max(0._r8,dum)
                 dum = min(1._r8,dum)
              else
                 dum = 1._r8
              end if

              qrtend(i,k)=qrtend(i,k)-dum*dumr(i,k)/deltat
              nrtend(i,k)=nrtend(i,k)-dum*dumnr(i,k)/deltat

              ! get mean size of rain = 1/lamr, add frozen rain to either snow or cloud ice
              ! depending on mean rain size

              call size_dist_param_basic(mg_rain_props, dumr(i,k), dumnr(i,k), &
                   lamr(i,k))

              if (lamr(i,k) < 1._r8/Dcs) then
                 qstend(i,k)=qstend(i,k)+dum*dumr(i,k)/deltat
                 nstend(i,k)=nstend(i,k)+dum*dumnr(i,k)/deltat
              else
                 qitend(i,k)=qitend(i,k)+dum*dumr(i,k)/deltat
                 nitend(i,k)=nitend(i,k)+dum*dumnr(i,k)/deltat
              end if

              ! heating tendency
              dum1 = xlf*dum*dumr(i,k)/deltat
              tlat(i,k)=tlat(i,k)+dum1

           end if
        end if
  end do
  open(11, file=trim(fout), status='replace')
  do i = 1, npts
     write(11, '(8(1x,es25.16e3))') -qrtend(i,1), -nrtend(i,1), qstend(i,1), nstend(i,1), &
          qitend(i,1), nitend(i,1), lamr(i,1), tlat(i,1)
  end do
  close(11)
  print *, 'MG2_RAIN_FRZ_OK npts=', npts
end program mg2_rain_frz_pts
