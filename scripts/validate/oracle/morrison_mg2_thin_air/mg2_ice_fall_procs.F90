! Point oracle for the CAM6 MG2 cloud-ice fall speeds of the post-process
! state.  Calls the UNMODIFIED micro_mg_utils.F90 size_dist_param_basic
! (cam_cesm2_1_rel_60) as micro_mg2_0.F90:2087-2089 does; the rhosu, rhof,
! ain, ajn lines (micro_mg2_0.F90:326, :895, :900-901) and the fall-speed
! block (:2119-2147) are copied verbatim.
!
! Point input (first line n, then one line per point):
!   T p rho dumi dumni      (in-cloud ice mass [kg/kg] and number [#/kg])
! Output (one line per point, 6 columns):
!   dumni_adj lami vtrmi vni rhof ain
! where vni = fni/(g rho) is the number-weighted speed.
! usage: mg2_ice_fall_procs points.txt out.txt dcs
program mg2_ice_fall_procs
  use micro_mg_utils, only: r8, ai, bi, aj, bj, qsmall, mg_ice_props, size_dist_param_basic
  use micro_mg2_0, only: micro_mg_init
  use wv_sat_methods, only: wv_sat_methods_init
  implicit none
  real(r8), parameter :: g = 9.80616_r8, rair = 287.05_r8, rh2o = 461.51_r8
  real(r8), parameter :: cpair = 1004.64_r8, tmelt = 273.15_r8
  real(r8), parameter :: latvap = 2.501e6_r8, latice = 3.337e5_r8
  integer :: n, i
  character(len=512) :: fin, fout, adcs
  character(128) :: err
  character(len=16) :: pfm = 'in_cloud'
  real(r8) :: dcs, rhosu, irad, ifrac
  real(r8) :: gamma_bi_plus1, gamma_bi_plus4, gamma_bj_plus1, gamma_bj_plus4
  real(r8), allocatable, dimension(:) :: t, p, rho, dumi, dumni, rhof, ain, ajn, lami, &
       vtrmi, fi, fni
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, adcs); read(adcs, *) dcs
  call wv_sat_methods_init(r8, tmelt, 273.16_r8, 373.16_r8, 20._r8, rair/rh2o, err)
  call micro_mg_init(kind=r8, gravit=g, rair=rair, rh2o=rh2o, cpair=cpair, &
       tmelt_in=tmelt, latvap=latvap, latice=latice, rhmini_in=0.8_r8, &
       micro_mg_dcs=dcs, microp_uniform_in=.false., do_cldice_in=.true., &
       use_hetfrz_classnuc_in=.true., micro_mg_precip_frac_method_in=pfm, &
       micro_mg_berg_eff_factor_in=1._r8, allow_sed_supersat_in=.false., &
       do_sb_physics_in=.false., nccons_in=.false., nicons_in=.false., &
       ncnst_in=1.e8_r8, ninst_in=0._r8, errstring=err)
  ! verbatim micro_mg2_0.F90:341-344
  gamma_bi_plus1=gamma(1._r8+bi)
  gamma_bi_plus4=gamma(4._r8+bi)
  gamma_bj_plus1=gamma(1._r8+bj)
  gamma_bj_plus4=gamma(4._r8+bj)
  open(10, file=trim(fin), status='old'); read(10, *) n
  allocate(t(n), p(n), rho(n), dumi(n), dumni(n), rhof(n), ain(n), ajn(n), lami(n), &
       vtrmi(n), fi(n), fni(n))
  do i = 1, n
     read(10, *) t(i), p(i), rho(i), dumi(i), dumni(i)
  end do
  close(10)
  ! verbatim micro_mg2_0.F90:326, :895, :900-901
  rhosu = 85000._r8/(rair * tmelt)
  rhof=(rhosu/rho)**0.54_r8
  ain=ai*(rhosu/rho)**0.35_r8
  ajn=aj*(rhosu/rho)**0.35_r8
  ! micro_mg2_0.F90:2088-2089
  call size_dist_param_basic(mg_ice_props, dumi, dumni, lami, n)
  vtrmi = 0._r8
  do i = 1, n
     ! verbatim micro_mg2_0.F90:2121-2147 (column index dropped)
        if (dumi(i).ge.qsmall) then

           vtrmi(i)=min(ain(i)*gamma_bi_plus4/(6._r8*lami(i)**bi), &
                1.2_r8*rhof(i))

           fi(i) = g*rho(i)*vtrmi(i)
           fni(i) = g*rho(i)* &
                min(ain(i)*gamma_bi_plus1/lami(i)**bi,1.2_r8*rhof(i))

           ! adjust the ice fall velocity for smaller (r < 20 um) ice
           ! particles (blend over 18-20 um)
           irad = 1.5_r8 / lami(i) * 1e6_r8
           ifrac = min(1._r8, max(0._r8, (irad - 18._r8) / 2._r8))

           if (ifrac .lt. 1._r8) then
              vtrmi(i) = ifrac * vtrmi(i) + &
                 (1._r8 - ifrac) * &
                 min(ajn(i)*gamma_bj_plus4/(6._r8*lami(i)**bj), &
                 1.2_r8*rhof(i))

              fi(i) = g*rho(i)*vtrmi(i)
              fni(i) = ifrac * fni(i) + &
                 (1._r8 - ifrac) * &
                 g*rho(i)* &
                 min(ajn(i)*gamma_bj_plus1/lami(i)**bj,1.2_r8*rhof(i))
           end if
        else
           fi(i) = 0._r8
           fni(i)= 0._r8
        end if
  end do
  open(11, file=trim(fout), status='replace')
  do i = 1, n
     write(11, '(6(1x,es25.16e3))') dumni(i), lami(i), vtrmi(i), fni(i)/(g*rho(i)), rhof(i), ain(i)
  end do
  close(11)
  print *, 'MG2_ICE_FALL_OK n=', n
end program mg2_ice_fall_procs
