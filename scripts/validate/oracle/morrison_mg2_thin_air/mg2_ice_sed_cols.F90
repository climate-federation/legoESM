! Column oracle for the CAM6 MG2 cloud-ice sedimentation in micro_mg2_0.F90
! order: the ice is sized AND transported from the post-process state
! qi + qitend*deltat (:2063-2064, :2088-2089, :2119-2147, :2206-2214), with the
! sub-step count, the clear-air sublimation of ice falling into a smaller
! cloud fraction and the surface ice flux of :2232-2314.  Calls the
! UNMODIFIED micro_mg_utils.F90 size_dist_param_basic (cam_cesm2_1_rel_60);
! the rhosu, rhof, ain, ajn lines (:326, :895, :900-901), the fall-speed block
! and the ice part of the sedimentation loop are copied verbatim, the column
! index kept and the cloud-water, rain and snow parts left out.
!
! Input (first line: ncol nlev deltat, then per column nlev lines, top first):
!   qi ni qitend nitend icldm rho pdel     ([kg/kg], [#/kg], per second, -, kg/m3, Pa)
! Output (per column nlev lines): qitend nitend qisedten qvlat tlat
! then one line per column: preci nstep   (preci in m/s as MG2 reports it).
! usage: mg2_ice_sed_cols cols_in.txt cols_out.txt dcs
program mg2_ice_sed_cols
  use micro_mg_utils, only: r8, ai, bi, aj, bj, qsmall, mg_ice_props, size_dist_param_basic
  use micro_mg2_0, only: micro_mg_init
  use wv_sat_methods, only: wv_sat_methods_init
  implicit none
  real(r8), parameter :: g = 9.80616_r8, rair = 287.05_r8, rh2o = 461.51_r8
  real(r8), parameter :: cpair = 1004.64_r8, tmelt = 273.15_r8
  real(r8), parameter :: latvap = 2.501e6_r8, latice = 3.337e5_r8
  integer :: ncol, nlev, i, k, n, nstep
  character(len=512) :: fin, fout, adcs
  character(128) :: err
  character(len=16) :: pfm = 'in_cloud'
  real(r8) :: dcs, deltat, rhosu, irad, ifrac, xxls
  real(r8) :: dum1, faltndi, faltndni, faltndqie
  real(r8) :: gamma_bi_plus1, gamma_bi_plus4, gamma_bj_plus1, gamma_bj_plus4
  real(r8), allocatable, dimension(:,:) :: qi, ni, qitend, nitend, icldm, rho, pdel, pdel_inv, &
       dumi, dumni, lami, rhof, ain, ajn, vtrmi, fi, fni, qisedten, qvlat, tlat
  real(r8), allocatable :: falouti(:), faloutni(:), preci(:)
  integer, allocatable :: nsteps(:)
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
  ! micro_mg2_0.F90:317 (xxlv = latvap, xlf = latice as micro_mg_init sets them)
  xxls = latvap + latice
  ! verbatim micro_mg2_0.F90:341-344
  gamma_bi_plus1=gamma(1._r8+bi)
  gamma_bi_plus4=gamma(4._r8+bi)
  gamma_bj_plus1=gamma(1._r8+bj)
  gamma_bj_plus4=gamma(4._r8+bj)
  open(10, file=trim(fin), status='old'); read(10, *) ncol, nlev, deltat
  allocate(qi(ncol,nlev), ni(ncol,nlev), qitend(ncol,nlev), nitend(ncol,nlev), &
       icldm(ncol,nlev), rho(ncol,nlev), pdel(ncol,nlev), pdel_inv(ncol,nlev), &
       dumi(ncol,nlev), dumni(ncol,nlev), lami(ncol,nlev), rhof(ncol,nlev), &
       ain(ncol,nlev), ajn(ncol,nlev), vtrmi(ncol,nlev), fi(ncol,nlev), fni(ncol,nlev), &
       qisedten(ncol,nlev), qvlat(ncol,nlev), tlat(ncol,nlev))
  allocate(falouti(nlev), faloutni(nlev), preci(ncol), nsteps(ncol))
  do i = 1, ncol
     do k = 1, nlev
        read(10, *) qi(i,k), ni(i,k), qitend(i,k), nitend(i,k), icldm(i,k), rho(i,k), pdel(i,k)
     end do
  end do
  close(10)
  qisedten = 0._r8; qvlat = 0._r8; tlat = 0._r8; preci = 0._r8
  ! verbatim micro_mg2_0.F90:326, :895, :900-901
  rhosu = 85000._r8/(rair * tmelt)
  rhof=(rhosu/rho)**0.54_r8
  ain=ai*(rhosu/rho)**0.35_r8
  ajn=aj*(rhosu/rho)**0.35_r8
  ! verbatim micro_mg2_0.F90:2063-2064 (ice lines)
  do k=1,nlev
     do i=1,ncol
        dumi(i,k) = (qi(i,k)+qitend(i,k)*deltat)/icldm(i,k)
        dumni(i,k) = max((ni(i,k)+nitend(i,k)*deltat)/icldm(i,k),0._r8)
     enddo
  enddo
  ! micro_mg2_0.F90:2088-2089
  do k=1,nlev
     call size_dist_param_basic(mg_ice_props, dumi(:,k), dumni(:,k), &
          lami(:,k), ncol)
  enddo
  ! verbatim micro_mg2_0.F90:2119-2147
  do k=1,nlev
     do i=1,ncol
        if (dumi(i,k).ge.qsmall) then

           vtrmi(i,k)=min(ain(i,k)*gamma_bi_plus4/(6._r8*lami(i,k)**bi), &
                1.2_r8*rhof(i,k))

           fi(i,k) = g*rho(i,k)*vtrmi(i,k)
           fni(i,k) = g*rho(i,k)* &
                min(ain(i,k)*gamma_bi_plus1/lami(i,k)**bi,1.2_r8*rhof(i,k))

           ! adjust the ice fall velocity for smaller (r < 20 um) ice
           ! particles (blend over 18-20 um)
           irad = 1.5_r8 / lami(i,k) * 1e6_r8
           ifrac = min(1._r8, max(0._r8, (irad - 18._r8) / 2._r8))

           if (ifrac .lt. 1._r8) then
              vtrmi(i,k) = ifrac * vtrmi(i,k) + &
                 (1._r8 - ifrac) * &
                 min(ajn(i,k)*gamma_bj_plus4/(6._r8*lami(i,k)**bj), &
                 1.2_r8*rhof(i,k))

              fi(i,k) = g*rho(i,k)*vtrmi(i,k)
              fni(i,k) = ifrac * fni(i,k) + &
                 (1._r8 - ifrac) * &
                 g*rho(i,k)* &
                 min(ajn(i,k)*gamma_bj_plus1/lami(i,k)**bj,1.2_r8*rhof(i,k))
           end if
        else
           fi(i,k) = 0._r8
           fni(i,k)= 0._r8
        end if
     enddo
  enddo
  ! verbatim micro_mg2_0.F90:2206-2207, :2213 (ice lines)
  do k=1,nlev
     do i=1,ncol
        dumi(i,k) = (qi(i,k)+qitend(i,k)*deltat)
        dumni(i,k) = max((ni(i,k)+nitend(i,k)*deltat),0._r8)
        if (dumi(i,k).lt.qsmall) dumni(i,k)=0._r8
     enddo
  enddo
  ! verbatim micro_mg2_0.F90:2218-2222
  do k=1,nlev
     do i=1,ncol
       pdel_inv(i,k) = 1._r8/pdel(i,k)
     enddo
  enddo
  ! verbatim micro_mg2_0.F90:2232-2314, ice only (do_cldice = .true.)
  do i=1,ncol
     nstep = 1 + int(max( &
          maxval( fi(i,:)*pdel_inv(i,:)), &
          maxval(fni(i,:)*pdel_inv(i,:))) &
          * deltat)
     nsteps(i) = nstep

     do n = 1,nstep

        falouti  = fi(i,:)  * dumi(i,:)
        faloutni = fni(i,:) * dumni(i,:)

        ! top of model

        k = 1

        ! add fallout terms to microphysical tendencies
        faltndi = falouti(k)/pdel(i,k)
        faltndni = faloutni(k)/pdel(i,k)
        qitend(i,k) = qitend(i,k)-faltndi/nstep
        nitend(i,k) = nitend(i,k)-faltndni/nstep

        ! sedimentation tendency for output
        qisedten(i,k)=qisedten(i,k)-faltndi/nstep

        dumi(i,k) = dumi(i,k)-faltndi*deltat/nstep
        dumni(i,k) = dumni(i,k)-faltndni*deltat/nstep

        do k = 2,nlev

           dum1=icldm(i,k)/icldm(i,k-1)
           dum1=min(dum1,1._r8)

           faltndqie=(falouti(k)-falouti(k-1))/pdel(i,k)
           faltndi=(falouti(k)-dum1*falouti(k-1))/pdel(i,k)
           faltndni=(faloutni(k)-dum1*faloutni(k-1))/pdel(i,k)

           ! add fallout terms to eulerian tendencies

           qitend(i,k) = qitend(i,k)-faltndi/nstep
           nitend(i,k) = nitend(i,k)-faltndni/nstep

           ! sedimentation tendency for output
           qisedten(i,k)=qisedten(i,k)-faltndi/nstep

           ! add terms to to evap/sub of cloud water

           qvlat(i,k)=qvlat(i,k)-(faltndqie-faltndi)/nstep

           tlat(i,k)=tlat(i,k)+(faltndqie-faltndi)*xxls/nstep

           dumi(i,k) = dumi(i,k)-faltndi*deltat/nstep
           dumni(i,k) = dumni(i,k)-faltndni*deltat/nstep

        end do

        preci(i) = preci(i)+falouti(nlev)/g/real(nstep)/1000._r8

     end do
  end do
  open(11, file=trim(fout), status='replace')
  do i = 1, ncol
     do k = 1, nlev
        write(11, '(5(1x,es25.16e3))') qitend(i,k), nitend(i,k), qisedten(i,k), qvlat(i,k), tlat(i,k)
     end do
  end do
  do i = 1, ncol
     write(11, '(1x,es25.16e3,1x,i8)') preci(i), nsteps(i)
  end do
  close(11)
  print *, 'MG2_ICE_SED_OK ncol=', ncol, ' nlev=', nlev
end program mg2_ice_sed_cols
