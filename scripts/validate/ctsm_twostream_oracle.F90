program ts_oracle
  ! Build: gfortran -ffree-line-length-none ctsm_twostream_oracle.F90 -o o && ./o < ctsm_twostream_oracle_cases.txt
  ! Input row: elai esai f_leaf(unused) xl coszen fcansno rho_vis tau_vis rho_nir tau_nir alb_g_vis alb_g_nir
  ! Output row: albd_vis albi_vis albd_nir albi_nir.  Feeds tests/land/unit/test_canopy_snow_albedo.py::FORTRAN.
  ! Only edit: xl(patch%itype(p)) -> xl(p) (one patch).
  ! Executes the VERBATIM CTSM 5.1 SurfaceAlbedoMod.F90 TwoStream lines
  ! 1320-1338 (preamble) and 1381-1518 (per-band albedo) on one patch.
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: numrad = 2
  real(r8) :: omegas(numrad) = (/0.8_r8, 0.4_r8/), betads = 0.5_r8, betais = 0.5_r8
  real(r8) :: tfrz = 273.15_r8
  logical :: lSFonly = .false., snowveg_affects_radiation = .true.
  integer :: p, c, ib, ncase, k
  real(r8) :: coszen(1), xl(1), chil(1), gdir(1), twostext(1), avmu(1), temp0(1), temp2(1)
  real(r8) :: elai(1), esai(1), fcansno(1), fwet(1), t_veg(1)
  real(r8) :: rho(1,numrad), tau(1,numrad), omega(1,numrad)
  real(r8) :: albgrd(1,numrad), albgri(1,numrad), albsod(1,numrad), albsoi(1,numrad)
  real(r8) :: albd(1,numrad), albi(1,numrad), ftid(1,numrad), ftdd(1,numrad), fabd(1,numrad)
  real(r8) :: albdSF(1,numrad), albiSF(1,numrad), ftii(1,numrad), fabi(1,numrad)
  real(r8) :: fabd_sun(1,numrad), fabd_sha(1,numrad), fabi_sun(1,numrad), fabi_sha(1,numrad)
  real(r8) :: cosz, phi1, phi2, temp1, omegal, asu, betadl, betail, tmp0, tmp1, tmp2, tmp3, tmp4
  real(r8) :: tmp5, tmp6, tmp7, tmp8, tmp9, betad, betai, b, c1, d, f, h, sigma, p1, p2, p3, p4
  real(r8) :: t1, s1, s2, u1, u2, u3, d1, d2, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, a1, a2
  real(r8) :: rl, tl, rs, ts, fl
  p = 1; c = 1; fwet = 0._r8; t_veg = 260._r8
  read(*,*) ncase
  do k = 1, ncase
    read(*,*) elai(1), esai(1), fl, xl(1), coszen(1), fcansno(1), &
              rho(1,1), tau(1,1), rho(1,2), tau(1,2), albgrd(1,1), albgrd(1,2)
    albgri = albgrd; albsod = albgrd; albsoi = albgrd
    ! ---- CTSM lines 1320-1336 ----
       ! note that the following limit only acts on cosz values > 0 and less than
       ! 0.001, not on values cosz = 0, since these zero have already been filtered
       ! out in filter_vegsol
       cosz = max(0.001_r8, coszen(p))

       chil(p) = min( max(xl(p), -0.4_r8), 0.6_r8 )
       if (abs(chil(p)) <= 0.01_r8) chil(p) = 0.01_r8
       phi1 = 0.5_r8 - 0.633_r8*chil(p) - 0.330_r8*chil(p)*chil(p)
       phi2 = 0.877_r8 * (1._r8-2._r8*phi1)
       gdir(p) = phi1 + phi2*cosz
       twostext(p) = gdir(p)/cosz
       avmu(p) = ( 1._r8 - phi1/phi2 * log((phi1+phi2)/phi1) ) / phi2
       ! Restrict this calculation of temp0. We have seen cases where small temp0
       ! can cause unrealistic single scattering albedo (asu) associated with the
       ! log calculation in temp2 below, thereby eventually causing a negative soil albedo
       ! See bugzilla bug 2431: http://bugs.cgd.ucar.edu/show_bug.cgi?id=2431
       temp0(p) = max(gdir(p) + phi2*cosz,1.e-6_r8)
       temp1 = phi1*cosz
       temp2(p) = ( 1._r8 - temp1/temp0(p) * log((temp1+temp0(p))/temp1) )
    do ib = 1, numrad
    ! ---- CTSM lines 1381-1518 ----
          omegal = rho(p,ib) + tau(p,ib)
          asu = 0.5_r8*omegal*gdir(p)/temp0(p) *temp2(p)
          betadl = (1._r8+avmu(p)*twostext(p))/(omegal*avmu(p)*twostext(p))*asu
          betail = 0.5_r8 * ((rho(p,ib)+tau(p,ib)) + (rho(p,ib)-tau(p,ib)) &
                 * ((1._r8+chil(p))/2._r8)**2) / omegal

          if ( lSFonly .or. ( (.not. snowveg_affects_radiation) .and. (t_veg(p) > tfrz) ) ) then
             ! Keep omega, betad, and betai as they are (for Snow free case or
             ! when there is no snow
             tmp0 = omegal
             tmp1 = betadl
             tmp2 = betail
          else
             ! Adjust omega, betad, and betai for intercepted snow
             if (snowveg_affects_radiation) then
                tmp0 =   (1._r8-fcansno(p))*omegal        + fcansno(p)*omegas(ib)
                tmp1 = ( (1._r8-fcansno(p))*omegal*betadl + fcansno(p)*omegas(ib)*betads ) / tmp0
                tmp2 = ( (1._r8-fcansno(p))*omegal*betail + fcansno(p)*omegas(ib)*betais ) / tmp0
             else
                tmp0 =   (1._r8-fwet(p))*omegal        + fwet(p)*omegas(ib)
                tmp1 = ( (1._r8-fwet(p))*omegal*betadl + fwet(p)*omegas(ib)*betads ) / tmp0
                tmp2 = ( (1._r8-fwet(p))*omegal*betail + fwet(p)*omegas(ib)*betais ) / tmp0
             end if
          end if  ! end Snow free

          omega(p,ib) = tmp0
          betad = tmp1
          betai = tmp2

          ! Common terms

          b = 1._r8 - omega(p,ib) + omega(p,ib)*betai
          c1 = omega(p,ib)*betai
          tmp0 = avmu(p)*twostext(p)
          d = tmp0 * omega(p,ib)*betad
          f = tmp0 * omega(p,ib)*(1._r8-betad)
          tmp1 = b*b - c1*c1
          h = sqrt(tmp1) / avmu(p)
          sigma = tmp0*tmp0 - tmp1
          p1 = b + avmu(p)*h
          p2 = b - avmu(p)*h
          p3 = b + tmp0
          p4 = b - tmp0

          ! Absorbed, reflected, transmitted fluxes per unit incoming radiation
          ! for full canopy

          t1 = min(h*(elai(p)+esai(p)), 40._r8)
          s1 = exp(-t1)
          t1 = min(twostext(p)*(elai(p)+esai(p)), 40._r8)
          s2 = exp(-t1)

          ! Direct beam
          if ( .not. lSFonly )then
             u1 = b - c1/albgrd(c,ib)
             u2 = b - c1*albgrd(c,ib)
             u3 = f + c1*albgrd(c,ib)
          else
             ! Snow Free (SF) only 
             ! albsod instead of albgrd here:
             u1 = b - c1/albsod(c,ib)
             u2 = b - c1*albsod(c,ib)
             u3 = f + c1*albsod(c,ib)
          end if
          tmp2 = u1 - avmu(p)*h
          tmp3 = u1 + avmu(p)*h
          d1 = p1*tmp2/s1 - p2*tmp3*s1
          tmp4 = u2 + avmu(p)*h
          tmp5 = u2 - avmu(p)*h
          d2 = tmp4/s1 - tmp5*s1
          h1 = -d*p4 - c1*f
          tmp6 = d - h1*p3/sigma
          tmp7 = ( d - c1 - h1/sigma*(u1+tmp0) ) * s2
          h2 = ( tmp6*tmp2/s1 - p2*tmp7 ) / d1
          h3 = - ( tmp6*tmp3*s1 - p1*tmp7 ) / d1
          h4 = -f*p3 - c1*d
          tmp8 = h4/sigma
          tmp9 = ( u3 - tmp8*(u2-tmp0) ) * s2
          h5 = - ( tmp8*tmp4/s1 + tmp9 ) / d2
          h6 = ( tmp8*tmp5*s1 + tmp9 ) / d2
          if ( .not. lSFonly )then
            albd(p,ib) = h1/sigma + h2 + h3
            ftid(p,ib) = h4*s2/sigma + h5*s1 + h6/s1
            ftdd(p,ib) = s2
            fabd(p,ib) = 1._r8 - albd(p,ib) - (1._r8-albgrd(c,ib))*ftdd(p,ib) - (1._r8-albgri(c,ib))*ftid(p,ib)
          else
            albdSF(p,ib) = h1/sigma + h2 + h3
          end if
          

          a1 = h1 / sigma * (1._r8 - s2*s2) / (2._r8 * twostext(p)) &
             + h2         * (1._r8 - s2*s1) / (twostext(p) + h) &
             + h3         * (1._r8 - s2/s1) / (twostext(p) - h)

          a2 = h4 / sigma * (1._r8 - s2*s2) / (2._r8 * twostext(p)) &
             + h5         * (1._r8 - s2*s1) / (twostext(p) + h) &
             + h6         * (1._r8 - s2/s1) / (twostext(p) - h)
          if ( .not. lSFonly )then
            fabd_sun(p,ib) = (1._r8 - omega(p,ib)) * ( 1._r8 - s2 + 1._r8 / avmu(p) * (a1 + a2) )
            fabd_sha(p,ib) = fabd(p,ib) - fabd_sun(p,ib)
          end if

          ! Diffuse
          if ( .not. lSFonly )then
            u1 = b - c1/albgri(c,ib)
            u2 = b - c1*albgri(c,ib)
          else
             ! Snow Free (SF) only 
             ! albsoi instead of albgri here:
            u1 = b - c1/albsoi(c,ib)
            u2 = b - c1*albsoi(c,ib)
          end if
          tmp2 = u1 - avmu(p)*h
          tmp3 = u1 + avmu(p)*h
          d1 = p1*tmp2/s1 - p2*tmp3*s1
          tmp4 = u2 + avmu(p)*h
          tmp5 = u2 - avmu(p)*h
          d2 = tmp4/s1 - tmp5*s1
          h7 = (c1*tmp2) / (d1*s1)
          h8 = (-c1*tmp3*s1) / d1
          h9 = tmp4 / (d2*s1)
          h10 = (-tmp5*s1) / d2

  
          ! Final Snow Free albedo
          if ( lSFonly )then
            albiSF(p,ib) = h7 + h8
          else
            ! For non snow Free case, adjustments continue
            albi(p,ib) = h7 + h8
            ftii(p,ib) = h9*s1 + h10/s1
            fabi(p,ib) = 1._r8 - albi(p,ib) - (1._r8-albgri(c,ib))*ftii(p,ib)

            a1 = h7 * (1._r8 - s2*s1) / (twostext(p) + h) +  h8 * (1._r8 - s2/s1) / (twostext(p) - h)
            a2 = h9 * (1._r8 - s2*s1) / (twostext(p) + h) + h10 * (1._r8 - s2/s1) / (twostext(p) - h)

            fabi_sun(p,ib) = (1._r8 - omega(p,ib)) / avmu(p) * (a1 + a2)
            fabi_sha(p,ib) = fabi(p,ib) - fabi_sun(p,ib)
          end if
    end do
    write(*,'(4es25.16)') albd(1,1), albi(1,1), albd(1,2), albi(1,2)
  end do
end program ts_oracle
