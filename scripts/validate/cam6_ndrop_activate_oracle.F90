! CAM6 ndrop.F90 activate_modal + maxsat oracle (B7 substitutions), for
! tests/atmosphere/hydrostatic/unit/test_cam6_activation_oracle.py.
! Build: gfortran -O0 -ffree-line-length-none -fno-fast-math cam6_ndrop_activate_oracle.F90 -o o && ./o < cam6_ndrop_activate_oracle_cases.txt
! The two subroutines below are VERBATIM ndrop.F90:1294-1700 and :1704-1750 (CAM6 reference tree).
! Shim (the ONLY substitutions; user decision B7, 2026-10-16): physconst values = legoesm.constants
! (read from the first input line, printed by the generator with repr precision); qsat = legoesm.thermo
! Tetens + smooth softplus floor/cap (saturation_specific_humidity); erf = Fortran intrinsic
! (CAM: shr_spfn_erf); endrun aborts the oracle.  ndrop_init lines 113-125 / 153-156 reproduced in
! set_mode_constants / set_constants with the same expressions.
module oracle_shim
  use iso_c_binding, only: c_double
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: iulog = 6
  real(r8) :: pi, rhoh2o, mwh2o, r_universal, rh2o, rair, cpair, latvap, gravit, tmelt, epsilo
  real(r8) :: zero, third, twothird, sixth, sq2, sqpi, t0, surften, aten, alogaten, alog2, alog3
  real(r8), allocatable :: alogsig(:), exp45logsig(:), f1(:), f2(:)
  interface
    function c_log1p(x) bind(c, name='log1p')
      import :: c_double
      real(c_double), value :: x
      real(c_double) :: c_log1p
    end function c_log1p
  end interface
contains
  subroutine endrun(msg)
    character(len=*), intent(in) :: msg
    write(*,*) 'ENDRUN ', msg
    stop 2
  end subroutine endrun
  function softplus(x) result(y)          ! jax.nn.softplus = logaddexp(x, 0)
    real(r8), intent(in) :: x
    real(r8) :: y
    y = max(x, 0._r8) + c_log1p(exp(-abs(x)))
  end function softplus
  subroutine qsat(t, p, es, qs)           ! legoesm.thermo.saturation_specific_humidity
    real(r8), intent(in) :: t, p
    real(r8), intent(out) :: es, qs
    real(r8) :: tc, denom, w
    tc = max(t, 150._r8) - tmelt
    es = 611.2_r8*exp(17.67_r8*tc/(tc + 243.5_r8))
    denom = softplus(p - es - 1.0_r8) + 1.0_r8
    w = epsilo*es/denom
    w = 1.0_r8 - softplus(20.0_r8*(1.0_r8 - w))/20.0_r8
    qs = w/(1.0_r8 + w)
  end subroutine qsat
  subroutine set_constants()
    zero     = 0._r8
    third    = 1._r8/3._r8
    twothird = 2._r8*third
    sixth    = 1._r8/6._r8
    sq2      = sqrt(2._r8)
    sqpi     = sqrt(pi)
    aten     = 2._r8*mwh2o*surften/(r_universal*t0*rhoh2o)
    alogaten = log(aten)
    alog2    = log(2._r8)
    alog3    = log(3._r8)
  end subroutine set_constants
  subroutine set_mode_constants(nmode, sigmag)
    integer, intent(in) :: nmode
    real(r8), intent(in) :: sigmag(nmode)
    integer :: m
    if (allocated(alogsig)) deallocate(alogsig, exp45logsig, f1, f2)
    allocate(alogsig(nmode), exp45logsig(nmode), f1(nmode), f2(nmode))
    do m = 1, nmode
      alogsig(m)     = log(sigmag(m))
      exp45logsig(m) = exp(4.5_r8*alogsig(m)*alogsig(m))
      f1(m)          = 0.5_r8*exp(2.5_r8*alogsig(m)*alogsig(m))
      f2(m)          = 1._r8 + 0.25_r8*alogsig(m)
    end do
  end subroutine set_mode_constants

subroutine activate_modal(wbar, sigw, wdiab, wminf, wmaxf, tair, rhoair,  &
   na, nmode, volume, hygro,  &
   fn, fm, fluxn, fluxm, flux_fullact, smax_prescribed, in_cloud_in, smax_f)

   !      calculates number, surface, and mass fraction of aerosols activated as CCN
   !      calculates flux of cloud droplets, surface area, and aerosol mass into cloud
   !      assumes an internal mixture within each of up to nmode multiple aerosol modes
   !      a gaussiam spectrum of updrafts can be treated.

   !      mks units

   !      Abdul-Razzak and Ghan, A parameterization of aerosol activation.
   !      2. Multiple aerosol types. J. Geophys. Res., 105, 6837-6844.


   !      input

   real(r8), intent(in) :: wbar          ! grid cell mean vertical velocity (m/s)
   real(r8), intent(in) :: sigw          ! subgrid standard deviation of vertical vel (m/s)
   real(r8), intent(in) :: wdiab         ! diabatic vertical velocity (0 if adiabatic)
   real(r8), intent(in) :: wminf         ! minimum updraft velocity for integration (m/s)
   real(r8), intent(in) :: wmaxf         ! maximum updraft velocity for integration (m/s)
   real(r8), intent(in) :: tair          ! air temperature (K)
   real(r8), intent(in) :: rhoair        ! air density (kg/m3)
   real(r8), intent(in) :: na(:)      ! aerosol number concentration (/m3)
   integer,  intent(in) :: nmode      ! number of aerosol modes
   real(r8), intent(in) :: volume(:)  ! aerosol volume concentration (m3/m3)
   real(r8), intent(in) :: hygro(:)   ! hygroscopicity of aerosol mode

   !      output

   real(r8), intent(out) :: fn(:)      ! number fraction of aerosols activated
   real(r8), intent(out) :: fm(:)      ! mass fraction of aerosols activated
   real(r8), intent(out) :: fluxn(:)   ! flux of activated aerosol number fraction into cloud (cm/s)
   real(r8), intent(out) :: fluxm(:)   ! flux of activated aerosol mass fraction into cloud (cm/s)
   real(r8), intent(out) :: flux_fullact   ! flux of activated aerosol fraction assuming 100% activation (cm/s)
   !    rce-comment
   !    used for consistency check -- this should match (ekd(k)*zs(k))
   !    also, fluxm/flux_fullact gives fraction of aerosol mass flux
   !       that is activated
  
   !      optional
   real(r8), optional, intent(in) :: smax_prescribed  ! prescribed max. supersaturation for secondary activation
   logical,  optional, intent(in) :: in_cloud_in      ! switch to modify calculations when above cloud base
   real(r8), optional, intent(in) :: smax_f           ! droplet and rain size distr factor in the smax calculation
                                                      ! used when in_cloud=.true.

   !      local

   integer, parameter:: nx=200
   integer iquasisect_option, isectional
   real(r8) integ,integf
   real(r8), parameter :: p0 = 1013.25e2_r8    ! reference pressure (Pa)
   real(r8) xmin(nmode),xmax(nmode) ! ln(r) at section interfaces
   real(r8) volmin(nmode),volmax(nmode) ! volume at interfaces
   real(r8) tmass ! total aerosol mass concentration (g/cm3)
   real(r8) sign(nmode)    ! geometric standard deviation of size distribution
   real(r8) rm ! number mode radius of aerosol at max supersat (cm)
   real(r8) pres ! pressure (Pa)
   real(r8) path ! mean free path (m)
   real(r8) diff ! diffusivity (m2/s)
   real(r8) conduct ! thermal conductivity (Joule/m/sec/deg)
   real(r8) diff0,conduct0
   real(r8) es ! saturation vapor pressure
   real(r8) qs ! water vapor saturation mixing ratio
   real(r8) dqsdt ! change in qs with temperature
   real(r8) dqsdp ! change in qs with pressure
   real(r8) g ! thermodynamic function (m2/s)
   real(r8) zeta(nmode), eta(nmode)
   real(r8) lnsmax ! ln(smax)
   real(r8) alpha
   real(r8) gamma
   real(r8) beta
   real(r8) sqrtg
   real(r8) :: amcube(nmode) ! cube of dry mode radius (m)
   real(r8) :: smcrit(nmode) ! critical supersatuation for activation
   real(r8) :: lnsm(nmode) ! ln(smcrit)
   real(r8) smc(nmode) ! critical supersaturation for number mode radius
   real(r8) sumflx_fullact
   real(r8) sumflxn(nmode)
   real(r8) sumflxm(nmode)
   real(r8) sumfn(nmode)
   real(r8) sumfm(nmode)
   real(r8) fnold(nmode)   ! number fraction activated
   real(r8) fmold(nmode)   ! mass fraction activated
   real(r8) wold,gold
   real(r8) alogam
   real(r8) rlo,rhi,xint1,xint2,xint3,xint4
   real(r8) wmin,wmax,w,dw,dwmax,dwmin,wnuc,dwnew,wb
   real(r8) dfmin,dfmax,fnew,fold,fnmin,fnbar,fsbar,fmbar
   real(r8) alw,sqrtalw
   real(r8) smax
   real(r8) x,arg
   real(r8) xmincoeff,xcut,volcut,surfcut
   real(r8) z,z1,z2,wf1,wf2,zf1,zf2,gf1,gf2,gf
   real(r8) etafactor1,etafactor2(nmode),etafactor2max
   real(r8) grow
   character(len=*), parameter :: subname='activate_modal'

   logical :: in_cloud
   integer m,n
   !      numerical integration parameters
   real(r8), parameter :: eps=0.3_r8,fmax=0.99_r8,sds=3._r8

   real(r8), parameter :: namin=1.e6_r8   ! minimum aerosol number concentration (/m3)

   integer ndist(nx)  ! accumulates frequency distribution of integration bins required
   data ndist/nx*0/
   save ndist

   if (present(in_cloud_in)) then
      if (.not. present(smax_f)) call endrun('activate_modal error: smax_f must be supplied when in_cloud is used')
      in_cloud = in_cloud_in
   else
      in_cloud = .false.
   end if

   fn(:)=0._r8
   fm(:)=0._r8
   fluxn(:)=0._r8
   fluxm(:)=0._r8
   flux_fullact=0._r8

   if(nmode.eq.1.and.na(1).lt.1.e-20_r8)return

   if(sigw.le.1.e-5_r8.and.wbar.le.0._r8)return

   if ( present( smax_prescribed ) ) then
      if (smax_prescribed <= 0.0_r8) return
   end if

   pres=rair*rhoair*tair
   diff0=0.211e-4_r8*(p0/pres)*(tair/t0)**1.94_r8
   conduct0=(5.69_r8+0.017_r8*(tair-t0))*4.186e2_r8*1.e-5_r8 ! convert to J/m/s/deg
   call qsat(tair, pres, es, qs)
   dqsdt=latvap/(rh2o*tair*tair)*qs
   alpha=gravit*(latvap/(cpair*rh2o*tair*tair)-1._r8/(rair*tair))
   gamma=(1.0_r8+latvap/cpair*dqsdt)/(rhoair*qs)
   etafactor2max=1.e10_r8/(alpha*wmaxf)**1.5_r8 ! this should make eta big if na is very small.

   grow  = 1._r8/(rhoh2o/(diff0*rhoair*qs)  &
           + latvap*rhoh2o/(conduct0*tair)*(latvap/(rh2o*tair) - 1._r8))
   sqrtg = sqrt(grow)
   beta  = 2._r8*pi*rhoh2o*grow*gamma

   do m=1,nmode

      if(volume(m).gt.1.e-39_r8.and.na(m).gt.1.e-39_r8)then
         !            number mode radius (m)
         !           write(iulog,*)'alogsig,volc,na=',alogsig(m),volc(m),na(m)
         amcube(m)=(3._r8*volume(m)/(4._r8*pi*exp45logsig(m)*na(m)))  ! only if variable size dist
         !           growth coefficent Abdul-Razzak & Ghan 1998 eqn 16
         !           should depend on mean radius of mode to account for gas kinetic effects
         !           see Fountoukis and Nenes, JGR2005 and Meskhidze et al., JGR2006
         !           for approriate size to use for effective diffusivity.
         etafactor2(m)=1._r8/(na(m)*beta*sqrtg)
         if(hygro(m).gt.1.e-10_r8)then
            smc(m)=2._r8*aten*sqrt(aten/(27._r8*hygro(m)*amcube(m))) ! only if variable size dist
         else
            smc(m)=100._r8
         endif
         !	    write(iulog,*)'sm,hygro,amcube=',smcrit(m),hygro(m),amcube(m)
      else
         smc(m)=1._r8
         etafactor2(m)=etafactor2max ! this should make eta big if na is very small.
      endif
      lnsm(m)=log(smc(m)) ! only if variable size dist
      !	 write(iulog,'(a,i4,4g12.2)')'m,na,amcube,hygro,sm,lnsm=', &
      !                   m,na(m),amcube(m),hygro(m),sm(m),lnsm(m)
   enddo

   if(sigw.gt.1.e-5_r8)then ! spectrum of updrafts

      wmax=min(wmaxf,wbar+sds*sigw)
      wmin=max(wminf,-wdiab)
      wmin=max(wmin,wbar-sds*sigw)
      w=wmin
      dwmax=eps*sigw
      dw=dwmax
      dfmax=0.2_r8
      dfmin=0.1_r8
      if (wmax <= w) return
      do m=1,nmode
         sumflxn(m)=0._r8
         sumfn(m)=0._r8
         fnold(m)=0._r8
         sumflxm(m)=0._r8
         sumfm(m)=0._r8
         fmold(m)=0._r8
      enddo
      sumflx_fullact=0._r8

      fold=0._r8
      wold=0._r8
      gold=0._r8

      dwmin = min( dwmax, 0.01_r8 )
      do n = 1, nx

100      wnuc=w+wdiab
         !           write(iulog,*)'wnuc=',wnuc
         alw=alpha*wnuc
         sqrtalw=sqrt(alw)
         etafactor1=alw*sqrtalw

         do m=1,nmode
            eta(m)=etafactor1*etafactor2(m)
            zeta(m)=twothird*sqrtalw*aten/sqrtg
         enddo

         if ( present( smax_prescribed ) ) then
            smax = smax_prescribed
         else
            call maxsat(zeta,eta,nmode,smc,smax)
         endif
         !	      write(iulog,*)'w,smax=',w,smax

         lnsmax=log(smax)

         x=twothird*(lnsm(nmode)-lnsmax)/(sq2*alogsig(nmode))
         fnew=0.5_r8*(1._r8-erf(x))


         dwnew = dw
         if(fnew-fold.gt.dfmax.and.n.gt.1)then
            !              reduce updraft increment for greater accuracy in integration
            if (dw .gt. 1.01_r8*dwmin) then
               dw=0.7_r8*dw
               dw=max(dw,dwmin)
               w=wold+dw
               go to 100
            else
               dwnew = dwmin
            endif
         endif

         if(fnew-fold.lt.dfmin)then
            !              increase updraft increment to accelerate integration
            dwnew=min(1.5_r8*dw,dwmax)
         endif
         fold=fnew

         z=(w-wbar)/(sigw*sq2)
         g=exp(-z*z)
         fnmin=1._r8
         xmincoeff=alogaten-twothird*(lnsmax-alog2)-alog3

         do m=1,nmode
            !              modal
            x=twothird*(lnsm(m)-lnsmax)/(sq2*alogsig(m))
            fn(m)=0.5_r8*(1._r8-erf(x))
            fnmin=min(fn(m),fnmin)
            !               integration is second order accurate
            !               assumes linear variation of f*g with w
            fnbar=(fn(m)*g+fnold(m)*gold)
            arg=x-1.5_r8*sq2*alogsig(m)
            fm(m)=0.5_r8*(1._r8-erf(arg))
            fmbar=(fm(m)*g+fmold(m)*gold)
            wb=(w+wold)
            if(w.gt.0._r8)then
               sumflxn(m)=sumflxn(m)+sixth*(wb*fnbar           &
                  +(fn(m)*g*w+fnold(m)*gold*wold))*dw
               sumflxm(m)=sumflxm(m)+sixth*(wb*fmbar           &
                  +(fm(m)*g*w+fmold(m)*gold*wold))*dw
            endif
            sumfn(m)=sumfn(m)+0.5_r8*fnbar*dw
            !	       write(iulog,'(a,9g10.2)')'lnsmax,lnsm(m),x,fn(m),fnold(m),g,gold,fnbar,dw=',lnsmax,lnsm(m),x,fn(m),fnold(m),g,gold,fnbar,dw
            fnold(m)=fn(m)
            sumfm(m)=sumfm(m)+0.5_r8*fmbar*dw
            fmold(m)=fm(m)
         enddo
         !           same form as sumflxm but replace the fm with 1.0
         sumflx_fullact = sumflx_fullact &
            + sixth*(wb*(g+gold) + (g*w+gold*wold))*dw
         !            sumg=sumg+0.5_r8*(g+gold)*dw
         gold=g
         wold=w
         dw=dwnew
         if (n > 1 .and. (w > wmax .or. fnmin > fmax)) exit
         w=w+dw
         if (n == nx) then
            write(iulog,*)'do loop is too short in activate'
            write(iulog,*)'wmin=',wmin,' w=',w,' wmax=',wmax,' dw=',dw
            write(iulog,*)'wbar=',wbar,' sigw=',sigw,' wdiab=',wdiab
            write(iulog,*)'wnuc=',wnuc
            write(iulog,*)'na=',(na(m),m=1,nmode)
            write(iulog,*)'fn=',(fn(m),m=1,nmode)
            !   dump all subr parameters to allow testing with standalone code
            !   (build a driver that will read input and call activate)
            write(iulog,*)'wbar,sigw,wdiab,tair,rhoair,nmode='
            write(iulog,*) wbar,sigw,wdiab,tair,rhoair,nmode
            write(iulog,*)'na=',na
            write(iulog,*)'volume=', (volume(m),m=1,nmode)
            write(iulog,*)'hydro='
            write(iulog,*) hygro
            call endrun(subname)
         end if

      enddo

      ndist(n)=ndist(n)+1
      if(w.lt.wmaxf)then

         !            contribution from all updrafts stronger than wmax
         !            assuming constant f (close to fmax)
         wnuc=w+wdiab

         z1=(w-wbar)/(sigw*sq2)
         z2=(wmaxf-wbar)/(sigw*sq2)
         g=exp(-z1*z1)
         integ=sigw*0.5_r8*sq2*sqpi*(erf(z2)-erf(z1))
         !            consider only upward flow into cloud base when estimating flux
         wf1=max(w,zero)
         zf1=(wf1-wbar)/(sigw*sq2)
         gf1=exp(-zf1*zf1)
         wf2=max(wmaxf,zero)
         zf2=(wf2-wbar)/(sigw*sq2)
         gf2=exp(-zf2*zf2)
         gf=(gf1-gf2)
         integf=wbar*sigw*0.5_r8*sq2*sqpi*(erf(zf2)-erf(zf1))+sigw*sigw*gf

         do m=1,nmode
            sumflxn(m)=sumflxn(m)+integf*fn(m)
            sumfn(m)=sumfn(m)+fn(m)*integ
            sumflxm(m)=sumflxm(m)+integf*fm(m)
            sumfm(m)=sumfm(m)+fm(m)*integ
         enddo
         !           same form as sumflxm but replace the fm with 1.0
         sumflx_fullact = sumflx_fullact + integf
         !            sumg=sumg+integ
      endif


      do m=1,nmode
         fn(m)=sumfn(m)/(sq2*sqpi*sigw)
         !            fn(m)=sumfn(m)/(sumg)
         if(fn(m).gt.1.01_r8)then
            write(iulog,*)'fn=',fn(m),' > 1 in activate'
            write(iulog,*)'w,m,na,amcube=',w,m,na(m),amcube(m)
            write(iulog,*)'integ,sumfn,sigw=',integ,sumfn(m),sigw
            call endrun('activate')
         endif
         fluxn(m)=sumflxn(m)/(sq2*sqpi*sigw)
         fm(m)=sumfm(m)/(sq2*sqpi*sigw)
         !            fm(m)=sumfm(m)/(sumg)
         if(fm(m).gt.1.01_r8)then
            write(iulog,*)'fm=',fm(m),' > 1 in activate'
         endif
         fluxm(m)=sumflxm(m)/(sq2*sqpi*sigw)
      enddo
      !        same form as fluxm
      flux_fullact = sumflx_fullact/(sq2*sqpi*sigw)

   else

      !        single updraft
      wnuc=wbar+wdiab

      if(wnuc.gt.0._r8)then

         w=wbar
              
         if(in_cloud) then

            if (smax_f > 0._r8) then
               smax = alpha*w/(2.0_r8*pi*rhoh2o*grow*gamma*smax_f)
            else
               smax = 1.e-20_r8
            end if

         else ! at cloud base
            alw        = alpha*wnuc
            sqrtalw    = sqrt(alw)
            etafactor1 = alw*sqrtalw

            do m = 1, nmode
               eta(m)  = etafactor1*etafactor2(m)
               zeta(m) = twothird*sqrtalw*aten/sqrtg
            end do
            if ( present(smax_prescribed) ) then
               smax = smax_prescribed
            else
               call maxsat(zeta, eta, nmode, smc, smax)
            end if
         end if

         lnsmax=log(smax)
         xmincoeff=alogaten-twothird*(lnsmax-alog2)-alog3


         do m=1,nmode
            !                 modal
            x=twothird*(lnsm(m)-lnsmax)/(sq2*alogsig(m))
            fn(m)=0.5_r8*(1._r8-erf(x))
            arg=x-1.5_r8*sq2*alogsig(m)
            fm(m)=0.5_r8*(1._r8-erf(arg))
            if(wbar.gt.0._r8)then
               fluxn(m)=fn(m)*w
               fluxm(m)=fm(m)*w
            endif
         enddo
         flux_fullact = w
      endif

   endif

end subroutine activate_modal

subroutine maxsat(zeta,eta,nmode,smc,smax)

   !      calculates maximum supersaturation for multiple
   !      competing aerosol modes.

   !      Abdul-Razzak and Ghan, A parameterization of aerosol activation.
   !      2. Multiple aerosol types. J. Geophys. Res., 105, 6837-6844.

   integer,  intent(in)  :: nmode ! number of modes
   real(r8), intent(in)  :: smc(nmode) ! critical supersaturation for number mode radius
   real(r8), intent(in)  :: zeta(nmode)
   real(r8), intent(in)  :: eta(nmode)
   real(r8), intent(out) :: smax ! maximum supersaturation
   integer  :: m  ! mode index
   real(r8) :: sum, g1, g2, g1sqrt, g2sqrt

   do m=1,nmode
      if(zeta(m).gt.1.e5_r8*eta(m).or.smc(m)*smc(m).gt.1.e5_r8*eta(m))then
         !            weak forcing. essentially none activated
         smax=1.e-20_r8
      else
         !            significant activation of this mode. calc activation all modes.
         exit
      endif
          ! No significant activation in any mode.  Do nothing.
      if (m == nmode) return

   enddo

   sum=0.0_r8
   do m=1,nmode
      if(eta(m).gt.1.e-20_r8)then
         g1=zeta(m)/eta(m)
         g1sqrt=sqrt(g1)
         g1=g1sqrt*g1
         g2=smc(m)/sqrt(eta(m)+3._r8*zeta(m))
         g2sqrt=sqrt(g2)
         g2=g2sqrt*g2
         sum=sum+(f1(m)*g1+f2(m)*g2)/(smc(m)*smc(m))
      else
         sum=1.e20_r8
      endif
   enddo

   smax=1._r8/sqrt(sum)

end subroutine maxsat

end module oracle_shim

program ndrop_activate_oracle
  use oracle_shim
  implicit none
  integer :: ncase, k, nmode, m
  real(r8) :: wbar, sigw, wdiab, wminf, wmaxf, tair, rhoair, flux_fullact
  real(r8), allocatable :: na(:), volume(:), hygro(:), sigmag(:), fn(:), fm(:), fluxn(:), fluxm(:)
  read(*,*) pi, rhoh2o, mwh2o, r_universal, rh2o, rair, cpair, latvap, gravit, tmelt, epsilo, t0, surften
  call set_constants()
  read(*,*) ncase
  do k = 1, ncase
    read(*,*) nmode
    allocate(na(nmode), volume(nmode), hygro(nmode), sigmag(nmode), fn(nmode), fm(nmode), fluxn(nmode), fluxm(nmode))
    read(*,*) wbar, sigw, wdiab, wminf, wmaxf, tair, rhoair
    read(*,*) (na(m), m=1,nmode)
    read(*,*) (volume(m), m=1,nmode)
    read(*,*) (hygro(m), m=1,nmode)
    read(*,*) (sigmag(m), m=1,nmode)
    call set_mode_constants(nmode, sigmag)
    call activate_modal(wbar, sigw, wdiab, wminf, wmaxf, tair, rhoair, na, nmode, volume, hygro, &
                        fn, fm, fluxn, fluxm, flux_fullact)
    write(*,'(*(ES25.17E3))') (fn(m), m=1,nmode), (fm(m), m=1,nmode), (fluxn(m), m=1,nmode), &
                              (fluxm(m), m=1,nmode), flux_fullact
    deallocate(na, volume, hygro, sigmag, fn, fm, fluxn, fluxm)
  end do
end program ndrop_activate_oracle
