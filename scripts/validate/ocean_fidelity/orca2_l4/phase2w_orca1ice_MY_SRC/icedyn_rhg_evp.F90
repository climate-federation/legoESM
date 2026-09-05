MODULE icedyn_rhg_evp
   !!======================================================================
   !!                     ***  MODULE  icedyn_rhg_evp  ***
   !!   Sea-Ice dynamics : rheology Elasto-Viscous-Plastic
   !!======================================================================
   !! History :   -   !  2007-03  (M.A. Morales Maqueda, S. Bouillon) Original code
   !!            3.0  !  2008-03  (M. Vancoppenolle) adaptation to new model
   !!             -   !  2008-11  (M. Vancoppenolle, S. Bouillon, Y. Aksenov) add surface tilt in ice rheolohy
   !!            3.3  !  2009-05  (G.Garric)    addition of the evp case
   !!            3.4  !  2011-01  (A. Porter)   dynamical allocation
   !!            3.5  !  2012-08  (R. Benshila) AGRIF
   !!            3.6  !  2016-06  (C. Rousset)  Rewriting + landfast ice + mEVP (Bouillon 2013)
   !!            3.7  !  2017     (C. Rousset)  add aEVP (Kimmritz 2016-2017)
   !!            4.0  !  2018     (many people) SI3 [aka Sea Ice cube]
   !!----------------------------------------------------------------------
#if defined key_si3
   !!----------------------------------------------------------------------
   !!   'key_si3'                                       SI3 sea-ice model
   !!----------------------------------------------------------------------
   !!   ice_dyn_rhg_evp : computes ice velocities from EVP rheology
   !!   rhg_evp_rst     : read/write EVP fields in ice restart
   !!----------------------------------------------------------------------
   USE par_ice              ! SI3 parameters
   USE par_icedyn           ! SI3 dynamics parameters
   USE phycst               ! Physical constant
   USE sbc_oce       , ONLY : ln_ice_embd, nn_fsbc, ssh_m
   USE sbc_ice       , ONLY : utau_ice, vtau_ice, snwice_mass, snwice_mass_b
   USE ice                  ! sea-ice: ice variables
   USE icevar        , ONLY : ice_var_sshdyn
   USE icedyn_rdgrft , ONLY : ice_strength
   USE bdy_oce       , ONLY : ln_bdy
   USE bdyice
#if defined key_agrif
   USE agrif_ice_interp
#endif
   !
   USE in_out_manager ! I/O manager
   USE iom            ! I/O manager library
   USE lib_mpp        ! MPP library
   USE lib_fortran    ! Fortran routines library
   USE lbclnk         ! lateral boundary conditions (or mpp links)
   USE prtctl         ! Print control

   USE netcdf         ! NetCDF library for convergence test
   IMPLICIT NONE
   PRIVATE

   PUBLIC   ice_dyn_rhg_evp   ! called by icedyn_rhg.F90
   PUBLIC   rhg_evp_rst       ! called by icedyn_rhg.F90

   !! for convergence tests
   INTEGER ::   ncvgid   ! netcdf file id
   INTEGER ::   nvarid   ! netcdf variable id
   REAL(wp), DIMENSION(:,:), ALLOCATABLE ::   fimask   ! mask at F points for the ice
   
   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/ICE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE ice_dyn_rhg_evp( kt, Kmm, pstress1_i, pstress2_i, pstress12_i, pshear_i, pdivu_i, pdelta_i )
      !!-------------------------------------------------------------------
      !!                 ***  SUBROUTINE ice_dyn_rhg_evp  ***
      !!                             EVP-C-grid
      !!
      !! ** purpose : determines sea ice drift from wind stress, ice-ocean
      !!  stress and sea-surface slope. Ice-ice interaction is described by
      !!  a non-linear elasto-viscous-plastic (EVP) law including shear
      !!  strength and a bulk rheology (Hunke and Dukowicz, 2002).
      !!
      !!  The points in the C-grid look like this, dear reader
      !!
      !!                              (ji,jj)
      !!                                 |
      !!                                 |
      !!                      (ji-1,jj)  |  (ji,jj)
      !!                             ---------
      !!                            |         |
      !!                            | (ji,jj) |------(ji,jj)
      !!                            |         |
      !!                             ---------
      !!                     (ji-1,jj-1)     (ji,jj-1)
      !!
      !! ** Inputs  : - wind forcing (stress), oceanic currents
      !!                ice, snow and ponds total volume (vt_i, vt_s, vt_ip, vt_il) per unit area
      !!
      !! ** Action  : - compute u_ice, v_ice : the components of the
      !!                sea-ice velocity vector
      !!              - compute delta_i, shear_i, divu_i, which are inputs
      !!                of the ice thickness distribution
      !!
      !! ** Steps   : 0) compute mask at F point
      !!              1) Compute ice snow mass, ice strength
      !!              2) Compute wind, oceanic stresses, mass terms and
      !!                 coriolis terms of the momentum equation
      !!              3) Solve the momentum equation (iterative procedure)
      !!              4) Recompute delta, shear and divergence
      !!                 (which are inputs of the ITD) & store stress
      !!                 for the next time step
      !!              5) Diagnostics including charge ellipse
      !!
      !! ** Notes   : There is the possibility to use aEVP from the nice work of Kimmritz et al. (2016 & 2017)
      !!              by setting up ln_aEVP=T (i.e. changing alpha and beta parameters).
      !!              This is an upgraded version of mEVP from Bouillon et al. 2013
      !!              (i.e. more stable and better convergence)
      !!
      !! References : Hunke and Dukowicz, JPO97
      !!              Bouillon et al., Ocean Modelling 2009
      !!              Bouillon et al., Ocean Modelling 2013
      !!              Kimmritz et al., Ocean Modelling 2016 & 2017
      !!-------------------------------------------------------------------
      INTEGER                    , INTENT(in   ) ::   kt                                    ! time step
      INTEGER                    , INTENT(in   ) ::   Kmm                                   ! ocean time level index
      REAL(wp), DIMENSION(:,:)   , INTENT(inout) ::   pstress1_i, pstress2_i, pstress12_i   !
      REAL(wp), DIMENSION(A2D(0)), INTENT(  out) ::   pshear_i  , pdivu_i   , pdelta_i      !
      !!
      INTEGER ::   ji, jj       ! dummy loop indices
      INTEGER ::   jter         ! local integers
      !
      REAL(wp) ::   zdtevp, z1_dtevp                                    ! time step for subcycling
      REAL(wp) ::   ecc2, z1_ecc2                                       ! square of yield ellipse eccenticity
      REAL(wp) ::   zalph1, z1_alph1, zalph2, z1_alph2                  ! alpha coef from Bouillon 2009 or Kimmritz 2017
      REAl(wp) ::   zbetau, zbetav
      REAL(wp) ::   zm1, zm2, zm3, zmassU, zmassV, zvU, zvV             ! ice/snow mass and volume
      REAL(wp) ::   zp_delf, zds2, zdt, zdt2, zdiv, zdiv2               ! temporary scalars
      REAL(wp) ::   zTauO, zTauB, zRHS, zvel                            ! temporary scalars
      REAL(wp) ::   zkt                                                 ! isotropic tensile strength for landfast ice
      REAL(wp) ::   zvCr                                                ! critical ice volume above which ice is landfast
      !
      REAL(wp) ::   zfac_x, zfac_y
      !
      REAL(wp), DIMENSION(jpi,jpj)       ::   zdelta, zp_delt                 ! delta, P/delta at T points
      REAL(wp), DIMENSION(jpi,jpj)       ::   zbeta                           ! beta coef from Kimmritz 2017
      !
      REAL(wp), DIMENSION(jpi,jpj)       ::   zdt_m                           ! (dt / ice-snow_mass) on T points
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   zaU  , zaV                      ! ice fraction on U/V points
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   zmU_t, zmV_t                    ! (ice-snow_mass / dt) on U/V points
      REAL(wp), DIMENSION(jpi,jpj)       ::   zmf                             ! coriolis parameter at T points
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   v_oceU, u_oceV, v_iceU, u_iceV  ! ocean/ice u/v component on V/U points
      !
      REAL(wp), DIMENSION(jpi,jpj)       ::   zds                             ! shear
      REAL(wp), DIMENSION(A2D(0))        ::   zten_i, zshear                  ! tension, shear
      REAL(wp), DIMENSION(jpi,jpj)       ::   zs1, zs2, zs12                  ! stress tensor components
      REAL(wp), DIMENSION(jpi,jpj)       ::   zsshdyn                         ! array used for the calculation of ice surface slope:
      !                                                                       !    ocean surface (ssh_m) if ice is not embedded
      !                                                                       !    ice bottom surface if ice is embedded
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   zfU  , zfV                      ! internal stresses
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   zspgU, zspgV                    ! surface pressure gradient at U/V points
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   zCorU, zCorV                    ! Coriolis stress array
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   ztaux_ai, ztauy_ai              ! ice-atm. stress at U-V points
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   ztaux_oi, ztauy_oi              ! ice-ocean stress at U-V points
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   ztaux_bi, ztauy_bi              ! ice-OceanBottom stress at U-V points (landfast)
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   ztaux_base, ztauy_base          ! ice-bottom stress at U-V points (landfast)
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   zdragx_oi, zdragy_oi            ! ice-ocean drags at U-V points
      !
      REAL(wp), DIMENSION(jpi,jpj)       ::   zmsk
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   zmsk01x, zmsk01y                ! dummy arrays
      REAL(wp), DIMENSION(A2D(nn_hls-1)) ::   zmsk00x, zmsk00y                ! mask for ice presence

      REAL(wp), PARAMETER          ::   zepsi  = 1.0e-20_wp             ! tolerance parameter
      REAL(wp), PARAMETER          ::   zmmin  = 1._wp                  ! ice mass (kg/m2)  below which ice velocity becomes very small
      REAL(wp), PARAMETER          ::   zamin  = 0.001_wp               ! ice concentration below which ice velocity becomes very small
      !! --- check convergence
      REAL(wp), DIMENSION(A2D(0))  ::   zmsk00, zmsk15
      REAL(wp), DIMENSION(A2D(0))  ::   zu_ice, zv_ice
      !! --- diags
      REAL(wp) ::   zsig1, zsig2, zsig12, zfac, z1_strength
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   zsig_I, zsig_II, zsig1_p, zsig2_p
      !! --- SIMIP diags
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   zdiag_xmtrp_ice ! X-component of ice mass transport (kg/s)
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   zdiag_ymtrp_ice ! Y-component of ice mass transport (kg/s)
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   zdiag_xmtrp_snw ! X-component of snow mass transport (kg/s)
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   zdiag_ymtrp_snw ! Y-component of snow mass transport (kg/s)
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   zdiag_xatrp     ! X-component of area transport (m2/s)
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   zdiag_yatrp     ! Y-component of area transport (m2/s)
      !! -- advect fields at the rheology time step for the calculation of strength
      !!    it seems that convergence is worse when ll_advups=true. So it is not really a good idea
      LOGICAL  ::   ll_advups = .FALSE.
      REAL(wp) ::   zdt_ups
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) ::   za_i_ups, zv_i_ups   ! tracers advected upstream
      !!-------------------------------------------------------------------

      IF( kt == nit000 .AND. lwp )   WRITE(numout,*) '-- ice_dyn_rhg_evp: EVP sea-ice rheology'
      !
      DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )
         IF( at_i(ji,jj) < epsi10 ) THEN   ;   zmsk(ji,jj) = 0._wp
         ELSE                              ;   zmsk(ji,jj) = 1._wp   ;   ENDIF
      END_2D
      ! for diagnostics and convergence tests
      DO_2D( 0, 0, 0, 0 )
         IF( at_i(ji,jj) < epsi06 ) THEN   ;   zmsk00(ji,jj) = 0._wp
         ELSE                              ;   zmsk00(ji,jj) = 1._wp   ;   ENDIF
      END_2D
      IF( nn_rhg_chkcvg > 0 ) THEN
         DO_2D( 0, 0, 0, 0 )
            IF( at_i(ji,jj) < 0.15_wp ) THEN   ;   zmsk15(ji,jj) = 0._wp
            ELSE                               ;   zmsk15(ji,jj) = 1._wp   ;   ENDIF
         END_2D
      ENDIF
      !
      !------------------------------------------------------------------------------!
      ! 0) mask at F points for the ice
      !------------------------------------------------------------------------------!
      IF( kt == nit000 ) THEN
         ! ocean/land mask
         ALLOCATE( fimask(jpi,jpj) )
         IF( rn_ishlat == 0._wp ) THEN
            DO_2D( 0, 0, 0, 0 )
               fimask(ji,jj) = tmask(ji,jj,1) * tmask(ji+1,jj,1) * tmask(ji,jj+1,1) * tmask(ji+1,jj+1,1)
            END_2D
         ELSE
            DO_2D( 0, 0, 0, 0 )
               fimask(ji,jj) = tmask(ji,jj,1) * tmask(ji+1,jj,1) * tmask(ji,jj+1,1) * tmask(ji+1,jj+1,1)
               ! Lateral boundary conditions on velocity (modify fimask)
               IF( fimask(ji,jj) == 0._wp ) THEN
                  fimask(ji,jj) = rn_ishlat * MIN( 1._wp , MAX( umask(ji,jj,1), umask(ji  ,jj+1,1), &
                     &                                          vmask(ji,jj,1), vmask(ji+1,jj  ,1) ) )
               ENDIF
            END_2D
         ENDIF
         CALL lbc_lnk( 'icedyn_rhg_evp', fimask, 'F', 1._wp )
      ENDIF
      !------------------------------------------------------------------------------!
      ! 1) define some variables and initialize arrays
      !------------------------------------------------------------------------------!

      ! ecc2: square of yield ellipse eccenticrity
      ecc2    = rn_ecc * rn_ecc
      z1_ecc2 = 1._wp / ecc2

      ! alpha parameters (Bouillon 2009)
      IF( .NOT. ln_aEVP ) THEN
         zdtevp   = rDt_ice / REAL( nn_nevp )
         zalph1 =   2._wp * rn_relast * REAL( nn_nevp )
         zalph2 = zalph1 * z1_ecc2

         z1_alph1 = 1._wp / ( zalph1 + 1._wp )
         z1_alph2 = 1._wp / ( zalph2 + 1._wp )
      ELSE
         zdtevp   = rDt_ice
         ! zalpha parameters set later on adaptatively
      ENDIF
      z1_dtevp = 1._wp / zdtevp

      ! Initialise stress tensor
      zs1 (:,:) = pstress1_i (:,:)
      zs2 (:,:) = pstress2_i (:,:)
      zs12(:,:) = pstress12_i(:,:)

      ! Ice strength
      CALL ice_strength

      ! landfast param from Lemieux(2016): add isotropic tensile strength (following Konig Beatty and Holland, 2010)
      IF( ln_landfast_L16 ) THEN   ;   zkt = rn_lf_tensile
      ELSE                         ;   zkt = 0._wp
      ENDIF
      !
      !------------------------------------------------------------------------------!
      ! 2) Wind / ocean stress, mass terms, coriolis terms
      !------------------------------------------------------------------------------!
      ! sea surface height
      !    embedded sea ice: compute representative ice top surface
      !    non-embedded sea ice: use ocean surface for slope calculation
      zsshdyn(:,:) = ice_var_sshdyn( ssh_m, snwice_mass, snwice_mass_b)

      DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )
         zm1          = ( rhos * vt_s(ji,jj) + rhoi * vt_i(ji,jj) + rhow * (vt_ip(ji,jj) + vt_il(ji,jj)) ) ! Ice/snow mass at U-V points
         zmf  (ji,jj) = zm1 * ff_t(ji,jj)                                                                  ! Coriolis at T points (m*f)
         zdt_m(ji,jj) = zdtevp / MAX( zm1, zmmin )                                                         ! dt/m at T points (for alpha and beta coefficients)
      END_2D
      
      DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )

         ! ice fraction at U-V points
         zaU(ji,jj) = 0.5_wp * ( at_i(ji,jj)*e1e2t(ji,jj) + at_i(ji+1,jj  )*e1e2t(ji+1,jj  ) ) * r1_e1e2u(ji,jj) * umask(ji,jj,1)
         zaV(ji,jj) = 0.5_wp * ( at_i(ji,jj)*e1e2t(ji,jj) + at_i(ji  ,jj+1)*e1e2t(ji  ,jj+1) ) * r1_e1e2v(ji,jj) * vmask(ji,jj,1)

         ! Ice/snow mass at U-V points
         zm1 = ( rhos * vt_s(ji  ,jj  ) + rhoi * vt_i(ji  ,jj  ) + rhow * (vt_ip(ji  ,jj  ) + vt_il(ji  ,jj  )) )
         zm2 = ( rhos * vt_s(ji+1,jj  ) + rhoi * vt_i(ji+1,jj  ) + rhow * (vt_ip(ji+1,jj  ) + vt_il(ji+1,jj  )) )
         zm3 = ( rhos * vt_s(ji  ,jj+1) + rhoi * vt_i(ji  ,jj+1) + rhow * (vt_ip(ji  ,jj+1) + vt_il(ji  ,jj+1)) )
         !
         zmassU = 0.5_wp * ( zm1 * e1e2t(ji,jj) + zm2 * e1e2t(ji+1,jj  ) ) * r1_e1e2u(ji,jj) * umask(ji,jj,1)
         zmassV = 0.5_wp * ( zm1 * e1e2t(ji,jj) + zm3 * e1e2t(ji  ,jj+1) ) * r1_e1e2v(ji,jj) * vmask(ji,jj,1)

         ! Ocean currents at U-V points
         ! (brackets added to fix the order of floating point operations for the North Pole reproducibility)
         v_oceU(ji,jj)   = 0.25_wp * ( (v_oce(ji,jj) + v_oce(ji  ,jj-1)) + (v_oce(ji+1,jj  ) + v_oce(ji+1,jj-1)) ) * umask(ji,jj,1)
         u_oceV(ji,jj)   = 0.25_wp * ( (u_oce(ji,jj) + u_oce(ji-1,jj  )) + (u_oce(ji  ,jj+1) + u_oce(ji-1,jj+1)) ) * vmask(ji,jj,1)

         ! m/dt
         zmU_t(ji,jj)    = zmassU * z1_dtevp
         zmV_t(ji,jj)    = zmassV * z1_dtevp

         ! Drag ice-atm.
         !     Note the use of 0.5*(2-umask) in order to unmask the stress along coastlines
         !      and the use of MAX(tmask(i,j),tmask(i+1,j) is to mask tau over ice shelves
         ztaux_ai(ji,jj) = zaU(ji,jj) * 0.5_wp * ( utau_ice(ji,jj) + utau_ice(ji+1,jj) ) * &
            &                                    ( 2. - umask(ji,jj,1) ) * MAX( tmask(ji,jj,1), tmask(ji+1,jj,1) )
         ztauy_ai(ji,jj) = zaV(ji,jj) * 0.5_wp * ( vtau_ice(ji,jj) + vtau_ice(ji,jj+1) ) * &
            &                                    ( 2. - vmask(ji,jj,1) ) * MAX( tmask(ji,jj,1), tmask(ji,jj+1,1) )

         ! Drag ice-ocean
         !     Note the use of 0.5*(2-umask) in order to unmask the stress along coastlines
         !      and the use of MAX(tmask(i,j),tmask(i+1,j) is to mask tau over ice shelves
         zdragx_oi(ji,jj) = rho0 * zaU(ji,jj) * 0.5_wp * ( drag_io(ji,jj) + drag_io(ji+1,jj) ) * &
            &                                            ( 2. - umask(ji,jj,1) ) * MAX( tmask(ji,jj,1), tmask(ji+1,jj,1) )
         zdragy_oi(ji,jj) = rho0 * zaV(ji,jj) * 0.5_wp * ( drag_io(ji,jj) + drag_io(ji,jj+1) ) * &
            &                                            ( 2. - vmask(ji,jj,1) ) * MAX( tmask(ji,jj,1), tmask(ji,jj+1,1) )
         
         ! Surface pressure gradient (- m*g*GRAD(ssh)) at U-V points
         zspgU(ji,jj)    = - zmassU * grav * ( zsshdyn(ji+1,jj) - zsshdyn(ji,jj) ) * r1_e1u(ji,jj)
         zspgV(ji,jj)    = - zmassV * grav * ( zsshdyn(ji,jj+1) - zsshdyn(ji,jj) ) * r1_e2v(ji,jj)

         ! masks
         IF( zmassU > 0._wp ) THEN   ;   zmsk00x(ji,jj) = 1._wp
         ELSE                        ;   zmsk00x(ji,jj) = 0._wp   ;   ENDIF
         IF( zmassV > 0._wp ) THEN   ;   zmsk00y(ji,jj) = 1._wp
         ELSE                        ;   zmsk00y(ji,jj) = 0._wp   ;   ENDIF
            
         ! switches
         IF( zmassU <= zmmin .AND. zaU(ji,jj) <= zamin ) THEN   ;   zmsk01x(ji,jj) = 0._wp
         ELSE                                                   ;   zmsk01x(ji,jj) = 1._wp   ;   ENDIF
         IF( zmassV <= zmmin .AND. zaV(ji,jj) <= zamin ) THEN   ;   zmsk01y(ji,jj) = 0._wp
         ELSE                                                   ;   zmsk01y(ji,jj) = 1._wp   ;   ENDIF

      END_2D
      !
      !                                  !== Landfast ice parameterization ==!
      !
      IF( ln_landfast_L16 ) THEN         !-- Lemieux 2016
         DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
            ! ice thickness at U-V points
            zvU = 0.5_wp * ( vt_i(ji,jj)*e1e2t(ji,jj) + vt_i(ji+1,jj  )*e1e2t(ji+1,jj  ) ) * r1_e1e2u(ji,jj) * umask(ji,jj,1)
            zvV = 0.5_wp * ( vt_i(ji,jj)*e1e2t(ji,jj) + vt_i(ji  ,jj+1)*e1e2t(ji  ,jj+1) ) * r1_e1e2v(ji,jj) * vmask(ji,jj,1)
            ! ice-bottom stress at U points
            IF( icb_umask(ji,jj) == 0._wp ) THEN
               zvCr = zaU(ji,jj) * rn_lf_depfra * hu(ji,jj,Kmm)
               ztaux_base(ji,jj) = - rn_lf_bfr * MAX( 0._wp, zvU - zvCr ) * EXP( -rn_crhg * ( 1._wp - zaU(ji,jj) ) )
            ELSE
               ztaux_base(ji,jj) = - rn_lf_bfr * icb_umask(ji,jj)
            ENDIF
            ! ice-bottom stress at V points
            IF( icb_vmask(ji,jj) == 0._wp ) THEN
               zvCr = zaV(ji,jj) * rn_lf_depfra * hv(ji,jj,Kmm)
               ztauy_base(ji,jj) = - rn_lf_bfr * MAX( 0._wp, zvV - zvCr ) * EXP( -rn_crhg * ( 1._wp - zaV(ji,jj) ) )
            ELSE
               ztauy_base(ji,jj) = - rn_lf_bfr * icb_vmask(ji,jj)
            ENDIF
         END_2D
         DO_2D( 0, 0, 0, 0 )
            ! ice_bottom stress at T points
            IF( icb_tmask(ji,jj) == 0._wp ) THEN
               zvCr = at_i(ji,jj) * rn_lf_depfra * ht(ji,jj,Kmm)
               tau_icebfr(ji,jj) = - rn_lf_bfr * MAX( 0._wp, vt_i(ji,jj) - zvCr ) * EXP( -rn_crhg * ( 1._wp - at_i(ji,jj) ) )
            ELSE
               tau_icebfr(ji,jj) = - rn_lf_bfr * icb_tmask(ji,jj)
            ENDIF
         END_2D
      ELSE                               !-- no landfast or landfast read in a file
         DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
            ! ice-bottom stress at U, V points
            ztaux_base(ji,jj) = 0._wp
            ztauy_base(ji,jj) = 0._wp
         END_2D
         DO_2D( 0, 0, 0, 0 )
            ! ice-bottom stress at T points
            tau_icebfr(ji,jj) = 0._wp
         END_2D
      ENDIF

      ! nn_fsbc=2: retain the first three executed SI3 calls (ocean kt 1,3,5).
      IF( kt <= nit000 + 4 ) CALL l4_orca1ice_dyn_dump( kt, Kmm, pstress1_i, &
         & pstress2_i, pstress12_i, zaU, zaV, ztaux_ai, ztauy_ai, zdragx_oi, &
         & zdragy_oi, ztaux_base, ztauy_base, zmU_t, zmV_t )

      !------------------------------------------------------------------------------!
      ! 3) Solution of the momentum equation, iterative procedure
      !------------------------------------------------------------------------------!
      !
      !                                               ! ==================== !
      DO jter = 1 , nn_nevp                           !    loop over jter    !
         !                                            ! ==================== !
         ! convergence test
         IF( nn_rhg_chkcvg == 1 .OR. nn_rhg_chkcvg == 2  ) THEN
            DO_2D( 0, 0, 0, 0 )
               zu_ice(ji,jj) = u_ice(ji,jj) * umask(ji,jj,1) ! velocity at previous time step
               zv_ice(ji,jj) = v_ice(ji,jj) * vmask(ji,jj,1)
            END_2D
         ENDIF

         ! --- divergence, tension & shear (Appendix B of Hunke & Dukowicz, 2002) --- !
         DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )

            ! shear at F points
            zds(ji,jj) = ( ( u_ice(ji,jj+1) * r1_e1u(ji,jj+1) - u_ice(ji,jj) * r1_e1u(ji,jj) ) * e1f(ji,jj) * e1f(ji,jj)   &
               &         + ( v_ice(ji+1,jj) * r1_e2v(ji+1,jj) - v_ice(ji,jj) * r1_e2v(ji,jj) ) * e2f(ji,jj) * e2f(ji,jj)   &
               &         ) * r1_e1e2f(ji,jj) * fimask(ji,jj)

         END_2D

         DO_2D( 0, 0, 0, 0 )

            ! shear**2 at T points (doc eq. A16)
            zds2 = ( ( zds(ji,jj  )*zds(ji,jj  )*e1e2f(ji,jj  ) + zds(ji-1,jj  )*zds(ji-1,jj  )*e1e2f(ji-1,jj  ) ) & ! add () for
               &   + ( zds(ji,jj-1)*zds(ji,jj-1)*e1e2f(ji,jj-1) + zds(ji-1,jj-1)*zds(ji-1,jj-1)*e1e2f(ji-1,jj-1) ) & ! NP repro
               &   ) * 0.25_wp * r1_e1e2t(ji,jj)

            ! divergence at T points
            zdiv  = ( ( e2u(ji,jj) * u_ice(ji,jj) - e2u(ji-1,jj) * u_ice(ji-1,jj) )   &   ! add () for
               &    + ( e1v(ji,jj) * v_ice(ji,jj) - e1v(ji,jj-1) * v_ice(ji,jj-1) )   &   ! NP repro
               &    ) * r1_e1e2t(ji,jj)
            zdiv2 = zdiv * zdiv

            ! tension at T points
            zdt  = ( ( u_ice(ji,jj) * r1_e2u(ji,jj) - u_ice(ji-1,jj) * r1_e2u(ji-1,jj) ) * e2t(ji,jj) * e2t(ji,jj)   &
               &   - ( v_ice(ji,jj) * r1_e1v(ji,jj) - v_ice(ji,jj-1) * r1_e1v(ji,jj-1) ) * e1t(ji,jj) * e1t(ji,jj)   &
               &   ) * r1_e1e2t(ji,jj)
            zdt2 = zdt * zdt

            ! delta at T points
            zdelta(ji,jj) = SQRT( zdiv2 + ( zdt2 + zds2 ) * z1_ecc2 ) * zmsk(ji,jj)        ! zmsk is for reducing cpu

            ! P/delta at T points
            zp_delt(ji,jj) = strength(ji,jj) / ( zdelta(ji,jj) + rn_creepl ) * zmsk(ji,jj) ! zmsk is for reducing cpu

         END_2D
         CALL lbc_lnk( 'icedyn_rhg_evp', zdelta, 'T', 1.0_wp, zp_delt, 'T', 1.0_wp )

         !
         DO_2D( nn_hls-1, nn_hls, nn_hls-1, nn_hls )   ! loop ends at jpi,jpj so that no lbc_lnk are needed for zs1 and zs2

            ! divergence at T points (duplication to avoid communications)
            zdiv  = ( ( e2u(ji,jj) * u_ice(ji,jj) - e2u(ji-1,jj) * u_ice(ji-1,jj) )   &   ! add () for
               &    + ( e1v(ji,jj) * v_ice(ji,jj) - e1v(ji,jj-1) * v_ice(ji,jj-1) )   &   ! NP repro
               &    ) * r1_e1e2t(ji,jj)

            ! tension at T points (duplication to avoid communications)
            zdt  = ( ( u_ice(ji,jj) * r1_e2u(ji,jj) - u_ice(ji-1,jj) * r1_e2u(ji-1,jj) ) * e2t(ji,jj) * e2t(ji,jj)   &
               &   - ( v_ice(ji,jj) * r1_e1v(ji,jj) - v_ice(ji,jj-1) * r1_e1v(ji,jj-1) ) * e1t(ji,jj) * e1t(ji,jj)   &
               &   ) * r1_e1e2t(ji,jj)

            ! alpha for aEVP
            !   gamma = 0.5*P/(delta+creepl) * (c*pi)**2/Area * dt/m
            !   alpha = beta = sqrt(4*gamma)
            IF( ln_aEVP ) THEN
               zalph1   = MAX( 50._wp, rpi * SQRT( 0.5_wp * zp_delt(ji,jj) * r1_e1e2t(ji,jj) * zdt_m(ji,jj) ) )
               z1_alph1 = 1._wp / ( zalph1 + 1._wp )
               zalph2   = zalph1
               z1_alph2 = z1_alph1
               ! explicit:
               ! z1_alph1 = 1._wp / zalph1
               ! z1_alph2 = 1._wp / zalph1
               ! zalph1 = zalph1 - 1._wp
               ! zalph2 = zalph1
            ENDIF

            ! stress at T points (zkt/=0 if landfast)
            zs1(ji,jj) = ( zs1(ji,jj)*zalph1 + zp_delt(ji,jj) * ( zdiv*(1._wp + zkt) - zdelta(ji,jj)*(1._wp - zkt) ) ) &
               &         * z1_alph1 * zmsk(ji,jj) ! zmsk is for reducing cpu
            zs2(ji,jj) = ( zs2(ji,jj)*zalph2 + zp_delt(ji,jj) * ( zdt * z1_ecc2 * (1._wp + zkt) ) ) &
               &         * z1_alph2 * zmsk(ji,jj) ! zmsk is for reducing cpu

         END_2D

         ! Save beta at T-points for further computations
         IF( ln_aEVP ) THEN
            DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )
               zbeta(ji,jj) = MAX( 50._wp, rpi * SQRT( 0.5_wp * zp_delt(ji,jj) * r1_e1e2t(ji,jj) * zdt_m(ji,jj) ) )
            END_2D
         ENDIF

         DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )

            ! alpha for aEVP
            IF( ln_aEVP ) THEN
               zalph2   = MAX( zbeta(ji,jj), zbeta(ji+1,jj), zbeta(ji,jj+1), zbeta(ji+1,jj+1) )
               z1_alph2 = 1._wp / ( zalph2 + 1._wp )
               ! explicit:
               ! z1_alph2 = 1._wp / zalph2
               ! zalph2 = zalph2 - 1._wp
            ENDIF

            ! P/delta at F points
            ! (brackets added to fix the order of floating point operations for the North Pole reproducibility)
            zp_delf = 0.25_wp * ( ( zp_delt(ji,jj) + zp_delt(ji+1,jj) ) + ( zp_delt(ji,jj+1) + zp_delt(ji+1,jj+1) ) )

            ! stress at F points (zkt/=0 if landfast)
            zs12(ji,jj)= ( zs12(ji,jj) * zalph2 + zp_delf * ( zds(ji,jj) * z1_ecc2 * (1._wp + zkt) ) * 0.5_wp ) &
               &         * z1_alph2 

         END_2D

         ! --- Ice internal stresses (Appendix C of Hunke and Dukowicz, 2002) --- !
         ! (brackets added to fix the order of floating point operations for halo 1 - halo 2 compatibility)
         DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
            !                   !--- U points
            zfU(ji,jj) = 0.5_wp * ( ( ( zs1(ji+1,jj) - zs1(ji,jj) ) * e2u(ji,jj)                                           &
               &                    + ( zs2(ji+1,jj) * e2t(ji+1,jj) * e2t(ji+1,jj) - zs2(ji,jj) * e2t(ji,jj) * e2t(ji,jj)  &
               &                      ) * r1_e2u(ji,jj) )                                                                  &
               &                  + ( zs12(ji,jj) * e1f(ji,jj) * e1f(ji,jj) - zs12(ji,jj-1) * e1f(ji,jj-1) * e1f(ji,jj-1)  &
               &                    ) * 2._wp * r1_e1u(ji,jj)                                                              &
               &                  ) * r1_e1e2u(ji,jj)
            !
            !                !--- V points
            zfV(ji,jj) = 0.5_wp * ( ( ( zs1(ji,jj+1) - zs1(ji,jj) ) * e1v(ji,jj)                                           &
               &                    - ( zs2(ji,jj+1) * e1t(ji,jj+1) * e1t(ji,jj+1) - zs2(ji,jj) * e1t(ji,jj) * e1t(ji,jj)  &
               &                      ) * r1_e1v(ji,jj) )                                                                  &
               &                  + ( zs12(ji,jj) * e2f(ji,jj) * e2f(ji,jj) - zs12(ji-1,jj) * e2f(ji-1,jj) * e2f(ji-1,jj)  &
               &                    ) * 2._wp * r1_e2v(ji,jj)                                                              &
               &                  ) * r1_e1e2v(ji,jj)
            !
            !                !--- ice currents at U-V point, warning: add () for NP repro
            v_iceU(ji,jj) = 0.25_wp * ( ( v_ice(ji,jj) + v_ice(ji,jj-1) ) + ( v_ice(ji+1,jj) + v_ice(ji+1,jj-1) ) ) * umask(ji,jj,1)
            u_iceV(ji,jj) = 0.25_wp * ( ( u_ice(ji,jj) + u_ice(ji-1,jj) ) + ( u_ice(ji,jj+1) + u_ice(ji-1,jj+1) ) ) * vmask(ji,jj,1)
            !
         END_2D
         !
         ! --- Computation of ice velocity --- !
         !  Bouillon et al. 2013 (eq 47-48) => unstable unless alpha, beta vary as in Kimmritz 2016 & 2017
         !  Bouillon et al. 2009 (eq 34-35) => stable
         !
         ! aEVP formulation given by eq.6 from Kimmritz et al. 2016, but taken the last term implicitly (as in eq. 8)
         ! u(p+1) - u(p) = 1/beta * ( dt/m * RHS(p+1) + u(n) - u(p+1) )
         !    with RHS = tau_ai + tau_oi + tau_bi + Coriolis + Spg
         !     and tau_oi = ztau0 * ( uoce - u(p+1) )
         !         tau_bi = ztauB *          u(p+1)
         ! Hence:
         ! u(p+1) = 1/(m/dt*(beta+1)+ztauO-ztauB) * (m/dt*(beta*u(p)+u(n))+RHS+ztauO*u(p))
         !
         IF( MOD(jter,2) == 0 ) THEN ! even iterations
            !
            DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
               !                 !--- tau_io/(v_oce - v_ice)
               zTauO = zdragy_oi(ji,jj) *  &
                  &    SQRT( ( v_ice (ji,jj) - v_oce (ji,jj) ) * ( v_ice (ji,jj) - v_oce (ji,jj) )  &
                  &        + ( u_iceV(ji,jj) - u_oceV(ji,jj) ) * ( u_iceV(ji,jj) - u_oceV(ji,jj) ) )
               !                 !--- Ocean-to-Ice stress
               ztauy_oi(ji,jj) = zTauO * ( v_oce(ji,jj) - v_ice(ji,jj) )
               !
               !                 !--- tau_bottom/v_ice
               zvel  = 5.e-05_wp + SQRT( v_ice(ji,jj) * v_ice(ji,jj) + u_iceV(ji,jj) * u_iceV(ji,jj) )
               zTauB = ztauy_base(ji,jj) / zvel
               !                 !--- OceanBottom-to-Ice stress
               ztauy_bi(ji,jj) = zTauB * v_ice(ji,jj)
               !
               !                 !--- Coriolis at V-points (energy conserving formulation)
               zCorV(ji,jj)  = - 0.25_wp * r1_e2v(ji,jj) *  &
                  &    ( zmf(ji,jj  ) * ( e2u(ji,jj  ) * u_ice(ji,jj  ) + e2u(ji-1,jj  ) * u_ice(ji-1,jj  ) )  &
                  &    + zmf(ji,jj+1) * ( e2u(ji,jj+1) * u_ice(ji,jj+1) + e2u(ji-1,jj+1) * u_ice(ji-1,jj+1) ) )
               !
               !                 !--- Sum of external forces (explicit solution) = F + tau_ia + Coriolis + spg + tau_io
               zRHS = zfV(ji,jj) + ztauy_ai(ji,jj) + zCorV(ji,jj) + zspgV(ji,jj) + ztauy_oi(ji,jj)
               !
               IF( ln_aEVP ) THEN   !--- ice velocity using aEVP (Kimmritz et al 2016 & 2017)
                  !
                  zbetav = MAX( zbeta(ji,jj), zbeta(ji,jj+1) )
                  !
                  IF( ( zRHS + ztauy_base(ji,jj) ) < 0._wp .AND. zRHS >= 0._wp ) THEN ! static friction => slow decrease to v=0
                     v_ice(ji,jj) = ( v_ice_b(ji,jj) + v_ice(ji,jj) * MAX( 0._wp, zbetav - zdtevp*rn_lf_relax ) ) / (zbetav+1._wp)
                  ELSE
                     v_ice(ji,jj) = ( zmV_t(ji,jj) * ( zbetav * v_ice(ji,jj) + v_ice_b(ji,jj) ) + zRHS + zTauO * v_ice(ji,jj) ) &
                        &           / MAX( zepsi, zmV_t(ji,jj) * ( zbetav + 1._wp ) + zTauO - zTauB )
                  ENDIF
                  !
               ELSE                 !--- ice velocity using EVP implicit formulation (cf Madec doc & Bouillon 2009)
                  !
                  IF( ( zRHS + ztauy_base(ji,jj) ) < 0._wp .AND. zRHS >= 0._wp ) THEN ! static friction => slow decrease to v=0
                     v_ice(ji,jj) = v_ice(ji,jj) * MAX( 0._wp, 1._wp - zdtevp * rn_lf_relax )
                  ELSE
                     v_ice(ji,jj) = ( zmV_t(ji,jj) * v_ice(ji,jj) + zRHS + zTauO * v_ice(ji,jj) ) &
                        &           / MAX( zepsi, zmV_t(ji,jj) + zTauO - zTauB )
                  ENDIF
                  !
               ENDIF
               ! v_ice = v_oce/100 if mass < zmmin & conc < zamin
               v_ice(ji,jj) = ( v_ice(ji,jj)*zmsk01y(ji,jj) + v_oce(ji,jj) * 0.01_wp * (1._wp - zmsk01y(ji,jj)) ) * zmsk00y(ji,jj)
               ! Reduce value of v_ice drastically where masks for landfast or grounded icebergs are /=0
               ! In theory one could set v_ice=0 but it may lead to instabilities (cf Noe Pirlet)
               v_ice(ji,jj) = v_ice(ji,jj) * ( 1._wp - 0.99_wp * fast_vmask(ji,jj) ) !!* ( 1._wp - 0.99_wp * icb_vmask(ji,jj) ) 
            END_2D
            !
            !
            DO_2D( 0, 0, 0, 0 )
               !                 !--- tau_io/(u_oce - u_ice)
               zTauO = zdragx_oi(ji,jj) *  &
                  &    SQRT( ( u_ice (ji,jj) - u_oce (ji,jj) ) * ( u_ice (ji,jj) - u_oce (ji,jj) )  &
                  &        + ( v_iceU(ji,jj) - v_oceU(ji,jj) ) * ( v_iceU(ji,jj) - v_oceU(ji,jj) ) )
               !                 !--- Ocean-to-Ice stress
               ztaux_oi(ji,jj) = zTauO * ( u_oce(ji,jj) - u_ice(ji,jj) )
               !
               !                 !--- tau_bottom/u_ice
               zvel  = 5.e-05_wp + SQRT( v_iceU(ji,jj) * v_iceU(ji,jj) + u_ice(ji,jj) * u_ice(ji,jj) )
               zTauB = ztaux_base(ji,jj) / zvel
               !                 !--- OceanBottom-to-Ice stress
               ztaux_bi(ji,jj) = zTauB * u_ice(ji,jj)
               !
               !                 !--- Coriolis at U-points (energy conserving formulation)
               zCorU(ji,jj)  =   0.25_wp * r1_e1u(ji,jj) *  &
                  &    ( zmf(ji  ,jj) * ( e1v(ji  ,jj) * v_ice(ji  ,jj) + e1v(ji  ,jj-1) * v_ice(ji  ,jj-1) )  &
                  &    + zmf(ji+1,jj) * ( e1v(ji+1,jj) * v_ice(ji+1,jj) + e1v(ji+1,jj-1) * v_ice(ji+1,jj-1) ) )
               !
               !                 !--- Sum of external forces (explicit solution) = F + tau_ia + Coriolis + spg + tau_io
               zRHS = zfU(ji,jj) + ztaux_ai(ji,jj) + zCorU(ji,jj) + zspgU(ji,jj) + ztaux_oi(ji,jj)
               !
               IF( ln_aEVP ) THEN !--- ice velocity using aEVP (Kimmritz et al 2016 & 2017)
                  !
                  zbetau = MAX( zbeta(ji,jj), zbeta(ji+1,jj) )
                  !
                  IF( ( zRHS + ztaux_base(ji,jj) ) < 0._wp .AND. zRHS >= 0._wp ) THEN ! static friction => slow decrease to v=0
                     u_ice(ji,jj) = ( u_ice_b(ji,jj) + u_ice(ji,jj) * MAX( 0._wp, zbetau - zdtevp*rn_lf_relax ) ) / (zbetau+1._wp)
                  ELSE
                     u_ice(ji,jj) = ( zmU_t(ji,jj) * ( zbetau * u_ice(ji,jj) + u_ice_b(ji,jj) ) + zRHS + zTauO * u_ice(ji,jj) ) &
                        &           / MAX( zepsi, zmU_t(ji,jj) * ( zbetau + 1._wp ) + zTauO - zTauB )
                  ENDIF
                  !
               ELSE               !--- ice velocity using EVP implicit formulation (cf Madec doc & Bouillon 2009)
                  !
                  IF( ( zRHS + ztaux_base(ji,jj) ) < 0._wp .AND. zRHS >= 0._wp ) THEN ! static friction => slow decrease to v=0
                     u_ice(ji,jj) = u_ice(ji,jj) * MAX( 0._wp, 1._wp - zdtevp * rn_lf_relax )
                  ELSE
                     u_ice(ji,jj) = ( zmU_t(ji,jj) * u_ice(ji,jj) + zRHS + zTauO * u_ice(ji,jj) ) &
                        &           / MAX( zepsi, zmU_t(ji,jj) + zTauO - zTauB )
                  ENDIF
                  !
               ENDIF
               ! u_ice = u_oce/100 if mass < zmmin & conc < zamin
               u_ice(ji,jj) = ( u_ice(ji,jj)*zmsk01x(ji,jj) + u_oce(ji,jj) * 0.01_wp * (1._wp - zmsk01x(ji,jj)) ) * zmsk00x(ji,jj)
               ! Reduce value of u_ice drastically where masks for landfast or grounded icebergs are /=0
               ! In theory one could set u_ice=0 but it may lead to instabilities (cf Noe Pirlet)
               u_ice(ji,jj) = u_ice(ji,jj) * ( 1._wp - 0.99_wp * fast_umask(ji,jj) ) !!* ( 1._wp - 0.99_wp * icb_umask(ji,jj) ) 
               !
            END_2D
            !
            CALL lbc_lnk( 'icedyn_rhg_evp', u_ice, 'U', -1.0_wp, v_ice, 'V', -1.0_wp, ldfull=.true. )
            !
         ELSE ! odd iterations
            !
            DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
               !                 !--- tau_io/(u_oce - u_ice)
               zTauO = zdragx_oi(ji,jj) *  &
                  &    SQRT( ( u_ice (ji,jj) - u_oce (ji,jj) ) * ( u_ice (ji,jj) - u_oce (ji,jj) )  &
                  &        + ( v_iceU(ji,jj) - v_oceU(ji,jj) ) * ( v_iceU(ji,jj) - v_oceU(ji,jj) ) )
               !                 !--- Ocean-to-Ice stress
               ztaux_oi(ji,jj) = zTauO * ( u_oce(ji,jj) - u_ice(ji,jj) )
               !
               !                 !--- tau_bottom/u_ice
               zvel  = 5.e-05_wp + SQRT( v_iceU(ji,jj) * v_iceU(ji,jj) + u_ice(ji,jj) * u_ice(ji,jj) )
               zTauB = ztaux_base(ji,jj) / zvel
               !                 !--- OceanBottom-to-Ice stress
               ztaux_bi(ji,jj) = zTauB * u_ice(ji,jj)
               !
               !                 !--- Coriolis at U-points (energy conserving formulation)
               zCorU(ji,jj)  =   0.25_wp * r1_e1u(ji,jj) *  &
                  &    ( zmf(ji  ,jj) * ( e1v(ji  ,jj) * v_ice(ji  ,jj) + e1v(ji  ,jj-1) * v_ice(ji  ,jj-1) )  &
                  &    + zmf(ji+1,jj) * ( e1v(ji+1,jj) * v_ice(ji+1,jj) + e1v(ji+1,jj-1) * v_ice(ji+1,jj-1) ) )
               !
               !                 !--- Sum of external forces (explicit solution) = F + tau_ia + Coriolis + spg + tau_io
               zRHS = zfU(ji,jj) + ztaux_ai(ji,jj) + zCorU(ji,jj) + zspgU(ji,jj) + ztaux_oi(ji,jj)
               !
               IF( ln_aEVP ) THEN !--- ice velocity using aEVP (Kimmritz et al 2016 & 2017)
                  !
                  zbetau = MAX( zbeta(ji,jj), zbeta(ji+1,jj) )
                  !
                  IF( ( zRHS + ztaux_base(ji,jj) ) < 0._wp .AND. zRHS >= 0._wp ) THEN ! static friction => slow decrease to v=0
                     u_ice(ji,jj) = ( u_ice_b(ji,jj) + u_ice(ji,jj) * MAX( 0._wp, zbetau - zdtevp*rn_lf_relax ) ) / (zbetau+1._wp)
                  ELSE
                     u_ice(ji,jj) = ( zmU_t(ji,jj) * ( zbetau * u_ice(ji,jj) + u_ice_b(ji,jj) ) + zRHS + zTauO * u_ice(ji,jj) ) &
                        &           / MAX( zepsi, zmU_t(ji,jj) * ( zbetau + 1._wp ) + zTauO - zTauB )
                  ENDIF
                  !
               ELSE               !--- ice velocity using EVP implicit formulation (cf Madec doc & Bouillon 2009)
                  !
                  IF( ( zRHS + ztaux_base(ji,jj) ) < 0._wp .AND. zRHS >= 0._wp ) THEN ! static friction => slow decrease to v=0
                     u_ice(ji,jj) = u_ice(ji,jj) * MAX( 0._wp, 1._wp - zdtevp * rn_lf_relax )
                  ELSE
                     u_ice(ji,jj) = ( zmU_t(ji,jj) * u_ice(ji,jj) + zRHS + zTauO * u_ice(ji,jj) ) &
                        &           / MAX( zepsi, zmU_t(ji,jj) + zTauO - zTauB )
                  ENDIF
                  !
               ENDIF
               ! u_ice = u_oce/100 if mass < zmmin & conc < zamin
               u_ice(ji,jj) = ( u_ice(ji,jj)*zmsk01x(ji,jj) + u_oce(ji,jj) * 0.01_wp * (1._wp - zmsk01x(ji,jj)) ) * zmsk00x(ji,jj)
               ! Reduce value of u_ice drastically where masks for landfast or grounded icebergs are /=0
               ! In theory one could set u_ice=0 but it may lead to instabilities (cf Noe Pirlet)
               u_ice(ji,jj) = u_ice(ji,jj) * ( 1._wp - 0.99_wp * fast_umask(ji,jj) ) !!* ( 1._wp - 0.99_wp * icb_umask(ji,jj) ) 
               !
            END_2D
            !
            !
            DO_2D( 0, 0, 0, 0 )
               !                 !--- tau_io/(v_oce - v_ice)
               zTauO = zdragy_oi(ji,jj) *  & 
                  &    SQRT( ( v_ice (ji,jj) - v_oce (ji,jj) ) * ( v_ice (ji,jj) - v_oce (ji,jj) )  &
                  &        + ( u_iceV(ji,jj) - u_oceV(ji,jj) ) * ( u_iceV(ji,jj) - u_oceV(ji,jj) ) )
               !                 !--- Ocean-to-Ice stress
               ztauy_oi(ji,jj) = zTauO * ( v_oce(ji,jj) - v_ice(ji,jj) )
               !
               !                 !--- tau_bottom/v_ice
               zvel  = 5.e-05_wp + SQRT( v_ice(ji,jj) * v_ice(ji,jj) + u_iceV(ji,jj) * u_iceV(ji,jj) )
               zTauB = ztauy_base(ji,jj) / zvel
               !                 !--- OceanBottom-to-Ice stress
               ztauy_bi(ji,jj) = zTauB * v_ice(ji,jj)
               !
               !                 !--- Coriolis at v-points (energy conserving formulation)
               zCorV(ji,jj)  = - 0.25_wp * r1_e2v(ji,jj) *  &
                  &    ( zmf(ji,jj  ) * ( e2u(ji,jj  ) * u_ice(ji,jj  ) + e2u(ji-1,jj  ) * u_ice(ji-1,jj  ) )  &
                  &    + zmf(ji,jj+1) * ( e2u(ji,jj+1) * u_ice(ji,jj+1) + e2u(ji-1,jj+1) * u_ice(ji-1,jj+1) ) )
               !
               !                 !--- Sum of external forces (explicit solution) = F + tau_ia + Coriolis + spg + tau_io
               zRHS = zfV(ji,jj) + ztauy_ai(ji,jj) + zCorV(ji,jj) + zspgV(ji,jj) + ztauy_oi(ji,jj)
               !
               IF( ln_aEVP ) THEN !--- ice velocity using aEVP (Kimmritz et al 2016 & 2017)
                  !
                  zbetav = MAX( zbeta(ji,jj), zbeta(ji,jj+1) )
                  !
                  IF( ( zRHS + ztauy_base(ji,jj) ) < 0._wp .AND. zRHS >= 0._wp ) THEN ! static friction => slow decrease to v=0
                     v_ice(ji,jj) = ( v_ice_b(ji,jj) + v_ice(ji,jj) * MAX( 0._wp, zbetav - zdtevp*rn_lf_relax ) ) / (zbetav+1._wp)
                  ELSE
                     v_ice(ji,jj) = ( zmV_t(ji,jj) * ( zbetav * v_ice(ji,jj) + v_ice_b(ji,jj) ) + zRHS + zTauO * v_ice(ji,jj) ) &
                        &           / MAX( zepsi, zmV_t(ji,jj) * ( zbetav + 1._wp ) + zTauO - zTauB )
                  ENDIF
                  !
               ELSE               !--- ice velocity using EVP implicit formulation (cf Madec doc & Bouillon 2009)
                  !
                  IF( ( zRHS + ztauy_base(ji,jj) ) < 0._wp .AND. zRHS >= 0._wp ) THEN ! static friction => slow decrease to v=0
                     v_ice(ji,jj) = v_ice(ji,jj) * MAX( 0._wp, 1._wp - zdtevp * rn_lf_relax )
                  ELSE
                     v_ice(ji,jj) = ( zmV_t(ji,jj) * v_ice(ji,jj) + zRHS + zTauO * v_ice(ji,jj) ) &
                        &           / MAX( zepsi, zmV_t(ji,jj) + zTauO - zTauB )
                  ENDIF
                  !
               ENDIF
               ! v_ice = v_oce/100 if mass < zmmin & conc < zamin
               v_ice(ji,jj) = ( v_ice(ji,jj)*zmsk01y(ji,jj) + v_oce(ji,jj) * 0.01_wp * (1._wp - zmsk01y(ji,jj)) ) * zmsk00y(ji,jj)
               ! Reduce value of v_ice drastically where masks for landfast or grounded icebergs are /=0
               ! In theory one could set v_ice=0 but it may lead to instabilities (cf Noe Pirlet)
               v_ice(ji,jj) = v_ice(ji,jj) * ( 1._wp - 0.99_wp * fast_vmask(ji,jj) ) !!* ( 1._wp - 0.99_wp * icb_vmask(ji,jj) ) 
               !
            END_2D
            !
            CALL lbc_lnk( 'icedyn_rhg_evp', u_ice, 'U', -1.0_wp, v_ice, 'V', -1.0_wp, ldfull=.true. )
            !
         ENDIF
         !
#if defined key_agrif
!!       CALL agrif_interp_ice( 'U', jter, nn_nevp )
!!       CALL agrif_interp_ice( 'V', jter, nn_nevp )
         CALL agrif_interp_ice( 'U' )
         CALL agrif_interp_ice( 'V' )
#endif
         IF( ln_bdy )   CALL bdy_ice_dyn( 'U' )
         IF( ln_bdy )   CALL bdy_ice_dyn( 'V' )
         !
         ! convergence test
         IF( nn_rhg_chkcvg == 2 )   CALL rhg_cvg( kt, jter, nn_nevp, u_ice, v_ice, zu_ice, zv_ice, zmsk15 )
         !
         !
         ! --- change strength according to advected a_i and v_i (upstream for now) --- !
         IF( ll_advups .AND. ln_str_H79 ) THEN
            !
            IF( jter == 1 ) THEN                               ! init
               ALLOCATE( za_i_ups(jpi,jpj,jpl), zv_i_ups(jpi,jpj,jpl) )
               zdt_ups = rDt_ice / REAL( nn_nevp )
               za_i_ups(:,:,:) = a_i(:,:,:)
               zv_i_ups(:,:,:) = v_i(:,:,:)
            ELSE
               CALL lbc_lnk( 'icedyn_rhg_evp', za_i_ups, 'T', 1.0_wp, zv_i_ups, 'T', 1.0_wp )               
            ENDIF
            !
            CALL rhg_upstream( jter, zdt_ups, u_ice, v_ice, za_i_ups )   ! upstream advection: a_i
            CALL rhg_upstream( jter, zdt_ups, u_ice, v_ice, zv_i_ups )   ! upstream advection: v_i
            !
            DO_2D( 0, 0, 0, 0 )    ! strength
               strength(ji,jj) = rn_pstar * SUM( zv_i_ups(ji,jj,:) ) * EXP( -rn_crhg * ( 1._wp - SUM( za_i_ups(ji,jj,:) ) ) )
            END_2D
            !
            IF( jter == nn_nevp ) THEN
               DEALLOCATE( za_i_ups, zv_i_ups )
            ENDIF
         ENDIF
         !                                                ! ==================== !
      END DO                                              !  end loop over jter  !
      !                                                   ! ==================== !
      IF( ln_aEVP )   CALL iom_put( 'beta_evp' , zbeta )
      !
      IF( ll_advups .AND. ln_str_H79 )   CALL lbc_lnk( 'icedyn_rhg_evp', strength, 'T', 1.0_wp )
      !
      !------------------------------------------------------------------------------!
      ! 4) Recompute delta, shear and div (inputs for mechanical redistribution)
      !------------------------------------------------------------------------------!
      DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )

         ! shear at F points
         zds(ji,jj) = ( ( u_ice(ji,jj+1) * r1_e1u(ji,jj+1) - u_ice(ji,jj) * r1_e1u(ji,jj) ) * e1f(ji,jj) * e1f(ji,jj)   &
            &         + ( v_ice(ji+1,jj) * r1_e2v(ji+1,jj) - v_ice(ji,jj) * r1_e2v(ji,jj) ) * e2f(ji,jj) * e2f(ji,jj)   &
            &         ) * r1_e1e2f(ji,jj) * fimask(ji,jj)

      END_2D

      DO_2D( 0, 0, 0, 0 )   ! no vector loop

         ! tension**2 at T points
         zdt  = ( ( u_ice(ji,jj) * r1_e2u(ji,jj) - u_ice(ji-1,jj) * r1_e2u(ji-1,jj) ) * e2t(ji,jj) * e2t(ji,jj)   &
            &   - ( v_ice(ji,jj) * r1_e1v(ji,jj) - v_ice(ji,jj-1) * r1_e1v(ji,jj-1) ) * e1t(ji,jj) * e1t(ji,jj)   &
            &   ) * r1_e1e2t(ji,jj) * (1._wp - fast_tmask(ji,jj)) * (1._wp - icb_tmask(ji,jj))
         zdt2 = zdt * zdt

         zten_i(ji,jj) = zdt

         ! shear**2 at T points (doc eq. A16)
         zds2 = ( ( zds(ji,jj  ) * zds(ji,jj  ) * e1e2f(ji,jj  ) + zds(ji-1,jj  ) * zds(ji-1,jj  ) * e1e2f(ji-1,jj  ) )  & ! add () 
            &   + ( zds(ji,jj-1) * zds(ji,jj-1) * e1e2f(ji,jj-1) + zds(ji-1,jj-1) * zds(ji-1,jj-1) * e1e2f(ji-1,jj-1) )  & ! NP rep
            &   ) * 0.25_wp * r1_e1e2t(ji,jj) * (1._wp - fast_tmask(ji,jj)) * (1._wp - icb_tmask(ji,jj))

         ! maximum shear rate at T points (includes tension, output only)
         pshear_i(ji,jj) = SQRT( zdt2 + zds2 ) * zmsk(ji,jj) 

         ! shear at T-points
         zshear(ji,jj)   = SQRT( zds2 ) * zmsk(ji,jj)

         ! divergence at T points
         pdivu_i(ji,jj) = ( ( e2u(ji,jj) * u_ice(ji,jj) - e2u(ji-1,jj) * u_ice(ji-1,jj) )  & ! add () for NP repro
            &             + ( e1v(ji,jj) * v_ice(ji,jj) - e1v(ji,jj-1) * v_ice(ji,jj-1) )  &
            &             ) * r1_e1e2t(ji,jj) * zmsk(ji,jj) * (1._wp - fast_tmask(ji,jj)) * (1._wp - icb_tmask(ji,jj))

         ! delta at T points
         zdelta(ji,jj)   = SQRT( pdivu_i(ji,jj) * pdivu_i(ji,jj) + ( zdt2 + zds2 ) * z1_ecc2 ) * zmsk(ji,jj) 

         ! delta* at T points (pdelta_i)
         IF( zdelta(ji,jj) > 0._wp ) THEN   ;   pdelta_i(ji,jj) = zdelta(ji,jj) !!clem + rn_creepl
         ELSE                               ;   pdelta_i(ji,jj) = 0._wp
         ENDIF
                           ! it seems that deformation used for advection and mech redistribution is delta*
                           ! MV in principle adding creep limit is a regularization for viscosity not for delta
                           ! delta_star should not (in my view) be used in a replacement for delta
      END_2D

      CALL lbc_lnk( 'icedyn_rhg_evp', zs1   , 'T', 1._wp, zs2   , 'T', 1._wp, zs12  , 'F', 1._wp )

      ! --- Store the stress tensor for the next time step --- !
      pstress1_i (:,:) = zs1 (:,:)
      pstress2_i (:,:) = zs2 (:,:)
      pstress12_i(:,:) = zs12(:,:)
      !
      ! 5) diagnostics
      !------------------------------------------------------------------------------!
      ! --- ice-ocean, ice-atm. & ice-oceanbottom(landfast) stresses --- !
      IF( iom_use('utau_oi') )   CALL iom_put( 'utau_oi' , ztaux_oi(A2D(0)) * zmsk00 )
      IF( iom_use('vtau_oi') )   CALL iom_put( 'vtau_oi' , ztauy_oi(A2D(0)) * zmsk00 )
      IF( iom_use('utau_ai') )   CALL iom_put( 'utau_ai' , ztaux_ai(A2D(0)) * zmsk00 )
      IF( iom_use('vtau_ai') )   CALL iom_put( 'vtau_ai' , ztauy_ai(A2D(0)) * zmsk00 )
      IF( iom_use('utau_bi') )   CALL iom_put( 'utau_bi' , ztaux_bi(A2D(0)) * zmsk00 )
      IF( iom_use('vtau_bi') )   CALL iom_put( 'vtau_bi' , ztauy_bi(A2D(0)) * zmsk00 )

      ! --- divergence, shear and strength --- !
      IF( iom_use('icediv') )   CALL iom_put( 'icediv' , pdivu_i (A2D(0)) * zmsk00 )   ! divergence
      IF( iom_use('iceshe') )   CALL iom_put( 'iceshe' , pshear_i(A2D(0)) * zmsk00 )   ! shear
      IF( iom_use('icestr') )   CALL iom_put( 'icestr' , strength(A2D(0)) * zmsk00 )   ! strength
      IF( iom_use('icedlt') )   CALL iom_put( 'icedlt' , zdelta  (A2D(0)) * zmsk00 )   ! delta

      ! --- Stress tensor invariants (SIMIP diags) --- !
      IF( iom_use('normstr') .OR. iom_use('sheastr') ) THEN
         !
         ALLOCATE( zsig_I(A2D(0)) , zsig_II(A2D(0)) )
         !
         DO_2D( 0, 0, 0, 0 )
            ! Ice stresses
            ! sigma1, sigma2, sigma12 are some recombination of the stresses (HD MWR002, Bouillon et al., OM2013)
            ! not to be confused with stress tensor components, stress invariants, or stress principal components
            zfac             =   strength(ji,jj) / ( zdelta(ji,jj) + rn_creepl )          ! viscosity
            zsig1            =   zfac * ( pdivu_i(ji,jj) - zdelta(ji,jj) )
            zsig2            =   zfac * z1_ecc2 * zten_i(ji,jj)
            zsig12           =   zfac * z1_ecc2 * zshear(ji,jj) * 0.5_wp

            ! Stress invariants (sigma_I, sigma_II, Coon 1974, Feltham 2008)
            zsig_I (ji,jj)   =   0.5_wp * zsig1 
            zsig_II(ji,jj)   =   0.5_wp * SQRT ( zsig2 * zsig2 + 4._wp * zsig12 * zsig12 )

         END_2D
         !
         IF( iom_use('normstr') )   CALL iom_put( 'normstr', zsig_I (:,:) * zmsk00 ) ! Normal stress
         IF( iom_use('sheastr') )   CALL iom_put( 'sheastr', zsig_II(:,:) * zmsk00 ) ! Maximum shear stress

         DEALLOCATE ( zsig_I, zsig_II )

      ENDIF

      ! --- Normalized stress tensor principal components --- !
      ! This are used to plot the normalized yield curve, see Lemieux & Dupont, 2020
      ! Recommendation 1 : we use ice strength, not replacement pressure
      ! Recommendation 2 : for EVP, no need to use viscosities at last iteration (stress is properly iterated)
      IF( iom_use('sig1_pnorm') .OR. iom_use('sig2_pnorm') ) THEN
         !
         ALLOCATE( zsig1_p(A2D(0)) , zsig2_p(A2D(0)) , zsig_I(A2D(0)) , zsig_II(A2D(0)) )
         !
         DO_2D( 0, 0, 0, 0 )

            ! For EVP solvers, ice stresses at current iterates can be used
            !                        following Lemieux & Dupont (2020)
            zfac             =   strength(ji,jj) / ( zdelta(ji,jj) + rn_creepl )
            zsig1            =   zfac * ( pdivu_i(ji,jj) - zdelta(ji,jj) )
            zsig2            =   zfac * z1_ecc2 * zten_i(ji,jj)
            zsig12           =   zfac * z1_ecc2 * zshear(ji,jj) * 0.5_wp

            ! Stress invariants (sigma_I, sigma_II, Coon 1974, Feltham 2008), T-point
            zsig_I(ji,jj)    =   0.5_wp * zsig1                                         ! normal stress
            zsig_II(ji,jj)   =   0.5_wp * SQRT ( zsig2 * zsig2 + 4._wp * zsig12 * zsig12 ) ! max shear stress

            ! Normalized  principal stresses (used to display the ellipse)
            z1_strength      =   1._wp / MAX( 1._wp, strength(ji,jj) )
            zsig1_p(ji,jj)   =   ( zsig_I(ji,jj) + zsig_II(ji,jj) ) * z1_strength
            zsig2_p(ji,jj)   =   ( zsig_I(ji,jj) - zsig_II(ji,jj) ) * z1_strength
         END_2D
         !
         CALL iom_put( 'sig1_pnorm' , zsig1_p(:,:) * zmsk00 )
         CALL iom_put( 'sig2_pnorm' , zsig2_p(:,:) * zmsk00 )

         DEALLOCATE( zsig1_p , zsig2_p , zsig_I, zsig_II )

      ENDIF

      ! --- SIMIP --- !
      IF( iom_use('dssh_dx') )   CALL iom_put( 'dssh_dx' , zspgU(A2D(0)) * zmsk00 )   ! Sea-surface tilt term in force balance (x)
      IF( iom_use('dssh_dy') )   CALL iom_put( 'dssh_dy' , zspgV(A2D(0)) * zmsk00 )   ! Sea-surface tilt term in force balance (y)
      IF( iom_use('corstrx') )   CALL iom_put( 'corstrx' , zCorU(A2D(0)) * zmsk00 )   ! Coriolis force term in force balance (x)
      IF( iom_use('corstry') )   CALL iom_put( 'corstry' , zCorV(A2D(0)) * zmsk00 )   ! Coriolis force term in force balance (y)
      IF( iom_use('intstrx') )   CALL iom_put( 'intstrx' , zfU  (A2D(0)) * zmsk00 )   ! Internal force term in force balance (x)
      IF( iom_use('intstry') )   CALL iom_put( 'intstry' , zfV  (A2D(0)) * zmsk00 )   ! Internal force term in force balance (y)

      IF(  iom_use('xmtrpice') .OR. iom_use('ymtrpice') .OR. &
         & iom_use('xmtrpsnw') .OR. iom_use('ymtrpsnw') .OR. iom_use('xatrp') .OR. iom_use('yatrp') ) THEN
         !
         ALLOCATE( zdiag_xmtrp_ice(A2D(0)) , zdiag_ymtrp_ice(A2D(0)) , &
            &      zdiag_xmtrp_snw(A2D(0)) , zdiag_ymtrp_snw(A2D(0)) , zdiag_xatrp(A2D(0)) , zdiag_yatrp(A2D(0)) )
         !
         DO_2D( 0, 0, 0, 0 )
            ! 2D ice mass, snow mass, area transport arrays (X, Y)
            zfac_x = 0.5 * u_ice(ji,jj) * e2u(ji,jj) * zmsk00(ji,jj)
            zfac_y = 0.5 * v_ice(ji,jj) * e1v(ji,jj) * zmsk00(ji,jj)

            zdiag_xmtrp_ice(ji,jj) = rhoi * zfac_x * ( vt_i(ji+1,jj) + vt_i(ji,jj) ) ! ice mass transport, X-component
            zdiag_ymtrp_ice(ji,jj) = rhoi * zfac_y * ( vt_i(ji,jj+1) + vt_i(ji,jj) ) !        ''           Y-   ''

            zdiag_xmtrp_snw(ji,jj) = rhos * zfac_x * ( vt_s(ji+1,jj) + vt_s(ji,jj) ) ! snow mass transport, X-component
            zdiag_ymtrp_snw(ji,jj) = rhos * zfac_y * ( vt_s(ji,jj+1) + vt_s(ji,jj) ) !          ''          Y-   ''

            zdiag_xatrp(ji,jj)     = zfac_x * ( at_i(ji+1,jj) + at_i(ji,jj) )        ! area transport,      X-component
            zdiag_yatrp(ji,jj)     = zfac_y * ( at_i(ji,jj+1) + at_i(ji,jj) )        !        ''            Y-   ''

         END_2D

         CALL iom_put( 'xmtrpice' , zdiag_xmtrp_ice )   ! X-component of sea-ice mass transport (kg/s)
         CALL iom_put( 'ymtrpice' , zdiag_ymtrp_ice )   ! Y-component of sea-ice mass transport
         CALL iom_put( 'xmtrpsnw' , zdiag_xmtrp_snw )   ! X-component of snow mass transport (kg/s)
         CALL iom_put( 'ymtrpsnw' , zdiag_ymtrp_snw )   ! Y-component of snow mass transport
         CALL iom_put( 'xatrp'    , zdiag_xatrp     )   ! X-component of ice area transport
         CALL iom_put( 'yatrp'    , zdiag_yatrp     )   ! Y-component of ice area transport

         DEALLOCATE( zdiag_xmtrp_ice , zdiag_ymtrp_ice , &
            &        zdiag_xmtrp_snw , zdiag_ymtrp_snw , zdiag_xatrp , zdiag_yatrp )

      ENDIF
      !
      ! --- convergence tests --- !
      IF( nn_rhg_chkcvg == 1 .OR. nn_rhg_chkcvg == 2 ) THEN
         IF( iom_use('uice_cvg') ) THEN
            IF( ln_aEVP ) THEN   ! output: beta * ( u(t=nn_nevp) - u(t=nn_nevp-1) )
               CALL iom_put( 'uice_cvg', MAX( ABS( u_ice(A2D(0)) - zu_ice(:,:) ) * zbeta(A2D(0)) * umask(A2D(0),1) , &
                  &                           ABS( v_ice(A2D(0)) - zv_ice(:,:) ) * zbeta(A2D(0)) * vmask(A2D(0),1) ) * zmsk15(:,:) )
            ELSE                 ! output: nn_nevp * ( u(t=nn_nevp) - u(t=nn_nevp-1) )
               CALL iom_put( 'uice_cvg', REAL( nn_nevp ) * MAX( ABS( u_ice(A2D(0)) - zu_ice(:,:) ) * umask(A2D(0),1) , &
                  &                                             ABS( v_ice(A2D(0)) - zv_ice(:,:) ) * vmask(A2D(0),1) ) * zmsk15(:,:) )
            ENDIF
         ENDIF
      ENDIF
      !
   END SUBROUTINE ice_dyn_rhg_evp


   SUBROUTINE rhg_cvg( kt, kiter, kitermax, pu, pv, pub, pvb, pmsk15 )
      !!----------------------------------------------------------------------
      !!                    ***  ROUTINE rhg_cvg  ***
      !!
      !! ** Purpose :   check convergence of oce rheology
      !!
      !! ** Method  :   create a file ice_cvg.nc containing the convergence of ice velocity
      !!                during the sub timestepping of rheology so as:
      !!                  uice_cvg = MAX( u(t+1) - u(t) , v(t+1) - v(t) )
      !!                This routine is called every sub-iteration, so it is cpu expensive
      !!
      !! ** Note    :   for the first sub-iteration, uice_cvg is set to 0 (too large otherwise)
      !!----------------------------------------------------------------------
      INTEGER ,                    INTENT(in) ::   kt, kiter, kitermax       ! ocean time-step index
      REAL(wp), DIMENSION(:,:)   , INTENT(in) ::   pu, pv                    ! now velocities
      REAL(wp), DIMENSION(A2D(0)), INTENT(in) ::   pub, pvb                  ! before velocities
      REAL(wp), DIMENSION(A2D(0)), INTENT(in) ::   pmsk15
      !!
      INTEGER           ::   it, idtime, istatus
      INTEGER           ::   ji, jj          ! dummy loop indices
      REAL(wp)          ::   zresm           ! local real
      CHARACTER(len=20) ::   clname
      LOGICAL           ::   ll_maxcvg
      REAL(wp), DIMENSION(A2D(0),2) ::   zres
      REAL(wp), DIMENSION(2)        ::   ztmp
      !!----------------------------------------------------------------------
      ll_maxcvg = .FALSE.
      !
      ! create file
      IF( kt == nit000 .AND. kiter == 1 ) THEN
         !
         IF( lwp ) THEN
            WRITE(numout,*)
            WRITE(numout,*) 'rhg_cvg : ice rheology convergence control'
            WRITE(numout,*) '~~~~~~~'
         ENDIF
         !
         IF( lwm ) THEN
            clname = 'ice_cvg.nc'
            IF( .NOT. Agrif_Root() )   clname = TRIM(Agrif_CFixed())//"_"//TRIM(clname)
            istatus = NF90_CREATE( TRIM(clname), NF90_CLOBBER, ncvgid )
            istatus = NF90_DEF_DIM( ncvgid, 'time'  , NF90_UNLIMITED, idtime )
            istatus = NF90_DEF_VAR( ncvgid, 'uice_cvg', NF90_DOUBLE , (/ idtime /), nvarid )
            istatus = NF90_ENDDEF(ncvgid)
         ENDIF
         !
      ENDIF

      ! time
      it = ( kt - nit000 ) * kitermax + kiter

      ! convergence
      IF( kiter == 1 ) THEN ! remove the first iteration for calculations of convergence (always very large)
         zresm = 0._wp
      ELSE
         zresm = 0._wp
         IF( ll_maxcvg ) THEN   ! error max over the domain
            DO_2D( 0, 0, 0, 0 )
               zresm = MAX( zresm, MAX( ABS( pu(ji,jj) - pub(ji,jj) ) * umask(ji,jj,1), &
                  &                     ABS( pv(ji,jj) - pvb(ji,jj) ) * vmask(ji,jj,1) ) * pmsk15(ji,jj) )
            END_2D
            CALL mpp_max( 'icedyn_rhg_evp', zresm )
         ELSE                   ! error averaged over the domain
            DO_2D( 0, 0, 0, 0 )
               zres(ji,jj,1) = MAX( ABS( pu(ji,jj) - pub(ji,jj) ) * umask(ji,jj,1), &
                  &                 ABS( pv(ji,jj) - pvb(ji,jj) ) * vmask(ji,jj,1) ) * pmsk15(ji,jj)
               zres(ji,jj,2) = pmsk15(ji,jj)
            END_2D
            ztmp(:) = glob_2Dsum( 'icedyn_rhg_evp', zres )
            IF( ztmp(2) /= 0._wp )   zresm = ztmp(1) / ztmp(2)
         ENDIF
      ENDIF

      IF( lwm ) THEN
         ! write variables
         istatus = NF90_PUT_VAR( ncvgid, nvarid, (/zresm/), (/it/), (/1/) )
         ! close file
         IF( kt == nitend - nn_fsbc + 1 .AND. kiter == kitermax )   istatus = NF90_CLOSE(ncvgid)
      ENDIF

   END SUBROUTINE rhg_cvg


   SUBROUTINE rhg_evp_rst( cdrw, kt )
      !!---------------------------------------------------------------------
      !!                   ***  ROUTINE rhg_evp_rst  ***
      !!
      !! ** Purpose :   Read or write RHG file in restart file
      !!
      !! ** Method  :   use of IOM library
      !!----------------------------------------------------------------------
      CHARACTER(len=*) , INTENT(in) ::   cdrw   ! "READ"/"WRITE" flag
      INTEGER, OPTIONAL, INTENT(in) ::   kt     ! ice time-step
      !
      INTEGER  ::   iter            ! local integer
      INTEGER  ::   id1, id2, id3   ! local integers
      !!----------------------------------------------------------------------
      !
      IF( TRIM(cdrw) == 'READ' ) THEN        ! Read/initialize
         !                                   ! ---------------
         IF( ln_rstart ) THEN                   !* Read the restart file
            !
            id1 = iom_varid( numrir, 'stress1_i' , ldstop = .FALSE. )
            id2 = iom_varid( numrir, 'stress2_i' , ldstop = .FALSE. )
            id3 = iom_varid( numrir, 'stress12_i', ldstop = .FALSE. )
            !
            IF( MIN( id1, id2, id3 ) > 0 ) THEN      ! fields exist
               CALL iom_get( numrir, jpdom_auto, 'stress1_i' , stress1_i , cd_type = 'T' )
               CALL iom_get( numrir, jpdom_auto, 'stress2_i' , stress2_i , cd_type = 'T' )
               CALL iom_get( numrir, jpdom_auto, 'stress12_i', stress12_i, cd_type = 'F' )
            ELSE                                     ! start rheology from rest
               IF(lwp) WRITE(numout,*)
               IF(lwp) WRITE(numout,*) '   ==>>>   previous run without rheology, set stresses to 0'
               stress1_i (:,:) = 0._wp
               stress2_i (:,:) = 0._wp
               stress12_i(:,:) = 0._wp
            ENDIF
         ELSE                                   !* Start from rest
            IF(lwp) WRITE(numout,*)
            IF(lwp) WRITE(numout,*) '   ==>>>   start from rest: set stresses to 0'
            stress1_i (:,:) = 0._wp
            stress2_i (:,:) = 0._wp
            stress12_i(:,:) = 0._wp
         ENDIF
         !
      ELSEIF( TRIM(cdrw) == 'WRITE' ) THEN   ! Create restart file
         !                                   ! -------------------
         IF(lwp) WRITE(numout,*) '---- rhg-rst ----'
         iter = kt + nn_fsbc - 1             ! ice restarts are written at kt == nitrst - nn_fsbc + 1
         !
         CALL iom_rstput( iter, nitrst, numriw, 'stress1_i' , stress1_i )
         CALL iom_rstput( iter, nitrst, numriw, 'stress2_i' , stress2_i )
         CALL iom_rstput( iter, nitrst, numriw, 'stress12_i', stress12_i )
         !
      ENDIF
      !
   END SUBROUTINE rhg_evp_rst

   SUBROUTINE rhg_upstream( jter, pdt, pu, pv, pt )
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE rhg_upstream  ***
      !!
      !! **  Purpose :   compute the upstream fluxes and upstream guess of tracer
      !!----------------------------------------------------------------------
      INTEGER                    , INTENT(in   ) ::   jter
      REAL(wp)                   , INTENT(in   ) ::   pdt              ! tracer time-step
      REAL(wp), DIMENSION(:,:  ) , INTENT(in   ) ::   pu, pv           ! 2 ice velocity components
      REAL(wp), DIMENSION(:,:,:) , INTENT(inout) ::   pt               ! tracer fields
      !
      INTEGER  ::   ji, jj, jl    ! dummy loop indices
      REAL(wp) ::   ztra          ! local scalar
      LOGICAL  ::   ll_upsxy = .TRUE.
      REAL(wp), DIMENSION(jpi,jpj) ::   zfu_ups, zfv_ups, zpt   ! upstream fluxes and tracer guess
      !!----------------------------------------------------------------------
      DO jl = 1, jpl
         IF( .NOT. ll_upsxy ) THEN         !** no alternate directions **!
            !
            DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )
               zfu_ups(ji,jj) = MAX(pu(ji,jj)*e2u(ji,jj), 0._wp) * pt(ji,jj,jl) + MIN(pu(ji,jj)*e2u(ji,jj), 0._wp) * pt(ji+1,jj,jl)
               zfv_ups(ji,jj) = MAX(pv(ji,jj)*e1v(ji,jj), 0._wp) * pt(ji,jj,jl) + MIN(pv(ji,jj)*e1v(ji,jj), 0._wp) * pt(ji,jj+1,jl)
            END_2D
            !
         ELSE                              !** alternate directions **!
            !
            IF( MOD(jter,2) == 1 ) THEN   !==  odd ice time step:  adv_x then adv_y  ==!
               !
               DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls )       !-- flux in x-direction
                  zfu_ups(ji,jj) = MAX( pu(ji,jj)*e2u(ji,jj), 0._wp ) * pt(ji  ,jj,jl) + &
                     &             MIN( pu(ji,jj)*e2u(ji,jj), 0._wp ) * pt(ji+1,jj,jl)
               END_2D
               !
               DO_2D( nn_hls-1, nn_hls-1, nn_hls, nn_hls )     !-- first guess of tracer from u-flux
                  ztra       = - ( zfu_ups(ji,jj) - zfu_ups(ji-1,jj) )
                  zpt(ji,jj) =   ( pt(ji,jj,jl) + ztra * pdt * r1_e1e2t(ji,jj) ) * tmask(ji,jj,1)
               END_2D
               !
               DO_2D( nn_hls-1, nn_hls-1, nn_hls, nn_hls-1 )   !-- flux in y-direction
                  zfv_ups(ji,jj) = MAX( pv(ji,jj)*e1v(ji,jj), 0._wp ) * zpt(ji,jj  ) + &
                     &             MIN( pv(ji,jj)*e1v(ji,jj), 0._wp ) * zpt(ji,jj+1)
               END_2D
               !
            ELSE                          !==  even ice time step:  adv_y then adv_x  ==!
               !
               DO_2D( nn_hls, nn_hls, nn_hls, nn_hls-1 )       !-- flux in y-direction
                  zfv_ups(ji,jj) = MAX( pv(ji,jj)*e1v(ji,jj), 0._wp ) * pt(ji,jj  ,jl) + &
                     &             MIN( pv(ji,jj)*e1v(ji,jj), 0._wp ) * pt(ji,jj+1,jl)
               END_2D
               !
               DO_2D( nn_hls, nn_hls, nn_hls-1, nn_hls-1 )     !-- first guess of tracer from v-flux
                  ztra       = - ( zfv_ups(ji,jj) - zfv_ups(ji,jj-1) )
                  zpt(ji,jj) =   ( pt(ji,jj,jl) + ztra * pdt * r1_e1e2t(ji,jj) ) * tmask(ji,jj,1)
               END_2D
               !
               DO_2D( nn_hls, nn_hls-1, nn_hls-1, nn_hls-1 )   !-- flux in x-direction
                  zfu_ups(ji,jj) = MAX( pu(ji,jj)*e2u(ji,jj), 0._wp ) * zpt(ji  ,jj) + &
                     &             MIN( pu(ji,jj)*e2u(ji,jj), 0._wp ) * zpt(ji+1,jj)
               END_2D
               !
            ENDIF
            !
         ENDIF
         !
         DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
            ztra         = - (  ( zfu_ups(ji,jj) - zfu_ups(ji-1,jj) )   & ! add () for NP repro
               &              + ( zfv_ups(ji,jj) - zfv_ups(ji,jj-1) )  )
            pt(ji,jj,jl) =   ( pt(ji,jj,jl) + ztra * pdt * r1_e1e2t(ji,jj) ) * tmask(ji,jj,1)
         END_2D
      END DO
      !
   END SUBROUTINE rhg_upstream


   SUBROUTINE l4_orca1ice_dyn_dump( kt, Kmm, ps1, ps2, ps12, pzaU, pzaV, &
      & ptaux_ai, ptauy_ai, pdragx, pdragy, ptaux_base, ptauy_base, pmU, pmV )
      !! WRITE-only dynamics-entry/landfast frame after NEMO has evaluated
      !! ice_strength and the Lemieux-2016 basal-stress operands, but before
      !! the EVP iteration (icedyn_rhg_evp.F90:254-374).
      INTEGER, INTENT(in) :: kt, Kmm
      REAL(wp), DIMENSION(:,:), INTENT(in) :: ps1, ps2, ps12, pzaU, pzaV
      REAL(wp), DIMENSION(:,:), INTENT(in) :: ptaux_ai, ptauy_ai, pdragx, pdragy
      REAL(wp), DIMENSION(:,:), INTENT(in) :: ptaux_base, ptauy_base, pmU, pmV
      INTEGER :: l4_unit, l4_ios, l4_payload, jf
      INTEGER, PARAMETER :: l4_nf = 28, l4_ns = 5
      INTEGER, DIMENSION(3,l4_nf) :: l4_ext
      CHARACTER(LEN=16), PARAMETER :: l4_magic = 'NEMO_L4_O1IDYN1'
      CHARACTER(LEN=96) :: l4_name
      IF( .NOT.lwp ) RETURN

      l4_ext(:, 1) = (/ SHAPE(u_ice), 1 /)
      l4_ext(:, 2) = (/ SHAPE(v_ice), 1 /)
      l4_ext(:, 3) = (/ SHAPE(utau_ice), 1 /)
      l4_ext(:, 4) = (/ SHAPE(vtau_ice), 1 /)
      l4_ext(:, 5) = (/ SHAPE(strength), 1 /)
      l4_ext(:, 6) = (/ SHAPE(at_i), 1 /)
      l4_ext(:, 7) = (/ SHAPE(vt_i), 1 /)
      l4_ext(:, 8) = (/ SHAPE(vt_s), 1 /)
      l4_ext(:, 9) = (/ SHAPE(tau_icebfr), 1 /)
      l4_ext(:,10) = (/ SHAPE(icb_tmask), 1 /)
      l4_ext(:,11) = (/ SHAPE(fast_tmask), 1 /)
      l4_ext(:,12) = (/ SHAPE(pzaU), 1 /)
      l4_ext(:,13) = (/ SHAPE(pzaV), 1 /)
      l4_ext(:,14) = (/ SHAPE(ptaux_ai), 1 /)
      l4_ext(:,15) = (/ SHAPE(ptauy_ai), 1 /)
      l4_ext(:,16) = (/ SHAPE(pdragx), 1 /)
      l4_ext(:,17) = (/ SHAPE(pdragy), 1 /)
      l4_ext(:,18) = (/ SHAPE(ptaux_base), 1 /)
      l4_ext(:,19) = (/ SHAPE(ptauy_base), 1 /)
      l4_ext(:,20) = (/ SHAPE(pmU), 1 /)
      l4_ext(:,21) = (/ SHAPE(pmV), 1 /)
      l4_ext(:,22) = (/ SHAPE(ht(:,:,Kmm)), 1 /)
      l4_ext(:,23) = (/ SHAPE(hu(:,:,Kmm)), 1 /)
      l4_ext(:,24) = (/ SHAPE(hv(:,:,Kmm)), 1 /)
      l4_ext(:,25) = (/ SHAPE(ps1), 1 /)
      l4_ext(:,26) = (/ SHAPE(ps2), 1 /)
      l4_ext(:,27) = (/ SHAPE(ps12), 1 /)
      l4_ext(:,28) = SHAPE(a_i)
      l4_payload = l4_ns
      DO jf = 1, l4_nf
         l4_payload = l4_payload + PRODUCT(l4_ext(:,jf))
      END DO

      WRITE(l4_name,'("oracle_orca1ice_dyn_kt",I8.8,".bin")') kt
      OPEN( NEWUNIT=l4_unit, FILE=TRIM(l4_name), STATUS='REPLACE', ACCESS='STREAM', &
         & FORM='UNFORMATTED', ACTION='WRITE', IOSTAT=l4_ios )
      IF( l4_ios /= 0 ) CALL ctl_stop('l4_orca1ice_dyn_dump: open failed')
      WRITE(l4_unit) l4_magic
      WRITE(l4_unit) 1, kt, Kmm, jpi, jpj, jpl, nlay_i, nlay_s, &
         & STORAGE_SIZE(1._wp), l4_nf, l4_ns, l4_payload
      WRITE(l4_unit) l4_ext
      WRITE(l4_unit) rn_lf_depfra, rn_lf_bfr, rn_lf_relax, rn_lf_tensile, rn_crhg
      WRITE(l4_unit) l4_canon_2d(u_ice,'U'), l4_canon_2d(v_ice,'V'), &
         & l4_canon_2d(utau_ice,'T'), l4_canon_2d(vtau_ice,'T'), &
         & l4_canon_2d(strength,'T'), l4_canon_2d(at_i,'T'), &
         & l4_canon_2d(vt_i,'T'), l4_canon_2d(vt_s,'T'), &
         & l4_canon_2d(tau_icebfr,'T'), l4_canon_2d(icb_tmask,'T'), &
         & l4_canon_2d(fast_tmask,'T'), l4_canon_2d(pzaU,'U'), &
         & l4_canon_2d(pzaV,'V'), l4_canon_2d(ptaux_ai,'U'), &
         & l4_canon_2d(ptauy_ai,'V'), l4_canon_2d(pdragx,'U'), &
         & l4_canon_2d(pdragy,'V'), l4_canon_2d(ptaux_base,'U'), &
         & l4_canon_2d(ptauy_base,'V'), l4_canon_2d(pmU,'U'), &
         & l4_canon_2d(pmV,'V'), l4_canon_2d(ht(:,:,Kmm),'T'), &
         & l4_canon_2d(hu(:,:,Kmm),'U'), l4_canon_2d(hv(:,:,Kmm),'V'), &
         & l4_canon_2d(ps1,'T'), l4_canon_2d(ps2,'T'), &
         & l4_canon_2d(ps12,'F'), l4_canon_3d(a_i,'T')
      CLOSE(l4_unit)
   END SUBROUTINE l4_orca1ice_dyn_dump

#include "l4_oracle_canon_subroutines.h90"

#else
   !!----------------------------------------------------------------------
   !!   Default option         Empty module           NO SI3 sea-ice model
   !!----------------------------------------------------------------------
#endif

   !!==============================================================================
END MODULE icedyn_rhg_evp
