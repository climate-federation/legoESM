MODULE sbcblk
   !!======================================================================
   !!                       ***  MODULE  sbcblk  ***
   !! Ocean forcing:  momentum, heat and freshwater flux formulation
   !!                         Aerodynamic Bulk Formulas
   !!                        SUCCESSOR OF "sbcblk_core"
   !!=====================================================================
   !! History :  1.0  !  2004-08  (U. Schweckendiek)  Original CORE code
   !!            2.0  !  2005-04  (L. Brodeau, A.M. Treguier)  improved CORE bulk and its user interface
   !!            3.0  !  2006-06  (G. Madec)  sbc rewritting
   !!             -   !  2006-12  (L. Brodeau)  Original code for turb_core
   !!            3.2  !  2009-04  (B. Lemaire)  Introduce iom_put
   !!            3.3  !  2010-10  (S. Masson)  add diurnal cycle
   !!            3.4  !  2011-11  (C. Harris)  Fill arrays required by CICE
   !!            3.7  !  2014-06  (L. Brodeau)  simplification and optimization of CORE bulk
   !!            4.0  !  2016-06  (L. Brodeau)  sbcblk_core becomes sbcblk and is not restricted to the CORE algorithm anymore
   !!                 !                        ==> based on AeroBulk (https://github.com/brodeau/aerobulk/)
   !!            4.0  !  2016-10  (G. Madec)  introduce a sbc_blk_init routine
   !!            4.0  !  2016-10  (M. Vancoppenolle)  Introduce conduction flux emulator (M. Vancoppenolle)
   !!            4.0  !  2019-03  (F. Lemarié & G. Samson)  add ABL compatibility (ln_abl=TRUE)
   !!            4.2  !  2020-12  (L. Brodeau) Introduction of various air-ice bulk parameterizations + improvements
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   sbc_blk_init  : initialisation of the chosen bulk formulation as ocean surface boundary condition
   !!   sbc_blk       : bulk formulation as ocean surface boundary condition
   !!   blk_oce_1     : computes pieces of momentum, heat and freshwater fluxes over ocean for ABL model  (ln_abl=TRUE)
   !!   blk_oce_2     : finalizes momentum, heat and freshwater fluxes computation over ocean after the ABL step  (ln_abl=TRUE)
   !!             sea-ice case only :
   !!   blk_ice_1   : provide the air-ice stress
   !!   blk_ice_2   : provide the heat and mass fluxes at air-ice interface
   !!   blk_ice_qcn   : provide ice surface temperature and snow/ice conduction flux (emulating conduction flux)
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and tracers
   USE dom_oce        ! ocean space and time domain
   USE phycst         ! physical constants
   USE fldread        ! read input fields
   USE sbc_oce        ! Surface boundary condition: ocean fields
   USE trc_oce        ! share SMS/Ocean variables
   USE cyclone        ! Cyclone 10m wind form trac of cyclone centres
   USE sbcdcy         ! surface boundary condition: diurnal cycle
   USE sbcwave , ONLY :   cdn_wave ! wave module
   USE lib_fortran    ! to use key_nosignedzero and glob_2Dsum
   !
#if defined key_si3
   USE par_ice        ! SI3 parameters
   USE sbc_ice        ! Surface boundary condition: ice fields #LB? ok to be in 'key_si3' ???
   USE ice     , ONLY : u_ice, v_ice, a_i_b, at_i_b, t_su, hfx_err_dif, drag_ia
   USE icevar         ! for CALL ice_var_snwblow
   USE sbcblk_algo_ice_an05
   USE sbcblk_algo_ice_lu12
   USE sbcblk_algo_ice_lg15
#endif
   USE sbcblk_algo_ncar     ! => turb_ncar     : NCAR - (formerly known as CORE, Large & Yeager, 2009)
   USE sbcblk_algo_coare3p0 ! => turb_coare3p0 : COAREv3.0 (Fairall et al. 2003)
   USE sbcblk_algo_coare3p6 ! => turb_coare3p6 : COAREv3.6 (Fairall et al. 2018 + Edson et al. 2013)
   USE sbcblk_algo_ecmwf    ! => turb_ecmwf    : ECMWF (IFS cycle 45r1)
   USE sbcblk_algo_andreas  ! => turb_andreas  : Andreas et al. 2015
   USE sbcblk_algo_mfs      ! => turb_mfs      : MFS/BS Copernicus (Petenuzzo et al. 2010 doi:10.1029/2009JC005631)
   !
   USE iom            ! I/O manager library
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! distribued memory computing library
   USE lbclnk         ! ocean lateral boundary conditions (or mpp link)
   USE prtctl         ! Print control

   USE sbc_phy        ! Catalog of functions for physical/meteorological parameters in the marine boundary layer

   IMPLICIT NONE
   PRIVATE

   PUBLIC   sbc_blk_init  ! called in sbcmod
   PUBLIC   sbc_blk       ! called in sbcmod
   PUBLIC   blk_oce_1     ! called in sbcabl
   PUBLIC   blk_oce_2     ! called in sbcabl
#if defined key_si3
   PUBLIC   blk_ice_1     ! routine called in icesbc
   PUBLIC   blk_ice_2     ! routine called in icesbc
   PUBLIC   blk_ice_qcn   ! routine called in icesbc
#endif

   INTEGER , PUBLIC, PARAMETER ::   jp_wndi  =  1   ! index of 10m wind velocity (i-component) (m/s)    at T-point
   INTEGER , PUBLIC, PARAMETER ::   jp_wndj  =  2   ! index of 10m wind velocity (j-component) (m/s)    at T-point
   INTEGER , PUBLIC, PARAMETER ::   jp_tair  =  3   ! index of 10m air temperature             (Kelvin)
   INTEGER , PUBLIC, PARAMETER ::   jp_humi  =  4   ! index of specific humidity               (kg/kg)
   INTEGER , PUBLIC, PARAMETER ::   jp_qsr   =  5   ! index of solar heat                      (W/m2)
   INTEGER , PUBLIC, PARAMETER ::   jp_qlw   =  6   ! index of Long wave                       (W/m2)
   INTEGER , PUBLIC, PARAMETER ::   jp_prec  =  7   ! index of total precipitation (rain+snow) (Kg/m2/s)
   INTEGER , PUBLIC, PARAMETER ::   jp_snow  =  8   ! index of snow (solid prcipitation)       (kg/m2/s)
   INTEGER , PUBLIC, PARAMETER ::   jp_slp   =  9   ! index of sea level pressure              (Pa)
   INTEGER , PUBLIC, PARAMETER ::   jp_uoatm = 10   ! index of surface current (i-component)
   !                                                !          seen by the atmospheric forcing (m/s) at T-point
   INTEGER , PUBLIC, PARAMETER ::   jp_voatm = 11   ! index of surface current (j-component)
   !                                                !          seen by the atmospheric forcing (m/s) at T-point
   INTEGER , PUBLIC, PARAMETER ::   jp_cc    = 12   ! index of cloud cover                     (-)      range:0-1
   INTEGER , PUBLIC, PARAMETER ::   jp_hpgi  = 13   ! index of ABL geostrophic wind or hpg (i-component) (m/s) at T-point
   INTEGER , PUBLIC, PARAMETER ::   jp_hpgj  = 14   ! index of ABL geostrophic wind or hpg (j-component) (m/s) at T-point
   INTEGER , PUBLIC, PARAMETER ::   jpfld    = 14   ! maximum number of files to read

   ! Warning: keep this structure allocatable for Agrif...
   TYPE(FLD), PUBLIC, ALLOCATABLE, DIMENSION(:) ::   sf   ! structure of input atmospheric fields (file informations, fields read)

   !                           !!* Namelist namsbc_blk : bulk parameters
   LOGICAL  ::   ln_NCAR        ! "NCAR"      algorithm   (Large and Yeager 2008)
   LOGICAL  ::   ln_COARE_3p0   ! "COARE 3.0" algorithm   (Fairall et al. 2003)
   LOGICAL  ::   ln_COARE_3p6   ! "COARE 3.6" algorithm   (Edson et al. 2013)
   LOGICAL  ::   ln_ECMWF       ! "ECMWF"     algorithm   (IFS cycle 45r1)
   LOGICAL  ::   ln_ANDREAS     ! "ANDREAS"   algorithm   (Andreas et al. 2015)
   LOGICAL  ::   ln_MFS         ! "MFS"       algorithm   (Petenuzzo et al 2010)
   !
   LOGICAL          ::   ln_Cx_ice_cst    ! use constant air-ice bulk transfer coefficients (value given in namelist's rn_Cd_ia,rn_Ce_ia & rn_Ch_ia)
   REAL(wp), PUBLIC ::   rn_Cd_ia
   REAL(wp), PUBLIC ::   rn_Ce_ia, rn_Ch_ia   ! L3 bulk oracle: expose resolved operands to WRITE-only recorder
   LOGICAL          ::   ln_Cx_ice_AN05   ! air-ice bulk transfer coefficients based on Andreas et al., 2005
   LOGICAL          ::   ln_Cx_ice_LU12   ! air-ice bulk transfer coefficients based on Lupkes et al., 2012
   LOGICAL          ::   ln_Cx_ice_LG15   ! air-ice bulk transfer coefficients based on Lupkes & Gryanik, 2015
   LOGICAL , PUBLIC ::   ln_Cx_ice_frm = .FALSE.   !: use form drags, force default to false as sbcblk not called in coupled mode
   INTEGER , PUBLIC ::   nn_frm           !: = 1 : affects momentum and heat transfer coefficient
                                          !:       for ocean-ice and atmos-ice (default)
                                          !: = 2 : affects only momentum transfer coefficient
                                          !:       for ocean-ice and atmos-ice
                                          !: = 3 : affect momentum and heat transfer coefficient (atmos-ice),
                                          !:       but only momentum transfer
                                          ! coefficient (ocean-ice)
   REAL(wp), PUBLIC ::   rn_Cs_io         !: ice-ocn skin drag [0.0005,0.005]
   REAL(wp), PUBLIC ::   rn_Cs_ia         !: ice-air skin drag [0.0001,0.001]
   REAL(wp), PUBLIC ::   rn_Cr_ia         !: ridge/sail drag coefficient [0,1]
   REAL(wp), PUBLIC ::   rn_Cr_io         !: ridge/keel form drag coefficient [0,1]
   REAL(wp), PUBLIC ::   rn_Cf_ia         !: floe edge atm [0,1]
   REAL(wp), PUBLIC ::   rn_Cf_io         !: floe edge ocean [0,1]
   !
   LOGICAL  ::   ln_crt_fbk     ! Add surface current feedback to the wind stress computation  (Renault et al. 2020)
   REAL(wp) ::   rn_stau_a      ! Alpha and Beta coefficients of Renault et al. 2020, eq. 10: Stau = Alpha * Wnd + Beta
   REAL(wp) ::   rn_stau_b      !
   !
   REAL(wp)         ::   rn_pfac   ! multiplication factor for precipitation
   REAL(wp), PUBLIC ::   rn_efac   ! multiplication factor for evaporation
   REAL(wp)         ::   rn_zqt    ! z(q,t) : height of humidity and temperature measurements
   REAL(wp)         ::   rn_zu     ! z(u)   : height of wind measurements
   !
   INTEGER          :: nn_iter_algo   !  Number of iterations in bulk param. algo ("stable ABL + weak wind" requires more)

   REAL(wp), ALLOCATABLE, DIMENSION(:,:) ::   theta_zu, q_zu   ! air temp. and spec. hum. at wind speed height (L15 bulk scheme)
   REAL(wp), PUBLIC, ALLOCATABLE, DIMENSION(:,:) ::   precip   ! precipitation (after optional conversion from [m] to [Kg/m2/s])

#if defined key_si3
   REAL(wp), PUBLIC, ALLOCATABLE, DIMENSION(:,:) :: Cd_ice , Ch_ice , Ce_ice   !#LB transfert coefficients over ice
   REAL(wp), PUBLIC, ALLOCATABLE, DIMENSION(:,:) :: theta_zu_i, q_zu_i         !#LB fixme ! air temp. and spec. hum. over ice at wind speed height (L15 bulk scheme)
   REAL(wp), PUBLIC, SAVE :: l3_qsat_ice, l3_theta_ice, l3_qlw_ice, l3_qsb_ice, l3_dqlw_ice
#endif


   LOGICAL  ::   ln_skin_cs     ! use the cool-skin (only available in ECMWF and COARE algorithms) !LB
   LOGICAL  ::   ln_skin_wl     ! use the warm-layer parameterization (only available in ECMWF and COARE algorithms) !LB
   LOGICAL  ::   ln_humi_sph    ! humidity read in files ("sn_humi") is specific humidity [kg/kg] if .true. !LB
   LOGICAL  ::   ln_humi_dpt    ! humidity read in files ("sn_humi") is dew-point temperature [K] if .true. !LB
   LOGICAL  ::   ln_humi_rlh    ! humidity read in files ("sn_humi") is relative humidity     [%] if .true. !LB
   LOGICAL  ::   ln_tair_pot    ! temperature read in files ("sn_tair") is already potential temperature (not absolute)
   LOGICAL  ::   ln_prec_met    ! precipitation read in files ("sn_prec") is in m and has to be converted to kg/m2/s
   !
   INTEGER  ::   nhumi          ! choice of the bulk algorithm
   !                            ! associated indices:
   INTEGER, PARAMETER :: np_humi_sph = 1
   INTEGER, PARAMETER :: np_humi_dpt = 2
   INTEGER, PARAMETER :: np_humi_rlh = 3

   INTEGER  ::   nblk           ! choice of the bulk algorithm
   !                            ! associated indices:
   INTEGER, PARAMETER ::   np_NCAR      = 1   ! "NCAR" algorithm        (Large and Yeager 2008)
   INTEGER, PARAMETER ::   np_COARE_3p0 = 2   ! "COARE 3.0" algorithm   (Fairall et al. 2003)
   INTEGER, PARAMETER ::   np_COARE_3p6 = 3   ! "COARE 3.6" algorithm   (Edson et al. 2013)
   INTEGER, PARAMETER ::   np_ECMWF     = 4   ! "ECMWF" algorithm       (IFS cycle 45r1)
   INTEGER, PARAMETER ::   np_ANDREAS   = 5   ! "ANDREAS" algorithm     (Andreas et al. 2015)
   INTEGER, PARAMETER ::   np_MFS       = 6   ! "MFS" algorithm         (Petenuzzo et al. 2010)
   !#LB:
#if defined key_si3
   ! Same, over sea-ice:
   INTEGER  ::   nblk_ice           ! choice of the bulk algorithm
   !                            ! associated indices:
   INTEGER, PARAMETER ::   np_ice_cst  = 1   ! constant transfer coefficients
   INTEGER, PARAMETER ::   np_ice_an05 = 2   ! Andreas et al., 2005
   INTEGER, PARAMETER ::   np_ice_lu12 = 3   ! Lupkes el al., 2012
   INTEGER, PARAMETER ::   np_ice_lg15 = 4   ! Lupkes & Gryanik, 2015
   INTEGER, PARAMETER ::   np_ice_frm  = 5   ! Tsamadoes et al., 2014
#endif
   !LB.



   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "read_nml_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   INTEGER FUNCTION sbc_blk_alloc()
      !!-------------------------------------------------------------------
      !!             ***  ROUTINE sbc_blk_alloc ***
      !!-------------------------------------------------------------------
      ALLOCATE( theta_zu(A2D(0)), q_zu(A2D(0)), precip(A2D(0)), STAT=sbc_blk_alloc )
      CALL mpp_sum ( 'sbcblk', sbc_blk_alloc )
      IF( sbc_blk_alloc /= 0 )   CALL ctl_stop( 'STOP', 'sbc_blk_alloc: failed to allocate arrays' )
   END FUNCTION sbc_blk_alloc

#if defined key_si3
   INTEGER FUNCTION sbc_blk_ice_alloc()
      !!-------------------------------------------------------------------
      !!             ***  ROUTINE sbc_blk_ice_alloc ***
      !!-------------------------------------------------------------------
      ALLOCATE( Cd_ice(A2D(0)), Ch_ice(A2D(0)), Ce_ice(A2D(0)), theta_zu_i(A2D(0)), q_zu_i(A2D(0)), STAT=sbc_blk_ice_alloc )
      CALL mpp_sum ( 'sbcblk', sbc_blk_ice_alloc )
      IF( sbc_blk_ice_alloc /= 0 )   CALL ctl_stop( 'STOP', 'sbc_blk_ice_alloc: failed to allocate arrays' )
   END FUNCTION sbc_blk_ice_alloc
#endif


   SUBROUTINE sbc_blk_init
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE sbc_blk_init  ***
      !!
      !! ** Purpose :   choose and initialize a bulk formulae formulation
      !!
      !! ** Method  :
      !!
      !!----------------------------------------------------------------------
      INTEGER  ::   jfpr                  ! dummy loop indice and argument
      INTEGER  ::   ios, ierror, ioptio   ! Local integer
      !!
      CHARACTER(len=100)            ::   cn_dir                ! Root directory for location of atmospheric forcing files
      TYPE(FLD_N), DIMENSION(jpfld) ::   slf_i                 ! array of namelist informations on the fields to read
      TYPE(FLD_N) ::   sn_wndi, sn_wndj , sn_humi, sn_qsr      ! informations about the fields to be read
      TYPE(FLD_N) ::   sn_qlw , sn_tair , sn_prec, sn_snow     !       "                        "
      TYPE(FLD_N) ::   sn_slp , sn_uoatm, sn_voatm             !       "                        "
      TYPE(FLD_N) ::   sn_cc, sn_hpgi, sn_hpgj                 !       "                        "
      INTEGER     ::   ipka                                    ! number of levels in the atmospheric variable
      NAMELIST/namsbc_blk/ ln_NCAR, ln_COARE_3p0, ln_COARE_3p6, ln_ECMWF, ln_ANDREAS, ln_MFS, &   ! bulk algorithm
         &                 rn_zqt, rn_zu, nn_iter_algo, ln_skin_cs, ln_skin_wl,        &
         &                 rn_pfac, rn_efac,                                           &
         &                 ln_crt_fbk, rn_stau_a, rn_stau_b,                           &   ! current feedback
         &                 ln_humi_sph, ln_humi_dpt, ln_humi_rlh, ln_tair_pot,         &
         &                 ln_prec_met,                                                &
         &                 ln_Cx_ice_cst,                                              &
         &                 rn_Cd_ia, rn_Ce_ia, rn_Ch_ia,                               &
         &                 ln_Cx_ice_frm, nn_frm,                                      &
         &                 rn_Cs_io, rn_Cs_ia, rn_Cr_ia, rn_Cr_io, rn_Cf_ia, rn_Cf_io, &
         &                 ln_Cx_ice_AN05, ln_Cx_ice_LU12, ln_Cx_ice_LG15,             &
         &                 cn_dir,                                                     &
         &                 sn_wndi, sn_wndj, sn_qsr, sn_qlw ,                          &   ! input fields
         &                 sn_tair, sn_humi, sn_prec, sn_snow, sn_slp,                 &
         &                 sn_uoatm, sn_voatm, sn_cc, sn_hpgi, sn_hpgj

      ! cool-skin / warm-layer !LB
      !!---------------------------------------------------------------------
      !
      !                                      ! allocate sbc_blk_core array
      IF( sbc_blk_alloc()     /= 0 )   CALL ctl_stop( 'STOP', 'sbc_blk : unable to allocate standard arrays' )
      !
#if defined key_si3
      IF( sbc_blk_ice_alloc() /= 0 )   CALL ctl_stop( 'STOP', 'sbc_blk : unable to allocate standard ice arrays' )
#endif
      !
      !                             !** read bulk namelist
      READ_NML_REF(numnam,namsbc_blk)
      READ_NML_CFG(numnam,namsbc_blk)
      IF(lwm) WRITE( numond, namsbc_blk )
      !
      !                             !** initialization of the chosen bulk formulae (+ check)
      !                                   !* select the bulk chosen in the namelist and check the choice
      ioptio = 0
      IF( ln_NCAR      ) THEN
         nblk =  np_NCAR        ;   ioptio = ioptio + 1
      ENDIF
      IF( ln_COARE_3p0 ) THEN
         nblk =  np_COARE_3p0   ;   ioptio = ioptio + 1
      ENDIF
      IF( ln_COARE_3p6 ) THEN
         nblk =  np_COARE_3p6   ;   ioptio = ioptio + 1
      ENDIF
      IF( ln_ECMWF     ) THEN
         nblk =  np_ECMWF       ;   ioptio = ioptio + 1
      ENDIF
      IF( ln_ANDREAS   ) THEN
         nblk =  np_ANDREAS     ;   ioptio = ioptio + 1
      ENDIF
      IF( ln_MFS       ) THEN
         nblk =  np_MFS         ;   ioptio = ioptio + 1
      ENDIF
      IF( ioptio /= 1 )   CALL ctl_stop( 'sbc_blk_init: Choose one and only one bulk algorithm' )

      !                             !** initialization of the cool-skin / warm-layer parametrization
      IF( ln_skin_cs .OR. ln_skin_wl ) THEN
         !! Some namelist sanity tests:
         IF( ln_NCAR .OR. ln_ANDREAS .OR. ln_MFS )      &
            & CALL ctl_stop( 'sbc_blk_init: Cool-skin/warm-layer param. not compatible with current bulk algorithm' )
         !IF( nn_fsbc /= 1 ) &
         !   & CALL ctl_stop( 'sbc_blk_init: Please set "nn_fsbc" to 1 when using cool-skin/warm-layer param.')
      END IF

      IF( ln_skin_wl ) THEN
         !! Check if the frequency of downwelling solar flux input makes sense and if ln_dm2dc=T if it is daily!
         IF( (sn_qsr%freqh  < 0.).OR.(sn_qsr%freqh  > 24.) ) &
            & CALL ctl_stop( 'sbc_blk_init: Warm-layer param. (ln_skin_wl) not compatible with freq. of solar flux > daily' )
         IF( (sn_qsr%freqh == 24.).AND.(.NOT. ln_dm2dc) ) &
            & CALL ctl_stop( 'sbc_blk_init: Please set ln_dm2dc=T for warm-layer param. (ln_skin_wl) to work properly' )
      END IF

      IF ( ln_MFS .AND. .NOT. ln_humi_dpt )    &
          & CALL ctl_stop( 'sbc_blk_init: MFS bulk only works with dew point temperature (ln_humi_dpt)')

      ioptio = 0
      IF( ln_humi_sph ) THEN
         nhumi =  np_humi_sph    ;   ioptio = ioptio + 1
      ENDIF
      IF( ln_humi_dpt ) THEN
         nhumi =  np_humi_dpt    ;   ioptio = ioptio + 1
      ENDIF
      IF( ln_humi_rlh ) THEN
         nhumi =  np_humi_rlh    ;   ioptio = ioptio + 1
      ENDIF
      IF( ioptio /= 1 )   CALL ctl_stop( 'sbc_blk_init: Choose one and only one type of air humidity' )
      !
      IF( ln_dm2dc ) THEN                 !* check: diurnal cycle on Qsr
         IF( sn_qsr%freqh /= 24. )   CALL ctl_stop( 'sbc_blk_init: ln_dm2dc=T only with daily short-wave input' )
         IF( sn_qsr%ln_tint ) THEN
            CALL ctl_warn( 'sbc_blk_init: ln_dm2dc=T daily qsr time interpolation done by sbcdcy module',   &
               &           '              ==> We force time interpolation = .false. for qsr' )
            sn_qsr%ln_tint = .false.
         ENDIF
      ENDIF

#if defined key_si3
      ioptio = 0
      IF( ln_Cx_ice_cst ) THEN
         nblk_ice =  np_ice_cst     ;   ioptio = ioptio + 1
      ENDIF
      IF( ln_Cx_ice_AN05 ) THEN
         nblk_ice =  np_ice_an05   ;   ioptio = ioptio + 1
      ENDIF
      IF( ln_Cx_ice_LU12 ) THEN
         nblk_ice =  np_ice_lu12    ;   ioptio = ioptio + 1
      ENDIF
      IF( ln_Cx_ice_LG15 ) THEN
         nblk_ice =  np_ice_lg15   ;   ioptio = ioptio + 1
      ENDIF
      IF( ln_Cx_ice_frm ) THEN
         nblk_ice =  np_ice_frm     ;   ioptio = ioptio + 1
      ENDIF
      IF( ioptio /= 1 )   CALL ctl_stop( 'sbc_blk_init: Choose one and only one ice-atm bulk algorithm' )
#endif


      !                                   !* set the bulk structure
      !                                      !- store namelist information in an array
      !
      slf_i(jp_wndi ) = sn_wndi    ;   slf_i(jp_wndj ) = sn_wndj
      slf_i(jp_qsr  ) = sn_qsr     ;   slf_i(jp_qlw  ) = sn_qlw
      slf_i(jp_tair ) = sn_tair    ;   slf_i(jp_humi ) = sn_humi
      slf_i(jp_prec ) = sn_prec    ;   slf_i(jp_snow ) = sn_snow
      slf_i(jp_slp  ) = sn_slp     ;   slf_i(jp_cc   ) = sn_cc
      slf_i(jp_uoatm) = sn_uoatm   ;   slf_i(jp_voatm) = sn_voatm
      slf_i(jp_hpgi ) = sn_hpgi    ;   slf_i(jp_hpgj ) = sn_hpgj
      !
      IF( .NOT. ln_abl ) THEN   ! force to not use jp_hpgi and jp_hpgj, should already be done in namelist_* but we never know...
         slf_i(jp_hpgi)%clname = 'NOT USED'
         slf_i(jp_hpgj)%clname = 'NOT USED'
      ENDIF
      !
      !                                      !- allocate the bulk structure
      ALLOCATE( sf(jpfld), STAT=ierror )
      IF( ierror > 0 )   CALL ctl_stop( 'STOP', 'sbc_blk_init: unable to allocate sf structure' )
      !
      !                                      !- fill the bulk structure with namelist informations
      CALL fld_fill( sf, slf_i, cn_dir, 'sbc_blk_init', 'surface boundary condition -- bulk formulae', 'namsbc_blk' )
      sf(jp_wndi )%zsgn = -1._wp   ;   sf(jp_wndj )%zsgn = -1._wp   ! vector field at T point: overwrite default definition of zsgn
      sf(jp_uoatm)%zsgn = -1._wp   ;   sf(jp_voatm)%zsgn = -1._wp   ! vector field at T point: overwrite default definition of zsgn
      sf(jp_hpgi )%zsgn = -1._wp   ;   sf(jp_hpgj )%zsgn = -1._wp   ! vector field at T point: overwrite default definition of zsgn
      !
      DO jfpr= 1, jpfld
         !
         IF(   ln_abl    .AND.                                                      &
            &    ( jfpr == jp_wndi .OR. jfpr == jp_wndj .OR. jfpr == jp_humi .OR.   &
            &      jfpr == jp_hpgi .OR. jfpr == jp_hpgj .OR. jfpr == jp_tair     )  ) THEN
            ipka = jpka   ! ABL: some fields are 3D input
         ELSE
            ipka = 1
         ENDIF
         !
         ALLOCATE( sf(jfpr)%fnow(A2D(0),ipka) )
         !
         IF( TRIM(sf(jfpr)%clrootname) == 'NOT USED' ) THEN    !--  not used field  --!   (only now allocated and set to default)
            IF(     jfpr == jp_slp ) THEN
               sf(jfpr)%fnow(:,:,1:ipka) = 101325._wp   ! use standard pressure in Pa
            ELSEIF( jfpr == jp_prec .OR. jfpr == jp_snow .OR. jfpr == jp_uoatm .OR. jfpr == jp_voatm ) THEN
               sf(jfpr)%fnow(:,:,1:ipka) = 0._wp        ! no precip or no snow or no surface currents
            ELSEIF( jfpr == jp_wndi .OR. jfpr == jp_wndj ) THEN
               sf(jfpr)%fnow(:,:,1:ipka) = 0._wp
            ELSEIF( jfpr == jp_hpgi .OR. jfpr == jp_hpgj ) THEN
               IF( .NOT. ln_abl ) THEN
                  DEALLOCATE( sf(jfpr)%fnow )   ! deallocate as not used in this case
               ELSE
                  sf(jfpr)%fnow(:,:,1:ipka) = 0._wp
               ENDIF
            ELSEIF( jfpr == jp_cc  ) THEN
               sf(jp_cc)%fnow(:,:,1:ipka) = pp_cldf
            ELSEIF( jfpr ==jp_qsr .OR. jfpr==jp_qlw ) THEN
               sf(jfpr)%fnow(:,:,1:ipka) = 0._wp
            ELSE
               WRITE(ctmp1,*) 'sbc_blk_init: no default value defined for field number', jfpr
               CALL ctl_stop( ctmp1 )
            ENDIF
         ELSE                                                  !-- used field  --!
            IF( sf(jfpr)%ln_tint )   ALLOCATE( sf(jfpr)%fdta(A2D(0),ipka,2) )   ! allocate array for temporal interpolation
            !
            IF( sf(jfpr)%freqh > 0. .AND. MOD( NINT(3600. * sf(jfpr)%freqh), nn_fsbc * NINT(rn_Dt) ) /= 0 )   &
         &  CALL ctl_warn( 'sbc_blk_init: sbcmod timestep rn_Dt*nn_fsbc is NOT a submultiple of atmospheric forcing frequency.',   &
         &                 '               This is not ideal. You should consider changing either rn_Dt or nn_fsbc value...' )
         ENDIF
      END DO
      !
      IF( ln_abl ) THEN       ! ABL: read 3D fields for wind, temperature, humidity and pressure gradient
         rn_zqt = ght_abl(2)          ! set the bulk altitude to ABL first level
         rn_zu  = ght_abl(2)
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) '   ABL formulation: overwrite rn_zqt & rn_zu with ABL first level altitude'
      ENDIF
      !
      !
      IF(lwp) THEN                     !** Control print
         !
         WRITE(numout,*)                  !* namelist
         WRITE(numout,*) '   Namelist namsbc_blk (other than data information):'
         WRITE(numout,*) '      "NCAR"      algorithm   (Large and Yeager 2008)      ln_NCAR      = ', ln_NCAR
         WRITE(numout,*) '      "COARE 3.0" algorithm   (Fairall et al. 2003)       ln_COARE_3p0 = ', ln_COARE_3p0
         WRITE(numout,*) '      "COARE 3.6" algorithm (Fairall 2018 + Edson al 2013) ln_COARE_3p6 = ', ln_COARE_3p6
         WRITE(numout,*) '      "ECMWF"     algorithm   (IFS cycle 45r1)             ln_ECMWF     = ', ln_ECMWF
         WRITE(numout,*) '      "ANDREAS"   algorithm   (Andreas et al. 2015)       ln_ANDREAS   = ', ln_ANDREAS
         WRITE(numout,*) '      "MFS"       algorithm   (Petenuzzo et al. 2010)     ln_MFS       = ', ln_MFS
         WRITE(numout,*) '      Air temperature and humidity reference height (m)   rn_zqt       = ', rn_zqt
         WRITE(numout,*) '      Wind vector reference height (m)                    rn_zu        = ', rn_zu
         WRITE(numout,*) '      factor applied on precipitation (total & snow)      rn_pfac      = ', rn_pfac
         WRITE(numout,*) '      factor applied on evaporation                       rn_efac      = ', rn_efac
         WRITE(numout,*) '         (form absolute (=0) to relative winds(=1))'
         WRITE(numout,*) '      use surface current feedback on wind stress         ln_crt_fbk   = ', ln_crt_fbk
         IF(ln_crt_fbk) THEN
         WRITE(numout,*) '         Renault et al. 2020, eq. 10: Stau = Alpha * Wnd + Beta'
         WRITE(numout,*) '            Alpha                                         rn_stau_a    = ', rn_stau_a
         WRITE(numout,*) '            Beta                                          rn_stau_b    = ', rn_stau_b
         ENDIF
         !
         WRITE(numout,*)
         SELECT CASE( nblk )              !* Print the choice of bulk algorithm
         CASE( np_NCAR      )   ;   WRITE(numout,*) '   ==>>>   "NCAR" algorithm        (Large and Yeager 2008)'
         CASE( np_COARE_3p0 )   ;   WRITE(numout,*) '   ==>>>   "COARE 3.0" algorithm   (Fairall et al. 2003)'
         CASE( np_COARE_3p6 )   ;   WRITE(numout,*) '   ==>>>   "COARE 3.6" algorithm (Fairall 2018+Edson et al. 2013)'
         CASE( np_ECMWF     )   ;   WRITE(numout,*) '   ==>>>   "ECMWF" algorithm       (IFS cycle 45r1)'
         CASE( np_ANDREAS   )   ;   WRITE(numout,*) '   ==>>>   "ANDREAS" algorithm (Andreas et al. 2015)'
         CASE( np_MFS       )   ;   WRITE(numout,*) '   ==>>    "MFS/BS Copernicus" algorithm (Petenuzzo et al. 2010)'
         END SELECT
         !
         WRITE(numout,*)
         WRITE(numout,*) '      use cool-skin  parameterization (SSST)  ln_skin_cs  = ', ln_skin_cs
         WRITE(numout,*) '      use warm-layer parameterization (SSST)  ln_skin_wl  = ', ln_skin_wl
         !
         WRITE(numout,*)
         SELECT CASE( nhumi )              !* Print the choice of air humidity
         CASE( np_humi_sph )   ;   WRITE(numout,*) '   ==>>>   air humidity is SPECIFIC HUMIDITY     [kg/kg]'
         CASE( np_humi_dpt )   ;   WRITE(numout,*) '   ==>>>   air humidity is DEW-POINT TEMPERATURE [K]'
         CASE( np_humi_rlh )   ;   WRITE(numout,*) '   ==>>>   air humidity is RELATIVE HUMIDITY     [%]'
         END SELECT
         !
         !#LB:
#if defined key_si3
         IF( nn_ice > 0 ) THEN
            WRITE(numout,*)
            WRITE(numout,*) '      use constant ice-atm bulk transfer coeff.           ln_Cx_ice_cst  = ', ln_Cx_ice_cst
            WRITE(numout,*) '      use ice-atm bulk coeff. from Andreas et al., 2005   ln_Cx_ice_AN05 = ', ln_Cx_ice_AN05
            WRITE(numout,*) '      use ice-atm bulk coeff. from Lupkes et al., 2012    ln_Cx_ice_LU12 = ', ln_Cx_ice_LU12
            WRITE(numout,*) '      use ice-atm bulk coeff. from Lupkes & Gryanik, 2015 ln_Cx_ice_LG15 = ', ln_Cx_ice_LG15
            WRITE(numout,*) '      use form-drag param from Tsamadoes et al.,2014      ln_Cx_ice_frm  = ', ln_Cx_ice_frm
         ENDIF
         WRITE(numout,*)
         SELECT CASE( nblk_ice )              !* Print the choice of bulk algorithm
         CASE( np_ice_cst  )
            WRITE(numout,*) '   ==>>>   Constant bulk transfer coefficients over sea-ice:'
            WRITE(numout,*) '      => Cd_ice, Ce_ice, Ch_ice =', REAL(rn_Cd_ia,4), REAL(rn_Ce_ia,4), REAL(rn_Ch_ia,4)
            IF( (rn_Cd_ia<0._wp).OR.(rn_Cd_ia>1.E-2_wp).OR.(rn_Ce_ia<0._wp).OR.(rn_Ce_ia>1.E-2_wp).OR.(rn_Ch_ia<0._wp).OR.(rn_Ch_ia>1.E-2_wp) ) &
               & CALL ctl_stop( 'Be realistic in your pick of Cd_ice, Ce_ice & Ch_ice ! (0 < Cx < 1.E-2)')
         CASE( np_ice_an05 )   ;   WRITE(numout,*) '   ==>>> bulk algo over ice: Andreas et al, 2005'
         CASE( np_ice_lu12 )   ;   WRITE(numout,*) '   ==>>> bulk algo over ice: Lupkes et al, 2012'
         CASE( np_ice_lg15 )   ;   WRITE(numout,*) '   ==>>> bulk algo over ice: Lupkes & Gryanik, 2015'
         CASE( np_ice_frm )
            WRITE(numout,*) '   ==>>> form-drag param with nn_frm = ', nn_frm
            WRITE(numout,*) '   1 : affects momentum and heat transfer coefficient'
            WRITE(numout,*) '       for ocean-ice and atmos-ice (default)'
            WRITE(numout,*) '   2 : affects only momentum transfer coefficient'
            WRITE(numout,*) '       for ocean-ice and atmos-ice'
            WRITE(numout,*) '   3 : affect momentum and heat transfer coefficient (atmos-ice),'
            WRITE(numout,*) '       but only momentum transfer coefficient (ocean-ice)'
            WRITE(numout,*) '                with rn_Cs_io    = ', rn_Cs_io
            WRITE(numout,*) '                with rn_Cs_ia    = ', rn_Cs_ia
            WRITE(numout,*) '                with rn_Cr_ia    = ', rn_Cr_ia
            WRITE(numout,*) '                with rn_Cr_io    = ', rn_Cr_io
            WRITE(numout,*) '                with rn_Cf_ia    = ', rn_Cf_ia
            WRITE(numout,*) '                with rn_Cf_io    = ', rn_Cf_io
         END SELECT
#endif
         !#LB.
         !
      ENDIF
      !
   END SUBROUTINE sbc_blk_init


   SUBROUTINE sbc_blk( kt )
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE sbc_blk  ***
      !!
      !! ** Purpose :   provide at each time step the surface ocean fluxes
      !!              (momentum, heat, freshwater and runoff)
      !!
      !! ** Method  :
      !!              (1) READ each fluxes in NetCDF files:
      !!      the wind velocity (i-component) at z=rn_zu  (m/s) at T-point
      !!      the wind velocity (j-component) at z=rn_zu  (m/s) at T-point
      !!      the specific humidity           at z=rn_zqt (kg/kg)
      !!      the air temperature             at z=rn_zqt (Kelvin)
      !!      the solar heat                              (W/m2)
      !!      the Long wave                               (W/m2)
      !!      the total precipitation (rain+snow)         (Kg/m2/s)
      !!      the snow (solid precipitation)              (kg/m2/s)
      !!      ABL dynamical forcing (i/j-components of either hpg or geostrophic winds)
      !!              (2) CALL blk_oce_1 and blk_oce_2
      !!
      !!      C A U T I O N : never mask the surface stress fields
      !!                      the stress is assumed to be in the (i,j) mesh referential
      !!
      !! ** Action  :   defined at each time-step at the air-sea interface
      !!              - utau, vtau  i- and j-component of the wind stress at T-point
      !!              - taum        wind stress module at T-point
      !!              - wndm        wind speed  module at T-point over free ocean or leads in presence of sea-ice
      !!              - qns, qsr    non-solar and solar heat fluxes
      !!              - emp         upward mass flux (evapo. - precip.)
      !!              - sfx         salt flux due to freezing/melting (non-zero only if ice is present)
      !!
      !! ** References :   Large & Yeager, 2004 / Large & Yeager, 2008
      !!                   Brodeau et al. Ocean Modelling 2010
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt   ! ocean time step
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(A2D(0)) ::   zssq, zcd_du, zsen, zlat, zevp, zpre, ztheta, zqlwn
      REAL(wp) :: ztst
      LOGICAL  :: llerr
      INTEGER  :: ji, jj
      !!----------------------------------------------------------------------
      !
      CALL fld_read( kt, nn_fsbc, sf )             ! input fields provided at the current time-step

      ! Sanity/consistence test on humidity at first time step to detect potential screw-up:
      IF( kt == nit000 ) THEN
         ! mean humidity over ocean on proc
         ztst =   glob_2Dsum( 'sbcblk', sf(jp_humi)%fnow(:,:,1) * e1e2t(A2D(0)) * smask0(:,:) ) &
            &   / glob_2Dsum( 'sbcblk', e1e2t(A2D(0)) * smask0(:,:) )
         llerr = .FALSE.
         SELECT CASE( nhumi )
         CASE( np_humi_sph ) ! specific humidity => expect: 0. <= something < 0.065 [kg/kg] (0.061 is saturation at 45degC !!!)
            IF( (ztst <   0._wp) .OR. (ztst > 0.065_wp) )   llerr = .TRUE.
         CASE( np_humi_dpt ) ! dew-point temperature => expect: 110. <= something < 320. [K]
            IF( (ztst < 110._wp) .OR. (ztst >  320._wp) )   llerr = .TRUE.
         CASE( np_humi_rlh ) ! relative humidity => expect: 0. <= something < 100. [%]
            IF( (ztst <   0._wp) .OR. (ztst >  100._wp) )   llerr = .TRUE.
         END SELECT
         IF(llerr) THEN
            WRITE(ctmp1,'("   Error on mean humidity value: ",f10.5)') ztst
            CALL ctl_stop( 'STOP', ctmp1, 'Something is wrong with air humidity!!!', &
               &   ' ==> check the unit in your input files'       , &
               &   ' ==> check consistence of namelist choice: specific? relative? dew-point?', &
               &   ' ==> ln_humi_sph -> [kg/kg] | ln_humi_rlh -> [%] | ln_humi_dpt -> [K] !!!' )
         ENDIF
         IF(lwp) THEN
            WRITE(numout,*) ''
            WRITE(numout,*) ' Global mean humidity at kt = nit000: ', ztst
            WRITE(numout,*) ' === Sanity/consistence test on air humidity sucessfuly passed! ==='
            WRITE(numout,*) ''
         ENDIF
      ENDIF   !IF( kt == nit000 )
      !                                            ! compute the surface ocean fluxes using bulk formulea
      IF( MOD( kt - 1, nn_fsbc ) == 0 ) THEN

         ! Specific humidity of air at z=rn_zqt
         SELECT CASE( nhumi )
         CASE( np_humi_sph )
            q_air_zt(:,:) = sf(jp_humi )%fnow(:,:,1)      ! what we read in file is already a spec. humidity!
         CASE( np_humi_dpt )
            IF((kt==nit000).AND.lwp) WRITE(numout,*) ' *** sbc_blk() => computing q_air out of dew-point and P !'
            q_air_zt(:,:) = q_sat( sf(jp_humi )%fnow(:,:,1), sf(jp_slp  )%fnow(:,:,1) )
         CASE( np_humi_rlh )
            IF((kt==nit000).AND.lwp) WRITE(numout,*) ' *** sbc_blk() => computing q_air out of RH, t_air and slp !' !LBrm
            q_air_zt(:,:) = q_air_rh( 0.01_wp*sf(jp_humi )%fnow(:,:,1), &
               &                      sf(jp_tair )%fnow(:,:,1), sf(jp_slp  )%fnow(:,:,1) ) !#LB: 0.01 => RH is % percent in file
         END SELECT

         ! Potential temperature of air at z=rn_zqt (most reanalysis products provide absolute temp., not potential temp.)
         IF( ln_tair_pot ) THEN
            ! temperature read into file is already potential temperature, do nothing...
            IF((kt==nit000).AND.lwp) WRITE(numout,*) ' *** sbc_blk() => air temperature already converted to POTENTIAL!'
            theta_air_zt(:,:) = sf(jp_tair )%fnow(:,:,1)
         ELSE
            ! temperature read into file is ABSOLUTE temperature (that's the case for ECMWF products for example...)
            IF((kt==nit000).AND.lwp) WRITE(numout,*) ' *** sbc_blk() => air temperature converted from ABSOLUTE to POTENTIAL!'
            zpre(:,:)         = pres_temp( q_air_zt(:,:), sf(jp_slp)%fnow(:,:,1), rn_zqt, pta=sf(jp_tair)%fnow(:,:,1) )
            theta_air_zt(:,:) = theta_exner( sf(jp_tair)%fnow(:,:,1), zpre(:,:) )
         ENDIF
         !
         IF( ln_prec_met ) THEN
            precip(:,:) = MAX(0._wp, sf(jp_prec)%fnow(:,:,1) * 1000._wp /(sf(jp_prec)%freqh * 3600._wp ) )
         ELSE
            precip(:,:) = sf(jp_prec)%fnow(:,:,1)
         ENDIF
         !
         CALL blk_oce_1( kt, sf(jp_wndi )%fnow(:,:,1), sf(jp_wndj )%fnow(:,:,1),   &   !   <<= in
            &                theta_air_zt(:,:), q_air_zt(:,:),                     &   !   <<= in
            &                sf(jp_slp  )%fnow(:,:,1), sst_m, ssu_m, ssv_m,        &   !   <<= in
            &                sf(jp_uoatm)%fnow(:,:,1), sf(jp_voatm)%fnow(:,:,1),   &   !   <<= in
            &                sf(jp_qsr  )%fnow(:,:,1), sf(jp_qlw  )%fnow(:,:,1),   &   !   <<= in (wl/cs)
            &                tsk_m, zssq, zcd_du, zsen, zlat, zevp, zqlwn )            !   =>> out

         CALL blk_oce_2(     theta_air_zt(:,:),                                    &   !   <<= in
            &                sf(jp_qlw  )%fnow(:,:,1), precip ,                    &   !   <<= in
            &                sf(jp_snow )%fnow(:,:,1), tsk_m,                      &   !   <<= in
            &                zsen, zlat, zevp, zqlwn )                                 !   <=> in out
      ENDIF
      !
#if defined key_cice
      IF( MOD( kt - 1, nn_fsbc ) == 0 )   THEN
         qlw_ice(:,:,1)   = sf(jp_qlw )%fnow(:,:,1)
         IF( ln_dm2dc ) THEN
            qsr_ice(:,:,1) = sbc_dcy( sf(jp_qsr)%fnow(:,:,1) )
         ELSE
            qsr_ice(:,:,1) =          sf(jp_qsr)%fnow(:,:,1)
         ENDIF
         tatm_ice(:,:) = sf(jp_tair)%fnow(:,:,1)    !#LB: should it be POTENTIAL temperature (theta_air_zt) instead ????
         qatm_ice(:,:) = q_air_zt(:,:)
         tprecip(:,:)  = precip                  * rn_pfac
         sprecip(:,:)  = sf(jp_snow)%fnow(:,:,1) * rn_pfac
         wndi_ice(:,:) = sf(jp_wndi)%fnow(:,:,1)
         wndj_ice(:,:) = sf(jp_wndj)%fnow(:,:,1)
      ENDIF
#endif

#if defined key_top
      IF( ln_trcdc2dm )  THEN      !  diurnal cycle in TOP
         IF( MOD( kt - 1, nn_fsbc ) == 0 ) THEN
            IF( ln_dm2dc )  THEN
              DO_2D( 0, 0, 0, 0 )
                qsr_mean(ji,jj) = ( 1. - albo )  * sf(jp_qsr)%fnow(ji,jj,1)  * smask0(ji,jj)
              END_2D
            ELSE
                ncpl_qsr_freq = sf(jp_qsr)%freqh * 3600 !   qsr_mean will be computed in TOP
            ENDIF
         ENDIF
      ENDIF
#endif
      !
   END SUBROUTINE sbc_blk


   SUBROUTINE blk_oce_1( kt, pwndi, pwndj, ptair, pqair,         &    ! <<= in
      &                      pslp , pst  , pu   , pv,            &    ! <<= in
      &                      puatm, pvatm, pdqsr , pdqlw ,       &    ! <<= in
      &                      ptsk , pssq , pcd_du, psen, plat, pevp, qlwn ) ! =>> out
      !!---------------------------------------------------------------------
      !!                     ***  ROUTINE blk_oce_1  ***
      !!
      !! ** Purpose :   if ln_blk=T, computes surface momentum, heat and freshwater fluxes
      !!                if ln_abl=T, computes Cd x |U|, Ch x |U|, Ce x |U| for ABL integration
      !!
      !! ** Method  :   bulk formulae using atmospheric fields from :
      !!                if ln_blk=T, atmospheric fields read in sbc_read
      !!                if ln_abl=T, the ABL model at previous time-step
      !!
      !! ** Outputs : - pssq    : surface humidity used to compute latent heat flux (kg/kg)
      !!              - pcd_du  : Cd x |dU| at T-points  (m/s)
      !!              - psen    : sensible heat flux (W/m^2)
      !!              - plat    : latent heat flux   (W/m^2)
      !!              - pevp    : evaporation        (mm/s) #lolo
      !!---------------------------------------------------------------------
      INTEGER , INTENT(in   )                    ::   kt     ! time step index
      REAL(wp), INTENT(inout), DIMENSION(A2D(0)) ::   pwndi  ! atmospheric wind at T-point              [m/s]
      REAL(wp), INTENT(inout), DIMENSION(A2D(0)) ::   pwndj  ! atmospheric wind at T-point              [m/s]
      REAL(wp), INTENT(in   ), DIMENSION(A2D(0)) ::   pqair  ! specific humidity at T-points            [kg/kg]
      REAL(wp), INTENT(in   ), DIMENSION(A2D(0)) ::   ptair  ! potential temperature at T-points        [Kelvin]
      REAL(wp), INTENT(in   ), DIMENSION(A2D(0)) ::   pslp   ! sea-level pressure                       [Pa]
      REAL(wp), INTENT(in   ), DIMENSION(A2D(nn_hls)) :: pst ! surface temperature                      [Celsius]
      REAL(wp), INTENT(in   ), DIMENSION(A2D(nn_hls)) :: pu  ! surface current at U-point (i-component) [m/s]
      REAL(wp), INTENT(in   ), DIMENSION(A2D(nn_hls)) :: pv  ! surface current at V-point (j-component) [m/s]
      REAL(wp), INTENT(in   ), DIMENSION(A2D(0)) ::   puatm  ! surface current seen by the atm at T-point (i-component) [m/s]
      REAL(wp), INTENT(in   ), DIMENSION(A2D(0)) ::   pvatm  ! surface current seen by the atm at T-point (j-component) [m/s]
      REAL(wp), INTENT(in   ), DIMENSION(A2D(0)) ::   pdqsr  ! downwelling solar (shortwave) radiation at surface [W/m^2]
      REAL(wp), INTENT(in   ), DIMENSION(A2D(0)) ::   pdqlw  ! downwelling longwave radiation at surface [W/m^2]
      REAL(wp), INTENT(  out), DIMENSION(A2D(0)) ::   ptsk   ! skin temp. (or SST if CS & WL not used)  [Celsius]
      REAL(wp), INTENT(  out), DIMENSION(A2D(0)) ::   pssq   ! specific humidity at pst                 [kg/kg]
      REAL(wp), INTENT(  out), DIMENSION(A2D(0)) ::   pcd_du
      REAL(wp), INTENT(  out), DIMENSION(A2D(0)) ::   psen
      REAL(wp), INTENT(  out), DIMENSION(A2D(0)) ::   plat
      REAL(wp), INTENT(  out), DIMENSION(A2D(0)) ::   pevp
      REAL(wp), INTENT(  out), OPTIONAL, DIMENSION(A2D(0)) ::   qlwn ! net longwave radiation at surface (MFS bulk only) [W/m^2]
      !
      INTEGER  ::   ji, jj               ! dummy loop indices
      REAL(wp) ::   zztmp                ! local variable
      REAL(wp) ::   zstmax, zstau
#if defined key_cyclone
      REAL(wp), DIMENSION(A2D(0)) ::   zwnd_i, zwnd_j    ! wind speed components at T-point
#endif
      REAL(wp), DIMENSION(A2D(0)) ::   zU_zu             ! bulk wind speed at height zu  [m/s]
      REAL(wp), DIMENSION(A2D(0)) ::   zcd_oce           ! momentum transfert coefficient over ocean
      REAL(wp), DIMENSION(A2D(0)) ::   zch_oce           ! sensible heat transfert coefficient over ocean
      REAL(wp), DIMENSION(A2D(0)) ::   zce_oce           ! latent   heat transfert coefficient over ocean
      REAL(wp), DIMENSION(A2D(0)) ::   zsspt             ! potential sea-surface temperature [K]
      REAL(wp), DIMENSION(A2D(0)) ::   zpre, ztabs       ! air pressure [Pa] & absolute temperature [K]
      REAL(wp), DIMENSION(A2D(0)) ::   zrspeed           ! reltive windspeed factor (MFS bulk only)
      REAL(wp), DIMENSION(A2D(0)) ::   zztmp1, zztmp2
      !!---------------------------------------------------------------------
      !
      ! local scalars ( place there for vector optimisation purposes)
      !                           ! Temporary conversion from Celcius to Kelvin (and set minimum value far above 0 K)
      ptsk(A2D(0)) = pst(A2D(0)) + rt0  ! by default: skin temperature = "bulk SST" (will remain this way if NCAR algorithm used!)

      ! sea surface potential temperature [K]
      zsspt(:,:) = theta_exner( ptsk(:,:), pslp(:,:) )

      ! --- cloud cover --- !
      cloud_fra(:,:) = sf(jp_cc)%fnow(:,:,1)

      ! ----------------------------------------------------------------------------- !
      !      0   Wind components and module at T-point relative to the moving ocean   !
      ! ----------------------------------------------------------------------------- !

      ! ... components ( U10m - U_oce ) at T-point (unmasked)
#if defined key_cyclone
      zwnd_i(:,:) = 0._wp
      zwnd_j(:,:) = 0._wp
      CALL wnd_cyc( kt, zwnd_i, zwnd_j )    ! add analytical tropical cyclone (Vincent et al. JGR 2012)
      DO_2D( 0, 0, 0, 0 )
         zwnd_i(ji,jj) = pwndi(ji,jj) + zwnd_i(ji,jj)
         zwnd_j(ji,jj) = pwndj(ji,jj) + zwnd_j(ji,jj)
         ! ... scalar wind at T-point (not masked)
         wndm(ji,jj) = SQRT( zwnd_i(ji,jj) * zwnd_i(ji,jj) + zwnd_j(ji,jj) * zwnd_j(ji,jj) )
      END_2D
#else
      ! ... scalar wind module at T-point (not masked)
      DO_2D( 0, 0, 0, 0 )
         wndm(ji,jj) = SQRT( pwndi(ji,jj) * pwndi(ji,jj) + pwndj(ji,jj) * pwndj(ji,jj) )
      END_2D
#endif
      ! ----------------------------------------------------------------------------- !
      !      I   Solar FLUX                                                           !
      ! ----------------------------------------------------------------------------- !

      ! ocean albedo assumed to be constant + modify now Qsr to include the diurnal cycle
      IF( ln_dm2dc ) THEN
         qsr(:,:) = ( 1._wp - albo ) * sbc_dcy( pdqsr(:,:) ) * smask0(:,:)
      ELSE
         qsr(:,:) = ( 1._wp - albo ) *          pdqsr(:,:)   * smask0(:,:)
      ENDIF

      ! ----------------------------------------------------------------------------- !
      !     II   Turbulent FLUXES                                                     !
      ! ----------------------------------------------------------------------------- !

      ! specific humidity at SST
      pssq(:,:) = rdct_qsat_salt * q_sat( ptsk(:,:), pslp(:,:) )

      ! Backup "bulk SST" and associated spec. hum.
      IF( ln_skin_cs .OR. ln_skin_wl ) THEN
         zztmp1(:,:) = zsspt(:,:)
         zztmp2(:,:) = pssq (:,:)
      ENDIF

      ! transfer coefficients (Cd, Ch, Ce at T-point, and more)
      SELECT CASE( nblk )   ! user-selected bulk parameterization
         !
      CASE( np_NCAR      )
         CALL turb_ncar    (     rn_zqt, rn_zu, zsspt, ptair, pssq, pqair, wndm, &
            &                zcd_oce, zch_oce, zce_oce, theta_zu, q_zu, zU_zu ,  &
            &                nb_iter=nn_iter_algo )
      CASE( np_COARE_3p0 )
         CALL turb_coare3p0( kt, rn_zqt, rn_zu, zsspt, ptair, pssq, pqair, wndm, &
            &                ln_skin_cs, ln_skin_wl,                             &
            &                zcd_oce, zch_oce, zce_oce, theta_zu, q_zu, zU_zu,   &
            &                nb_iter=nn_iter_algo,                               &
            &                Qsw=qsr(:,:), rad_lw=pdqlw(:,:), slp=pslp(:,:) )
      CASE( np_COARE_3p6 )
         CALL turb_coare3p6( kt, rn_zqt, rn_zu, zsspt, ptair, pssq, pqair, wndm, &
            &                ln_skin_cs, ln_skin_wl,                             &
            &                zcd_oce, zch_oce, zce_oce, theta_zu, q_zu, zU_zu,   &
            &                nb_iter=nn_iter_algo,                               &
            &                Qsw=qsr(:,:), rad_lw=pdqlw(:,:), slp=pslp(:,:) )
      CASE( np_ECMWF     )
         CALL turb_ecmwf   ( kt, rn_zqt, rn_zu, zsspt, ptair, pssq, pqair, wndm, &
            &                ln_skin_cs, ln_skin_wl,                             &
            &                zcd_oce, zch_oce, zce_oce, theta_zu, q_zu, zU_zu,   &
            &                nb_iter=nn_iter_algo,                               &
            &                Qsw=qsr(:,:), rad_lw=pdqlw(:,:), slp=pslp(:,:) )
      CASE( np_ANDREAS   )
         CALL turb_andreas (     rn_zqt, rn_zu, zsspt, ptair, pssq, pqair, wndm, &
            &                zcd_oce, zch_oce, zce_oce, theta_zu, q_zu, zU_zu,   &
            &                nb_iter=nn_iter_algo   )
      !
       CASE( np_MFS      )
         CALL turb_mfs     (     rn_zqt, rn_zu,  ptsk, ptair, sf(jp_humi)%fnow(:,:,1), &
          &                pssq, wndm, pwndi,pwndj, pu, pv,         &
          &                cloud_fra, qsr, zsspt,            &
          &                zcd_oce, zch_oce, zce_oce,               &
          &                theta_zu, q_zu, zU_zu, zrspeed, rhoa, qlwn,     &
          &                nb_iter=nn_iter_algo, slp=pslp(:,:)   )
      CASE DEFAULT
         CALL ctl_stop( 'STOP', 'sbc_oce: non-existing bulk parameterizaton selected' )
      END SELECT

      ! outputs
      IF( iom_use('Cd_oce') )   CALL iom_put( "Cd_oce",   zcd_oce * smask0(:,:) )
      IF( iom_use('Ce_oce') )   CALL iom_put( "Ce_oce",   zce_oce * smask0(:,:) )
      IF( iom_use('Ch_oce') )   CALL iom_put( "Ch_oce",   zch_oce * smask0(:,:) )
      !! LB: mainly here for debugging purpose:
      IF( iom_use('theta_zt') ) CALL iom_put( "theta_zt", (ptair-rt0) * smask0(:,:) ) ! potential temperature at z=zt
      IF( iom_use('q_zt') )     CALL iom_put( "q_zt",     pqair       * smask0(:,:) ) ! specific humidity       "
      IF( iom_use('theta_zu') ) CALL iom_put( "theta_zu", (theta_zu -rt0) * smask0(:,:) ) ! potential temperature at z=zu
      IF( iom_use('q_zu') )     CALL iom_put( "q_zu",     q_zu        * smask0(:,:) ) ! specific humidity       "
      IF( iom_use('ssq') )      CALL iom_put( "ssq",      pssq        * smask0(:,:) ) ! saturation specific humidity at z=0
      IF( iom_use('wspd_blk') ) CALL iom_put( "wspd_blk", zU_zu       * smask0(:,:) ) ! bulk wind speed at z=zu

      ! In the presence of sea-ice we do not use the cool-skin/warm-layer update of zsspt, pssq & ptsk from turb_*()
      IF( ln_skin_cs .OR. ln_skin_wl ) THEN
         WHERE ( fr_i(A2D(0)) > 0.001_wp )
            zsspt(:,:) = zztmp1(:,:)
            pssq (:,:) = zztmp2(:,:)
         END WHERE
         ! apply potential temperature increment to abolute SST
         ptsk(:,:) = ptsk(:,:) + ( zsspt(:,:) - zztmp1(:,:) )
      END IF

      !  Turbulent fluxes over ocean  => BULK_FORMULA @ sbc_phy.F90
      ! -------------------------------------------------------------

      IF( ln_abl ) THEN         !==  ABL formulation  ==!   multiplication by rho_air and turbulent fluxes computation done in ablstp

         DO_2D( 0, 0, 0, 0 )
            zztmp = zU_zu(ji,jj)
            wndm(ji,jj)   = zztmp                   ! Store zU_zu in wndm to compute ustar2 in ablmod
            pcd_du(ji,jj) = zztmp * zcd_oce(ji,jj)
            psen(ji,jj)   = zztmp * zch_oce(ji,jj)
            pevp(ji,jj)   = zztmp * zce_oce(ji,jj)
            zpre(ji,jj)   = pres_temp( pqair(ji,jj), pslp(ji,jj), rn_zu, ptpot=ptair(ji,jj), pta=ztabs(ji,jj) )
            rhoa(ji,jj)   = rho_air( ztabs(ji,jj), pqair(ji,jj), zpre(ji,jj) )
         END_2D

      ELSE                      !==  BLK formulation  ==!   turbulent fluxes computation
         IF( .NOT. ln_MFS) THEN
            DO_2D( 0, 0, 0, 0 )
               zpre(ji,jj) = pres_temp( q_zu(ji,jj), pslp(ji,jj), rn_zu, ptpot=theta_zu(ji,jj), pta=ztabs(ji,jj) )
               rhoa(ji,jj) = rho_air( ztabs(ji,jj), q_zu(ji,jj), zpre(ji,jj) )
            END_2D
         ENDIF

         CALL BULK_FORMULA( rn_zu, zsspt(:,:), pssq(:,:), theta_zu(:,:), q_zu(:,:), &
            &                      zcd_oce(:,:), zch_oce(:,:), zce_oce(:,:),        &
            &                      wndm(:,:), zU_zu(:,:), pslp(:,:), rhoa(:,:),     &
            &                      taum(:,:), psen(:,:), plat(:,:),                 &
            &                      pEvap=pevp(:,:), pfact_evap=rn_efac )

         psen(:,:) = psen(:,:) * smask0(:,:)
         plat(:,:) = plat(:,:) * smask0(:,:)
         taum(:,:) = taum(:,:) * smask0(:,:)
         pevp(:,:) = pevp(:,:) * smask0(:,:)

         DO_2D( 0, 0, 0, 0 )
            IF( wndm(ji,jj) > 0._wp ) THEN
               zztmp = taum(ji,jj) / wndm(ji,jj)
#if defined key_cyclone
               utau(ji,jj) = zztmp * zwnd_i(ji,jj)
               vtau(ji,jj) = zztmp * zwnd_j(ji,jj)
#else
               utau(ji,jj) = zztmp * pwndi(ji,jj)
               vtau(ji,jj) = zztmp * pwndj(ji,jj)
#endif
            ELSE
               utau(ji,jj) = 0._wp
               vtau(ji,jj) = 0._wp
            ENDIF
         END_2D

         IF( ln_crt_fbk ) THEN   ! aply eq. 10 and 11 of Renault et al. 2020 (doi: 10.1029/2019MS001715)
            zstmax = MIN( rn_stau_a * 3._wp + rn_stau_b, 0._wp )   ! set the max value of Stau corresponding to a wind of 3 m/s (<0)
            DO_2D( 0, 0, 0, 0 )
               zstau = MIN( rn_stau_a * wndm(ji,jj) + rn_stau_b, zstmax ) * smask0(ji,jj)   ! stau (<0) must be smaller than zstmax
               utau(ji,jj) = utau(ji,jj) + zstau * ( 0.5_wp * ( pu(ji-1,jj  ) + pu(ji,jj) ) - puatm(ji,jj) )
               vtau(ji,jj) = vtau(ji,jj) + zstau * ( 0.5_wp * ( pv(ji  ,jj-1) + pv(ji,jj) ) - pvatm(ji,jj) )
               taum(ji,jj) = SQRT( utau(ji,jj) * utau(ji,jj) + vtau(ji,jj) * vtau(ji,jj) )
            END_2D
            CALL lbc_lnk( 'sbcblk', utau, 'T', -1._wp, vtau, 'T', -1._wp )
         ENDIF

         IF(ln_MFS) THEN ! use relative wind (MFS bulk only)
            DO_2D( 0, 0, 0, 0 )
               utau(ji,jj) = zrspeed(ji,jj) * utau(ji,jj)/wndm(ji,jj)
               vtau(ji,jj) = zrspeed(ji,jj) * vtau(ji,jj)/wndm(ji,jj)
               taum(ji,jj) = SQRT( utau(ji,jj) * utau(ji,jj) + vtau(ji,jj)* vtau(ji,jj))
            END_2D
            CALL lbc_lnk( 'sbcblk', utau, 'T', -1._wp, vtau, 'T', -1._wp )
         ENDIF

         ! Saving open-ocean wind-stress (module and components)
         CALL iom_put( "taum_oce", taum(:,:) )   ! wind stress module
         !                                       ! LB: These 2 lines below mostly here for 'STATION_ASF' test-case
         CALL iom_put( "utau_oce", utau(:,:) )   ! utau
         CALL iom_put( "vtau_oce", vtau(:,:) )   ! vtau

         IF(sn_cfctl%l_prtctl) THEN
            CALL prt_ctl( tab2d_1=pssq   , clinfo1=' blk_oce_1: pssq   : ', mask1=tmask )
            CALL prt_ctl( tab2d_1=wndm   , clinfo1=' blk_oce_1: wndm   : ', mask1=tmask )
            CALL prt_ctl( tab2d_1=utau   , clinfo1=' blk_oce_1: utau   : ', mask1=tmask,   &
               &          tab2d_2=vtau   , clinfo2='            vtau   : ', mask2=tmask )
            CALL prt_ctl( tab2d_1=zcd_oce, clinfo1=' blk_oce_1: Cd     : ', mask1=tmask )
         ENDIF
         !
      ENDIF ! ln_blk / ln_abl

      ptsk(:,:) = ( ptsk(:,:) - rt0 ) * smask0(:,:)  ! Back to Celsius

      IF( ln_skin_cs .OR. ln_skin_wl ) THEN
         CALL iom_put( "t_skin" ,  ptsk                        )  ! T_skin in Celsius
         CALL iom_put( "dt_skin" , ptsk(A2D(0)) - pst(A2D(0))  )  ! T_skin - SST temperature difference
      ENDIF
      !
   END SUBROUTINE blk_oce_1


   SUBROUTINE blk_oce_2( ptair, pdqlw, pprec, psnow, &   ! <<= in
      &                   ptsk, psen, plat, pevp, qlwn     )   ! <<= in
      !!---------------------------------------------------------------------
      !!                     ***  ROUTINE blk_oce_2  ***
      !!
      !! ** Purpose :   finalize the momentum, heat and freshwater fluxes computation
      !!                at the ocean surface at each time step knowing Cd, Ch, Ce and
      !!                atmospheric variables (from ABL or external data)
      !!
      !! ** Outputs : - utau    : i-component of the stress at T-point  (N/m2)
      !!              - vtau    : j-component of the stress at T-point  (N/m2)
      !!              - taum    : Wind stress module at T-point         (N/m2)
      !!              - wndm    : Wind speed module at T-point          (m/s)
      !!              - qsr     : Solar heat flux over the ocean        (W/m2)
      !!              - qns     : Non Solar heat flux over the ocean    (W/m2)
      !!              - emp     : evaporation minus precipitation       (kg/m2/s)
      !!---------------------------------------------------------------------
      REAL(wp), INTENT(in), DIMENSION(A2D(0)) ::   ptair   ! potential temperature of air #LB: confirm!
      REAL(wp), INTENT(in), DIMENSION(A2D(0)) ::   pdqlw   ! downwelling longwave radiation at surface [W/m^2]
      REAL(wp), INTENT(in), DIMENSION(A2D(0)) ::   pprec
      REAL(wp), INTENT(in), DIMENSION(A2D(0)) ::   psnow
      REAL(wp), INTENT(in), DIMENSION(A2D(0)) ::   ptsk   ! SKIN surface temperature   [Celsius]
      REAL(wp), INTENT(in), DIMENSION(A2D(0)) ::   psen
      REAL(wp), INTENT(in), DIMENSION(A2D(0)) ::   plat
      REAL(wp), INTENT(in), DIMENSION(A2D(0)) ::   pevp
      REAL(wp), INTENT(in), OPTIONAL, DIMENSION(A2D(0)) ::   qlwn   ! net longwave radiation at surface (MFS only) [W/m^2]
      !
      INTEGER  ::   ji, jj               ! dummy loop indices
      REAL(wp) ::   zztmp,zz1,zz2,zz3    ! local variable
      REAL(wp), DIMENSION(A2D(0)) ::   zqlw              ! net long wave radiative heat flux
      REAL(wp), DIMENSION(A2D(0)) ::   zcptrain, zcptsnw, zcptn ! Heat content per unit mass (J/kg)
      !!---------------------------------------------------------------------
      !
      DO_2D( 0, 0, 0, 0 )
         ! Heat content per unit mass (J/kg)
         zcptrain(ji,jj) = (      ptair(ji,jj)        - rt0 ) * rcp  * smask0(ji,jj)
         zcptsnw (ji,jj) = ( MIN( ptair(ji,jj), rt0 ) - rt0 ) * rcpi * smask0(ji,jj)
         zcptn   (ji,jj) =        ptsk (ji,jj)                * rcp  * smask0(ji,jj)
         !
      END_2D
      ! ----------------------------------------------------------------------------- !
      !     III    Net longwave radiative FLUX                                        !
      ! ----------------------------------------------------------------------------- !
      !! #LB: now moved after Turbulent fluxes because must use the skin temperature rather than bulk SST
      !! (ptsk is skin temperature if ln_skin_cs==.TRUE. .OR. ln_skin_wl==.TRUE.)
      zqlw(:,:) = qlw_net( pdqlw(:,:), ptsk(:,:)+rt0 )
      IF( ln_MFS ) zqlw(:,:) = qlwn(:,:)   ! net LW already computed by MFS
      ! ----------------------------------------------------------------------------- !
      !     IV    Total FLUXES                                                       !
      ! ----------------------------------------------------------------------------- !
      !
      DO_2D( 0, 0, 0, 0 )
         emp (ji,jj) = ( pevp(ji,jj) - pprec(ji,jj) * rn_pfac ) * smask0(ji,jj)          ! mass flux (evap. - precip.)
         !
         qns(ji,jj) =    zqlw(ji,jj) + psen(ji,jj) + plat(ji,jj)                     &   ! Downward Non Solar
            &         - psnow(ji,jj) * rn_pfac * rLfus                               &   ! remove latent melting heat for solid precip
            &         -  pevp(ji,jj) * zcptn(ji,jj)                                  &   ! remove evap heat content at SST
            &         + ( pprec(ji,jj) - psnow(ji,jj) ) * rn_pfac * zcptrain(ji,jj)  &   ! add liquid precip heat content at Tair
            &         + psnow(ji,jj) * rn_pfac * zcptsnw(ji,jj)                          ! add solid  precip heat content at min(Tair,Tsnow)
         qns(ji,jj) = qns(ji,jj) * smask0(ji,jj)
      END_2D
      !
#if defined key_si3
      IF ( nn_ice == 2 ) THEN
         DO_2D( 0, 0, 0, 0 )
            qns_oce(ji,jj) = zqlw(ji,jj) + psen(ji,jj) + plat(ji,jj)                  ! non solar without emp (only needed by SI3)
            qsr_oce(ji,jj) = qsr(ji,jj)
         END_2D
      ENDIF
#endif
      !
      CALL iom_put( "rho_air"  , rhoa*smask0(:,:) )       ! output air density [kg/m^3]
      CALL iom_put( "evap_oce" , pevp )                    ! evaporation
      CALL iom_put( "qlw_oce"  , zqlw )                    ! output downward longwave heat over the ocean
      CALL iom_put( "qsb_oce"  , psen )                    ! output downward sensible heat over the ocean
      CALL iom_put( "qla_oce"  , plat )                    ! output downward latent   heat over the ocean
      tprecip(:,:) = pprec(:,:) * rn_pfac * smask0(:,:)   ! output total precipitation [kg/m2/s]
      sprecip(:,:) = psnow(:,:) * rn_pfac * smask0(:,:)   ! output solid precipitation [kg/m2/s]
      CALL iom_put( 'snowpre', sprecip )                   ! Snow
      CALL iom_put( 'precip' , tprecip )                   ! Total precipitation
      !
      IF ( nn_ice == 0 ) THEN
         CALL iom_put( "qemp_oce" , qns-zqlw-psen-plat )   ! output downward heat content of E-P over the ocean
         CALL iom_put( "qns_oce"  ,   qns  )               ! output downward non solar heat over the ocean
         CALL iom_put( "qsr_oce"  ,   qsr  )               ! output downward solar heat over the ocean
         CALL iom_put( "qt_oce"   ,   qns+qsr )            ! output total downward heat over the ocean
      ENDIF
      !
      IF(sn_cfctl%l_prtctl) THEN
         CALL prt_ctl(tab2d_1=zqlw , clinfo1=' blk_oce_2: zqlw  : ', mask1=tmask )
         CALL prt_ctl(tab2d_1=psen , clinfo1=' blk_oce_2: psen  : ', mask1=tmask )
         CALL prt_ctl(tab2d_1=plat , clinfo1=' blk_oce_2: plat  : ', mask1=tmask )
         CALL prt_ctl(tab2d_1=qns  , clinfo1=' blk_oce_2: qns   : ', mask1=tmask )
         CALL prt_ctl(tab2d_1=emp  , clinfo1=' blk_oce_2: emp   : ', mask1=tmask )
      ENDIF
      !
   END SUBROUTINE blk_oce_2


#if defined key_si3
   !!----------------------------------------------------------------------
   !!   'key_si3'                                       SI3 sea-ice model
   !!----------------------------------------------------------------------
   !!   blk_ice_1   : provide the air-ice stress
   !!   blk_ice_2   : provide the heat and mass fluxes at air-ice interface
   !!   blk_ice_qcn : provide ice surface temperature and snow/ice conduction flux (emulating conduction flux)
   !!----------------------------------------------------------------------

   SUBROUTINE blk_ice_1( pwndi, pwndj, ptair, pqair, pslp, ptsui,     &   ! inputs
      &                  putaui, pvtaui, pseni, pevpi, pssqi, pcd_dui )   ! optional outputs
      !!---------------------------------------------------------------------
      !!                     ***  ROUTINE blk_ice_1  ***
      !!
      !! ** Purpose :   provide the surface boundary condition over sea-ice
      !!
      !! ** Method  :   compute momentum using bulk formulation
      !!                formulea, ice variables and read atmospheric fields.
      !!                NB: ice drag coefficient is assumed to be a constant
      !!---------------------------------------------------------------------
      REAL(wp) , INTENT(in   ), DIMENSION(A2D(0)  ) ::   pslp    ! sea-level pressure [Pa]
      REAL(wp) , INTENT(in   ), DIMENSION(A2D(0)  ) ::   pwndi   ! atmospheric wind at T-point [m/s]
      REAL(wp) , INTENT(in   ), DIMENSION(A2D(0)  ) ::   pwndj   ! atmospheric wind at T-point [m/s]
      REAL(wp) , INTENT(in   ), DIMENSION(A2D(0)  ) ::   ptair   ! atmospheric potential temperature at T-point [K]
      REAL(wp) , INTENT(in   ), DIMENSION(A2D(0)  ) ::   pqair   ! atmospheric specific humidity at T-point [kg/kg]
      REAL(wp) , INTENT(in   ), DIMENSION(A2D(0)  ) ::   ptsui   ! sea-ice surface temperature [K]
      REAL(wp) , INTENT(  out), DIMENSION(A2D(nn_hls)), OPTIONAL ::   putaui  ! if ln_blk
      REAL(wp) , INTENT(  out), DIMENSION(A2D(nn_hls)), OPTIONAL ::   pvtaui  ! if ln_blk
      REAL(wp) , INTENT(  out), DIMENSION(A2D(0)  ), OPTIONAL ::   pseni   ! if ln_abl
      REAL(wp) , INTENT(  out), DIMENSION(A2D(0)  ), OPTIONAL ::   pevpi   ! if ln_abl
      REAL(wp) , INTENT(  out), DIMENSION(A2D(0)  ), OPTIONAL ::   pssqi   ! if ln_abl
      REAL(wp) , INTENT(  out), DIMENSION(A2D(0)  ), OPTIONAL ::   pcd_dui ! if ln_abl
      !
      INTEGER  ::   ji, jj    ! dummy loop indices
      REAL(wp) ::   zztmp                           ! temporary scalars
      REAL(wp), DIMENSION(A2D(0)) ::   ztmp, zsipt  ! temporary array
      REAL(wp), DIMENSION(A2D(0)) ::   zmsk00       ! O% concentration ice mask
      !!---------------------------------------------------------------------
      !
      ! treshold for outputs
      DO_2D( 0, 0, 0, 0 )
         IF( fr_i(ji,jj) >= 1.e-06_wp  ) THEN ; zmsk00(ji,jj) = 1._wp ! 1 if ice    , 0 if no ice
         ELSE                                 ; zmsk00(ji,jj) = 0._wp
         ENDIF
      END_2D

      ! ------------------------------------------------------------ !
      !    Wind module relative to the moving ice ( U10m - U_ice )   !
      ! ------------------------------------------------------------ !
      ! C-grid ice dynamics :   U & V-points (same as ocean)
      DO_2D( 0, 0, 0, 0 )
         wndm_ice(ji,jj) = SQRT( pwndi(ji,jj) * pwndi(ji,jj) + pwndj(ji,jj) * pwndj(ji,jj) )
      END_2D
      !
      ! potential sea-ice surface temperature [K]
      zsipt(:,:) = theta_exner( ptsui(:,:), pslp(:,:) )

      ! sea-ice <-> atmosphere bulk transfer coefficients
      SELECT CASE( nblk_ice )   ! user-selected bulk parameterization
         !
      CASE( np_ice_cst  )
         ! Constant bulk transfer coefficients over sea-ice:
         Cd_ice(:,:) = rn_Cd_ia
         Ch_ice(:,:) = rn_Ch_ia
         Ce_ice(:,:) = rn_Ce_ia
         ! no height adjustment, keeping zt values:
         theta_zu_i(:,:) = ptair(:,:)
         q_zu_i(:,:)     = pqair(:,:)
         !
      CASE( np_ice_an05 )  ! from Andreas(2005) equations
         ztmp(:,:) = q_sat( ptsui(:,:), pslp(:,:), l_ice=.TRUE. ) ! temporary array for SSQ
         CALL turb_ice_an05( rn_zqt, rn_zu, zsipt, ptair, ztmp, pqair, wndm_ice,       &
            &                      Cd_ice, Ch_ice, Ce_ice, theta_zu_i, q_zu_i )
         !
      CASE( np_ice_lu12 )  ! from Lupkes(2012) equations
         ztmp(:,:) = q_sat( ptsui(:,:), pslp(:,:), l_ice=.TRUE. ) ! temporary array for SSQ
         CALL turb_ice_lu12( rn_zqt, rn_zu, zsipt, ptair, ztmp, pqair, wndm_ice, fr_i, &
            &                      Cd_ice, Ch_ice, Ce_ice, theta_zu_i, q_zu_i )
         !
      CASE( np_ice_lg15 )  ! from Lupkes and Gryanik (2015) equations
         ztmp(:,:) = q_sat( ptsui(:,:), pslp(:,:), l_ice=.TRUE. ) ! temporary array for SSQ
         CALL turb_ice_lg15( rn_zqt, rn_zu, zsipt, ptair, ztmp, pqair, wndm_ice, fr_i, &
            &                      Cd_ice, Ch_ice, Ce_ice, theta_zu_i, q_zu_i )
         !
      CASE ( np_ice_frm )
         Cd_ice(:,:) = drag_ia(A2D(0))
         IF ( nn_frm == 1 .or. nn_frm == 3 ) THEN
           Ch_ice(:,:) = drag_ia(A2D(0))
           Ce_ice(:,:) = drag_ia(A2D(0))
         ELSE
           Ch_ice(:,:) = rn_Ch_ia
           Ce_ice(:,:) = rn_Ce_ia
         END IF
         ! no height adjustment, keeping zt values:
         theta_zu_i(:,:) = ptair(:,:)
         q_zu_i(:,:)     = pqair(:,:)
         !
      END SELECT


      IF( ln_blk ) THEN
         ! ---------------------------------------------------- !
         !    Wind stress relative to nonmoving ice ( U10m )    !
         ! ---------------------------------------------------- !
         ! supress moving ice in wind stress computation as we don't know how to do it properly...
         DO_2D( 0, 0, 0, 0 )
            zztmp         = rhoa(ji,jj) * Cd_ice(ji,jj) * wndm_ice(ji,jj)
            putaui(ji,jj) = zztmp * pwndi(ji,jj)
            pvtaui(ji,jj) = zztmp * pwndj(ji,jj)
         END_2D

         ! outputs
         !        LB: not weighted by the ice concentration
         IF( iom_use('taum_ice') ) CALL iom_put( 'taum_ice', SQRT( putaui*putaui + pvtaui*pvtaui ) * zmsk00 )
         !        LB: These 2 lines below mostly here for 'STATION_ASF' test-case
         IF( iom_use('utau_ice') ) CALL iom_put( "utau_ice", putaui * zmsk00 )
         IF( iom_use('vtau_ice') ) CALL iom_put( "vtau_ice", pvtaui * zmsk00 )
         !
         IF(sn_cfctl%l_prtctl)  CALL prt_ctl( tab2d_1=putaui  , clinfo1=' blk_ice: putaui : ', mask1=tmask   &
            &                               , tab2d_2=pvtaui  , clinfo2='          pvtaui : ', mask2=tmask )
      ELSE ! ln_abl

         DO_2D( 0, 0, 0, 0 )
            pcd_dui(ji,jj) = wndm_ice(ji,jj) * Cd_ice(ji,jj)
            pseni  (ji,jj) = wndm_ice(ji,jj) * Ch_ice(ji,jj)
            pevpi  (ji,jj) = wndm_ice(ji,jj) * Ce_ice(ji,jj)
         END_2D
         pssqi(:,:) = q_sat( ptsui(:,:), pslp(:,:), l_ice=.TRUE. ) ! more accurate way to obtain ssq

      ENDIF ! ln_blk  / ln_abl
      !
      ! outputs
      IF( iom_use('Cd_ice') ) CALL iom_put( "Cd_ice", Cd_ice * zmsk00 )
      IF( iom_use('Ce_ice') ) CALL iom_put( "Ce_ice", Ce_ice * zmsk00 )
      IF( iom_use('Ch_ice') ) CALL iom_put( "Ch_ice", Ch_ice * zmsk00 )
      !
      IF(sn_cfctl%l_prtctl)  CALL prt_ctl(tab2d_1=wndm_ice  , clinfo1=' blk_ice: wndm_ice : ', mask1=tmask )
      !
   END SUBROUTINE blk_ice_1


   SUBROUTINE blk_ice_2( ptsu, phs, phi, palb, ptair, pqair, pslp, pdqlw, pprec, psnow  )
      !!---------------------------------------------------------------------
      !!                     ***  ROUTINE blk_ice_2  ***
      !!
      !! ** Purpose :   provide the heat and mass fluxes at air-ice interface
      !!
      !! ** Method  :   compute heat and freshwater exchanged
      !!                between atmosphere and sea-ice using bulk formulation
      !!                formulea, ice variables and read atmmospheric fields.
      !!
      !! caution : the net upward water flux has with mm/day unit
      !!---------------------------------------------------------------------
      REAL(wp), DIMENSION(A2D(nn_hls),jpl), INTENT(in)  ::   ptsu   ! sea ice surface temperature [K]
      REAL(wp), DIMENSION(A2D(nn_hls),jpl), INTENT(in)  ::   phs    ! snow thickness
      REAL(wp), DIMENSION(A2D(nn_hls),jpl), INTENT(in)  ::   phi    ! ice thickness
      REAL(wp), DIMENSION(A2D(0     ),jpl), INTENT(in)  ::   palb   ! ice albedo (all skies)
      REAL(wp), DIMENSION(A2D(0     )    ), INTENT(in)  ::   ptair  ! potential temperature of air #LB: okay ???
      REAL(wp), DIMENSION(A2D(0     )    ), INTENT(in)  ::   pqair  ! specific humidity of air
      REAL(wp), DIMENSION(A2D(0     )    ), INTENT(in)  ::   pslp
      REAL(wp), DIMENSION(A2D(0     )    ), INTENT(in)  ::   pdqlw
      REAL(wp), DIMENSION(A2D(0     )    ), INTENT(in)  ::   pprec
      REAL(wp), DIMENSION(A2D(0     )    ), INTENT(in)  ::   psnow
      !!
      INTEGER  ::   ji, jj, jl               ! dummy loop indices
      REAL(wp) ::   zst, zst3, zsq, zsipt    ! local variable
      REAL(wp) ::   zcoef_dqlw, zcoef_dqla   !   -      -
      REAL(wp) ::   zztmp, zzblk, zztmp1, z1_rLsub   !   -      -
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   zmsk   ! temporary mask for prt_ctl
      REAL(wp), DIMENSION(A2D(0),jpl) ::   z_qlw         ! long wave heat flux over ice
      REAL(wp), DIMENSION(A2D(0),jpl) ::   z_qsb         ! sensible  heat flux over ice
      REAL(wp)                        ::   z_dqlw        ! long wave heat sensitivity over ice
      REAL(wp)                        ::   z_dqsb        ! sensible  heat sensitivity over ice
      REAL(wp)                        ::   ztmp          ! temporary scalar
      REAL(wp), DIMENSION(A2D(0))     ::   zevap, zsnw   ! evaporation and snw distribution after wind blowing (SI3)
      REAL(wp), DIMENSION(A2D(0))     ::   ztri
      REAL(wp), DIMENSION(A2D(0))     ::   zcptrain, zcptsnw, zcptn ! Heat content per unit mass (J/kg)
      !!---------------------------------------------------------------------
      !
      zcoef_dqlw = 4._wp * emiss_i * stefan             ! local scalars
      zztmp = 1. / ( 1. - albo )

      ! Heat content per unit mass (J/kg)
      zcptrain(:,:) = (      ptair(:,:)        - rt0 ) * rcp  * smask0(:,:)
      zcptsnw (:,:) = ( MIN( ptair(:,:), rt0 ) - rt0 ) * rcpi * smask0(:,:)
      zcptn   (:,:) =        sst_m(A2D(0))             * rcp  * smask0(:,:)
      !
      !                                     ! ========================== !
      DO jl = 1, jpl                        !  Loop over ice categories  !
         !                                  ! ========================== !
         DO_2D( 0, 0, 0, 0 )

            zst   = ptsu(ji,jj,jl)                                ! surface temperature of sea-ice [K]
            zsq   = q_sat( zst, pslp(ji,jj), l_ice=.TRUE. )       ! surface saturation specific humidity when ice present
            zsipt = theta_exner( zst, pslp(ji,jj) )               ! potential sea-ice surface temperature [K]

            ! ----------------------------!
            !      I   Radiative FLUXES   !
            ! ----------------------------!
            ! Short Wave (sw)
            qsr_ice(ji,jj,jl) = zztmp * ( 1. - palb(ji,jj,jl) ) * qsr(ji,jj)

            ! Long  Wave (lw)
            zst3 = zst * zst * zst
            z_qlw(ji,jj,jl)   = emiss_i * ( pdqlw(ji,jj) - stefan * zst * zst3 ) * smask0(ji,jj)
            ! lw sensitivity
            z_dqlw = zcoef_dqlw * zst3

            ! ----------------------------!
            !     II    Turbulent FLUXES  !
            ! ----------------------------!

            ! ... turbulent heat fluxes with Ch_ice recalculated in blk_ice_1

            ! Common term in bulk F. equations...
            zzblk = rhoa(ji,jj) * wndm_ice(ji,jj)

            ! Sensible Heat
            zztmp1 = zzblk * rCp_air * Ch_ice(ji,jj)
            z_qsb (ji,jj,jl) = zztmp1 * (zsipt - theta_zu_i(ji,jj))
            z_dqsb = zztmp1                        ! ==> Qsens sensitivity (Dqsb_ice/Dtn_ice)

            ! Latent Heat
            zztmp1 = zzblk * rLsub * Ce_ice(ji,jj)
            qla_ice (ji,jj,jl) = zztmp1 * (zsq - q_zu_i(ji,jj))
            dqla_ice(ji,jj,jl) = zztmp1*dq_sat_dt_ice(zst, pslp(ji,jj)) ! ==> Qlat sensitivity  (dQlat/dT)
            !                                                                 !#LB: dq_sat_dt_ice() in "sbc_phy.F90"

            ! ----------------------------!
            !     III    Total FLUXES     !
            ! ----------------------------!
            ! Downward Non Solar flux
            qns_ice (ji,jj,jl) =     z_qlw (ji,jj,jl) - z_qsb (ji,jj,jl) - qla_ice (ji,jj,jl)
            ! Total non solar heat flux sensitivity for ice
            dqns_ice(ji,jj,jl) = - ( z_dqlw + z_dqsb + dqla_ice(ji,jj,jl) ) !#LB: correct signs ????
            IF( ji == Nis0 .AND. jj == Njs0 .AND. jl == 1 ) THEN
               l3_qsat_ice  = zsq
               l3_theta_ice = zsipt
               l3_qlw_ice   = z_qlw(ji,jj,jl)
               l3_qsb_ice   = z_qsb(ji,jj,jl)
               l3_dqlw_ice  = z_dqlw
            ENDIF

         END_2D
         !
      END DO
      !
      z1_rLsub = 1._wp / rLsub
      DO_2D( 0, 0, 0, 0 )
         ! --- precipitation --- !
         tprecip(ji,jj) = pprec(ji,jj) * rn_pfac * smask0(ji,jj)  ! total precipitation [kg/m2/s]
         sprecip(ji,jj) = psnow(ji,jj) * rn_pfac * smask0(ji,jj)  ! solid precipitation [kg/m2/s]

         ! --- evaporation --- !
         zevap(ji,jj) = emp(ji,jj) + tprecip(ji,jj)   ! evaporation over ocean  !LB: removed rn_efac here, correct???
         DO jl = 1, jpl
            evap_ice (ji,jj,jl) = rn_efac * qla_ice (ji,jj,jl) * z1_rLsub    ! sublimation
            devap_ice(ji,jj,jl) = rn_efac * dqla_ice(ji,jj,jl) * z1_rLsub    ! d(sublimation)/dT
         ENDDO
      END_2D

      zsnw(:,:) = 0._wp
      CALL ice_var_snwblow( (1.-at_i_b(:,:)), zsnw(:,:) )  ! snow distribution over ice after wind blowing
      DO_2D( 0, 0, 0, 0 )
         ! --- evaporation minus precipitation --- !
         emp_oce(ji,jj) = ( 1._wp - at_i_b(ji,jj) ) * zevap(ji,jj) - ( tprecip(ji,jj) - sprecip(ji,jj) ) - sprecip(ji,jj) * (1._wp - zsnw(ji,jj) )
         emp_ice(ji,jj) = SUM( a_i_b(ji,jj,:) * evap_ice(ji,jj,:) ) - sprecip(ji,jj) * zsnw(ji,jj)
         emp_tot(ji,jj) = emp_oce(ji,jj) + emp_ice(ji,jj)

         ! --- heat flux associated with emp --- !
         qemp_oce(ji,jj) = - ( 1._wp - at_i_b(ji,jj) ) * zevap(ji,jj) * zcptn(ji,jj)         & ! evap at sst
            &          + ( tprecip(ji,jj) - sprecip(ji,jj) )   *   zcptrain(ji,jj)         & ! liquid precip at Tair
            &          +   sprecip(ji,jj) * ( 1._wp - zsnw(ji,jj) ) * ( zcptsnw (ji,jj) - rLfus ) ! solid precip at min(Tair,Tsnow)
         qemp_ice(ji,jj) = sprecip(ji,jj) *           zsnw(ji,jj)   * ( zcptsnw (ji,jj) - rLfus ) ! solid precip (only)

         ! --- total solar and non solar fluxes --- !
         qns_tot(ji,jj) = ( 1._wp - at_i_b(ji,jj) ) * qns_oce(ji,jj) + SUM( a_i_b(ji,jj,:) * qns_ice(ji,jj,:) )  &
            &           + qemp_ice(ji,jj) + qemp_oce(ji,jj)
         qsr_tot(ji,jj) = ( 1._wp - at_i_b(ji,jj) ) * qsr_oce(ji,jj) + SUM( a_i_b(ji,jj,:) * qsr_ice(ji,jj,:) )

         ! --- heat content of precip over ice in J/m3 (to be used in 1D-thermo) --- !
         qprec_ice(ji,jj) = rhos * ( zcptsnw(ji,jj) - rLfus )

         ! --- heat content of evap over ice in W/m2 (to be used in 1D-thermo) ---
         DO jl = 1, jpl
            qevap_ice(ji,jj,jl) = 0._wp ! should be -evap_ice(ji,jj,jl)*( ( Tice - rt0 ) * rcpi * smask0(ji,jj) )
            !                         ! But we do not have Tice => consider it at 0degC => evap=0
         ENDDO
      END_2D

      ! --- shortwave radiation transmitted thru the surface scattering layer (W/m2) --- !
      IF( nn_qtrice == 0 ) THEN
         ! formulation derived from Grenfell and Maykut (1977), where transmission rate
         !    1) depends on cloudiness
         !    2) is 0 when there is any snow
         !    3) tends to 1 for thin ice
         ztri(:,:) = 0.18 * ( 1.0 - cloud_fra(:,:) ) + 0.35 * cloud_fra(:,:)  ! surface transmission when hi>10cm
         DO jl = 1, jpl
            DO_2D( 0, 0, 0, 0 )
               IF( phs(ji,jj,jl) <= 0._wp ) THEN
                  ztmp = ztri(ji,jj)                                       ! constant (ztri) when hi>10cm
                  IF( phi(ji,jj,jl) <  0.1_wp )   &                        ! linear decrease from hi=0 to 10cm
                     & ztmp = ztmp + ( 1._wp - ztmp ) * ( 1._wp - phi(ji,jj,jl) * 10._wp )
                  qtr_ice_top(ji,jj,jl) = qsr_ice(ji,jj,jl) * ztmp
               ELSE                                                        ! zero when hs>0
                  qtr_ice_top(ji,jj,jl) = 0._wp
               ENDIF
            END_2D
         ENDDO
      ELSEIF( nn_qtrice == 1 ) THEN
         ! formulation is derived from the thesis of M. Lebrun (2019).
         !    It represents the best fit using several sets of observations
         !    It comes with snow conductivities adapted to freezing/melting conditions (see icethd_zdf_bl99.F90)
         qtr_ice_top(:,:,:) = 0.3_wp * qsr_ice(:,:,:)
      ENDIF
      !
      CALL iom_put( 'snowpre', sprecip )                 ! Snow precipitation
      CALL iom_put( 'precip' , tprecip )                 ! Total precipitation
      IF( iom_use('evap_ao_cea') .OR. iom_use('hflx_evap_cea') ) THEN
         CALL iom_put( 'evap_ao_cea'  , zevap(:,:) * ( 1._wp - at_i_b(:,:) ) * smask0(:,:)              )   ! ice-free oce evap (cell average)
         CALL iom_put( 'hflx_evap_cea', zevap(:,:) * ( 1._wp - at_i_b(:,:) ) * smask0(:,:) * zcptn(:,:) )   ! heat flux from evap (cell average)
      ENDIF
      IF( iom_use('rain') .OR. iom_use('rain_ao_cea') .OR. iom_use('hflx_rain_cea') ) THEN
         CALL iom_put( 'rain'         ,   tprecip(:,:) - sprecip(:,:)                             )          ! liquid precipitation
         CALL iom_put( 'rain_ao_cea'  , ( tprecip(:,:) - sprecip(:,:) ) * ( 1._wp - at_i_b(:,:) ) )          ! liquid precipitation over ocean (cell average)
         CALL iom_put( 'hflx_rain_cea', ( tprecip(:,:) - sprecip(:,:) ) * zcptrain(:,:) )                    ! heat flux from rain (cell average)
      ENDIF
      IF(  iom_use('snow_ao_cea')   .OR. iom_use('snow_ai_cea')      .OR. &
         & iom_use('hflx_snow_cea') .OR. iom_use('hflx_snow_ao_cea') .OR. iom_use('hflx_snow_ai_cea')  )  THEN
         CALL iom_put( 'snow_ao_cea'     , sprecip(:,:)                            * ( 1._wp - zsnw(:,:) ) ) ! Snow over ice-free ocean  (cell average)
         CALL iom_put( 'snow_ai_cea'     , sprecip(:,:)                            *           zsnw(:,:)   ) ! Snow over sea-ice         (cell average)
         CALL iom_put( 'hflx_snow_cea'   , sprecip(:,:) * ( zcptsnw(:,:) - rLfus ) )                         ! heat flux from snow (cell average)
         CALL iom_put( 'hflx_snow_ao_cea', sprecip(:,:) * ( zcptsnw(:,:) - rLfus ) * ( 1._wp - zsnw(:,:) ) ) ! heat flux from snow (over ocean)
         CALL iom_put( 'hflx_snow_ai_cea', sprecip(:,:) * ( zcptsnw(:,:) - rLfus ) *           zsnw(:,:)   ) ! heat flux from snow (over ice)
      ENDIF
      IF( iom_use('hflx_prec_cea') ) THEN                                                                    ! heat flux from precip (cell average)
         CALL iom_put('hflx_prec_cea' ,    sprecip(:,:)                  * ( zcptsnw (:,:) - rLfus )  &
            &                          + ( tprecip(:,:) - sprecip(:,:) ) *   zcptrain(:,:) )
      ENDIF
      IF( iom_use('subl_ai_cea') .OR. iom_use('hflx_subl_cea') ) THEN
         CALL iom_put( 'subl_ai_cea'  , SUM( a_i_b(:,:,:) *  evap_ice(:,:,:), dim=3 ) * smask0(:,:) ) ! Sublimation over sea-ice (cell average)
         CALL iom_put( 'hflx_subl_cea', SUM( a_i_b(:,:,:) * qevap_ice(:,:,:), dim=3 ) * smask0(:,:) ) ! Heat flux from sublimation (cell average)
      ENDIF
      !
      IF(sn_cfctl%l_prtctl) THEN
         ALLOCATE(zmsk(A2D(0),jpl))
         DO jl = 1, jpl
            zmsk(:,:,jl) = smask0(:,:)
         END DO
         CALL prt_ctl(tab3d_1=qla_ice , clinfo1=' blk_ice: qla_ice  : ', mask1=zmsk,   &
            &         tab3d_2=z_qsb   , clinfo2=' z_qsb    : '         , mask2=zmsk, kdim=jpl)
         CALL prt_ctl(tab3d_1=z_qlw   , clinfo1=' blk_ice: z_qlw    : ', mask1=zmsk,   &
            &         tab3d_2=dqla_ice, clinfo2=' dqla_ice : '         , mask2=zmsk, kdim=jpl)
!!$         CALL prt_ctl(tab3d_1=z_dqsb  , clinfo1=' blk_ice: z_dqsb   : ', mask1=zmsk,   &
!!$            &         tab3d_2=z_dqlw  , clinfo2=' z_dqlw   : '         , mask2=zmsk, kdim=jpl)
         CALL prt_ctl(tab3d_1=dqns_ice, clinfo1=' blk_ice: dqns_ice : ', mask1=zmsk,   &
            &         tab3d_2=qsr_ice , clinfo2=' qsr_ice  : '         , mask2=zmsk, kdim=jpl)
         CALL prt_ctl(tab3d_1=ptsu(A2D(0),:), clinfo1=' blk_ice: ptsu     : ', mask1=zmsk,   &
            &         tab3d_2=qns_ice , clinfo2=' qns_ice  : '         , mask2=zmsk, kdim=jpl)
         CALL prt_ctl(tab2d_1=tprecip , clinfo1=' blk_ice: tprecip  : ', mask1=tmask,   &
            &         tab2d_2=sprecip , clinfo2=' sprecip  : '         , mask2=tmask        )
         DEALLOCATE(zmsk)
      ENDIF

      !#LB:
      ! air-ice heat flux components that are not written from ice_stp()@icestp.F90:
      IF( iom_use('qla_ice') )  CALL iom_put( 'qla_ice', SUM( - qla_ice * a_i_b, dim=3 ) ) !#LB: sign consistent with what's done for ocean
      IF( iom_use('qsb_ice') )  CALL iom_put( 'qsb_ice', SUM( -   z_qsb * a_i_b, dim=3 ) ) !#LB:     ==> negative => loss of heat for sea-ice
      IF( iom_use('qlw_ice') )  CALL iom_put( 'qlw_ice', SUM(     z_qlw * a_i_b, dim=3 ) )
      !#LB.

   END SUBROUTINE blk_ice_2


   SUBROUTINE blk_ice_qcn( ld_virtual_itd, ptb, phs, phi,      &   ! <<== in
      &                                    pqcn_ice, pqml_ice, &   ! ==>> out
      &                                    pqns_ice, ptsu )        ! ==>> inout
      !!---------------------------------------------------------------------
      !!                     ***  ROUTINE blk_ice_qcn  ***
      !!
      !! ** Purpose :   Compute surface temperature and snow/ice conduction flux
      !!                to force sea ice / snow thermodynamics
      !!                in the case conduction flux is emulated
      !!
      !! ** Method  :   compute surface energy balance assuming neglecting heat storage
      !!                following the 0-layer Semtner (1976) approach
      !!
      !! ** Outputs : - ptsu    : sea-ice / snow surface temperature (K)
      !!              - qcn_ice : surface inner conduction flux (W/m2)
      !!
      !!---------------------------------------------------------------------
      LOGICAL                             , INTENT(in   ) ::   ld_virtual_itd  ! single-category option
      REAL(wp), DIMENSION(A2D(0     )    ), INTENT(in   ) ::   ptb             ! sea ice base temperature
      REAL(wp), DIMENSION(A2D(nn_hls),jpl), INTENT(in   ) ::   phs             ! snow thickness
      REAL(wp), DIMENSION(A2D(nn_hls),jpl), INTENT(in   ) ::   phi             ! sea ice thickness
      REAL(wp), DIMENSION(A2D(0     ),jpl), INTENT(  out) ::   pqcn_ice
      REAL(wp), DIMENSION(A2D(0     ),jpl), INTENT(  out) ::   pqml_ice
      REAL(wp), DIMENSION(A2D(0     ),jpl), INTENT(inout) ::   pqns_ice
      REAL(wp), DIMENSION(A2D(nn_hls),jpl), INTENT(inout) ::   ptsu            ! sea ice / snow surface temperature
     !
      INTEGER , PARAMETER ::   nit = 10                  ! number of iterations
      REAL(wp), PARAMETER ::   zepsilon = 0.1_wp         ! characteristic thickness for enhanced conduction
      !
      INTEGER  ::   ji, jj, jl           ! dummy loop indices
      INTEGER  ::   iter                 ! local integer
      REAL(wp) ::   zfac, zfac2, zfac3   ! local scalars
      REAL(wp) ::   zkeff_h, ztsu, ztsu0 !
      REAL(wp) ::   zqc, zqnet           !
      REAL(wp) ::   zhe, zqa0            !
      REAL(wp), DIMENSION(A2D(0),jpl) ::   zgfac   ! enhanced conduction factor
      !!---------------------------------------------------------------------

      ! -------------------------------------!
      !      I   Enhanced conduction factor  !
      ! -------------------------------------!
      ! Emulates the enhancement of conduction by unresolved thin ice (ld_virtual_itd = T)
      ! Fichefet and Morales Maqueda, JGR 1997
      !
      zgfac(:,:,:) = 1._wp

      IF( ld_virtual_itd ) THEN
         !
         zfac  = 1._wp /  ( rcnd_s + rcnd_i )
         zfac2 = EXP(1._wp) * 0.5_wp * zepsilon
         zfac3 = 2._wp / zepsilon
         !
         DO jl = 1, jpl
            DO_2D( 0, 0, 0, 0 )
               zhe = ( rcnd_s * phi(ji,jj,jl) + rcnd_i * phs(ji,jj,jl) ) * zfac                              ! Effective thickness
               IF( zhe >=  zfac2 )   zgfac(ji,jj,jl) = MIN( 2._wp, 0.5_wp * ( 1._wp + LOG( zhe * zfac3 ) ) ) ! Enhanced conduction factor
            END_2D
         END DO
         !
      ENDIF

      ! -------------------------------------------------------------!
      !      II   Surface temperature and conduction flux            !
      ! -------------------------------------------------------------!
      !
      zfac = rcnd_i * rcnd_s
      !
      DO jl = 1, jpl
         DO_2D( 0, 0, 0, 0 )
            !
            zkeff_h = zfac * zgfac(ji,jj,jl) / &                                    ! Effective conductivity of the snow-ice system divided by thickness
               &      ( rcnd_i * phs(ji,jj,jl) + rcnd_s * MAX( 0.01, phi(ji,jj,jl) ) )
            ztsu    = ptsu(ji,jj,jl)                                                ! Store current iteration temperature
            ztsu0   = ptsu(ji,jj,jl)                                                ! Store initial surface temperature
            zqa0    = qsr_ice(ji,jj,jl) - qtr_ice_top(ji,jj,jl) + pqns_ice(ji,jj,jl) ! Net initial atmospheric heat flux
            !
            DO iter = 1, nit     ! --- Iterative loop
               zqc   = zkeff_h * ( ztsu - ptb(ji,jj) )                              ! Conduction heat flux through snow-ice system (>0 downwards)
               zqnet = zqa0 + dqns_ice(ji,jj,jl) * ( ztsu - ptsu(ji,jj,jl) ) - zqc  ! Surface energy budget
               ztsu  = ztsu - zqnet / ( dqns_ice(ji,jj,jl) - zkeff_h )              ! Temperature update
            END DO
            !
            ptsu    (ji,jj,jl) = MIN( rt0, ztsu )
            pqcn_ice(ji,jj,jl) = zkeff_h * ( ptsu(ji,jj,jl) - ptb(ji,jj) )
            pqns_ice(ji,jj,jl) = pqns_ice(ji,jj,jl) + dqns_ice(ji,jj,jl) * ( ptsu(ji,jj,jl) - ztsu0 )
            pqml_ice(ji,jj,jl) = ( qsr_ice(ji,jj,jl) - qtr_ice_top(ji,jj,jl) + pqns_ice(ji,jj,jl) - pqcn_ice(ji,jj,jl) )  &
               &   * MAX( 0._wp , SIGN( 1._wp, ptsu(ji,jj,jl) - rt0 ) )

            ! --- Diagnose the heat loss due to changing non-solar flux (as in icethd_zdf_bl99) --- !
            hfx_err_dif(ji,jj) = hfx_err_dif(ji,jj) - ( dqns_ice(ji,jj,jl) * ( ptsu(ji,jj,jl) - ztsu0 ) ) * a_i_b(ji,jj,jl)

         END_2D
         !
      END DO
      !
   END SUBROUTINE blk_ice_qcn

#endif

   !!======================================================================
END MODULE sbcblk
