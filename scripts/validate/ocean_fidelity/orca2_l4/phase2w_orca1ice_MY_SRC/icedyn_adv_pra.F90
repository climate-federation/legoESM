MODULE icedyn_adv_pra
   !!======================================================================
   !!                       ***  MODULE icedyn_adv_pra   ***
   !!   sea-ice : advection => Prather scheme
   !!======================================================================
   !! History :       !  2008-03  (M. Vancoppenolle) original code
   !!            4.0  !  2018     (many people)      SI3 [aka Sea Ice cube]
   !!--------------------------------------------------------------------
#if defined key_si3
   !!----------------------------------------------------------------------
   !!   'key_si3'                                       SI3 sea-ice model
   !!----------------------------------------------------------------------
   !!   ice_dyn_adv_pra : advection of sea ice using Prather scheme
   !!   adv_x, adv_y    : Prather scheme applied in i- and j-direction, resp.
   !!   adv_pra_init    : initialisation of the Prather scheme
   !!   adv_pra_rst     : read/write Prather field in ice restart file, or initialized to zero
   !!----------------------------------------------------------------------
   USE par_ice        ! SI3 parameters
   USE dom_oce, ONLY : nn_hls, tmask, umask, vmask, ssmask, ssumask, ssvmask, ssfmask
   USE phycst         ! physical constant
   USE ice            ! sea-ice variables
   USE sbc_oce , ONLY : nn_fsbc   ! frequency of sea-ice call
   USE icevar         ! sea-ice: operations
   !
   USE in_out_manager ! I/O manager
   USE iom            ! I/O manager library
   USE lib_mpp        ! MPP library
   USE lbclnk         ! lateral boundary conditions (or mpp links)

   IMPLICIT NONE
   PRIVATE

   PUBLIC   ice_dyn_adv_pra   ! called by icedyn_adv
   PUBLIC   adv_pra_init      ! called by icedyn_adv

   ! Moments for advection
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   sxice, syice, sxxice, syyice, sxyice   ! ice thickness
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   sxsn , sysn , sxxsn , syysn , sxysn    ! snow thickness
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   sxa  , sya  , sxxa  , syya  , sxya     ! ice concentration
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   sxsal, sysal, sxxsal, syysal, sxysal   ! ice salinity
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   sxage, syage, sxxage, syyage, sxyage   ! ice age
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:,:) ::   sxc0 , syc0 , sxxc0 , syyc0 , sxyc0    ! snow layers heat content
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:,:) ::   sxe  , sye  , sxxe  , syye  , sxye     ! ice layers heat content
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:,:) ::   sxsi , sysi , sxxsi , syysi , sxysi    ! ice layers salt content
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   sxap , syap , sxxap , syyap , sxyap    ! melt pond fraction
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   sxvp , syvp , sxxvp , syyvp , sxyvp    ! melt pond volume
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   sxvl , syvl , sxxvl , syyvl , sxyvl    ! melt pond lid volume

   !! * Substitutions
#  include "do_loop_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/ICE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE ice_dyn_adv_pra(         kt, pu_ice, pv_ice, ph_i, ph_s, ph_ip,  &
      &                        pato_i, pv_i, pv_s, psv_i, poa_i, pa_i, pa_ip, pv_ip, pv_il, pe_s, pe_i, pszv_i )
      !!----------------------------------------------------------------------
      !!                **  routine ice_dyn_adv_pra  **
      !!
      !! ** purpose :   Computes and adds the advection trend to sea-ice
      !!
      !! ** method  :   Uses Prather second order scheme that advects tracers
      !!                but also their quadratic forms. The method preserves
      !!                tracer structures by conserving second order moments.
      !!
      !! Reference:  Prather, 1986, JGR, 91, D6. 6671-6681.
      !!----------------------------------------------------------------------
      INTEGER                     , INTENT(in   ) ::   kt         ! time step
      REAL(wp), DIMENSION(:,:)    , INTENT(in   ) ::   pu_ice     ! ice i-velocity
      REAL(wp), DIMENSION(:,:)    , INTENT(in   ) ::   pv_ice     ! ice j-velocity
      REAL(wp), DIMENSION(:,:,:)  , INTENT(in   ) ::   ph_i       ! ice thickness
      REAL(wp), DIMENSION(:,:,:)  , INTENT(in   ) ::   ph_s       ! snw thickness
      REAL(wp), DIMENSION(:,:,:)  , INTENT(in   ) ::   ph_ip      ! ice pond thickness
      REAL(wp), DIMENSION(:,:)    , INTENT(inout) ::   pato_i     ! open water area
      REAL(wp), DIMENSION(:,:,:)  , INTENT(inout) ::   pv_i       ! ice volume
      REAL(wp), DIMENSION(:,:,:)  , INTENT(inout) ::   pv_s       ! snw volume
      REAL(wp), DIMENSION(:,:,:)  , INTENT(inout) ::   psv_i      ! salt content
      REAL(wp), DIMENSION(:,:,:)  , INTENT(inout) ::   poa_i      ! age content
      REAL(wp), DIMENSION(:,:,:)  , INTENT(inout) ::   pa_i       ! ice concentration
      REAL(wp), DIMENSION(:,:,:)  , INTENT(inout) ::   pa_ip      ! melt pond fraction
      REAL(wp), DIMENSION(:,:,:)  , INTENT(inout) ::   pv_ip      ! melt pond volume
      REAL(wp), DIMENSION(:,:,:)  , INTENT(inout) ::   pv_il      ! melt pond lid thickness
      REAL(wp), DIMENSION(:,:,:,:), INTENT(inout) ::   pe_s       ! snw heat content
      REAL(wp), DIMENSION(:,:,:,:), INTENT(inout) ::   pe_i       ! ice heat content
      REAL(wp), DIMENSION(:,:,:,:), INTENT(inout) ::   pszv_i     ! ice salt content
      !
      INTEGER  ::   ji, jj, jk, jl, jt, ihls ! dummy loop indices
      INTEGER  ::   icycle                   ! number of sub-timestep for the advection
      REAL(wp) ::   zdt, z1_dt               !   -      -
      REAL(wp) ::   zati2
      REAL(wp) ::   zcfl
      REAL(wp), DIMENSION(A2D(0))         ::   zati1
      REAL(wp), DIMENSION(jpi,jpj)        ::   zudy, zvdx
      REAL(wp), DIMENSION(jpi,jpj)        ::   zh_i, zh_s, zhi_max, zhs_max, zhip_max, zsi_max
      REAL(wp), DIMENSION(jpi,jpj,nlay_i) ::   ze_i, zei_max, zszi_max
      REAL(wp), DIMENSION(jpi,jpj,nlay_s) ::   ze_s, zes_max
      REAL(wp), DIMENSION(jpi,jpj)        ::   zarea
      REAL(wp), DIMENSION(jpi,jpj)        ::   z0ice, z0snw, z0ai, z0oi
      REAL(wp), DIMENSION(jpi,jpj,nlay_s) ::   z0es
      REAL(wp), DIMENSION(jpi,jpj,nlay_i) ::   z0ei
      REAL(wp), ALLOCATABLE, DIMENSION(:,:)    ::   z0ap , z0vp, z0vl, zh_ip
      REAL(wp), ALLOCATABLE, DIMENSION(:,:)    ::   z0smi, zs_i 
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:)  ::   z0si , zsz_i
      !! diagnostics
      REAL(wp), DIMENSION(A2D(0))         ::   zdiag_adv_mass, zdiag_adv_salt, zdiag_adv_heat
      !!----------------------------------------------------------------------
      !
      IF( kt == nit000 .AND. lwp )   WRITE(numout,*) '-- ice_dyn_adv_pra: Prather advection scheme'
      CALL l4_dump_pra( kt, 0 )
      IF( kt <= nit000 + 4 ) CALL l4_orca1ice_pra_dump( kt, 0 )
      !
      IF( ln_pnd_LEV .OR. ln_pnd_TOPO ) ALLOCATE( z0ap(jpi,jpj), z0vp(jpi,jpj), z0vl(jpi,jpj), zh_ip(jpi,jpj) )
      IF( nn_icesal == 4 ) THEN   ;     ALLOCATE( z0si (jpi,jpj,nlay_i), zsz_i(jpi,jpj,nlay_i) )
      ELSE                        ;     ALLOCATE( z0smi(jpi,jpj)       , zs_i (jpi,jpj)        )
      ENDIF
      
      ! --- If ice drift is too fast, use  subtime steps for advection (CFL test for stability) --- !
      !        Note: the advection split is applied at the next time-step in order to avoid blocking global comm.
      !              this should not affect too much the stability
      zcfl =            MAXVAL( ABS( pu_ice(:,:) ) * rDt_ice * r1_e1u(:,:) )
      zcfl = MAX( zcfl, MAXVAL( ABS( pv_ice(:,:) ) * rDt_ice * r1_e2v(:,:) ) )

      CALL mpp_max( 'icedyn_adv_pra', zcfl, cdelay = 'cflice' )

      IF    ( zcfl > 1.5 ) THEN   ;   icycle = 3
      ELSEIF( zcfl >  .5 ) THEN   ;   icycle = 2
      ELSE                        ;   icycle = 1
      ENDIF
!!$      !!test clem
!!$      icycle=3
!!$      !!test clem      
      zdt = rDt_ice / REAL(icycle)
      z1_dt = 1._wp / zdt

      ! --- transport --- !
      zudy(:,:) = pu_ice(:,:) * e2u(:,:)
      zvdx(:,:) = pv_ice(:,:) * e1v(:,:)
     
      IF( lwp .AND. icycle > 1 )   WRITE(numout,*) 'icedyn_adv: CFL=',NINT(zcfl*10)/10.,' => number of cycles=',icycle
      !---------------!
      !== advection ==!
      !---------------!
      DO jt = 1, icycle
         
         ! record at_i before advection (for open water)
         zati1(:,:) = SUM( pa_i(A2D(0),:), dim=3 )

         IF( icycle == 1 ) THEN   ;   ihls = 0                   ! optimization
         ELSE                     ;   ihls = MAX( 0, nn_hls - jt )
         ENDIF
         !
         !                       ! =================== !
         !                       ! Start cat loop here !
         !                       ! =================== !
         DO jl = 1, jpl
            
            ! --- Record max of the surrounding 9-pts (for call Hbig) --- !
            ! thickness
            zh_i (:,:) = ph_i (:,:,jl)
            zh_s (:,:) = ph_s (:,:,jl)
            CALL icemax2D_pra( ihls, zh_i , zhi_max )
            CALL icemax2D_pra( ihls, zh_s , zhs_max )

            IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN
               zh_ip(:,:) = ph_ip(:,:,jl)
               CALL icemax2D_pra( ihls, zh_ip, zhip_max)
            ENDIF
            !
            ! enthalpies
            DO jk = 1, nlay_i
               WHERE( pv_i(:,:,jl) >= epsi10 ) ; ze_i(:,:,jk) = pe_i(:,:,jk,jl) / pv_i(:,:,jl)
               ELSEWHERE                       ; ze_i(:,:,jk) = 0._wp
               END WHERE
            END DO
            DO jk = 1, nlay_s
               WHERE( pv_s(:,:,jl) >= epsi10 ) ; ze_s(:,:,jk) = pe_s(:,:,jk,jl) / pv_s(:,:,jl)
               ELSEWHERE                       ; ze_s(:,:,jk) = 0._wp
               END WHERE
            END DO
            CALL icemax3D_pra( ihls, ze_i , zei_max )
            CALL icemax3D_pra( ihls, ze_s , zes_max )
            !
            ! salt content
            IF( nn_icesal == 4 ) THEN
               !
               DO jk = 1, nlay_i
                  WHERE( pv_i(:,:,jl) >= epsi10 ) ; zsz_i(:,:,jk) = pszv_i(:,:,jk,jl) / pv_i(:,:,jl)
                  ELSEWHERE                       ; zsz_i(:,:,jk) = 0._wp
                  END WHERE
               END DO
               CALL icemax3D_pra( ihls, zsz_i , zszi_max )
               !
            ELSE
               !
               WHERE( pv_i(:,:,jl) >= epsi10 ) ; zs_i(:,:) = psv_i(:,:,jl) / pv_i(:,:,jl)
               ELSEWHERE                       ; zs_i(:,:) = 0._wp
               END WHERE
               CALL icemax2D_pra( ihls, zs_i , zsi_max )
               !
            ENDIF
         
            ! diagnostics
            DO_2D( 0, 0, 0, 0 )
               zdiag_adv_mass(ji,jj) =   pv_i (ji,jj,jl) * rhoi + pv_s (ji,jj,jl) * rhos &
                  &                    + pv_ip(ji,jj,jl) * rhow + pv_il(ji,jj,jl) * rhow
               zdiag_adv_heat(ji,jj) = - SUM( pe_i(ji,jj,1:nlay_i,jl) ) - SUM( pe_s(ji,jj,1:nlay_s,jl) )
            END_2D
            IF( nn_icesal == 4 ) THEN
               DO_2D( 0, 0, 0, 0 )
                  zdiag_adv_salt(ji,jj) = SUM( pszv_i(ji,jj,1:nlay_i,jl) ) * rhoi
               END_2D
            ELSE
                DO_2D( 0, 0, 0, 0 )
                  zdiag_adv_salt(ji,jj) = psv_i(ji,jj,jl) * rhoi
               END_2D
            ENDIF
            !

            ! --- transported fields --- !
            DO_2D( ihls+1, ihls+1, ihls+1, ihls+1 )
               zarea(ji,jj) = e1e2t(ji,jj)
               z0snw(ji,jj) = pv_s (ji,jj,jl) * e1e2t(ji,jj)          ! Snow volume
               z0ice(ji,jj) = pv_i (ji,jj,jl) * e1e2t(ji,jj)          ! Ice  volume
               z0ai (ji,jj) = pa_i (ji,jj,jl) * e1e2t(ji,jj)          ! Ice area
               z0oi (ji,jj) = poa_i(ji,jj,jl) * e1e2t(ji,jj)          ! Age content
            END_2D
            DO_3D( ihls+1, ihls+1, ihls+1, ihls+1, 1, nlay_s )
               z0es(ji,jj,jk) = pe_s(ji,jj,jk,jl) * e1e2t(ji,jj)      ! Snow heat content
            END_3D
            DO_3D( ihls+1, ihls+1, ihls+1, ihls+1, 1, nlay_i )
               z0ei(ji,jj,jk) = pe_i  (ji,jj,jk,jl) * e1e2t(ji,jj)    ! Ice  heat content
            END_3D
            IF( nn_icesal == 4 ) THEN
               DO_3D( ihls+1, ihls+1, ihls+1, ihls+1, 1, nlay_i )
                  z0si(ji,jj,jk) = pszv_i(ji,jj,jk,jl) * e1e2t(ji,jj) ! Ice  salt content
               END_3D
            ELSE
               DO_2D( ihls+1, ihls+1, ihls+1, ihls+1 )
                  z0smi(ji,jj) = psv_i(ji,jj,jl) * e1e2t(ji,jj)
               END_2D
            ENDIF
            IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN
               DO_2D( ihls+1, ihls+1, ihls+1, ihls+1 )
                  z0ap(ji,jj) = pa_ip(ji,jj,jl) * e1e2t(ji,jj)        ! Melt pond fraction
                  z0vp(ji,jj) = pv_ip(ji,jj,jl) * e1e2t(ji,jj)        ! Melt pond volume
                  z0vl(ji,jj) = pv_il(ji,jj,jl) * e1e2t(ji,jj)        ! Melt pond lid volume
               END_2D
            ENDIF
            !
            ! ----------------------- !
            ! ==> start advection <== !
            ! ----------------------- !
            !                                                                  !--------------------------------------------!
            IF( MOD( (kt - 1) / nn_fsbc , 2 ) ==  MOD( (jt - 1) , 2 ) ) THEN   !==  odd ice time step:  adv_x then adv_y  ==!
               !                                                               !--------------------------------------------!
               ! ---------------------- !
               ! == mandatory fields == !
               ! ---------------------- !
               CALL adv_x( ihls, jl, zdt , zudy , 1._wp , zarea , z0ice , sxice , sxxice , syice , syyice , sxyice )    !--- ice volume
               CALL adv_y( ihls, jl, zdt , zvdx , 0._wp , zarea , z0ice , sxice , sxxice , syice , syyice , sxyice )
               CALL adv_x( ihls, jl, zdt , zudy , 1._wp , zarea , z0snw , sxsn  , sxxsn  , sysn  , syysn  , sxysn  )    !--- snow volume
               CALL adv_y( ihls, jl, zdt , zvdx , 0._wp , zarea , z0snw , sxsn  , sxxsn  , sysn  , syysn  , sxysn  )
               CALL adv_x( ihls, jl, zdt , zudy , 1._wp , zarea , z0ai  , sxa   , sxxa   , sya   , syya   , sxya   )    !--- ice concentration
               CALL adv_y( ihls, jl, zdt , zvdx , 0._wp , zarea , z0ai  , sxa   , sxxa   , sya   , syya   , sxya   )
               CALL adv_x( ihls, jl, zdt , zudy , 1._wp , zarea , z0oi  , sxage , sxxage , syage , syyage , sxyage )    !--- ice age
               CALL adv_y( ihls, jl, zdt , zvdx , 0._wp , zarea , z0oi  , sxage , sxxage , syage , syyage , sxyage )
               !
               DO jk = 1, nlay_s                                                                                        !--- snow heat content
                  CALL adv_x( ihls, jl, zdt, zudy, 1._wp, zarea, z0es (:,:,jk)  , sxc0(:,:,jk,:),   &
                     &                           sxxc0(:,:,jk,:), syc0(:,:,jk,:), syyc0(:,:,jk,:), sxyc0(:,:,jk,:) )
                  CALL adv_y( ihls, jl, zdt, zvdx, 0._wp, zarea, z0es (:,:,jk)  , sxc0(:,:,jk,:),   &
                     &                           sxxc0(:,:,jk,:), syc0(:,:,jk,:), syyc0(:,:,jk,:), sxyc0(:,:,jk,:) )
               END DO
               DO jk = 1, nlay_i                                                                                        !--- ice heat content
                  CALL adv_x( ihls, jl, zdt, zudy, 1._wp, zarea, z0ei(:,:,jk)  , sxe(:,:,jk,:),   &
                     &                               sxxe(:,:,jk,:), sye(:,:,jk,:), syye(:,:,jk,:), sxye(:,:,jk,:) )
                  CALL adv_y( ihls, jl, zdt, zvdx, 0._wp, zarea, z0ei(:,:,jk)  , sxe(:,:,jk,:),   &
                     &                               sxxe(:,:,jk,:), sye(:,:,jk,:), syye(:,:,jk,:), sxye(:,:,jk,:) )
               END DO
               ! --------------------- !
               ! == optional fields == !
               ! --------------------- !
               IF( nn_icesal == 4 ) THEN
                  DO jk = 1, nlay_i                                                                                     !--- ice salt content
                     CALL adv_x( ihls, jl, zdt, zudy, 1._wp, zarea, z0si(:,:,jk)  , sxsi (:,:,jk,:),   &
                        &                          sxxsi(:,:,jk,:), sysi(:,:,jk,:), syysi(:,:,jk,:), sxysi(:,:,jk,:) )
                     CALL adv_y( ihls, jl, zdt, zvdx, 0._wp, zarea, z0si(:,:,jk)  , sxsi (:,:,jk,:),   &
                        &                          sxxsi(:,:,jk,:), sysi(:,:,jk,:), syysi(:,:,jk,:), sxysi(:,:,jk,:) )
                  END DO
               ELSE
                  CALL adv_x( ihls, jl, zdt , zudy , 1._wp , zarea , z0smi , sxsal , sxxsal , sysal , syysal , sxysal ) !--- ice salinity
                  CALL adv_y( ihls, jl, zdt , zvdx , 0._wp , zarea , z0smi , sxsal , sxxsal , sysal , syysal , sxysal )
               ENDIF
               !
               IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN
                  CALL adv_x( ihls, jl, zdt , zudy , 1._wp , zarea , z0ap , sxap , sxxap , syap , syyap , sxyap )       !--- melt pond fraction
                  CALL adv_y( ihls, jl, zdt , zvdx , 0._wp , zarea , z0ap , sxap , sxxap , syap , syyap , sxyap )
                  CALL adv_x( ihls, jl, zdt , zudy , 1._wp , zarea , z0vp , sxvp , sxxvp , syvp , syyvp , sxyvp )       !--- melt pond volume
                  CALL adv_y( ihls, jl, zdt , zvdx , 0._wp , zarea , z0vp , sxvp , sxxvp , syvp , syyvp , sxyvp )
                  CALL adv_x( ihls, jl, zdt , zudy , 1._wp , zarea , z0vl , sxvl , sxxvl , syvl , syyvl , sxyvl )       !--- melt pond lid volume
                  CALL adv_y( ihls, jl, zdt , zvdx , 0._wp , zarea , z0vl , sxvl , sxxvl , syvl , syyvl , sxyvl )
               ENDIF
               !                                                               !--------------------------------------------!
            ELSE                                                               !== even ice time step:  adv_y then adv_x  ==!
               !                                                               !--------------------------------------------!
               ! ---------------------- !
               ! == mandatory fields == !
               ! ---------------------- !
               CALL adv_y( ihls, jl, zdt , zvdx , 1._wp , zarea , z0ice , sxice , sxxice , syice , syyice , sxyice )    !--- ice volume
               CALL adv_x( ihls, jl, zdt , zudy , 0._wp , zarea , z0ice , sxice , sxxice , syice , syyice , sxyice )
               CALL adv_y( ihls, jl, zdt , zvdx , 1._wp , zarea , z0snw , sxsn  , sxxsn  , sysn  , syysn  , sxysn  )    !--- snow volume
               CALL adv_x( ihls, jl, zdt , zudy , 0._wp , zarea , z0snw , sxsn  , sxxsn  , sysn  , syysn  , sxysn  )
               CALL adv_y( ihls, jl, zdt , zvdx , 1._wp , zarea , z0ai  , sxa   , sxxa   , sya   , syya   , sxya   )    !--- ice concentration
               CALL adv_x( ihls, jl, zdt , zudy , 0._wp , zarea , z0ai  , sxa   , sxxa   , sya   , syya   , sxya   )
               CALL adv_y( ihls, jl, zdt , zvdx , 1._wp , zarea , z0oi  , sxage , sxxage , syage , syyage , sxyage )    !--- ice age
               CALL adv_x( ihls, jl, zdt , zudy , 0._wp , zarea , z0oi  , sxage , sxxage , syage , syyage , sxyage )
               !
               DO jk = 1, nlay_s                                                                                        !--- snow heat content
                  CALL adv_y( ihls, jl, zdt, zvdx, 1._wp, zarea, z0es (:,:,jk)  , sxc0(:,:,jk,:),   &
                     &                           sxxc0(:,:,jk,:), syc0(:,:,jk,:), syyc0(:,:,jk,:), sxyc0(:,:,jk,:) )
                  CALL adv_x( ihls, jl, zdt, zudy, 0._wp, zarea, z0es (:,:,jk)  , sxc0(:,:,jk,:),   &
                     &                           sxxc0(:,:,jk,:), syc0(:,:,jk,:), syyc0(:,:,jk,:), sxyc0(:,:,jk,:) )
               END DO
               DO jk = 1, nlay_i                                                                                        !--- ice heat content
                  CALL adv_y( ihls, jl, zdt, zvdx, 1._wp, zarea, z0ei(:,:,jk)  , sxe(:,:,jk,:),   &
                     &                               sxxe(:,:,jk,:), sye(:,:,jk,:), syye(:,:,jk,:), sxye(:,:,jk,:) )
                  CALL adv_x( ihls, jl, zdt, zudy, 0._wp, zarea, z0ei(:,:,jk)  , sxe(:,:,jk,:),   &
                     &                               sxxe(:,:,jk,:), sye(:,:,jk,:), syye(:,:,jk,:), sxye(:,:,jk,:) )
               END DO
               ! --------------------- !
               ! == optional fields == !
               ! --------------------- !
               IF( nn_icesal == 4 ) THEN
                  DO jk = 1, nlay_i                                                                                     !--- ice salt content
                     CALL adv_y( ihls, jl, zdt, zvdx, 1._wp, zarea, z0si(:,:,jk)  , sxsi (:,:,jk,:),   &
                        &                          sxxsi(:,:,jk,:), sysi(:,:,jk,:), syysi(:,:,jk,:), sxysi(:,:,jk,:) )
                     CALL adv_x( ihls, jl, zdt, zudy, 0._wp, zarea, z0si(:,:,jk)  , sxsi (:,:,jk,:),   &
                        &                          sxxsi(:,:,jk,:), sysi(:,:,jk,:), syysi(:,:,jk,:), sxysi(:,:,jk,:) )
                  END DO
               ELSE
                  CALL adv_y( ihls, jl, zdt , zvdx , 1._wp , zarea , z0smi , sxsal , sxxsal , sysal , syysal , sxysal ) !--- ice salinity
                  CALL adv_x( ihls, jl, zdt , zudy , 0._wp , zarea , z0smi , sxsal , sxxsal , sysal , syysal , sxysal )
               ENDIF
               !
               IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN
                  CALL adv_y( ihls, jl, zdt , zvdx , 1._wp , zarea , z0ap , sxap , sxxap , syap , syyap , sxyap )       !--- melt pond fraction
                  CALL adv_x( ihls, jl, zdt , zudy , 0._wp , zarea , z0ap , sxap , sxxap , syap , syyap , sxyap )
                  CALL adv_y( ihls, jl, zdt , zvdx , 1._wp , zarea , z0vp , sxvp , sxxvp , syvp , syyvp , sxyvp )       !--- melt pond volume
                  CALL adv_x( ihls, jl, zdt , zudy , 0._wp , zarea , z0vp , sxvp , sxxvp , syvp , syyvp , sxyvp )
                  CALL adv_y( ihls, jl, zdt , zvdx , 1._wp , zarea , z0vl , sxvl , sxxvl , syvl , syyvl , sxyvl )       !--- melt pond lid volume
                  CALL adv_x( ihls, jl, zdt , zudy , 0._wp , zarea , z0vl , sxvl , sxxvl , syvl , syyvl , sxyvl )
               ENDIF
               !
            ENDIF

            ! --- Recover the properties from their contents --- !
            DO_2D( ihls, ihls, ihls, ihls )
               pv_i (ji,jj,jl) = z0ice(ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
               pv_s (ji,jj,jl) = z0snw(ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
               poa_i(ji,jj,jl) = z0oi (ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
               pa_i (ji,jj,jl) = z0ai (ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
            END_2D
            DO_3D( ihls, ihls, ihls, ihls, 1, nlay_s )
               pe_s(ji,jj,jk,jl) = z0es(ji,jj,jk) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
            END_3D
            DO_3D( ihls, ihls, ihls, ihls, 1, nlay_i )
               pe_i(ji,jj,jk,jl) = z0ei(ji,jj,jk) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
            END_3D
            IF( nn_icesal == 4 ) THEN
               DO_3D( ihls, ihls, ihls, ihls, 1, nlay_i )
                  pszv_i(ji,jj,jk,jl) = z0si(ji,jj,jk) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
               END_3D
            ELSE
               DO_2D( ihls, ihls, ihls, ihls )
                  psv_i(ji,jj,jl) = z0smi(ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
               END_2D
            ENDIF
            IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN
               DO_2D( ihls, ihls, ihls, ihls )
                  pa_ip(ji,jj,jl) = z0ap(ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
                  pv_ip(ji,jj,jl) = z0vp(ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
                  pv_il(ji,jj,jl) = z0vl(ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
               END_2D
            ENDIF
                        
            ! --- diagnostics --- !
            DO_2D( 0, 0, 0, 0 )
               diag_adv_mass(ji,jj) = diag_adv_mass(ji,jj) + ( pv_i (ji,jj,jl) * rhoi + pv_s (ji,jj,jl) * rhos &
                  &                                        +   pv_ip(ji,jj,jl) * rhow + pv_il(ji,jj,jl) * rhow &
                  &                                          - zdiag_adv_mass(ji,jj) ) * r1_Dt_ice
               diag_adv_heat(ji,jj) = diag_adv_heat(ji,jj) + ( -SUM( pe_i(ji,jj,1:nlay_i,jl) ) -SUM( pe_s(ji,jj,1:nlay_s,jl) ) &
                  &                                          - zdiag_adv_heat(ji,jj) ) * r1_Dt_ice
            END_2D
            IF( nn_icesal == 4 ) THEN
               DO_2D( 0, 0, 0, 0 )
                  diag_adv_salt(ji,jj) = diag_adv_salt(ji,jj) + ( SUM( pszv_i(ji,jj,1:nlay_i,jl) ) * rhoi &
                     &                                          - zdiag_adv_salt(ji,jj) ) * r1_Dt_ice
               END_2D
            ELSE
                DO_2D( 0, 0, 0, 0 )
                  diag_adv_salt(ji,jj) = diag_adv_salt(ji,jj) + ( psv_i(ji,jj,jl) * rhoi &
                     &                                          - zdiag_adv_salt(ji,jj) ) * r1_Dt_ice
               END_2D
            ENDIF

            ! --- Make sure ice thickness is not too big --- !
            !     (because ice thickness can be too large where ice concentration is very small)
            CALL Hbig_pra( ihls, jl, rDt_ice, zhi_max, zhs_max, zhip_max, zsi_max, zes_max, zei_max, zszi_max, &
               &            pv_i, pv_s, pa_i, pa_ip, pv_ip, psv_i, pe_s, pe_i, pszv_i )
            !
            ! --- Ensure snow load is not too big --- !
            CALL Hsnow_pra( ihls, jl, rDt_ice, pv_i, pv_s, pa_i, pa_ip, pe_s )
            !
         END DO
         !                       ! ================= !
         !                       ! End cat loop here !
         !                       ! ================= !
         !
         ! --- Ensure non-negative fields --- !
         !     Remove negative values (conservation is ensured)
         !     (because advected fields are not perfectly bounded and tiny negative values can occur, e.g. -1.e-20)
         CALL ice_var_zapneg( ihls, rDt_ice, pv_i, pv_s, psv_i, poa_i, pa_i, pa_ip, pv_ip, pv_il, pe_s, pe_i, pszv_i )
         !
         ! derive open water from ice concentration
         DO_2D( 0, 0, 0, 0 )
            zati2 = SUM( pa_i(ji,jj,:) )
            pato_i(ji,jj) = MAX( 0._wp, pato_i(ji,jj) - ( zati2 - zati1(ji,jj) )            &
               &                                      - (   ( zudy(ji,jj) - zudy(ji-1,jj) ) &   ! ad () for NP repro
               &                                          + ( zvdx(ji,jj) - zvdx(ji,jj-1) ) ) * r1_e1e2t(ji,jj) * zdt )
         END_2D
         ! note: no need of lbc_lnk for open water (never used in the halos)
         !
         ! --- Lateral boundary conditions --- !
         !     caution: for gradients (sx and sy) the sign changes
         !              plus, one needs ldfull=T to deal with the NorthFold
            !
            IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN
               CALL lbc_lnk( 'icedyn_adv_pra', pv_i  , 'T', 1._wp, sxice , 'T', -1._wp, syice , 'T', -1._wp  & ! ice volume
                  &                          , sxxice, 'T', 1._wp, syyice, 'T',  1._wp, sxyice, 'T',  1._wp  &
                  &                          , pv_s  , 'T', 1._wp, sxsn  , 'T', -1._wp, sysn  , 'T', -1._wp  & ! snw volume
                  &                          , sxxsn , 'T', 1._wp, syysn , 'T',  1._wp, sxysn , 'T',  1._wp  &
                  &                          , psv_i , 'T', 1._wp, sxsal , 'T', -1._wp, sysal , 'T', -1._wp  & ! ice salinity
                  &                          , sxxsal, 'T', 1._wp, syysal, 'T',  1._wp, sxysal, 'T',  1._wp  &
                  &                          , pa_i  , 'T', 1._wp, sxa   , 'T', -1._wp, sya   , 'T', -1._wp  & ! ice concentration
                  &                          , sxxa  , 'T', 1._wp, syya  , 'T',  1._wp, sxya  , 'T',  1._wp  &
                  &                          , poa_i , 'T', 1._wp, sxage , 'T', -1._wp, syage , 'T', -1._wp  & ! ice age
                  &                          , sxxage, 'T', 1._wp, syyage, 'T',  1._wp, sxyage, 'T',  1._wp  &
                  &                          , pa_ip , 'T', 1._wp, sxap  , 'T', -1._wp, syap  , 'T', -1._wp  & ! melt pond fraction
                  &                          , sxxap , 'T', 1._wp, syyap , 'T',  1._wp, sxyap , 'T',  1._wp  &
                  &                          , pv_ip , 'T', 1._wp, sxvp  , 'T', -1._wp, syvp  , 'T', -1._wp  & ! melt pond volume
                  &                          , sxxvp , 'T', 1._wp, syyvp , 'T',  1._wp, sxyvp , 'T',  1._wp  &
                  &                          , pv_il , 'T', 1._wp, sxvl  , 'T', -1._wp, syvl  , 'T', -1._wp  & ! melt pond lid volume
                  &                          , sxxvl , 'T', 1._wp, syyvl , 'T',  1._wp, sxyvl,  'T',  1._wp, ldfull = .TRUE. )
            ELSE
               CALL lbc_lnk( 'icedyn_adv_pra', pv_i  , 'T', 1._wp, sxice , 'T', -1._wp, syice , 'T', -1._wp  & ! ice volume
                  &                          , sxxice, 'T', 1._wp, syyice, 'T',  1._wp, sxyice, 'T',  1._wp  &
                  &                          , pv_s  , 'T', 1._wp, sxsn  , 'T', -1._wp, sysn  , 'T', -1._wp  & ! snw volume
                  &                          , sxxsn , 'T', 1._wp, syysn , 'T',  1._wp, sxysn , 'T',  1._wp  &
                  &                          , psv_i , 'T', 1._wp, sxsal , 'T', -1._wp, sysal , 'T', -1._wp  & ! ice salinity
                  &                          , sxxsal, 'T', 1._wp, syysal, 'T',  1._wp, sxysal, 'T',  1._wp  &
                  &                          , pa_i  , 'T', 1._wp, sxa   , 'T', -1._wp, sya   , 'T', -1._wp  & ! ice concentration
                  &                          , sxxa  , 'T', 1._wp, syya  , 'T',  1._wp, sxya  , 'T',  1._wp  &
                  &                          , poa_i , 'T', 1._wp, sxage , 'T', -1._wp, syage , 'T', -1._wp  & ! ice age
                  &                          , sxxage, 'T', 1._wp, syyage, 'T',  1._wp, sxyage, 'T',  1._wp, ldfull = .TRUE. )
            ENDIF
            IF( nn_icesal == 4 ) THEN
               CALL lbc_lnk( 'icedyn_adv_pra', pe_s  , 'T', 1._wp, sxc0  , 'T', -1._wp, syc0  , 'T', -1._wp  & ! snw enthalpy
                  &                          , sxxc0 , 'T', 1._wp, syyc0 , 'T',  1._wp, sxyc0 , 'T',  1._wp  &
                  &                          , pe_i  , 'T', 1._wp, sxe   , 'T', -1._wp, sye   , 'T', -1._wp  & ! ice enthalpy
                  &                          , sxxe  , 'T', 1._wp, syye  , 'T',  1._wp, sxye  , 'T',  1._wp  &
                  &                          , pszv_i, 'T', 1._wp, sxsi  , 'T', -1._wp, sysi  , 'T', -1._wp  & ! ice salt content
                  &                          , sxxsi , 'T', 1._wp, syysi , 'T',  1._wp, sxysi , 'T',  1._wp, ldfull = .TRUE. )
            ELSE
               CALL lbc_lnk( 'icedyn_adv_pra', pe_s  , 'T', 1._wp, sxc0  , 'T', -1._wp, syc0  , 'T', -1._wp  & ! snw enthalpy
                  &                          , sxxc0 , 'T', 1._wp, syyc0 , 'T',  1._wp, sxyc0 , 'T',  1._wp  &
                  &                          , pe_i  , 'T', 1._wp, sxe   , 'T', -1._wp, sye   , 'T', -1._wp  & ! ice enthalpy
                  &                          , sxxe  , 'T', 1._wp, syye  , 'T',  1._wp, sxye  , 'T',  1._wp, ldfull = .TRUE. )
            ENDIF

            CALL lbc_lnk( 'icedyn_adv_pra', pato_i  , 'T', 1._wp, ldfull = .TRUE. )
            CALL lbc_lnk( 'icedyn_adv_pra', wfx_res  , 'T', 1._wp, hfx_res  , 'T', 1._wp, sfx_res  , 'T', 1._wp, ldfull = .TRUE. )

            !
         !
      END DO ! jt
      !
      !
      CALL l4_dump_pra( kt, 1 )                          ! Lane-4 registered EXIT moments
      IF( kt <= nit000 + 4 ) CALL l4_orca1ice_pra_dump( kt, 1 )
      IF( lrst_ice )   CALL adv_pra_rst( 'WRITE', kt )   !* write Prather fields in the restart file
      !
      !
      ! --- Deallocate arrays --- !
      IF( ln_pnd_LEV .OR. ln_pnd_TOPO ) DEALLOCATE( z0ap , z0vp, z0vl, zh_ip )
      IF( nn_icesal == 4 ) THEN   ;     DEALLOCATE( z0si , zsz_i )
      ELSE                        ;     DEALLOCATE( z0smi, zs_i  )
      ENDIF
      !
   END SUBROUTINE ice_dyn_adv_pra


   SUBROUTINE l4_orca1ice_pra_dump( kt, kframe )
      !! WRITE-only self-describing Prather moments for the resolved ORCA1
      !! target (nn_icesal=2, ponds off).  Model arrays are never assigned.
      INTEGER, INTENT(in) :: kt, kframe
      INTEGER :: l4_unit, l4_ios, l4_payload, jf
      INTEGER, PARAMETER :: l4_nf = 35
      INTEGER, DIMENSION(4,l4_nf) :: l4_ext
      CHARACTER(LEN=16), PARAMETER :: l4_magic = 'NEMO_L4_O1IPRA1'
      CHARACTER(LEN=96) :: l4_name
      IF( .NOT.lwp ) RETURN
      IF( nn_icesal /= 2 .OR. ln_pnd_LEV .OR. ln_pnd_TOPO ) &
         & CALL ctl_stop('l4_orca1ice_pra_dump: selector contract changed')

      l4_ext(:, 1) = (/ SHAPE(sxice), 1 /)
      l4_ext(:, 2) = (/ SHAPE(syice), 1 /)
      l4_ext(:, 3) = (/ SHAPE(sxxice), 1 /)
      l4_ext(:, 4) = (/ SHAPE(syyice), 1 /)
      l4_ext(:, 5) = (/ SHAPE(sxyice), 1 /)
      l4_ext(:, 6) = (/ SHAPE(sxsn), 1 /)
      l4_ext(:, 7) = (/ SHAPE(sysn), 1 /)
      l4_ext(:, 8) = (/ SHAPE(sxxsn), 1 /)
      l4_ext(:, 9) = (/ SHAPE(syysn), 1 /)
      l4_ext(:,10) = (/ SHAPE(sxysn), 1 /)
      l4_ext(:,11) = (/ SHAPE(sxa), 1 /)
      l4_ext(:,12) = (/ SHAPE(sya), 1 /)
      l4_ext(:,13) = (/ SHAPE(sxxa), 1 /)
      l4_ext(:,14) = (/ SHAPE(syya), 1 /)
      l4_ext(:,15) = (/ SHAPE(sxya), 1 /)
      l4_ext(:,16) = (/ SHAPE(sxage), 1 /)
      l4_ext(:,17) = (/ SHAPE(syage), 1 /)
      l4_ext(:,18) = (/ SHAPE(sxxage), 1 /)
      l4_ext(:,19) = (/ SHAPE(syyage), 1 /)
      l4_ext(:,20) = (/ SHAPE(sxyage), 1 /)
      l4_ext(:,21) = SHAPE(sxc0)
      l4_ext(:,22) = SHAPE(syc0)
      l4_ext(:,23) = SHAPE(sxxc0)
      l4_ext(:,24) = SHAPE(syyc0)
      l4_ext(:,25) = SHAPE(sxyc0)
      l4_ext(:,26) = SHAPE(sxe)
      l4_ext(:,27) = SHAPE(sye)
      l4_ext(:,28) = SHAPE(sxxe)
      l4_ext(:,29) = SHAPE(syye)
      l4_ext(:,30) = SHAPE(sxye)
      l4_ext(:,31) = (/ SHAPE(sxsal), 1 /)
      l4_ext(:,32) = (/ SHAPE(sysal), 1 /)
      l4_ext(:,33) = (/ SHAPE(sxxsal), 1 /)
      l4_ext(:,34) = (/ SHAPE(syysal), 1 /)
      l4_ext(:,35) = (/ SHAPE(sxysal), 1 /)
      l4_payload = 0
      DO jf = 1, l4_nf
         l4_payload = l4_payload + PRODUCT(l4_ext(:,jf))
      END DO

      WRITE(l4_name,'("oracle_orca1ice_prather_kt",I8.8,"_f",I1,".bin")') kt, kframe
      OPEN(NEWUNIT=l4_unit, FILE=TRIM(l4_name), STATUS='REPLACE', ACCESS='STREAM', &
         & FORM='UNFORMATTED', ACTION='WRITE', IOSTAT=l4_ios)
      IF( l4_ios /= 0 ) CALL ctl_stop('l4_orca1ice_pra_dump: open failed')
      WRITE(l4_unit) l4_magic
      WRITE(l4_unit) 1, kt, kframe, jpi, jpj, jpl, nlay_i, nlay_s, &
         & STORAGE_SIZE(1._wp), l4_nf, l4_payload
      WRITE(l4_unit) l4_ext
      WRITE(l4_unit) l4_canon_3d(sxice,'T'), l4_canon_3d(syice,'T'), &
         & l4_canon_3d(sxxice,'T'), l4_canon_3d(syyice,'T'), l4_canon_3d(sxyice,'T'), &
         & l4_canon_3d(sxsn,'T'), l4_canon_3d(sysn,'T'), l4_canon_3d(sxxsn,'T'), &
         & l4_canon_3d(syysn,'T'), l4_canon_3d(sxysn,'T'), l4_canon_3d(sxa,'T'), &
         & l4_canon_3d(sya,'T'), l4_canon_3d(sxxa,'T'), l4_canon_3d(syya,'T'), &
         & l4_canon_3d(sxya,'T'), l4_canon_3d(sxage,'T'), l4_canon_3d(syage,'T'), &
         & l4_canon_3d(sxxage,'T'), l4_canon_3d(syyage,'T'), l4_canon_3d(sxyage,'T'), &
         & l4_canon_4d(sxc0,'T'), l4_canon_4d(syc0,'T'), l4_canon_4d(sxxc0,'T'), &
         & l4_canon_4d(syyc0,'T'), l4_canon_4d(sxyc0,'T'), l4_canon_4d(sxe,'T'), &
         & l4_canon_4d(sye,'T'), l4_canon_4d(sxxe,'T'), l4_canon_4d(syye,'T'), &
         & l4_canon_4d(sxye,'T'), l4_canon_3d(sxsal,'T'), l4_canon_3d(sysal,'T'), &
         & l4_canon_3d(sxxsal,'T'), l4_canon_3d(syysal,'T'), l4_canon_3d(sxysal,'T')
      CLOSE(l4_unit)
   END SUBROUTINE l4_orca1ice_pra_dump


   SUBROUTINE l4_dump_pra( kt, kstage )
      !! Lane-4 read-only complete Prather-moment frame. Optional families are
      !! guarded by the same selectors that own their allocation.
      INTEGER, INTENT(in) :: kt, kstage
      INTEGER :: iunit, ios
      CHARACTER(LEN=96) :: filename
      CHARACTER(LEN=16), PARAMETER :: cmagic = 'NEMO_L4_PRA_001'
      IF( .NOT.lwp ) RETURN
      IF( STORAGE_SIZE(1._wp) /= 64 ) CALL ctl_stop( 'lane4 Prather dump requires 64-bit wp' )
      WRITE(filename,'("oracle_si3_prather_kt",I8.8,"_s",I1,".bin")') kt, kstage
      OPEN( NEWUNIT=iunit, FILE=TRIM(filename), ACCESS='STREAM', FORM='UNFORMATTED', &
         &  STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 ) CALL ctl_stop( 'cannot open lane4 Prather frame' )
      WRITE(iunit) cmagic, 1, kt, kstage, jpi, jpj, jpl, nlay_i, nlay_s, nn_icesal, &
         & MERGE(1,0,ln_pnd_LEV .OR. ln_pnd_TOPO), STORAGE_SIZE(1._wp)
      WRITE(iunit) l4_canon_3d(sxice,'T'), l4_canon_3d(syice,'T'), &
         & l4_canon_3d(sxxice,'T'), l4_canon_3d(syyice,'T'), l4_canon_3d(sxyice,'T'), &
         & l4_canon_3d(sxsn,'T'), l4_canon_3d(sysn,'T'), l4_canon_3d(sxxsn,'T'), &
         & l4_canon_3d(syysn,'T'), l4_canon_3d(sxysn,'T')
      WRITE(iunit) l4_canon_3d(sxa,'T'), l4_canon_3d(sya,'T'), l4_canon_3d(sxxa,'T'), &
         & l4_canon_3d(syya,'T'), l4_canon_3d(sxya,'T'), l4_canon_3d(sxage,'T'), &
         & l4_canon_3d(syage,'T'), l4_canon_3d(sxxage,'T'), l4_canon_3d(syyage,'T'), &
         & l4_canon_3d(sxyage,'T')
      WRITE(iunit) l4_canon_4d(sxc0,'T'), l4_canon_4d(syc0,'T'), &
         & l4_canon_4d(sxxc0,'T'), l4_canon_4d(syyc0,'T'), l4_canon_4d(sxyc0,'T'), &
         & l4_canon_4d(sxe,'T'), l4_canon_4d(sye,'T'), l4_canon_4d(sxxe,'T'), &
         & l4_canon_4d(syye,'T'), l4_canon_4d(sxye,'T')
      WRITE(iunit) l4_canon_3d(sxsal,'T'), l4_canon_3d(sysal,'T'), &
         & l4_canon_3d(sxxsal,'T'), l4_canon_3d(syysal,'T'), l4_canon_3d(sxysal,'T')
      IF( nn_icesal == 4 ) WRITE(iunit) l4_canon_4d(sxsi,'T'), l4_canon_4d(sysi,'T'), &
         & l4_canon_4d(sxxsi,'T'), l4_canon_4d(syysi,'T'), l4_canon_4d(sxysi,'T')
      IF( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN
         WRITE(iunit) l4_canon_3d(sxap,'T'), l4_canon_3d(syap,'T'), &
            & l4_canon_3d(sxxap,'T'), l4_canon_3d(syyap,'T'), l4_canon_3d(sxyap,'T'), &
            & l4_canon_3d(sxvp,'T'), l4_canon_3d(syvp,'T'), l4_canon_3d(sxxvp,'T'), &
            & l4_canon_3d(syyvp,'T'), l4_canon_3d(sxyvp,'T')
         WRITE(iunit) l4_canon_3d(sxvl,'T'), l4_canon_3d(syvl,'T'), &
            & l4_canon_3d(sxxvl,'T'), l4_canon_3d(syyvl,'T'), l4_canon_3d(sxyvl,'T')
      ENDIF
      CLOSE(iunit)
      WRITE(numout,*) 'LANE4_PRATHER_DUMP ', kt, kstage, TRIM(filename)
   END SUBROUTINE l4_dump_pra


   SUBROUTINE adv_x( ihls, jcat, pdt, put , pcrh, psm , ps0 ,   &
      &              psx, psxx, psy , psyy, psxy )
      !!----------------------------------------------------------------------
      !!                **  routine adv_x  **
      !!
      !! ** purpose :   Computes and adds the advection trend to sea-ice
      !!                variable on x axis
      !!----------------------------------------------------------------------
      INTEGER                   , INTENT(in   ) ::   ihls               ! loop index
      INTEGER                   , INTENT(in   ) ::   jcat               ! category
      REAL(wp)                  , INTENT(in   ) ::   pdt                ! time step
      REAL(wp)                  , INTENT(in   ) ::   pcrh               ! call adv_x then adv_y (=1) or the opposite (=0)
      REAL(wp), DIMENSION(:,:)  , INTENT(in   ) ::   put                ! i-direction ice velocity at U-point [m/s]
      REAL(wp), DIMENSION(:,:)  , INTENT(inout) ::   psm                ! area
      REAL(wp), DIMENSION(:,:)  , INTENT(inout) ::   ps0                ! field to be advected
      REAL(wp), DIMENSION(:,:,:), INTENT(inout) ::   psx , psy          ! 1st moments
      REAL(wp), DIMENSION(:,:,:), INTENT(inout) ::   psxx, psyy, psxy   ! 2nd moments
      !!
      INTEGER  ::   ji, jj                               ! dummy loop indices
      INTEGER  ::   ji0, jj0                             ! dummy loop indices
      REAL(wp) ::   z1_3
      REAL(wp) ::   zs1max, zslpmax                      ! local scalars
      REAL(wp) ::   zs1new, zalf , zalf2, zalf3          !   -      -
      REAL(wp) ::   zs2new, z1malf, z1malf2, z1malf3     !   -      -
      REAL(wp) ::   zpsm, zps0
      REAL(wp) ::   zpsx, zpsy, zpsxx, zpsyy, zpsxy
      REAL(wp), DIMENSION(jpi,jpj) ::   zf0 , zfx  , zfy            ! 2D workspace
      REAL(wp), DIMENSION(jpi,jpj) ::   zfm , zfxx , zfyy  , zfxy   !  -      -
      !-----------------------------------------------------------------------
      ! in order to avoid lbc_lnk (communications):
      !    jj loop must be 1:jpj   if adv_x is called first
      !                and 2:jpj-1 if adv_x is called second
      ji0 = 1 + ihls
      jj0 = NINT(pcrh) + ihls
      !
      z1_3 = 1._wp / 3._wp
      ! Limitation of moments.
      DO_2D( ji0, ji0, jj0, jj0 )
         !
         zpsm  = psm (ji,jj) ! optimization
         zps0  = ps0 (ji,jj)
         zpsx  = psx (ji,jj,jcat)
         zpsxx = psxx(ji,jj,jcat)
         zpsy  = psy (ji,jj,jcat)
         zpsyy = psyy(ji,jj,jcat)
         zpsxy = psxy(ji,jj,jcat)

         !  Initialize volumes of boxes (=area if adv_x first called, =psm otherwise)
         zpsm = MAX( pcrh * e1e2t(ji,jj) + ( 1._wp - pcrh ) * zpsm , epsi20 )
         !
         zslpmax = MAX( 0._wp, zps0 )
         zps0    = zslpmax
         !
         IF( zslpmax > 0._wp ) THEN
            zs1max = 1.5_wp * zslpmax
            zs1new = MIN( zs1max, MAX( -zs1max, zpsx ) )
            zs2new = MIN( 2._wp * zslpmax - z1_3 * ABS( zs1new ), MAX( ABS( zs1new ) - zslpmax, zpsxx ) )
            !
            zpsx  = zs1new  * tmask(ji,jj,1)
            zpsxx = zs2new  * tmask(ji,jj,1)
            zpsy  = zpsy    * tmask(ji,jj,1)
            zpsyy = zpsyy   * tmask(ji,jj,1)
            zpsxy = MIN( zslpmax, MAX( -zslpmax, zpsxy ) ) * tmask(ji,jj,1)
         ELSE
            zpsx  = 0._wp
            zpsxx = 0._wp
            zpsy  = 0._wp
            zpsyy = 0._wp
            zpsxy = 0._wp
         ENDIF
         !
         !  Calculate fluxes and moments between boxes i<-->i+1
         !                                !  Flux from i to i+1 WHEN u GT 0
         IF( put(ji,jj) >= 0._wp ) THEN
            !
            zalf    = put(ji,jj) * pdt / zpsm
            z1malf  = 1._wp - zalf
            !
            zalf2   =   zalf  *   zalf
            zalf3   =   zalf2 *   zalf
            z1malf2 = z1malf  * z1malf
            z1malf3 = z1malf2 * z1malf
            !
            zfm (ji,jj) = zalf  *   zpsm
            zf0 (ji,jj) = zalf  * ( zps0  +         z1malf * ( zpsx + (z1malf - zalf) * zpsxx ) )
            zfx (ji,jj) = zalf2 * ( zpsx  + 3._wp * z1malf *   zpsxx )
            zfxx(ji,jj) = zalf3 *   zpsxx
            zfy (ji,jj) = zalf  * ( zpsy  +         z1malf *   zpsxy )
            zfyy(ji,jj) = zalf  *   zpsyy
            zfxy(ji,jj) = zalf2 *   zpsxy
            !
            !                                !  Readjust moments remaining in the box.
            zpsm  = zpsm    -   zfm (ji,jj)
            zps0  = zps0    -   zf0 (ji,jj)
            zpsx  = z1malf2 * ( zpsx - 3._wp * zalf * zpsxx )
            zpsxx = z1malf3 *   zpsxx
            zpsy  = zpsy    -   zfy (ji,jj)
            zpsyy = zpsyy   -   zfyy(ji,jj)
            zpsxy = z1malf2 *   zpsxy
         ELSE
            zfm (ji,jj) = 0._wp
            zf0 (ji,jj) = 0._wp
            zfx (ji,jj) = 0._wp
            zfxx(ji,jj) = 0._wp
            zfy (ji,jj) = 0._wp
            zfyy(ji,jj) = 0._wp
            zfxy(ji,jj) = 0._wp
         ENDIF
         !
         psm (ji,jj) = zpsm ! optimization
         ps0 (ji,jj) = zps0
         psx (ji,jj,jcat) = zpsx
         psxx(ji,jj,jcat) = zpsxx
         psy (ji,jj,jcat) = zpsy
         psyy(ji,jj,jcat) = zpsyy
         psxy(ji,jj,jcat) = zpsxy
         !
      END_2D

      DO_2D( ji0, ji0-1, jj0, jj0 )
         !                                !  Flux from i+1 to i when u LT 0.
         IF( put(ji,jj) < 0._wp ) THEN
            !
            zalf   = - put(ji,jj) * pdt / psm(ji+1,jj)
            z1malf =   1._wp - zalf
            !
            zalf2  = zalf  * zalf
            zalf3  = zalf2 * zalf
            
            zfm (ji,jj) = zfm (ji,jj) + zalf  *   psm (ji+1,jj)
            zf0 (ji,jj) = zf0 (ji,jj) + zalf  * ( ps0 (ji+1,jj) &
               &                                - z1malf * ( psx (ji+1,jj,jcat) - ( z1malf - zalf ) * psxx(ji+1,jj,jcat) ) )
            zfxx(ji,jj) = zfxx(ji,jj) + zalf3 *              psxx(ji+1,jj,jcat)
            zfy (ji,jj) = zfy (ji,jj) + zalf  * (            psy (ji+1,jj,jcat) -          z1malf   * psxy(ji+1,jj,jcat) )
            zfyy(ji,jj) = zfyy(ji,jj) + zalf  *              psyy(ji+1,jj,jcat)
            zfxy(ji,jj) = zfxy(ji,jj) + zalf2 *              psxy(ji+1,jj,jcat)
         ENDIF
         !
      END_2D

      DO_2D( ji0-1, ji0-1, jj0, jj0 )
         !
         zpsm  = psm (ji,jj) ! optimization
         zps0  = ps0 (ji,jj)
         zpsx  = psx (ji,jj,jcat)
         zpsxx = psxx(ji,jj,jcat)
         zpsy  = psy (ji,jj,jcat)
         zpsyy = psyy(ji,jj,jcat)
         zpsxy = psxy(ji,jj,jcat)
         !                                !  Readjust moments remaining in the box.
         IF( put(ji-1,jj) < 0._wp ) THEN
            !
            zalf    = - put(ji-1,jj) * pdt / psm(ji,jj)
            z1malf  =   1._wp - zalf
            !
            z1malf2 = z1malf  * z1malf
            z1malf3 = z1malf2 * z1malf
            !
            zpsm  = zpsm    -   zfm (ji-1,jj)
            zps0  = zps0    -   zf0 (ji-1,jj)
            zpsx  = z1malf2 * ( zpsx + 3._wp * zalf * zpsxx )
            zpsxx = z1malf3 *   zpsxx
            zpsy  = zpsy    -   zfy (ji-1,jj)
            zpsyy = zpsyy   -   zfyy(ji-1,jj)
            zpsxy = z1malf2 *   zpsxy
         ENDIF

         !   Put the temporary moments into appropriate neighboring boxes.
         !                                !   Flux from i to i+1 IF u GT 0.
         IF( put(ji-1,jj) >= 0._wp ) THEN
            !
            zpsm = zpsm + zfm(ji-1,jj)
            !
            zalf    = zfm(ji-1,jj) / zpsm
            z1malf  = 1._wp - zalf
            !
            zalf2   =   zalf *   zalf
            z1malf2 = z1malf * z1malf
            !
            zps0  = zps0 +               zf0 (ji-1,jj)
            zpsx  = (            zalf  * zfx (ji-1,jj) + z1malf  * zpsx  ) + 3._wp * ( zalf * zps0 - z1malf * zf0(ji-1,jj) )
            zpsxx = (            zalf2 * zfxx(ji-1,jj) + z1malf2 * zpsxx ) &
               &  + 5._wp * (    zalf  * z1malf * ( zpsx - zfx(ji-1,jj) ) - (z1malf-zalf) * (zalf * zps0 - z1malf * zf0(ji-1,jj)) )
            zpsxy =         (    zalf  * zfxy(ji-1,jj) + z1malf * zpsxy )  & ! do not move this line (it depends on zpsy)
               &  + 3._wp * ( -z1malf  * zfy (ji-1,jj) +   zalf * zpsy  )
            zpsy  = zpsy  +              zfy (ji-1,jj)
            zpsyy = zpsyy +              zfyy(ji-1,jj)
         ENDIF
         
         !                                !  Flux from i+1 to i IF u LT 0.
         IF( put(ji,jj) < 0._wp ) THEN
            !
            zpsm = zpsm + zfm(ji,jj)
            !
            zalf    = zfm(ji,jj) / zpsm
            z1malf  = 1._wp - zalf
            !
            zalf2   =   zalf *   zalf
            z1malf2 = z1malf * z1malf
            !
            zps0  = zps0 +              zf0 (ji,jj)
            zpsx  = (           zalf  * zfx (ji,jj) + z1malf  * zpsx ) + 3._wp * ( -zalf * zps0 + z1malf * zf0(ji,jj) )
            zpsxx = (           zalf2 * zfxx(ji,jj) + z1malf2 * zpsxx ) &
               &  + 5._wp * (   zalf  * z1malf * ( -zpsx + zfx (ji,jj) ) + (z1malf-zalf) * (-zalf * zps0 + z1malf * zf0(ji,jj)) ) 
            zpsxy =         (   zalf  * zfxy(ji,jj) + z1malf * zpsxy )  & ! do not move this line (it depends on zpsy)
               &  + 3._wp * ( z1malf  * zfy (ji,jj) -   zalf * zpsy  ) 
            zpsy  = zpsy  +             zfy (ji,jj)
            zpsyy = zpsyy +             zfyy(ji,jj)
         ENDIF
         !
         psm (ji,jj) = zpsm  ! optimization
         ps0 (ji,jj) = zps0
         psx (ji,jj,jcat) = zpsx
         psxx(ji,jj,jcat) = zpsxx
         psy (ji,jj,jcat) = zpsy
         psyy(ji,jj,jcat) = zpsyy
         psxy(ji,jj,jcat) = zpsxy
         !
      END_2D
      !
   END SUBROUTINE adv_x


   SUBROUTINE adv_y( ihls, jcat, pdt, pvt , pcrh, psm , ps0 ,   &
      &              psx, psxx, psy , psyy, psxy )
      !!---------------------------------------------------------------------
      !!                **  routine adv_y  **
      !!
      !! ** purpose :   Computes and adds the advection trend to sea-ice
      !!                variable on y axis
      !!---------------------------------------------------------------------
      INTEGER                   , INTENT(in   ) ::   ihls               ! loop index
      INTEGER                   , INTENT(in   ) ::   jcat               ! category
      REAL(wp)                  , INTENT(in   ) ::   pdt                ! time step
      REAL(wp)                  , INTENT(in   ) ::   pcrh               ! call adv_x then adv_y (=1) or the opposite (=0)
      REAL(wp), DIMENSION(:,:)  , INTENT(in   ) ::   pvt                ! j-direction ice velocity at V-point [m/s]
      REAL(wp), DIMENSION(:,:)  , INTENT(inout) ::   psm                ! area
      REAL(wp), DIMENSION(:,:)  , INTENT(inout) ::   ps0                ! field to be advected
      REAL(wp), DIMENSION(:,:,:), INTENT(inout) ::   psx , psy          ! 1st moments
      REAL(wp), DIMENSION(:,:,:), INTENT(inout) ::   psxx, psyy, psxy   ! 2nd moments
      !!
      INTEGER  ::   ji, jj                               ! dummy loop indices
      INTEGER  ::   ji0, jj0                             ! dummy loop indices
      REAL(wp) ::   z1_3
      REAL(wp) ::   zs1max, zslpmax                      ! local scalars
      REAL(wp) ::   zs1new, zalf , zalf2, zalf3          !   -      -
      REAL(wp) ::   zs2new, z1malf, z1malf2, z1malf3     !   -      -
      REAL(wp) ::   zpsm, zps0
      REAL(wp) ::   zpsx, zpsy, zpsxx, zpsyy, zpsxy
      REAL(wp), DIMENSION(jpi,jpj) ::   zf0 , zfx  , zfy            ! 2D workspace
      REAL(wp), DIMENSION(jpi,jpj) ::   zfm , zfxx , zfyy  , zfxy   !  -      -
      !---------------------------------------------------------------------
      ! in order to avoid lbc_lnk (communications):
      !    ji loop must be 1:jpi   if adv_y is called first
      !                and 2:jpi-1 if adv_y is called second
      ji0 = NINT(pcrh) + ihls
      jj0 = 1 + ihls
      !
      z1_3 = 1._wp / 3._wp
      ! Limitation of moments.
      DO_2D( ji0, ji0, jj0, jj0 )
         !
         zpsm  = psm (ji,jj) ! optimization
         zps0  = ps0 (ji,jj)
         zpsx  = psx (ji,jj,jcat)
         zpsxx = psxx(ji,jj,jcat)
         zpsy  = psy (ji,jj,jcat)
         zpsyy = psyy(ji,jj,jcat)
         zpsxy = psxy(ji,jj,jcat)
         !
         !  Initialize volumes of boxes (=area if adv_y first called, =psm otherwise)
         zpsm = MAX( pcrh * e1e2t(ji,jj) + ( 1._wp - pcrh ) * zpsm , epsi20  )
         !
         zslpmax = MAX( 0._wp, zps0 )
         zps0    = zslpmax
         !
         IF( zslpmax > 0._wp ) THEN
            zs1max = 1.5_wp * zslpmax
            zs1new = MIN( zs1max, MAX( -zs1max, zpsy ) )
            zs2new = MIN( 2._wp * zslpmax - z1_3 * ABS( zs1new ), MAX( ABS( zs1new ) - zslpmax, zpsyy ) )
            !
            zpsx  = zpsx    * tmask(ji,jj,1)
            zpsxx = zpsxx   * tmask(ji,jj,1)
            zpsy  = zs1new  * tmask(ji,jj,1)
            zpsyy = zs2new  * tmask(ji,jj,1)
            zpsxy = MIN( zslpmax, MAX( -zslpmax, zpsxy ) ) * tmask(ji,jj,1)
         ELSE
            zpsx  = 0._wp
            zpsxx = 0._wp
            zpsy  = 0._wp
            zpsyy = 0._wp
            zpsxy = 0._wp
         ENDIF
         !
         !  Calculate fluxes and moments between boxes j<-->j+1
         !                                !  Flux from j to j+1 WHEN v GT 0
         IF( pvt(ji,jj) >= 0._wp ) THEN
            !
            zalf    = pvt(ji,jj) * pdt / zpsm
            z1malf  = 1._wp - zalf
            !
            zalf2   =   zalf  *   zalf
            zalf3   =   zalf2 *   zalf
            z1malf2 = z1malf  * z1malf
            z1malf3 = z1malf2 * z1malf
            !
            zfm (ji,jj) = zalf  *   zpsm
            zf0 (ji,jj) = zalf  * ( zps0  +         z1malf * ( zpsy + (z1malf - zalf) * zpsyy ) )
            zfy (ji,jj) = zalf2 * ( zpsy  + 3._wp * z1malf *   zpsyy )
            zfyy(ji,jj) = zalf3 *   zpsyy 
            zfx (ji,jj) = zalf  * ( zpsx  +         z1malf *   zpsxy )
            zfxx(ji,jj) = zalf  *   zpsxx
            zfxy(ji,jj) = zalf2 *   zpsxy
            !
            !                                !  Readjust moments remaining in the box.
            zpsm  = zpsm    -   zfm (ji,jj)
            zps0  = zps0    -   zf0 (ji,jj)
            zpsy  = z1malf2 * ( zpsy - 3._wp * zalf * zpsyy )
            zpsyy = z1malf3 *   zpsyy
            zpsx  = zpsx    -   zfx (ji,jj)
            zpsxx = zpsxx   -   zfxx(ji,jj)
            zpsxy = z1malf2 *   zpsxy
         ELSE
            zfm (ji,jj) = 0._wp
            zf0 (ji,jj) = 0._wp
            zfx (ji,jj) = 0._wp
            zfxx(ji,jj) = 0._wp
            zfy (ji,jj) = 0._wp
            zfyy(ji,jj) = 0._wp
            zfxy(ji,jj) = 0._wp
         ENDIF
         !
         psm (ji,jj) = zpsm ! optimization
         ps0 (ji,jj) = zps0
         psx (ji,jj,jcat) = zpsx
         psxx(ji,jj,jcat) = zpsxx
         psy (ji,jj,jcat) = zpsy
         psyy(ji,jj,jcat) = zpsyy
         psxy(ji,jj,jcat) = zpsxy
         !
      END_2D
      !
      DO_2D( ji0, ji0, jj0, jj0-1 )
         !                                !  Flux from j+1 to j when v LT 0.
         IF( pvt(ji,jj) < 0._wp ) THEN
            !
            zalf   = - pvt(ji,jj) * pdt / psm(ji,jj+1)
            z1malf =   1._wp - zalf
            !
            zalf2  = zalf  * zalf
            zalf3  = zalf2 * zalf
            !
            zfm (ji,jj) = zfm (ji,jj) + zalf  *   psm (ji,jj+1)
            zf0 (ji,jj) = zf0 (ji,jj) + zalf  * ( ps0 (ji,jj+1) &
               &                                - z1malf * ( psy (ji,jj+1,jcat) - ( z1malf - zalf ) * psyy(ji,jj+1,jcat) ) )
            zfy (ji,jj) = zfy (ji,jj) + zalf2 * (            psy (ji,jj+1,jcat) -  3._wp * z1malf   * psyy(ji,jj+1,jcat) )
            zfyy(ji,jj) = zfyy(ji,jj) + zalf3 *              psyy(ji,jj+1,jcat)
            zfx (ji,jj) = zfx (ji,jj) + zalf  * (            psx (ji,jj+1,jcat) -          z1malf   * psxy(ji,jj+1,jcat) )
            zfxx(ji,jj) = zfxx(ji,jj) + zalf  *              psxx(ji,jj+1,jcat)
            zfxy(ji,jj) = zfxy(ji,jj) + zalf2 *              psxy(ji,jj+1,jcat)
         ENDIF
         !
      END_2D

      DO_2D( ji0, ji0, jj0-1, jj0-1 )
         !
         zpsm  = psm (ji,jj) ! optimization
         zps0  = ps0 (ji,jj)
         zpsx  = psx (ji,jj,jcat)
         zpsxx = psxx(ji,jj,jcat)
         zpsy  = psy (ji,jj,jcat)
         zpsyy = psyy(ji,jj,jcat)
         zpsxy = psxy(ji,jj,jcat)
         !                                !  Readjust moments remaining in the box.
         IF( pvt(ji,jj-1) < 0._wp ) THEN
            !
            zalf    = - pvt(ji,jj-1) * pdt / psm(ji,jj)
            z1malf  =   1._wp - zalf
            !
            z1malf2 = z1malf  * z1malf
            z1malf3 = z1malf2 * z1malf
            !
            zpsm  = zpsm    -   zfm(ji,jj-1)
            zps0  = zps0    -   zf0(ji,jj-1)
            zpsy  = z1malf2 * ( zpsy + 3._wp * zalf * zpsyy )
            zpsyy = z1malf3 *   zpsyy
            zpsx  = zpsx    -   zfx (ji,jj-1)
            zpsxx = zpsxx   -   zfxx(ji,jj-1)
            zpsxy = z1malf2 *   zpsxy
         ENDIF

         !   Put the temporary moments into appropriate neighboring boxes.
         !                                !   Flux from j to j+1 IF v GT 0.
         IF( pvt(ji,jj-1) >= 0._wp ) THEN
            !
            zpsm = zpsm + zfm(ji,jj-1)
            !
            zalf   = zfm(ji,jj-1) / zpsm
            z1malf = 1._wp - zalf
            !
            zalf2   =   zalf *   zalf
            z1malf2 = z1malf * z1malf            
            !
            zps0  = zps0 +               zf0 (ji,jj-1)
            zpsy  = (            zalf  * zfy (ji,jj-1) + z1malf  * zpsy  ) + 3._wp * ( zalf * zps0 - z1malf * zf0(ji,jj-1) )
            zpsyy = (            zalf2 * zfyy(ji,jj-1) + z1malf2 * zpsyy ) &
               &  + 5._wp * (    zalf  * z1malf * ( zpsy - zfy(ji,jj-1) ) - (z1malf-zalf) * (zalf * zps0 - z1malf * zf0(ji,jj-1)) )
            zpsxy =         (    zalf  * zfxy(ji,jj-1) + z1malf * zpsxy ) & ! do not move this line (it depends on zpsx)
               &  + 3._wp * ( -z1malf  * zfx (ji,jj-1) +   zalf * zpsx  ) 
            zpsx  = zpsx  +              zfx (ji,jj-1)
            zpsxx = zpsxx +              zfxx(ji,jj-1)
         ENDIF
         
         !                                !  Flux from j+1 to j IF v LT 0.
         IF( pvt(ji,jj) < 0._wp ) THEN
            !
            zpsm  = zpsm + zfm(ji,jj)
            !
            zalf   = zfm(ji,jj) / zpsm
            z1malf = 1._wp - zalf
            !
            zalf2   =   zalf *   zalf
            z1malf2 = z1malf * z1malf
            !
            zps0  = zps0 +              zf0 (ji,jj)
            zpsy  = (           zalf  * zfy (ji,jj) + z1malf  * zpsy ) + 3._wp * ( -zalf * zps0 + z1malf * zf0(ji,jj) )
            zpsyy = (           zalf2 * zfyy(ji,jj) + z1malf2 * zpsyy ) &
               &  + 5._wp * (   zalf  * z1malf * ( - zpsy + zfy(ji,jj) ) + (z1malf-zalf) * (-zalf * zps0 + z1malf * zf0(ji,jj)) )
            zpsxy =         (  zalf   * zfxy(ji,jj) + z1malf * zpsxy )  & ! do not move this line (it depends on zpsx)
               &  + 3._wp * ( z1malf  * zfx (ji,jj) - zalf   * zpsx  )
            zpsx  = zpsx  +             zfx (ji,jj)
            zpsxx = zpsxx +             zfxx(ji,jj)
         ENDIF
         !
         psm (ji,jj) = zpsm ! optimization
         ps0 (ji,jj) = zps0
         psx (ji,jj,jcat) = zpsx
         psxx(ji,jj,jcat) = zpsxx
         psy (ji,jj,jcat) = zpsy
         psyy(ji,jj,jcat) = zpsyy
         psxy(ji,jj,jcat) = zpsxy
         !
      END_2D
      !
   END SUBROUTINE adv_y


   SUBROUTINE Hbig_pra( ihls, jcat, pdt, phi_max, phs_max, phip_max, psi_max, pes_max, pei_max, pszi_max, &
      &                  pv_i, pv_s, pa_i, pa_ip, pv_ip, psv_i, pe_s, pe_i, pszv_i )
      !!-------------------------------------------------------------------
      !!                  ***  ROUTINE Hbig_pra  ***
      !!
      !! ** Purpose : Thickness correction in case advection scheme creates
      !!              abnormally tick ice or snow
      !!
      !! ** Method  : 1- check whether ice thickness is larger than the surrounding 9-points
      !!                 (before advection) and reduce it by adapting ice concentration
      !!              2- check whether snow thickness is larger than the surrounding 9-points
      !!                 (before advection) and reduce it by sending the excess in the ocean
      !!
      !! ** input   : Max thickness of the surrounding 9-points
      !!-------------------------------------------------------------------
      INTEGER                     , INTENT(in   ) ::   ihls                                  ! loop index
      INTEGER                     , INTENT(in   ) ::   jcat                                  ! category
      REAL(wp)                    , INTENT(in   ) ::   pdt                                   ! tracer time-step
      REAL(wp), DIMENSION(:,:)    , INTENT(in   ) ::   phi_max, phs_max, phip_max, psi_max   ! max ice thick from surrounding 9-pts
      REAL(wp), DIMENSION(:,:,:)  , INTENT(in   ) ::   pes_max
      REAL(wp), DIMENSION(:,:,:)  , INTENT(in   ) ::   pei_max, pszi_max
      REAL(wp), DIMENSION(:,:,:)  , INTENT(inout) ::   pv_i, pv_s, pa_i, pa_ip, pv_ip, psv_i
      REAL(wp), DIMENSION(:,:,:,:), INTENT(inout) ::   pe_s
      REAL(wp), DIMENSION(:,:,:,:), INTENT(inout) ::   pe_i
      REAL(wp), DIMENSION(:,:,:,:), INTENT(inout) ::   pszv_i
      !
      INTEGER  ::   ji, jj, jk         ! dummy loop indices
      REAL(wp) ::   z1_dt, zhip, zhi, zhs, zsi, zes, zei, zfra
      REAL(wp), DIMENSION(jpi,jpj) ::   zwfx_res, zhfx_res, zsfx_res  ! needed since loop is not (0,0,0,0)
      !!-------------------------------------------------------------------
      !
      DO_2D( ihls, ihls, ihls, ihls )
         zwfx_res(ji,jj) = 0._wp
         zhfx_res(ji,jj) = 0._wp
         zsfx_res(ji,jj) = 0._wp
      END_2D
      !
      z1_dt = 1._wp / pdt
      !
      DO_2D( ihls, ihls, ihls, ihls )
         IF ( pv_i(ji,jj,jcat) > 0._wp .AND. pa_i(ji,jj,jcat) > 0._wp ) THEN
            !
            !                               ! -- check h_ip -- !
            ! if h_ip is larger than the surrounding 9 pts => reduce h_ip and increase a_ip
            IF( ln_pnd_LEV .OR. ln_pnd_TOPO .AND. pv_ip(ji,jj,jcat) > 0._wp ) THEN
               zhip = pv_ip(ji,jj,jcat) / MAX( epsi20, pa_ip(ji,jj,jcat) )
               IF( zhip > phip_max(ji,jj) .AND. pa_ip(ji,jj,jcat) < 0.15 ) THEN
                  pa_ip(ji,jj,jcat) = pv_ip(ji,jj,jcat) / phip_max(ji,jj)
               ENDIF
            ENDIF
            !
            !                               ! -- check h_i -- !
            ! if h_i is larger than the surrounding 9 pts => reduce h_i and increase a_i
            zhi = pv_i(ji,jj,jcat) / pa_i(ji,jj,jcat)
            IF( zhi > phi_max(ji,jj) .AND. pa_i(ji,jj,jcat) < 0.15 ) THEN
               pa_i(ji,jj,jcat) = pv_i(ji,jj,jcat) / MIN( phi_max(ji,jj), hi_max(jpl) )   !-- bound h_i to hi_max (99 m)
            ENDIF
            !
            !                               ! -- check h_s -- !
            ! if h_s is larger than the surrounding 9 pts => put the snow excess in the ocean
            zhs = pv_s(ji,jj,jcat) / pa_i(ji,jj,jcat)
            IF( pv_s(ji,jj,jcat) > 0._wp .AND. zhs > phs_max(ji,jj) .AND. pa_i(ji,jj,jcat) < 0.15 ) THEN
               zfra = phs_max(ji,jj) / MAX( zhs, epsi20 )
               !
               zwfx_res(ji,jj) = zwfx_res(ji,jj) + ( pv_s(ji,jj,jcat) - pa_i(ji,jj,jcat) * phs_max(ji,jj) ) * rhos * z1_dt
               zhfx_res(ji,jj) = zhfx_res(ji,jj) - SUM( pe_s(ji,jj,1:nlay_s,jcat) ) * ( 1._wp - zfra ) * z1_dt ! W.m-2 <0
               !
               pe_s(ji,jj,1:nlay_s,jcat) = pe_s(ji,jj,1:nlay_s,jcat) * zfra
               pv_s(ji,jj,jcat)          = pa_i(ji,jj,jcat) * phs_max(ji,jj)
            ENDIF
            !
         ENDIF
      END_2D
      !
      !                                          ! -- check s_i -- !
      IF( nn_icesal == 4 ) THEN
         DO_3D( ihls, ihls, ihls, ihls, 1, nlay_i )
            IF ( pv_i(ji,jj,jcat) > 0._wp .AND. pa_i(ji,jj,jcat) > 0._wp ) THEN
               ! if szv_i/v_i is larger than the surrounding 9 pts => put the salt excess in the ocean
               zsi = pszv_i(ji,jj,jk,jcat) / pv_i(ji,jj,jcat)
               IF( zsi > pszi_max(ji,jj,jk) .AND. pa_i(ji,jj,jcat) < 0.15 ) THEN
                  zfra = pszi_max(ji,jj,jk) / zsi
                  zsfx_res(ji,jj) = zsfx_res(ji,jj) + pszv_i(ji,jj,jk,jcat) * ( 1._wp - zfra ) * rhoi * z1_dt
                  pszv_i(ji,jj,jk,jcat) = pszv_i(ji,jj,jk,jcat) * zfra
               ENDIF
            ENDIF
         END_3D
      ELSE
         DO_2D( ihls, ihls, ihls, ihls )
            IF ( pv_i(ji,jj,jcat) > 0._wp .AND. pa_i(ji,jj,jcat) > 0._wp ) THEN
               ! if s_i is larger than the surrounding 9 pts => put salt excess in the ocean
               zsi = psv_i(ji,jj,jcat) / pv_i(ji,jj,jcat)
               IF( zsi > psi_max(ji,jj) .AND. pa_i(ji,jj,jcat) < 0.15 ) THEN
                  zfra = psi_max(ji,jj) / zsi
                  zsfx_res(ji,jj) = zsfx_res(ji,jj) + psv_i(ji,jj,jcat) * ( 1._wp - zfra ) * rhoi * z1_dt
                  psv_i(ji,jj,jcat) = psv_i(ji,jj,jcat) * zfra
               ENDIF
               !
            ENDIF
         END_2D
      ENDIF
      !                                           ! -- check e_i/v_i -- !
      DO_3D( ihls, ihls, ihls, ihls, 1, nlay_i )
         IF ( pv_i(ji,jj,jcat) > 0._wp .AND. pa_i(ji,jj,jcat) > 0._wp ) THEN
            ! if e_i/v_i is larger than the surrounding 9 pts => put the heat excess in the ocean
            zei = pe_i(ji,jj,jk,jcat) / pv_i(ji,jj,jcat)
            IF( zei > pei_max(ji,jj,jk) .AND. pa_i(ji,jj,jcat) < 0.15 ) THEN
               zfra = pei_max(ji,jj,jk) / zei
               zhfx_res(ji,jj) = zhfx_res(ji,jj) - pe_i(ji,jj,jk,jcat) * ( 1._wp - zfra ) * z1_dt ! W.m-2 <0
               pe_i(ji,jj,jk,jcat) = pe_i(ji,jj,jk,jcat) * zfra
            ENDIF
         ENDIF
      END_3D
      !                                           ! -- check e_s/v_s -- !
      DO_3D( ihls, ihls, ihls, ihls, 1, nlay_s )
         IF ( pv_s(ji,jj,jcat) > 0._wp .AND. pa_i(ji,jj,jcat) > 0._wp ) THEN
            ! if e_s/v_s is larger than the surrounding 9 pts => put the heat excess in the ocean
            zes = pe_s(ji,jj,jk,jcat) / pv_s(ji,jj,jcat)
            IF( zes > pes_max(ji,jj,jk) .AND. pa_i(ji,jj,jcat) < 0.15 ) THEN
               zfra = pes_max(ji,jj,jk) / zes
               zhfx_res(ji,jj) = zhfx_res(ji,jj) - pe_s(ji,jj,jk,jcat) * ( 1._wp - zfra ) * z1_dt ! W.m-2 <0
               pe_s(ji,jj,jk,jcat) = pe_s(ji,jj,jk,jcat) * zfra
            ENDIF
         ENDIF
      END_3D
      !
      ! record residual fluxes
      DO_2D( 0, 0, 0, 0 )
         wfx_res(ji,jj) = wfx_res(ji,jj) + zwfx_res(ji,jj)
         hfx_res(ji,jj) = hfx_res(ji,jj) + zhfx_res(ji,jj)
         sfx_res(ji,jj) = sfx_res(ji,jj) + zsfx_res(ji,jj)
      END_2D
      !
   END SUBROUTINE Hbig_pra


   SUBROUTINE Hsnow_pra( ihls, jcat, pdt, pv_i, pv_s, pa_i, pa_ip, pe_s )
      !!-------------------------------------------------------------------
      !!                  ***  ROUTINE Hsnow_pra  ***
      !!
      !! ** Purpose : 1- Check snow load after advection
      !!              2- Correct pond concentration to avoid a_ip > a_i
      !!
      !! ** Method :  If snow load makes snow-ice interface to deplet below the ocean surface
      !!              then put the snow excess in the ocean
      !!
      !! ** Notes :   This correction is crucial because of the call to routine icecor afterwards
      !!              which imposes a mini of ice thick. (rn_himin). This imposed mini can artificially
      !!              make the snow very thick (if concentration decreases drastically)
      !!              This behavior has been seen in Ultimate-Macho and supposedly it can also be true for Prather
      !!-------------------------------------------------------------------
      INTEGER                     , INTENT(in   ) ::   ihls  ! loop index
      INTEGER                     , INTENT(in   ) ::   jcat  ! category
      REAL(wp)                    , INTENT(in   ) ::   pdt   ! tracer time-step
      REAL(wp), DIMENSION(:,:,:)  , INTENT(inout) ::   pv_i, pv_s, pa_i, pa_ip
      REAL(wp), DIMENSION(:,:,:,:), INTENT(inout) ::   pe_s
      !
      INTEGER  ::   ji, jj ! dummy loop indices
      REAL(wp) ::   z1_dt, zvs_excess, zfra
      REAL(wp), DIMENSION(jpi,jpj) ::   zwfx_res, zhfx_res  ! needed since loop is not (0,0,0,0)
      !!-------------------------------------------------------------------
      !
      DO_2D( ihls, ihls, ihls, ihls )
         zwfx_res(ji,jj) = 0._wp
         zhfx_res(ji,jj) = 0._wp
      END_2D
      !
      z1_dt = 1._wp / pdt
      !
      ! -- check snow load -- !
      DO_2D( ihls, ihls, ihls, ihls )
         IF ( pv_i(ji,jj,jcat) > 0._wp ) THEN
            !
            zvs_excess = MAX( 0._wp, pv_s(ji,jj,jcat) - pv_i(ji,jj,jcat) * (rho0-rhoi) * r1_rhos )
            !
            IF( zvs_excess > 0._wp ) THEN   ! snow-ice interface deplets below the ocean surface
               ! put snow excess in the ocean
               zfra = ( pv_s(ji,jj,jcat) - zvs_excess ) / MAX( pv_s(ji,jj,jcat), epsi20 )
               zwfx_res(ji,jj) = zwfx_res(ji,jj) + zvs_excess * rhos * z1_dt
               zhfx_res(ji,jj) = zhfx_res(ji,jj) - SUM( pe_s(ji,jj,1:nlay_s,jcat) ) * ( 1._wp - zfra ) * z1_dt ! W.m-2 <0
               ! correct snow volume and heat content
               pe_s(ji,jj,1:nlay_s,jcat) = pe_s(ji,jj,1:nlay_s,jcat) * zfra
               pv_s(ji,jj,jcat)          = pv_s(ji,jj,jcat) - zvs_excess
            ENDIF
            !
         ENDIF
         !-- correct pond concentration to avoid a_ip > a_i -- !
         pa_ip(ji,jj,jcat) = MIN( pa_ip(ji,jj,jcat), pa_i(ji,jj,jcat) )
      END_2D
      !
      ! record residual fluxes
      DO_2D( 0, 0, 0, 0 )
         wfx_res(ji,jj) = wfx_res(ji,jj) + zwfx_res(ji,jj)
         hfx_res(ji,jj) = hfx_res(ji,jj) + zhfx_res(ji,jj)
      END_2D
      !
   END SUBROUTINE Hsnow_pra


   SUBROUTINE adv_pra_init
      !!-------------------------------------------------------------------
      !!                  ***  ROUTINE adv_pra_init  ***
      !!
      !! ** Purpose :   allocate and initialize arrays for Prather advection
      !!-------------------------------------------------------------------
      INTEGER ::   ierr(4), ii, ierr_max
      !!-------------------------------------------------------------------
      !
      ierr(:) = 0
      ii = 0
      !                             !* allocate prather fields
      ! ---------------------- !
      ! == mandatory fields == !
      ! ---------------------- !
      ii = ii + 1
      ALLOCATE( sxice(jpi,jpj,jpl) , syice(jpi,jpj,jpl) , sxxice(jpi,jpj,jpl) , syyice(jpi,jpj,jpl) , sxyice(jpi,jpj,jpl) , &
         &      sxsn (jpi,jpj,jpl) , sysn (jpi,jpj,jpl) , sxxsn (jpi,jpj,jpl) , syysn (jpi,jpj,jpl) , sxysn (jpi,jpj,jpl) , &
         &      sxa  (jpi,jpj,jpl) , sya  (jpi,jpj,jpl) , sxxa  (jpi,jpj,jpl) , syya  (jpi,jpj,jpl) , sxya  (jpi,jpj,jpl) , &
         &      sxage(jpi,jpj,jpl) , syage(jpi,jpj,jpl) , sxxage(jpi,jpj,jpl) , syyage(jpi,jpj,jpl) , sxyage(jpi,jpj,jpl) , &
         &      sxc0 (jpi,jpj,nlay_s,jpl) , syc0 (jpi,jpj,nlay_s,jpl) , sxxc0(jpi,jpj,nlay_s,jpl) , &
         &      syyc0(jpi,jpj,nlay_s,jpl) , sxyc0(jpi,jpj,nlay_s,jpl)                             , &
         &      sxe  (jpi,jpj,nlay_i,jpl) , sye  (jpi,jpj,nlay_i,jpl) , sxxe (jpi,jpj,nlay_i,jpl) , &
         &      syye (jpi,jpj,nlay_i,jpl) , sxye (jpi,jpj,nlay_i,jpl)                             , &
         &      STAT = ierr(ii) )
      !
      ! --------------------- !
      ! == optional fields == !
      ! --------------------- !            
      ii = ii + 1             ! sxsal etc must be allocated for conveniency
      ALLOCATE( sxsal(jpi,jpj,jpl) , sysal(jpi,jpj,jpl) , sxxsal(jpi,jpj,jpl) , syysal(jpi,jpj,jpl) , sxysal(jpi,jpj,jpl) , &
         &      STAT = ierr(ii) )
      !
      ii = ii + 1
      IF( nn_icesal == 4 ) THEN
         ALLOCATE( sxsi (jpi,jpj,nlay_i,jpl) , sysi (jpi,jpj,nlay_i,jpl) , sxxsi(jpi,jpj,nlay_i,jpl) , &
            &      syysi(jpi,jpj,nlay_i,jpl) , sxysi(jpi,jpj,nlay_i,jpl)                             , &
            &      STAT = ierr(ii) )
      ENDIF
      !
      ii = ii + 1
      IF( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN
         ALLOCATE( sxap (jpi,jpj,jpl) , syap (jpi,jpj,jpl) , sxxap (jpi,jpj,jpl) , syyap (jpi,jpj,jpl) , sxyap (jpi,jpj,jpl) , &
            &      sxvp (jpi,jpj,jpl) , syvp (jpi,jpj,jpl) , sxxvp (jpi,jpj,jpl) , syyvp (jpi,jpj,jpl) , sxyvp (jpi,jpj,jpl) , &
            &      sxvl (jpi,jpj,jpl) , syvl (jpi,jpj,jpl) , sxxvl (jpi,jpj,jpl) , syyvl (jpi,jpj,jpl) , sxyvl (jpi,jpj,jpl) , &
            &      STAT = ierr(ii) )
      ENDIF
      !
      ierr_max = MAXVAL( ierr(:) )
      !
      CALL mpp_sum( 'icedyn_adv_pra', ierr_max )
      IF( ierr_max /= 0 )   CALL ctl_stop('STOP', 'adv_pra_init : unable to allocate ice arrays for Prather advection scheme')
      !
      CALL adv_pra_rst( 'READ' )    !* read or initialize all required files
      !
   END SUBROUTINE adv_pra_init


   SUBROUTINE adv_pra_rst( cdrw, kt )
      !!---------------------------------------------------------------------
      !!                   ***  ROUTINE adv_pra_rst  ***
      !!
      !! ** Purpose :   Read or write file in restart file
      !!
      !! ** Method  :   use of IOM library
      !!----------------------------------------------------------------------
      CHARACTER(len=*) , INTENT(in) ::   cdrw   ! "READ"/"WRITE" flag
      INTEGER, OPTIONAL, INTENT(in) ::   kt     ! ice time-step
      !
      INTEGER ::   jk, jl   ! dummy loop indices
      INTEGER ::   iter     ! local integer
      INTEGER ::   id1, id0      ! local integer
      CHARACTER(len=25) ::   znam
      CHARACTER(len=2)  ::   zchar, zchar1
      REAL(wp), DIMENSION(jpi,jpj,jpl) ::   z3d   ! 3D workspace
      !!----------------------------------------------------------------------
      !
      !                                      !==========================!
      IF( TRIM(cdrw) == 'READ' ) THEN        !==  Read or initialize  ==!
         !                                   !==========================!
         !
         IF( ln_rstart ) THEN   ;   id1 = iom_varid( numrir, 'sxice' , ldstop = .FALSE. )    ! file exist: id1>0
         ELSE                   ;   id1 = 0                                                  ! no restart: id1=0
         ENDIF
         !
         ! check size of the input fields
         id0=0
         DO jk = 1, 99
            WRITE(zchar1,'(I2.2)') jk
            znam = 'sxc0'//'_l'//zchar1
            IF( iom_varid( numrir, znam , ldstop = .FALSE. ) > 0 )   id0 = id0+1 
            znam = 'sxe'//'_l'//zchar1
            IF( iom_varid( numrir, znam , ldstop = .FALSE. ) > 0 )   id0 = id0+1 
         END DO
         
         IF( id1 > 0 .AND. id0 == (nlay_s+nlay_i) ) THEN   !**  Read the restart file  **!
            !
            ! ---------------------- !
            ! == mandatory fields == !
            ! ---------------------- !
            !                                                        ! ice thickness
            CALL iom_get( numrir, jpdom_auto, 'sxice' , sxice , psgn = -1._wp )
            CALL iom_get( numrir, jpdom_auto, 'syice' , syice , psgn = -1._wp )
            CALL iom_get( numrir, jpdom_auto, 'sxxice', sxxice )
            CALL iom_get( numrir, jpdom_auto, 'syyice', syyice )
            CALL iom_get( numrir, jpdom_auto, 'sxyice', sxyice )
            !                                                        ! snow thickness
            CALL iom_get( numrir, jpdom_auto, 'sxsn'  , sxsn  , psgn = -1._wp )
            CALL iom_get( numrir, jpdom_auto, 'sysn'  , sysn  , psgn = -1._wp )
            CALL iom_get( numrir, jpdom_auto, 'sxxsn' , sxxsn  )
            CALL iom_get( numrir, jpdom_auto, 'syysn' , syysn  )
            CALL iom_get( numrir, jpdom_auto, 'sxysn' , sxysn  )
            !                                                        ! ice concentration
            CALL iom_get( numrir, jpdom_auto, 'sxa'   , sxa   , psgn = -1._wp )
            CALL iom_get( numrir, jpdom_auto, 'sya'   , sya   , psgn = -1._wp )
            CALL iom_get( numrir, jpdom_auto, 'sxxa'  , sxxa   )
            CALL iom_get( numrir, jpdom_auto, 'syya'  , syya   )
            CALL iom_get( numrir, jpdom_auto, 'sxya'  , sxya   )
            !                                                        ! ice age
            CALL iom_get( numrir, jpdom_auto, 'sxage' , sxage , psgn = -1._wp )
            CALL iom_get( numrir, jpdom_auto, 'syage' , syage , psgn = -1._wp )
            CALL iom_get( numrir, jpdom_auto, 'sxxage', sxxage )
            CALL iom_get( numrir, jpdom_auto, 'syyage', syyage )
            CALL iom_get( numrir, jpdom_auto, 'sxyage', sxyage )
            !                                                        ! snow layers heat content
            DO jk = 1, nlay_s
               WRITE(zchar1,'(I2.2)') jk
               znam = 'sxc0'//'_l'//zchar1
               CALL iom_get( numrir, jpdom_auto, znam , z3d, psgn = -1._wp )   ;   sxc0 (:,:,jk,:) = z3d(:,:,:)
               znam = 'syc0'//'_l'//zchar1
               CALL iom_get( numrir, jpdom_auto, znam , z3d, psgn = -1._wp )   ;   syc0 (:,:,jk,:) = z3d(:,:,:)
               znam = 'sxxc0'//'_l'//zchar1
               CALL iom_get( numrir, jpdom_auto, znam , z3d )                  ;   sxxc0(:,:,jk,:) = z3d(:,:,:)
               znam = 'syyc0'//'_l'//zchar1
               CALL iom_get( numrir, jpdom_auto, znam , z3d )                  ;   syyc0(:,:,jk,:) = z3d(:,:,:)
               znam = 'sxyc0'//'_l'//zchar1
               CALL iom_get( numrir, jpdom_auto, znam , z3d )                  ;   sxyc0(:,:,jk,:) = z3d(:,:,:)
            END DO
            !                                                        ! ice layers heat content
            DO jk = 1, nlay_i
               WRITE(zchar1,'(I2.2)') jk
               znam = 'sxe'//'_l'//zchar1
               CALL iom_get( numrir, jpdom_auto, znam , z3d, psgn = -1._wp )   ;   sxe (:,:,jk,:) = z3d(:,:,:)
               znam = 'sye'//'_l'//zchar1
               CALL iom_get( numrir, jpdom_auto, znam , z3d, psgn = -1._wp )   ;   sye (:,:,jk,:) = z3d(:,:,:)
               znam = 'sxxe'//'_l'//zchar1
               CALL iom_get( numrir, jpdom_auto, znam , z3d )                  ;   sxxe(:,:,jk,:) = z3d(:,:,:)
               znam = 'syye'//'_l'//zchar1
               CALL iom_get( numrir, jpdom_auto, znam , z3d )                  ;   syye(:,:,jk,:) = z3d(:,:,:)
               znam = 'sxye'//'_l'//zchar1
               CALL iom_get( numrir, jpdom_auto, znam , z3d )                  ;   sxye(:,:,jk,:) = z3d(:,:,:)
            END DO
            !
            ! --------------------- !
            ! == optional fields == !
            ! --------------------- !            
            !                                                        ! ice salinity
            IF( iom_varid( numrir, 'sxsal', ldstop = .FALSE. ) > 0 ) THEN
               CALL iom_get( numrir, jpdom_auto, 'sxsal' , sxsal , psgn = -1._wp )
               CALL iom_get( numrir, jpdom_auto, 'sysal' , sysal , psgn = -1._wp )
               CALL iom_get( numrir, jpdom_auto, 'sxxsal', sxxsal )
               CALL iom_get( numrir, jpdom_auto, 'syysal', syysal )
               CALL iom_get( numrir, jpdom_auto, 'sxysal', sxysal )
            ELSE
               sxsal = 0._wp ; sysal = 0._wp ; sxxsal = 0._wp ; syysal = 0._wp ; sxysal = 0._wp               
            ENDIF
            !                                                        ! ice layers salt content
            IF( nn_icesal == 4 ) THEN
               IF( iom_varid( numrir, 'sxsi_l01' , ldstop = .FALSE. ) > 0 ) THEN
                  DO jk = 1, nlay_i
                     WRITE(zchar1,'(I2.2)') jk
                     znam = 'sxsi'//'_l'//zchar1
                     CALL iom_get( numrir, jpdom_auto, znam , z3d, psgn = -1._wp )   ;   sxsi (:,:,jk,:) = z3d(:,:,:)
                     znam = 'sysi'//'_l'//zchar1
                     CALL iom_get( numrir, jpdom_auto, znam , z3d, psgn = -1._wp )   ;   sysi (:,:,jk,:) = z3d(:,:,:)
                     znam = 'sxxsi'//'_l'//zchar1
                     CALL iom_get( numrir, jpdom_auto, znam , z3d )                  ;   sxxsi(:,:,jk,:) = z3d(:,:,:)
                     znam = 'syysi'//'_l'//zchar1
                     CALL iom_get( numrir, jpdom_auto, znam , z3d )                  ;   syysi(:,:,jk,:) = z3d(:,:,:)
                     znam = 'sxysi'//'_l'//zchar1
                     CALL iom_get( numrir, jpdom_auto, znam , z3d )                  ;   sxysi(:,:,jk,:) = z3d(:,:,:)
                  END DO
               ELSE
                  sxsi = 0._wp ; sysi = 0._wp ; sxxsi = 0._wp ; syysi = 0._wp ; sxysi = 0._wp               
               ENDIF
            ENDIF
            !
            IF( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN                   ! melt pond fraction
               IF( iom_varid( numrir, 'sxap', ldstop = .FALSE. ) > 0 ) THEN
                  CALL iom_get( numrir, jpdom_auto, 'sxap' , sxap , psgn = -1._wp )
                  CALL iom_get( numrir, jpdom_auto, 'syap' , syap , psgn = -1._wp )
                  CALL iom_get( numrir, jpdom_auto, 'sxxap', sxxap )
                  CALL iom_get( numrir, jpdom_auto, 'syyap', syyap )
                  CALL iom_get( numrir, jpdom_auto, 'sxyap', sxyap )
                  !                                                  ! melt pond volume
                  CALL iom_get( numrir, jpdom_auto, 'sxvp' , sxvp , psgn = -1._wp )
                  CALL iom_get( numrir, jpdom_auto, 'syvp' , syvp , psgn = -1._wp )
                  CALL iom_get( numrir, jpdom_auto, 'sxxvp', sxxvp )
                  CALL iom_get( numrir, jpdom_auto, 'syyvp', syyvp )
                  CALL iom_get( numrir, jpdom_auto, 'sxyvp', sxyvp )
               ELSE
                  sxap = 0._wp ; syap = 0._wp ; sxxap = 0._wp ; syyap = 0._wp ; sxyap = 0._wp
                  sxvp = 0._wp ; syvp = 0._wp ; sxxvp = 0._wp ; syyvp = 0._wp ; sxyvp = 0._wp
               ENDIF
                  !                                                  ! melt pond lid volume
               IF( iom_varid( numrir, 'sxvl', ldstop = .FALSE. ) > 0 ) THEN
                  CALL iom_get( numrir, jpdom_auto, 'sxvl' , sxvl , psgn = -1._wp )
                  CALL iom_get( numrir, jpdom_auto, 'syvl' , syvl , psgn = -1._wp )
                  CALL iom_get( numrir, jpdom_auto, 'sxxvl', sxxvl )
                  CALL iom_get( numrir, jpdom_auto, 'syyvl', syyvl )
                  CALL iom_get( numrir, jpdom_auto, 'sxyvl', sxyvl )
               ELSE
                  sxvl = 0._wp ; syvl = 0._wp ; sxxvl = 0._wp ; syyvl = 0._wp ; sxyvl = 0._wp
               ENDIF
            ENDIF
            !
         ELSE                                   !**  start advection from rest  **!
            !
            IF(lwp) WRITE(numout,*) '   ==>>   start from rest OR previous run without Prather, set moments to 0'
            !
            sxice = 0._wp   ;   syice = 0._wp   ;   sxxice = 0._wp   ;   syyice = 0._wp   ;   sxyice = 0._wp      ! ice thickness
            sxsn  = 0._wp   ;   sysn  = 0._wp   ;   sxxsn  = 0._wp   ;   syysn  = 0._wp   ;   sxysn  = 0._wp      ! snow thickness
            sxa   = 0._wp   ;   sya   = 0._wp   ;   sxxa   = 0._wp   ;   syya   = 0._wp   ;   sxya   = 0._wp      ! ice concentration
            sxage = 0._wp   ;   syage = 0._wp   ;   sxxage = 0._wp   ;   syyage = 0._wp   ;   sxyage = 0._wp      ! ice age
            sxc0  = 0._wp   ;   syc0  = 0._wp   ;   sxxc0  = 0._wp   ;   syyc0  = 0._wp   ;   sxyc0  = 0._wp      ! snow layers heat content
            sxe   = 0._wp   ;   sye   = 0._wp   ;   sxxe   = 0._wp   ;   syye   = 0._wp   ;   sxye   = 0._wp      ! ice layers heat content
            !
            sxsal = 0._wp   ;   sysal = 0._wp   ;   sxxsal = 0._wp   ;   syysal = 0._wp   ;   sxysal = 0._wp      ! ice salinity
            IF( nn_icesal == 4 ) THEN
               sxsi = 0._wp ;   sysi  = 0._wp   ;   sxxsi  = 0._wp   ;   syysi  = 0._wp   ;   sxysi  = 0._wp      ! ice layers salt content               
            ENDIF
            IF( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN
               sxap = 0._wp ;   syap = 0._wp    ;   sxxap = 0._wp    ;   syyap = 0._wp    ;   sxyap = 0._wp       ! melt pond fraction
               sxvp = 0._wp ;   syvp = 0._wp    ;   sxxvp = 0._wp    ;   syyvp = 0._wp    ;   sxyvp = 0._wp       ! melt pond volume
               sxvl = 0._wp ;   syvl = 0._wp    ;   sxxvl = 0._wp    ;   syyvl = 0._wp    ;   sxyvl = 0._wp       ! melt pond lid volume
            ENDIF
         ENDIF
         !
         !                                   !=====================================!
      ELSEIF( TRIM(cdrw) == 'WRITE' ) THEN   !==  write in the ice restart file  ==!
         !                                   !=====================================!
         IF(lwp) WRITE(numout,*) '----  ice-adv-rst  ----'
         iter = kt + nn_fsbc - 1             ! ice restarts are written at kt == nitrst - nn_fsbc + 1
         !
         !
         ! In case Prather scheme is used for advection, write second order moments
         ! ------------------------------------------------------------------------
         !
         ! ---------------------- !
         ! == mandatory fields == !
         ! ---------------------- !
         !                                                           ! ice thickness
         CALL iom_rstput( iter, nitrst, numriw, 'sxice' , sxice  )
         CALL iom_rstput( iter, nitrst, numriw, 'syice' , syice  )
         CALL iom_rstput( iter, nitrst, numriw, 'sxxice', sxxice )
         CALL iom_rstput( iter, nitrst, numriw, 'syyice', syyice )
         CALL iom_rstput( iter, nitrst, numriw, 'sxyice', sxyice )
         !                                                           ! snow thickness
         CALL iom_rstput( iter, nitrst, numriw, 'sxsn'  , sxsn   )
         CALL iom_rstput( iter, nitrst, numriw, 'sysn'  , sysn   )
         CALL iom_rstput( iter, nitrst, numriw, 'sxxsn' , sxxsn  )
         CALL iom_rstput( iter, nitrst, numriw, 'syysn' , syysn  )
         CALL iom_rstput( iter, nitrst, numriw, 'sxysn' , sxysn  )
         !                                                           ! ice concentration
         CALL iom_rstput( iter, nitrst, numriw, 'sxa'   , sxa    )
         CALL iom_rstput( iter, nitrst, numriw, 'sya'   , sya    )
         CALL iom_rstput( iter, nitrst, numriw, 'sxxa'  , sxxa   )
         CALL iom_rstput( iter, nitrst, numriw, 'syya'  , syya   )
         CALL iom_rstput( iter, nitrst, numriw, 'sxya'  , sxya   )
         !                                                           ! ice age
         CALL iom_rstput( iter, nitrst, numriw, 'sxage' , sxage  )
         CALL iom_rstput( iter, nitrst, numriw, 'syage' , syage  )
         CALL iom_rstput( iter, nitrst, numriw, 'sxxage', sxxage )
         CALL iom_rstput( iter, nitrst, numriw, 'syyage', syyage )
         CALL iom_rstput( iter, nitrst, numriw, 'sxyage', sxyage )
         !                                                           ! snow layers heat content
         DO jk = 1, nlay_s
            WRITE(zchar1,'(I2.2)') jk
            znam = 'sxc0'//'_l'//zchar1  ;   z3d(:,:,:) = sxc0 (:,:,jk,:)
            CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
            znam = 'syc0'//'_l'//zchar1  ;   z3d(:,:,:) = syc0 (:,:,jk,:)
            CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
            znam = 'sxxc0'//'_l'//zchar1 ;   z3d(:,:,:) = sxxc0(:,:,jk,:)
            CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
            znam = 'syyc0'//'_l'//zchar1 ;   z3d(:,:,:) = syyc0(:,:,jk,:)
            CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
            znam = 'sxyc0'//'_l'//zchar1 ;   z3d(:,:,:) = sxyc0(:,:,jk,:)
            CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
         END DO
         !                                                           ! ice layers heat content
         DO jk = 1, nlay_i
            WRITE(zchar1,'(I2.2)') jk
            znam = 'sxe'//'_l'//zchar1   ;   z3d(:,:,:) = sxe (:,:,jk,:)
            CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
            znam = 'sye'//'_l'//zchar1   ;   z3d(:,:,:) = sye (:,:,jk,:)
            CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
            znam = 'sxxe'//'_l'//zchar1  ;   z3d(:,:,:) = sxxe(:,:,jk,:)
            CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
            znam = 'syye'//'_l'//zchar1  ;   z3d(:,:,:) = syye(:,:,jk,:)
            CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
            znam = 'sxye'//'_l'//zchar1  ;   z3d(:,:,:) = sxye(:,:,jk,:)
            CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
         END DO
         !
         ! --------------------- !
         ! == optional fields == !
         ! --------------------- !
         !
         IF( nn_icesal == 4 ) THEN
            !                                                        ! ice layers salt content
            DO jk = 1, nlay_i
               WRITE(zchar1,'(I2.2)') jk
               znam = 'sxsi'//'_l'//zchar1   ;   z3d(:,:,:) = sxsi (:,:,jk,:)
               CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
               znam = 'sysi'//'_l'//zchar1   ;   z3d(:,:,:) = sysi (:,:,jk,:)
               CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
               znam = 'sxxsi'//'_l'//zchar1  ;   z3d(:,:,:) = sxxsi(:,:,jk,:)
               CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
               znam = 'syysi'//'_l'//zchar1  ;   z3d(:,:,:) = syysi(:,:,jk,:)
               CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
               znam = 'sxysi'//'_l'//zchar1  ;   z3d(:,:,:) = sxysi(:,:,jk,:)
               CALL iom_rstput( iter, nitrst, numriw, znam , z3d )
            END DO
            !
         ELSE
            !                                                        ! ice salinity
            CALL iom_rstput( iter, nitrst, numriw, 'sxsal' , sxsal  )
            CALL iom_rstput( iter, nitrst, numriw, 'sysal' , sysal  )
            CALL iom_rstput( iter, nitrst, numriw, 'sxxsal', sxxsal )
            CALL iom_rstput( iter, nitrst, numriw, 'syysal', syysal )
            CALL iom_rstput( iter, nitrst, numriw, 'sxysal', sxysal )
            !
         ENDIF
         !
         IF( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN
            !                                                        ! melt pond fraction
            CALL iom_rstput( iter, nitrst, numriw, 'sxap' , sxap  )
            CALL iom_rstput( iter, nitrst, numriw, 'syap' , syap  )
            CALL iom_rstput( iter, nitrst, numriw, 'sxxap', sxxap )
            CALL iom_rstput( iter, nitrst, numriw, 'syyap', syyap )
            CALL iom_rstput( iter, nitrst, numriw, 'sxyap', sxyap )
            !                                                        ! melt pond volume
            CALL iom_rstput( iter, nitrst, numriw, 'sxvp' , sxvp  )
            CALL iom_rstput( iter, nitrst, numriw, 'syvp' , syvp  )
            CALL iom_rstput( iter, nitrst, numriw, 'sxxvp', sxxvp )
            CALL iom_rstput( iter, nitrst, numriw, 'syyvp', syyvp )
            CALL iom_rstput( iter, nitrst, numriw, 'sxyvp', sxyvp )
            !                                                        ! melt pond lid volume
            CALL iom_rstput( iter, nitrst, numriw, 'sxvl' , sxvl  )
            CALL iom_rstput( iter, nitrst, numriw, 'syvl' , syvl  )
            CALL iom_rstput( iter, nitrst, numriw, 'sxxvl', sxxvl )
            CALL iom_rstput( iter, nitrst, numriw, 'syyvl', syyvl )
            CALL iom_rstput( iter, nitrst, numriw, 'sxyvl', sxyvl )
            !
         ENDIF
         !
      ENDIF
      !
   END SUBROUTINE adv_pra_rst

   SUBROUTINE icemax2D_pra( ihls, pice , pmax )
      !!---------------------------------------------------------------------
      !!                   ***  ROUTINE icemax2D_pra ***
      !! ** Purpose :  compute the max of the 9 points around
      !!----------------------------------------------------------------------
      INTEGER                 , INTENT(in ) ::   ihls   ! loop index
      REAL(wp), DIMENSION(:,:), INTENT(in ) ::   pice   ! input
      REAL(wp), DIMENSION(:,:), INTENT(out) ::   pmax   ! output
      !
!!$      REAL(wp), DIMENSION(Nis0-ihls:Nie0+ihls) ::   zmax1, zmax2
      REAL(wp), DIMENSION(1:jpi) ::   zmax1, zmax2
      REAL(wp)                   ::   zmax3
      INTEGER  ::   ji, jj   ! dummy loop indices
      !!----------------------------------------------------------------------
      ! basic version: get the max of epsi20 + 9 neighbours
!!$      DO_2D( ihls, ihls, ihls, ihls )
!!$         pmax(ji,jj) = MAX( epsi20, pice(ji-1,jj-1), pice(ji,jj-1), pice(ji+1,jj-1), &
!!$            &                       pice(ji-1,jj  ), pice(ji,jj  ), pice(ji+1,jj  ), &
!!$            &                       pice(ji-1,jj+1), pice(ji,jj+1), pice(ji+1,jj+1) )
!!$      END_2D
      ! optimized version : does a little bit more than 2 max of epsi20 + 3 neighbours
      DO ji = Nis0-ihls, Nie0+ihls
         zmax1(ji) = MAX( epsi20, pice(ji,Njs0-1-ihls), pice(ji-1,Njs0-1-ihls), pice(ji+1,Njs0-1-ihls) )
         zmax2(ji) = MAX( epsi20, pice(ji,Njs0  -ihls), pice(ji-1,Njs0  -ihls), pice(ji+1,Njs0  -ihls) )
      END DO
      DO_2D( ihls, ihls, ihls, ihls )
         zmax3 = MAX( epsi20, pice(ji,jj+1), pice(ji-1,jj+1), pice(ji+1,jj+1) )
         pmax(ji,jj) = MAX( epsi20, zmax1(ji), zmax2(ji), zmax3 )
         zmax1(ji) = zmax2(ji)
         zmax2(ji) = zmax3
      END_2D
   END SUBROUTINE icemax2D_pra

   SUBROUTINE icemax3D_pra( ihls, pice , pmax )
      !!---------------------------------------------------------------------
      !!                   ***  ROUTINE icemax3D_pra ***
      !! ** Purpose :  compute the max of the 9 points around
      !!----------------------------------------------------------------------
      INTEGER                   , INTENT(in ) ::   ihls   ! loop index
      REAL(wp), DIMENSION(:,:,:), INTENT(in ) ::   pice   ! input
      REAL(wp), DIMENSION(:,:,:), INTENT(out) ::   pmax   ! output
      !
!!$      REAL(wp), DIMENSION(Nis0-ihls:Nie0+ihls) ::   zmax1, zmax2
      REAL(wp), DIMENSION(1:jpi) ::   zmax1, zmax2
      REAL(wp)                   ::   zmax3
      INTEGER  ::   jlay, ji, jj, jk   ! dummy loop indices
      !!----------------------------------------------------------------------
      jlay = SIZE( pice , 3 )   ! size of input arrays
      ! basic version: get the max of epsi20 + 9 neighbours
!!$      DO jk = 1, jlay
!!$         DO_2D( ihls, ihls, ihls, ihls )
!!$            pmax(ji,jj,jk) = MAX( epsi20, pice(ji-1,jj-1,jk), pice(ji,jj-1,jk), pice(ji+1,jj-1,jk), &
!!$               &                          pice(ji-1,jj  ,jk), pice(ji,jj  ,jk), pice(ji+1,jj  ,jk), &
!!$               &                          pice(ji-1,jj+1,jk), pice(ji,jj+1,jk), pice(ji+1,jj+1,jk) )
!!$         END_2D
!!$      END DO
      ! optimized version : does a little bit more than 2 max of epsi20 + 3 neighbours
      DO jk = 1, jlay
         DO ji = Nis0-ihls, Nie0+ihls
            zmax1(ji) = MAX( epsi20, pice(ji,Njs0-1-ihls,jk), pice(ji-1,Njs0-1-ihls,jk), pice(ji+1,Njs0-1-ihls,jk) )
            zmax2(ji) = MAX( epsi20, pice(ji,Njs0  -ihls,jk), pice(ji-1,Njs0  -ihls,jk), pice(ji+1,Njs0  -ihls,jk) )
         END DO
         DO_2D( ihls, ihls, ihls, ihls )
            zmax3 = MAX( epsi20, pice(ji,jj+1,jk), pice(ji-1,jj+1,jk), pice(ji+1,jj+1,jk) )
            pmax(ji,jj,jk) = MAX( epsi20, zmax1(ji), zmax2(ji), zmax3 )
            zmax1(ji) = zmax2(ji)
            zmax2(ji) = zmax3
         END_2D
      END DO
   END SUBROUTINE icemax3D_pra

#else
   !!----------------------------------------------------------------------
   !!   Default option            Dummy module        NO SI3 sea-ice model
   !!----------------------------------------------------------------------
#endif

   !!======================================================================
#include "l4_oracle_canon_subroutines.h90"

END MODULE icedyn_adv_pra
