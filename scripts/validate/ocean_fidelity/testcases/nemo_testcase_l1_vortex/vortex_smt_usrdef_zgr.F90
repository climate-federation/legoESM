MODULE usrdef_zgr
   !!======================================================================
   !!                       ***  MODULE  usrdef_zgr  ***
   !!
   !!                   ===  VORTEX_SMT configuration  ===
   !!
   !! User defined : vertical coordinate system of a user configuration
   !!======================================================================
   !! History :  4.0  ! 2017-11  (J. Chanut)  Original VORTEX code
   !!            5.0  ! 2026-10  legoESM NEMO-fidelity campaign, decision 88:
   !!                            the SAME VORTEX box with a Gaussian seamount
   !!                            and z-coordinate PARTIAL STEPS.
   !!----------------------------------------------------------------------
   !!
   !! PROVENANCE OF EVERY STATEMENT BELOW (NEMO 5.0.2, this tree).
   !!
   !!   * zgr_z (the 1-D reference coordinate) is tests/VORTEX/MY_SRC/
   !!     usrdef_zgr.F90:89-161, carried VERBATIM.  The 10 uniform 500 m
   !!     levels are unchanged.
   !!   * the land mask (k_top) is tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:
   !!     113-115 -- z2d = 1, lbc_lnk, k_top = NINT(z2d).  It reproduces
   !!     exactly the mask tests/VORTEX/MY_SRC/usrdef_zgr.F90:187-193 builds
   !!     (same lbc_lnk on a T-field of a closed box; VORTEX then takes
   !!     k_top = MIN(1,k_bot)).
   !!   * the bottom level and the minimum partial cell are
   !!     tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:202-212 (the lk_vco_1d3d
   !!     "zps-coordinate (partial bottom-steps)" branch):
   !!         ze3min = 0.1_wp * rn_dz
   !!         k_bot(:,:) = jpkm1
   !!         DO jk = jpkm1, 1, -1
   !!            WHERE( zht(:,:) < pdepw_1d(jk) + ze3min )  k_bot(:,:) = jk-1
   !!         END DO
   !!     with the land mask applied the way OVERFLOW's other branches apply
   !!     it (`k_bot = jk * k_top`, OVERFLOW usrdef_zgr.F90:143/160/197).
   !!   * the partial e3t is OVERFLOW usrdef_zgr.F90:221-225:
   !!         ik = k_bot(ji,jj)
   !!         pe3t(ji,jj,ik  ) = MIN( zht, pdepw_1d(ik+1) ) - pdepw_1d(ik)
   !!         pe3t(ji,jj,ik+1) = pe3t(ji,jj,ik)
   !!   * OVERFLOW then sets pe3u = pe3v = pe3f = pe3t, which it documents
   !!     as valid ONLY for its own 1-D-in-j bathymetry
   !!     (usrdef_zgr.F90:228-231: "HERE OVERFLOW configuration ... e3
   !!     increases with i-index and identical with j-index").  A SEAMOUNT
   !!     varies in both directions, so that shortcut is NOT portable and is
   !!     NOT taken.  NEMO's general zps rule for a two-dimensional
   !!     bathymetry is tools/DOMAINcfg/src/domzgr.F90::zgr_zps:1162-1170
   !!     and :1191-1197:
   !!         e3u_0(ji,jj,jk) = MIN( e3t_0(ji,jj,jk), e3t_0(ji+1,jj,jk) )
   !!         e3v_0(ji,jj,jk) = MIN( e3t_0(ji,jj,jk), e3t_0(ji,jj+1,jk) )
   !!         e3f_0(ji,jj,jk) = MIN( e3v_0(ji,jj,jk), e3v_0(ji+1,jj,jk) )
   !!     with lbc_lnk on e3u/e3v BEFORE e3f is formed (:1177-1178) and on
   !!     e3f after (:1198).  That is what this module transcribes.
   !!     DELIBERATELY NOT PORTED from zgr_zps, each with its reason
   !!     (round 2's adversarial review re-checked all four against source):
   !!      - the duplication of row jj=1 onto jj=2 (:1205-1209, flagged
   !!        "!!gm bug ?" in NEMO).  It is UNCONDITIONAL there, not
   !!        ORCA-specific -- an earlier version of this comment said
   !!        otherwise and was wrong.  Omitting it is still exact here
   !!        because kfillmode = jpfillcopy on the lbc_lnk above copies the
   !!        inner edge into the southern halo, which is the same operation.
   !!      - the ln_isfcav branches: every ice-shelf statement in zgr_zps is
   !!        inside IF(ln_isfcav), and there are no cavities here.
   !!      - the `WHERE(e3==0)` repair (:1180-1185, :1201-1203): unreachable,
   !!        because all four arrays are pre-filled from pe3t_1d below and
   !!        the only zero source in zgr_zps is its zero-filling lbc_lnk,
   !!        which jpfillcopy replaces.
   !!   * only e3t/e3u/e3v/e3f are 3-D: under key_vco_1d3d dom_zgr calls
   !!     usr_def_zgr with exactly those four optional arrays
   !!     (src/OCE/DOM/domzgr.F90:265-269) and applies lbc_lnk to all four
   !!     itself afterwards (:273-274).  e3w, gdept and gdepw stay 1-D.
   !!
   !! THE ONE cpp KEY THIS CONFIGURATION CHANGES, and why it is forced, not
   !! chosen: src/OCE/DOM/domzgr.F90:259 is
   !!     IF( l_zps ) CALL ctl_stop( 'STOP','domzgr: key_vco_1d and l_zps=T
   !!                                are incompatible. Fix usrdef_zgr !' )
   !! so partial steps CANNOT be run under the certified VORTEX cards'
   !! key_vco_1d.  NEMO's own name for the partial-cell key is key_vco_1d3d
   !! (domzgr.F90:164,:262 "z-partial cells"; OVERFLOW's zps branch is
   !! guarded by lk_vco_1d3d).  key_qco and key_RK3 are unchanged.
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   usr_def_zgr   : user defined vertical coordinate system
   !!      zgr_z      : reference 1D z-coordinate
   !!---------------------------------------------------------------------
   USE oce            ! ocean variables
   USE dom_oce        ! ocean domain
   USE depth_e3       ! depth <=> e3
   USE usrdef_nam     ! User defined : namelist variables (rn_dz)
   !
   USE in_out_manager ! I/O manager
   USE lbclnk         ! ocean lateral boundary conditions (or mpp link)
   USE lib_mpp        ! distributed memory computing library

   IMPLICIT NONE
   PRIVATE

   PUBLIC   usr_def_zgr        ! called by domzgr.F90

   ! --- Gaussian seamount, user decision 88 (2026-10-03) ---------------
   ! h(x,y) = H0 - A * exp( -((x-x0)^2 + (y-y0)^2) / L^2 )
   ! (x0,y0) is the vortex's initial centre shifted 300 km WEST.  The
   ! vortex is centred on (glamt,gphit) = (0,0) km: its temperature and
   ! velocity anomalies are exp(-(zx^2+zy^2)/zlambda^2) with zx =
   ! glamt*1.e3, zy = gphit*1.e3 (tests/VORTEX/MY_SRC/usrdef_istate.F90:
   ! 79-80 and :86, and :98-99/:105 for the velocity), so x0 = -300 km.
   ! glamt/gphit are in KILOMETRES in this configuration
   ! (tests/VORTEX/MY_SRC/usrdef_hgr.F90:78 "Position coordinates (in
   ! kilometers)"), and for nn_rot = 0 glamt increases with the i index
   ! (usrdef_hgr.F90:108), so WEST is the -glamt direction.
   REAL(wp), PARAMETER ::   pp_smt_H0 =  5000._wp    ! far-field depth           [m]
   REAL(wp), PARAMETER ::   pp_smt_A  =  1000._wp    ! seamount height           [m]
   REAL(wp), PARAMETER ::   pp_smt_L  =   150.e3_wp  ! Gaussian radius           [m]
   REAL(wp), PARAMETER ::   pp_smt_x0 =  -300.e3_wp  ! centre, i-direction       [m]
   REAL(wp), PARAMETER ::   pp_smt_y0 =     0.e3_wp  ! centre, j-direction       [m]

   !! * Substitutions
#  include "do_loop_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS             

   SUBROUTINE usr_def_zgr( ld_zco  , ld_zps  , ld_sco  , ld_isfcav,    &   ! type of vertical coordinate
      &                    k_top   , k_bot                        ,    &   ! top & bottom ocean level
      &                    pdept_1d, pdepw_1d, pe3t_1d , pe3w_1d  ,    &   ! 1D reference vertical coordinate
      &                    pe3t  , pe3u  , pe3v   , pe3f ,             &   ! vertical scale factors
      &                    pdept , pdepw ,                             &   ! 3D t & w-points depth
      &                    pe3w  , pe3uw , pe3vw                       )   ! vertical scale factors
      !!---------------------------------------------------------------------
      !!              ***  ROUTINE usr_def_zgr  ***
      !!
      !! ** Purpose :   User defined the vertical coordinates
      !!                VORTEX_SMT: Gaussian seamount, z-coordinate with
      !!                partial steps.
      !!----------------------------------------------------------------------
      LOGICAL                   , INTENT(out) ::   ld_zco, ld_zps, ld_sco      ! vertical coordinate flags
      LOGICAL                   , INTENT(out) ::   ld_isfcav                   ! under iceshelf cavity flag
      INTEGER , DIMENSION(:,:)  , INTENT(out) ::   k_top, k_bot                ! first & last ocean level
      REAL(wp), DIMENSION(:)    , INTENT(out) ::   pdept_1d, pdepw_1d          ! 1D grid-point depth     [m]
      REAL(wp), DIMENSION(:)    , INTENT(out) ::   pe3t_1d , pe3w_1d           ! 1D grid-point depth     [m]
      REAL(wp), DIMENSION(:,:,:), OPTIONAL, INTENT(out) ::   pdept, pdepw                ! grid-point depth        [m]
      REAL(wp), DIMENSION(:,:,:), OPTIONAL, INTENT(out) ::   pe3t , pe3u , pe3v , pe3f   ! vertical scale factors  [m]
      REAL(wp), DIMENSION(:,:,:), OPTIONAL, INTENT(out) ::   pe3w , pe3uw, pe3vw         ! i-scale factors 
      !
      INTEGER  ::   ji, jj, jk        ! dummy loop indices
      INTEGER  ::   ik                ! local integer
      INTEGER  ::   ikbmin, ikbmax    ! global k_bot extrema (control print)
      REAL(wp) ::   ze3min            ! local scalar
      REAL(wp) ::   zhtmin, zhtmax    ! global bathymetry extrema (control print)
      REAL(wp) ::   ze3min_g, ze3max_g ! global bottom-e3t extrema (control print)
      REAL(wp) ::   zx, zy            ! local scalars
      REAL(wp), DIMENSION(jpi,jpj) ::   zht, z2d   ! 2D workspace
      !!----------------------------------------------------------------------
      !
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'usr_def_zgr : VORTEX_SMT configuration (Gaussian seamount, z-partial-step closed box ocean)'
      IF(lwp) WRITE(numout,*) '~~~~~~~~~~~'
      !
      !
      ! type of vertical coordinate
      ! ---------------------------
      ld_zco    = .FALSE.        ! VORTEX_SMT case:  z-coordinate WITH partial steps, no ocean cavities
      ld_zps    = .TRUE.
      ld_sco    = .FALSE.
      ld_isfcav = .FALSE.
      !
      !
      ! Build the vertical coordinate system
      ! ------------------------------------
      CALL zgr_z( pdept_1d, pdepw_1d, pe3t_1d , pe3w_1d )   ! Reference z-coordinate system (VORTEX, unchanged)
      !
      !                       !==  UNmasked meter bathymetry  ==!
      !
      ! Gaussian seamount on an otherwise flat 5000 m bottom.  glamt/gphit
      ! are in kilometres (usrdef_hgr.F90:78), so they are converted to
      ! metres here exactly as usrdef_istate.F90:79-80 does.
      DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )
         zx = glamt(ji,jj) * 1.e3_wp - pp_smt_x0
         zy = gphit(ji,jj) * 1.e3_wp - pp_smt_y0
         zht(ji,jj) = pp_smt_H0 - pp_smt_A * EXP( - ( zx*zx + zy*zy ) / ( pp_smt_L * pp_smt_L ) )
      END_2D
      !
      !                       !==  top masked level bathymetry  ==!
      !
      ! no ocean cavities : top ocean level is ONE, except over land.  The
      ! basin is closed, so lbc_lnk zeroes the surrounding land points.
      z2d(:,:) = 1._wp                                   ! OVERFLOW usrdef_zgr.F90:113-115
      CALL lbc_lnk( 'usrdef_zgr', z2d, 'T', 1._wp )
      k_top(:,:) = NINT( z2d(:,:) )
      !
      !                       !==  zps-coordinate  ==!   (partial bottom-steps)
      !
      ze3min = 0.1_wp * rn_dz                            ! OVERFLOW usrdef_zgr.F90:204
      IF(lwp) WRITE(numout,*) '   minimum thickness of the partial cells = 10 % of e3 = ', ze3min
      !
      !                                !* bottom ocean compute from the depth of grid-points
      k_bot(:,:) = jpkm1                                 ! OVERFLOW usrdef_zgr.F90:209-212
      DO jk = jpkm1, 1, -1
         WHERE( zht(:,:) < pdepw_1d(jk) + ze3min )   k_bot(:,:) = jk-1
      END DO
      k_bot(:,:) = k_bot(:,:) * k_top(:,:)               ! land mask, OVERFLOW usrdef_zgr.F90:143/160/197
      !
      IF( PRESENT( pe3t ) ) THEN       !* 3-D t-level scale factors (key_vco_1d3d)
         !
         DO jk = 1, jpk                      ! initialization to the reference z-coordinate
            pe3t (:,:,jk) = pe3t_1d (jk)     ! OVERFLOW usrdef_zgr.F90:215-220
            pe3u (:,:,jk) = pe3t_1d (jk)
            pe3v (:,:,jk) = pe3t_1d (jk)
            pe3f (:,:,jk) = pe3t_1d (jk)
         END DO
         DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )
            ik = k_bot(ji,jj)                ! last wet level thickness
            IF( ik > 0 ) THEN                ! OVERFLOW usrdef_zgr.F90:221-225
               pe3t (ji,jj,ik  ) = MIN( zht(ji,jj) , pdepw_1d(ik+1) ) - pdepw_1d(ik)
               pe3t (ji,jj,ik+1) = pe3t (ji,jj,ik  )
            ENDIF
         END_2D
         !
         !                                   ! U-, V- and F-point scale factors:
         !                                   ! the minimum of the neighbouring T- (then V-) scale factors,
         !                                   ! tools/DOMAINcfg/src/domzgr.F90::zgr_zps:1162-1170, :1191-1197
         DO jk = 1, jpk
            DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls )
               pe3u(ji,jj,jk) = MIN( pe3t(ji,jj,jk), pe3t(ji+1,jj,jk) )
            END_2D
            DO_2D( nn_hls, nn_hls, nn_hls, nn_hls-1 )
               pe3v(ji,jj,jk) = MIN( pe3t(ji,jj,jk), pe3t(ji,jj+1,jk) )
            END_2D
         END DO
         CALL lbc_lnk( 'usrdef_zgr', pe3u, 'U', 1._wp, pe3v, 'V', 1._wp,   &
            &          kfillmode = jpfillcopy )          ! zgr_zps:1177-1178
         DO jk = 1, jpk
            DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls )
               pe3f(ji,jj,jk) = MIN( pe3v(ji,jj,jk), pe3v(ji+1,jj,jk) )
            END_2D
         END DO
         CALL lbc_lnk( 'usrdef_zgr', pe3f, 'F', 1._wp, kfillmode = jpfillcopy )   ! zgr_zps:1198
         !
         ! Control print: the seamount as the model actually resolved it.
         ! The reductions are GLOBAL and therefore COLLECTIVE, so they are
         ! taken by every rank BEFORE the IF(lwp) -- a reduction inside
         ! IF(lwp) is either rank-0-local (and mislabelled) or a deadlock.
         ! Round 2's adversarial review found the first of those here.
         ! They run over the INTERIOR window only: the outer global halo
         ! carries k_top = 0 after the lbc_lnk above, so including it would
         ! report k_bot min = 0 on every decomposition.
         zhtmin = MINVAL( zht(Nis0:Nie0,Njs0:Nje0) )   ;   CALL mpp_min( 'usrdef_zgr', zhtmin )
         zhtmax = MAXVAL( zht(Nis0:Nie0,Njs0:Nje0) )   ;   CALL mpp_max( 'usrdef_zgr', zhtmax )
         ikbmin = MINVAL( k_bot(Nis0:Nie0,Njs0:Nje0) ) ;   CALL mpp_min( 'usrdef_zgr', ikbmin )
         ikbmax = MAXVAL( k_bot(Nis0:Nie0,Njs0:Nje0) ) ;   CALL mpp_max( 'usrdef_zgr', ikbmax )
         ze3min_g = MINVAL( pe3t(Nis0:Nie0,Njs0:Nje0,1:jpkm1) )
         CALL mpp_min( 'usrdef_zgr', ze3min_g )
         ze3max_g = MAXVAL( pe3t(Nis0:Nie0,Njs0:Nje0,1:jpkm1) )
         CALL mpp_max( 'usrdef_zgr', ze3max_g )
         IF(lwp) THEN
            WRITE(numout,*)
            WRITE(numout,*) '    VORTEX_SMT seamount, as resolved GLOBALLY (mpp reductions):'
            WRITE(numout,"(10x,'H0 =',f9.2,'  A =',f9.2,'  L =',f10.1,'  x0 =',f10.1,'  y0 =',f10.1,'  [m]')")   &
               &            pp_smt_H0, pp_smt_A, pp_smt_L, pp_smt_x0, pp_smt_y0
            WRITE(numout,"(10x,'bathymetry  min =',f10.3,'  max =',f10.3,' m')") zhtmin, zhtmax
            WRITE(numout,"(10x,'k_bot       min =',i4,'  max =',i4)") ikbmin, ikbmax
            WRITE(numout,"(10x,'bottom e3t  min =',f10.4,'  max =',f10.4,' m')") ze3min_g, ze3max_g
            WRITE(numout,*) '      i-row through the seamount centre, RANK-0 SUBDOMAIN ONLY'
            WRITE(numout,*) '      (a local MINLOC; on a j-split the seamount row may sit on another rank):'
            jj = MINLOC( MINVAL( zht(:,:), DIM=1 ), DIM=1 )
            WRITE(numout,"(10x,' ji   glamt[km]   zht[m]   k_bot   e3t(k_bot)[m]   e3u(k_bot)[m]')")
            DO ji = 1, jpi
               ik = MAX( k_bot(ji,jj), 1 )
               WRITE(numout,"(10x,i4,f11.2,f10.3,i7,f15.4,f16.4)")                                                &
                  &         ji, glamt(ji,jj), zht(ji,jj), k_bot(ji,jj), pe3t(ji,jj,ik), pe3u(ji,jj,ik)
            END DO
         ENDIF
         !
      ENDIF
      !
   END SUBROUTINE usr_def_zgr


   SUBROUTINE zgr_z( pdept_1d, pdepw_1d, pe3t_1d , pe3w_1d )   ! 1D reference vertical coordinate
      !!----------------------------------------------------------------------
      !!                   ***  ROUTINE zgr_z  ***
      !!
      !! ** Purpose :   set the 1D depth of model levels and the resulting 
      !!              vertical scale factors.
      !!
      !!      VERBATIM from tests/VORTEX/MY_SRC/usrdef_zgr.F90:89-161.
      !!
      !! Reference : Marti, Madec & Delecluse, 1992, JGR, 97, No8, 12,763-12,766.
      !!             Madec and Imbard, 1996, Clim. Dyn.
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(:)    , INTENT(out) ::   pdept_1d, pdepw_1d   ! 1D grid-point depth        [m]
      REAL(wp), DIMENSION(:)    , INTENT(out) ::   pe3t_1d , pe3w_1d    ! 1D vertical scale factors  [m]
      !
      INTEGER  ::   jk       ! dummy loop indices
      REAL(wp) ::   zd       ! local scalar
      !!----------------------------------------------------------------------
      !
      zd = 5000._wp/REAL(jpkm1,wp)
      !
      IF(lwp) THEN            ! Parameter print
         WRITE(numout,*)
         WRITE(numout,*) '    zgr_z   : Reference vertical z-coordinates '
         WRITE(numout,*) '    ~~~~~~~'
         WRITE(numout,*) '       VORTEX case : uniform vertical grid :'
         WRITE(numout,*) '                     with thickness = ', zd
      ENDIF

      !
      ! 1D Reference z-coordinate    (using Madec & Imbard 1996 function)
      ! -------------------------
      !
      pdepw_1d(1) = 0._wp
      pdept_1d(1) = 0.5_wp * zd
      ! 
      DO jk = 2, jpk          ! depth at T and W-points
         pdepw_1d(jk) = pdepw_1d(jk-1) + zd 
         pdept_1d(jk) = pdept_1d(jk-1) + zd 
      END DO
      !
      !                       ! e3t and e3w from depth
      CALL depth_to_e3( pdept_1d, pdepw_1d, pe3t_1d, pe3w_1d ) 
      !
      !                       ! recompute depths from SUM(e3)  <== needed
      CALL e3_to_depth( pe3t_1d, pe3w_1d, pdept_1d, pdepw_1d ) 
      !
      IF(lwp) THEN                        ! control print
         WRITE(numout,*)
         WRITE(numout,*) '              Reference 1D z-coordinate depth and scale factors:'
         WRITE(numout, "(9x,' level  gdept_1d  gdepw_1d  e3t_1d   e3w_1d  ')" )
         WRITE(numout, "(10x, i4, 4f9.2)" ) ( jk, pdept_1d(jk), pdepw_1d(jk), pe3t_1d(jk), pe3w_1d(jk), jk = 1, jpk )
      ENDIF
      !
   END SUBROUTINE zgr_z

   !!======================================================================
END MODULE usrdef_zgr
