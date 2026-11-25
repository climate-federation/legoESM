MODULE usrdef_istate
   !!======================================================================
   !!                     ***  MODULE usrdef_istate   ***
   !!
   !!            === ORCA2G1 (grid ladder rung G1) configuration ===
   !!
   !! tests/TSUNAMI/MY_SRC/usrdef_istate.F90 adapted to ORCA2's coordinates:
   !! uniform T = 20, S = 30, u = v = 0 (TSUNAMI :65-68, verbatim) and
   !! TSUNAMI's ssh bump (TSUNAMI :94-102: amplitude 0.1 m, radius
   !! MAXVAL(zdist)/20, quarter-cosine profile) centred on the ORCA2 T point
   !! (ji,jj) = (np_g1_ji0, np_g1_jj0) of the 180 x 148 inner global domain.
   !! Distance is in degrees, with the longitude difference wrapped into
   !! [-180,180]; TSUNAMI's own distance is from the origin of its
   !! Cartesian (km) coordinates.
   !!======================================================================
   USE par_oce        ! ocean space and time domain
   USE dom_oce
   USE phycst         ! physical constants
   !
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! MPP library
   USE lbclnk         ! lateral boundary conditions - mpp exchanges

   IMPLICIT NONE
   PRIVATE

   PUBLIC   usr_def_istate       ! called by istate.F90
   PUBLIC   usr_def_istate_ssh   ! called by restart.F90 (rst_read_ssh)

   INTEGER, PARAMETER ::   np_g1_ji0 = 51   ! bump centre, inner global i (1..180): 180 E
   INTEGER, PARAMETER ::   np_g1_jj0 = 74   ! bump centre, inner global j (1..148): equator

   !! * Substitutions
#  include "do_loop_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE usr_def_istate( pdept, ptmask, pts, pu, pv )
      !!----------------------------------------------------------------------
      !!                   ***  ROUTINE usr_def_istate  ***
      !!
      !! ** Purpose :   uniform T/S, ocean at rest (TSUNAMI's statements)
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj,jpk)     , INTENT(in   ) ::   pdept   ! depth of t-point               [m]
      REAL(wp), DIMENSION(jpi,jpj,jpk)     , INTENT(in   ) ::   ptmask  ! t-point ocean mask             [m]
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(  out) ::   pts     ! T & S fields      [Celsius ; g/kg]
      REAL(wp), DIMENSION(jpi,jpj,jpk)     , INTENT(  out) ::   pu      ! i-component of the velocity  [m/s]
      REAL(wp), DIMENSION(jpi,jpj,jpk)     , INTENT(  out) ::   pv      ! j-component of the velocity  [m/s]
      !!----------------------------------------------------------------------
      !
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'usr_def_istate : ORCA2G1 configuration, uniform T/S, ocean at rest'
      IF(lwp) WRITE(numout,*) '~~~~~~~~~~~~~~   '
      !
      pts(:,:,:,jp_tem) = 20._wp
      pts(:,:,:,jp_sal) = 30._wp
      pu( :,:,:       ) =  0._wp
      pv( :,:,:       ) =  0._wp
      !
   END SUBROUTINE usr_def_istate


   SUBROUTINE usr_def_istate_ssh( ptmask, pssh )
      !!----------------------------------------------------------------------
      !!                   ***  ROUTINE usr_def_istate_ssh  ***
      !!
      !! ** Purpose :   TSUNAMI's ssh bump centred on (np_g1_ji0, np_g1_jj0)
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj,jpk)     , INTENT(in   ) ::   ptmask  ! t-point ocean mask   [m]
      REAL(wp), DIMENSION(jpi,jpj)         , INTENT(  out) ::   pssh    ! sea-surface height   [m]
      !
      INTEGER  ::   ji, jj
      REAL(wp), DIMENSION(jpi,jpj) ::  zdist
      REAL(wp)                     ::  zmax, zlam0, zphi0, zdlam
      !!----------------------------------------------------------------------
      !
      !                                   ! centre coordinates, known on every rank
      zlam0 = -HUGE(1._wp)   ;   zphi0 = -HUGE(1._wp)
      DO jj = mj0(np_g1_jj0+nn_hls,nn_hls), mj1(np_g1_jj0+nn_hls,nn_hls)
         DO ji = mi0(np_g1_ji0+nn_hls,nn_hls), mi1(np_g1_ji0+nn_hls,nn_hls)
            zlam0 = glamt(ji,jj)
            zphi0 = gphit(ji,jj)
         END DO
      END DO
      CALL mpp_max( 'usrdef_istate', zlam0 )
      CALL mpp_max( 'usrdef_istate', zphi0 )
      !
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'usr_def_istate_ssh : ORCA2G1 configuration, TSUNAMI ssh bump'
      IF(lwp) WRITE(numout,*) '   G1_BUMP_CENTRE ji, jj, glamt, gphit = ', np_g1_ji0, np_g1_jj0, zlam0, zphi0
      !
      DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )
         zdlam = glamt(ji,jj) - zlam0
         zdlam = zdlam - 360._wp * ANINT( zdlam / 360._wp )
         zdist(ji,jj) = SQRT( zdlam**2 + ( gphit(ji,jj) - zphi0 )**2 )
      END_2D
      zmax = MAXVAL( zdist ) / 20._wp
      CALL mpp_max( 'usrdef_istate', zmax )
      IF(lwp) WRITE(numout,*) '   G1_BUMP_RADIUS_DEG = ', zmax
      !
      pssh(:,:) = 0._wp
      DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )
         IF( zdist(ji,jj) <= zmax )   pssh(ji,jj) = 0.1 * COS( zdist(ji,jj) / zmax * rpi * 0.5_wp ) * ptmask(ji,jj,1)
      END_2D
      !
      CALL lbc_lnk('usrdef_istate', pssh, 'T',  1._wp )            ! apply boundary conditions
      !
   END SUBROUTINE usr_def_istate_ssh

   !!======================================================================
END MODULE usrdef_istate
