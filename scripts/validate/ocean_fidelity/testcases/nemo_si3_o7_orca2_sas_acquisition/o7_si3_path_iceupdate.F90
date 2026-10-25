! Fixture implementation of the post-flux update.  Its ice_alb call is the
! real second ice_alb path made by ice_update_flx in the production program.
MODULE iceupdate
   USE par_ice, ONLY: ln_pnd_alb
   USE ice, ONLY: t_su,h_i,h_s,a_ip_eff,h_ip,h_il
   USE sbc_oce, ONLY: cloud_fra
   USE sbc_ice, ONLY: alb_ice
   USE icealb, ONLY: ice_alb
   IMPLICIT NONE
CONTAINS
   SUBROUTINE ice_update_flx(kt)
      INTEGER,INTENT(in)::kt
      CALL ice_alb(ln_pnd_alb,t_su,h_i,h_s,a_ip_eff,h_ip,h_il,cloud_fra,alb_ice)
   END SUBROUTINE
   SUBROUTINE ice_update_tau(kt,u,v)
      INTEGER,INTENT(in)::kt
      REAL(8),INTENT(in)::u(:,:),v(:,:)
   END SUBROUTINE
   SUBROUTINE ice_update_init
   END SUBROUTINE
END MODULE iceupdate
