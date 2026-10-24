MODULE par_oce
   USE, INTRINSIC :: iso_fortran_env, ONLY : real64
   IMPLICIT NONE
   PUBLIC
   INTEGER, PARAMETER :: wp = real64
   INTEGER :: jpi = 10, jpj = 9, jpk = 2
   INTEGER :: nn_hls = 2
   INTEGER :: Ni0glo = 12, Nj0glo = 10
   INTEGER :: Nis0 = 3, Nie0 = 8, Njs0 = 3, Nje0 = 7
END MODULE par_oce

MODULE dom_oce
   USE par_oce
   IMPLICIT NONE
   PUBLIC
   INTEGER :: nimpp = 1, njmpp = 1
   REAL(wp) :: rn_Dt = 10800.0_wp
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: tmask, umask, vmask
END MODULE dom_oce

MODULE phycst
   USE par_oce
   IMPLICIT NONE
   PUBLIC
   REAL(wp), PARAMETER :: rt0 = 273.15_wp
END MODULE phycst

MODULE sbc_oce
   USE par_oce
   IMPLICIT NONE
   PUBLIC
   INTEGER :: nn_fsbc = 1
   LOGICAL :: ln_cpl = .FALSE.
   REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: sst_m, sss_m
END MODULE sbc_oce

MODULE sbc_ice
   USE par_oce
   IMPLICIT NONE
   PUBLIC
   REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: snwice_mass, snwice_mass_b, snwice_fmass
END MODULE sbc_ice

MODULE par_ice
   USE par_oce
   IMPLICIT NONE
   PUBLIC
   INTEGER :: jpl = 5
   INTEGER :: nlay_i = 10
   INTEGER :: nlay_s = 5
END MODULE par_ice

MODULE ice
   USE par_oce
   USE par_ice
   IMPLICIT NONE
   PUBLIC
   INTEGER :: nn_icesal = 4
   INTEGER :: nn_snwfra = 2
   LOGICAL :: ln_pnd = .TRUE.
   LOGICAL :: ln_pnd_LEV = .TRUE.
   LOGICAL :: ln_pnd_TOPO = .FALSE.
   LOGICAL :: ln_pnd_lids = .TRUE.
   LOGICAL :: lrst_ice = .FALSE.
   REAL(wp) :: rDt_ice = 10800.0_wp
   REAL(wp), ALLOCATABLE, DIMENSION(:) :: hi_max, hi_mean
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: v_i, v_s, sv_i, a_i, t_su
   REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: u_ice, v_ice
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: oa_i, a_ip, v_ip, v_il
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:,:) :: e_s, e_i, szv_i
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: cnd_ice, t1_ice, t_si
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: h_i, h_s, o_i, h_ip, h_il, a_ip_frac, a_ip_eff, s_i
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:,:) :: sz_i, t_i, t_s
   REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: vt_i, vt_s, at_i, at_ip, vt_ip, vt_il, ato_i
   REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: et_s, et_i, at_ip_eff, tm_su, st_i
   REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: hm_i, hm_s, tm_si, om_i, sm_i, tm_i, tm_s, hm_ip, hm_il
END MODULE ice

MODULE icedyn_adv_pra
   USE par_oce
   USE par_ice
   IMPLICIT NONE
   PUBLIC
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: sxice, syice, sxxice, syyice, sxyice
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: sxsn, sysn, sxxsn, syysn, sxysn
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: sxa, sya, sxxa, syya, sxya
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: sxage, syage, sxxage, syyage, sxyage
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:,:) :: sxc0, syc0, sxxc0, syyc0, sxyc0
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:,:) :: sxe, sye, sxxe, syye, sxye
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: sxsal, sysal, sxxsal, syysal, sxysal
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:,:) :: sxsi, sysi, sxxsi, syysi, sxysi
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: sxap, syap, sxxap, syyap, sxyap
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: sxvp, syvp, sxxvp, syyvp, sxyvp
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: sxvl, syvl, sxxvl, syyvl, sxyvl
END MODULE icedyn_adv_pra

MODULE in_out_manager
   IMPLICIT NONE
   PUBLIC
   LOGICAL :: lwp = .TRUE.
   INTEGER :: numout = 6
CONTAINS
   SUBROUTINE ctl_stop( cd1, cd2 )
      CHARACTER(len=*), INTENT(in) :: cd1, cd2
      WRITE(*,*) 'STOP: ', TRIM(cd1), ' ', TRIM(cd2)
      ERROR STOP 1
   END SUBROUTINE ctl_stop
END MODULE in_out_manager

MODULE lib_mpp
   IMPLICIT NONE
   PUBLIC
   INTEGER :: mpprank = 0
   INTEGER :: narea = 1
END MODULE lib_mpp
