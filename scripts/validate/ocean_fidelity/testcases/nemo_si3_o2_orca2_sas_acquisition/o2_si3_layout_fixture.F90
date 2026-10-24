PROGRAM o2_si3_layout_fixture
   !!======================================================================
   !! Compiled gfortran multi-rank layout fixture for SI3 rung O2.
   !! Links against the real o2_si3_state_writer.F90 and writes 4 MPI rank
   !! files for both O2SI3I01 (istate) and O1SI3R01 (entry and step 1).
   !!======================================================================
   USE, INTRINSIC :: iso_fortran_env, ONLY : int32, real64
   USE par_oce
   USE dom_oce
   USE oce
   USE phycst
   USE sbc_oce
   USE sbc_ice
   USE par_ice
   USE ice
   USE icedyn_adv_pra
   USE lib_mpp
   USE o2_si3_state_writer, ONLY : o2_record_istate_scalars, o2_write_istate_snapshot, o2_write_si3_state_snapshot

   IMPLICIT NONE

   INTEGER :: irank, rx, ry
   REAL(wp) :: rbase
   REAL(wp), DIMENSION(:,:), ALLOCATABLE :: ztfrz, zmsk, zht_i_ini, zht_s_ini, zat_i_ini
   REAL(wp), DIMENSION(:,:), ALLOCATABLE :: zsm_i_ini, ztm_i_ini, zt_su_ini, ztm_s_ini
   REAL(wp), DIMENSION(:,:), ALLOCATABLE :: zapnd_ini, zhpnd_ini, zhlid_ini

   jpi = 10
   jpj = 9
   jpk = 2
   jpt = 3
   nn_hls = 2
   Ni0glo = 12
   Nj0glo = 10
   Nis0 = 3
   Nie0 = 8
   Njs0 = 3
   Nje0 = 7
   jpl = 5
   nlay_i = 10
   nlay_s = 5

   ALLOCATE( hi_max(0:jpl), hi_mean(1:jpl) )
   hi_max(0:5) = (/ 0.0_wp, 0.45401674241555168_wp, 1.1293138230836322_wp, &
      &             2.1414589889479725_wp, 3.6705964099155439_wp, 99.0_wp /)
   hi_mean(1:5) = (/ 0.22700837120777584_wp, 0.79166528274959191_wp, &
      &              1.6353864060158023_wp, 2.9060276994317582_wp, 4.8352982049577680_wp /)

   ALLOCATE( tmask(jpi,jpj,jpk), umask(jpi,jpj,jpk), vmask(jpi,jpj,jpk) )
   ALLOCATE( ssmask(jpi,jpj), smask0(Nis0:Nie0,Njs0:Nje0), smask0_i(Nis0:Nie0,Njs0:Nje0) )
   ALLOCATE( e1e2t(jpi,jpj), ff_t(jpi,jpj), gphit(jpi,jpj), ssh(jpi,jpj,jpt) )
   ALLOCATE( sst_m(jpi,jpj), sss_m(jpi,jpj) )
   ALLOCATE( snwice_mass(jpi,jpj), snwice_mass_b(jpi,jpj), snwice_fmass(jpi,jpj) )
   ALLOCATE( t_bo(Nis0:Nie0,Njs0:Nje0), rn_amax_2d(jpi,jpj) )

   ALLOCATE( ztfrz(jpi,jpj), zmsk(jpi,jpj), zht_i_ini(jpi,jpj), zht_s_ini(jpi,jpj), zat_i_ini(jpi,jpj) )
   ALLOCATE( zsm_i_ini(jpi,jpj), ztm_i_ini(jpi,jpj), zt_su_ini(jpi,jpj), ztm_s_ini(jpi,jpj) )
   ALLOCATE( zapnd_ini(jpi,jpj), zhpnd_ini(jpi,jpj), zhlid_ini(jpi,jpj) )

   ALLOCATE( v_i(jpi,jpj,jpl), v_s(jpi,jpj,jpl), sv_i(jpi,jpj,jpl), a_i(jpi,jpj,jpl), t_su(jpi,jpj,jpl) )
   ALLOCATE( u_ice(jpi,jpj), v_ice(jpi,jpj) )
   ALLOCATE( oa_i(jpi,jpj,jpl), a_ip(jpi,jpj,jpl), v_ip(jpi,jpj,jpl), v_il(jpi,jpj,jpl) )
   ALLOCATE( e_s(jpi,jpj,nlay_s,jpl), e_i(jpi,jpj,nlay_i,jpl), szv_i(jpi,jpj,nlay_i,jpl) )
   ALLOCATE( cnd_ice(Nis0:Nie0,Njs0:Nje0,jpl), t1_ice(Nis0:Nie0,Njs0:Nje0,jpl), t_si(Nis0:Nie0,Njs0:Nje0,jpl) )

   ALLOCATE( sxice(jpi,jpj,jpl), syice(jpi,jpj,jpl), sxxice(jpi,jpj,jpl), syyice(jpi,jpj,jpl), sxyice(jpi,jpj,jpl) )
   ALLOCATE( sxsn(jpi,jpj,jpl), sysn(jpi,jpj,jpl), sxxsn(jpi,jpj,jpl), syysn(jpi,jpj,jpl), sxysn(jpi,jpj,jpl) )
   ALLOCATE( sxa(jpi,jpj,jpl), sya(jpi,jpj,jpl), sxxa(jpi,jpj,jpl), syya(jpi,jpj,jpl), sxya(jpi,jpj,jpl) )
   ALLOCATE( sxage(jpi,jpj,jpl), syage(jpi,jpj,jpl), sxxage(jpi,jpj,jpl), syyage(jpi,jpj,jpl), sxyage(jpi,jpj,jpl) )
   ALLOCATE( sxc0(jpi,jpj,nlay_s,jpl), syc0(jpi,jpj,nlay_s,jpl), sxxc0(jpi,jpj,nlay_s,jpl), syyc0(jpi,jpj,nlay_s,jpl), sxyc0(jpi,jpj,nlay_s,jpl) )
   ALLOCATE( sxe(jpi,jpj,nlay_i,jpl), sye(jpi,jpj,nlay_i,jpl), sxxe(jpi,jpj,nlay_i,jpl), syye(jpi,jpj,nlay_i,jpl), sxye(jpi,jpj,nlay_i,jpl) )
   ALLOCATE( sxsal(jpi,jpj,jpl), sysal(jpi,jpj,jpl), sxxsal(jpi,jpj,jpl), syysal(jpi,jpj,jpl), sxysal(jpi,jpj,jpl) )
   ALLOCATE( sxsi(jpi,jpj,nlay_i,jpl), sysi(jpi,jpj,nlay_i,jpl), sxxsi(jpi,jpj,nlay_i,jpl), syysi(jpi,jpj,nlay_i,jpl), sxysi(jpi,jpj,nlay_i,jpl) )
   ALLOCATE( sxap(jpi,jpj,jpl), syap(jpi,jpj,jpl), sxxap(jpi,jpj,jpl), syyap(jpi,jpj,jpl), sxyap(jpi,jpj,jpl) )
   ALLOCATE( sxvp(jpi,jpj,jpl), syvp(jpi,jpj,jpl), sxxvp(jpi,jpj,jpl), syyvp(jpi,jpj,jpl), sxyvp(jpi,jpj,jpl) )
   ALLOCATE( sxvl(jpi,jpj,jpl), syvl(jpi,jpj,jpl), sxxvl(jpi,jpj,jpl), syyvl(jpi,jpj,jpl), sxyvl(jpi,jpj,jpl) )

   ALLOCATE( h_i(jpi,jpj,jpl), h_s(jpi,jpj,jpl), o_i(jpi,jpj,jpl), h_ip(jpi,jpj,jpl), h_il(jpi,jpj,jpl) )
   ALLOCATE( a_ip_frac(Nis0:Nie0,Njs0:Nje0,jpl), a_ip_eff(Nis0:Nie0,Njs0:Nje0,jpl), s_i(jpi,jpj,jpl) )
   ALLOCATE( sz_i(jpi,jpj,nlay_i,jpl), t_i(jpi,jpj,nlay_i,jpl), t_s(jpi,jpj,nlay_s,jpl) )

   ALLOCATE( vt_i(jpi,jpj), vt_s(jpi,jpj), at_i(jpi,jpj), at_ip(jpi,jpj), vt_ip(jpi,jpj), vt_il(jpi,jpj), ato_i(jpi,jpj) )
   ALLOCATE( et_s(Nis0:Nie0,Njs0:Nje0), et_i(Nis0:Nie0,Njs0:Nje0), at_ip_eff(Nis0:Nie0,Njs0:Nje0), tm_su(Nis0:Nie0,Njs0:Nje0), st_i(Nis0:Nie0,Njs0:Nje0) )
   ALLOCATE( hm_i(Nis0:Nie0,Njs0:Nje0), hm_s(Nis0:Nie0,Njs0:Nje0), tm_si(Nis0:Nie0,Njs0:Nje0), om_i(Nis0:Nie0,Njs0:Nje0), sm_i(Nis0:Nie0,Njs0:Nje0) )
   ALLOCATE( tm_i(Nis0:Nie0,Njs0:Nje0), tm_s(Nis0:Nie0,Njs0:Nje0), hm_ip(Nis0:Nie0,Njs0:Nje0), hm_il(Nis0:Nie0,Njs0:Nje0) )

   DO irank = 0, 3
      mpprank = irank
      narea = irank + 1
      rx = MOD(irank, 2)
      ry = irank / 2
      nimpp = 1 + rx * 6
      njmpp = 1 + ry * 5
      rbase = 10000.0_wp * REAL(irank + 1, wp)

      CALL fill_3d(tmask, rbase + 1.0_wp)
      CALL fill_3d(umask, rbase + 2.0_wp)
      CALL fill_3d(vmask, rbase + 3.0_wp)
      CALL fill_2d(sst_m, rbase + 4.0_wp)
      CALL fill_2d(sss_m, rbase + 5.0_wp)
      CALL fill_2d(ssmask, rbase + 6.0_wp)
      CALL fill_a2d_2d(smask0, rbase + 6.25_wp)
      CALL fill_a2d_2d(smask0_i, rbase + 6.5_wp)
      CALL fill_2d(e1e2t, rbase + 7.0_wp)
      CALL fill_2d(ff_t, rbase + 7.5_wp)
      CALL fill_2d(gphit, rbase + 8.0_wp)
      CALL fill_3d(ssh, rbase + 8.5_wp)
      CALL fill_a2d_2d(t_bo, rbase + 9.0_wp)
      CALL fill_2d(rn_amax_2d, rbase + 9.5_wp)

      CALL fill_2d(ztfrz,     rbase + 140.0_wp)
      CALL fill_2d(zmsk,      rbase + 141.0_wp)
      CALL fill_2d(zht_i_ini, rbase + 142.0_wp)
      CALL fill_2d(zht_s_ini, rbase + 143.0_wp)
      CALL fill_2d(zat_i_ini, rbase + 144.0_wp)
      CALL fill_2d(zsm_i_ini, rbase + 145.0_wp)
      CALL fill_2d(ztm_i_ini, rbase + 146.0_wp)
      CALL fill_2d(zt_su_ini, rbase + 147.0_wp)
      CALL fill_2d(ztm_s_ini, rbase + 148.0_wp)
      CALL fill_2d(zapnd_ini, rbase + 149.0_wp)
      CALL fill_2d(zhpnd_ini, rbase + 150.0_wp)
      CALL fill_2d(zhlid_ini, rbase + 151.0_wp)

      CALL o2_record_istate_scalars( 1, 1, 2, ztfrz, zmsk, zht_i_ini, zht_s_ini, zat_i_ini, &
         &                           zsm_i_ini, ztm_i_ini, zt_su_ini, ztm_s_ini, zapnd_ini, zhpnd_ini, zhlid_ini )

      ! Update ssh after recording pre-istate ssh_before_kbb/kmm
      CALL fill_3d(ssh, rbase + 160.0_wp)

      CALL fill_3d(v_i,  rbase + 10.0_wp)
      CALL fill_3d(v_s,  rbase + 11.0_wp)
      CALL fill_3d(sv_i, rbase + 12.0_wp)
      CALL fill_3d(a_i,  rbase + 13.0_wp)
      CALL fill_3d(t_su, rbase + 14.0_wp)
      CALL fill_2d(u_ice, rbase + 15.0_wp)
      CALL fill_2d(v_ice, rbase + 16.0_wp)
      CALL fill_3d(oa_i, rbase + 17.0_wp)
      CALL fill_3d(a_ip, rbase + 18.0_wp)
      CALL fill_3d(v_ip, rbase + 19.0_wp)
      CALL fill_3d(v_il, rbase + 20.0_wp)
      CALL fill_4d(e_s,   rbase + 21.0_wp)
      CALL fill_4d(e_i,   rbase + 22.0_wp)
      CALL fill_4d(szv_i, rbase + 23.0_wp)
      CALL fill_a2d_3d(cnd_ice, rbase + 24.0_wp)
      CALL fill_a2d_3d(t1_ice,  rbase + 25.0_wp)
      CALL fill_a2d_3d(t_si,    rbase + 25.5_wp)
      CALL fill_2d(snwice_mass,   rbase + 26.0_wp)
      CALL fill_2d(snwice_mass_b, rbase + 27.0_wp)
      CALL fill_2d(snwice_fmass,  rbase + 28.0_wp)

      CALL fill_3d(sxice,  rbase + 30.0_wp)
      CALL fill_3d(syice,  rbase + 31.0_wp)
      CALL fill_3d(sxxice, rbase + 32.0_wp)
      CALL fill_3d(syyice, rbase + 33.0_wp)
      CALL fill_3d(sxyice, rbase + 34.0_wp)
      CALL fill_3d(sxsn,   rbase + 35.0_wp)
      CALL fill_3d(sysn,   rbase + 36.0_wp)
      CALL fill_3d(sxxsn,  rbase + 37.0_wp)
      CALL fill_3d(syysn,  rbase + 38.0_wp)
      CALL fill_3d(sxysn,  rbase + 39.0_wp)
      CALL fill_3d(sxa,    rbase + 40.0_wp)
      CALL fill_3d(sya,    rbase + 41.0_wp)
      CALL fill_3d(sxxa,   rbase + 42.0_wp)
      CALL fill_3d(syya,   rbase + 43.0_wp)
      CALL fill_3d(sxya,   rbase + 44.0_wp)
      CALL fill_3d(sxage,  rbase + 45.0_wp)
      CALL fill_3d(syage,  rbase + 46.0_wp)
      CALL fill_3d(sxxage, rbase + 47.0_wp)
      CALL fill_3d(syyage, rbase + 48.0_wp)
      CALL fill_3d(sxyage, rbase + 49.0_wp)
      CALL fill_4d(sxc0,   rbase + 50.0_wp)
      CALL fill_4d(syc0,   rbase + 51.0_wp)
      CALL fill_4d(sxxc0,  rbase + 52.0_wp)
      CALL fill_4d(syyc0,  rbase + 53.0_wp)
      CALL fill_4d(sxyc0,  rbase + 54.0_wp)
      CALL fill_4d(sxe,    rbase + 55.0_wp)
      CALL fill_4d(sye,    rbase + 56.0_wp)
      CALL fill_4d(sxxe,   rbase + 57.0_wp)
      CALL fill_4d(syye,   rbase + 58.0_wp)
      CALL fill_4d(sxye,   rbase + 59.0_wp)
      CALL fill_3d(sxsal,  rbase + 60.0_wp)
      CALL fill_3d(sysal,  rbase + 61.0_wp)
      CALL fill_3d(sxxsal, rbase + 62.0_wp)
      CALL fill_3d(syysal, rbase + 63.0_wp)
      CALL fill_3d(sxysal, rbase + 64.0_wp)
      CALL fill_4d(sxsi,   rbase + 65.0_wp)
      CALL fill_4d(sysi,   rbase + 66.0_wp)
      CALL fill_4d(sxxsi,  rbase + 67.0_wp)
      CALL fill_4d(syysi,  rbase + 68.0_wp)
      CALL fill_4d(sxysi,  rbase + 69.0_wp)
      CALL fill_3d(sxap,   rbase + 70.0_wp)
      CALL fill_3d(syap,   rbase + 71.0_wp)
      CALL fill_3d(sxxap,  rbase + 72.0_wp)
      CALL fill_3d(syyap,  rbase + 73.0_wp)
      CALL fill_3d(sxyap,  rbase + 74.0_wp)
      CALL fill_3d(sxvp,   rbase + 75.0_wp)
      CALL fill_3d(syvp,   rbase + 76.0_wp)
      CALL fill_3d(sxxvp,  rbase + 77.0_wp)
      CALL fill_3d(syyvp,  rbase + 78.0_wp)
      CALL fill_3d(sxyvp,  rbase + 79.0_wp)
      CALL fill_3d(sxvl,   rbase + 80.0_wp)
      CALL fill_3d(syvl,   rbase + 81.0_wp)
      CALL fill_3d(sxxvl,  rbase + 82.0_wp)
      CALL fill_3d(syyvl,  rbase + 83.0_wp)
      CALL fill_3d(sxyvl,  rbase + 84.0_wp)

      CALL fill_3d(h_i,  rbase + 90.0_wp)
      CALL fill_3d(h_s,  rbase + 91.0_wp)
      CALL fill_3d(o_i,  rbase + 92.0_wp)
      CALL fill_3d(h_ip, rbase + 93.0_wp)
      CALL fill_3d(h_il, rbase + 94.0_wp)
      CALL fill_a2d_3d(a_ip_frac, rbase + 95.0_wp)
      CALL fill_a2d_3d(a_ip_eff,  rbase + 96.0_wp)
      CALL fill_3d(s_i,  rbase + 97.0_wp)
      CALL fill_4d(sz_i, rbase + 98.0_wp)
      CALL fill_4d(t_i,  rbase + 99.0_wp)
      CALL fill_4d(t_s,  rbase + 100.0_wp)

      CALL fill_2d(vt_i,  rbase + 110.0_wp)
      CALL fill_2d(vt_s,  rbase + 111.0_wp)
      CALL fill_2d(at_i,  rbase + 112.0_wp)
      CALL fill_2d(at_ip, rbase + 113.0_wp)
      CALL fill_2d(vt_ip, rbase + 114.0_wp)
      CALL fill_2d(vt_il, rbase + 115.0_wp)
      CALL fill_2d(ato_i, rbase + 116.0_wp)
      CALL fill_a2d_2d(et_s,      rbase + 117.0_wp)
      CALL fill_a2d_2d(et_i,      rbase + 118.0_wp)
      CALL fill_a2d_2d(at_ip_eff, rbase + 119.0_wp)
      CALL fill_a2d_2d(tm_su,     rbase + 120.0_wp)
      CALL fill_a2d_2d(st_i,      rbase + 121.0_wp)
      CALL fill_a2d_2d(hm_i,      rbase + 122.0_wp)
      CALL fill_a2d_2d(hm_s,      rbase + 123.0_wp)
      CALL fill_a2d_2d(tm_si,     rbase + 124.0_wp)
      CALL fill_a2d_2d(om_i,      rbase + 125.0_wp)
      CALL fill_a2d_2d(sm_i,      rbase + 126.0_wp)
      CALL fill_a2d_2d(tm_i,      rbase + 127.0_wp)
      CALL fill_a2d_2d(tm_s,      rbase + 128.0_wp)
      CALL fill_a2d_2d(hm_ip,     rbase + 129.0_wp)
      CALL fill_a2d_2d(hm_il,     rbase + 130.0_wp)

      CALL o2_write_istate_snapshot( 1, 1, 2, 3.6e14_wp, 0.105_wp, &
         &                           2.0_wp, 2.0_wp, 1.0_wp, 0.2_wp, 0.2_wp, &
         &                           0.9_wp, 0.9_wp, 6.3_wp, 6.3_wp, 270.0_wp, 270.0_wp, &
         &                           270.0_wp, 270.0_wp, 270.0_wp, 270.0_wp, 0.2_wp, 0.2_wp, &
         &                           0.05_wp, 0.05_wp, 0.0_wp, 0.0_wp )
      CALL o2_write_si3_state_snapshot( 'entry', 0, .FALSE. )
      CALL o2_write_si3_state_snapshot( 'step', 1, .TRUE. )
   END DO

CONTAINS

   SUBROUTINE fill_2d( arr, offset )
      REAL(wp), DIMENSION(jpi,jpj), INTENT(out) :: arr
      REAL(wp), INTENT(in) :: offset
      INTEGER :: ji, jj
      DO jj = 1, jpj
         DO ji = 1, jpi
            arr(ji,jj) = offset + 100.0_wp * REAL(ji, wp) + REAL(jj, wp)
         END DO
      END DO
   END SUBROUTINE fill_2d

   SUBROUTINE fill_a2d_2d( arr, offset )
      REAL(wp), DIMENSION(Nis0:Nie0,Njs0:Nje0), INTENT(out) :: arr
      REAL(wp), INTENT(in) :: offset
      INTEGER :: ji, jj
      DO jj = Njs0, Nje0
         DO ji = Nis0, Nie0
            arr(ji,jj) = offset + 100.0_wp * REAL(ji, wp) + REAL(jj, wp)
         END DO
      END DO
   END SUBROUTINE fill_a2d_2d

   SUBROUTINE fill_3d( arr, offset )
      REAL(wp), DIMENSION(:,:,:), INTENT(out) :: arr
      REAL(wp), INTENT(in) :: offset
      INTEGER :: ji, jj, jl
      DO jl = 1, SIZE(arr, 3)
         DO jj = 1, jpj
            DO ji = 1, jpi
               arr(ji,jj,jl) = offset + 100.0_wp * REAL(ji, wp) + REAL(jj, wp) + 0.1_wp * REAL(jl, wp)
            END DO
         END DO
      END DO
   END SUBROUTINE fill_3d

   SUBROUTINE fill_a2d_3d( arr, offset )
      REAL(wp), DIMENSION(Nis0:Nie0,Njs0:Nje0,jpl), INTENT(out) :: arr
      REAL(wp), INTENT(in) :: offset
      INTEGER :: ji, jj, jl
      DO jl = 1, jpl
         DO jj = Njs0, Nje0
            DO ji = Nis0, Nie0
               arr(ji,jj,jl) = offset + 100.0_wp * REAL(ji, wp) + REAL(jj, wp) + 0.1_wp * REAL(jl, wp)
            END DO
         END DO
      END DO
   END SUBROUTINE fill_a2d_3d

   SUBROUTINE fill_4d( arr, offset )
      REAL(wp), DIMENSION(:,:,:,:), INTENT(out) :: arr
      REAL(wp), INTENT(in) :: offset
      INTEGER :: ji, jj, jk, jl
      DO jl = 1, SIZE(arr, 4)
         DO jk = 1, SIZE(arr, 3)
            DO jj = 1, jpj
               DO ji = 1, jpi
                  arr(ji,jj,jk,jl) = offset + 100.0_wp * REAL(ji, wp) + REAL(jj, wp) &
                     &             + 0.01_wp * REAL(jk, wp) + 0.1_wp * REAL(jl, wp)
               END DO
            END DO
         END DO
      END DO
   END SUBROUTINE fill_4d

END PROGRAM o2_si3_layout_fixture
