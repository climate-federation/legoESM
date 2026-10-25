MODULE o5_si3_writer
   !! Passive, write-only observer for SI3 ladder rung O5.
   !! Every public routine is called outside NEMO's ji/jj loops.  Array
   !! assignment and reduction are deliberately absent: records contain the
   !! exact binary64 allocations supplied by the patched call site.
   USE, INTRINSIC :: iso_fortran_env, ONLY : int32, real64
   USE par_kind, ONLY : wp
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: o5_context, o5_strength_entry, o5_strength_raw, o5_strength_smooth
   PUBLIC :: o5_aevp_entry_state, o5_aevp_entry_masks, o5_aevp_entry_metrics
   PUBLIC :: o5_fimask, o5_frame_delta, o5_frame_stress, o5_frame_velocity, o5_aevp_exit
   PUBLIC :: o5_rdgrft_entry, o5_rdgrft_exit

   CHARACTER(len=8), PARAMETER :: record_magic = 'O5SI3R01'
   INTEGER(int32), PARAMETER :: record_version = 1_int32
   INTEGER(int32), SAVE :: c_kt = -1, c_kmm = -1, c_rank = -1
   INTEGER(int32), SAVE :: c_jpi = -1, c_jpj = -1, c_jpl = -1, c_nn_hls = -1
   INTEGER(int32), SAVE :: c_ncom_stp = -1, c_nit000 = -1, c_jpni = -1, c_jpnj = -1
   INTEGER(int32), SAVE :: c_ln_nnogather = -1, c_nn_npfchk = -1
   REAL(real64), SAVE :: c_rdt_ice = -1.0_real64

CONTAINS

   SUBROUTINE open_record(iu, kind, jter, nfields)
      INTEGER, INTENT(out) :: iu
      INTEGER, INTENT(in) :: kind, jter, nfields
      INTEGER :: ios
      INTEGER(int32), DIMENSION(18) :: meta
      REAL(real64), DIMENSION(2) :: rmeta
      CHARACTER(len=256) :: name

      IF (c_kt < 0 .OR. c_jpi < 1 .OR. c_jpj < 1) ERROR STOP 'O5 context not initialized'
      WRITE(name, '("o5_si3_records/o5_kind",I2.2,"_step",I4.4,"_iter",I3.3,"_rank",I4.4,".bin")') &
         kind, c_kt, jter, c_rank
      OPEN(NEWUNIT=iu, FILE=TRIM(name), FORM='UNFORMATTED', ACCESS='STREAM', STATUS='REPLACE', &
         ACTION='WRITE', IOSTAT=ios)
      IF (ios /= 0) ERROR STOP 'O5 record open failed'
      meta = [INT(kind,int32), c_kt, INT(jter,int32), c_kmm, c_rank, c_jpi, c_jpj, c_jpl, &
              c_nn_hls, c_ncom_stp, c_nit000, c_jpni, c_jpnj, c_ln_nnogather, c_nn_npfchk, &
              INT(nfields,int32), 1_int32, 0_int32]
      rmeta = [c_rdt_ice, 0.0_real64]
      WRITE(iu) record_magic
      WRITE(iu) record_version
      WRITE(iu) meta
      WRITE(iu) rmeta
   END SUBROUTINE open_record

   SUBROUTINE o5_context(kt, kmm, rank, jpi, jpj, jpl, nn_hls, ncom_stp, nit000, jpni, jpnj, &
                         ln_nnogather, nn_npfchk, rdt_ice)
      INTEGER, INTENT(in) :: kt, kmm, rank, nn_hls, ncom_stp, nit000, jpni, jpnj
      INTEGER, INTENT(in) :: jpi, jpj, jpl
      LOGICAL, INTENT(in) :: ln_nnogather
      INTEGER, INTENT(in) :: nn_npfchk
      REAL(wp), INTENT(in) :: rdt_ice
      c_kt = INT(kt,int32); c_kmm = INT(kmm,int32); c_rank = INT(rank,int32)
      c_jpi = INT(jpi,int32); c_jpj = INT(jpj,int32); c_jpl = INT(jpl,int32)
      c_nn_hls = INT(nn_hls,int32); c_ncom_stp = INT(ncom_stp,int32)
      c_nit000 = INT(nit000,int32); c_jpni = INT(jpni,int32); c_jpnj = INT(jpnj,int32)
      c_ln_nnogather = MERGE(1_int32,0_int32,ln_nnogather); c_nn_npfchk = INT(nn_npfchk,int32)
      c_rdt_ice = REAL(rdt_ice,real64)
   END SUBROUTINE o5_context

   SUBROUTINE o5_strength_entry(a_i, v_i, at_i, tmask)
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: a_i, v_i
      REAL(wp), DIMENSION(:,:), INTENT(in) :: at_i, tmask
      INTEGER :: iu
      IF (SIZE(at_i,1) /= c_jpi .OR. SIZE(at_i,2) /= c_jpj .OR. SIZE(a_i,3) /= c_jpl) &
         ERROR STOP 'O5 strength extent/context mismatch'
      CALL open_record(iu, 1, 0, 4)
      WRITE(iu) a_i; WRITE(iu) v_i; WRITE(iu) at_i; WRITE(iu) tmask
      CLOSE(iu)
   END SUBROUTINE o5_strength_entry

   SUBROUTINE o5_strength_raw(at_i, strength)
      REAL(wp), DIMENSION(:,:), INTENT(in) :: at_i, strength
      INTEGER :: iu
      CALL open_record(iu, 2, 0, 2)
      WRITE(iu) at_i; WRITE(iu) strength
      CLOSE(iu)
   END SUBROUTINE o5_strength_raw

   SUBROUTINE o5_strength_smooth(strength)
      REAL(wp), DIMENSION(:,:), INTENT(in) :: strength
      INTEGER :: iu
      CALL open_record(iu, 3, 0, 1)
      WRITE(iu) strength
      CLOSE(iu)
   END SUBROUTINE o5_strength_smooth

   SUBROUTINE o5_aevp_entry_state(u, v, ub, vb, zs1, zs2, zs12, strength, at_i, &
                                  vt_i, vt_s, vt_ip, vt_il, utau, vtau, drag, &
                                  u_oce, v_oce, ssh_m, snw, snwb, zsshdyn, ff_t)
      REAL(wp), DIMENSION(:,:), INTENT(in) :: u, v, ub, vb, zs1, zs2, zs12, strength, at_i
      REAL(wp), DIMENSION(:,:), INTENT(in) :: vt_i, vt_s, vt_ip, vt_il, utau, vtau, drag
      REAL(wp), DIMENSION(:,:), INTENT(in) :: u_oce, v_oce, ssh_m, snw, snwb, zsshdyn, ff_t
      INTEGER :: iu
      CALL open_record(iu, 4, 0, 23)
      WRITE(iu) u; WRITE(iu) v; WRITE(iu) ub; WRITE(iu) vb
      WRITE(iu) zs1; WRITE(iu) zs2; WRITE(iu) zs12; WRITE(iu) strength; WRITE(iu) at_i
      WRITE(iu) vt_i; WRITE(iu) vt_s; WRITE(iu) vt_ip; WRITE(iu) vt_il
      WRITE(iu) utau; WRITE(iu) vtau; WRITE(iu) drag; WRITE(iu) u_oce; WRITE(iu) v_oce
      WRITE(iu) ssh_m; WRITE(iu) snw; WRITE(iu) snwb; WRITE(iu) zsshdyn; WRITE(iu) ff_t
      CLOSE(iu)
   END SUBROUTINE o5_aevp_entry_state

   SUBROUTINE o5_aevp_entry_masks(tmask, umask, vmask, fast_u, fast_v, fast_t, icb_u, icb_v, icb_t)
      REAL(wp), DIMENSION(:,:), INTENT(in) :: tmask, umask, vmask, fast_u, fast_v, fast_t
      REAL(wp), DIMENSION(:,:), INTENT(in) :: icb_u, icb_v, icb_t
      INTEGER :: iu
      CALL open_record(iu, 5, 0, 9)
      WRITE(iu) tmask; WRITE(iu) umask; WRITE(iu) vmask
      WRITE(iu) fast_u; WRITE(iu) fast_v; WRITE(iu) fast_t
      WRITE(iu) icb_u; WRITE(iu) icb_v; WRITE(iu) icb_t
      CLOSE(iu)
   END SUBROUTINE o5_aevp_entry_masks

   SUBROUTINE o5_aevp_entry_metrics(e1t,e2t,e1u,e2u,e1v,e2v,e1f,e2f, &
                                    a_t,a_u,a_v,a_f,r1_e1u,r1_e2v,r1_at,r1_au,r1_av,r1_af)
      REAL(wp), DIMENSION(:,:), INTENT(in) :: e1t,e2t,e1u,e2u,e1v,e2v,e1f,e2f
      REAL(wp), DIMENSION(:,:), INTENT(in) :: a_t,a_u,a_v,a_f,r1_e1u,r1_e2v,r1_at,r1_au,r1_av,r1_af
      INTEGER :: iu
      CALL open_record(iu, 6, 0, 18)
      WRITE(iu) e1t; WRITE(iu) e2t; WRITE(iu) e1u; WRITE(iu) e2u
      WRITE(iu) e1v; WRITE(iu) e2v; WRITE(iu) e1f; WRITE(iu) e2f
      WRITE(iu) a_t; WRITE(iu) a_u; WRITE(iu) a_v; WRITE(iu) a_f
      WRITE(iu) r1_e1u; WRITE(iu) r1_e2v; WRITE(iu) r1_at; WRITE(iu) r1_au
      WRITE(iu) r1_av; WRITE(iu) r1_af
      CLOSE(iu)
   END SUBROUTINE o5_aevp_entry_metrics

   SUBROUTINE o5_fimask(fimask)
      REAL(wp), DIMENSION(:,:), INTENT(in) :: fimask
      INTEGER :: iu
      CALL open_record(iu, 7, 0, 1); WRITE(iu) fimask; CLOSE(iu)
   END SUBROUTINE o5_fimask

   SUBROUTINE o5_frame_delta(jter, zdelta, zp_delt)
      INTEGER, INTENT(in) :: jter
      REAL(wp), DIMENSION(:,:), INTENT(in) :: zdelta, zp_delt
      INTEGER :: iu
      CALL open_record(iu, 8, jter, 2)
      WRITE(iu) zdelta; WRITE(iu) zp_delt
      CLOSE(iu)
   END SUBROUTINE o5_frame_delta

   SUBROUTINE o5_frame_stress(jter, zs1, zs2, zs12)
      INTEGER, INTENT(in) :: jter
      REAL(wp), DIMENSION(:,:), INTENT(in) :: zs1, zs2, zs12
      INTEGER :: iu
      CALL open_record(iu, 9, jter, 3)
      WRITE(iu) zs1; WRITE(iu) zs2; WRITE(iu) zs12
      CLOSE(iu)
   END SUBROUTINE o5_frame_stress

   SUBROUTINE o5_frame_velocity(jter, u, v)
      INTEGER, INTENT(in) :: jter
      REAL(wp), DIMENSION(:,:), INTENT(in) :: u, v
      INTEGER :: iu
      CALL open_record(iu, 10, jter, 2); WRITE(iu) u; WRITE(iu) v; CLOSE(iu)
   END SUBROUTINE o5_frame_velocity

   SUBROUTINE o5_aevp_exit(stress1, stress2, stress12, u, v, divu, shear, delta)
      REAL(wp), DIMENSION(:,:), INTENT(in) :: stress1, stress2, stress12, u, v, divu, shear, delta
      INTEGER :: iu
      CALL open_record(iu, 11, 0, 8)
      WRITE(iu) stress1; WRITE(iu) stress2; WRITE(iu) stress12
      WRITE(iu) u; WRITE(iu) v; WRITE(iu) divu; WRITE(iu) shear; WRITE(iu) delta
      CLOSE(iu)
   END SUBROUTINE o5_aevp_exit

   SUBROUTINE o5_rdgrft_entry(a_i, v_i)
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: a_i, v_i
      INTEGER :: iu
      CALL open_record(iu, 12, 0, 2); WRITE(iu) a_i; WRITE(iu) v_i; CLOSE(iu)
   END SUBROUTINE o5_rdgrft_entry

   SUBROUTINE o5_rdgrft_exit(a_i, v_i)
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: a_i, v_i
      INTEGER :: iu
      CALL open_record(iu, 13, 0, 2); WRITE(iu) a_i; WRITE(iu) v_i; CLOSE(iu)
   END SUBROUTINE o5_rdgrft_exit

END MODULE o5_si3_writer
