MODULE round340_surface_writer
   USE par_oce, ONLY: wp
   USE lib_mpp, ONLY: mpprank
   USE sbc_oce, ONLY: sst_m, sss_m, ssu_m, ssv_m, ssh_m, e3t_m, frq_m, &
      & utau, vtau, utauU, vtauV, taum, wndm, qsr, qns, emp, sfx, rnf, fr_i
   USE sbcrnf, ONLY: rnf_tsc
   USE sbcssr, ONLY: erp, qrp, coefice
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r340_surface_frame, r340_s1_turb_frame, r340_s1_flux_frame
CONTAINS
   SUBROUTINE r340_surface_frame(kt, kid)
      INTEGER, INTENT(in) :: kt, kid
      INTEGER :: iu, i0, i1, j0, j1
      CHARACTER(LEN=128) :: name
      CHARACTER(LEN=16) :: magic
      magic = 'NEMO_R340_S4_3'
      WRITE(name,'("oracle_r340_surface_k",I8.8,"_b",I2.2,"_r",I4.4,".bin")') kt,kid,mpprank
      OPEN(NEWUNIT=iu,FILE=TRIM(name),ACCESS='STREAM',FORM='UNFORMATTED',STATUS='NEW',ACTION='WRITE')
      WRITE(iu) magic
      i0=LBOUND(qsr,1); i1=UBOUND(qsr,1); j0=LBOUND(qsr,2); j1=UBOUND(qsr,2)
      WRITE(iu) 3,kt,kid,mpprank,i1-i0+1,j1-j0+1,64
      IF(kid >= 1) WRITE(iu) sst_m(i0:i1,j0:j1),sss_m(i0:i1,j0:j1), &
         & ssu_m(i0:i1,j0:j1),ssv_m(i0:i1,j0:j1),ssh_m(i0:i1,j0:j1), &
         & e3t_m(i0:i1,j0:j1),frq_m(i0:i1,j0:j1)
      IF(kid >= 3) WRITE(iu) utau(i0:i1,j0:j1),vtau(i0:i1,j0:j1), &
         & emp(i0:i1,j0:j1),qsr,qns,sfx,taum,wndm
      IF(kid >= 5) WRITE(iu) fr_i(i0:i1,j0:j1)
      IF(kid >= 7) WRITE(iu) rnf(i0:i1,j0:j1),rnf_tsc(:,:,1),rnf_tsc(:,:,2)
      IF(kid >= 8) WRITE(iu) erp,qrp,coefice
      IF(kid >= 11) WRITE(iu) utauU(i0:i1,j0:j1),vtauV(i0:i1,j0:j1)
      CLOSE(iu)
   END SUBROUTINE r340_surface_frame

   SUBROUTINE r340_s1_turb_frame(kt,ptair,pqair,pst,pu,pv,ptsk,pssq,pcd,pch,pce, &
      & prhoa,psen,plat,pevp,pqsr,putau,pvtau,ptaum,pwndm)
      INTEGER, INTENT(in) :: kt
      REAL(wp), INTENT(in) :: ptair(:,:),pqair(:,:),pst(:,:),pu(:,:),pv(:,:),ptsk(:,:),pssq(:,:)
      REAL(wp), INTENT(in) :: pcd(:,:),pch(:,:),pce(:,:),prhoa(:,:),psen(:,:),plat(:,:),pevp(:,:)
      REAL(wp), INTENT(in) :: pqsr(:,:),putau(:,:),pvtau(:,:),ptaum(:,:),pwndm(:,:)
      INTEGER :: iu
      CHARACTER(LEN=128) :: name
      CHARACTER(LEN=16) :: magic
      magic='NEMO_R340_S1T_1'
      WRITE(name,'("oracle_r340_s1t_k",I8.8,"_r",I4.4,".bin")') kt,mpprank
      OPEN(NEWUNIT=iu,FILE=TRIM(name),ACCESS='STREAM',FORM='UNFORMATTED',STATUS='NEW',ACTION='WRITE')
      WRITE(iu) magic
      WRITE(iu) 1,kt,0,mpprank,SIZE(ptair,1),SIZE(ptair,2),64
      WRITE(iu) ptair,pqair,pst,pu,pv,ptsk,pssq,pcd,pch,pce,prhoa,psen,plat,pevp,pqsr,putau,pvtau,ptaum,pwndm
      CLOSE(iu)
   END SUBROUTINE r340_s1_turb_frame

   SUBROUTINE r340_s1_flux_frame(kt,pprec,psnow,pdqlw,pqlwn,pqns,pemp)
      INTEGER, INTENT(in) :: kt
      REAL(wp), INTENT(in) :: pprec(:,:),psnow(:,:),pdqlw(:,:),pqlwn(:,:),pqns(:,:),pemp(:,:)
      INTEGER :: iu
      CHARACTER(LEN=128) :: name
      CHARACTER(LEN=16) :: magic
      magic='NEMO_R340_S1F_1'
      WRITE(name,'("oracle_r340_s1f_k",I8.8,"_r",I4.4,".bin")') kt,mpprank
      OPEN(NEWUNIT=iu,FILE=TRIM(name),ACCESS='STREAM',FORM='UNFORMATTED',STATUS='NEW',ACTION='WRITE')
      WRITE(iu) magic
      WRITE(iu) 1,kt,1,mpprank,SIZE(pprec,1),SIZE(pprec,2),64
      WRITE(iu) pprec,psnow,pdqlw,pqlwn,pqns,pemp
      CLOSE(iu)
   END SUBROUTINE r340_s1_flux_frame
END MODULE round340_surface_writer
