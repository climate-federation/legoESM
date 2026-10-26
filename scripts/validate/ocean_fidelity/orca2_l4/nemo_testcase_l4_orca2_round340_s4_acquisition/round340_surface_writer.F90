MODULE round340_surface_writer
   USE par_oce, ONLY: wp, jpi, jpj
   USE lib_mpp, ONLY: mpprank
   USE sbc_oce, ONLY: sst_m, sss_m, ssu_m, ssv_m, ssh_m, e3t_m, frq_m, &
      & utau, vtau, utauU, vtauV, taum, wndm, qsr, qns, emp, sfx, rnf, fr_i
   USE sbcrnf, ONLY: rnf_tsc
   USE sbcssr, ONLY: erp, qrp, coefice
   USE sbcfwb, ONLY: a_fwb, a_fwb_b
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r340_surface_frame, r340_s1_turb_frame, r340_s1_flux_frame
CONTAINS
   SUBROUTINE r340_surface_frame(kt, kid)
      INTEGER, INTENT(in) :: kt, kid
      INTEGER :: iu
      CHARACTER(LEN=128) :: name
      CHARACTER(LEN=16) :: magic
      magic = 'NEMO_R340_S4_2'
      WRITE(name,'("oracle_r340_surface_k",I8.8,"_b",I2.2,"_r",I4.4,".bin")') kt,kid,mpprank
      OPEN(NEWUNIT=iu,FILE=TRIM(name),ACCESS='STREAM',FORM='UNFORMATTED',STATUS='NEW',ACTION='WRITE')
      WRITE(iu) magic
      WRITE(iu) 2,kt,kid,mpprank,jpi,jpj,64
      WRITE(iu) LBOUND(qsr,1),UBOUND(qsr,1),LBOUND(qsr,2),UBOUND(qsr,2)
      WRITE(iu) sst_m,sss_m,ssu_m,ssv_m,ssh_m,e3t_m,frq_m
      WRITE(iu) utau,vtau,utauU,vtauV,emp,rnf,fr_i
      WRITE(iu) qsr,qns,sfx,taum,wndm,rnf_tsc,erp,qrp,coefice
      WRITE(iu) a_fwb,a_fwb_b
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
