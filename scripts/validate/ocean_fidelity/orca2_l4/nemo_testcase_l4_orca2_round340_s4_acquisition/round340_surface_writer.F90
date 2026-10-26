MODULE round340_surface_writer
   USE par_oce, ONLY: wp, jpi, jpj
   USE lib_mpp, ONLY: mpprank
   USE sbc_oce, ONLY: sst_m, sss_m, ssu_m, ssv_m, ssh_m, e3t_m, frq_m, &
      & utau, vtau, utauU, vtauV, taum, wndm, qsr, qns, emp, sfx
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r340_surface_frame
CONTAINS
   SUBROUTINE r340_surface_frame(kt, kid)
      INTEGER, INTENT(in) :: kt, kid
      INTEGER :: iu
      CHARACTER(LEN=128) :: name
      CHARACTER(LEN=16) :: magic
      magic = 'NEMO_R340_S4_1'
      WRITE(name,'("oracle_r340_surface_k",I8.8,"_b",I2.2,"_r",I4.4,".bin")') kt,kid,mpprank
      OPEN(NEWUNIT=iu,FILE=TRIM(name),ACCESS='STREAM',FORM='UNFORMATTED',STATUS='NEW',ACTION='WRITE')
      WRITE(iu) magic
      WRITE(iu) 1,kt,kid,mpprank,jpi,jpj,64
      WRITE(iu) LBOUND(qsr,1),UBOUND(qsr,1),LBOUND(qsr,2),UBOUND(qsr,2)
      WRITE(iu) sst_m,sss_m,ssu_m,ssv_m,ssh_m,e3t_m,frq_m
      WRITE(iu) utau,vtau,utauU,vtauV
      WRITE(iu) qsr,qns,emp,sfx,taum,wndm
      CLOSE(iu)
   END SUBROUTINE r340_surface_frame
END MODULE round340_surface_writer
