MODULE round340_state_writer
   USE par_oce, ONLY: wp, jpi, jpj, jpk, jpts
   USE lib_mpp, ONLY: mpprank
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r340_state_frame
CONTAINS
   SUBROUTINE r340_state_frame(kt,kstage,klevel,pts,pu,pv,pssh)
      INTEGER, INTENT(in) :: kt,kstage,klevel
      REAL(wp), INTENT(in) :: pts(jpi,jpj,jpk,jpts),pu(jpi,jpj,jpk),pv(jpi,jpj,jpk),pssh(jpi,jpj)
      INTEGER :: iu
      CHARACTER(LEN=128) :: name
      CHARACTER(LEN=16) :: magic
      magic='NEMO_R340_STATE1'
      WRITE(name,'("oracle_r340_state_k",I8.8,"_s",I1,"_r",I4.4,".bin")') kt,kstage,mpprank
      OPEN(NEWUNIT=iu,FILE=TRIM(name),ACCESS='STREAM',FORM='UNFORMATTED',STATUS='NEW',ACTION='WRITE')
      WRITE(iu) magic
      WRITE(iu) 1,kt,kstage,klevel,mpprank,jpi,jpj,jpk,jpts,64
      WRITE(iu) pts,pu,pv,pssh
      CLOSE(iu)
   END SUBROUTINE r340_state_frame
END MODULE round340_state_writer
