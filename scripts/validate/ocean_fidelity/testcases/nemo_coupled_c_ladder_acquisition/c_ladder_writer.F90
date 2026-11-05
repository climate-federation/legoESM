MODULE c_ladder_writer
   USE, INTRINSIC :: iso_fortran_env, ONLY: int32
   USE par_oce, ONLY: wp, jpi, jpj, jpk, jpts, jp_tem, jp_sal
   USE oce, ONLY: ts, uu, vv, ssh, fraqsr_1lev
   USE dom_oce, ONLY: tmask, umask, vmask, e3t_3d, r3t
   USE sbc_oce, ONLY: ssu_m, ssv_m, sst_m, sss_m, ssh_m, e3t_m, frq_m, &
      & utau, vtau, taum, utauU, vtauV, qsr, qns, emp, sfx, fr_i, fwfice
   USE sbcapr, ONLY: ssh_ib
   USE lib_mpp, ONLY: mpprank
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: c_record_ssm, c_record_sbc, c_record_step
CONTAINS
   SUBROUTINE open_c(kind,kt,kbb,kmm,nfields,u)
      INTEGER,INTENT(in)::kind,kt,kbb,kmm,nfields
      INTEGER,INTENT(out)::u
      CHARACTER(LEN=8),PARAMETER::magic='CLAD0001'
      CHARACTER(LEN=192)::path
      INTEGER(int32)::meta(11)
      WRITE(path,'("c_ladder_records/c_kind",I1,"_step",I4.4,"_rank",I4.4,".bin")') kind,kt,mpprank
      OPEN(NEWUNIT=u,FILE=TRIM(path),ACCESS='stream',FORM='unformatted',STATUS='replace',ACTION='write')
      meta=[kind,kt,kbb,kmm,mpprank,jpi,jpj,jpk,jpts,STORAGE_SIZE(1._wp),nfields]
      WRITE(u) magic,INT(1,int32),meta
   END SUBROUTINE open_c

   SUBROUTINE c_record_ssm(kind,kt,kbb,kmm,lmean,usect,aprdyn)
      INTEGER,INTENT(in)::kind,kt,kbb,kmm
      LOGICAL,INTENT(in)::lmean,usect,aprdyn
      INTEGER::u
      IF(kind==1) THEN
         CALL open_c(kind,kt,kbb,kmm,20,u)
         WRITE(u) MERGE(1_int32,0_int32,lmean),MERGE(1_int32,0_int32,usect),MERGE(1_int32,0_int32,aprdyn)
         WRITE(u) ts(:,:,1,jp_tem,kmm),ts(:,:,1,jp_sal,kmm),uu(:,:,1,kbb),vv(:,:,1,kbb),ssh(:,:,kmm)
         WRITE(u) ssh_ib,e3t_3d(:,:,1),r3t(:,:,kmm),tmask(:,:,1),fraqsr_1lev
         WRITE(u) ssu_m,ssv_m,sst_m,sss_m,ssh_m,e3t_m,frq_m
      ELSEIF(kind==2) THEN
         CALL open_c(kind,kt,kbb,kmm,7,u)
         WRITE(u) ssu_m,ssv_m,sst_m,sss_m,ssh_m,e3t_m,frq_m
      ELSE
         ERROR STOP 'C-ladder invalid SSM kind'
      ENDIF
      CLOSE(u)
   END SUBROUTINE c_record_ssm

   SUBROUTINE c_record_sbc(kind,kt,kbb,kmm)
      INTEGER,INTENT(in)::kind,kt,kbb,kmm
      INTEGER::u
      IF(kind==3) THEN
         CALL open_c(kind,kt,kbb,kmm,7,u)
         WRITE(u) ssu_m,ssv_m,sst_m,sss_m,ssh_m,e3t_m,frq_m
      ELSEIF(kind==4) THEN
         CALL open_c(kind,kt,kbb,kmm,14,u)
         WRITE(u) utau,vtau,taum,utauU,vtauV,qsr,qns,emp,sfx,fr_i,fwfice
         WRITE(u) tmask(:,:,1),umask(:,:,1),vmask(:,:,1)
      ELSE
         ERROR STOP 'C-ladder invalid SBC kind'
      ENDIF
      CLOSE(u)
   END SUBROUTINE c_record_sbc

   SUBROUTINE c_record_step(kind,kt,kbb,kmm)
      INTEGER,INTENT(in)::kind,kt,kbb,kmm
      INTEGER::u
      IF(kind/=5 .AND. kind/=6) ERROR STOP 'C-ladder invalid step kind'
      CALL open_c(kind,kt,kbb,kmm,4,u)
      WRITE(u) ts(:,:,:,:,kbb),uu(:,:,:,kbb),vv(:,:,:,kbb),ssh(:,:,kbb)
      CLOSE(u)
   END SUBROUTINE c_record_step
END MODULE c_ladder_writer
