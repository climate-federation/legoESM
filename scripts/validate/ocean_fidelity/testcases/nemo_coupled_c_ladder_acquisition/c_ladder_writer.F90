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
      LOGICAL::have_ssh_ib
      IF(kind==1) THEN
         IF(.NOT.(ALLOCATED(ts).AND.ALLOCATED(uu).AND.ALLOCATED(vv).AND.ALLOCATED(ssh).AND. &
            & ALLOCATED(e3t_3d).AND.ALLOCATED(r3t).AND.ALLOCATED(tmask).AND. &
            & ALLOCATED(fraqsr_1lev).AND.ALLOCATED(ssu_m).AND.ALLOCATED(ssv_m).AND. &
            & ALLOCATED(sst_m).AND.ALLOCATED(sss_m).AND.ALLOCATED(ssh_m).AND. &
            & ALLOCATED(e3t_m).AND.ALLOCATED(frq_m))) ERROR STOP 'C-ladder kind 1 required array unallocated'
         have_ssh_ib=ALLOCATED(ssh_ib)
         IF(aprdyn.NEQV.have_ssh_ib) ERROR STOP 'C-ladder kind 1 ln_apr_dyn/ssh_ib allocation mismatch'
         CALL open_c(kind,kt,kbb,kmm,19+MERGE(1,0,have_ssh_ib),u)
         WRITE(u) MERGE(1_int32,0_int32,lmean),MERGE(1_int32,0_int32,usect),MERGE(1_int32,0_int32,aprdyn)
         WRITE(u) ts(:,:,1,jp_tem,kmm),ts(:,:,1,jp_sal,kmm),uu(:,:,1,kbb),vv(:,:,1,kbb),ssh(:,:,kmm)
         IF(have_ssh_ib) WRITE(u) ssh_ib
         WRITE(u) e3t_3d(:,:,1),r3t(:,:,kmm),tmask(:,:,1),fraqsr_1lev
         WRITE(u) ssu_m,ssv_m,sst_m,sss_m,ssh_m,e3t_m,frq_m
      ELSEIF(kind==2) THEN
         IF(.NOT.(ALLOCATED(ssu_m).AND.ALLOCATED(ssv_m).AND.ALLOCATED(sst_m).AND. &
            & ALLOCATED(sss_m).AND.ALLOCATED(ssh_m).AND.ALLOCATED(e3t_m).AND. &
            & ALLOCATED(frq_m))) ERROR STOP 'C-ladder kind 2 required array unallocated'
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
         IF(.NOT.(ALLOCATED(ssu_m).AND.ALLOCATED(ssv_m).AND.ALLOCATED(sst_m).AND. &
            & ALLOCATED(sss_m).AND.ALLOCATED(ssh_m).AND.ALLOCATED(e3t_m).AND. &
            & ALLOCATED(frq_m))) ERROR STOP 'C-ladder kind 3 required array unallocated'
         CALL open_c(kind,kt,kbb,kmm,7,u)
         WRITE(u) ssu_m,ssv_m,sst_m,sss_m,ssh_m,e3t_m,frq_m
      ELSEIF(kind==4) THEN
         IF(.NOT.(ALLOCATED(utau).AND.ALLOCATED(vtau).AND.ALLOCATED(taum).AND. &
            & ALLOCATED(utauU).AND.ALLOCATED(vtauV).AND.ALLOCATED(qsr).AND. &
            & ALLOCATED(qns).AND.ALLOCATED(emp).AND.ALLOCATED(sfx).AND. &
            & ALLOCATED(fr_i).AND.ALLOCATED(fwfice).AND.ALLOCATED(tmask).AND. &
            & ALLOCATED(umask).AND.ALLOCATED(vmask))) ERROR STOP 'C-ladder kind 4 required array unallocated'
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
      IF(.NOT.(ALLOCATED(ts).AND.ALLOCATED(uu).AND.ALLOCATED(vv).AND.ALLOCATED(ssh))) &
         & ERROR STOP 'C-ladder step required array unallocated'
      CALL open_c(kind,kt,kbb,kmm,4,u)
      WRITE(u) ts(:,:,:,:,kbb),uu(:,:,:,kbb),vv(:,:,:,kbb),ssh(:,:,kbb)
      CLOSE(u)
   END SUBROUTINE c_record_step
END MODULE c_ladder_writer
