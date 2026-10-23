PROGRAM dynzdf_round328_layout_fixture
   USE iso_fortran_env, ONLY : real64
   USE dom_oce
   USE dynzdf_round328_writer
   IMPLICIT NONE
   INTEGER :: rank, jj, phase, component, ji, jk, unit, ios
   CHARACTER(LEN=8) :: argument
   CHARACTER(LEN=16) :: magic
   CHARACTER(LEN=64) :: filename
   REAL(wp) :: puu(3,2,jpk,3), pvv(3,2,jpk,3), avm(4,3,jpk)
   REAL(wp) :: puub(3,2), pvvb(3,2), pbot(4,3), ptop(4,3)
   REAL(wp) :: utau(3,2), vtau(3,2), urow(3,jpkm1), vrow(3,jpkm1)
   REAL(wp) :: lower(3,jpkm1), diagonal(3,jpkm1), upper(3,jpkm1)
   REAL(wp) :: stage_t(3,2,jpk), stage_s(3,2,jpk)
   REAL(wp) :: stage_u(3,2,jpk), stage_v(3,2,jpk), stage_ssh(3,2)

   CALL GET_COMMAND_ARGUMENT(1,argument)
   READ(argument,*) rank
   narea=rank+1
   DO component=1,3; DO jk=1,jpk; DO jj=1,2; DO ji=1,3
      puu(ji,jj,jk,component)=1000._wp*component+100._wp*jk+10._wp*jj+ji
      pvv(ji,jj,jk,component)=puu(ji,jj,jk,component)+0.5_wp
   END DO; END DO; END DO; END DO
   DO jk=1,jpk; DO jj=1,3; DO ji=1,4
      avm(ji,jj,jk)=REAL(100*jk+10*jj+ji,wp)
   END DO; END DO; END DO
   puub=7._wp; pvvb=8._wp; pbot=9._wp; ptop=10._wp
   utau=11._wp; vtau=12._wp
   CALL r328_begin(1,1,2,3,3,puu,pvv,avm,puub,pvvb,pbot,ptop,utau,vtau)
   DO phase=1,4; DO jj=1,2
      urow=REAL(100*phase+jj,wp); vrow=urow+0.25_wp
      CALL r328_state_row(phase,jj,urow,vrow)
   END DO; END DO
   DO component=1,2; DO phase=1,5; DO jj=1,2
      lower=REAL(1000*component+100*phase+10*jj+1,wp)
      diagonal=REAL(1000*component+100*phase+10*jj+2,wp)
      upper=REAL(1000*component+100*phase+10*jj+3,wp)
      CALL r328_matrix_row(component,phase,jj,lower,diagonal,upper)
   END DO; END DO; END DO
   DO component=1,2; DO jj=1,2
      urow=REAL(9000+100*component+jj,wp)
      CALL r328_solve_row(component,jj,urow)
   END DO; END DO
   CALL r328_finish(puu,pvv,3)

   stage_t=0._wp; stage_s=0._wp; stage_u=0._wp; stage_v=0._wp; stage_ssh=0._wp
   stage_u(:,:,1:jpkm1)=puu(:,:,1:jpkm1,3)
   stage_v(:,:,1:jpkm1)=pvv(:,:,1:jpkm1,3)
   IF(rank == 0) THEN
      filename='oracle_step_entry_kt00000002.bin'
   ELSE
      filename='oracle_step_entry_rank0001_kt00000002.bin'
   ENDIF
   magic='NEMO_L1_ENTRY_1'
   OPEN(NEWUNIT=unit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
      & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
   IF(ios /= 0) ERROR STOP 2
   WRITE(unit) magic,1,2,3,3,2,jpk,2,64
   WRITE(unit) stage_t,stage_s,stage_u,stage_v,stage_ssh
   CLOSE(unit)

   IF(rank == 0) THEN
      magic='NEMO_L1_STAGE_1'
      OPEN(NEWUNIT=unit,FILE='oracle_stage_kt00000001_s3.bin',ACCESS='STREAM', &
         & FORM='UNFORMATTED',STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) ERROR STOP 2
      WRITE(unit) magic,1,1,3,3,3,2,jpk,2,64
      WRITE(unit) stage_t,stage_s,stage_u,stage_v,stage_ssh
      CLOSE(unit)
   ENDIF
END PROGRAM dynzdf_round328_layout_fixture
