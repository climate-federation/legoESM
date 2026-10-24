PROGRAM tke_round331_layout_fixture
   USE dom_oce
   USE tke_round331_writer
   IMPLICIT NONE
   INTEGER :: rank,ji,jj,jk
   CHARACTER(LEN=8) :: argument
   REAL(wp) :: rank_offset
   REAL(wp) :: interior(3,2,3),sh2(3,2,3),bottom(3,2),top(3,2)
   REAL(wp) :: hmlp(3,2),sh2row(3,3),fullrow(3,4)
   REAL(wp) :: diag(3,4),upper(3,4),lower(3,4)
   REAL(wp) :: en(3,2,4),avm(3,2,3),avt(3,2,3),wmask(3,2,4)
   INTEGER :: nmln(3,2)

   CALL GET_COMMAND_ARGUMENT(1,argument)
   READ(argument,*) rank
   narea=rank+1
   rank_offset=10000._wp*rank
   DO jk=1,3; DO jj=1,2; DO ji=1,3
      interior(ji,jj,jk)=rank_offset+100._wp*jk+10._wp*jj+ji
      sh2(ji,jj,jk)=0.01_wp*interior(ji,jj,jk)
   END DO; END DO; END DO
   bottom=rank_offset+9._wp; top=rank_offset+10._wp
   nmln=3; hmlp=20._wp+rank
   CALL r331_begin(1,1,1,3,interior)
   CALL r331_sh2(sh2)
   CALL r331_drag(1,bottom); CALL r331_drag(2,top)
   CALL r331_mxl(nmln,hmlp)
   DO jj=1,2
      sh2row=sh2(:,jj,:)
      CALL r331_tke_inputs(jj,sh2row,bottom(:,jj),top(:,jj),nmln(:,jj),hmlp(:,jj))
      fullrow(:,1)=rank_offset+1._wp
      fullrow(:,2:4)=interior(:,jj,:)+0.1_wp
      CALL r331_after_bc(jj,fullrow)
      diag=2._wp; upper=-0.25_wp; lower=-0.5_wp
      CALL r331_after_matrix(jj,fullrow,diag,upper,lower)
      fullrow(:,2:4)=interior(:,jj,:)+0.25_wp
      CALL r331_after_solve(jj,fullrow)
   END DO
   wmask=1._wp
   en(:,:,1)=rank_offset+1._wp
   en(:,:,2:4)=interior+0.5_wp
   avm=interior; avt=interior+0.25_wp
   CALL r331_finish(en,avm,avt,wmask)
END PROGRAM tke_round331_layout_fixture
