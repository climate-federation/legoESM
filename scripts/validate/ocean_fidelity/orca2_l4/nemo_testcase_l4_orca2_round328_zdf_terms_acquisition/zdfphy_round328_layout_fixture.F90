PROGRAM zdfphy_round328_layout_fixture
   USE, INTRINSIC :: iso_fortran_env, ONLY : real64
   USE dom_oce
   USE zdfphy_round328_writer
   IMPLICIT NONE
   INTEGER :: rank,ji,jj,jk,unit,ios
   CHARACTER(LEN=8) :: argument
   CHARACTER(LEN=16) :: magic
   REAL(wp) :: avm(3,2,3),avt(3,2,3),avs(3,2,3),en(3,2,3)
   REAL(wp) :: avm_k(3,2,3),avt_k(3,2,3),rn2(3,2,3),wmask(3,2,3)
   REAL(wp) :: r3t(3,2),ht0(3,2),wave(3,2),emx(3,2),reb(3,2)
   INTEGER :: mbkt(3,2)
   REAL(wp) :: full_avm(7,6,jpk),interior(3,2,jpk),zero3(3,2,jpk)

   CALL GET_COMMAND_ARGUMENT(1,argument)
   READ(argument,*) rank
   narea=rank+1
   DO jk=1,3; DO jj=1,2; DO ji=1,3
      avm_k(ji,jj,jk)=100._wp*jk+10._wp*jj+ji
      avt_k(ji,jj,jk)=avm_k(ji,jj,jk)+0.25_wp
      en(ji,jj,jk)=avm_k(ji,jj,jk)+0.5_wp
      rn2(ji,jj,jk)=avm_k(ji,jj,jk)+0.75_wp
      wmask(ji,jj,jk)=1._wp
   END DO; END DO; END DO
   wmask(3,2,3)=0._wp
   avm_k=avm_k*wmask; avt_k=avt_k*wmask
   avm=avm_k; avt=avt_k; avs=-3._wp
   r3t=0.01_wp; ht0=100._wp; mbkt=4
   CALL r328_zp_begin(1,1,1,3,avm,avt,avs,en,avm_k,avt_k,rn2,r3t,ht0,mbkt,wmask)
   DO jk=2,jpkm1; DO jj=1,2; DO ji=1,3
      CALL r328_zp_ddm_cell(ji,jj,jk,wmask(ji,jj,jk-1), &
         & 2._wp*wmask(ji,jj,jk-1),2._wp*wmask(ji,jj,jk-1))
   END DO; END DO; END DO
   avm=(avm+2._wp)*wmask; avs=(avt+2._wp)*wmask; avt=(avt+1._wp)*wmask
   CALL r328_zp_boundary2(avm,avt,avs)
   DO jk=2,jpkm1
      wave=0.5_wp*wmask(:,:,jk-1); emx=REAL(jk,wp); reb=10._wp*jk
      CALL r328_zp_iwm_level(jk,wave,emx,reb)
   END DO
   avm=(avm+0.5_wp)*wmask; avt=(avt+0.5_wp)*wmask
   avs=(avs+0.25_wp)*wmask
   avm(3,2,3)=-0._wp; avt(3,2,3)=-0._wp; avs(3,2,3)=-0._wp
   CALL r328_zp_finish(avm,avt,avs)

   full_avm=0._wp; interior=0._wp; zero3=0._wp
   full_avm(3:5,3:4,2:4)=avm
   interior(:,:,2:4)=avt
   full_avm(5,4,4)=0._wp; interior(3,2,4)=0._wp
   magic='NEMO_L4_ZDF___2'
   OPEN(NEWUNIT=unit,FILE='oracle_zdf_entry_kt00000001.bin',ACCESS='STREAM', &
      & FORM='UNFORMATTED',STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
   IF(ios /= 0) ERROR STOP 2
   WRITE(unit) magic,2,1,1,7,6,jpk,64
   WRITE(unit) full_avm,interior
   interior=0._wp; interior(:,:,2:4)=avs
   interior(3,2,4)=0._wp
   WRITE(unit) interior,zero3
   CLOSE(unit)
END PROGRAM zdfphy_round328_layout_fixture
