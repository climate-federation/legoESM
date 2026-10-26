MODULE round336_trazdf_writer
   !! Passive, outside-loop replay of the kt=2 stage-3 TRA ZDF solve.
   USE, INTRINSIC :: iso_fortran_env, ONLY : int64
   USE par_kind       , ONLY : wp
   USE par_oce        , ONLY : jpi,jpj,jpk,jpkm1,jpts,jp_tem,jp_sal
   USE dom_oce        , ONLY : narea,ntsi,ntei,ntsj,ntej,e3t_3d,e3w_1d,r3t,tmask
   USE in_out_manager , ONLY : nit000
   USE zdf_oce        , ONLY : avt,avs
   USE ldfslp         , ONLY : akz
   USE lib_mpp        , ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r336_begin,r336_capture
   REAL(wp), ALLOCATABLE, SAVE :: saved_krhs(:,:,:,:)

CONTAINS

   SUBROUTINE r336_begin(kt,Krhs,pt)
      INTEGER, INTENT(in) :: kt,Krhs
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(in) :: pt
      IF(kt /= nit000+1) RETURN
      IF(ALLOCATED(saved_krhs)) CALL ctl_stop('round336: duplicate trazdf begin')
      IF(SIZE(pt,1)/=jpi .OR. SIZE(pt,2)/=jpj .OR. SIZE(pt,3)/=jpk .OR. &
         & SIZE(pt,4)/=jpts) CALL ctl_stop('round336: trazdf begin extent moved')
      ALLOCATE(saved_krhs(jpi,jpj,jpk,jpts))
      saved_krhs=pt(:,:,:,:,Krhs)
   END SUBROUTINE r336_begin

   SUBROUTINE r336_capture(kt,Kbb,Kmm,Krhs,Kaa,p2dt,pt,ld_aimp,ld_mfc,ld_ddm,ld_slp,ld_msc)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm,Krhs,Kaa
      REAL(wp), INTENT(in) :: p2dt
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(in) :: pt
      LOGICAL, INTENT(in) :: ld_aimp,ld_mfc,ld_ddm,ld_slp,ld_msc
      REAL(wp), ALLOCATABLE :: effective(:,:,:,:),e3w_kmm(:,:,:)
      REAL(wp), ALLOCATABLE :: e3t_bb(:,:,:),e3t_mm(:,:,:),e3t_aa(:,:,:)
      REAL(wp), ALLOCATABLE :: kbb_field(:,:,:,:),krhs_field(:,:,:,:),content(:,:,:,:)
      REAL(wp), ALLOCATABLE :: lower(:,:,:,:),diagonal(:,:,:,:),upper(:,:,:,:)
      REAL(wp), ALLOCATABLE :: eliminated(:,:,:,:),forward(:,:,:,:),backsub(:,:,:,:)
      REAL(wp), ALLOCATABLE :: actual(:,:,:,:),seen(:,:,:,:)
      INTEGER :: unit,ios,ji,jj,jk,jn
      REAL(wp) :: zrhs
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=96) :: filename

      IF(kt /= nit000+1) RETURN
      IF(.NOT.ALLOCATED(saved_krhs)) CALL ctl_stop('round336: missing trazdf begin')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round336: trazdf record requires fp64')
      IF(ld_aimp .OR. ld_mfc .OR. .NOT.ld_ddm .OR. .NOT.ld_slp .OR. .NOT.ld_msc) &
         & CALL ctl_stop('round336: trazdf selector set moved')
      IF(SIZE(pt,1)/=jpi .OR. SIZE(pt,2)/=jpj .OR. SIZE(pt,3)/=jpk .OR. &
         & SIZE(pt,4)/=jpts) CALL ctl_stop('round336: trazdf input extent moved')
      ALLOCATE(effective(jpi,jpj,jpk,jpts),e3w_kmm(jpi,jpj,jpk))
      ALLOCATE(e3t_bb(jpi,jpj,jpk),e3t_mm(jpi,jpj,jpk),e3t_aa(jpi,jpj,jpk))
      ALLOCATE(kbb_field(jpi,jpj,jpk,jpts),krhs_field(jpi,jpj,jpk,jpts), &
         & content(jpi,jpj,jpk,jpts),lower(jpi,jpj,jpk,jpts), &
         & diagonal(jpi,jpj,jpk,jpts),upper(jpi,jpj,jpk,jpts), &
         & eliminated(jpi,jpj,jpk,jpts),forward(jpi,jpj,jpk,jpts), &
         & backsub(jpi,jpj,jpk,jpts),actual(jpi,jpj,jpk,jpts), &
         & seen(jpi,jpj,jpk,jpts))

      effective=-0._wp; e3w_kmm=-0._wp
      e3t_bb=-0._wp; e3t_mm=-0._wp; e3t_aa=-0._wp
      kbb_field=pt(:,:,:,:,Kbb); krhs_field=saved_krhs
      content=-0._wp; lower=-0._wp; diagonal=-0._wp; upper=-0._wp
      eliminated=-0._wp; forward=-0._wp; backsub=-0._wp
      actual=pt(:,:,:,:,Kaa); seen=0._wp

      DO jk=1,jpk; DO jj=1,jpj; DO ji=1,jpi
         e3w_kmm(ji,jj,jk)=e3w_1d(jk)*(1._wp+r3t(ji,jj,Kmm))
         e3t_bb(ji,jj,jk)=e3t_3d(ji,jj,jk)*(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,jk))
         e3t_mm(ji,jj,jk)=e3t_3d(ji,jj,jk)*(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk))
         e3t_aa(ji,jj,jk)=e3t_3d(ji,jj,jk)*(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))
      END DO; END DO; END DO

      DO jj=ntsj,ntej
         DO jn=1,jpts
            effective(:,jj,1,jn)=0._wp
            IF(jn==jp_tem) THEN
               DO jk=2,jpk; DO ji=ntsi,ntei
                  effective(ji,jj,jk,jn)=avt(ji,jj,jk)+akz(ji,jj,jk)
               END DO; END DO
            ELSE IF(jn==jp_sal) THEN
               DO jk=2,jpk; DO ji=ntsi,ntei
                  effective(ji,jj,jk,jn)=avs(ji,jj,jk)+akz(ji,jj,jk)
               END DO; END DO
            ELSE
               CALL ctl_stop('round336: active tracer count moved')
            ENDIF
            DO jk=1,jpkm1; DO ji=ntsi,ntei
               lower(ji,jj,jk,jn)=-p2dt*effective(ji,jj,jk,jn)/ &
                  & (e3w_1d(jk)*(1._wp+r3t(ji,jj,Kmm)))
               upper(ji,jj,jk,jn)=-p2dt*effective(ji,jj,jk+1,jn)/ &
                  & (e3w_1d(jk+1)*(1._wp+r3t(ji,jj,Kmm)))
               diagonal(ji,jj,jk,jn)=e3t_3d(ji,jj,jk)* &
                  & (1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))- &
                  & (lower(ji,jj,jk,jn)+upper(ji,jj,jk,jn))
            END DO; END DO
            DO ji=ntsi,ntei
               eliminated(ji,jj,1,jn)=diagonal(ji,jj,1,jn)
            END DO
            DO jk=2,jpkm1; DO ji=ntsi,ntei
               eliminated(ji,jj,jk,jn)=diagonal(ji,jj,jk,jn)- &
                  & lower(ji,jj,jk,jn)*upper(ji,jj,jk-1,jn)/eliminated(ji,jj,jk-1,jn)
            END DO; END DO
            DO ji=ntsi,ntei
               content(ji,jj,1,jn)=e3t_3d(ji,jj,1)* &
                  & (1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,1))*kbb_field(ji,jj,1,jn)+ &
                  & p2dt*(e3t_3d(ji,jj,1)* &
                  & (1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,1)))*krhs_field(ji,jj,1,jn)
               forward(ji,jj,1,jn)=content(ji,jj,1,jn)
            END DO
            DO jk=2,jpkm1; DO ji=ntsi,ntei
               zrhs=e3t_3d(ji,jj,jk)*(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,jk))* &
                  & kbb_field(ji,jj,jk,jn)+p2dt*(e3t_3d(ji,jj,jk)* &
                  & (1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))*krhs_field(ji,jj,jk,jn)
               content(ji,jj,jk,jn)=zrhs
               forward(ji,jj,jk,jn)=zrhs-lower(ji,jj,jk,jn)/ &
                  & eliminated(ji,jj,jk-1,jn)*forward(ji,jj,jk-1,jn)
            END DO; END DO
            DO ji=ntsi,ntei
               backsub(ji,jj,jpkm1,jn)=forward(ji,jj,jpkm1,jn)/ &
                  & eliminated(ji,jj,jpkm1,jn)*tmask(ji,jj,jpkm1)
               seen(ji,jj,jpkm1,jn)=1._wp
            END DO
            DO jk=jpk-2,1,-1; DO ji=ntsi,ntei
               backsub(ji,jj,jk,jn)=(forward(ji,jj,jk,jn)- &
                  & upper(ji,jj,jk,jn)*backsub(ji,jj,jk+1,jn))/ &
                  & eliminated(ji,jj,jk,jn)*tmask(ji,jj,jk)
               seen(ji,jj,jk,jn)=1._wp
            END DO; END DO
         END DO
      END DO

      WRITE(filename,'("oracle_r336_trazdf_kt",I8.8,"_r",I4.4,".bin")') kt,narea-1
      OPEN(NEWUNIT=unit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios/=0) CALL ctl_stop('round336: cannot open trazdf record')
      magic='NEMO_L4_R336TZ'
      WRITE(unit) magic,1,kt,Kbb,Kmm,Krhs,Kaa,narea-1,jpi,jpj,jpk,jpts, &
         & STORAGE_SIZE(1._wp),25,MERGE(1,0,ld_ddm),MERGE(1,0,ld_msc)
      CALL put3(unit,'avt',avt); CALL put3(unit,'avs',avs); CALL put3(unit,'akz',akz)
      CALL put3(unit,'e3t_3d',e3t_3d); CALL put1(unit,'e3w_1d',e3w_1d)
      CALL put2(unit,'r3t_Kbb',r3t(:,:,Kbb)); CALL put2(unit,'r3t_Kmm',r3t(:,:,Kmm))
      CALL put2(unit,'r3t_Kaa',r3t(:,:,Kaa))
      CALL put4(unit,'effective_K',effective); CALL put3(unit,'e3w_Kmm',e3w_kmm)
      CALL put3(unit,'e3t_Kbb',e3t_bb); CALL put3(unit,'e3t_Kmm',e3t_mm)
      CALL put3(unit,'e3t_Kaa',e3t_aa); CALL put4(unit,'Kbb_tracer',kbb_field)
      CALL put4(unit,'Krhs_tracer',krhs_field); CALL put4(unit,'content_rhs',content)
      CALL put4(unit,'lower',lower); CALL put4(unit,'diagonal',diagonal)
      CALL put4(unit,'upper',upper); CALL put4(unit,'eliminated',eliminated)
      CALL put4(unit,'forward',forward); CALL put4(unit,'backsub',backsub)
      CALL put4(unit,'actual_Kaa',actual); CALL put4(unit,'solve_seen',seen)
      CALL put3(unit,'tmask',tmask)
      CLOSE(unit)
      DEALLOCATE(effective,e3w_kmm,e3t_bb,e3t_mm,e3t_aa,kbb_field,krhs_field, &
         & content,lower,diagonal,upper,eliminated,forward,backsub,actual,seen)
      DEALLOCATE(saved_krhs)
   END SUBROUTINE r336_capture

   SUBROUTINE put4(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: value
      CHARACTER(LEN=24) :: field
      field=name; WRITE(unit) field
      WRITE(unit) 4,SIZE(value,1),SIZE(value,2),SIZE(value,3),SIZE(value,4),1,1,1,1
      WRITE(unit) checksum4(value); WRITE(unit) value
   END SUBROUTINE put4

   SUBROUTINE put3(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=24) :: field
      field=name; WRITE(unit) field
      WRITE(unit) 3,SIZE(value,1),SIZE(value,2),SIZE(value,3),1,1,1,1,1
      WRITE(unit) checksum3(value); WRITE(unit) value
   END SUBROUTINE put3

   SUBROUTINE put2(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      CHARACTER(LEN=24) :: field
      field=name; WRITE(unit) field
      WRITE(unit) 2,SIZE(value,1),SIZE(value,2),1,1,1,1,1,1
      WRITE(unit) checksum2(value); WRITE(unit) value
   END SUBROUTINE put2

   SUBROUTINE put1(unit,name,value)
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:), INTENT(in) :: value
      CHARACTER(LEN=24) :: field
      field=name; WRITE(unit) field
      WRITE(unit) 1,SIZE(value,1),1,1,1,1,1,1,1
      WRITE(unit) checksum1(value); WRITE(unit) value
   END SUBROUTINE put1

   FUNCTION checksum4(value) RESULT(checksum)
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: value
      INTEGER(int64) :: checksum,word
      INTEGER :: i,j,k,n
      checksum=0_int64
      DO n=1,SIZE(value,4); DO k=1,SIZE(value,3); DO j=1,SIZE(value,2); DO i=1,SIZE(value,1)
         word=TRANSFER(value(i,j,k,n),word); checksum=IEOR(checksum,word)
      END DO; END DO; END DO; END DO
   END FUNCTION checksum4

   FUNCTION checksum3(value) RESULT(checksum)
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      INTEGER(int64) :: checksum,word
      INTEGER :: i,j,k
      checksum=0_int64
      DO k=1,SIZE(value,3); DO j=1,SIZE(value,2); DO i=1,SIZE(value,1)
         word=TRANSFER(value(i,j,k),word); checksum=IEOR(checksum,word)
      END DO; END DO; END DO
   END FUNCTION checksum3

   FUNCTION checksum2(value) RESULT(checksum)
      REAL(wp), DIMENSION(:,:), INTENT(in) :: value
      INTEGER(int64) :: checksum,word
      INTEGER :: i,j
      checksum=0_int64
      DO j=1,SIZE(value,2); DO i=1,SIZE(value,1)
         word=TRANSFER(value(i,j),word); checksum=IEOR(checksum,word)
      END DO; END DO
   END FUNCTION checksum2

   FUNCTION checksum1(value) RESULT(checksum)
      REAL(wp), DIMENSION(:), INTENT(in) :: value
      INTEGER(int64) :: checksum,word
      INTEGER :: i
      checksum=0_int64
      DO i=1,SIZE(value,1)
         word=TRANSFER(value(i),word); checksum=IEOR(checksum,word)
      END DO
   END FUNCTION checksum1
END MODULE round336_trazdf_writer
