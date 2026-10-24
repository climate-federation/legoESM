PROGRAM write_round332_fixture
   USE dom_oce
   USE round332_writer
   IMPLICIT NONE
   INTEGER :: ji,jj,jk,rank
   CHARACTER(LEN=16) :: arg
   REAL(wp) :: rank_offset,add
   REAL(wp), ALLOCATABLE :: before(:,:,:),added(:,:,:),after(:,:,:),vwet(:,:,:)
   REAL(wp), ALLOCATABLE :: ztsw(:,:),ztse(:,:),ztnw(:,:),ztne(:,:),zwx(:,:)
   REAL(wp), ALLOCATABLE :: row_m(:,:),row_d(:,:),row_p(:,:)

   CALL get_command_argument(1,arg)
   READ(arg,*) rank
   narea=rank+1
   rank_offset=REAL(10*rank,wp)
   ALLOCATE(before(ntei-ntsi+1,ntej-ntsj+1,jpkm1))
   ALLOCATE(added(ntei-ntsi+1,ntej-ntsj+1,jpkm1))
   ALLOCATE(after(ntei-ntsi+1,ntej-ntsj+1,jpkm1))
   ALLOCATE(vwet(ntei-ntsi+1,ntej-ntsj+1,jpkm1))
   ALLOCATE(ztsw(ntsi-1:ntei+1,ntsj-1:ntej+1))
   ALLOCATE(ztse(ntsi-1:ntei+1,ntsj-1:ntej+1))
   ALLOCATE(ztnw(ntsi-1:ntei+1,ntsj-1:ntej+1))
   ALLOCATE(ztne(ntsi-1:ntei+1,ntsj-1:ntej+1))
   ALLOCATE(zwx(ntsi-1:ntei+1,ntsj-1:ntej+1))
   ALLOCATE(row_m(ntei-ntsi+1,jpkm1),row_d(ntei-ntsi+1,jpkm1))
   ALLOCATE(row_p(ntei-ntsi+1,jpkm1-1))
   ztsw=1._wp+rank_offset
   ztse=2._wp+rank_offset
   ztnw=3._wp+rank_offset
   ztne=4._wp+rank_offset
   zwx=0.125_wp
   ! This source cell makes the literal zva expression below evaluate to -0.
   zwx(ntsi-1:ntsi,ntsj:ntsj+1)=0._wp
   before=5._wp+rank_offset
   vwet=1._wp
   vwet(2,1,1)=0._wp
   before(1,1,:)=-0._wp
   after=before
   DO jk=1,jpkm1
      DO jj=ntsj,ntej
         DO ji=ntsi,ntei
            add = - (1._wp/12._wp) * &
               & (  ( ztsw(ji,jj+1) * zwx(ji-1,jj+1) + ztse(ji,jj+1) * zwx(ji,jj+1) ) &
               &  + ( ztnw(ji,jj) * zwx(ji-1,jj) + ztne(ji,jj) * zwx(ji,jj) ) )
            added(ji-ntsi+1,jj-ntsj+1,jk)=add
            after(ji-ntsi+1,jj-ntsj+1,jk)=before(ji-ntsi+1,jj-ntsj+1,jk)+add
         END DO
      END DO
   END DO
   CALL r332_vor_begin(1,1,3,before,vwet)
   CALL r332_vor_add(added)
   CALL r332_vor_done(after)

   DO jj=ntsj,ntej
      row_m=2._wp+rank_offset+REAL(jj,wp)
      row_d=3._wp+rank_offset+REAL(jj,wp)
      row_p=0.5_wp
      row_p(ntei-ntsi+1,jpkm1-1)=1._wp
      IF(jj==ntsj) THEN
         row_m(1,:)=-0._wp
         row_d(1,:)=-0._wp
         row_p(1,:)=-0._wp
      ENDIF
      CALL r332_tke_row(1,1,jj,row_m,row_d,row_p)
   END DO
END PROGRAM write_round332_fixture
