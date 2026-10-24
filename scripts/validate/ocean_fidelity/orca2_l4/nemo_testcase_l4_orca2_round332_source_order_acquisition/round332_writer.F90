MODULE round332_writer
   !! WRITE-only kt=1 source-order discriminator for ORCA2 round 332.
   USE dom_oce
   USE in_out_manager, ONLY : nit000
   USE dynzdf_round328_writer, ONLY : checksum3
   USE lib_mpp, ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r332_vor_begin, r332_vor_add, r332_vor_done
   PUBLIC :: r332_tke_row

   INTEGER, SAVE :: nkt=-1,nkmm=-1,nkrhs=-1,nx=0,ny=0,nz=0,nrows=0
   LOGICAL, SAVE :: active=.FALSE.,have_add=.FALSE.,have_vor=.FALSE.,written=.FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: vor_before(:,:,:),vor_add(:,:,:),vor_after(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: mxlm(:,:,:),mxld(:,:,:),pdlr(:,:,:)

CONTAINS

   SUBROUTINE allocate_arrays
      IF(ALLOCATED(vor_before)) RETURN
      nx=ntei-ntsi+1; ny=ntej-ntsj+1; nz=jpkm1
      ALLOCATE(vor_before(nx,ny,nz),vor_add(nx,ny,nz),vor_after(nx,ny,nz), &
         & mxlm(nx,ny,nz),mxld(nx,ny,nz),pdlr(nx,ny,nz))
      vor_before=0._wp; vor_add=0._wp; vor_after=0._wp
      mxlm=0._wp; mxld=0._wp; pdlr=0._wp
   END SUBROUTINE allocate_arrays

   SUBROUTINE r332_vor_begin(kt,Kmm,Krhs,pbefore)
      INTEGER, INTENT(in) :: kt,Kmm,Krhs
      REAL(wp), INTENT(in) :: pbefore(:,:,:)
      IF(kt /= nit000 .OR. have_vor .OR. active) RETURN
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round332: requires fp64')
      CALL allocate_arrays
      IF(nkt /= -1 .AND. (nkt /= kt .OR. nkmm /= Kmm)) &
         & CALL ctl_stop('round332: TKE/vorticity indices moved')
      IF(ANY(SHAPE(pbefore) /= (/nx,ny,nz/))) &
         & CALL ctl_stop('round332: vorticity before shape')
      nkt=kt; nkmm=Kmm; nkrhs=Krhs
      vor_before=pbefore
      active=.TRUE.
   END SUBROUTINE r332_vor_begin

   SUBROUTINE r332_vor_add(padd)
      REAL(wp), INTENT(in) :: padd(:,:,:)
      IF(.NOT.active) RETURN
      IF(have_add .OR. ANY(SHAPE(padd) /= (/nx,ny,nz/))) &
         & CALL ctl_stop('round332: vorticity replay shape')
      vor_add=padd
      have_add=.TRUE.
   END SUBROUTINE r332_vor_add

   SUBROUTINE r332_vor_done(pafter)
      REAL(wp), INTENT(in) :: pafter(:,:,:)
      IF(.NOT.active) RETURN
      IF(.NOT.have_add .OR. ANY(SHAPE(pafter) /= (/nx,ny,nz/))) &
         & CALL ctl_stop('round332: vorticity after shape')
      vor_after=pafter
      active=.FALSE.; have_vor=.TRUE.
      CALL maybe_write
   END SUBROUTINE r332_vor_done

   SUBROUTINE put3(unit,name,value,k0)
      INTEGER, INTENT(in) :: unit,k0
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value(:,:,:)
      CHARACTER(LEN=24) :: field
      field=name
      WRITE(unit) field
      WRITE(unit) 3,SIZE(value,1),SIZE(value,2),SIZE(value,3),ntsi,ntsj,k0
      WRITE(unit) checksum3(value)
      WRITE(unit) value
   END SUBROUTINE put3

   SUBROUTINE r332_tke_row(Kbb,Kmm,jj,pmxlm,pmxld,ppdlr)
      INTEGER, INTENT(in) :: Kbb,Kmm,jj
      REAL(wp), INTENT(in) :: pmxlm(:,:),pmxld(:,:),ppdlr(:,:)
      INTEGER :: j
      IF(written) RETURN
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round332: requires fp64')
      CALL allocate_arrays
      IF(nkt == -1) THEN
         nkt=nit000; nkmm=Kmm
      ELSE IF(nkt /= nit000 .OR. nkmm /= Kmm) THEN
         CALL ctl_stop('round332: TKE indices moved')
      ENDIF
      IF(Kbb /= 1) CALL ctl_stop('round332: unexpected TKE before index')
      j=jj-ntsj+1
      IF(j<1 .OR. j>ny .OR. ANY(SHAPE(pmxlm) /= (/nx,nz/)) .OR. &
         & ANY(SHAPE(pmxld) /= (/nx,nz/)) .OR. ANY(SHAPE(ppdlr) /= (/nx,nz-1/))) &
         & CALL ctl_stop('round332: TKE row shape')
      mxlm(:,j,:)=pmxlm; mxld(:,j,:)=pmxld; pdlr(:,j,2:nz)=ppdlr
      nrows=nrows+1
      IF(nrows > ny) CALL ctl_stop('round332: duplicate TKE row')
      CALL maybe_write
   END SUBROUTINE r332_tke_row

   SUBROUTINE maybe_write
      INTEGER :: unit,ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=96) :: filename
      IF(written .OR. .NOT.have_vor .OR. nrows /= ny) RETURN
      WRITE(filename,'("oracle_r332_source_kt",I8.8,"_r",I4.4,".bin")') nkt,narea-1
      OPEN(NEWUNIT=unit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round332: cannot open record')
      magic='NEMO_L4_R332SO'
      WRITE(unit) magic,1,nkt,nkmm,nkrhs,narea-1,nx,ny,nz,ntsi,ntsj, &
         & STORAGE_SIZE(1._wp),6
      CALL put3(unit,'stage1_v_before',vor_before,1)
      CALL put3(unit,'stage1_v_add',vor_add,1)
      CALL put3(unit,'stage1_v_after',vor_after,1)
      CALL put3(unit,'tke_zmxlm',mxlm,1)
      CALL put3(unit,'tke_zmxld',mxld,1)
      CALL put3(unit,'tke_pdlr',pdlr,1)
      CLOSE(unit)
      written=.TRUE.
   END SUBROUTINE maybe_write

END MODULE round332_writer
