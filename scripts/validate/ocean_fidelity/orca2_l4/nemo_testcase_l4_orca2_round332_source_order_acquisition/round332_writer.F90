MODULE round332_writer
   !! WRITE-only kt=1 source-order and kt=2 TKE discriminators.
   USE dom_oce
   USE in_out_manager, ONLY : nit000
   USE dynzdf_round328_writer, ONLY : checksum3
   USE lib_mpp, ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r332_vor_begin, r332_vor_add, r332_vor_done
   PUBLIC :: r332_tke_begin, r332_tke_row, r332_raw_postfold_v

   INTEGER, SAVE :: nkt=-1,nkmm=-1,nkrhs=-1,nx=0,ny=0,nz=0,nrows=0
   INTEGER, SAVE :: nrows2=0
   INTEGER, SAVE :: current_kt=-1,current_kbb=-1,current_kmm=-1
   LOGICAL, SAVE :: active=.FALSE.,have_add=.FALSE.,have_vor=.FALSE.,written=.FALSE.
   LOGICAL, SAVE :: written2=.FALSE.
   LOGICAL, SAVE :: raw_written=.FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: vor_before(:,:,:),vor_add(:,:,:),vor_after(:,:,:),vor_mask(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: mxlm(:,:,:),mxld(:,:,:),pdlr(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: mxlm2(:,:,:),mxld2(:,:,:),pdlr2(:,:,:)

CONTAINS

   SUBROUTINE allocate_arrays
      IF(ALLOCATED(vor_before)) RETURN
      nx=ntei-ntsi+1; ny=ntej-ntsj+1; nz=jpkm1
      ALLOCATE(vor_before(nx,ny,nz),vor_add(nx,ny,nz),vor_after(nx,ny,nz),vor_mask(nx,ny,nz), &
         & mxlm(nx,ny,nz),mxld(nx,ny,nz),pdlr(nx,ny,nz), &
         & mxlm2(nx,ny,nz),mxld2(nx,ny,nz),pdlr2(nx,ny,nz))
      vor_before=0._wp; vor_add=0._wp; vor_after=0._wp
      vor_mask=0._wp; mxlm=0._wp; mxld=0._wp; pdlr=0._wp
      mxlm2=0._wp; mxld2=0._wp; pdlr2=0._wp
   END SUBROUTINE allocate_arrays

   SUBROUTINE r332_vor_begin(kt,Kmm,Krhs,pbefore,pvmask)
      INTEGER, INTENT(in) :: kt,Kmm,Krhs
      REAL(wp), INTENT(in) :: pbefore(:,:,:),pvmask(:,:,:)
      IF(kt /= nit000 .OR. have_vor .OR. active) RETURN
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round332: requires fp64')
      CALL allocate_arrays
      IF(nkt /= -1 .AND. (nkt /= kt .OR. nkmm /= Kmm)) &
         & CALL ctl_stop('round332: TKE/vorticity indices moved')
      IF(ANY(SHAPE(pbefore) /= (/nx,ny,nz/)) .OR. &
         & ANY(SHAPE(pvmask) /= (/nx,ny,nz/))) &
         & CALL ctl_stop('round332: vorticity before/mask shape')
      nkt=kt; nkmm=Kmm; nkrhs=Krhs
      vor_before=pbefore; vor_mask=pvmask
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

   SUBROUTINE r332_tke_begin(kt,Kbb,Kmm)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm
      IF(kt == nit000 .OR. kt == nit000+1) THEN
         current_kt=kt; current_kbb=Kbb; current_kmm=Kmm
      ELSE
         current_kt=-1; current_kbb=-1; current_kmm=-1
      ENDIF
   END SUBROUTINE r332_tke_begin

   SUBROUTINE r332_tke_row(Kbb,Kmm,jj,pmxlm,pmxld,ppdlr)
      INTEGER, INTENT(in) :: Kbb,Kmm,jj
      REAL(wp), INTENT(in) :: pmxlm(:,:),pmxld(:,:),ppdlr(:,:)
      INTEGER :: j,record_kt
      record_kt=current_kt
      IF(record_kt == -1) RETURN
      IF(Kbb /= current_kbb .OR. Kmm /= current_kmm) &
         & CALL ctl_stop('round336: TKE saved levels moved')
      IF(record_kt == nit000 .AND. written) RETURN
      IF(record_kt == nit000+1 .AND. written2) RETURN
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round332: requires fp64')
      CALL allocate_arrays
      j=jj-ntsj+1
      IF(j<1 .OR. j>ny .OR. ANY(SHAPE(pmxlm) /= (/nx,nz/)) .OR. &
         & ANY(SHAPE(pmxld) /= (/nx,nz/)) .OR. ANY(SHAPE(ppdlr) /= (/nx,nz-1/))) &
         & CALL ctl_stop('round332: TKE row shape')
      IF(record_kt == nit000) THEN
         IF(nkt == -1) THEN
            nkt=record_kt; nkmm=Kmm
         ELSE IF(nkt /= record_kt .OR. nkmm /= Kmm) THEN
            CALL ctl_stop('round332: TKE indices moved')
         ENDIF
         IF(Kbb /= 1 .OR. Kmm /= 1) CALL ctl_stop('round332: kt1 TKE levels moved')
         mxlm(:,j,:)=pmxlm; mxld(:,j,:)=pmxld; pdlr(:,j,2:nz)=ppdlr
         nrows=nrows+1
         IF(nrows > ny) CALL ctl_stop('round332: duplicate TKE row')
         CALL maybe_write
      ELSE
         IF(Kbb /= 3 .OR. Kmm /= 3) CALL ctl_stop('round336: kt2 TKE levels moved')
         mxlm2(:,j,:)=pmxlm; mxld2(:,j,:)=pmxld; pdlr2(:,j,2:nz)=ppdlr
         nrows2=nrows2+1
         IF(nrows2 > ny) CALL ctl_stop('round336: duplicate kt2 TKE row')
         IF(nrows2 == ny) CALL write_tke2(record_kt,Kbb,Kmm)
      ENDIF
   END SUBROUTINE r332_tke_row

   SUBROUTINE write_tke2(kt,Kbb,Kmm)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm
      INTEGER :: unit,ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=96) :: filename
      IF(written2) CALL ctl_stop('round336: duplicate kt2 TKE record')
      WRITE(filename,'("oracle_r336_tke_mxl_kt",I8.8,"_r",I4.4,".bin")') kt,narea-1
      OPEN(NEWUNIT=unit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round336: cannot open kt2 TKE record')
      magic='NEMO_L4_R336TK2'
      WRITE(unit) magic,1,kt,Kbb,Kmm,narea-1,nx,ny,nz,ntsi,ntsj, &
         & STORAGE_SIZE(1._wp),3
      CALL put3(unit,'tke_zmxlm',mxlm2,1)
      CALL put3(unit,'tke_zmxld',mxld2,1)
      CALL put3(unit,'tke_pdlr',pdlr2,1)
      CLOSE(unit)
      written2=.TRUE.
   END SUBROUTINE write_tke2

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
         & STORAGE_SIZE(1._wp),7
      CALL put3(unit,'stage1_v_before',vor_before,1)
      CALL put3(unit,'stage1_v_add',vor_add,1)
      CALL put3(unit,'stage1_v_after',vor_after,1)
      CALL put3(unit,'stage1_v_mask',vor_mask,1)
      CALL put3(unit,'tke_zmxlm',mxlm,1)
      CALL put3(unit,'tke_zmxld',mxld,1)
      CALL put3(unit,'tke_pdlr',pdlr,1)
      CLOSE(unit)
      written=.TRUE.
   END SUBROUTINE maybe_write

   SUBROUTINE r332_raw_postfold_v(kt,kstg,Kaa,pv)
      INTEGER, INTENT(in) :: kt,kstg,Kaa
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: pv
      INTEGER :: unit,ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=24) :: field
      CHARACTER(LEN=96) :: filename
      IF(kt /= nit000 .OR. kstg /= 1 .OR. raw_written) RETURN
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round332: raw V requires fp64')
      WRITE(filename,'("oracle_r332_raw_v_kt",I8.8,"_r",I4.4,".bin")') kt,narea-1
      OPEN(NEWUNIT=unit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round332: cannot open raw V record')
      magic='NEMO_L4_R332RV'
      WRITE(unit) magic,1,kt,kstg,Kaa,narea-1,jpi,jpj,jpk,STORAGE_SIZE(1._wp),1
      field='raw_postfold_v'
      WRITE(unit) field
      WRITE(unit) 3,jpi,jpj,jpk,1,1,1
      WRITE(unit) checksum3(pv)
      WRITE(unit) pv
      CLOSE(unit)
      raw_written=.TRUE.
   END SUBROUTINE r332_raw_postfold_v

END MODULE round332_writer
