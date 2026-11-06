! Standalone driver: AeroBulk 1.0.0 TURB_COARE3P0 and TURB_NCAR on identical inputs.
! Usage: aerobulk_drv <infile> <outfile> <niter>
! infile: line 1 = N zt zu ; then N rows: sst t_zt q_s q_zt U slp
!   sst  : surface (potential) temperature [K]
!   t_zt : POTENTIAL air temperature at zt [K]
!   q_s  : surface specific humidity [kg/kg]     (passed as-is; no q_sat recomputed)
!   q_zt : air specific humidity at zt [kg/kg]
!   U    : scalar wind speed at zu [m/s]
!   slp  : pressure [Pa] (only used by the helper functions q_sat/rho below)
! outfile: one row per point, COARE3P0 block then NCAR block, then helpers.
PROGRAM aerobulk_drv
   USE mod_const
   USE mod_phymbl
   USE mod_blk_coare3p0
   USE mod_blk_ncar
   IMPLICIT NONE
   INTEGER :: n, i, niter
   REAL(wp) :: zt, zu
   CHARACTER(len=512) :: fin, fout, cniter
   REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: sst, tzt, qs, qzt, U, slp, &
      & Cd, Ch, Ce, tzu, qzu, Ub, CdN, ChN, CeN, z0, us, xL, UN10, sst2, qs2
   CALL GET_COMMAND_ARGUMENT(1, fin)
   CALL GET_COMMAND_ARGUMENT(2, fout)
   CALL GET_COMMAND_ARGUMENT(3, cniter)
   READ(cniter,*) niter
   nb_iter = niter
   OPEN(10, file=TRIM(fin), status='old')
   READ(10,*) n, zt, zu
   ALLOCATE(sst(n,1), tzt(n,1), qs(n,1), qzt(n,1), U(n,1), slp(n,1), Cd(n,1), Ch(n,1), Ce(n,1), &
      & tzu(n,1), qzu(n,1), Ub(n,1), CdN(n,1), ChN(n,1), CeN(n,1), z0(n,1), us(n,1), xL(n,1), UN10(n,1), &
      & sst2(n,1), qs2(n,1))
   DO i = 1, n
      READ(10,*) sst(i,1), tzt(i,1), qs(i,1), qzt(i,1), U(i,1), slp(i,1)
   END DO
   CLOSE(10)
   OPEN(20, file=TRIM(fout), status='replace')
   WRITE(20,'(a,i6,a,es24.16,a,es24.16,a,i4,a,es24.16,a,es24.16)') '# N=', n, ' grav=', grav, ' rctv0=', rctv0, &
      & ' nb_iter=', nb_iter, ' vkarmn=', vkarmn, ' Cx_min=', Cx_min
   ! --- COARE 3.0 (no skin) ---
   sst2 = sst ; qs2 = qs
   CALL TURB_COARE3P0( 1, zt, zu, sst2, tzt, qs2, qzt, U, .FALSE., .FALSE., &
      &                Cd, Ch, Ce, tzu, qzu, Ub, CdN=CdN, ChN=ChN, CeN=CeN, xz0=z0, xu_star=us, xL=xL, xUN10=UN10 )
   DO i = 1, n
      WRITE(20,'(a,13es25.16e3)') 'C ', Cd(i,1), Ch(i,1), Ce(i,1), tzu(i,1), qzu(i,1), Ub(i,1), &
         & CdN(i,1), ChN(i,1), CeN(i,1), z0(i,1), us(i,1), xL(i,1), UN10(i,1)
   END DO
   ! --- NCAR (Large & Yeager 2004/2009) ---
   sst2 = sst ; qs2 = qs
   CALL TURB_NCAR( zt, zu, sst2, tzt, qs2, qzt, U, Cd, Ch, Ce, tzu, qzu, Ub, &
      &            CdN=CdN, ChN=ChN, CeN=CeN, xz0=z0, xu_star=us, xL=xL, xUN10=UN10 )
   DO i = 1, n
      WRITE(20,'(a,13es25.16e3)') 'N ', Cd(i,1), Ch(i,1), Ce(i,1), tzu(i,1), qzu(i,1), Ub(i,1), &
         & CdN(i,1), ChN(i,1), CeN(i,1), z0(i,1), us(i,1), xL(i,1), UN10(i,1)
   END DO
   ! --- thermodynamic helpers used by BULK_FORMULA / aerobulk_compute ---
   DO i = 1, n
      WRITE(20,'(a,5es25.16e3)') 'H ', cp_air(qzt(i,1)), L_vap(sst(i,1)), q_sat(sst(i,1), slp(i,1)), &
         & visc_air(tzt(i,1)), rho_air(tzt(i,1), qzt(i,1), slp(i,1))
   END DO
   CLOSE(20)
END PROGRAM aerobulk_drv
