! Minimal stand-ins for the OpenIFS 49r1 modules that the IFS-ecRad interface
! routines (radiation_scheme, su_radiation_scheme, liquid/ice_effective_radius,
! cloud_overlap_decorr_len, radheatn) USE but whose real 49r1 versions drag in
! the whole model (MPI, OOPS model object, SPP pattern generator). YOM_YGFL
! (TYPE_GFLD incl. NACTAERO) is the core build's ecRad 1.7.2 copy, reused. Each type carries ONLY the members those routines reference;
! every member is set by ifs49_driver.F90 (values cited there) or carries the
! 49r1 type's own default (cited). The real 49r1 TERAD, TRADIATION, TEAERATM,
! TERDI, TECLDP, TECLD, TEPHY, TEPHLI, SPP_PERT/SPP_MODEL types are used inside.

MODULE MPL_MYRANK_MOD
! serial harness: only reached when RADIATION_SCHEME writes bad-flux files
! (NDUMPBADINPUTS/NDUMPINPUTS /= 0), which the harness never enables
IMPLICIT NONE
CONTAINS
INTEGER FUNCTION MPL_MYRANK()
  MPL_MYRANK = 1
END FUNCTION MPL_MYRANK
END MODULE MPL_MYRANK_MOD

MODULE YOMRIP
USE PARKIND1, ONLY : JPRB
IMPLICIT NONE
TYPE TRIP
  REAL(KIND=JPRB) :: TSTEP         ! model timestep (s) (yomrip.F90:104)
END TYPE TRIP
END MODULE YOMRIP

MODULE SPP_MOD
! TSPP_CONFIG with the 49r1 SPP_MODEL type (real spp_gen_mod.F90) and a
! pointer record holding only the perturbation pointers the interface reads
! (all read inside IF (LSPP) only; LSPP=.FALSE. is the TSPP_CONFIG default,
! spp_mod.F90:61, and stochastic physics is off, user decision A18)
USE PARKIND1, ONLY : JPIM, JPRB
USE SPP_GEN_MOD, ONLY : SPP_MODEL
IMPLICIT NONE
TYPE TSPP_PPTR
  INTEGER(KIND=JPIM) :: ZDECORR = -1, ZSIGQCW = -1, ZRADEFFL = -1, ZRADEFFI = -1, PHR = -1
END TYPE TSPP_PPTR
TYPE TSPP_CONFIG
  LOGICAL :: LSPP = .FALSE.
  TYPE(SPP_MODEL) :: SM
  TYPE(TSPP_PPTR) :: PPTR
  REAL(KIND=JPRB) :: XPRESS_PHR_ST
END TYPE TSPP_CONFIG
END MODULE SPP_MOD

MODULE TYPE_MODEL
! the members of the 49r1 MODEL object (type_model.F90) that the interface reads
USE YOERAD, ONLY : TERAD
USE YOERADIATION, ONLY : TRADIATION
USE YOEAERATM, ONLY : TEAERATM
USE YOERDI, ONLY : TERDI
USE YOECLDP, ONLY : TECLDP
USE YOECLD, ONLY : TECLD
USE YOEPHY, ONLY : TEPHY
USE YOEPHLI, ONLY : TEPHLI
USE SPP_MOD, ONLY : TSPP_CONFIG
USE YOM_YGFL, ONLY : TYPE_GFLD
USE YOMRIP, ONLY : TRIP
IMPLICIT NONE
TYPE TML_PHY_RAD
  TYPE(TERAD) :: YRERAD
  TYPE(TRADIATION) :: YRADIATION
  TYPE(TEAERATM) :: YREAERATM
  TYPE(TERDI) :: YRERDI
END TYPE TML_PHY_RAD
TYPE TML_PHY_EC
  TYPE(TECLDP) :: YRECLDP
  TYPE(TECLD) :: YRECLD
  TYPE(TEPHY) :: YREPHY
END TYPE TML_PHY_EC
TYPE TML_GCONF
  TYPE(TSPP_CONFIG) :: YRSPP_CONFIG
  TYPE(TYPE_GFLD) :: YGFL
  TYPE(TRIP) :: YRRIP
END TYPE TML_GCONF
TYPE TML_PHY_SLIN
  TYPE(TEPHLI) :: YREPHLI
END TYPE TML_PHY_SLIN
TYPE MODEL
  TYPE(TML_PHY_RAD) :: YRML_PHY_RAD
  TYPE(TML_PHY_EC) :: YRML_PHY_EC
  TYPE(TML_GCONF) :: YRML_GCONF
  TYPE(TML_PHY_SLIN) :: YRML_PHY_SLIN
END TYPE MODEL
END MODULE TYPE_MODEL
