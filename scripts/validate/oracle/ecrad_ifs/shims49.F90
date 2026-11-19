! Module-name shims for the 49r1 ecRad oracle build (no code, no values):
! 49r1 RRTMG sources USE YOMLUN/YOMMP0/YOMCST; the serial stand-ins from
! ecRad 1.7.2 are named YOMLUN_ECRAD/YOMCST_ECRAD. YOMMP0 is the serial task layout.
MODULE YOMLUN
USE YOMLUN_ECRAD
END MODULE YOMLUN
MODULE YOMMP0
! serial harness: one task, task 1 reads the RRTMG tables (rrtm_kgb*/srtm_kgb*)
INTEGER :: NPROC = 1
INTEGER :: MYPROC = 1
END MODULE YOMMP0
MODULE YOMCST
USE YOMCST_ECRAD
END MODULE YOMCST
