"""RRTMG's band of each g-point, for feeding 49r1 Fortran stage dumps to ecRad's solvers.

The lane now runs RRTMGP g-points; these dumps were made with RRTMG.  Run lengths
are yoerrtftr.NGB and yoesrtm.NGBSW as the 49r1 oracle held them.
"""
import numpy as np

BAND_LW = np.repeat(np.arange(16), [10, 12, 16, 14, 16, 8, 12, 8, 12, 6, 8, 8, 4, 2, 2, 2])
BAND_SW = np.repeat(np.arange(14), [6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12])
