"""Literal NumPy transcription of CAM6 ``clubb_intr.F90:2345-2376``
(cam_cesm2_1_rel), the non-``zmconv_microp`` branch with the shallow
detrainment ``dlf2 = 0``: the host's treatment of the Zhang-McFarlane
detrained condensate ``dlf``.  Constants are CAM's literals, so a test
against this function certifies the port, not legoESM's constants.
"""
from __future__ import annotations

import numpy as np

LATICE = 3.337e5     # shr_const_latice [J/kg]
CPAIR = 1.00464e3    # shr_const_cpdair [J/kg/K]  # const-ok: CAM cpair literal (oracle)


def cam6_clubb_detrain_oracle(dlf, t):
    """Returns (dq_c, dq_i, dN_c, dN_i, dT_dt) exactly as the Fortran loop."""
    dlf = np.asarray(dlf, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    dlf2 = np.zeros_like(dlf)
    dum1 = np.where(t > 268.15, 0.0, np.where(t < 238.15, 1.0, (268.15 - t) / 30.0))
    dq_c = dlf * (1.0 - dum1)
    dq_i = dlf * dum1
    dn_c = (3.0 * (np.maximum(0.0, dlf - dlf2) * (1.0 - dum1))
            / (4.0 * 3.14 * 8.e-6 ** 3 * 997.0)
            + 3.0 * (dlf2 * (1.0 - dum1)) / (4.0 * 3.14 * 10.e-6 ** 3 * 997.0))
    dn_i = (3.0 * (np.maximum(0.0, dlf - dlf2) * dum1)
            / (4.0 * 3.14 * 25.e-6 ** 3 * 500.0)
            + 3.0 * (dlf2 * dum1) / (4.0 * 3.14 * 50.e-6 ** 3 * 500.0))
    s = dlf * dum1 * LATICE
    return dq_c, dq_i, dn_c, dn_i, s / CPAIR
