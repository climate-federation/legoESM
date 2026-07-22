"""
JAX/Python translation of the CLM orbital parameters module.

Holds the four orbital parameters shared across modules.
Populated by a call to :func:`shr_orb_mod.shr_orb_params` during
model initialization.

Original Fortran module: clm_varorb
"""

# Orbital eccentricity factor — Fortran line 13
eccen: float = 0.0

# Earth's obliquity in radians — Fortran line 17
obliqr: float = 0.0

# Mean longitude of perihelion at the vernal equinox (radians) — Fortran line 18
lambm0: float = 0.0

# Moving vernal equinox longitude of perihelion plus pi (radians) — Fortran line 19
mvelpp: float = 0.0
