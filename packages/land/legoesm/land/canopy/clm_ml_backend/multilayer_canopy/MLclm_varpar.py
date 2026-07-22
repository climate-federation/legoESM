"""
JAX/Python translation of the multilayer canopy model parameters module.

Original Fortran module: MLclm_varpar
"""

# Number of layers in multilayer canopy model — Fortran line 15
nlevmlcan: int = 100

# Number of leaf types (sunlit and shaded) — Fortran line 16
nleaf: int = 2

# Sunlit leaf index — Fortran line 17
isun: int = 1

# Shaded leaf index — Fortran line 18
isha: int = 2
