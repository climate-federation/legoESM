"""
JAX translation of clmSoilOptionMod Fortran module.

Offline soil configuration options for simulations uncoupled from CLM.
Controls the snow/soil layer convention (CLM4.5 vs CLM5.0) and the
number of soil layers over which a moisture adjustment is applied.
Values are set via namelist in the original Fortran.

Original Fortran module: clmSoilOptionMod
Fortran lines 1-17
"""

# CLM snow/soil layer convention — Fortran line 13
# Options: 'CLM4_5' or 'CLM5_0'
# Used only for offline simulations uncoupled to CLM, for backward
# compatibility with older simulations that used CLM4.5 soils.
clm_phys: str = "CLM5_0"

# Number of soil layers to apply soil moisture adjustment — Fortran line 14
# Set to 0 to disable the adjustment entirely.
nlev_soil_adjust: int = 0

# Path to soil moisture adjustment file — set by main.py from namelist
fin_soil_adjust: str = ""
