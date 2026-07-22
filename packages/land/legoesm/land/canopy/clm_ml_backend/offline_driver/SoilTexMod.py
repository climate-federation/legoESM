"""
JAX/Python translation of the soil texture parameters module.

Soil texture classes and hydraulic parameters following:

- Cosby et al. 1984. Water Resources Research 20:682-690
  (sand, silt, clay fractions).
- Clapp and Hornberger. 1978. Water Resources Research 14:601-604
  (watsat, smpsat, hksat, bsw).

All arrays are 1-indexed in Fortran; Python tuples use 0-based
indexing with 11 elements (``ntex = 11``).

Original Fortran module: SoilTexMod
"""

# Number of soil texture classes — Fortran line 14
ntex: int = 11

# Soil texture class names — Fortran lines 28-29
soil_tex: tuple = (
    "sand",
    "loamy sand",
    "sandy loam",
    "silty loam",
    "loam",
    "sandy clay loam",
    "silty clay loam",
    "clay loam",
    "sandy clay",
    "silty clay",
    "clay",
)

# Sand fraction — Fortran line 31
sand_tex: tuple = (
    0.92,
    0.82,
    0.58,
    0.17,
    0.43,
    0.58,
    0.10,
    0.32,
    0.52,
    0.06,
    0.22,
)

# Silt fraction — Fortran line 32
silt_tex: tuple = (
    0.05,
    0.12,
    0.32,
    0.70,
    0.39,
    0.15,
    0.56,
    0.34,
    0.06,
    0.47,
    0.20,
)

# Clay fraction — Fortran line 33
clay_tex: tuple = (
    0.03,
    0.06,
    0.10,
    0.13,
    0.18,
    0.27,
    0.34,
    0.34,
    0.42,
    0.47,
    0.58,
)

# Volumetric soil water at saturation (porosity) — Fortran lines 37-38
watsat_tex: tuple = (
    0.395,
    0.410,
    0.435,
    0.485,
    0.451,
    0.420,
    0.477,
    0.476,
    0.426,
    0.492,
    0.482,
)

# Soil matric potential at saturation (mm) — Fortran lines 40-41
smpsat_tex: tuple = (
    -121.0,
    -90.0,
    -218.0,
    -786.0,
    -478.0,
    -299.0,
    -356.0,
    -630.0,
    -153.0,
    -490.0,
    -405.0,
)

# Hydraulic conductivity at saturation (mm H2O/min) — Fortran lines 43-44
hksat_tex: tuple = (
    10.560,
    9.380,
    2.080,
    0.432,
    0.417,
    0.378,
    0.102,
    0.147,
    0.130,
    0.062,
    0.077,
)

# Clapp and Hornberger "b" parameter — Fortran lines 46-47
bsw_tex: tuple = (
    4.05,
    4.38,
    4.90,
    5.30,
    5.39,
    7.12,
    7.75,
    8.52,
    10.40,
    10.40,
    11.40,
)
