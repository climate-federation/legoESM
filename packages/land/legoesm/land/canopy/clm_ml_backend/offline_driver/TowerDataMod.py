"""
JAX translation of TowerDataMod Fortran module.

Parameters for flux tower sites used in offline CLM simulations.
Provides site identifiers, geographic coordinates, vegetation and
soil properties, canopy structure parameters, and meteorological
forcing time steps for 15 AmeriFlux and CHATS tower sites.

Original Fortran module: TowerDataMod
Fortran lines 1-130
"""

import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Mutable scalar: current tower site index — Fortran line 14
# Set by controlMod.control() after matching tower_name to tower_id.
# ---------------------------------------------------------------------------

tower_num: int = 0  # Tower site index (1-based into arrays below; 0 = unset)


# ---------------------------------------------------------------------------
# Dimension constant — Fortran line 16
# ---------------------------------------------------------------------------

ntower: int = 15  # Number of tower sites


# ---------------------------------------------------------------------------
# Tower site identifiers — Fortran lines 33-35
# Index convention: 1-based (index 0 unused), matching Fortran array(ntower).
# All arrays have length ntower + 1; index 0 is a placeholder.
# ---------------------------------------------------------------------------

# Tower site names — Fortran: character(len=6) :: tower_id(ntower)
tower_id: list[str] = [
    "",  # index 0: unused
    "US-Ha1",  #  1
    "US-Ho1",  #  2
    "US-MMS",  #  3
    "US-UMB",  #  4
    "US-Dk3",  #  5
    "US-Me2",  #  6
    "US-Var",  #  7
    "US-IB1",  #  8
    "US-Ne3",  #  9
    "US-ARM",  # 10
    "US-Bo1",  # 11
    "US-Dk1",  # 12
    "US-Dk2",  # 13
    "CHATS7",  # 14
    "UMBSmw",  # 15
]

# ---------------------------------------------------------------------------
# Geographic coordinates — Fortran lines 37-44
# ---------------------------------------------------------------------------

# Latitude of tower (degrees) — Fortran: real(r8) :: tower_lat(ntower)
tower_lat: Array = jnp.array(
    [
        0.00,  # index 0: unused
        42.54,  #  1  US-Ha1
        45.20,  #  2  US-Ho1
        39.32,  #  3  US-MMS
        45.56,  #  4  US-UMB
        35.98,  #  5  US-Dk3
        44.45,  #  6  US-Me2
        38.41,  #  7  US-Var
        41.86,  #  8  US-IB1
        41.18,  #  9  US-Ne3
        36.61,  # 10  US-ARM
        40.01,  # 11  US-Bo1
        35.97,  # 12  US-Dk1
        35.97,  # 13  US-Dk2
        38.49,  # 14  CHATS7
        45.56,  # 15  UMBSmw
    ],
    dtype=jnp.float64,
)

# Longitude of tower (degrees) — Fortran: real(r8) :: tower_lon(ntower)
tower_lon: Array = jnp.array(
    [
        0.00,  # index 0: unused
        -72.17,  #  1  US-Ha1
        -68.74,  #  2  US-Ho1
        -86.41,  #  3  US-MMS
        -84.71,  #  4  US-UMB
        -79.09,  #  5  US-Dk3
        -121.56,  #  6  US-Me2
        -120.95,  #  7  US-Var
        -88.22,  #  8  US-IB1
        -96.44,  #  9  US-Ne3
        -97.49,  # 10  US-ARM
        -88.29,  # 11  US-Bo1
        -79.09,  # 12  US-Dk1
        -79.10,  # 13  US-Dk2
        -121.84,  # 14  CHATS7
        -84.71,  # 15  UMBSmw
    ],
    dtype=jnp.float64,
)

# ---------------------------------------------------------------------------
# Vegetation — Fortran lines 46-48
# ---------------------------------------------------------------------------

# CLM PFT for tower site — Fortran: integer :: tower_pft(ntower)
tower_pft: Array = jnp.array(
    [
        0,  # index 0: unused
        7,  #  1  US-Ha1  broadleaf_deciduous_temperate_tree
        2,  #  2  US-Ho1  needleleaf_evergreen_boreal_tree
        7,  #  3  US-MMS  broadleaf_deciduous_temperate_tree
        7,  #  4  US-UMB  broadleaf_deciduous_temperate_tree
        1,  #  5  US-Dk3  needleleaf_evergreen_temperate_tree
        2,  #  6  US-Me2  needleleaf_evergreen_boreal_tree
        13,  #  7  US-Var  c3_non-arctic_grass
        15,  #  8  US-IB1  c3_crop
        15,  #  9  US-Ne3  c3_crop
        15,  # 10  US-ARM  c3_crop
        15,  # 11  US-Bo1  c3_crop
        13,  # 12  US-Dk1  c3_non-arctic_grass
        7,  # 13  US-Dk2  broadleaf_deciduous_temperate_tree
        7,  # 14  CHATS7  broadleaf_deciduous_temperate_tree
        7,  # 15  UMBSmw  broadleaf_deciduous_temperate_tree
    ],
    dtype=jnp.int32,
)

# ---------------------------------------------------------------------------
# Soil properties — Fortran lines 50-77
# ---------------------------------------------------------------------------

# Soil texture class name — Fortran: character(len=15) :: tower_tex(ntower)
# Used when tower_sand/tower_clay are both < 0.
tower_tex: list[str] = [
    "",  # index 0: unused
    "loam",  #  1  US-Ha1
    "sandy loam",  #  2  US-Ho1
    "clay",  #  3  US-MMS
    "sand",  #  4  US-UMB
    "sandy loam",  #  5  US-Dk3
    "sandy loam",  #  6  US-Me2
    "silty loam",  #  7  US-Var
    "silty clay loam",  #  8  US-IB1
    "clay loam",  #  9  US-Ne3
    "clay",  # 10  US-ARM
    "silty loam",  # 11  US-Bo1
    "sandy loam",  # 12  US-Dk1
    "sandy loam",  # 13  US-Dk2
    "silty clay loam",  # 14  CHATS7
    "sand",  # 15  UMBSmw
]

# Percent sand (used when >= 0; -999 triggers texture class lookup)
# Fortran: real(r8) :: tower_sand(ntower)
tower_sand: Array = jnp.array(
    [
        0.0,  # index 0: unused
        -999.0,  #  1  US-Ha1
        -999.0,  #  2  US-Ho1
        -999.0,  #  3  US-MMS
        -999.0,  #  4  US-UMB
        -999.0,  #  5  US-Dk3
        -999.0,  #  6  US-Me2
        -999.0,  #  7  US-Var
        -999.0,  #  8  US-IB1
        -999.0,  #  9  US-Ne3
        -999.0,  # 10  US-ARM
        -999.0,  # 11  US-Bo1
        -999.0,  # 12  US-Dk1
        -999.0,  # 13  US-Dk2
        10.0,  # 14  CHATS7
        -999.0,  # 15  UMBSmw
    ],
    dtype=jnp.float64,
)

# Percent clay (used when >= 0; -999 triggers texture class lookup)
# Fortran: real(r8) :: tower_clay(ntower)
tower_clay: Array = jnp.array(
    [
        0.0,  # index 0: unused
        -999.0,  #  1  US-Ha1
        -999.0,  #  2  US-Ho1
        -999.0,  #  3  US-MMS
        -999.0,  #  4  US-UMB
        -999.0,  #  5  US-Dk3
        -999.0,  #  6  US-Me2
        -999.0,  #  7  US-Var
        -999.0,  #  8  US-IB1
        -999.0,  #  9  US-Ne3
        -999.0,  # 10  US-ARM
        -999.0,  # 11  US-Bo1
        -999.0,  # 12  US-Dk1
        -999.0,  # 13  US-Dk2
        35.0,  # 14  CHATS7
        -999.0,  # 15  UMBSmw
    ],
    dtype=jnp.float64,
)

# Soil organic matter (kg/m3) — Fortran lines 67-70
# Fortran: real(r8) :: tower_organic(ntower)
tower_organic: Array = jnp.array(
    [
        0.0,  # index 0: unused
        0.0,  #  1  US-Ha1
        0.0,  #  2  US-Ho1
        0.0,  #  3  US-MMS
        0.0,  #  4  US-UMB
        0.0,  #  5  US-Dk3
        0.0,  #  6  US-Me2
        0.0,  #  7  US-Var
        0.0,  #  8  US-IB1
        0.0,  #  9  US-Ne3
        0.0,  # 10  US-ARM
        0.0,  # 11  US-Bo1
        0.0,  # 12  US-Dk1
        0.0,  # 13  US-Dk2
        50.0,  # 14  CHATS7
        0.0,  # 15  UMBSmw
    ],
    dtype=jnp.float64,
)

# CLM soil color class — Fortran lines 72-74
# Fortran: integer :: tower_isoicol(ntower)
tower_isoicol: Array = jnp.array(
    [
        0,  # index 0: unused
        18,  #  1  US-Ha1
        16,  #  2  US-Ho1
        15,  #  3  US-MMS
        17,  #  4  US-UMB
        15,  #  5  US-Dk3
        20,  #  6  US-Me2
        17,  #  7  US-Var
        15,  #  8  US-IB1
        13,  #  9  US-Ne3
        13,  # 10  US-ARM
        15,  # 11  US-Bo1
        15,  # 12  US-Dk1
        15,  # 13  US-Dk2
        15,  # 14  CHATS7
        17,  # 15  UMBSmw
    ],
    dtype=jnp.int32,
)

# Depth to bedrock (m) — Fortran lines 76-79
# Fortran: real(r8) :: tower_zbed(ntower)
tower_zbed: Array = jnp.array(
    [
        0.0,  # index 0: unused
        50.0,  #  1  US-Ha1
        50.0,  #  2  US-Ho1
        50.0,  #  3  US-MMS
        50.0,  #  4  US-UMB
        50.0,  #  5  US-Dk3
        50.0,  #  6  US-Me2
        50.0,  #  7  US-Var
        50.0,  #  8  US-IB1
        50.0,  #  9  US-Ne3
        50.0,  # 10  US-ARM
        50.0,  # 11  US-Bo1
        50.0,  # 12  US-Dk1
        50.0,  # 13  US-Dk2
        2.0,  # 14  CHATS7
        50.0,  # 15  UMBSmw
    ],
    dtype=jnp.float64,
)

# ---------------------------------------------------------------------------
# Canopy and measurement structure — Fortran lines 81-100
# ---------------------------------------------------------------------------

# Flux tower height (m); -999 triggers a default of 30 m — Fortran lines 81-85
# Fortran: real(r8) :: tower_ht(ntower)
tower_ht: Array = jnp.array(
    [
        0.0,  # index 0: unused
        30.0,  #  1  US-Ha1
        29.0,  #  2  US-Ho1
        48.0,  #  3  US-MMS
        46.0,  #  4  US-UMB
        22.0,  #  5  US-Dk3
        32.0,  #  6  US-Me2
        2.5,  #  7  US-Var
        4.0,  #  8  US-IB1
        6.0,  #  9  US-Ne3
        -999.0,  # 10  US-ARM  (triggers default 30 m)
        6.0,  # 11  US-Bo1
        5.0,  # 12  US-Dk1
        42.0,  # 13  US-Dk2
        23.0,  # 14  CHATS7
        46.0,  # 15  UMBSmw
    ],
    dtype=jnp.float64,
)

# Canopy height (m) — Fortran lines 87-91
# Fortran: real(r8) :: tower_canht(ntower)
tower_canht: Array = jnp.array(
    [
        0.0,  # index 0: unused
        23.0,  #  1  US-Ha1
        20.0,  #  2  US-Ho1
        27.0,  #  3  US-MMS
        21.0,  #  4  US-UMB
        17.0,  #  5  US-Dk3
        14.0,  #  6  US-Me2
        0.6,  #  7  US-Var
        0.9,  #  8  US-IB1
        0.9,  #  9  US-Ne3
        0.5,  # 10  US-ARM
        0.9,  # 11  US-Bo1
        0.5,  # 12  US-Dk1
        25.0,  # 13  US-Dk2
        10.0,  # 14  CHATS7
        21.0,  # 15  UMBSmw
    ],
    dtype=jnp.float64,
)

# Fine root biomass (g biomass/m2); -999 = unavailable — Fortran lines 93-97
# Fortran: real(r8) :: tower_root(ntower)
tower_root: Array = jnp.array(
    [
        0.0,  # index 0: unused
        500.0,  #  1  US-Ha1
        500.0,  #  2  US-Ho1
        500.0,  #  3  US-MMS
        500.0,  #  4  US-UMB
        500.0,  #  5  US-Dk3
        500.0,  #  6  US-Me2
        -999.0,  #  7  US-Var
        -999.0,  #  8  US-IB1
        -999.0,  #  9  US-Ne3
        -999.0,  # 10  US-ARM
        -999.0,  # 11  US-Bo1
        500.0,  # 12  US-Dk1
        500.0,  # 13  US-Dk2
        500.0,  # 14  CHATS7
        500.0,  # 15  UMBSmw
    ],
    dtype=jnp.float64,
)

# ---------------------------------------------------------------------------
# Beta distribution parameters for leaf and stem area density profiles
# Fortran lines 99-117
# Shape (ntower+1, 2): index 0 unused; column 0 = param 1, column 1 = param 2.
# -999 triggers PFT-specified values.
# ---------------------------------------------------------------------------

# Leaf area density beta distribution parameters — Fortran lines 99-107
# Fortran: real(r8) :: tower_pbeta_lai(ntower,2)
tower_pbeta_lai: Array = jnp.array(
    [
        [0.0, 0.0],  # index 0: unused
        [-999.0, -999.0],  #  1  US-Ha1
        [-999.0, -999.0],  #  2  US-Ho1
        [-999.0, -999.0],  #  3  US-MMS
        [-999.0, -999.0],  #  4  US-UMB
        [-999.0, -999.0],  #  5  US-Dk3
        [11.5, 3.5],  #  6  US-Me2
        [-999.0, -999.0],  #  7  US-Var
        [-999.0, -999.0],  #  8  US-IB1
        [-999.0, -999.0],  #  9  US-Ne3
        [-999.0, -999.0],  # 10  US-ARM
        [-999.0, -999.0],  # 11  US-Bo1
        [-999.0, -999.0],  # 12  US-Dk1
        [-999.0, -999.0],  # 13  US-Dk2
        [2.6, 1.3],  # 14  CHATS7
        [-999.0, -999.0],  # 15  UMBSmw
    ],
    dtype=jnp.float64,
)

# Stem area density beta distribution parameters — Fortran lines 109-117
# Fortran: real(r8) :: tower_pbeta_sai(ntower,2)
tower_pbeta_sai: Array = jnp.array(
    [
        [0.0, 0.0],  # index 0: unused
        [-999.0, -999.0],  #  1  US-Ha1
        [-999.0, -999.0],  #  2  US-Ho1
        [-999.0, -999.0],  #  3  US-MMS
        [-999.0, -999.0],  #  4  US-UMB
        [-999.0, -999.0],  #  5  US-Dk3
        [11.5, 3.5],  #  6  US-Me2
        [-999.0, -999.0],  #  7  US-Var
        [-999.0, -999.0],  #  8  US-IB1
        [-999.0, -999.0],  #  9  US-Ne3
        [-999.0, -999.0],  # 10  US-ARM
        [-999.0, -999.0],  # 11  US-Bo1
        [-999.0, -999.0],  # 12  US-Dk1
        [-999.0, -999.0],  # 13  US-Dk2
        [1.8, 1.3],  # 14  CHATS7
        [-999.0, -999.0],  # 15  UMBSmw
    ],
    dtype=jnp.float64,
)

# ---------------------------------------------------------------------------
# Time step of forcing data — Fortran lines 119-121
# ---------------------------------------------------------------------------

# Forcing data time step (minutes) — Fortran: integer :: tower_time(ntower)
tower_time: Array = jnp.array(
    [
        0,  # index 0: unused
        60,  #  1  US-Ha1
        30,  #  2  US-Ho1
        60,  #  3  US-MMS
        60,  #  4  US-UMB
        30,  #  5  US-Dk3
        30,  #  6  US-Me2
        30,  #  7  US-Var
        30,  #  8  US-IB1
        60,  #  9  US-Ne3
        30,  # 10  US-ARM
        30,  # 11  US-Bo1
        30,  # 12  US-Dk1
        30,  # 13  US-Dk2
        30,  # 14  CHATS7
        60,  # 15  UMBSmw
    ],
    dtype=jnp.int32,
)
