"""DCMIP-2025 non-hydrostatic test-case package."""

from .test_case_1 import dcmip25_tc1_init, dcmip25_tc1_topography
from .test_case_2 import (
    dcmip25_tc2_init,
    dcmip25_tc2a_topography,
    dcmip25_tc2b_topography,
)
from .test_case_3 import dcmip25_tc3_init
from .common import apply_small_earth_scaling

__all__ = [
    "apply_small_earth_scaling",
    "dcmip25_tc1_init",
    "dcmip25_tc1_topography",
    "dcmip25_tc2_init",
    "dcmip25_tc2a_topography",
    "dcmip25_tc2b_topography",
    "dcmip25_tc3_init",
]
