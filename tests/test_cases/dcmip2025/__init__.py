"""Compatibility wrapper for relocated DCMIP-2025 test cases."""

from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025 import (
    apply_small_earth_scaling,
    dcmip25_tc1_init,
    dcmip25_tc1_topography,
    dcmip25_tc2_init,
    dcmip25_tc2a_topography,
    dcmip25_tc2b_topography,
    dcmip25_tc3_init,
)

__all__ = [
    "apply_small_earth_scaling",
    "dcmip25_tc1_init",
    "dcmip25_tc1_topography",
    "dcmip25_tc2_init",
    "dcmip25_tc2a_topography",
    "dcmip25_tc2b_topography",
    "dcmip25_tc3_init",
]
