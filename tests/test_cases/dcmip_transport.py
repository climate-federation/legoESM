"""Compatibility wrapper for relocated DCMIP-2012 transport test cases."""

from tests.atmosphere.hydrostatic.test_cases.dcmip_transport import *  # noqa: F401,F403
from tests.atmosphere.hydrostatic.test_cases.dcmip_transport import (  # noqa: F401
    _height_from_sigma,
    _mountain_height,
)
