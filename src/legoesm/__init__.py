"""legoESM: A Differentiable Earth System Model in JAX.

legoESM is a next-generation, fully differentiable Earth System Model
built from scratch in JAX. It spans weather-to-climate timescales, couples
atmosphere, ocean, land, and cryosphere through a unified interface, and
enables end-to-end gradient computation for data assimilation, parameter
estimation, and hybrid AI-physics modeling.
"""

__version__ = "0.1.0"

# Core infrastructure
from legoesm.core.field import Field
from legoesm.constants import *  # noqa: F401, F403

# Convenience time helpers
def hours(n: float) -> float:
    """Convert hours to seconds."""
    return n * 3600.0

def minutes(n: float) -> float:
    """Convert minutes to seconds."""
    return n * 60.0

def days(n: float) -> float:
    """Convert days to seconds."""
    return n * 86400.0

def years(n: float) -> float:
    """Convert years to seconds (365.25 days)."""
    return n * 365.25 * 86400.0
