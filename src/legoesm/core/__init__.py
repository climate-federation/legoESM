"""Core infrastructure for legoESM."""

from legoesm.core.cfl import cfl_check_and_adjust, cfl_max_dt
from legoesm.core.field import Field
# Legacy re-exports; canonical imports are from legoesm.runtime.
from legoesm.core.hardware import check_spectral_backend, get_backend
from legoesm.core.smooth import sigmoid_switch, smooth_max, smooth_min, smooth_clamp
