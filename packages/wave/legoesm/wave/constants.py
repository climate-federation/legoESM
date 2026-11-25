"""Constants of the ported NeuralSWM wave kernel.

These are the values the released weights were trained with.  They are
deliberately NOT taken from :mod:`legoesm.constants`: the dispersion
relation and the spherical metric were built with them, so substituting
the legoESM values would change the wavenumber and the grid spacing the
network sees.

Recorded differences from ``legoesm.constants``:

===============  =====================  ==================
quantity         NeuralSWM (here)       legoESM
===============  =====================  ==================
gravity [m/s2]   9.806                  9.80616
Earth radius [m] 6366197.7236758135     6.371229e6
===============  =====================  ==================

The gravity difference is 1.6e-5 relative and the radius difference is
7.9e-4 relative; both are small but they shift the discretised
trajectory, so the port keeps the oracle values in :mod:`legoesm.constants`
and the parity test compares against NeuralSWM, not against a re-derived model.

Provenance: NeuralSWM commit 83634cc, ``physics_jax/physics.py`` and
``neuralwave_jax/architecture.py``.
"""

from __future__ import annotations

from legoesm import constants

# physics_jax/physics.py:21 (``solve_wavenumber(gravity=9.806)``).
GRAVITY = constants.g_neuralswm

# neuralwave_jax/architecture.py:55 (``_BASE_METADATA["r_earth"]``).
R_EARTH = constants.R_earth_neuralswm

# The Stokes diagnostic uses its own Newton solver with g = 9.81, not the
# model's 9.806 (NeuralSWM inference/jax_core.py:386).  Kept so the ported
# Stokes output matches the released diagnostic.
STOKES_GRAVITY = constants.g_neuralswm_stokes

# Source-term scales baked into the trained parameterizations:
#   parameterizations.py:255  -> stot = ... * 0.001
#   integrator.py:110          -> update = stot * (dt * 0.01)
# They are part of the trained model's units, not tunables.
SOURCE_TERM_SCALE = 0.001
SOURCE_TIME_SCALE = 0.01

# Default spectral grid of the released weights: 24 directions x 30
# frequencies (``architecture.py:52``).
SPEC_SHAPE = (24, 30)

# Wavenumber solver: fixed bisection count.  WADEN v2 (46aadc7) performs no
# decimal rounding; the legacy NeuralSWM v1 path did, but this port follows
# the WADEN reference.
WAVENUM_MAX_ITER = 50
