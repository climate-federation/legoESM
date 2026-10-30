"""Configuration for the 3D Monte-Carlo ray-tracing radiation scheme.

See ``docs/specs/mc3d_raytracer.md``. This is a *spatial solver* config only —
spectral optics (per-g-point ``optical_depth``/``ssa``/``asymmetry_factor``) come
from the existing RRTMGP optics; nothing here re-derives constants or saturation.

Parameter hygiene: ``photons_per_pixel``/``max_iterations``/``n_batches`` are
integer *counts* (loop-counts-never-trainable rule — ints are not
``__param_spec__``-eligible, so they can never reach the trainable collector).
``knull_floor`` is a numerics majorant floor, excluded (tier 0).
"""

from __future__ import annotations

from typing import NamedTuple

# --- Woodcock null-collision tracking (numerics) ---
# Floor on the majorant extinction so the free-path sampler never divides by a
# (near-)zero majorant in optically empty columns. microhh uses the same 1e-3.
_KNULL_FLOOR_DEFAULT = 1.0e-3  # [1/m]
# Sharpness of the smooth free-path step floor (RULE 5): the step is
# smooth_max(eps, tau/k_null, s). microhh's hard ``max(eps, tau/k_null)``
# (raytracer_kernels.cu:212) is reached at s = inf. At s = 1e6 1/m the floor
# adds log(2)/s ~ 6.9e-7 m at u = 0 and < 1e-40 m for steps >= 1e-4 m.
_FREE_PATH_FLOOR_SHARPNESS_DEFAULT = 1.0e6  # [1/m]


class MC3DRadiationConfig(NamedTuple):
  """Numerics knobs for the 3D Monte-Carlo ray tracer (plane LES/CRM only)."""

  # Photons launched per horizontal column per g-point. Variance ~ 1/sqrt(N).
  photons_per_pixel: int = 128
  # Photon batches the per-column photons are split into and ``lax.scan``-ed
  # over, bounding peak photon-state memory independently of photon count.
  n_batches: int = 1
  # Hard cap on Woodcock tracking steps per photon (free-path + null-collision
  # events). A photon hitting the cap is terminated as escaped (logged in tests).
  max_iterations: int = 4096
  # Majorant extinction floor [1/m]; see module constant above.
  knull_floor: float = _KNULL_FLOOR_DEFAULT
  # Coarse majorant-grid block sizes in fine cells (Phase 1.1 speedup). 0 = no
  # coarsening (one coarse cell per axis = global scalar majorant). A smaller
  # block tightens the majorant in clear air beside thick clouds, cutting
  # null-collision steps; the delta-tracking answer is unbiased for any block
  # size. knull_coarsen_xy MUST divide nx and ny.
  knull_coarsen_xy: int = 0
  knull_coarsen_z: int = 0
  # Base PRNG seed (offset further per-rank for photon sharding, Phase 4).
  seed: int = 0
  # Quasi-random (Halton low-discrepancy) photon-launch sampling for variance
  # reduction (fewer photons for a target error -> less GPU memory). Opt-in;
  # default off keeps pure pseudo-random (threefry) launch. See qrng.py.
  use_qrng: bool = False
  # Cloud scattering phase: True -> sample the microhh Mie LUT (mie.py) by
  # effective radius + band; False (default) -> Henyey-Greenstein with the cloud
  # asymmetry g. Mie is the oracle-faithful cloud phase (RRTMGP optics path).
  use_mie: bool = False
  # Smooth free-path step floor sharpness [1/m]; inf = microhh hard max.
  free_path_floor_sharpness: float = _FREE_PATH_FLOOR_SHARPNESS_DEFAULT


# Excluded from trainable collection: counts are int (auto-exempt) and
# ``knull_floor`` is a numerics regulariser, not a closure parameter.
__param_spec__ = {
    "knull_floor": {
        "units": "1/m",
        "bounds": (1e-6, 1e-1),
        "tunable_tier": 0,
        "transform": "none",
        "category": "numerics",
        "reference": "microhh rte-rrtmgp-cpp create_knull_grid (1e-3 floor)",
        "shape": None,
    },
    "free_path_floor_sharpness": {
        "units": "1/m",
        "bounds": (1e3, float("inf")),
        "tunable_tier": 0,
        "transform": "none",
        "category": "numerics",
        "reference": "microhh rte-rrtmgp-cpp raytracer_kernels.cu:212 max(eps, tau/k_null) at inf",
        "shape": None,
    },
}
