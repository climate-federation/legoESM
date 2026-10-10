# Copyright 2024 The swirl_jatmos Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of a radiative transfer solver."""

import functools
from pathlib import Path
from typing import TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.rrtmgp import constants
from legoesm.atmosphere.physics.radiation.rrtmgp.config.radiative_transfer import (
    OpticsParameters,
    RRTMOptics as RRTMOpticsConfig,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import atmospheric_state
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import constants as optics_constants
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.lookup_volume_mixing_ratio import (
    LookupVolumeMixingRatio,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.optics import optics_factory
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "RRTMGP correlated-k radiation solver (solve_columns): builds the gas/"
        "cloud/aerosol optics per g-point and integrates the two-stream LW+SW "
        "radiative transfer to broadband fluxes and radiative heating rates."
    ),
    "inputs": {
        "T": "K", "p_full": "Pa", "p_half": "Pa", "sfc_temperature": "K",
        "q_v": "kg/kg", "cos_zenith": "1 (cos solar zenith angle)",
        "sfc_albedo": "1 (surface shortwave albedo)",
        "sfc_emissivity": "1 (surface longwave emissivity)",
        "o3_vmr": "mol/mol", "cloud_path_liq": "kg/m^2",
        "cloud_path_ice": "kg/m^2", "cloud_r_eff_liq": "m",
        "cloud_r_eff_ice": "m", "cloud_fraction": "1",
        "aerosol_optical_depth": "1",
    },
    "outputs": {
        "lw_flux_up": "W/m^2", "lw_flux_down": "W/m^2",
        "sw_flux_up": "W/m^2", "sw_flux_down": "W/m^2",
        "heating_rate": "K/s", "lw_heating_rate": "K/s",
        "sw_heating_rate": "K/s",
    },
    "sign_convention": (
        "heating_rate dT/dt>0 warms the layer; fluxes positive in their named "
        "direction (up/down); net radiative flux OUT of a layer cools it "
        "(heating_rate = -g/c_p * dF_net/dp); optical depth tau>=0, "
        "0<=ssa<=1, |g|<=1; cos_zenith>=0 for illuminated columns (nighttime SW "
        "zeroed). Photons enter/leave at TOA and the surface, so the column "
        "energy budget is OPEN (accounted, not conserved)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Pincus, Mlawer & Delamere (2019), JAMES, doi:10.1029/2019MS001621 "
        "(RRTMGP); swirl_jatmos two-stream port."
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_rrtmgp_stratosphere.py + "
        "tests/unit/test_physics_radiation.py: clear-sky LW cooling / SW heating "
        "profiles; differentiable in the continuous optical inputs (T, q_v, VMR, "
        "cloud paths), with only the integer band/g-point index selection static."
    ),
}

# ---------------------------------------------------------------------------
# Default RRTMGP data file paths (Zarr preferred, NetCDF fallback)
# ---------------------------------------------------------------------------
_ZARR_DIR = Path(__file__).parent.parent.parent.parent / "data" / "zarr"
_NC_DIR = Path(__file__).parent / "optics" / "rrtmgp_data"


# Idealized ozone Gaussian profile + well-mixed gas mole fractions (fixed).
_O3_SIGMA_TROP = 0.9
_O3_SIGMA_STRAT = 1.5
_O3_PEAK_VMR = 9.0e-6
_O2_MOLE_FRACTION = 0.20948
_N2_MOLE_FRACTION = 0.78084

def _default_data_path(basename_nc: str) -> str:
    """Return path to Zarr store if available, else fall back to NetCDF."""
    zarr_path = _ZARR_DIR / basename_nc.replace(".nc", ".zarr")
    if zarr_path.exists():
        return str(zarr_path)
    return str(_NC_DIR / basename_nc)


_DEFAULT_LW_GAS = _default_data_path("rrtmgp-gas-lw-g128.nc")
DEFAULT_SW_GAS = _default_data_path("rrtmgp-gas-sw-g112.nc")
_DEFAULT_LW_CLOUD = _default_data_path("cloudysky_lw.nc")
_DEFAULT_SW_CLOUD = _default_data_path("cloudysky_sw.nc")

# Module-level optics cache (shared across RRTMGP instances)
_legoesm_optics_cache: dict = {}


# ---------------------------------------------------------------------------
# Helper functions for the legoESM column solver
# ---------------------------------------------------------------------------

def _add_halos(f_3d):
    """Add 1-cell vertical halos via linear extrapolation.

    Input shape (ncol, 1, nlev), output shape (ncol, 1, nlev+2).
    """
    bottom_halo = 2 * f_3d[:, :, 0:1] - f_3d[:, :, 1:2]
    top_halo = 2 * f_3d[:, :, -1:] - f_3d[:, :, -2:-1]
    return jnp.concatenate([bottom_halo, f_3d, top_halo], axis=2)


def _standard_o3_profile(p_full):
    """Climatological ozone VMR profile, US Std Atm 1976 piecewise fit.

    Skewed log-Gaussian peaking at 9 ppm near 10 hPa with separate
    widths for the tropospheric (``sigma_trop = 0.9`` in natural log of
    pressure) and stratospheric (``sigma_strat = 1.5``) sides, plus a
    constant tropospheric **background** of 20 ppb captured via
    ``jnp.maximum``.  The skewed shape captures the asymmetric real
    profile (sharp fall through the tropopause, gentle decay into the
    mesosphere); the background captures the well-mixed tropospheric
    ozone the Gaussian alone underestimates by 2-3 orders of magnitude
    near the surface.

    Reference values at canonical levels vs. US Std Atm 1976::

        level    real      old (σ=1.5,A=8)   iter-3 (skew+bg, A=9)
        1000 hPa  ~25 ppb  ~70 ppb           20 ppb  (background)
         500 hPa  ~50 ppb  ~570 ppb          20 ppb  (background)
         200 hPa ~100 ppb  ~1.5 ppm          35 ppb  (Gaussian)
         100 hPa ~250 ppb  ~2.5 ppm         ~340 ppb (Gaussian)
          30 hPa  ~5 ppm   ~5.6 ppm          ~4.3 ppm
          10 hPa  ~9 ppm   ~8 ppm             9 ppm (peak)
           1 hPa  ~3 ppm   ~2.5 ppm          ~2.8 ppm
         0.1 hPa  ~80 ppb  ~5 ppb            ~80 ppb

    NOT meant as a high-fidelity climatology — drivers should provide
    an external ``o3_vmr`` field for production runs.  This fallback
    only ensures that radiative transfer sees a non-trivial ozone
    column when no ozone source is configured.

    The skewed-Gaussian transition has a derivative discontinuity at
    ``p = 10 hPa`` but is C0-continuous and finite everywhere; the
    ``jnp.maximum`` with the background introduces a second
    sub-differentiable transition where the Gaussian tail crosses
    20 ppb (around p ~ 250 hPa).  Both are AD-safe (finite gradients
    on each side).
    """
    p_hPa = p_full / 100.0
    log_p = jnp.log(p_hPa)
    log_p_peak = jnp.log(10.0)
    sigma_trop = _O3_SIGMA_TROP
    sigma_strat = _O3_SIGMA_STRAT
    sigma = jnp.where(log_p > log_p_peak, sigma_trop, sigma_strat)
    arg = (log_p - log_p_peak) / sigma
    o3_gauss = _O3_PEAK_VMR * jnp.exp(-0.5 * arg * arg)
    # 20 ppb tropospheric background (US Std Atm 1976 surface value).
    o3_background = 2.0e-8
    o3 = jnp.maximum(o3_gauss, o3_background)
    return jnp.clip(o3, 1.0e-10, None)


def standard_o3_profile(p_full):
    """Public alias of the climatological ozone-VMR profile.

    Use this when an external ozone source is not configured but RRTMGP
    must still see a realistic ozone column — driver code historically
    initialised ``o3_vmr`` to zeros, which RRTMGP then clipped to
    ``1e-10`` and which silently disabled stratospheric heating.  Pass
    the result of ``standard_o3_profile(p_full)`` instead of zeros, or
    pass ``None`` to let RRTMGP build the same profile internally.
    """
    return _standard_o3_profile(p_full)


def _cast_optics_f64_to_f32(obj, _seen=None):
    """Recursively cast every float64 array leaf in ``obj`` to float32.

    The RRTMGP optics objects are plain Python classes (``RRTMOptics``)
    holding stdlib ``@dataclasses.dataclass(frozen=True)`` lookup tables
    (``gas_optics_lw/sw``, ``cloud_optics_lw/sw``) whose fields are
    ``jax.Array`` tables.  NONE of these are registered JAX pytrees, so
    ``jax.tree_util.tree_map`` treats each as a single opaque leaf and casts
    NOTHING — the silent no-op this replaces (fp32 heating came out
    bit-identical to fp64 because the tables stayed float64).  This walks the
    structure by hand: float64 arrays are cast; frozen dataclasses are rebuilt
    via ``dataclasses.replace``; dicts / lists / tuples are mapped; and plain
    objects with a ``__dict__`` (e.g. ``RRTMOptics``) have each attribute cast
    in place.  Integer index tables and non-float leaves are left untouched.

    An ``id``-keyed ``_seen`` set breaks reference cycles in the plain-object
    graph (``OpticsScheme`` subclasses bind ``functools.partial``/closures that
    capture ``self`` and other optics objects, so a naive deep walk recurses
    forever — the ``RecursionError`` this guards against).  Callables, modules,
    and types are skipped (they hold no float tables and are common cycle
    waypoints).  Used only on the ``compute_fp32`` path.
    """
    import dataclasses as _dc
    import types as _types

    if _seen is None:
        _seen = set()

    # Array leaf (jax or numpy): cast float64 -> float32, keep everything else
    # (int index tables, already-float32, bool) as is.
    _dtype = getattr(obj, "dtype", None)
    if _dtype is not None and hasattr(obj, "astype"):
        return obj.astype(jnp.float32) if _dtype == jnp.float64 else obj
    # Skip leaves that hold no tables and are common cycle waypoints.
    if (obj is None or isinstance(obj, (str, bytes, int, float, bool,
                                        _types.ModuleType, type))
            or callable(obj)):
        return obj
    # Cycle guard: only mutable containers/objects can form cycles.
    _oid = id(obj)
    if _oid in _seen:
        return obj
    _seen.add(_oid)

    # Frozen / plain stdlib dataclass instance: rebuild changed float fields.
    if _dc.is_dataclass(obj) and not isinstance(obj, type):
        changes = {}
        for _f in _dc.fields(obj):
            _v = getattr(obj, _f.name)
            _nv = _cast_optics_f64_to_f32(_v, _seen)
            if _nv is not _v:
                changes[_f.name] = _nv
        return _dc.replace(obj, **changes) if changes else obj
    if isinstance(obj, dict):
        return {_k: _cast_optics_f64_to_f32(_v, _seen) for _k, _v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cast_optics_f64_to_f32(_v, _seen) for _v in obj)
    # Plain object (e.g. RRTMOptics, which is mutable): cast each data attribute
    # in place and return the same object.  Methods/bound closures are skipped
    # by the callable guard above when recursed into.
    if hasattr(obj, "__dict__"):
        for _a, _v in vars(obj).items():
            if callable(_v):
                continue
            _nv = _cast_optics_f64_to_f32(_v, _seen)
            if _nv is not _v:
                setattr(obj, _a, _nv)
        return obj
    return obj


def _resolve_aerosol_sw_optics(config, optics_lib):
    """SW aerosol single-scattering albedo + asymmetry fed to the two-stream
    solver.  Returns Python-float SCALARS (grey aerosol, the historical default
    -> byte-identical) or ``(n_bnd_sw,)`` per-band jnp arrays when
    ``config.aerosol_ssa_bands`` / ``aerosol_g_bands`` are set.  The per-band
    length is validated against the loaded SW gas-optics band count (a wrong
    length is a config error, not a silent out-of-range gather)."""
    ssa = config.aerosol_ssa
    g = config.aerosol_g
    bands_ssa = getattr(config, "aerosol_ssa_bands", None)
    bands_g = getattr(config, "aerosol_g_bands", None)
    if bands_ssa is None and bands_g is None:
        return ssa, g
    n_bnd = int(optics_lib.gas_optics_sw.n_bnd)
    dtype = optics_lib.gas_optics_sw.kmajor.dtype
    if bands_ssa is not None:
        if len(bands_ssa) != n_bnd:
            raise ValueError(
                f"RRTMGPConfig.aerosol_ssa_bands has length {len(bands_ssa)}, "
                f"expected the SW band count {n_bnd}.")
        ssa = jnp.asarray(bands_ssa, dtype=dtype)
    if bands_g is not None:
        if len(bands_g) != n_bnd:
            raise ValueError(
                f"RRTMGPConfig.aerosol_g_bands has length {len(bands_g)}, "
                f"expected the SW band count {n_bnd}.")
        g = jnp.asarray(bands_g, dtype=dtype)
    return ssa, g


class RRTMGP:
  """Rapid Radiative Transfer Model for General Circulation Models (RRTMGP).

  legoESM integration API — construct with
  ``RRTMGP.from_legoesm_config(rrtmgp_config)`` and drive via
  ``solve_columns(...)``.  The swirl_jatmos-style ``__init__`` and
  ``compute_heating_rate`` API were removed in iter-22 after zero
  callers were found across the codebase.
  """

  @staticmethod
  def _optics_cache_key(config):
      """Hashable key for the optics-table cache.

      Includes only fields that change the loaded NetCDF tables
      (gas files, cloud files, table-shape inputs) plus the active
      x64 precision (table dtype depends on it).  Solver-behavior
      fields like ``use_scan`` deliberately omitted — the tables
      themselves are independent of how the solver traverses them.

      Iter-40: re-introduced ``include_clouds`` here BUT for a
      different reason than the pre-iter-36 state — iter-40 made
      ``RRTMOptics.__init__`` skip the cloud-table load when
      ``include_clouds=False``, so the constructed optics_lib is
      now genuinely different across the True/False configurations
      (one has cloud_optics_lw/sw populated, the other has them set
      to None).  Memory: clear-sky workflows save ~MB of cloud
      tables per cached entry.
      """
      import jax
      x64 = bool(jax.config.jax_enable_x64)
      return (config.lw_gas_file, config.sw_gas_file,
              config.lw_cloud_file, config.sw_cloud_file,
              config.include_clouds,
              config.co2_ppmv, config.ch4_ppbv, config.n2o_ppbv,
              x64, getattr(config, "compute_fp32", False))

  @staticmethod
  def _instance_cache_key(config):
      """Hashable key for the solver-instance cache.

      Extends the optics key with the behavior fields that change
      the *result* of ``solve_columns`` (or its compile-time graph)
      without changing the optics tables.

      Includes:
      - ``use_scan`` (issue #273): GPU auto-pick (``None`` ⇒ scan on
        GPU/TPU, for-loop on CPU) is honored even when an earlier call
        cached an explicit ``False``.
      - ``use_optimal_angle`` (iter-2): the optimal-angle path traces
        a different two-stream graph (per-band/per-column secant) than
        the fixed-1.66 path, so a config flip must rebuild.
      - ``S_0``, ``aerosol_ssa``, ``aerosol_g`` (iter-32 audit): these
        are NOT overridable per-call via ``solve_columns(...)`` kwargs
        — they are read off ``self._config`` every call.  Without
        them in the cache key, a later config that bumps e.g.
        ``aerosol_ssa = 0.95`` would silently reuse a solver instance
        built with ``aerosol_ssa = 0.93`` and apply the stale value.
      - ``sfc_emissivity``, ``sfc_albedo`` (iter-32 audit): these ARE
        overridable per call, but ``_resolve_surface_field(override,
        fallback)`` falls back to the config default when no override
        is passed.  AIMIP populates these with ``(ncol,)`` arrays
        (low-rank lat-lon expansion) — arrays aren't hashable, so we
        use ``id(...)`` for arrays (cache-correct as long as AIMIP
        doesn't mutate in place, which JAX immutability prevents).
        Scalar floats hash directly.
      """
      def _hashable(x):
          # Float / int / None / str / bool: hashable directly.
          try:
              hash(x)
              return x
          except TypeError:
              # 0-D arrays (jnp/np scalar shape == ()): convert to a
              # stable value-hashable form so two ``jnp.array(0.07)``
              # values share the cache key.  iter-39 + codex iter-40
              # review: ``float(x)`` fails on complex dtypes; gate
              # on dtype kind to keep ``float()`` safe for the
              # legitimate ``float`` / ``int`` / ``bool`` config
              # fields and fall through to ``id()`` for exotic
              # dtypes (e.g. complex, which RRTMGPConfig should
              # never see).
              shape = getattr(x, "shape", None)
              if shape == ():
                  dtype_kind = getattr(getattr(x, "dtype", None), "kind", None)
                  # 'f' float, 'i' int, 'b' bool, 'u' uint — all safely
                  # coerce to Python float.  'c' complex, 'O' object,
                  # 'U' unicode, etc. fall through to id().
                  if dtype_kind in ("f", "i", "b", "u"):
                      return float(x)
              # N-D arrays (or 0-D with non-numeric dtype): id-based
              # key.  Safe under JAX immutability; a new array (e.g.
              # fresh AIMIP fit per epoch) gets a new id and
              # rebuilds the cached instance.
              return id(x)

      # iter-40: ``include_clouds`` is now ALSO in the optics key
      # (because RRTMOptics conditionally loads cloud tables on it),
      # so it's redundant here.  Keeping it would not be a
      # correctness bug, just a minor duplicate that adds no info on
      # top of the optics key tuple.  Dropped to reduce tuple size.
      return (
          RRTMGP._optics_cache_key(config),
          config.use_scan,
          # g-point accumulation strategy changes the compiled RTE solve
          # (vmap block vs per-g-point scan) and its answer to re-association
          # tolerance — key on it so a batch_size/checkpoint change rebuilds
          # the cached solver instead of silently reusing the prior one.
          getattr(config, "gpoint_batch_size", 0),
          getattr(config, "gpoint_checkpoint", True),
          getattr(config, "use_optimal_angle", False),
          config.S_0,
          config.aerosol_ssa,
          config.aerosol_g,
          # Per-band aerosol optics (tuples => hashable): a change must rebuild
          # the cached solver just like the scalar ssa/g above.
          getattr(config, "aerosol_ssa_bands", None),
          getattr(config, "aerosol_g_bands", None),
          _hashable(config.sfc_emissivity),
          _hashable(config.sfc_albedo),
          _hashable(config.sfc_albedo_direct),
          # CAM-style overhead layer changes the solved column (nlev+1).
          getattr(config, "overhead_layer", False),
      )

  @staticmethod
  def _build_optics_and_vmr(config):
      """Build (or retrieve from cache) optics scheme and VMR library.

      Parameters
      ----------
      config : RRTMGPConfig
          legoESM radiation configuration.

      Returns
      -------
      (optics_lib, vmr_lib)
      """
      key = RRTMGP._optics_cache_key(config)
      if key not in _legoesm_optics_cache:
          lw_file = config.lw_gas_file or _DEFAULT_LW_GAS
          sw_file = config.sw_gas_file or DEFAULT_SW_GAS
          lw_cloud = config.lw_cloud_file or _DEFAULT_LW_CLOUD
          sw_cloud = config.sw_cloud_file or _DEFAULT_SW_CLOUD

          rrtm_optics = RRTMOpticsConfig(
              longwave_nc_filepath=lw_file,
              shortwave_nc_filepath=sw_file,
              cloud_longwave_nc_filepath=lw_cloud,
              cloud_shortwave_nc_filepath=sw_cloud,
          )
          optics_params = OpticsParameters(optics=rrtm_optics)

          # Build VMR library with global means from legoESM config.
          global_means = {
              optics_constants.DRY_AIR_KEY: optics_constants.DRY_AIR_VMR,
              "co2": config.co2_ppmv * 1.0e-6,
              "ch4": config.ch4_ppbv * 1.0e-9,
              "n2o": config.n2o_ppbv * 1.0e-9,
              "o2": _O2_MOLE_FRACTION,
              "n2": _N2_MOLE_FRACTION,
              "co": 1.5e-7,
              "ccl4": 7.5e-11,
              "cfc11": 2.2e-10,
              "cfc12": 5.0e-10,
              "cfc22": 2.4e-10,
              "cf4": 8.5e-11,
              "no2": 3.0e-10,
          }
          vmr_lib = LookupVolumeMixingRatio(
              global_means=global_means, profiles=None,
          )

          # iter-40: pass include_clouds to skip cloud-table load for
          # clear-sky-only workflows; ~MB saved per cached entry.
          optics_lib = optics_factory(
              optics_params, vmr_lib,
              include_clouds=config.include_clouds,
          )
          if getattr(config, "compute_fp32", False):
              # Cast the optics tables float64 -> float32 (the dycore keeps
              # fp64).  GATED OFF in production: the MPAS driver sets
              # ``compute_fp32=False`` because the fp32 RTE path still crashes
              # (the shortwave direct-beam recurrence re-promotes the scan carry
              # to float64) -- see ``_run_column`` + PR #343.  Retained inert so a
              # future kernel-wide precision audit can flip the flag.
              # ``_cast_optics_f64_to_f32`` walks the RRTMOptics object + its
              # frozen-dataclass tables by hand (they are NOT registered JAX
              # pytrees, so ``tree_map`` would no-op).
              vmr_lib = _cast_optics_f64_to_f32(vmr_lib)
              optics_lib = _cast_optics_f64_to_f32(optics_lib)
          _legoesm_optics_cache[key] = (optics_lib, vmr_lib)
      return _legoesm_optics_cache[key]

  @classmethod
  def from_legoesm_config(cls, config) -> 'RRTMGP':
      """Construct an RRTMGP solver from legoESM's RRTMGPConfig.

      Parameters
      ----------
      config : RRTMGPConfig
          legoESM radiation configuration.

      Returns
      -------
      RRTMGP
          Ready-to-use solver instance.
      """
      instance = object.__new__(cls)
      optics_lib, vmr_lib = cls._build_optics_and_vmr(config)
      # Store the VMR library directly — the per-call ``atmos_state``
      # in ``solve_columns`` carries the actual zenith / albedo /
      # emissivity for the call.  iter-33 dropped the redundant
      # ``instance.atmospheric_state`` field (previously a default
      # AtmosphericState whose only purpose was to expose ``vmr``).
      instance._vmr_lib = vmr_lib
      instance.optics_lib = optics_lib
      instance._config = config
      return instance

  @classmethod
  def preload(cls, config) -> None:
      """Preload optics tables outside JIT."""
      cls._build_optics_and_vmr(config)

  @classmethod
  def preload_mpi(cls, config) -> None:
      """MPI-aware preload: rank 0 reads, broadcasts to others."""
      try:
          from mpi4py import MPI
      except ImportError:
          cls.preload(config)
          return
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      if rank == 0:
          cls.preload(config)
      key = cls._optics_cache_key(config)
      data = _legoesm_optics_cache.get(key) if rank == 0 else None
      data = comm.bcast(data, root=0)
      if rank != 0:
          _legoesm_optics_cache[key] = data

  def solve_columns(
      self,
      T: jnp.ndarray,
      p_full: jnp.ndarray,
      p_half: jnp.ndarray,
      sfc_temperature: jnp.ndarray,
      q_v: jnp.ndarray,
      cos_zenith: jnp.ndarray,
      sfc_albedo: jnp.ndarray | float | None = None,
      sfc_albedo_direct: jnp.ndarray | float | None = None,
      sfc_emissivity: jnp.ndarray | float | None = None,
      o3_vmr: jnp.ndarray | None = None,
      cloud_path_liq: jnp.ndarray | None = None,
      cloud_path_ice: jnp.ndarray | None = None,
      cloud_path_liq_lw: jnp.ndarray | None = None,
      cloud_path_ice_lw: jnp.ndarray | None = None,
      cloud_r_eff_liq: jnp.ndarray | None = None,
      cloud_r_eff_ice: jnp.ndarray | None = None,
      cloud_fraction: jnp.ndarray | None = None,
      aerosol_optical_depth: jnp.ndarray | None = None,
      aerosol_absorption_optical_depth_lw: jnp.ndarray | None = None,
      solar_spectral_fraction: jnp.ndarray | None = None,
      ghg_vmr_override: dict | None = None,
      sw_optical_field_only: bool = False,
      lw_optical_field_only: bool = False,
      *,
      mcica_cloud_fraction: jnp.ndarray | None = None,
      clear_sky: bool = False,
      o3_top_vmr: jnp.ndarray | None = None,
  ):
      """Compute radiation for legoESM column arrays.

      This is the canonical entry point for the RRTMGP solver when used
      from legoESM's physics integration layer.  It handles:

      1. Reshaping (ncol, nlev) arrays to jax-rrtmgp's (ncol, 1, nlev+2)
      2. Adding 1-cell vertical halos
      3. Building VMR dict from config concentrations
      4. Computing atmospheric state and solving two-stream
      5. Stripping halos and reshaping back

      Parameters
      ----------
      T : jnp.ndarray
          Temperature at full levels (ncol, nlev) [K].
      p_full : jnp.ndarray
          Pressure at full levels (ncol, nlev) [Pa].
      p_half : jnp.ndarray
          Pressure at interface levels (ncol, nlev+1) [Pa].
      sfc_temperature : jnp.ndarray
          Surface temperature (ncol,) [K].
      q_v : jnp.ndarray
          Water vapor specific humidity (ncol, nlev) [kg/kg].
      cos_zenith : jnp.ndarray
          Cosine of solar zenith angle (ncol,).
      sfc_albedo : jnp.ndarray | float | None
          Per-column surface albedo override.
      sfc_emissivity : jnp.ndarray | float | None
          Per-column surface emissivity override.
      o3_vmr : jnp.ndarray | None
          External ozone VMR (ncol, nlev), index 0 = TOA.
      cloud_path_liq : jnp.ndarray | None
          Liquid water path per layer (ncol, nlev) [kg/m^2].
      cloud_path_ice : jnp.ndarray | None
          Ice water path per layer (ncol, nlev) [kg/m^2].
      cloud_path_liq_lw : jnp.ndarray | None
          Separate liquid water path for the LONGWAVE solve (ncol, nlev)
          [kg/m^2].  ``None`` (default) reuses ``cloud_path_liq`` for both
          streams (byte-identical legacy behaviour).  Populated by the
          ``two_column`` partial-coverage optics, whose coverage inversion
          is stream-specific (reflectance vs emissivity space).
      cloud_path_ice_lw : jnp.ndarray | None
          Separate ice water path for the LONGWAVE solve; see
          ``cloud_path_liq_lw``.
      cloud_r_eff_liq : jnp.ndarray | None
          Liquid cloud effective radius (ncol, nlev) [m].
      cloud_r_eff_ice : jnp.ndarray | None
          Ice cloud effective radius (ncol, nlev) [m].
      aerosol_optical_depth : jnp.ndarray | None
          Prescribed shortwave aerosol optical depth per layer
          (ncol, nlev).
      aerosol_absorption_optical_depth_lw : jnp.ndarray | None
          Prescribed longwave aerosol **absorption** optical depth per
          layer (ncol, nlev) — NOT extinction.  Added to the absorption
          optical depth with ssa=0 (LW scattering neglected; a caller
          holding extinction OD must scale by the absorption fraction
          1−ω first).  ``None`` (default) leaves the longwave solution
          byte-identical.
      solar_spectral_fraction : jnp.ndarray | None
          Per-g-point solar source weights (ngpt_sw,).
      ghg_vmr_override : dict | None
          Runtime GHG VMR overrides (e.g. ``{"co2": 4.15e-4}``).
      mcica_cloud_fraction : jnp.ndarray | None
          ``(ncol, nlev)`` layer cloud fraction, index 0 = model top.  Given,
          ``cloud_path_liq/ice`` are IN-CLOUD paths and each g-point solves
          its own maximum-random subcolumn (McICA).
      o3_top_vmr : jnp.ndarray | None
          Column-mean ozone VMR above the model top (ncol,), over
          ``0 < p < p_half[:, 0]``.  Used only by the transported overhead
          layer (``config.overhead_layer``); ``None`` => that layer takes the
          top model layer's ozone.

      Returns
      -------
      RadiationOutput
          Fluxes and heating rates.
      """
      from legoesm.atmosphere.physics.radiation.output import RadiationOutput

      config = self._config
      ncol, nlev = T.shape

      # --- 0. Determine working dtype ---
      # The optics tables are loaded at whatever precision JAX was configured
      # with at load time (float32 if x64 off, float64 if x64 on).  Promote
      # all inputs to match so that table lookups, lax.cond branches, and
      # lax.scan carries have consistent dtypes throughout the solver.
      _table_dtype = self.optics_lib.gas_optics_lw.kmajor.dtype
      # McICA per-column table shift, seeded like CAM's mcica_subcol_gen:
      # the fractional part of the lowest-layer pressure [Pa], taken before
      # the cast to the table dtype.  State-derived, so it is independent of
      # the domain decomposition and needs no PRNG.  Columns with IDENTICAL
      # bottom pressure (idealized uniform states) share a shift, as they
      # share a seed in CAM.
      _mcica_shift = (None if mcica_cloud_fraction is None
                      else jax.lax.stop_gradient(jnp.mod(p_full[:, -1], 1.0)))
      T = T.astype(_table_dtype)
      p_full = p_full.astype(_table_dtype)
      p_half = p_half.astype(_table_dtype)
      q_v = q_v.astype(_table_dtype)
      sfc_temperature = jnp.asarray(sfc_temperature).astype(_table_dtype)
      cos_zenith = jnp.asarray(cos_zenith).astype(_table_dtype)
      # Optional external-forcing arrays must match the working dtype
      # too: under ``compute_fp32`` a float64 ozone / aerosol column
      # (built by the driver under JAX x64) would re-promote the optical
      # depth and break the RTE ``lax.scan`` carry dtype — exactly the
      # PR #343 leak class, resurfaced when the MPAS/spectral paths
      # started threading per-step o3_vmr / aerosol_od (2026-06-10).
      if o3_vmr is not None:
          o3_vmr = jnp.asarray(o3_vmr).astype(_table_dtype)
      if aerosol_optical_depth is not None:
          aerosol_optical_depth = jnp.asarray(
              aerosol_optical_depth).astype(_table_dtype)
      if aerosol_absorption_optical_depth_lw is not None:
          aerosol_absorption_optical_depth_lw = jnp.asarray(
              aerosol_absorption_optical_depth_lw).astype(_table_dtype)
      if cloud_path_liq is not None:
          cloud_path_liq = jnp.asarray(cloud_path_liq).astype(_table_dtype)
      if cloud_path_ice is not None:
          cloud_path_ice = jnp.asarray(cloud_path_ice).astype(_table_dtype)
      if cloud_path_liq_lw is not None:
          cloud_path_liq_lw = jnp.asarray(
              cloud_path_liq_lw).astype(_table_dtype)
      if cloud_path_ice_lw is not None:
          cloud_path_ice_lw = jnp.asarray(
              cloud_path_ice_lw).astype(_table_dtype)
      if cloud_r_eff_liq is not None:
          cloud_r_eff_liq = jnp.asarray(cloud_r_eff_liq).astype(_table_dtype)
      if cloud_r_eff_ice is not None:
          cloud_r_eff_ice = jnp.asarray(cloud_r_eff_ice).astype(_table_dtype)
      if cloud_fraction is not None:
          cloud_fraction = jnp.asarray(cloud_fraction).astype(_table_dtype)
      if mcica_cloud_fraction is not None:
          mcica_cloud_fraction = jnp.asarray(
              mcica_cloud_fraction).astype(_table_dtype)
      if ghg_vmr_override is not None:
          # Cast every numeric override (array, Python float/int, list)
          # to the working dtype; only floating leaves are retyped so
          # integer flags (if any ever appear) pass through unchanged.
          def _cast_ghg(v):
              arr = jnp.asarray(v)
              if jnp.issubdtype(arr.dtype, jnp.floating):
                  return arr.astype(_table_dtype)
              return arr
          ghg_vmr_override = {
              k: _cast_ghg(v) for k, v in ghg_vmr_override.items()
          }
      if solar_spectral_fraction is not None:
          solar_spectral_fraction = jnp.asarray(
              solar_spectral_fraction).astype(_table_dtype)

      # --- 0b. CAM-style transported layer ABOVE the model top (opt-in) ---
      # CAM RRTMG's "extra layer": one layer from p_top up to a 1 Pa lid with
      # T, q_v of the top model layer, no cloud / aerosol, and the overhead
      # column-mean ozone, so the beam reaching the model top has already
      # been depleted by the ozone above it.  Its heating is discarded and
      # the returned TOA face is the lid (true top).
      # ponytail: the ~2 W/m2 SW absorbed above the model top LEAVES the
      # column budget (TOA net - surface net != sum of model-layer heating),
      # exactly as in CAM.  Not applied to the optics-only (3D MC) paths,
      # whose per-layer output must stay on the model grid.
      overhead = (bool(getattr(config, "overhead_layer", False))
                  and not (sw_optical_field_only or lw_optical_field_only))
      if overhead:
          p_top = p_half[:, :1]
          p_lid = jnp.minimum(p_top, 1.0)  # coeff-ok: 1 Pa lid (CAM-like ~0 Pa top)

          def _top(a, k0):
              return None if a is None else jnp.concatenate([k0, a], axis=1)

          def _zero(a):
              return None if a is None else _top(a, jnp.zeros_like(a[:, :1]))

          if o3_vmr is None:  # resolve model ozone first: fallback = o3(k0)
              o3_vmr = _standard_o3_profile(p_full)
          p_half = _top(p_half, p_lid)
          p_full = _top(p_full, 0.5 * (p_lid + p_top))
          T = _top(T, T[:, :1])
          q_v = _top(q_v, q_v[:, :1])
          o3_vmr = _top(o3_vmr, o3_vmr[:, :1] if o3_top_vmr is None
                        else jnp.asarray(o3_top_vmr).astype(
                            _table_dtype).reshape(ncol, 1))
          cloud_path_liq, cloud_path_ice = _zero(cloud_path_liq), _zero(cloud_path_ice)
          cloud_path_liq_lw = _zero(cloud_path_liq_lw)
          cloud_path_ice_lw = _zero(cloud_path_ice_lw)
          cloud_fraction = _zero(cloud_fraction)
          aerosol_optical_depth = _zero(aerosol_optical_depth)
          aerosol_absorption_optical_depth_lw = _zero(
              aerosol_absorption_optical_depth_lw)
          if cloud_r_eff_liq is not None:
              cloud_r_eff_liq = _top(cloud_r_eff_liq, cloud_r_eff_liq[:, :1])
          if cloud_r_eff_ice is not None:
              cloud_r_eff_ice = _top(cloud_r_eff_ice, cloud_r_eff_ice[:, :1])
          nlev = nlev + 1

      # --- 1. Reshape (ncol, nlev) -> (ncol, 1, nlev+2) with halos ---
      T_3d = _add_halos(T[:, None, ::-1])
      p_3d = _add_halos(p_full[:, None, ::-1])
      p_3d = jnp.clip(p_3d, 1.0, None)
      # Upper-clip q_v strictly below 1 so the (1 - q_v) denominator in
      # the VMR conversion is bounded away from zero.  0.99 is far above
      # any physically plausible specific humidity (peak tropical surface
      # values are ~0.025); the bound only ever fires on numerical
      # pathology during spin-up and prevents singular/negative VMRs.
      #
      # **Clip AFTER ``_add_halos``**: linear extrapolation of a steep
      # boundary profile can produce halo values OUTSIDE [0, 0.99]
      # (e.g. q_v = [0.99, 0.0, ...] extrapolates to halo = 1.98), which
      # would yield singular / negative h2o_vmr via the
      # ``1 − q_v`` denominator.  Clipping the interior alone is not
      # enough — the halo cells are passed straight to the RRTMGP
      # solve.  Codex iter-79 stop-time review.
      q_v_3d = _add_halos(q_v[:, None, ::-1])
      q_v_3d = jnp.clip(q_v_3d, 0.0, 0.99)  # coeff-ok: specific-humidity cap

      # --- 2. Build VMR fields ---
      mol_ratio = constants.R_V / constants.R_D
      h2o_vmr = mol_ratio * q_v_3d / (1.0 - q_v_3d)

      if o3_vmr is not None:
          # Clip AFTER ``_add_halos``: linear halo extrapolation of a steep
          # boundary ozone profile (e.g. [1e-5, 1e-10]) can extrapolate to a
          # NEGATIVE halo VMR (2·1e-10 − 1e-5 < 0), which is an unphysical
          # absorber concentration in the optics lookup.  Same fix class as
          # the q_v post-clip above (codex atm-radiation review).
          o3_3d = _add_halos(o3_vmr[:, None, ::-1])
          o3_3d = jnp.clip(o3_3d, 1.0e-10, None)
      else:
          o3_3d = _standard_o3_profile(p_3d)

      vmr_fields = {
          "h2o": h2o_vmr,
          "o3": o3_3d,
      }

      if ghg_vmr_override is not None:
          for gas_name, vmr_value in ghg_vmr_override.items():
              vmr_fields[gas_name] = jnp.full_like(p_3d, vmr_value)

      # Exact layer thickness from interface pressures.
      # p_half is (ncol, nlev+1) with index 0 = TOA (sigma=0 → p≈0).
      # dp = p_half[k+1] - p_half[k] is positive (p increases toward surface).
      # Reverse to surface-first order to match T_3d, p_3d, q_v_3d.
      dp_exact = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev) TOA-first
      dp_3d = dp_exact[:, None, ::-1]  # (ncol, 1, nlev) surface-first
      # Pad with halo values (replicate boundary layers)
      dp_3d = jnp.concatenate([
          dp_3d[:, :, :1], dp_3d, dp_3d[:, :, -1:],
      ], axis=2)
      dp_3d = jnp.clip(dp_3d, 1.0, None)
      if overhead:
          # Exact overhead mass p_top - p_lid, NOT the 1 Pa floor: a top at
          # or below 1 Pa gets a ~massless (not a phantom 1 Pa) layer.
          dp_3d = dp_3d.at[:, 0, -2].set(
              jnp.maximum(dp_exact[:, 0], 1.0e-6))  # coeff-ok: div-by-zero guard [Pa]

      mol_m_air = (constants.DRY_AIR_MOL_MASS
                   + constants.WATER_MOL_MASS * h2o_vmr)
      molecules = (dp_3d / constants.G) * constants.AVOGADRO / mol_m_air

      # --- 3. Build atmospheric state ---
      optics_lib = self.optics_lib
      vmr_lib = self._vmr_lib

      cos_z_col = jnp.clip(cos_zenith, 0.0, 1.0)
      zenith_col = jnp.arccos(cos_z_col)[:, None, None]

      def _resolve_surface_field(override, fallback):
          """Promote a per-column surface field to ``(ncol, 1)``.

          Accepts a Python scalar, a 0-D JAX scalar, or a ``(ncol,)``
          / ``(ncol, 1)`` array.  AIMIP's spatial surface
          parameterization populates ``config.sfc_albedo`` /
          ``config.sfc_emissivity`` with ``(ncol,)`` arrays produced
          by a low-rank lat-lon expansion; this branch preserves the
          legacy scalar path while supporting that AIMIP use case
          without forcing every call site to thread an explicit
          override kwarg through ``_call_radiation_backend``.
          """
          source = override if override is not None else fallback
          val = jnp.asarray(source, dtype=p_3d.dtype)
          if val.ndim == 0:
              return jnp.full((ncol, 1), val, dtype=p_3d.dtype)
          if val.ndim == 1:
              return val.reshape(ncol, 1)
          return val  # already (ncol, 1)

      eff_albedo = _resolve_surface_field(sfc_albedo, config.sfc_albedo)
      eff_emis = _resolve_surface_field(sfc_emissivity, config.sfc_emissivity)
      # RAD-3 direct/diffuse albedo split: the DIRECT-beam albedo defaults to
      # the diffuse eff_albedo (legacy single-albedo) unless a direct value is
      # given via the param or config.sfc_albedo_direct.
      _alb_dir_src = (
          sfc_albedo_direct if sfc_albedo_direct is not None
          else config.sfc_albedo_direct
      )
      eff_albedo_dir = (
          None if _alb_dir_src is None
          else _resolve_surface_field(_alb_dir_src, eff_albedo)
      )

      atmos_state = atmospheric_state.AtmosphericState(
          sfc_emis=eff_emis,
          sfc_alb=eff_albedo,
          sfc_alb_dir=eff_albedo_dir,
          zenith=zenith_col,
          irrad=config.S_0,
          vmr=vmr_lib,
          toa_flux_lw=0.0,
      )

      sfc_T_2d = sfc_temperature[:, None]

      # --- Cloud properties ---
      has_clouds = config.include_clouds and (
          cloud_path_liq is not None or cloud_path_ice is not None
      )
      if has_clouds:
          _zero = jnp.zeros((ncol, nlev), dtype=T.dtype)
          _r_min = jnp.full((ncol, nlev), 1.0e-6, dtype=T.dtype)
          _cpl = cloud_path_liq if cloud_path_liq is not None else _zero
          _cpi = cloud_path_ice if cloud_path_ice is not None else _zero
          _crl = cloud_r_eff_liq if cloud_r_eff_liq is not None else _r_min
          _cri = cloud_r_eff_ice if cloud_r_eff_ice is not None else _r_min
          # Clip AFTER ``_add_halos`` (same fix class as q_v / o3 / cf):
          # linear halo extrapolation of a boundary cloud-water step can
          # produce a NEGATIVE halo water path / sub-floor halo radius,
          # which becomes a negative cloud optical depth in the halo layer
          # (optics.py scales τ by these paths).  The halo flux is stripped
          # but its optics feed the interior recurrence — so re-floor the
          # halo-expanded fields, not just the interior.
          cpl_3d = jnp.clip(_add_halos(_cpl[:, None, ::-1]), 0.0, None)
          cpi_3d = jnp.clip(_add_halos(_cpi[:, None, ::-1]), 0.0, None)
          crl_3d = jnp.clip(_add_halos(_crl[:, None, ::-1]), 1.0e-6, None)
          cri_3d = jnp.clip(_add_halos(_cri[:, None, ::-1]), 1.0e-6, None)
          # Separate LONGWAVE water paths (two_column coverage optics): same
          # halo + post-clip treatment as the SW paths; default to the SW
          # paths so every caller not passing them is byte-identical.
          cpl_lw_3d = (cpl_3d if cloud_path_liq_lw is None else jnp.clip(
              _add_halos(cloud_path_liq_lw[:, None, ::-1]), 0.0, None))
          cpi_lw_3d = (cpi_3d if cloud_path_ice_lw is None else jnp.clip(
              _add_halos(cloud_path_ice_lw[:, None, ::-1]), 0.0, None))
          if cloud_fraction is not None:
              # Clip AFTER ``_add_halos``: linear halo extrapolation of a
              # boundary cloud-fraction step (e.g. [0, 1]) produces halo
              # values OUTSIDE [0, 1] (2·0 − 1 = −1, or 2·1 − 0 = 2).  The
              # halo layer's optical depth is scaled by cf in optics.py
              # (``optical_depth * cloud_fraction``); a NEGATIVE halo cf
              # injects an unphysical negative cloud optical depth into the
              # two-stream recurrence that feeds the interior fluxes (the
              # halo flux itself is stripped, but its optics are not).  Same
              # post-clip fix class as q_v / o3 above (codex review).
              cf_3d = _add_halos(cloud_fraction[:, None, ::-1])
              cf_3d = jnp.clip(cf_3d, 0.0, 1.0)
          else:
              cf_3d = None
      else:
          cpl_3d = cpi_3d = crl_3d = cri_3d = cf_3d = None
          cpl_lw_3d = cpi_lw_3d = None

      # McICA (CAM6 mcica_subcol_gen): each g-point sees its own
      # maximum-random cloud subcolumn.  The paths passed in are IN-CLOUD;
      # the g-point's mask zeroes its clear cells, then the paths get the
      # same halo + floor treatment as above.
      lw_path_fn = sw_path_fn = None
      if has_clouds and mcica_cloud_fraction is not None:
          if (cloud_fraction is not None or cloud_path_liq_lw is not None
                  or cloud_path_ice_lw is not None):
              raise ValueError(
                  "mcica_cloud_fraction takes in-cloud paths and its own "
                  "sampling; it cannot be combined with cloud_fraction "
                  "optical-depth scaling or separate longwave paths")
          if sw_optical_field_only or lw_optical_field_only:
              raise ValueError(
                  "mcica_cloud_fraction samples clouds per g-point inside the "
                  "flux solve; the optical-field-only outputs would return "
                  "unsampled in-cloud optics")
          from legoesm.atmosphere.physics.clouds.subcolumns import (
              generate_subcolumns)

          def _mcica_paths(n_gpt):
              mask = generate_subcolumns(mcica_cloud_fraction, n_gpt,
                                         shift=_mcica_shift)
              if overhead:
                  # the overhead layer is cloud-free; sample the MODEL layers
                  # only, so each g-point's model-layer subcolumn is the one
                  # it draws with the layer off (merge of fu/halo with main's
                  # McICA: mcica_cloud_fraction stays (ncol, nlev))
                  mask = jnp.concatenate(
                      [jnp.zeros_like(mask[:, :, :1]), mask], axis=2)

              def paths(igpt):
                  m = mask[igpt]
                  return tuple(
                      jnp.clip(_add_halos(
                          jnp.where(m, x, 0.0)[:, None, ::-1]), 0.0, None)
                      for x in (_cpl, _cpi))
              return paths

          lw_path_fn = _mcica_paths(optics_lib.n_gpt_lw)
          sw_path_fn = _mcica_paths(optics_lib.n_gpt_sw)

      # Optional aerosol optical depth (shortwave).  Clip AFTER ``_add_halos``
      # (same fix class as q_v / o3 / cf / cloud paths): linear halo
      # extrapolation can drive a boundary aerosol OD negative, which is an
      # unphysical (negative) optical depth feeding the interior recurrence.
      if aerosol_optical_depth is not None:
          aerosol_od_3d = jnp.clip(
              _add_halos(aerosol_optical_depth[:, None, ::-1]), 0.0, None,
          )
      else:
          aerosol_od_3d = None

      # Aerosol SW single-scattering albedo + asymmetry: scalar (grey, default)
      # or per-shortwave-band arrays (config.aerosol_ssa_bands/aerosol_g_bands).
      # Resolved once here and shared by the MC-optics short-circuit and the
      # full SW solve so the per-band selection is defined in exactly one place.
      aer_ssa, aer_g = _resolve_aerosol_sw_optics(config, optics_lib)

      # Optional aerosol optical depth (longwave, pure absorber)
      if aerosol_absorption_optical_depth_lw is not None:
          aerosol_od_lw_3d = jnp.clip(
              _add_halos(aerosol_absorption_optical_depth_lw[:, None, ::-1]),
              0.0, None,
          )
      else:
          aerosol_od_lw_3d = None

      # Optional spectral solar forcing
      if solar_spectral_fraction is not None:
          solar_weights = jnp.clip(jnp.asarray(solar_spectral_fraction), 0.0, None)
          denom = jnp.maximum(jnp.sum(solar_weights), 1.0e-30)
          solar_weights = solar_weights / denom
          if solar_weights.shape[0] != optics_lib.n_gpt_sw:
              raise ValueError(
                  "solar_spectral_fraction has wrong length: "
                  f"{solar_weights.shape[0]} (expected {optics_lib.n_gpt_sw})",
              )
      else:
          solar_weights = None

      # --- 3b. Optics-only short-circuit for the 3D MC ray tracer ---
      # Reuses the EXACT state above (no duplicated numerics); returns the
      # per-g-point shortwave optical field instead of solving transport. The
      # default path (sw_optical_field_only=False) is byte-identical.
      if clear_sky and (sw_optical_field_only or lw_optical_field_only):
          raise ValueError(
              "clear_sky adds clear-sky TOA fluxes to the RadiationOutput "
              "path; the optical-field-only outputs have no fluxes.")

      if sw_optical_field_only:
          sw_props = two_stream.compute_sw_optical_field(
              p_3d, T_3d, molecules, optics_lib, vmr_fields,
              cloud_r_eff_liq=crl_3d, cloud_path_liq=cpl_3d,
              cloud_r_eff_ice=cri_3d, cloud_path_ice=cpi_3d,
              cloud_fraction=cf_3d,
              aerosol_optical_depth=aerosol_od_3d,
              aerosol_single_scattering_albedo=aer_ssa,
              aerosol_asymmetry_factor=aer_g,
          )
          # Strip the singleton Y axis + vertical halos, flip back to the
          # legoESM TOA-first convention -> (n_gpt, ncol, nlev).
          hw = 1

          def _strip(a):
              return a[:, :, 0, hw:-hw][:, :, ::-1]

          tau_gpt = _strip(sw_props['optical_depth'])
          ssa_gpt = _strip(sw_props['ssa'])
          g_gpt = _strip(sw_props['asymmetry_factor'])
          rayleigh_frac_gpt = _strip(sw_props['rayleigh_frac'])
          # Cloud liquid effective radius [um] per cell for Mie-LUT sampling
          # (microhh LUT spans 2.5..21.5 um). Clear-sky -> the LUT floor.
          # crl_3d is 3D (ncol, 1, nlev+2) -- NOT per-g-point 4D -- so it needs a
          # 3D strip (drop singleton, trim halo, z-reverse to TOA-first), not the
          # 4D _strip used for the per-g-point optics.
          if crl_3d is not None:
              r_eff_um = crl_3d[:, 0, hw:-hw][:, ::-1] * 1.0e6   # (ncol, nlev)
          else:
              r_eff_um = jnp.full((ncol, nlev), 2.5, dtype=tau_gpt.dtype)  # coeff-ok: microhh Mie LUT lower r_eff bound [um]
          weights = solar_weights
          if weights is None:
              weights = optics_lib.solar_fraction_by_gpt
          # Beam-normal solar flux per g-point [W/m^2].
          solar_flux_normal = config.S_0 * weights
          return (tau_gpt, ssa_gpt, g_gpt, rayleigh_frac_gpt, r_eff_um,
                  solar_flux_normal)

      # --- 3c. Optics-only short-circuit for the 3D MC LW emission tracer ---
      # Reuses the EXACT state above; returns per-g-point LW absorption optical
      # depth + Planck sources instead of solving transport. Default path
      # (lw_optical_field_only=False) is byte-identical.
      if lw_optical_field_only:
          lw_props = two_stream.compute_lw_optical_field(
              p_3d, T_3d, molecules, optics_lib, sfc_T_2d, vmr_fields,
              cloud_r_eff_liq=crl_3d, cloud_path_liq=cpl_lw_3d,
              cloud_r_eff_ice=cri_3d, cloud_path_ice=cpi_lw_3d,
              cloud_fraction=cf_3d,
              aerosol_absorption_optical_depth=aerosol_od_lw_3d,
          )
          hw = 1

          def _strip3d(a):  # (n_gpt, ncol, 1, nlev+2) -> (n_gpt, ncol, nlev) TOA-first
              return a[:, :, 0, hw:-hw][:, :, ::-1]

          abs_od_gpt = _strip3d(lw_props['abs_optical_depth'])
          planck_gpt = _strip3d(lw_props['planck_src'])
          # Cell-face Planck (lower/upper-z faces) for Phase-3c linear-in-tau
          # in-cell emission. The strip+flip preserves each cell's lower-z /
          # upper-z face labeling (see compute_plane_lw_heating_spectral).
          planck_bot_gpt = _strip3d(lw_props['planck_src_bottom'])
          planck_top_gpt = _strip3d(lw_props['planck_src_top'])
          # Surface Planck is 2D per g-point (n_gpt, ncol, 1) -> (n_gpt, ncol).
          planck_sfc_gpt = lw_props['planck_src_sfc'][:, :, 0]
          return (abs_od_gpt, planck_gpt, planck_bot_gpt, planck_top_gpt,
                  planck_sfc_gpt)

      # --- 4. Solve LW ---
      lw_fluxes = two_stream.solve_lw(
          p_3d,
          T_3d,
          molecules,
          optics_lib,
          atmos_state,
          vmr_fields,
          sfc_T_2d,
          cloud_r_eff_liq=crl_3d,
          cloud_path_liq=cpl_lw_3d,
          cloud_r_eff_ice=cri_3d,
          cloud_path_ice=cpi_lw_3d,
          cloud_fraction=cf_3d,
          aerosol_absorption_optical_depth=aerosol_od_lw_3d,
          use_scan=config.use_scan,
          use_optimal_angle=getattr(config, "use_optimal_angle", False),
          gpoint_batch_size=getattr(config, "gpoint_batch_size", 0),
          gpoint_checkpoint=getattr(config, "gpoint_checkpoint", True),
          cloud_path_fn=lw_path_fn,
          clear_sky=clear_sky,
      )

      # --- 5. Solve SW ---
      sw_fluxes = two_stream.solve_sw(
          p_3d,
          T_3d,
          molecules,
          optics_lib,
          atmos_state,
          vmr_fields,
          cloud_r_eff_liq=crl_3d,
          cloud_path_liq=cpl_3d,
          cloud_r_eff_ice=cri_3d,
          cloud_path_ice=cpi_3d,
          cloud_fraction=cf_3d,
          aerosol_optical_depth=aerosol_od_3d,
          aerosol_single_scattering_albedo=aer_ssa,
          aerosol_asymmetry_factor=aer_g,
          solar_fraction_by_gpt=solar_weights,
          use_scan=config.use_scan,
          gpoint_batch_size=getattr(config, "gpoint_batch_size", 0),
          gpoint_checkpoint=getattr(config, "gpoint_checkpoint", True),
          cloud_path_fn=sw_path_fn,
          clear_sky=clear_sky,
      )

      # --- 6. Compute heating rates using exact layer thickness ---
      lw_hr_3d = two_stream.compute_heating_rate(
          lw_fluxes['flux_net'], p_3d, dp=dp_3d,
      )
      sw_hr_3d = two_stream.compute_heating_rate(
          sw_fluxes['flux_net'], p_3d, dp=dp_3d,
      )

      # --- 7. Strip halos, flip back to legoESM convention, reshape ---
      hw = 1
      lw_up = lw_fluxes['flux_up'][:, 0, hw:][:, ::-1]
      lw_down = lw_fluxes['flux_down'][:, 0, hw:][:, ::-1]
      sw_up = sw_fluxes['flux_up'][:, 0, hw:][:, ::-1]
      sw_down = sw_fluxes['flux_down'][:, 0, hw:][:, ::-1]

      lw_hr = lw_hr_3d[:, 0, hw:-hw][:, ::-1]
      sw_hr = sw_hr_3d[:, 0, hw:-hw][:, ::-1]

      if overhead:
          # Drop the overhead layer's heating and the model-top face: face 0
          # stays the TRUE top (lid), so TOA diagnostics see the full column.
          def _faces(a):
              return jnp.concatenate([a[:, :1], a[:, 2:]], axis=1)

          lw_up, lw_down = _faces(lw_up), _faces(lw_down)
          sw_up, sw_down = _faces(sw_up), _faces(sw_down)
          lw_hr, sw_hr = lw_hr[:, 1:], sw_hr[:, 1:]

      return RadiationOutput(
          lw_flux_up=lw_up,
          lw_flux_down=lw_down,
          sw_flux_up=sw_up,
          sw_flux_down=sw_down,
          heating_rate=lw_hr + sw_hr,
          lw_heating_rate=lw_hr,
          sw_heating_rate=sw_hr,
          # internal index -1 is the TOA face (== *_flux_up[:, 0] above)
          lw_flux_up_toa_clr=(lw_fluxes['flux_up_clr'][:, 0, -1]
                              if clear_sky else None),
          sw_flux_up_toa_clr=(sw_fluxes['flux_up_clr'][:, 0, -1]
                              if clear_sky else None),
      )

  def solve_columns_chunked(
      self,
      *,
      column_chunk_size: int,
      T: jnp.ndarray,
      p_full: jnp.ndarray,
      p_half: jnp.ndarray,
      sfc_temperature: jnp.ndarray,
      q_v: jnp.ndarray,
      cos_zenith: jnp.ndarray,
      **opt,
  ):
      """Column-chunked wrapper over :meth:`solve_columns`.

      Splits the leading column axis into ``ncol // column_chunk_size``
      fixed-size blocks and maps :meth:`solve_columns` over them with
      ``jax.lax.map``.  Radiation columns are physically INDEPENDENT (the
      k-distribution and two-stream solve have no horizontal coupling), so
      the result is NUMERICALLY EXACT — bit-for-bit identical to the
      unchunked call, not an approximation.  The point is COMPILE TIME: XLA
      lowers the per-block body ONCE at ``column_chunk_size`` columns, so the
      compiled graph size (and the highly super-linear rrtmgp JIT cost) is
      capped independent of the total ``ncol`` — this lets the higher
      horizontal resolutions (C24/C48 at L20) compile instead of stalling.

      LEVELS are NOT chunked: the vertical is coupled through the two-stream
      recurrence, so only the column axis is decomposable.

      ``jax.lax.map`` is scan-based, so the wrapper is fully reverse-mode
      differentiable (AD-safe) and adds no host/device thrash.

      Parameters
      ----------
      column_chunk_size : int
          Columns per compiled block.  ``0``/non-positive, or ``>= ncol``,
          disables chunking (a plain :meth:`solve_columns` call).  Must
          divide ``ncol`` exactly otherwise.
      T, p_full, p_half, sfc_temperature, q_v, cos_zenith :
          Required per-column inputs (leading axis ``ncol``); see
          :meth:`solve_columns`.
      **opt :
          Any optional :meth:`solve_columns` keyword.  A leaf whose leading
          axis equals ``ncol`` is treated as a per-column array and chunked
          alongside the required inputs; everything else (Python/0-D
          scalars, ``None``, the ``ghg_vmr_override`` dict, the per-g-point
          ``solar_spectral_fraction`` whose leading axis is ``n_gpt`` not
          ``ncol``) is broadcast unchanged to every block.

      Returns
      -------
      RadiationOutput
          Reassembled over ``ncol`` — identical to :meth:`solve_columns`.
      """
      ncol = T.shape[0]
      # Disabled / degenerate → exact plain call.  This precedes every
      # chunking-specific check so a disabled call is a TRUE passthrough:
      # it honours any optics-only flags and returns exactly what
      # solve_columns returns (no reshape overhead, no contract change).
      if (not column_chunk_size or column_chunk_size <= 0
              or ncol <= column_chunk_size):
          return self.solve_columns(
              T=T, p_full=p_full, p_half=p_half,
              sfc_temperature=sfc_temperature, q_v=q_v,
              cos_zenith=cos_zenith, **opt,
          )
      # --- chunking is ACTIVE from here down ---
      # The optics-only short-circuits return a per-g-point TUPLE whose leaf
      # layout ((n_gpt, ncol, nlev)) carries no leading (n_block, chunk)
      # column axis, so the reassembly reshape below would corrupt it.
      # Chunking supports the RadiationOutput path only.
      if opt.get("sw_optical_field_only") or opt.get("lw_optical_field_only"):
          raise ValueError(
              "solve_columns_chunked supports only the RadiationOutput path; "
              "sw_optical_field_only / lw_optical_field_only return a "
              "per-g-point tuple incompatible with column chunking — call "
              "solve_columns directly for the optics-only tracers."
          )
      if ncol % column_chunk_size != 0:
          raise ValueError(
              f"column_chunk_size {column_chunk_size} must divide ncol "
              f"{ncol} exactly (radiation columns are independent, but the "
              "lax.map reshape requires equal-size blocks). Note that ncol "
              "here is the SOLVER's column count: with max-random overlap "
              "it is cloud_n_subcolumns * the grid's column count, so a "
              "block size chosen against the grid alone need not divide it."
          )
      n_block = ncol // column_chunk_size

      # Per-column CONFIG surface fallbacks (sfc_albedo / sfc_emissivity /
      # sfc_albedo_direct) are read at GLOBAL ncol inside solve_columns via
      # ``_resolve_surface_field(override, self._config.<field>)`` whenever the
      # explicit per-call override is None.  Only EXPLICIT overrides (in opt)
      # are chunked here, so a per-column ``(ncol, ...)`` config fallback would
      # be read at global ncol inside each chunk-local solve → size mismatch
      # and a broken / non-exact result.  Fail loudly and tell the caller to
      # pass the field explicitly (which IS chunked); the production AMIP path
      # already passes sfc_albedo / sfc_emissivity as explicit args, so this
      # never fires there — it protects the direct-call contract + exactness.
      cfg = getattr(self, "_config", None)
      for _sfc_name in ("sfc_albedo", "sfc_emissivity", "sfc_albedo_direct"):
          _fallback = getattr(cfg, _sfc_name, None)
          if (opt.get(_sfc_name) is None and hasattr(_fallback, "shape")
                  and getattr(_fallback, "ndim", 0) >= 1
                  and _fallback.shape[0] == ncol):
              raise ValueError(
                  f"solve_columns_chunked: config fallback '{_sfc_name}' is a "
                  f"per-column ({ncol},...) array but no explicit '{_sfc_name}' "
                  "override was passed. The config surface field is read at "
                  "global ncol inside each chunk block (via "
                  "_resolve_surface_field) → size mismatch under chunking. "
                  f"Pass '{_sfc_name}' as an explicit per-call argument (it is "
                  "chunked) instead of relying on the per-column config field."
              )

      # Split every per-column array on its leading axis into
      # (n_block, column_chunk_size, ...).  This row-major reshape is
      # order-preserving and the reassembly reshape below is its exact
      # inverse, so no column is reordered — the exactness guarantee.
      def _is_col_array(x):
          return (hasattr(x, "shape") and getattr(x, "ndim", 0) >= 1
                  and x.shape[0] == ncol)

      def _split(x):
          return x.reshape(n_block, column_chunk_size, *x.shape[1:])

      mapped = {
          "T": _split(T), "p_full": _split(p_full), "p_half": _split(p_half),
          "sfc_temperature": _split(sfc_temperature), "q_v": _split(q_v),
          "cos_zenith": _split(cos_zenith),
      }
      # ``solar_spectral_fraction`` is a per-g-point vector (leading axis
      # n_gpt, NOT ncol) that must be broadcast to every block, never split.
      # Excluded by name so it stays static even in the degenerate
      # ncol == n_gpt_sw case where the shape heuristic alone would misfire
      # (splitting it → each block sees a short weight vector → solve_columns
      # raises on the wrong g-point length).
      _non_column_opt = ("solar_spectral_fraction",)
      static = {}
      for k, v in opt.items():
          if k not in _non_column_opt and _is_col_array(v):
              mapped[k] = _split(v)
          else:
              static[k] = v

      # Checkpointed per block: lax.map is a scan, so without this the
      # backward pass keeps every block's radiation activations at once and
      # the chunking buys compile time but no memory.  With it the backward
      # recomputes one block at a time, so reverse-mode scratch is bounded by
      # ``column_chunk_size`` columns instead of ``ncol`` -- the AMIP WB arm
      # needs it because max-random overlap hands the solver n_sub*ncol
      # sub-columns (measured 490 GiB of scratch at 8 sub-columns, T63/L32).
      # Forward-only callers are unaffected (checkpoint is a no-op there) and
      # the values are unchanged either way.  prevent_cse=False because this
      # body already runs inside a scan (same reason as the g-point block
      # loop in rte/two_stream.py).
      @functools.partial(
          jax.checkpoint,
          policy=jax.checkpoint_policies.nothing_saveable,
          prevent_cse=False,
      )
      def _one(chunk):
          return self.solve_columns(**chunk, **static)

      out = jax.lax.map(_one, mapped)  # leaves: (n_block, chunk, ...)
      # Merge (n_block, chunk) back to ncol — the exact inverse of _split.
      return jax.tree.map(lambda a: a.reshape(ncol, *a.shape[2:]), out)
