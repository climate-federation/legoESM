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

"""Implementations of `OpticsScheme`s and a factory method."""

from collections.abc import Mapping
from typing import Callable, TypeAlias, cast

import logging
import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.radiation.rrtmgp.config import radiative_transfer
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cloud_optics
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import gas_optics
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_cloud_optics
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_base
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_longwave
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_shortwave
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_volume_mixing_ratio
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics_base
from typing_extensions import override

Array: TypeAlias = jax.Array
AbstractLookupGasOptics: TypeAlias = (
    lookup_gas_optics_base.AbstractLookupGasOptics
)
LookupCloudOptics: TypeAlias = lookup_cloud_optics.LookupCloudOptics
LookupGasOpticsLongwave: TypeAlias = (
    lookup_gas_optics_longwave.LookupGasOpticsLongwave
)
LookupGasOpticsShortwave: TypeAlias = (
    lookup_gas_optics_shortwave.LookupGasOpticsShortwave
)
LookupVolumeMixingRatio: TypeAlias = (
    lookup_volume_mixing_ratio.LookupVolumeMixingRatio
)

_EPSILON = 1e-6


class RRTMOptics(optics_base.OpticsScheme):
  """The Rapid Radiative Transfer Model (RRTM) optics scheme implementation."""

  def __init__(
      self,
      vmr_lib: LookupVolumeMixingRatio,
      params: radiative_transfer.OpticsParameters,
      include_clouds: bool = True,
  ):
    """
    Args:
      vmr_lib: Lookup table of volume mixing ratios.
      params: Optics parameters (gas + cloud NetCDF file paths).
      include_clouds: When False, skip loading the cloud_optics_lw /
        cloud_optics_sw tables.  iter-40 memory optimisation:
        ``RRTMOptics`` is constructed from cached entries
        (``_legoesm_optics_cache``), and clear-sky-only workflows
        previously paid for the ~MB cloud-table load every time.
        ``solve_columns`` only invokes the cloud branch when
        ``has_clouds=config.include_clouds and (cloud_path_liq is not
        None or cloud_path_ice is not None)``, so the cloud_optics
        attributes are safely ``None``-able when include_clouds=False.
    """
    super().__init__()
    assert isinstance(params.optics, radiative_transfer.RRTMOptics)
    rrtm_params = params.optics
    self.vmr_lib = vmr_lib
    if include_clouds:
      self.cloud_optics_lw = lookup_cloud_optics.from_data_file(
          rrtm_params.cloud_longwave_nc_filepath
      )
      self.cloud_optics_sw = lookup_cloud_optics.from_data_file(
          rrtm_params.cloud_shortwave_nc_filepath
      )
    else:
      # Clear-sky configuration: skip the cloud-table load entirely.
      # solve_columns' has_clouds gate ensures the cloud branch is
      # never invoked, so these attributes stay None-safe.
      self.cloud_optics_lw = None
      self.cloud_optics_sw = None
    self.gas_optics_lw = lookup_gas_optics_longwave.from_data_file(
        rrtm_params.longwave_nc_filepath
    )
    self.gas_optics_sw = lookup_gas_optics_shortwave.from_data_file(
        rrtm_params.shortwave_nc_filepath
    )
    # Type narrowing
    assert self.gas_optics_lw is not None
    assert self.gas_optics_sw is not None

  def _od_fn(
      self,
      is_lw: bool,
      igpt: Array,
      molecules: Array,
      temperature: Array,
      pressure: Array,
      vmr_fields: dict[int, Array] | None,
  ) -> Array:
    """The actual optical_depth calculation."""
    logging.info('Calling optical depth graph.')
    lookup_gas_optics = self.gas_optics_lw if is_lw else self.gas_optics_sw
    return gas_optics.compute_minor_optical_depth(
        lookup_gas_optics,
        self.vmr_lib,
        molecules,
        temperature,
        pressure,
        igpt,
        vmr_fields,
    ) + gas_optics.compute_major_optical_depth(
        lookup_gas_optics,
        self.vmr_lib,
        molecules,
        temperature,
        pressure,
        igpt,
        vmr_fields,
    )

  def _optical_depth_fn(
      self, igpt: Array, is_lw: bool
  ) -> Callable[[Array, Array, Array, dict[int, Array] | None], Array]:
    """Create a function for computing the optical depth.

    Args:
      igpt: The g-point that will be used to index into the RRTMGP lookup table.
      is_lw: If `True`, uses the longwave lookup. Otherwise, uses the shortwave
        lookup.

    Returns:
      A callable that computes the optical depth given fields for the number of
      molecules per area, pressure, and volume mixing ratio of various gases.
    """

    def od_fn(
        molecules: Array,
        temperature: Array,
        pressure: Array,
        vmr_fields: dict[int, Array] | None,
    ) -> Array:
      return self._od_fn(
          is_lw, igpt, molecules, temperature, pressure, vmr_fields
      )

    return od_fn

  def _rayl_fn(
      self,
      igpt: Array,
      molecules: Array,
      temperature: Array,
      pressure: Array,
      vmr_fields: dict[int, Array] | None,
  ) -> Array:
    """The actual Rayleigh scattering calculation."""
    logging.info('Calling Rayleigh scattering graph.')
    return gas_optics.compute_rayleigh_optical_depth(
        self.gas_optics_sw,
        self.vmr_lib,
        molecules,
        temperature,
        pressure,
        igpt,
        vmr_fields,
    )

  def rayleigh_scattering_fn(
      self, igpt: Array
  ) -> Callable[[Array, Array, Array, dict[int, Array] | None], Array]:
    """Create a function for computing Rayleigh scattering.

    Args:
      igpt: The g-point that will be used to index into the RRTMGP lookup table.

    Returns:
      A callable that computes the Rayleigh scattering optical depth given
      fields for the number of molecules per area, temperature, pressure, and
      and volume mixing ratio of various gases.
    """

    def rayl_fn(
        molecules: Array,
        temperature: Array,
        pressure: Array,
        vmr_fields: dict[int, Array] | None,
    ) -> Array:
      return self._rayl_fn(igpt, molecules, temperature, pressure, vmr_fields)

    return rayl_fn

  def _pf_fn(
      self,
      igpt: Array,
      pressure: Array,
      temperature: Array,
      vmr_fields: dict[int, Array] | None = None,
  ) -> Array:
    """The actual Planck fraction calculation."""
    logging.info('Calling Planck fraction graph.')
    return gas_optics.compute_planck_fraction(
        self.gas_optics_lw,
        self.vmr_lib,
        pressure,
        temperature,
        igpt,
        vmr_fields,
    )

  def planck_fraction_fn(
      self,
      igpt: Array,
  ):
    """Create a function for computing the Planck fraction.

    Args:
      igpt: The g-point that will be used to index into the RRTMGP lookup table.

    Returns:
      A callable that computes the Planck fraction given fields for pressure,
      temperature and volume mixing ratio of various gases.
    """

    def pf_fn(
        pressure: Array,
        temperature: Array,
        vmr_fields: dict[int, Array] | None = None,
    ) -> Array:
      return self._pf_fn(igpt, pressure, temperature, vmr_fields)

    return pf_fn

  def _ps_fn(
      self,
      igpt: Array,
      planck_fraction: Array,
      temperature: Array,
  ) -> Array:
    """The actual Planck source calculation."""
    logging.info('Calling Planck source graph.')
    return gas_optics.compute_planck_sources(
        self.gas_optics_lw, planck_fraction, temperature, igpt
    )

  def planck_src_fn(
      self,
      igpt: Array,
  ):
    """Create a function for computing the Planck source.

    Args:
      igpt: The g-point that will be used to index into the RRTMGP lookup table.

    Returns:
      A callable that computes the Planck source given fields for the
      precomputed pointwise Planck fraction and temperature.
    """

    def ps_fn(planck_fraction: Array, temperature: Array) -> Array:
      return self._ps_fn(igpt, planck_fraction, temperature)

    return ps_fn

  def _cloud_props(
      self, ibnd, is_lw, r_eff_liq, cloud_path_liq, r_eff_ice, cloud_path_ice
  ) -> dict[str, Array]:
    """The actual cloud optical properties calculation."""
    logging.info('Calling cloud optical properties graph.')
    cloud_lookup = self.cloud_optics_lw if is_lw else self.cloud_optics_sw
    # iter-41: defensive guard.  The public entry point
    # ``_combine_gas_and_cloud_properties`` already errors out when
    # ``cloud_optics_*`` is None (iter-40 include_clouds=False
    # path), but ``_cloud_props`` is exposed via
    # ``cloud_properties_fn`` so a direct caller could still hit it.
    assert cloud_lookup is not None, (
        "_cloud_props called with cloud_optics_lw/sw=None; "
        "rebuild RRTMOptics with include_clouds=True."
    )
    return cloud_optics.compute_optical_properties(
        cloud_lookup,
        cloud_path_liq,
        cloud_path_ice,
        r_eff_liq,
        r_eff_ice,
        ibnd=ibnd,
    )

  def cloud_properties_fn(
      self,
      ibnd: Array,
      is_lw: bool,
  ):
    """Create a function for computing cloud optical properties.

    Args:
      ibnd: The spectral band index that will be used to index into the lookup
        tables for cloud absorption coefficients.
      is_lw: If `True`, uses the longwave lookup. Otherwise, uses the shortwave
        lookup.

    Returns:
      A callable that returns a dictionary containing the cloud optical depth,
      single-scattering albedo, and asymmetry factor.
    """

    def cloud_props_fn(r_eff_liq, cloud_path_liq, r_eff_ice, cloud_path_ice):
      return self._cloud_props(
          ibnd,
          is_lw,
          r_eff_liq,
          cloud_path_liq,
          r_eff_ice,
          cloud_path_ice,
      )

    return cloud_props_fn

  def _apply_delta_scaling_for_cloud(
      self,
      cloud_optical_props: Mapping[str, Array],
  ) -> dict[str, Array]:
    """Delta-scales optical properties for shortwave bands."""
    optical_depth = cloud_optical_props['optical_depth']
    ssa = cloud_optical_props['ssa']
    g = cloud_optical_props['asymmetry_factor']

    # Apply delta scaling.  Use ``safe_divide`` instead of
    # ``num / jnp.maximum(denom, eps)`` for the two divides — the
    # latter is forward-safe but has a ``-num / denom**2`` reverse-
    # mode VJP that overflows when ``denom`` is at the floor
    # (``denom**2 = eps**2 = 1e-12`` underflows to 0 under fp64).
    # Same pattern as the iter-14 restoration of commit 59407953;
    # codex iter-18 review flagged this as a survivor.
    wf = ssa * g**2
    cloud_tau = (1 - wf) * optical_depth
    cloud_ssa = safe_divide(ssa - wf, 1 - wf, eps=_EPSILON, fill=0.0)
    cloud_asy = safe_divide(g - g**2, 1 - g**2, eps=_EPSILON, fill=0.0)

    return {
        'optical_depth': cloud_tau,
        'ssa': cloud_ssa,
        'asymmetry_factor': cloud_asy,
    }

  def add_cloud_optical_properties(
      self,
      igpt: Array,
      gas_optical_props: Mapping[str, Array],
      is_lw: bool,
      cloud_r_eff_liq: Array | None = None,
      cloud_path_liq: Array | None = None,
      cloud_r_eff_ice: Array | None = None,
      cloud_path_ice: Array | None = None,
      cloud_fraction: Array | None = None,
  ) -> Mapping[str, Array]:
    """Add cloud optics to already-computed gas optics for g-point ``igpt``.

    The combination ``compute_lw/sw_optical_properties`` apply; with no cloud
    path the gas properties are returned unchanged.  Lets a caller that needs
    both the clear and the cloudy optics compute the gas optics once.
    """
    if cloud_path_liq is None and cloud_path_ice is None:
      return gas_optical_props
    return self._combine_gas_and_cloud_properties(
        igpt,
        gas_optical_props,
        is_lw=is_lw,
        radius_eff_liq=cloud_r_eff_liq,
        cloud_path_liq=cloud_path_liq,
        radius_eff_ice=cloud_r_eff_ice,
        cloud_path_ice=cloud_path_ice,
        cloud_fraction=cloud_fraction,
    )

  def _combine_gas_and_cloud_properties(
      self,
      igpt: Array,
      optical_props: Mapping[str, Array],
      is_lw: bool,
      radius_eff_liq: Array | None = None,
      cloud_path_liq: Array | None = None,
      radius_eff_ice: Array | None = None,
      cloud_path_ice: Array | None = None,
      cloud_fraction: Array | None = None,
  ) -> dict[str, Array]:
    """Combine the gas optical properties with the cloud optical properties.

    When ``cloud_fraction`` is provided, cloud optical depths are scaled
    by the fractional cloud cover so that partially-cloudy grid cells
    have proportionally reduced cloud radiative effect.
    """
    # iter-41 codex review: explicit None-guard for the cloud-optics
    # lookups.  When ``RRTMOptics`` was constructed with
    # ``include_clouds=False`` (iter-40), the cloud_optics_lw/sw
    # attributes are ``None`` and the public ``solve_columns`` gate
    # ensures cloud paths are also None — so this branch isn't
    # reached.  But a direct caller that bypasses ``solve_columns``
    # and invokes ``compute_lw_optical_properties`` with cloud paths
    # would hit ``None.cloud_optics_*`` here.  Raise a clear error
    # instead of letting the AttributeError propagate.
    cloud_lookup = self.cloud_optics_lw if is_lw else self.cloud_optics_sw
    if cloud_lookup is None:
      raise ValueError(
          "RRTMOptics was constructed with include_clouds=False but "
          "cloud_path_liq or cloud_path_ice is non-None.  Either rebuild "
          "the solver with include_clouds=True or omit the cloud kwargs."
      )
    gas_lookup = self.gas_optics_lw if is_lw else self.gas_optics_sw
    assert gas_lookup is not None  # Type narrowing.

    # If any of the input cloud states are `None`, replace them with zeros.
    if radius_eff_liq is None:
      radius_eff_liq = jnp.zeros_like(optical_props['ssa'])
    if cloud_path_liq is None:
      cloud_path_liq = jnp.zeros_like(optical_props['ssa'])
    if radius_eff_ice is None:
      radius_eff_ice = jnp.zeros_like(optical_props['ssa'])
    if cloud_path_ice is None:
      cloud_path_ice = jnp.zeros_like(optical_props['ssa'])

    ibnd = gas_lookup.g_point_to_bnd[igpt]

    compute_cloud_properties_fn = self.cloud_properties_fn(ibnd, is_lw)
    cloud_optical_props = compute_cloud_properties_fn(
        radius_eff_liq, cloud_path_liq, radius_eff_ice, cloud_path_ice
    )

    # Scale cloud optical depth by cloud fraction for partial coverage
    if cloud_fraction is not None:
      cloud_optical_props = {
          'optical_depth': cloud_optical_props['optical_depth'] * cloud_fraction,
          'ssa': cloud_optical_props['ssa'],
          'asymmetry_factor': cloud_optical_props['asymmetry_factor'],
      }

    if not is_lw:
      cloud_optical_props = self._apply_delta_scaling_for_cloud(
          cloud_optical_props
      )
    return self.combine_optical_properties(optical_props, cloud_optical_props)

  @override
  def compute_lw_optical_properties(
      self,
      pressure: Array,
      temperature: Array,
      molecules: Array,
      igpt: Array,
      vmr_fields: dict[int, Array] | None = None,
      cloud_r_eff_liq: Array | None = None,
      cloud_path_liq: Array | None = None,
      cloud_r_eff_ice: Array | None = None,
      cloud_path_ice: Array | None = None,
      cloud_fraction: Array | None = None,
  ) -> dict[str, Array]:
    """Compute the monochromatic longwave optical properties.

    Uses the RRTM optics scheme to compute the longwave optical depth, albedo,
    and asymmetry factor. These raw optical properties can be further
    transformed downstream to better suit the assumptions of the particular
    radiative transfer solver being used.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules / m^2].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].
      cloud_fraction: Cloud fraction per layer [0, 1] for partial-coverage
        scaling of cloud optical depth.

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The longwave optical depth.
        'ssa': The longwave single-scattering albedo.
        'asymmetry_factor': The longwave asymmetry factor.
    """
    optical_depth_fn = self._optical_depth_fn(igpt, is_lw=True)
    optical_depth_lw = optical_depth_fn(
        molecules, temperature, pressure, vmr_fields
    )
    optical_props = {
        'optical_depth': optical_depth_lw,
        'ssa': jnp.zeros_like(optical_depth_lw),
        'asymmetry_factor': jnp.zeros_like(optical_depth_lw),
    }
    return self.add_cloud_optical_properties(
        igpt, optical_props, True, cloud_r_eff_liq, cloud_path_liq,
        cloud_r_eff_ice, cloud_path_ice, cloud_fraction=cloud_fraction)

  @override
  def compute_sw_optical_properties(
      self,
      pressure: Array,
      temperature: Array,
      molecules: Array,
      igpt: Array,
      vmr_fields: dict[int, Array] | None = None,
      cloud_r_eff_liq: Array | None = None,
      cloud_path_liq: Array | None = None,
      cloud_r_eff_ice: Array | None = None,
      cloud_path_ice: Array | None = None,
      cloud_fraction: Array | None = None,
  ) -> dict[str, Array]:
    """Compute the monochromatic shortwave optical properties.

    Uses the RRTM optics scheme to compute the shortwave optical depth, albedo,
    and asymmetry factor. These raw optical properties can be further
    transformed downstream to better suit the assumptions of the particular
    radiative transfer solver being used.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules / m^2].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].
      cloud_fraction: Cloud fraction per layer [0, 1] for partial-coverage
        scaling of cloud optical depth.

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The shortwave optical depth.
        'ssa': The shortwave single-scattering albedo.
        'asymmetry_factor': The shortwave asymmetry factor.
    """
    optical_depth_fn = self._optical_depth_fn(igpt, is_lw=False)
    optical_depth_sw = optical_depth_fn(
        molecules, temperature, pressure, vmr_fields
    )

    rayl_fn = self.rayleigh_scattering_fn(igpt)
    rayleigh_scattering = rayl_fn(molecules, temperature, pressure, vmr_fields)

    optical_depth_sw = optical_depth_sw + rayleigh_scattering

    ssa = jnp.where(
        optical_depth_sw > 0,
        rayleigh_scattering / optical_depth_sw,
        jnp.zeros_like(rayleigh_scattering),
    )

    gas_optical_props = {
        'optical_depth': optical_depth_sw,
        'ssa': ssa,
        'asymmetry_factor': jnp.zeros_like(ssa),
    }
    return self.add_cloud_optical_properties(
        igpt, gas_optical_props, False, cloud_r_eff_liq, cloud_path_liq,
        cloud_r_eff_ice, cloud_path_ice, cloud_fraction=cloud_fraction)

  @override
  def compute_planck_sources(
      self,
      pressure: Array,
      temperature: Array,
      igpt: Array,
      vmr_fields: dict[int, Array] | None = None,
      sfc_temperature: Array | None = None,
  ) -> dict[str, Array]:
    """Compute the monochromatic Planck sources given the atmospheric state.

    This requires interpolating the temperature to z faces.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      sfc_temperature: The optional surface temperature [K], 2D field.

    Returns:
      A dictionary containing the Planck source at the cell center
      (`planck_src`), the top cell boundary (`planck_src_top`), the bottom cell
      boundary (`planck_src_bottom`) and, if a `sfc_temperature` argument was
      provided, the surface cell boundary (`planck_src_sfc`). Note that the
      surface source will only be valid for the replicas in the first
      computational layer, as the local temperature field is used to compute it.
    """
    assert sfc_temperature is not None, 'sfc_temperature is required.'

    temperature_bottom, temperature_top = optics_base.reconstruct_face_values(
        temperature, f_lower_bc=sfc_temperature
    )
    planck_fraction_fn = self.planck_fraction_fn(igpt)
    planck_fraction = planck_fraction_fn(pressure, temperature, vmr_fields)

    planck_src_fn = self.planck_src_fn(igpt)
    planck_src = planck_src_fn(planck_fraction, temperature)
    planck_src_top = planck_src_fn(planck_fraction, temperature_top)
    planck_src_bottom = planck_src_fn(planck_fraction, temperature_bottom)

    planck_srcs = {
        'planck_src': planck_src,
        'planck_src_top': planck_src_top,
        'planck_src_bottom': planck_src_bottom,
    }

    if sfc_temperature is not None:
      # Extract the first interior node for surface calculations.
      planck_fraction_0 = planck_fraction[:, :, 1]

      planck_srcs['planck_src_sfc'] = planck_src_fn(
          planck_fraction_0, sfc_temperature
      )

    return planck_srcs

  @override
  @property
  def n_gpt_lw(self) -> int:
    """The number of g-points in the longwave bands."""
    self.gas_optics_lw = cast(LookupGasOpticsLongwave, self.gas_optics_lw)
    return self.gas_optics_lw.n_gpt

  @override
  @property
  def n_gpt_sw(self) -> int:
    """The number of g-points in the shortwave bands."""
    self.gas_optics_sw = cast(LookupGasOpticsShortwave, self.gas_optics_sw)
    return self.gas_optics_sw.n_gpt

  @override
  @property
  def solar_fraction_by_gpt(self) -> Array:
    """Mapping from g-point to the fraction of total solar radiation."""
    self.gas_optics_sw = cast(LookupGasOpticsShortwave, self.gas_optics_sw)
    return self.gas_optics_sw.solar_src_scaled


def optics_factory(
    params: radiative_transfer.OpticsParameters,
    vmr_lib: LookupVolumeMixingRatio | None = None,
    include_clouds: bool = True,
) -> optics_base.OpticsScheme:
  """Construct an instance of `OpticsScheme`.

  Args:
    params: The optics parameters.
    vmr_lib: An instance of `LookupVolumeMixingRatio` containing gas
      concentrations.
    include_clouds: When False, the constructed ``RRTMOptics`` will
      have ``cloud_optics_lw = cloud_optics_sw = None`` and skip the
      cloud-table load.  See ``RRTMOptics.__init__`` docstring for
      the safety argument (iter-40 memory optimisation).

  Returns:
    An instance of `OpticsScheme`.
  """
  if isinstance(params.optics, radiative_transfer.RRTMOptics):
    assert vmr_lib is not None, '`vmr_lib` is required for `RRTMOptics`.'
    return RRTMOptics(vmr_lib, params, include_clouds=include_clouds)
  # iter-61 removed the ``GrayAtmosphereOptics`` impl path.  That class
  # carried its own ``compute_planck_sources`` based on
  # Schneider 2004 / O'Gorman 2008 gray-atmosphere lapse-rate Planck —
  # NOT the RRTMGP correlated-k Planck source.  The RRTMGP Planck path
  # (``RRTMOptics.compute_planck_sources`` → ``gas_optics.planck_source``)
  # is untouched and remains the only Planck source legoESM uses.
  raise ValueError(
      f'Unsupported optics scheme: {type(params.optics).__name__!r}. '
      'Only RRTMOptics is supported in legoESM (iter-61 dropped the '
      'swirl_jatmos GrayAtmosphereOptics path; use '
      'legoesm.atmosphere.physics.radiation.gray for gray radiation).'
  )
