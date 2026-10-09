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

"""A library for solving the two-stream radiative transfer equation."""

from typing import TypeAlias, cast

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.radiation.rrtmgp import constants
from legoesm.atmosphere.physics.radiation.rrtmgp import kernel_ops
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import atmospheric_state
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_base
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics_base
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import monochromatic_two_stream

Array: TypeAlias = jax.Array
AbstractLookupGasOptics: TypeAlias = (
    lookup_gas_optics_base.AbstractLookupGasOptics
)
AtmosphericState: TypeAlias = atmospheric_state.AtmosphericState

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Spectral two-stream RTE solver (solve_lw/solve_sw): maps per-g-point "
        "gas/cloud/aerosol optics to Meador-Weaver reflectance/transmittance and "
        "integrates the vertical transport, summing over g-points to broadband "
        "up/down/net fluxes; compute_heating_rate converts flux divergence to K/s."
    ),
    "inputs": {
        "pressure": "Pa", "temperature": "K",
        "molecules": "molecules/m^2", "sfc_temperature": "K",
        "vmr_fields": "mol/mol (per-gas volume mixing ratio)",
        "cloud_path_liq": "kg/m^2", "cloud_path_ice": "kg/m^2",
        "cloud_r_eff_liq": "m", "cloud_r_eff_ice": "m",
        "cloud_fraction": "1", "aerosol_optical_depth": "1",
        "flux_net": "W/m^2", "dp": "Pa",
    },
    "outputs": {
        "flux_up": "W/m^2", "flux_down": "W/m^2", "flux_net": "W/m^2",
        "heating_rate": "K/s",
    },
    "sign_convention": (
        "fluxes positive in their named direction (flux_up upward, flux_down "
        "downward); flux_net = flux_up - flux_down (positive-up); heating_rate "
        "dT/dt>0 warms and is -g/c_p * dF_net/dp so net flux OUT of a layer "
        "cools it; optical depth tau>=0, 0<=ssa<=1, |g|<=1 (clipped); nighttime "
        "SW columns are zeroed. Per-layer reflectance+transmittance+absorptance=1 "
        "(energy-consistent), but photons leave at TOA/surface so the column "
        "budget is OPEN (accounted, not conserved)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Meador & Weaver (1980), J. Atmos. Sci. 37, 630-643; Pincus, Mlawer & "
        "Delamere (2019), doi:10.1029/2019MS001621 (RRTMGP two-stream)."
    ),
    "idealized_test": (
        "tests/unit/test_two_stream_top_flux.py: near-TOA flux extrapolation is "
        "range-limited (no super-physical TOA SW / negative OLR); differentiable "
        "in the continuous optics, integer g-point index selection static; "
        "checkpointed g-point scan for reverse-mode memory."
    ),
}


# Default aerosol optical properties (fixed scheme defaults).
_AEROSOL_SSA_DEFAULT = 0.93
_AEROSOL_ASYM_DEFAULT = 0.70

def _compute_local_properties_lw(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    igpt: Array,
    optics_lib: optics_base.OpticsScheme,
    vmr_fields: dict[int, Array] | None = None,
    sfc_temperature: Array | float | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    cloud_fraction: Array | None = None,
    lw_diffusive_factor: float | Array = monochromatic_two_stream._LW_DIFFUSIVE_FACTOR,
    precomputed_lw_optical_props: dict[str, Array] | None = None,
    precomputed_planck_srcs: dict[str, Array] | None = None,
) -> dict[str, Array]:
  """Compute local optical properties for longwave radiative transfer.

  ``precomputed_lw_optical_props`` lets the caller share an already-
  computed optics dict (e.g. the optimal-angle path in ``solve_lw``
  needs the optical depth to derive the per-column secant, and would
  otherwise repeat the table interpolation here).  When ``None`` the
  function calls ``optics_lib.compute_lw_optical_properties`` itself.
  XLA's CSE pass already deduplicates identical-input calls under
  JIT, but threading the dict through keeps the graph compact and
  makes the dependency explicit.
  """
  if isinstance(sfc_temperature, float):
    # Create a plane for the surface temperature representation.
    nx, ny, _ = temperature.shape
    sfc_temperature = sfc_temperature * jnp.ones(
        (nx, ny), dtype=temperature.dtype
    )

  if precomputed_lw_optical_props is not None:
    lw_optical_props = precomputed_lw_optical_props
  else:
    # Compute optical properties: `optical_depth`, `ssa`, & `asymmetry_factor`.
    lw_optical_props = optics_lib.compute_lw_optical_properties(
        pressure,
        temperature,
        molecules,
        igpt,
        vmr_fields,
        cloud_r_eff_liq,
        cloud_path_liq,
        cloud_r_eff_ice,
        cloud_path_ice,
        cloud_fraction=cloud_fraction,
    )

  # Compute Planck sources: `planck_src`, `planck_src_bottom`, `planck_src_top`,
  # and `planck_src_sfc`.
  if precomputed_planck_srcs is not None:
    planck_srcs = precomputed_planck_srcs
  else:
    planck_srcs = optics_lib.compute_planck_sources(
        pressure, temperature, igpt, vmr_fields, sfc_temperature=sfc_temperature
    )

  halo_width = 1
  sfc_src = planck_srcs.get(
      'planck_src_sfc', planck_srcs['planck_src_bottom'][:, :, halo_width]
  )

  # Compute combined Planck sources.  Output keys are `planck_src_bottom` and
  # `planck_src_top`.
  combined_srcs = monochromatic_two_stream.lw_combine_sources(planck_srcs)

  # Compute `t_diff`, `r_diff`, `src_up`, and `src_down`.
  src_and_properties = monochromatic_two_stream.lw_cell_source_and_properties(
      lw_optical_props['optical_depth'],
      lw_optical_props['ssa'],
      combined_srcs['planck_src_bottom'],
      combined_srcs['planck_src_top'],
      lw_optical_props['asymmetry_factor'],
      lw_diffusive_factor=lw_diffusive_factor,
  )
  src_and_properties['sfc_src'] = sfc_src

  return src_and_properties


def _reindex_vmr_fields(
    vmr_fields: dict[str, Array], gas_optics_lib: AbstractLookupGasOptics
) -> dict[int, Array]:
  """Converts the chemical formulas of the gas species to RRTM indices."""
  return {gas_optics_lib.idx_gases[k]: v for k, v in vmr_fields.items()}


def _compute_optimal_lw_secant(
    optical_depth: Array,
    band_idx: Array,
    optimal_angle_fit: Array,
    halo_width: int = 1,
) -> Array:
  """Compute upstream RRTMGP's optimal longwave diffusivity secant.

  Replicates ``rte-rrtmgp``'s ``compute_optimal_angles``: a per-band linear
  fit on the column transmissivity ``trans = exp(-sum_z tau)``::

      secant(col, gpt) = optimal_angle_fit[band, 0] * trans
                       + optimal_angle_fit[band, 1]

  Operates on the per-g-point optical depth slice ``(ncol, 1, nlev+2)``,
  excluding halo cells from the sum since halos carry linearly-extrapolated
  values that are stripped before the recurrent integration anyway.

  Args:
    optical_depth: ``(ncol, 1, nlev+2)`` per-g-point optical depth slice.
    band_idx: 0-D scalar with the spectral band corresponding to the
      current g-point (``g_point_to_bnd[igpt]``).
    optimal_angle_fit: ``(n_bnd, 2)`` polynomial-fit coefficients loaded
      from the longwave gas-optics file.
    halo_width: Vertical halo width to exclude from the column sum.

  Returns:
    ``(ncol, 1, 1)`` secant ready to broadcast against the
    ``(ncol, 1, nlev+2)`` optical-depth array.
  """
  # Interior column (halos excluded) total optical depth.
  hw = halo_width
  if hw > 0:
    tau_interior = optical_depth[:, :, hw:-hw]
  else:
    tau_interior = optical_depth
  # ``jnp.maximum(tau, 0.0)`` is a deliberate departure from the literal
  # upstream formula ``tau_total = sum(tau)`` (codex iter-2 review, LOW).
  # Upstream is invoked on freshly computed positive optical depths so the
  # difference is zero in practice; in legoESM the same array is reused
  # downstream after the recurrence strips halos, but during scan tracing
  # a halo cell whose interpolated tau briefly dipped below zero would
  # otherwise inject a negative term into ``trans_total`` and amplify
  # ``-secant`` errors.  Clamping to zero matches the physical meaning of
  # "no optical depth" and keeps ``exp(-tau_total) ∈ [0, 1]``.  Interior
  # cells (the ones that actually contribute) almost always have
  # ``tau >= 0`` from the kmajor/kminor lookup, so this clamp acts only
  # as a guard.
  tau_total = jnp.sum(jnp.maximum(tau_interior, 0.0), axis=-1, keepdims=True)
  # ``tau_total`` shape ``(ncol, 1, 1)``.  Compute column transmissivity.
  trans_total = jnp.exp(-tau_total)
  c0 = optimal_angle_fit[band_idx, 0]
  c1 = optimal_angle_fit[band_idx, 1]
  return c0 * trans_total + c1


def _gpoint_cloud_paths(igpt, cloud_path_liq, cloud_path_ice, cloud_path_fn):
  """Cloud water paths seen by g-point ``igpt``.

  ``cloud_path_fn(igpt) -> (liq, ice)`` supplies McICA paths (each g-point
  its own cloud subcolumn); without it every g-point sees the same paths.
  """
  if cloud_path_fn is None:
    return cloud_path_liq, cloud_path_ice
  return cloud_path_fn(igpt)


def _accumulate_over_gpoints(step_fn, n_gpt, init_val, gpoint_batch_size,
                             checkpoint=True):
  """Sum the per-g-point flux contributions produced by ``step_fn``.

  ``step_fn(igpt, cumulative) -> cumulative + flux(igpt)`` is the shared
  per-g-point map-reduce body of :func:`solve_lw` / :func:`solve_sw`.  The
  g-point axis is embarrassingly parallel; only the vertical recurrence
  *inside* ``step_fn`` is sequential.

  ``gpoint_batch_size <= 0`` (default) — the memory-frugal path: a
  ``jax.lax.scan`` over g-points with each step wrapped in
  ``jax.checkpoint(nothing_saveable, prevent_cse=True)`` so the backward pass
  recomputes one g-point at a time (≈1 MiB/col instead of ≈21 MiB/col, making
  ~1° training feasible on a 24 GiB GPU under reverse-mode AD).  This is the
  REQUIRED path for ``eqx.filter_value_and_grad`` at high resolution.

  ``gpoint_batch_size > 0`` — the throughput path for FORWARD / inference:
  process g-points in parallel blocks of ``gpoint_batch_size`` via ``jax.vmap``,
  scanning over the blocks to bound peak memory.  The sequential g-point scan
  launches one tiny kernel per g-point and starves the GPU (~26x slower in a
  GPU microbench); blocking restores parallelism.  The forward result differs
  from the scan path only by summation re-association (validated bit/ulp-close).
  Reverse-mode-AD safe when ``checkpoint`` (gpoint_checkpoint) is True: the
  block scan body is checkpointed (nothing_saveable), so backward recompute
  bounds peak memory to a single ``gpoint_batch_size`` block; with checkpoint
  False the scan is plain (forward/inference).
  """
  if gpoint_batch_size and gpoint_batch_size > 0:
    bs = int(gpoint_batch_size)
    # ``step_fn(ig, zeros)`` == flux(ig) since step_fn adds to the carry.
    zero = jax.tree.map(jnp.zeros_like, init_val)
    flux_fn = lambda ig: step_fn(ig, zero)
    n_blocks = -(-n_gpt // bs)  # ceil

    def block_step(carry, b):
      igpts = b * bs + jnp.arange(bs)
      valid = igpts < n_gpt
      # Clamp out-of-range padding indices to a valid g-point (0) so the
      # optics-table gathers stay in bounds, then mask their contribution.
      igpts_safe = jnp.where(valid, igpts, 0)
      block = jax.vmap(flux_fn)(igpts_safe)  # leaves: (bs, ...)
      block_sum = jax.tree.map(
          lambda f: jnp.sum(
              jnp.where(valid.reshape((bs,) + (1,) * (f.ndim - 1)), f, 0),
              axis=0,
          ),
          block,
      )
      new = jax.tree.map(lambda c, s: c + s.astype(c.dtype), carry, block_sum)
      return new, None

    # Checkpoint the block body for reverse-mode AD: without this the
    # uncheckpointed scan saves every block's activations, so backward memory
    # scales to ALL g-points (n_blocks x bs), not one block — the OOM the batch
    # path is meant to avoid.  ``block_step`` is ONE vmapped body (bs g-points),
    # so nothing_saveable recompute bounds backward memory to a single block
    # while the compile stays a single reused kernel (no prevent_cse Ng-fold
    # blow-up — that pathology is specific to the per-g-point scan path).
    block_step_ckpt = (
        jax.checkpoint(
            block_step,
            # prevent_cse=False (jax.checkpoint defaults it True): inside a scan
            # CSE-disabling is unnecessary and would emit a distinct body per
            # block; keep ONE reused compiled body. JAX recommends False here.
            prevent_cse=False,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        if checkpoint else block_step
    )
    fluxes, _ = jax.lax.scan(block_step_ckpt, init_val, jnp.arange(n_blocks))
    return fluxes

  # Memory-frugal checkpointed scan (training / default).
  def _scan_step(carry, igpt):
    return step_fn(igpt, carry), None

  if not checkpoint:
    # FORWARD path: a PLAIN scan (no jax.checkpoint, no prevent_cse).  The
    # checkpoint's prevent_cse=True is ONLY for reverse-mode AD memory; it
    # DISABLES common-subexpression elimination, which forces XLA to emit a
    # distinct compiled body per g-point instead of ONE reused scan-body
    # kernel.  That code blow-up is what overflows the XLA-CPU LLVM-JIT
    # executable code region (rrtmgp CPU "Failed to materialize symbols") and
    # bloats GPU/TPU compile.  A plain scan compiles ONE small body and is
    # answer-identical (no AD memory benefit, but a forward run needs none).
    fluxes, _ = jax.lax.scan(_scan_step, init_val, jnp.arange(n_gpt))
    return fluxes

  _scan_step_ckpt = jax.checkpoint(
      _scan_step,
      prevent_cse=True,
      policy=jax.checkpoint_policies.nothing_saveable,
  )
  fluxes, _ = jax.lax.scan(_scan_step_ckpt, init_val, jnp.arange(n_gpt))
  return fluxes


def solve_lw(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics_base.OpticsScheme,
    atmos_state: AtmosphericState,
    vmr_fields: dict[str, Array] | None = None,
    sfc_temperature: Array | float | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    cloud_fraction: Array | None = None,
    aerosol_absorption_optical_depth: Array | None = None,
    use_scan: bool | None = None,
    use_optimal_angle: bool = False,
    gpoint_batch_size: int = 0,
    gpoint_checkpoint: bool = True,
    cloud_path_fn=None,
    clear_sky: bool = False,
) -> dict[str, Array]:
  """Solves two-stream radiative transfer equation over the longwave spectrum.

  Local optical properties like optical depth, single-scattering albedo, and
  asymmetry factor are computed using an optics library and transformed to
  two-stream approximations of reflectance and transmittance. The sources of
  longwave radiation are the Planck sources, which are a function only of
  temperature. To obtain the cell-centered directional Planck sources, the
  sources are first computed at the cell boundaries and the net source
  emanating from the grid cell is determined. Each spectral interval,
  represented by a g-point, is a separate radiative transfer problem, and can
  be computed in parallel. Finally, the independently solved fluxes are summed
  over the full spectrum to yield the final upwelling and downwelling fluxes.

  Args:
    pressure: The pressure field [Pa].
    temperature: The temperature field [K].
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m²].
    optics_lib: An instance of an optics library.
    atmos_state: An instance containing the atmospheric state.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by the chemical formula.
    sfc_temperature: The optional surface temperature represented as either a 2D
      field or as a scalar [K].
    cloud_r_eff_liq: The effective radius of cloud droplets [m].
    cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
      [kg/m²].
    cloud_r_eff_ice: The effective radius of cloud ice particles [m].
    cloud_path_ice: The cloud ice water path in each atmospheric grid cell
      [kg/m²].
    aerosol_absorption_optical_depth: Optional prescribed longwave aerosol
      **absorption** optical depth per layer [-] (NOT extinction): it is
      added directly to the absorption optical depth with single-scattering
      albedo 0.  Longwave aerosol scattering is neglected (the dominant LW
      aerosol effect is absorption/emission); a caller holding extinction
      optical depth must pre-multiply by the LW absorption fraction
      (1 − ω) before passing it here.  ``None`` (the default) leaves the
      longwave solution byte-identical.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling longwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling longwave radiative flux at face i - 1/2.
      `flux_net`: The net longwave radiative flux at face i - 1/2.
  """
  optics_lib = cast(optics.RRTMOptics, optics_lib)
  if vmr_fields is not None:
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    vmr_fields = _reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_lw)

  # Resolve the optimal-angle table once, outside the scan body, so the
  # branch is selected at trace time and does not introduce a Python ``if``
  # on a traced value inside the scan.  Raise explicitly when the caller
  # requested ``use_optimal_angle=True`` but the gas-optics file does not
  # ship ``optimal_angle_fit`` — silently degrading to the fixed Fu-Liou
  # 1.66 contradicts the config contract documented on
  # ``RRTMGPConfig.use_optimal_angle`` (codex iter-2 review, MEDIUM).
  optimal_angle_fit = None
  if use_optimal_angle:
    candidate = None
    if hasattr(optics_lib, 'gas_optics_lw'):
      candidate = getattr(optics_lib.gas_optics_lw, 'optimal_angle_fit', None)
    if candidate is None:
      raise ValueError(
          "solve_lw(use_optimal_angle=True) requires the longwave "
          "gas-optics file to ship 'optimal_angle_fit' (added to "
          "rrtmgp-gas-lw-* in rte-rrtmgp >= 1.7).  Either upgrade the "
          "data file or set use_optimal_angle=False to keep the fixed "
          "Fu-Liou 1.66 diffusivity secant."
      )
    optimal_angle_fit = candidate

  def _lw_gpoint_fluxes(igpt, precomputed_props, liq_g, ice_g,
                        planck_srcs=None):
    """One g-point's LW fluxes from its optics (gas, or gas + cloud)."""
    if aerosol_absorption_optical_depth is not None:
      # Prescribed longwave aerosol as a pure-absorbing layer
      # (single-scattering albedo 0): add its absorption optical depth to
      # the background gas+cloud optical depth and dilute the combined ssa
      # accordingly.  The asymmetry factor of the (scattering) background
      # is unchanged because the aerosol contributes no scattering
      # (g_tot = tau_bg w_bg g_bg / (tau_tot w_tot) = g_bg).  Injected
      # into ``precomputed_props`` *before* the optimal-angle secant so
      # the per-band diffusivity sees the aerosol-inclusive transmissivity
      # (the secant is fit on exp(-sum tau)); the same dict then feeds the
      # source-and-properties solve, keeping both paths consistent.
      # ``safe_divide`` avoids the -a/b^2 VJP overflow at the tau floor
      # (same rationale as the shortwave aerosol mix in ``solve_sw``).
      tau_bg = jnp.maximum(precomputed_props['optical_depth'], 1.0e-12)
      tau_aer = jnp.maximum(aerosol_absorption_optical_depth, 0.0)
      tau_tot = tau_bg + tau_aer
      w_tot = jnp.clip(
          safe_divide(
              tau_bg * precomputed_props['ssa'], tau_tot, eps=1.0e-12, fill=0.0,
          ),
          0.0,
          1.0,
      )
      precomputed_props = {
          'optical_depth': tau_tot,
          'ssa': w_tot,
          'asymmetry_factor': precomputed_props['asymmetry_factor'],
      }
    # Bound LW ssa / asymmetry to physical ranges before the two-stream solve
    # (same BUG-B robustness fix as the shortwave path): with no LW aerosol
    # forcing the cloud optics feed through unclamped, and an ssa>1 from a
    # drifted-state cloud makes the LW two-stream emit a super-physical /
    # negative TOA OLR (rlut<0).  No-op for valid (in-range) optics.
    precomputed_props = {
        **precomputed_props,
        'ssa': jnp.clip(precomputed_props['ssa'], 0.0, 1.0),
        'asymmetry_factor': jnp.clip(
            precomputed_props['asymmetry_factor'], -1.0, 1.0,
        ),
    }
    if optimal_angle_fit is not None:
      band_idx = optics_lib.gas_optics_lw.g_point_to_bnd[igpt]
      lw_diffusive_factor = _compute_optimal_lw_secant(
          precomputed_props['optical_depth'], band_idx, optimal_angle_fit
      )
    else:
      lw_diffusive_factor = monochromatic_two_stream._LW_DIFFUSIVE_FACTOR

    optical_props_2stream = _compute_local_properties_lw(
        pressure,
        temperature,
        molecules,
        igpt,
        optics_lib,
        vmr_fields,
        sfc_temperature,
        cloud_r_eff_liq,
        liq_g,
        cloud_r_eff_ice,
        ice_g,
        cloud_fraction=cloud_fraction,
        lw_diffusive_factor=lw_diffusive_factor,
        precomputed_lw_optical_props=precomputed_props,
        precomputed_planck_srcs=planck_srcs,
    )

    # Boundary conditions.
    sfc_src = optical_props_2stream['sfc_src']
    toa_flux_down_lw = atmos_state.toa_flux_lw * jnp.ones_like(sfc_src)
    sfc_emissivity_lw = atmos_state.sfc_emis * jnp.ones_like(sfc_src)

    fluxes = monochromatic_two_stream.lw_transport(
        optical_props_2stream['t_diff'],
        optical_props_2stream['r_diff'],
        optical_props_2stream['src_up'],
        optical_props_2stream['src_down'],
        toa_flux_down_lw,
        sfc_src,
        sfc_emissivity_lw,
        use_scan,
    )
    return fluxes

  def step_fn(igpt, cumulative_flux):
    liq_g, ice_g = _gpoint_cloud_paths(
        igpt, cloud_path_liq, cloud_path_ice, cloud_path_fn)
    if not clear_sky:
      # Compute the LW optics once per g-point; reuse for both the
      # optimal-angle secant and the source-and-properties solve.
      precomputed_props = optics_lib.compute_lw_optical_properties(
          pressure, temperature, molecules, igpt, vmr_fields,
          cloud_r_eff_liq, liq_g,
          cloud_r_eff_ice, ice_g,
          cloud_fraction=cloud_fraction,
      )
      fluxes = _lw_gpoint_fluxes(igpt, precomputed_props, liq_g, ice_g)
    else:
      # Clear-sky fluxes alongside the all-sky ones: gas optics and Planck
      # sources are computed ONCE and shared (their table reads sit behind
      # optimization barriers, so XLA would not merge two separate calls).
      gas_props = optics_lib.compute_lw_optical_properties(
          pressure, temperature, molecules, igpt, vmr_fields)
      cloudy_props = optics_lib.add_cloud_optical_properties(
          igpt, gas_props, True, cloud_r_eff_liq, liq_g,
          cloud_r_eff_ice, ice_g, cloud_fraction=cloud_fraction)
      sfc_t = sfc_temperature
      if isinstance(sfc_t, float):
        sfc_t = sfc_t * jnp.ones(temperature.shape[:2], dtype=temperature.dtype)
      planck_srcs = optics_lib.compute_planck_sources(
          pressure, temperature, igpt, vmr_fields, sfc_temperature=sfc_t)
      fluxes = _lw_gpoint_fluxes(igpt, cloudy_props, liq_g, ice_g, planck_srcs)
      fluxes_clr = _lw_gpoint_fluxes(igpt, gas_props, None, None, planck_srcs)
      fluxes = {**fluxes, **{k + '_clr': v for k, v in fluxes_clr.items()}}
    # cumulative_flux keys: 'flux_up', 'flux_down', 'flux_net' (+ '_clr')
    # Coerce each g-point's flux to the ACCUMULATOR dtype before adding.
    # Under ``compute_fp32`` the scan carry (``cumulative_flux``, init
    # ``zeros_like(temperature)``) is float32, but the per-g-point transport
    # solve re-promotes to float64 via stray x64 constants in the RTE kernel,
    # so a bare ``jnp.add`` returns float64 and ``lax.scan`` rejects the
    # carry-in != carry-out dtype mismatch.  Casting the contribution to the
    # carry dtype keeps the accumulator at ``temperature``'s precision (float32
    # for fp32 runs, float64 otherwise -> byte-identical no-op on the default
    # path).
    return jax.tree.map(
        lambda _f, _c: _c + _f.astype(_c.dtype), fluxes, cumulative_flux,
    )

  flux_keys = ['flux_up', 'flux_down', 'flux_net']
  if clear_sky:
    flux_keys = flux_keys + [k + '_clr' for k in flux_keys]
  init_val = {key: jnp.zeros_like(temperature) for key in flux_keys}

  # Accumulate each g-point's flux contribution.  ``gpoint_batch_size==0``
  # keeps the memory-frugal checkpointed scan (training); ``>0`` uses chunked
  # ``vmap`` over the parallel g-point axis for forward throughput.  See
  # :func:`_accumulate_over_gpoints`.
  fluxes = _accumulate_over_gpoints(
      step_fn, optics_lib.n_gpt_lw, init_val, gpoint_batch_size,
      checkpoint=gpoint_checkpoint,
  )
  # Index -1 (the top halo position) carries the PHYSICAL boundary fluxes the
  # recurrence placed there (down = the TOA incident init, up = down*albedo +
  # emission at the same face); the top model layer is heated by their
  # divergence against index -2.  They must not be overwritten.
  return fluxes


def compute_sw_optical_props_gpt(
    igpt: Array,
    optics_lib,
    pressure: Array,
    temperature: Array,
    molecules: Array,
    vmr_fields: dict | None,
    cloud_r_eff_liq: Array | None,
    cloud_path_liq: Array | None,
    cloud_r_eff_ice: Array | None,
    cloud_path_ice: Array | None,
    cloud_fraction: Array | None,
    aerosol_optical_depth: Array | None,
    aerosol_single_scattering_albedo: float | Array,
    aerosol_asymmetry_factor: float | Array,
) -> dict[str, Array]:
  """Per-g-point shortwave optical properties (optical_depth / ssa / asymmetry).

  Shared by ``solve_sw`` (two-stream transport) and ``compute_sw_optical_field``
  (3D Monte-Carlo ray tracer) so the gas+cloud optics lookup, aerosol mixing,
  and physical-range clipping are defined exactly once (no duplicated numerics).
  ``vmr_fields`` must already be reindexed to RRTM gas indices by the caller.
  """
  sw_optical_props = optics_lib.compute_sw_optical_properties(
      pressure,
      temperature,
      molecules,
      igpt,
      vmr_fields,
      cloud_r_eff_liq,
      cloud_path_liq,
      cloud_r_eff_ice,
      cloud_path_ice,
      cloud_fraction=cloud_fraction,
  )
  return _mix_sw_aerosol_and_clip(
      igpt, optics_lib, sw_optical_props, aerosol_optical_depth,
      aerosol_single_scattering_albedo, aerosol_asymmetry_factor)


def _mix_sw_aerosol_and_clip(
    igpt: Array,
    optics_lib,
    sw_optical_props: dict[str, Array],
    aerosol_optical_depth: Array | None,
    aerosol_single_scattering_albedo: float | Array,
    aerosol_asymmetry_factor: float | Array,
) -> dict[str, Array]:
  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g."""
  if aerosol_optical_depth is not None:
    # Per-band aerosol optics: ``aerosol_single_scattering_albedo`` /
    # ``aerosol_asymmetry_factor`` may be a SCALAR (grey aerosol, historical
    # default -> byte-identical) OR a ``(n_bnd_sw,)`` array of per-shortwave-band
    # values (real aerosols absorb/scatter very differently in the UV/visible vs
    # the near-IR).  When per-band, select this g-point's band value via the
    # SAME g-point->band map the LW diffusivity path uses (g_point_to_bnd);
    # a 0-D input broadcasts unchanged.
    ssa_aer = jnp.asarray(aerosol_single_scattering_albedo)
    g_aer = jnp.asarray(aerosol_asymmetry_factor)
    if ssa_aer.ndim >= 1 or g_aer.ndim >= 1:
      band_idx = optics_lib.gas_optics_sw.g_point_to_bnd[igpt]
      if ssa_aer.ndim >= 1:
        ssa_aer = ssa_aer[band_idx]
      if g_aer.ndim >= 1:
        g_aer = g_aer[band_idx]
    tau_bg = jnp.maximum(sw_optical_props['optical_depth'], 1.0e-12)
    tau_aer = jnp.maximum(aerosol_optical_depth, 0.0)
    tau_tot = tau_bg + tau_aer
    w_bg = sw_optical_props['ssa']
    g_bg = sw_optical_props['asymmetry_factor']
    w_num = tau_bg * w_bg + tau_aer * ssa_aer
    # AD-safe SW optical-property mixing (restored from commit 59407953
    # after AIMIP-#312 merge reverted it).  ``a / jnp.maximum(b, eps)``
    # has a ``-a/b**2`` VJP that overflows when ``b`` is at the floor —
    # for cloud-free, low-water-vapor stratospheric layers ``tau_tot``
    # can reach the 1e-12 floor and the backward propagates NaN to every
    # upstream traced parameter whose state path touches gas absorption.
    # ``safe_divide`` masks the bad branch before the divide.  The outer
    # ``jnp.clip`` preserves the original output range; with ``fill=0.0``
    # the bad branch lands inside that range.
    w_tot = jnp.clip(
        safe_divide(w_num, tau_tot, eps=1.0e-12, fill=0.0),
        0.0,
        1.0,
    )
    g_num = (
        tau_bg * w_bg * g_bg
        + tau_aer * ssa_aer * g_aer
    )
    g_denom = tau_tot * jnp.maximum(w_tot, 1.0e-12)
    g_tot = jnp.clip(
        safe_divide(g_num, g_denom, eps=1.0e-12, fill=0.0),
        -1.0,
        1.0,
    )
    sw_optical_props = {
        'optical_depth': tau_tot,
        'ssa': w_tot,
        'asymmetry_factor': g_tot,
    }
  # Bound the single-scattering albedo / asymmetry to their physical ranges
  # before downstream use.  The aerosol-mixing branch above already clips
  # ssa∈[0,1] / g∈[-1,1], but with no aerosol forcing the cloud optics feed
  # through UNCLAMPED — and on a drifted coupled state the cloud
  # parameterisation can emit ssa>1 / |g|>1, which drives the Meador-Weaver
  # reflectance/transmittance above 1 so the two-stream AMPLIFIES the flux
  # (super-physical TOA SW down; BUG-B, 2026-06-15).  Physical optics are
  # already in range ⇒ a no-op for valid inputs.
  sw_optical_props = {
      **sw_optical_props,
      'ssa': jnp.clip(sw_optical_props['ssa'], 0.0, 1.0),
      'asymmetry_factor': jnp.clip(
          sw_optical_props['asymmetry_factor'], -1.0, 1.0,
      ),
  }
  return sw_optical_props


def compute_sw_optical_field(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics_base.OpticsScheme,
    vmr_fields: dict[str, Array] | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    cloud_fraction: Array | None = None,
    aerosol_optical_depth: Array | None = None,
    aerosol_single_scattering_albedo: float | Array = _AEROSOL_SSA_DEFAULT,
    aerosol_asymmetry_factor: float | Array = _AEROSOL_ASYM_DEFAULT,
) -> dict[str, Array]:
  """Stack per-g-point shortwave optical fields for the 3D MC ray tracer.

  Runs the SAME per-g-point optics path as ``solve_sw`` (via
  ``compute_sw_optical_props_gpt``) but, instead of solving two-stream
  transport, stacks the optical depth / ssa / asymmetry over all shortwave
  g-points. Returns a dict with keys ``optical_depth`` / ``ssa`` /
  ``asymmetry_factor``, each shaped ``(n_gpt_sw, X, Y, Z)`` (same X,Y,Z as the
  input fields, including any vertical halo cells — the caller strips them).
  """
  optics_lib = cast(optics.RRTMOptics, optics_lib)
  if vmr_fields is not None:
    vmr_fields = _reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_sw)
  n_gpt = optics_lib.n_gpt_sw

  def gpt_props(igpt):
    props = compute_sw_optical_props_gpt(
        igpt, optics_lib, pressure, temperature, molecules, vmr_fields,
        cloud_r_eff_liq, cloud_path_liq, cloud_r_eff_ice, cloud_path_ice,
        cloud_fraction, aerosol_optical_depth,
        aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
    )
    # Gas Rayleigh scattering optical depth (the gas scatter is ALL Rayleigh) as
    # a fraction of TOTAL scattering (gas + cloud + aerosol) -> the per-cell
    # probability that a scatter event is Rayleigh, for the explicit-phase split.
    rayl_scat = optics_lib.rayleigh_scattering_fn(igpt)(
        molecules, temperature, pressure, vmr_fields)
    tot_scat = props['ssa'] * props['optical_depth']
    # safe_divide (not a floored raw divide) so the masked-branch VJP can't
    # overflow as tot_scat -> 0, consistent with ssa/g above. fill=1.0: a cell
    # with negligible total scattering is gas-only -> any scatter there is
    # Rayleigh (matches the old clip-to-1 limit, not 0).
    rayleigh_frac = jnp.clip(
        safe_divide(rayl_scat, tot_scat, eps=1.0e-30, fill=1.0), 0.0, 1.0)
    return {**props, 'rayleigh_frac': rayleigh_frac}

  # Sequential map over g-points (memory-frugal, like the solve_sw scan):
  # stacks leaves to (n_gpt, X, Y, Z) without materializing all g-points'
  # table lookups concurrently.
  return jax.lax.map(gpt_props, jnp.arange(n_gpt))


def compute_lw_optical_field(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics_base.OpticsScheme,
    sfc_temperature: Array,
    vmr_fields: dict[str, Array] | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    cloud_fraction: Array | None = None,
    aerosol_absorption_optical_depth: Array | None = None,
) -> dict[str, Array]:
  """Stack per-g-point longwave ABSORPTION optical depth + Planck sources for
  the 3D Monte-Carlo emission tracer.

  Returns a dict with leaves stacked over the longwave g-points:
  ``abs_optical_depth`` ``(n_gpt, X, Y, Z)`` = ``optical_depth * (1 - ssa)``
  (LW scattering neglected by the emission MC), ``planck_src`` ``(n_gpt, X,Y,Z)``
  cell-center Planck IRRADIANCE [W/m^2] (= pi * radiance), and
  ``planck_src_sfc`` ``(n_gpt, X, Y)`` surface Planck irradiance. Same per-g-point
  optics path as ``solve_lw`` (no duplicated numerics).
  """
  optics_lib = cast(optics.RRTMOptics, optics_lib)
  if vmr_fields is not None:
    vmr_fields = _reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_lw)
  n_gpt = optics_lib.n_gpt_lw

  def gpt_field(igpt):
    props = optics_lib.compute_lw_optical_properties(
        pressure, temperature, molecules, igpt, vmr_fields,
        cloud_r_eff_liq, cloud_path_liq, cloud_r_eff_ice, cloud_path_ice,
        cloud_fraction=cloud_fraction)
    planck = optics_lib.compute_planck_sources(
        pressure, temperature, igpt, vmr_fields,
        sfc_temperature=sfc_temperature)
    ssa = jnp.clip(props['ssa'], 0.0, 1.0)
    abs_od = jnp.maximum(props['optical_depth'], 0.0) * (1.0 - ssa)
    if aerosol_absorption_optical_depth is not None:
      # LW aerosol is a pure absorber (ssa=0): add its absorption OD directly,
      # matching the solve_lw aerosol treatment.
      abs_od = abs_od + jnp.maximum(aerosol_absorption_optical_depth, 0.0)
    return {
        'abs_optical_depth': abs_od,
        'planck_src': planck['planck_src'],
        'planck_src_bottom': planck['planck_src_bottom'],
        'planck_src_top': planck['planck_src_top'],
        'planck_src_sfc': planck['planck_src_sfc'],
    }

  return jax.lax.map(gpt_field, jnp.arange(n_gpt))


def solve_sw(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics_base.OpticsScheme,
    atmos_state: AtmosphericState,
    vmr_fields: dict[str, Array] | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    cloud_fraction: Array | None = None,
    aerosol_optical_depth: Array | None = None,
    aerosol_single_scattering_albedo: float | Array = _AEROSOL_SSA_DEFAULT,
    aerosol_asymmetry_factor: float | Array = _AEROSOL_ASYM_DEFAULT,
    solar_fraction_by_gpt: Array | None = None,
    use_scan: bool | None = None,
    gpoint_batch_size: int = 0,
    gpoint_checkpoint: bool = True,
    cloud_path_fn=None,
    clear_sky: bool = False,
) -> dict[str, Array]:
  """Solves the two-stream radiative transfer equation for shortwave.

  Local optical properties like optical depth, single-scattering albedo, and
  asymmetry factor are computed using an optics library and transformed to
  two-stream approximations of reflectance and transmittance. The sources of
  shortwave radiation are determined by the diffuse propagation of direct
  solar radiation through the layered atmosphere. Each spectral interval,
  represented by a g-point, is a separate radiative transfer problem, and can
  be computed in parallel. Finally, the independently solved fluxes are summed
  over the full spectrum to yield the final upwelling and downwelling fluxes.

  Args:
    pressure: The pressure field [Pa].
    temperature: The temperature field [K].
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m²].
    optics_lib: An instance of an optics library.
    atmos_state: An instance containing the atmospheric state.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index.
    cloud_r_eff_liq: The effective radius of cloud droplets [m].
    cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
      [kg/m²].
    cloud_r_eff_ice: The effective radius of cloud ice particles [m].
    cloud_path_ice: The cloud ice water path in each atmospheric grid cell
      [kg/m²].
    aerosol_optical_depth: Optional aerosol optical depth per layer (same
      shape as `temperature`), added to SW extinction.
    aerosol_single_scattering_albedo: Bulk aerosol single-scattering albedo.
    aerosol_asymmetry_factor: Bulk aerosol asymmetry factor.
    solar_fraction_by_gpt: Optional external spectral solar weights by g-point.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling shortwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling shortwave radiative flux at face i - 1/2.
      `flux_net`: The net shortwave radiative flux at face i - 1/2.
  """
  zenith = atmos_state.zenith
  optics_lib = cast(optics.RRTMOptics, optics_lib)
  if vmr_fields is not None:
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    vmr_fields = _reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_sw)

  # --- Per-column day/night handling ---
  # ``zenith`` may be a scalar (single-column) or an array with a column
  # dimension, e.g. shape ``(ncol, 1)``.  To avoid division-by-zero in
  # ``exp(-tau / cos(zenith))`` for nighttime columns (cos(zenith) ~ 0),
  # we clamp the zenith used in the solve to at most ~89.4 degrees
  # (cos > 0.01).  After the solve, nighttime columns are zeroed out.
  _ZENITH_MAX = jnp.arccos(jnp.asarray(0.01, dtype=temperature.dtype))  # coeff-ok: min cos(zenith) floor
  safe_zenith = jnp.minimum(zenith, _ZENITH_MAX)

  # Build a per-column boolean mask that is True for daytime columns.
  # Works for both scalar zenith and array zenith.
  is_day_col = zenith < 0.5 * jnp.pi  # shape () or (ncol, 1)

  # Check whether *any* column is illuminated to short-circuit a global
  # nighttime domain (preserves the original optimisation).
  any_day = jnp.any(is_day_col)

  def _sw_gpoint_fluxes(igpt, sw_optical_props):
    """One g-point's SW fluxes from its final (aerosol-mixed) optics."""
    optical_props_2stream = monochromatic_two_stream.sw_cell_properties(
        safe_zenith,
        sw_optical_props['optical_depth'],
        sw_optical_props['ssa'],
        sw_optical_props['asymmetry_factor'],
    )

    # Surface albedo: broadcast per-column array or scalar to 2D plane
    # with the same horizontal sharding as the temperature. The DIFFUSE
    # reflection uses sfc_alb; the DIRECT beam uses sfc_alb_dir when set
    # (RAD-3 SAM direct/diffuse split), else falls back to sfc_alb.
    sfc_albedo = atmos_state.sfc_alb * jnp.ones_like(temperature[:, :, 0])
    _alb_dir_src = (
        atmos_state.sfc_alb if atmos_state.sfc_alb_dir is None
        else atmos_state.sfc_alb_dir
    )
    sfc_albedo_dir = _alb_dir_src * jnp.ones_like(temperature[:, :, 0])

    # Monochromatic top of atmosphere flux.
    if solar_fraction_by_gpt is None:
      spectral_weight = optics_lib.solar_fraction_by_gpt[igpt]
    else:
      spectral_weight = solar_fraction_by_gpt[igpt]
    solar_flux = atmos_state.irrad * spectral_weight
    toa_flux = solar_flux * jnp.ones_like(temperature[:, :, 0])

    sources_2stream = monochromatic_two_stream.sw_cell_source(
        t_dir=optical_props_2stream['t_dir'],
        r_dir=optical_props_2stream['r_dir'],
        optical_depth=sw_optical_props['optical_depth'],
        toa_flux=toa_flux,
        sfc_albedo_direct=sfc_albedo_dir,
        zenith=safe_zenith,
        use_scan=use_scan,
    )

    sw_fluxes = monochromatic_two_stream.sw_transport(
        t_diff=optical_props_2stream['t_diff'],
        r_diff=optical_props_2stream['r_diff'],
        src_up=sources_2stream['src_up'],
        src_down=sources_2stream['src_down'],
        sfc_src=sources_2stream['sfc_src'],
        sfc_albedo=sfc_albedo,
        flux_down_dir=sources_2stream['flux_down_dir'],
        use_scan=use_scan,
    )
    return sw_fluxes

  def step_fn(igpt, partial_fluxes):
    liq_g, ice_g = _gpoint_cloud_paths(
        igpt, cloud_path_liq, cloud_path_ice, cloud_path_fn)
    if not clear_sky:
      # Per-g-point gas+cloud+aerosol optics (shared with the MC ray tracer).
      sw_optical_props = compute_sw_optical_props_gpt(
          igpt, optics_lib, pressure, temperature, molecules, vmr_fields,
          cloud_r_eff_liq, liq_g, cloud_r_eff_ice, ice_g,
          cloud_fraction, aerosol_optical_depth,
          aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
      )
      sw_fluxes = _sw_gpoint_fluxes(igpt, sw_optical_props)
    else:
      # Clear-sky alongside all-sky with the gas optics computed ONCE (see
      # solve_lw); aerosol is mixed into each, as the separate passes did.
      gas_props = optics_lib.compute_sw_optical_properties(
          pressure, temperature, molecules, igpt, vmr_fields)
      cloudy_props = optics_lib.add_cloud_optical_properties(
          igpt, gas_props, False, cloud_r_eff_liq, liq_g,
          cloud_r_eff_ice, ice_g, cloud_fraction=cloud_fraction)
      _aer = (aerosol_optical_depth, aerosol_single_scattering_albedo,
              aerosol_asymmetry_factor)
      sw_fluxes = _sw_gpoint_fluxes(
          igpt, _mix_sw_aerosol_and_clip(igpt, optics_lib, cloudy_props, *_aer))
      sw_fluxes_clr = _sw_gpoint_fluxes(
          igpt, _mix_sw_aerosol_and_clip(igpt, optics_lib, gas_props, *_aer))
      sw_fluxes = {**sw_fluxes,
                   **{k + '_clr': v for k, v in sw_fluxes_clr.items()}}
    # Cast each g-point contribution to the accumulator dtype (see solve_lw):
    # under ``compute_fp32`` the carry (``partial_fluxes``, init
    # ``zeros_like(temperature)``) is float32 but the transport solve
    # re-promotes to float64, so a bare ``jnp.add`` would break the scan carry
    # dtype invariant.  No-op on the default float64 path.
    total_sw_fluxes = jax.tree.map(
        lambda _f, _c: _c + _f.astype(_c.dtype), sw_fluxes, partial_fluxes,
    )
    return total_sw_fluxes

  flux_keys = ['flux_up', 'flux_down', 'flux_net']
  if clear_sky:
    flux_keys = flux_keys + [k + '_clr' for k in flux_keys]
  fluxes_0 = {key: jnp.zeros_like(temperature) for key in flux_keys}

  def _compute_fluxes(_):
    # Accumulate per-g-point flux: checkpointed scan when
    # ``gpoint_batch_size==0`` (training), chunked ``vmap`` over the parallel
    # g-point axis when ``>0`` (forward throughput).  See
    # :func:`_accumulate_over_gpoints` and the solve_lw companion.
    fluxes = _accumulate_over_gpoints(
        step_fn, optics_lib.n_gpt_sw, fluxes_0, gpoint_batch_size,
        checkpoint=gpoint_checkpoint,
    )
    # Index -1 holds the physical TOA fluxes from the recurrence (see solve_lw);
    # an earlier clipped extrapolation here zeroed the top layer's SW heating.

    # Zero out nighttime columns.  ``is_day_col`` broadcasts from
    # shape ``()`` or ``(ncol, 1)`` against ``(ncol, 1, nlev+2)``.
    day_mask_3d = jnp.asarray(is_day_col, dtype=temperature.dtype)
    for key in flux_keys:
      fluxes[key] = fluxes[key] * day_mask_3d
    return fluxes

  # Short-circuit: if the entire domain is nighttime, skip the solve.
  return jax.lax.cond(
      any_day,
      _compute_fluxes,
      lambda _: fluxes_0,
      operand=None,
  )


def compute_heating_rate(
    flux_net: Array,
    pressure: Array,
    dp: Array | None = None,
) -> Array:
  """Computes cell-center heating rate from pressure and net radiative flux.

  The net radiative flux corresponds to the bottom cell face. The difference
  of the net flux at the top face and that at the bottom face gives the total
  net flux out of the grid cell. Using the pressure difference across the grid
  cell, the net flux can be converted to a heating rate, in K/s.

  Args:
    flux_net: The net flux at the bottom face [W/m²].
    pressure: The pressure field [Pa].
    dp: Exact layer pressure thickness [Pa].  When provided, used directly
        instead of the centered-difference approximation from ``pressure``.

  Returns:
    The heating rate of the grid cell [K/s].
  """
  if dp is None:
      # Fallback: centered-difference approximation.
      dp = 0.5 * kernel_ops.centered_difference(pressure, dim=2)

  # Compute the forward pressure difference of fluxes on faces (like a
  # derivative of face_to_node).  This is the net upward flux out of the
  # cell; a positive value means the cell radiates away energy and cools.
  dflux = kernel_ops.forward_difference(flux_net, dim=2)

  # Heating rate at the grid cell center [K/s].  **Minus sign** (restored
  # from commit 0be22f0f after the AIMIP-#312 merge reverted it): net
  # flux *out* cools the cell.  ``abs(dp)`` so the sign is set by the
  # flux divergence alone, not by the vertical-axis orientation
  # (``solve_columns`` passes positive layer thickness; the legacy
  # ``RRTMGP.compute_heating_rate`` callpath passes a centered-difference
  # negative ``dp`` that abs() canonicalises).
  #
  # Pre-fix bug symptom: free-tropospheric LW heating was +2..+5 K/day
  # (radiative warming) instead of −1..−2 K/day (radiative cooling),
  # driving thermal runaway in long AMIP integrations (T̄ 261 → 293 K
  # over 120 days, NaN blowup at day 125).
  return -constants.G * dflux / jnp.abs(dp) / constants.CP_D
