"""#1226 online box heat-budget accumulator.

Instrument for the ACC-deficit causal chain: identifies WHICH tendency term
carries the +38.5 W/m^2 excess heat convergence into legoESM's DINO southern
channel box, by accumulating the box-integrated per-term heat tendency
DURING a multi-year run (a single-snapshot probe cannot resolve this — see
``tracer_tendency_compare.py``'s per-term buckets, which this module reuses
at daily cadence instead of one twin step).

Terms (mirrors ``tracer_tendency_compare.py``'s bucket split exactly, reusing
the SAME production functions — no re-derived numerics):
  ADV_H   — horizontal advection (incl. GM-bolus-through-FCT fold, this
            recipe's ``gm_bolus_advection="through_fct"``)
  ADV_V   — vertical advection
  ISO_REDI — GM/Redi tracer tendency (``gm_redi_tracer_tendency_latlon``,
            the same call ``tendency_probe.probe_latlon_cgrid`` makes,
            evaluated on the state's own T/S — the online run has no NEMO
            before-level twin, unlike ``tracer_tendency_compare.py``)
  FORCING — surface restoring + Jerlov SW penetration (the SAME functions
            ``apply_dino_lat_lon_surface_forcing`` calls)
  VERTMIX — NOT computed directly (the production vertical-mixing
            diffusivity comes from the model's own prognostic TKE closure,
            which lives in the run-loop carry, not on a bare
            ``LatLonCGridOceanState`` — recomputing it here would duplicate
            the TKE closure's numerics). Instead VERTMIX is the residual
            TOTAL - (ADV_H + ADV_V + ISO_REDI + FORCING), where TOTAL is the
            actual box heat-content tendency computed from consecutive
            state samples. This is EXACT by construction (closes the
            budget trivially for the box's total) and isolates the sampling
            error in the four directly-computed terms into one number
            instead of hiding it inside a re-derived vertmix estimate.

Sampling: intended for a daily cadence inside a multi-year run loop (the
caller decides N steps/sample). Diagnostics only — never mutates the
model state fed to ``model.step()``.

CAVEAT (endpoint-rate sampling, stated not hidden): each interval's ADV_H/
ADV_V/ISO_REDI/FORCING rate is evaluated ONCE, at the state CLOSING that
interval (not a midpoint or trapezoidal average), then multiplied by the
whole interval length. For a slowly-varying (multi-year-mean) signal this
is negligible; for the SEASONAL cycle DINO's forcing runs (nn_ann_cyc) a
daily endpoint sample carries a same-order-as-the-day's-drift bias
(bounded by day-length / forcing-timescale, ~0.2% of tau_T=11.85 days for
DINO — small, but real). Because VERTMIX is the residual, this sampling
error (along with the true vertical mixing and any intra-day
nonlinearity) all land in the VERTMIX bucket — a signal appearing in
VERTMIX is "vertmix + all fourth-term sampling/endpoint error," not proof
of vertical-mixing causation on its own. Only the four EXPLICIT terms
(ADV_H, ADV_V, ISO_REDI, FORCING) are directly attributable; use VERTMIX
as "everything else," consistent with its residual definition above.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
    divergence_cgrid,
    min_cell_to_uface,
    min_cell_to_vface,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _add_bolus_to_advecting_flux,
    _compute_advection_flux_div_pair,
    _static_kappa_redi_override,
)
from legoesm.ocean.experiments.dino import (
    dino_Q_sr_seasonal,
    dino_T_star_seasonal,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    gm_redi_tracer_tendency_latlon,
)
from legoesm.ocean.physics.shortwave_penetration import (
    ShortwavePenetrationConfig,
    shortwave_penetration_tendency,
)
from legoesm.ocean.physics.surface_forcing.config import (
    RestoringConfig,
    tau_from_flux_coefficient,
)
from legoesm.ocean.physics.surface_forcing.restoring import (
    restoring_surface_forcing,
)
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate,
    compute_layer_thickness,
    diagnose_w_from_flux_div,
)

TERM_NAMES = ("adv_h", "adv_v", "iso_redi", "forcing", "vertmix")


class _GridShim:
    """Matches ``tracer_tendency_compare.py``'s shim for
    ``restoring_surface_forcing`` (only ``grid_lat`` shape is read)."""

    def __init__(self, cell_mask):
        self.grid_lat = jnp.zeros_like(cell_mask)


def compute_box_heat_dT_terms(state, grid, z_coord, config, dino_cfg, forcing,
                              dt_model: float, t_seconds: float):
    """Per-cell dT/dt [degC/s] for ADV_H, ADV_V, ISO_REDI, FORCING.

    Reuses the exact production functions (``_compute_advection_flux_div_pair``,
    ``_add_bolus_to_advecting_flux``, ``gm_redi_tracer_tendency_latlon``,
    ``restoring_surface_forcing``, ``shortwave_penetration_tendency``) — same
    call pattern as ``tracer_tendency_compare.py`` but on the state's OWN
    T/S/eta (no NEMO before-level bridge; this is a lego-only online run).

    ``dt_model`` MUST be the model's own dynamical timestep (``DT``, e.g.
    2700 s for ``nemo_dino_kamm_mlf``), NOT the accumulator's (much longer)
    sampling interval — every one of these functions is ``dt``-SENSITIVE in
    its own right (physics-validator review, #1226):
    ``restoring_surface_forcing(implicit=True)`` uses ``eff_tau = tau + dt``
    (a 32x-too-large ``dt`` biases the restoring rate low by
    ``dt_step/(tau+dt_step)``, ~8% at daily sampling with DINO's
    ``tau_T``); ``_compute_advection_flux_div_pair``'s FCT limiter clips the
    antidiffusive flux against a low-order provisional update scaled by
    ``dt`` (wrong ``dt`` -> wrong limiter bound -> wrong flux, NOT just a
    scaling error); ``gm_redi_tracer_tendency_latlon``'s MSC
    (``ln_traldf_msc``) stability clamp on the vertical diagonal is also
    ``dt``-set. Passing the sampling interval here would silently corrupt
    all three rates, not merely rescale them — the caller (``sample()``)
    keeps the sampling interval OUT of this call and applies it only as the
    Joule-integration weight afterward.

    Returns
    -------
    dict with keys "adv_h", "adv_v", "iso_redi", "forcing" -> (n_lat, n_lon, nlev)
    """
    mask = state.land_mask.data
    h_k = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    h_u = min_cell_to_uface(h_k)
    h_v = min_cell_to_vface(h_k, grid)
    if isinstance(z_coord, OceanPartialCellCoordinate):
        u_mask_3d, v_mask_3d = compute_face_masks_3d(z_coord.is_active, grid)
        active_3d = z_coord.is_active.astype(h_k.dtype)
    else:
        u_mask_3d = state.u_mask.data[..., jnp.newaxis]
        v_mask_3d = state.v_mask.data[..., jnp.newaxis]
        active_3d = mask[:, :, jnp.newaxis] * jnp.ones_like(h_k)
    mass_flux_u = h_u * state.u.data * u_mask_3d
    mass_flux_v = h_v * state.v.data * v_mask_3d

    # Base (momentum/continuity) vertical transport — same call production
    # uses (ocean_model_latlon_cgrid.py:3475-3478) — needed regardless of
    # GM/Redi so the no-GM/Redi branch still has a valid w_baro.
    flux_div_k = divergence_cgrid(mass_flux_u, mass_flux_v, grid)
    w_baro = diagnose_w_from_flux_div(flux_div_k, z_coord, thickness_weighted=True)

    if getattr(config, "gm_redi", None) is not None:
        kappa_redi_ov = _static_kappa_redi_override(config.gm_redi, grid)
        # GM-bolus-through-FCT fold (tracer_tendency_compare.py): ONLY
        # exported/folded into the advecting flux when the recipe sets
        # gm_bolus_advection="through_fct" (nemo_dino_kamm_mlf). The other
        # supported mode, "centred", applies the bolus IN-OPERATOR inside
        # dT_gm directly (gm_redi_latlon_cgrid.py:1734-1736) — folding it
        # into advection too would double-count, and
        # return_bolus_transport=True is REJECTED by
        # gm_redi_tracer_tendency_latlon for any mode but through_fct.
        through_fct = getattr(config.gm_redi, "gm_bolus_advection", "centred") == "through_fct"
        gm_out = gm_redi_tracer_tendency_latlon(
            state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
            grid, z_coord, config.gm_redi,
            eos=config.eos, eos_linear=config.eos_linear,
            mask=mask, u_mask=state.u_mask.data, v_mask=state.v_mask.data,
            rho_0=config.constants.rho_0, g=config.constants.g,
            kappa_redi_override=kappa_redi_ov,
            return_bolus_transport=through_fct,
            dt=dt_model,
        )
        if through_fct:
            dT_gm, dS_gm, bolus = gm_out
            mass_flux_u, mass_flux_v, w_baro = _add_bolus_to_advecting_flux(
                bolus, mass_flux_u, mass_flux_v, u_mask_3d, v_mask_3d, grid, z_coord,
            )
        else:
            dT_gm, dS_gm = gm_out
    else:
        dT_gm = jnp.zeros_like(state.T.data)

    wall_fill_mask = active_3d if getattr(config, "tracer_wall_neumann_fill", True) else None
    (dh_T, dv_T), _ = _compute_advection_flux_div_pair(
        state.T.data, state.S.data, config.tracer_advection,
        mass_flux_u, mass_flux_v, w_baro, h_k, h_u, h_v, grid, dt_model,
        recon_fill_mask=wall_fill_mask,
        linssh_top_flux=getattr(z_coord, "linear_free_surface", False),
    )
    h_safe = jnp.maximum(h_k, 1e-10)
    dT_adv_h = -dh_T / h_safe * mask[:, :, jnp.newaxis]
    dT_adv_v = -dv_T / h_safe * mask[:, :, jnp.newaxis]

    dz_0 = float(z_coord.dz_ref[0])
    lat1 = forcing["lat_deg_1d"]
    if getattr(dino_cfg, "forcing_annual_cycle", False):
        T_star_2d = jnp.broadcast_to(
            dino_T_star_seasonal(lat1, t_seconds, dino_cfg)[:, None], mask.shape)
        Q_sr_2d = jnp.broadcast_to(
            dino_Q_sr_seasonal(lat1, t_seconds, dino_cfg)[:, None], mask.shape)
    else:
        T_star_2d = forcing["T_star_2d"]
        Q_sr_2d = forcing["Q_sr_2d"]
    tau_T = tau_from_flux_coefficient(dino_cfg.A_theta, dino_cfg.rho_0, dino_cfg.c_p, dz_0)
    tau_S = tau_from_flux_coefficient(dino_cfg.A_S, dino_cfg.rho_0, 1.0, dz_0)
    restoring_cfg = RestoringConfig(
        tau_T=tau_T, tau_S=tau_S, T_star_array=T_star_2d,
        S_star_array=forcing["S_star_2d"], subtract_qsr=True, implicit=True,
    )
    rest_out = restoring_surface_forcing(
        state.T.data, state.S.data, _GridShim(mask), restoring_cfg,
        sw_down=Q_sr_2d, dt=dt_model, rho_0=dino_cfg.rho_0, c_p=dino_cfg.c_p, dz_0=dz_0,
    )
    sw_cfg = ShortwavePenetrationConfig(water_type=dino_cfg.jerlov_water_type)
    dT_dt_sw = shortwave_penetration_tendency(
        sw_down=Q_sr_2d, z_coord_dz_ref=z_coord.dz_ref,
        z_coord_z_half_ref=z_coord.z_half_ref,
        jacobian=jnp.ones_like(state.eta.data), config=sw_cfg,
        rho_0=dino_cfg.rho_0, c_sw=dino_cfg.c_p,
    )
    dT_forcing = (rest_out.dT_dt + dT_dt_sw) * mask[:, :, jnp.newaxis]

    return {
        "adv_h": dT_adv_h,
        "adv_v": dT_adv_v,
        "iso_redi": dT_gm,
        "forcing": dT_forcing,
    }


class BoxHeatBudgetAccumulator:
    """Accumulates box-integrated per-term heat tendency [W] by depth band
    over a multi-year run, sampled at whatever cadence the caller calls
    ``.sample()`` (e.g. daily).

    Parameters
    ----------
    grid, z_coord, config, dino_cfg, forcing
        Same objects the run loop already built (``dino_lat_lon_model_config``,
        ``dino_config_for_recipe``, ``dino_lat_lon_surface_forcing_arrays``).
    dt_model : float
        The MODEL's own dynamical timestep [s] (e.g. 2700 for
        ``nemo_dino_kamm_mlf``) — passed to the physics operators
        (``restoring_surface_forcing``, the FCT advection limiter,
        ``gm_redi_tracer_tendency_latlon``'s MSC clamp) inside
        ``compute_box_heat_dT_terms``. This is DELIBERATELY separate from
        the (much longer) sampling interval passed to ``sample()`` —
        every one of those operators is dt-sensitive in a way that does
        NOT simply rescale with dt (see ``compute_box_heat_dT_terms``
        docstring), so feeding the sampling interval in as the physics dt
        would silently bias every explicit term (#1226 physics-validator
        review finding).
    row_slice : slice
        Box definition on the (n_lat,) axis — ``slice(12, 47)``, the
        "channel rows" convention shared by every #1226/#1317 instrument
        (``momentum_budget_diff.py``, ``budget_{pointwise,fullframe}.py``,
        ``tracer_tendency_compare.py``).
    depth_bands_m : tuple of (lo, hi) pairs
        Depth bands [m], e.g. ``((0, 200), (200, 1000), (1000, None))``.
    rho0, cp : float
        legoESM native convention (``constants.rho_ocean``, ``constants.c_sw``)
        by default — pass NEMO's (1026, 3991.86795711963) for an
        apples-to-apples comparison against the NEMO multi-year residual.
    """

    def __init__(self, grid, z_coord, config, dino_cfg, forcing, dt_model: float, *,
                 row_slice: slice = slice(12, 47),
                 depth_bands_m=((0.0, 200.0), (200.0, 1000.0), (1000.0, None)),
                 rho0: float = constants.rho_ocean, cp: float = constants.c_sw):
        self.grid = grid
        self.z_coord = z_coord
        self.config = config
        self.dino_cfg = dino_cfg
        self.forcing = forcing
        self.dt_model = float(dt_model)
        self.row_slice = row_slice
        self.depth_bands_m = tuple(depth_bands_m)
        self.rho0 = float(rho0)
        self.cp = float(cp)

        dz_ref = np.asarray(z_coord.dz_ref)
        z_cum = np.cumsum(dz_ref) - 0.5 * dz_ref  # cell-centre depth, positive down
        self._band_lev_masks = []
        for lo, hi in self.depth_bands_m:
            m = z_cum >= lo
            if hi is not None:
                m = m & (z_cum < hi)
            self._band_lev_masks.append(m)

        area = np.asarray(grid.area, dtype=np.float64)  # (n_lat, n_lon)
        self._box_area = float(area[row_slice, :].sum())

        self.n_samples = 0
        self.time_series_t = []
        self.time_series_H = {i: [] for i in range(len(self.depth_bands_m))}
        self.accum_W = {
            term: [0.0] * len(self.depth_bands_m) for term in TERM_NAMES
        }
        self._prev_H = None

    def _box_integral_W(self, dT_dt, mask, area, h_k):
        """(n_lat, n_lon, nlev) dT/dt [degC/s] -> per-band box power [W]."""
        dT_dt = np.asarray(dT_dt)
        vol = area[:, :, None] * np.asarray(h_k) * np.asarray(mask)[:, :, None]
        vol_box = vol[self.row_slice, :, :]
        dT_box = dT_dt[self.row_slice, :, :]
        out = []
        for lev_mask in self._band_lev_masks:
            w = self.rho0 * self.cp * float(np.sum(dT_box[:, :, lev_mask] * vol_box[:, :, lev_mask]))
            out.append(w)
        return out

    def _heat_content_J(self, T, mask, area, h_k):
        T = np.asarray(T)
        vol = area[:, :, None] * np.asarray(h_k) * np.asarray(mask)[:, :, None]
        vol_box = vol[self.row_slice, :, :]
        T_box = T[self.row_slice, :, :]
        out = {}
        for i, lev_mask in enumerate(self._band_lev_masks):
            out[i] = self.rho0 * self.cp * float(np.sum(T_box[:, :, lev_mask] * vol_box[:, :, lev_mask]))
        return out

    def sample(self, state, *, dt_step: float, t_seconds: float):
        """Accumulate one sample. ``dt_step`` = seconds between calls (the
        sampling interval, e.g. 86400 for daily); NOT the model's dynamical
        dt. Diagnostics only — ``state`` is read-only here.

        The FIRST call (no previous sample yet) only records the t=0
        heat-content baseline — there is no preceding interval to
        attribute a tendency to, so nothing is accumulated into
        ``accum_W`` until the SECOND call closes the first interval.
        """
        mask = np.asarray(state.land_mask.data) > 0.5
        area = np.asarray(self.grid.area, dtype=np.float64)
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m,
        )
        H = self._heat_content_J(np.asarray(state.T.data), mask, area, h_k)

        if self._prev_H is not None:
            # Rate evaluated at the state closing this interval (same
            # end-point convention as the heat-content sample it is
            # compared against). ``self.dt_model`` (NOT ``dt_step``) is the
            # physics timestep fed to the operators; ``dt_step`` is used
            # ONLY below as the Joule-integration weight over this sample
            # interval — see ``compute_box_heat_dT_terms`` docstring for
            # why conflating the two would bias every explicit term.
            terms = compute_box_heat_dT_terms(
                state, self.grid, self.z_coord, self.config, self.dino_cfg,
                self.forcing, self.dt_model, t_seconds,
            )
            interval_explicit_J = []
            for term in ("adv_h", "adv_v", "iso_redi", "forcing"):
                per_band_W = self._box_integral_W(terms[term], mask, area, h_k)
                interval_explicit_J.append(per_band_W)
            for i in range(len(self.depth_bands_m)):
                dH = H[i] - self._prev_H[i]
                d_explicit = sum(interval_explicit_J[j][i] for j in range(4)) * dt_step
                for j, term in enumerate(("adv_h", "adv_v", "iso_redi", "forcing")):
                    self.accum_W[term][i] += interval_explicit_J[j][i] * dt_step
                self.accum_W["vertmix"][i] += (dH - d_explicit)

        self._prev_H = H
        self.n_samples += 1
        self.time_series_t.append(t_seconds)
        for i in range(len(self.depth_bands_m)):
            self.time_series_H[i].append(H[i])

    def summary(self, total_seconds: float | None = None):
        """Return the closure-validated per-term budget.

        Parameters
        ----------
        total_seconds : float or None
            Elapsed accumulation window [s]. Default ``None`` derives it
            from the recorded sample times
            (``time_series_t[-1] - time_series_t[0]``) — the correct value
            whenever every ``.sample()`` call used the SAME ``dt_step``
            (the documented/only supported usage); pass explicitly only if
            the caller has a different convention.

        Returns
        -------
        dict with per-band, per-term W and W/m^2, plus the closure residual
        (sum of terms vs measured dH/dt) as the instrument's own validation
        gate.
        """
        if total_seconds is None:
            total_seconds = self.time_series_t[-1] - self.time_series_t[0]
        out = {"box_area_m2": self._box_area, "n_samples": self.n_samples,
               "bands": self.depth_bands_m, "terms": {}}
        dH_total_J = [
            self.time_series_H[i][-1] - self.time_series_H[i][0]
            for i in range(len(self.depth_bands_m))
        ]
        for i, band in enumerate(self.depth_bands_m):
            band_out = {}
            term_sum_J = 0.0
            for term in TERM_NAMES:
                J = self.accum_W[term][i]
                term_sum_J += J
                band_out[term] = {
                    "J": J,
                    "W_mean": J / total_seconds,
                    "W_per_m2": J / total_seconds / self._box_area,
                }
            residual_J = dH_total_J[i] - term_sum_J
            band_out["dH_J"] = dH_total_J[i]
            band_out["closure_residual_J"] = residual_J
            band_out["closure_residual_W_per_m2"] = (
                residual_J / total_seconds / self._box_area
            )
            out["terms"][band] = band_out
        return out


__all__ = (
    "TERM_NAMES",
    "BoxHeatBudgetAccumulator",
    "compute_box_heat_dT_terms",
)
