"""#1226 online box heat-budget accumulator.

Instrument for the ACC-deficit causal chain: identifies WHICH tendency term
carries the +38.5 W/m^2 excess heat convergence into legoESM's DINO southern
channel box, by accumulating the box-integrated per-term heat tendency
DURING a multi-year run (a single-snapshot probe cannot resolve this — see
``tracer_tendency_compare.py``'s per-term buckets, which this module reuses
at daily cadence instead of one twin step).

Terms (mirrors ``tracer_tendency_compare.py``'s bucket split exactly, reusing
the SAME production functions — no re-derived numerics):
  ADV_H    — horizontal advection (incl. GM-bolus-through-FCT fold, this
             recipe's ``gm_bolus_advection="through_fct"``)
  ADV_V    — vertical advection
  ISO_REDI — GM/Redi tracer tendency (``gm_redi_tracer_tendency_latlon``,
             the same call ``tendency_probe.probe_latlon_cgrid`` makes,
             evaluated on the state's own T/S — the online run has no NEMO
             before-level twin, unlike ``tracer_tendency_compare.py``). When
             ``config.gm_redi.implicit_K33=True`` (the DINO recipe default)
             this EXCLUDES the vertical isoneutral diagonal K_33 — see K33
             below.
  FORCING  — surface restoring + Jerlov SW penetration (the SAME functions
             ``apply_dino_lat_lon_surface_forcing`` calls)
  K33      — (#1226 instrument fix) the vertical isoneutral diffusivity's
             REALIZED backward-Euler increment, measured the SAME way as
             VERTMIX below (via ``model._apply_implicit_vertical_mixing``),
             isolated into its OWN bucket rather than folded into either
             ISO_REDI or VERTMIX. K_33 is computed by
             ``compute_isoneutral_K33_latlon`` — the IDENTICAL helper
             ``ocean_model_latlon_cgrid.py``'s own step loop calls (as
             ``k33_implicit``, e.g. line ~3905) and passes to
             ``_apply_implicit_vertical_mixing(K33_iso=...)`` (e.g. lines
             ~4350, ~4363, ~6591, ~6617, ~7056, ~7279) — and
             ``tendency_probe.py:263`` (the single-snapshot probe) already
             uses for its own K33 comparison. Kept SEPARATE (not folded
             into VERTMIX) because NEMO's ISO_REDI diagnostic grouping is
             ``ldf + (zdf - zdfp)`` — i.e. NEMO folds the isoneutral
             vertical part INTO its lateral-mixing bucket, not into pure
             vertical diffusion (``zdfp``). A caller wanting the
             NEMO-comparable grouping computes ``iso_redi + k33`` (matches
             NEMO ISO_REDI) and uses ``vertmix`` alone (matches NEMO
             ``zdfp``); folding K33 into ``vertmix`` here would make that
             NEMO-matching regrouping impossible to reconstruct after the
             fact (the two are summed inside one realized backward-Euler
             increment, inseparable once measured that way).
  VERTMIX  — DIRECTLY MEASURED (#1226 instrument fix): the model's own
             ``LatLonCGridOceanModel._apply_implicit_vertical_mixing``
             (T-only, ``do_momentum=False``) is called on the SAMPLED state
             with its own dt_model, and the REALIZED increment
             ``(T_after - T_before) / dt_model`` is box-integrated. This is
             NOT the explicit flux ``-rho0*cp*K*dT/dz`` — legoESM applies
             K_v IMPLICITLY (one backward-Euler solve per step), and when
             the implicit diffusion number ``K*dt/dz^2 >> 1`` (true for the
             convective-adjustment K~100 m^2/s case) that solve SATURATES:
             it drives the column toward homogeneity rather than moving the
             explicit flux, so the explicit formula overstates the realized
             heat transfer by an order of magnitude (measured 18x on this
             recipe: explicit -108 vs realized -5.93 W/m^2). The K-profile
             recompute reuses ``compute_vertical_K_profiles`` via the SAME
             fallback branch ``_apply_implicit_vertical_mixing`` already
             takes when ``K_v_phys is None`` (no re-derived closure
             numerics) — this is the identical fallback the online run
             loop itself takes whenever the K profile isn't threaded
             through from the step's own tendency computation, so it is
             the production closure, evaluated on the sampled state rather
             than an in-step intermediate (the same convention ISO_REDI
             already uses for GM/Redi above). This bucket EXCLUDES K33 (see
             above): the call leaves ``K33_iso`` at its ``None`` default, so
             the fold is skipped (``if K33_iso is not None``) and VERTMIX
             measures PURE vertical diffusion only, matching NEMO's
             ``zdfp``. #1226: that ``None`` is now CORRECT-BY-DESIGN rather
             than the accident it used to be — previously K33 was ALSO
             absent from ISO_REDI (dropped there by ``implicit_K33=True``),
             so it was missing from BOTH buckets, which invalidated the
             ISO_REDI/VERTMIX per-term split for oracle comparison (their
             SUM still closed, via the residual). See K33 above.

  RESIDUAL — TOTAL − (ADV_H + ADV_V + ISO_REDI + FORCING + K33 + VERTMIX),
             where TOTAL is the actual box heat-content tendency computed
             from consecutive state samples. With VERTMIX (and now K33)
             measured directly, this is a genuine CLOSURE DIAGNOSTIC — it
             should be ≈0 if every term is measured correctly, and a
             nonzero value signals INSTRUMENT ERROR (endpoint-rate sampling
             bias, an operator gap, or a masking mismatch between the
             direct terms and the box integrator), not a real, unattributed
             physical process. A signal in VERTMIX/K33 is now attributable
             to vertical mixing itself; a signal in RESIDUAL means
             "something the direct terms are not capturing" and should be
             investigated as an instrument gap, not folded silently into
             any physical term.

#1226 FIX (this module): ``compute_box_vertmix_dT`` previously called
``model._apply_implicit_vertical_mixing`` WITHOUT a ``K33_iso`` argument, so
it silently defaulted to ``None`` and K_33 was dropped from BOTH the
ISO_REDI bucket (excluded there by ``implicit_K33=True``) and the VERTMIX
bucket (never passed in here) — an instrument gap, not a real physical
"vertical mixing is 8x too weak" signal at 200-1000 m as an earlier
(retracted) reading of this accumulator concluded. K33 is now computed
explicitly (see K33 above) and threaded through: ``vertmix`` measures pure
diffusion (``K33_iso`` left at its ``None`` default, now CORRECT-BY-DESIGN
rather than accidental — see VERTMIX above), and the new ``k33`` bucket
measures the isoneutral-vertical increment on its own, so ``iso_redi + k33``
reproduces NEMO's ISO_REDI grouping exactly and only the individual
ISO_REDI/VERTMIX split (not their sums) was ever invalid.

NEMO-COMPARABLE GROUPING (read this before comparing any bucket to NEMO):
  NEMO ISO_REDI (``ldf + (zdf - zdfp)``)   ==  ``iso_redi + k33``
  NEMO ``zdfp``  (pure vertical diffusion) ==  ``vertmix``
``iso_redi + vertmix`` is NOT NEMO's ``ldf + zdf`` any more — post-fix it is
``ldf + zdfp``, i.e. it silently EXCLUDES K33. Any consumer still summing
``iso_redi + vertmix`` for a NEMO comparison is dropping the isoneutral
vertical diagonal, which is the very error this fix exists to remove.

Sampling: intended for a daily cadence inside a multi-year run loop (the
caller decides N steps/sample). Diagnostics only — never mutates the
model state fed to ``model.step()`` (the implicit-mixing probe solves on a
COPY of the sampled state's T field; its result is read, never written
back).

CAVEAT (endpoint-rate sampling, stated not hidden): each interval's ADV_H/
ADV_V/ISO_REDI/FORCING/K33/VERTMIX rate is evaluated ONCE, at the state CLOSING
that interval (not a midpoint or trapezoidal average), then multiplied by
the whole interval length. For a slowly-varying (multi-year-mean) signal
this is negligible; for the SEASONAL cycle DINO's forcing runs (nn_ann_cyc)
a daily endpoint sample carries a same-order-as-the-day's-drift bias
(bounded by day-length / forcing-timescale, ~0.2% of tau_T=11.85 days for
DINO — small, but real). This sampling error, plus any residual operator
gap, now lands in the RESIDUAL bucket (not VERTMIX) — use RESIDUAL as the
instrument's own error bar, and treat a RESIDUAL that is NOT small relative
to the band's other terms as a reason to distrust the decomposition for
that band/window, not as evidence of an unmodeled physical process.

PRECISION — fp64 STORAGE IS REQUIRED TO READ THE REALIZED-INCREMENT BUCKETS.
``k33`` and ``vertmix`` are measured as ``(T_after - T_before) / dt``, a
difference of two O(10) degC temperatures: a CATASTROPHIC CANCELLATION. At the
default fp32 policy any increment below ~``ULP(T)/dt`` (~2e-9 degC/s at
T~20 degC) is UNRESOLVABLE and reports as exactly 0 or a small ULP multiple.
MEASURED on the unit-test grid: at fp32 the k33 bucket's PEAK value was
``max|dT_k33| = 2.1193e-09``, exactly **1.00 ULP**, with 95% of cells
quantizing to 0 — and it was BIT-IDENTICAL between ``eos_depth="insitu"`` and
``"geometric"`` even though the underlying K33 field differed (rel-L2
2.003e-3), i.e. both conventions rounded to the same single quantum. The same
comparison at fp64 storage gives ``1.9587e-09`` at **4.96e+08 ULP** of margin,
with the eos_depth sensitivity now visible (``iso_redi`` rel-L2 3.810e-04).

Two consequences, both learned the hard way (#1226):
  * ``JAX_ENABLE_X64=1`` is NOT sufficient — it only PERMITS f64. The
    ``legoesm.core.precision`` policy is an INDEPENDENT axis and
    ``get_policy().storage`` defaults to float32. Constructors read it
    (``init_latlon_cgrid.py:97,172,224,284`` all do ``dtype =
    get_policy().storage``), so a state must be BUILT UNDER an fp64 policy —
    reusing a state constructed earlier silently stays fp32.
  * **A SMALL BUCKET READ UNDER fp32 IS NOT EVIDENCE OF PHYSICAL ABSENCE.**
    It is the false-negative twin of the dropped-``K33_iso`` bug above: the
    instrument lies by QUANTIZATION instead of by a missing kwarg, at
    identical downstream cost. A deep-ocean fp32 ``vertmix`` or ``k33`` of
    "≈0" is exactly the shape of the retracted "vertical mixing is 8x too
    weak at 200-1000 m" reading. Confirm fp64 storage BEFORE concluding a
    term contributes nothing.
The #1226 drivers gate on ``ocean.fidelity.precision_gate.require_fp64``, so
readings taken through them are sound; anything else must set the policy
itself.
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
    add_bolus_to_advecting_flux,
    compute_advection_flux_div_pair,
    static_kappa_redi_override,
)
from legoesm.ocean.experiments.dino import (
    dino_Q_sr_seasonal,
    dino_T_star_seasonal,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_isoneutral_K33_latlon,
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

TERM_NAMES = ("adv_h", "adv_v", "iso_redi", "forcing", "k33", "vertmix")


class _GridShim:
    """Matches ``tracer_tendency_compare.py``'s shim for
    ``restoring_surface_forcing`` (only ``grid_lat`` shape is read)."""

    def __init__(self, cell_mask):
        self.grid_lat = jnp.zeros_like(cell_mask)


def compute_box_heat_dT_terms(state, grid, z_coord, config, dino_cfg, forcing,
                              dt_model: float, t_seconds: float):
    """Per-cell dT/dt [degC/s] for ADV_H, ADV_V, ISO_REDI, FORCING, K33.

    Reuses the exact production functions (``compute_advection_flux_div_pair``,
    ``add_bolus_to_advecting_flux``, ``gm_redi_tracer_tendency_latlon``,
    ``compute_isoneutral_K33_latlon``, ``restoring_surface_forcing``,
    ``shortwave_penetration_tendency``) — same call pattern as
    ``tracer_tendency_compare.py``/``tendency_probe.py`` but on the state's
    OWN T/S/eta (no NEMO before-level bridge; this is a lego-only online run).

    ``dt_model`` MUST be the model's own dynamical timestep (``DT``, e.g.
    2700 s for ``nemo_dino_kamm_mlf``), NOT the accumulator's (much longer)
    sampling interval — every one of these functions is ``dt``-SENSITIVE in
    its own right (physics-validator review, #1226):
    ``restoring_surface_forcing(implicit=True)`` uses ``eff_tau = tau + dt``
    (a 32x-too-large ``dt`` biases the restoring rate low by
    ``dt_step/(tau+dt_step)``, ~8% at daily sampling with DINO's
    ``tau_T``); ``compute_advection_flux_div_pair``'s FCT limiter clips the
    antidiffusive flux against a low-order provisional update scaled by
    ``dt`` (wrong ``dt`` -> wrong limiter bound -> wrong flux, NOT just a
    scaling error); ``gm_redi_tracer_tendency_latlon``'s MSC
    (``ln_traldf_msc``) stability clamp on the vertical diagonal is also
    ``dt``-set. Passing the sampling interval here would silently corrupt
    all three rates, not merely rescale them — the caller (``sample()``)
    keeps the sampling interval OUT of this call and applies it only as the
    Joule-integration weight afterward.

    K33 is returned SEPARATELY from ISO_REDI (module docstring) as the
    per-cell dT/dt [degC/s] of the vertical isoneutral diffusivity's
    EXPLICIT flux-divergence form (``-d/dz(K33 dT/dz)`` via
    ``implicit_vertical_diffusion_ocean``'s realized increment on the
    state's own T, at ``dt_model`` -- the SAME construction
    ``tendency_probe.py:263-292`` already uses for its own K33 comparison,
    reused here rather than re-derived). This is a SEPARATE call from the
    one ``compute_box_vertmix_dT`` makes into
    ``model._apply_implicit_vertical_mixing`` (which realizes K33 folded
    into the SAME backward-Euler tridiagonal solve as the background K_v);
    the two are numerically close but not bit-identical (one shared solve
    vs two independent ones) -- acceptable for an additive decomposition
    where each bucket is individually a faithful realized-tendency
    measurement, not required to be a linear superposition of one shared
    solve. Zero (not None) when ``config.gm_redi`` is absent or
    ``implicit_K33`` is not set, so summing into ISO_REDI never introduces
    a spurious accounting gap on recipes that don't use it.

    Split-solve error magnitude (#1226 physics-validator review): for
    per-interface implicit diffusion numbers ``a = dt*K_v/dz^2`` (background,
    inside VERTMIX's solve) and ``b = dt*K33/dz^2`` (this call's solve), the
    fractional discrepancy between ``vertmix + k33`` and the model's own
    single combined ``solve(K_v + K33)`` scales like ``~a*b/(1+a+b)`` per
    interface -- second-order-small (sub-percent) whenever the background
    diffusivity is weak relative to K33 (the typical case: K33 from
    kappa_Redi*S^2 is usually << the background K_v away from strong mixed
    layers). Where OTHER diffusivities are large (strong TKE/KPP surface
    mixing, double-diffusion), ``a`` is O(1) and this split-solve
    approximation error migrates into RESIDUAL alongside the endpoint-
    sampling bias already documented in the module CAVEAT -- a nonzero
    RESIDUAL in a strongly-mixed band is therefore not necessarily an
    "instrument gap" in the sense the module docstring's RESIDUAL entry
    otherwise means; it can be this operator split itself. This is the SAME
    approximation ``tendency_probe.py`` already makes (its combined ISO_REDI
    total also mixes an explicit dT_gm with a separately-solved K33
    increment, not one shared tridiagonal system), so it is not a NEW gap
    this fix introduces, only one now visible via a second (K33) bucket.

    Returns
    -------
    dict with keys "adv_h", "adv_v", "iso_redi", "forcing", "k33" ->
    (n_lat, n_lon, nlev)
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
    # uses (ocean_model_latlon_cgrid.py:3537-3540) — needed regardless of
    # GM/Redi so the no-GM/Redi branch still has a valid w_baro.
    flux_div_k = divergence_cgrid(mass_flux_u, mass_flux_v, grid)
    w_baro = diagnose_w_from_flux_div(flux_div_k, z_coord, thickness_weighted=True)

    if getattr(config, "gm_redi", None) is not None:
        kappa_redi_ov, kappa_redi_v_ov = static_kappa_redi_override(config.gm_redi, grid)
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
            kappa_redi_v_override=kappa_redi_v_ov,
            return_bolus_transport=through_fct,
            dt=dt_model,
            # #1226: the DINO nemo_paper card sets eos_depth="geometric"
            # (dino.py:909,933 -> dino_lat_lon_model_config, dino.py:2874) and
            # production threads it into every GM/Redi + K33 call
            # (ocean_model_latlon_cgrid.py:3947,3921). Omitting it silently
            # takes the "insitu" default -- a DIFFERENT density convention
            # from NEMO's, which is exactly the omission that invalidated an
            # earlier probe measurement (fidelity_bar_gate.py:363 retraction).
            eos_depth=getattr(config, "eos_depth", "insitu"),
        )
        if through_fct:
            dT_gm, dS_gm, bolus = gm_out
            mass_flux_u, mass_flux_v, w_baro = add_bolus_to_advecting_flux(
                bolus, mass_flux_u, mass_flux_v, u_mask_3d, v_mask_3d, grid, z_coord,
            )
        else:
            dT_gm, dS_gm = gm_out

        if getattr(config.gm_redi, "implicit_K33", False):
            # #1226 FIX: K33 was previously dropped from BOTH buckets
            # (excluded from dT_gm above by implicit_K33=True, and never
            # passed to compute_box_vertmix_dT's _apply_implicit_vertical_
            # mixing call). Compute it explicitly here, the SAME construction
            # tendency_probe.py:263-292 already uses (compute_isoneutral_K33_
            # latlon -> zero at non-wet interfaces -> implicit_vertical_
            # diffusion_ocean realized increment) -- no re-derived numerics.
            K33 = compute_isoneutral_K33_latlon(
                state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
                grid, z_coord, config.gm_redi,
                eos=config.eos, eos_linear=config.eos_linear,
                mask=mask, rho_0=config.constants.rho_0, g=config.constants.g,
                kappa_redi_override=kappa_redi_ov,
                kappa_redi_v_override=kappa_redi_v_ov,
                u_mask=state.u_mask.data, v_mask=state.v_mask.data,
                dt=dt_model,
                # Same eos_depth requirement as the ISO_REDI call above --
                # production passes it here too (ocean_model_latlon_cgrid.py
                # :3921) and so does the sibling probe
                # (tendency_probe.py:275). Dropping it would compute K33 on
                # the WRONG density convention for the nemo_paper card.
                eos_depth=getattr(config, "eos_depth", "insitu"),
            )
            if isinstance(z_coord, OceanPartialCellCoordinate):
                # Same seafloor guard tendency_probe.py:283-285 applies:
                # zero K33 at non-wet interfaces so partial-cell bottom
                # cells never mix against below-bottom values.
                K33 = K33 * z_coord.is_active[..., 1:].astype(K33.dtype)
            from legoesm.ocean.physics.vertical_mixing import (
                build_dz_half,
                implicit_vertical_diffusion_ocean,
            )
            # h_k (computed above, z-star/partial-cell-aware per-cell
            # thickness) IS dz_cell -- reuse it rather than re-deriving the
            # Jacobian from scratch (that's what tendency_probe.py's
            # z_coord.dz_ref * J[..., None] amounts to; this function
            # already carries the true thickness in h_k).
            dz_half = build_dz_half(h_k)
            T_k33 = implicit_vertical_diffusion_ocean(
                state.T.data, K33, h_k, dz_half, dt_model)
            dT_k33 = (T_k33 - state.T.data) / dt_model * mask[:, :, jnp.newaxis]
        else:
            dT_k33 = jnp.zeros_like(state.T.data)
    else:
        dT_gm = jnp.zeros_like(state.T.data)
        dT_k33 = jnp.zeros_like(state.T.data)

    wall_fill_mask = active_3d if getattr(config, "tracer_wall_neumann_fill", True) else None
    (dh_T, dv_T), _ = compute_advection_flux_div_pair(
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
        "k33": dT_k33,
    }


def compute_box_vertmix_dT(state, model, surface_forcing, dt_model: float):
    """Per-cell dT/dt [degC/s] for VERTMIX — the REALIZED implicit-mixing
    increment, T-only (``do_momentum=False``).

    legoESM applies vertical diffusivity via ONE backward-Euler solve per
    step (:meth:`LatLonCGridOceanModel._apply_implicit_vertical_mixing`),
    not an explicit flux — so the tendency actually realized on the
    prognostic state is ``(T_after_solve - T_before_solve) / dt_model``,
    NOT ``-rho0*cp*K*dT/dz`` (that explicit formula overstates the
    realized transfer by up to ~18x when the implicit diffusion number
    ``K*dt/dz^2 >> 1`` saturates the solve).

    Calls the model's OWN method (no re-derived K-profile/solver numerics):
    ``K_v_phys=None``/``A_v_phys=None`` take the SAME fallback branch
    (``compute_vertical_K_profiles`` recomputed from the sampled state) the
    online run loop itself takes whenever the K profile isn't threaded
    through from an in-step tendency computation — the identical convention
    ``compute_box_heat_dT_terms`` already uses for ISO_REDI (recomputed from
    the state's own T/S, not an in-step carry). ``surface_forcing`` (the
    ``OceanSurfaceForcing`` the run loop passes to ``model.step()``) is
    threaded through so the wind-driven TKE surface production term is
    real, not silently zeroed. ``tke_source=None`` (the additive TKE
    energy-recycling source, an in-step-only intermediate) is the one input
    NOT reconstructable from a bare state; its omission means the
    PROGNOSTIC TKE budget itself sees no recycled dissipation for this
    one-off probe call, while the returned diffusivity/mixing (what
    heats/cools T) is otherwise the production closure evaluated on the
    sampled state.

    K33 (#1226 instrument fix): this call NEVER passes ``K33_iso`` (stays
    ``None``, i.e. pure vertical diffusion only) — the isoneutral vertical
    diagonal is measured SEPARATELY as its own ``k33`` bucket by
    ``compute_box_heat_dT_terms`` (via ``compute_isoneutral_K33_latlon`` +
    ``implicit_vertical_diffusion_ocean``, the same construction
    ``tendency_probe.py`` uses), NOT folded in here. Before this fix,
    ``K33_iso`` silently defaulted to ``None`` here TOO, but K33 was ALSO
    excluded from ISO_REDI (``implicit_K33=True`` drops it from the
    explicit flux) — so K33 was dropped from BOTH buckets, not just kept
    out of this one on purpose. That silent double-omission (not this
    function's still-``None`` ``K33_iso``) was the bug.

    Read-only: solves on ``state``'s own T (a fresh JAX value), returns the
    tendency array — never mutates or feeds back into ``model.step()``.

    Returns
    -------
    (n_lat, n_lon, nlev) array, degC/s, land-masked.
    """
    mask = state.land_mask.data
    tke_prognostic = model._tke_prognostic_active()
    tke_old = (state.tke.data if (tke_prognostic and state.tke is not None)
               else None)
    result = model._apply_implicit_vertical_mixing(
        state, dt_model, surface_forcing,
        do_tracers=True, do_momentum=False,
        tke_old=tke_old, tke_source=None,
        return_tke=tke_prognostic,
        n2_tracers=model._n2_before_advection_tracers(state),
        n2_tracers_before=model._n2_nemo_before_tracers(state),
    )
    state_new = result[0] if tke_prognostic else result
    dT_dt = (state_new.T.data - state.T.data) / dt_model
    return dT_dt * mask[:, :, jnp.newaxis]


class BoxHeatBudgetAccumulator:
    """Accumulates box-integrated per-term heat tendency [W] by depth band
    over a multi-year run, sampled at whatever cadence the caller calls
    ``.sample()`` (e.g. daily).

    Parameters
    ----------
    grid, z_coord, config, dino_cfg, forcing
        Same objects the run loop already built (``dino_lat_lon_model_config``,
        ``dino_config_for_recipe``, ``dino_lat_lon_surface_forcing_arrays``).
    model : LatLonCGridOceanModel
        The SAME model instance the run loop steps with. Required for the
        direct VERTMIX measurement — ``compute_box_vertmix_dT`` calls
        ``model._apply_implicit_vertical_mixing`` (the model's own
        production vertical-mixing solve) on each sampled state.
    dt_model : float
        The MODEL's own dynamical timestep [s] (e.g. 2700 for
        ``nemo_dino_kamm_mlf``) — passed to the physics operators
        (``restoring_surface_forcing``, the FCT advection limiter,
        ``gm_redi_tracer_tendency_latlon``'s MSC clamp, and now the
        implicit vertical-mixing solve) inside ``compute_box_heat_dT_terms``
        / ``compute_box_vertmix_dT``. This is DELIBERATELY separate from
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

    def __init__(self, grid, z_coord, config, dino_cfg, forcing, dt_model: float,
                 model=None, *,
                 row_slice: slice = slice(12, 47),
                 depth_bands_m=((0.0, 200.0), (200.0, 1000.0), (1000.0, None)),
                 rho0: float = constants.rho_ocean, cp: float = constants.c_sw):
        self.grid = grid
        self.z_coord = z_coord
        self.config = config
        self.dino_cfg = dino_cfg
        self.forcing = forcing
        self.model = model
        self.dt_model = float(dt_model)
        self.row_slice = row_slice
        self.depth_bands_m = tuple(depth_bands_m)
        self.rho0 = float(rho0)
        self.cp = float(cp)

        if model is not None:
            from legoesm.ocean.experiments.dino import dino_step_surface_forcing
            self._sf = dino_step_surface_forcing(forcing)
        else:
            self._sf = None

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
        # Cumulative totals (existing npz-key convention: backward compat).
        self.accum_W = {
            term: [0.0] * len(self.depth_bands_m) for term in TERM_NAMES
        }
        self.accum_residual_J = [0.0] * len(self.depth_bands_m)
        # Per-interval time series (NEW, #1226): each entry is the Joules
        # accumulated in the ONE interval closing at time_series_t[k] (k>=1)
        # -- lets a caller re-window (early/mid/late) without re-running.
        self.time_series_term_J = {
            term: [[] for _ in range(len(self.depth_bands_m))] for term in TERM_NAMES
        }
        self.time_series_residual_J = [[] for _ in range(len(self.depth_bands_m))]
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
            if self.model is None:
                raise ValueError(
                    "BoxHeatBudgetAccumulator: VERTMIX is now measured "
                    "DIRECTLY (#1226) via the model's own implicit-mixing "
                    "solve -- construct with model=<LatLonCGridOceanModel "
                    "instance used for model.step()>."
                )
            terms["vertmix"] = compute_box_vertmix_dT(
                state, self.model, self._sf, self.dt_model,
            )

            interval_term_J = {}
            for term in TERM_NAMES:
                per_band_W = self._box_integral_W(terms[term], mask, area, h_k)
                interval_term_J[term] = [w * dt_step for w in per_band_W]

            for i in range(len(self.depth_bands_m)):
                dH = H[i] - self._prev_H[i]
                term_sum = sum(interval_term_J[term][i] for term in TERM_NAMES)
                residual = dH - term_sum
                for term in TERM_NAMES:
                    self.accum_W[term][i] += interval_term_J[term][i]
                    self.time_series_term_J[term][i].append(interval_term_J[term][i])
                self.accum_residual_J[i] += residual
                self.time_series_residual_J[i].append(residual)

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
        dict with per-band, per-term W and W/m^2 (all 6 terms in
        ``TERM_NAMES``, VERTMIX directly measured and K33 tracked as its
        own bucket), plus the RESIDUAL (sum of the 6 measured terms vs
        measured dH/dt) — a genuine closure DIAGNOSTIC: a nonzero value
        signals instrument error (endpoint-rate sampling bias, an operator
        gap, or a masking mismatch), not an unattributed physical process
        (see module docstring).
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
            residual_J = self.accum_residual_J[i]
            band_out["dH_J"] = dH_total_J[i]
            band_out["residual"] = {
                "J": residual_J,
                "W_mean": residual_J / total_seconds,
                "W_per_m2": residual_J / total_seconds / self._box_area,
            }
            # Backward-compat aliases (pre-#1226 key names): closure_residual_*
            # was the ONLY residual before VERTMIX was measured directly; now
            # it is identical to "residual" above (both = dH - sum(6 terms)).
            band_out["closure_residual_J"] = residual_J
            band_out["closure_residual_W_per_m2"] = (
                residual_J / total_seconds / self._box_area
            )
            assert abs((term_sum_J + residual_J) - dH_total_J[i]) < (
                1e-6 * max(abs(dH_total_J[i]), 1.0)
            ), f"budget accounting bug for band {band}"
            out["terms"][band] = band_out
        return out


__all__ = (
    "TERM_NAMES",
    "BoxHeatBudgetAccumulator",
    "compute_box_heat_dT_terms",
    "compute_box_vertmix_dT",
)
