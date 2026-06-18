"""Spin off a forced column LES for each worst-performing column + diagnose K/w_e.

Stage 4–6 library of ``docs/COMPARE_REANALYSIS.md``: the orchestration that ties
the LES-side components into a runnable pipeline for one flagged column —

  manifest record + GCM column
    → regime + LES resolution      (les_regime, iter 12)
    → plane grid + stretched height coord with the column θ as θ_ref
    → SCM/large-scale forcing       (column_forcing iter 5 → plane physics_fn)
    → Newtonian top relaxation      (les_vertical_mapping, iter 13)
    → run the plane NH LES at LES resolution   (compressible_euler_plane)
    → diagnose the closure coefficient   (column_les_diagnosis, iter 14).

The data-agnostic orchestration (:func:`build_column_les_setup`,
:func:`run_column_les_pipeline`) is importable + unit-tested with the heavy LES
run injected as ``run_les_fn``; :func:`run_forced_les` is the real plane-dycore
time loop (reusing the ``run_les_plane`` machinery).  The CLI that loops a
manifest lives in ``scripts/run/run_column_les.py`` (imports this module).
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from legoesm.atmosphere.column_forcing import (
    ColumnLargeScaleState,
    build_column_scm_forcing,
    coriolis_f_c,
)
from legoesm.atmosphere.dynamics.column_les_diagnosis import (
    diagnose_column_coefficient,
)
from legoesm.atmosphere.dynamics.les_regime import (
    LESRegimeConfig,
    les_resolution_for_column,
)
from legoesm.atmosphere.dynamics.les_vertical_mapping import (
    build_top_relaxation,
    interpolate_column_to_les,
)


class ColumnLESConfig(NamedTuple):
    """Campaign config for the column-LES spin-off."""

    regime: LESRegimeConfig = LESRegimeConfig()
    relax_width_frac: float = 0.25   # relaxation layer = top fraction of the domain
    relax_tau_s: float = 600.0       # Newtonian relaxation timescale [s]
    p_sfc_Pa: float = 1.0e5
    diagnosis_method: str = "eddy_diffusivity"
    # The GCM CLUBBLiteConfig.l_mix_max, REQUIRED for the dimensionless
    # ``"clubb_coefficient"`` diagnosis (C_K = K_m/(ℓ·√wp2)) so the diagnosed
    # mixing length matches the GCM's; unused by the other methods.
    clubb_l_mix_max: float | None = None


def validate_column_les_config(config: ColumnLESConfig) -> None:
    """Reject non-physical campaign settings (relaxation, surface pressure)."""
    if not (0.0 < config.relax_width_frac <= 1.0):
        raise ValueError(
            f"relax_width_frac must be in (0, 1], got {config.relax_width_frac}."
        )
    if config.relax_tau_s <= 0.0:
        raise ValueError(f"relax_tau_s must be > 0, got {config.relax_tau_s}.")
    if config.p_sfc_Pa <= 0.0:
        raise ValueError(f"p_sfc_Pa must be > 0, got {config.p_sfc_Pa}.")
    if config.diagnosis_method == "clubb_coefficient" and (
        config.clubb_l_mix_max is None or config.clubb_l_mix_max <= 0.0
    ):
        raise ValueError(
            "diagnosis_method='clubb_coefficient' requires clubb_l_mix_max > 0 "
            "(the GCM CLUBBLiteConfig.l_mix_max)."
        )


def coefficient_value(diagnosis: Any, method: str):
    """Reduce a diagnosis object to the representative coefficient for ``method``.

    ``"eddy_diffusivity"`` → the ``K`` profile; ``"entrainment"`` → the scalar
    ``w_entrainment``.  Raises on an unexpected diagnosis/method pair rather than
    silently emitting ``NaN``.
    """
    if method == "eddy_diffusivity" and hasattr(diagnosis, "K"):
        return diagnosis.K
    if method == "clubb_coefficient" and hasattr(diagnosis, "C_K"):
        return diagnosis.C_K
    if method == "entrainment" and hasattr(diagnosis, "w_entrainment"):
        return diagnosis.w_entrainment
    raise ValueError(
        f"diagnosis {type(diagnosis).__name__} has no coefficient for method "
        f"{method!r}."
    )


class ColumnLESSetup(NamedTuple):
    """Everything needed to run + diagnose one column's LES."""

    grid: Any
    height_coord: Any
    forcing_physics: Callable        # plane physics_fn (large-scale forcing)
    relax_theta_target: jax.Array    # (nlev,) GCM θ on the LES grid
    relax_rate: jax.Array            # (nlev,) Newtonian rate [1/s]
    q_v_init: jax.Array              # (nlev,) GCM q_v on the LES grid (moist IC)
    regime: str
    resolution: Any
    f_c: float


def build_column_les_setup(
    *,
    cape_J_kg: float,
    lat_rad: float,
    gcm_z: jax.Array,
    gcm_theta: jax.Array,
    ls_state: ColumnLargeScaleState,
    config: ColumnLESConfig = ColumnLESConfig(),
) -> ColumnLESSetup:
    """Build the plane grid + height coord + forcing + relaxation for one column.

    ``cape_J_kg`` (from the manifest env tag) picks the regime/resolution;
    ``lat_rad`` sets the f-plane Coriolis; ``gcm_z`` / ``gcm_theta`` are the GCM
    column heights + potential temperature (ascending or top-down — interpolation
    sorts internally); ``ls_state`` (from the grid-side extractor, iter 10) is
    the large-scale forcing.  The LES top must lie within the GCM column
    (:func:`build_top_relaxation` raises otherwise).
    """
    from legoesm.atmosphere.dynamics.plane_large_scale_forcing import (
        make_plane_ls_forcing_physics,
    )
    from legoesm.grids.plane import create_plane_grid
    from legoesm.grids.vertical import create_stretched_height_coordinate

    validate_column_les_config(config)
    regime, res = les_resolution_for_column(cape_J_kg, config.regime)
    f_c = coriolis_f_c(lat_rad)
    grid = create_plane_grid(
        nx=res.nx, ny=res.ny, nlev=res.nlev, dx=res.dx_m, dy=res.dx_m,
        coriolis_mode="f_plane", f0=f_c,
    )
    # The GCM column θ is the LES reference θ(z): interpolate to whatever heights
    # the stretched coordinate builds.
    theta_ref_fn = lambda z: interpolate_column_to_les(gcm_z, gcm_theta, z)  # noqa: E731
    hc = create_stretched_height_coordinate(
        res.nlev, H=res.domain_top_m, dz_sfc=res.dz_sfc_m,
        theta_ref_fn=theta_ref_fn, p_sfc=config.p_sfc_Pa,
    )

    # Steady large-scale forcing profiles from the validated SCMForcing assembly.
    # The forcing lives on the GCM column levels (= gcm_z); interpolate each
    # channel onto the LES height grid (hc.z_full) before handing it to the
    # plane forcing physics, which expects (nlev_LES,) profiles.
    forcing = build_column_scm_forcing(ls_state)

    gcm_z = jnp.asarray(gcm_z)

    def _to_les(profile_fn):
        if profile_fn is None:
            return None
        profile = jnp.asarray(profile_fn(0.0))
        if profile.shape != gcm_z.shape:
            raise ValueError(
                f"forcing profile shape {profile.shape} != gcm_z shape "
                f"{gcm_z.shape}; the forcing must be co-located on the GCM column."
            )
        return interpolate_column_to_les(gcm_z, profile, hc.z_full)

    forcing_physics = make_plane_ls_forcing_physics(
        hc,
        w_ls=_to_les(forcing.subsidence_w),
        theta_adv=_to_les(forcing.theta_adv),
        qv_adv=_to_les(forcing.qv_adv),
    )

    # Wire the GCM column's geostrophic wind into the plane Coriolis as its
    # reference wind: the f-plane Coriolis term reads u_geo0/v_geo0 from the
    # height coordinate and applies f×(V − V_geo).  None (cubed-sphere /
    # equatorial columns, where the extractor leaves u_geo=None) keeps u_geo0=None
    # → the Coriolis falls back to f×V (no geostrophic target), unchanged.
    u_geo0 = _to_les(forcing.u_geo)
    v_geo0 = _to_les(forcing.v_geo)
    if u_geo0 is not None:
        hc = hc._replace(u_geo0=u_geo0, v_geo0=v_geo0)

    relax_width_m = config.relax_width_frac * res.domain_top_m
    target, rate = build_top_relaxation(
        hc.z_full, res.domain_top_m, gcm_z, gcm_theta,
        relax_width_m=relax_width_m, inv_tau=1.0 / config.relax_tau_s,
    )
    # Moist initial condition: the GCM column q_v interpolated onto the LES grid
    # (the LES must carry moisture so the resolved w'q_v' flux + the moist
    # diagnosis exist).
    q_v_init = interpolate_column_to_les(gcm_z, ls_state.q_v, hc.z_full)
    return ColumnLESSetup(
        grid=grid, height_coord=hc, forcing_physics=forcing_physics,
        relax_theta_target=target, relax_rate=rate, q_v_init=q_v_init,
        regime=regime, resolution=res, f_c=f_c,
    )


def run_column_les_pipeline(
    setup: ColumnLESSetup,
    run_les_fn: Callable[[ColumnLESSetup], Any],
    *,
    method: str = "eddy_diffusivity",
    qv_slot: int = 0,
    l_mix_max: float | None = None,
):
    """Run the LES (via ``run_les_fn``) and diagnose its closure coefficient.

    ``run_les_fn(setup)`` returns the finished plane LES state; the closure
    coefficient is then diagnosed with :func:`diagnose_column_coefficient`
    (``method`` raises on unknown).  ``run_les_fn`` is injected so the heavy run
    can be mocked in tests and swapped for :func:`run_forced_les` in production.
    ``l_mix_max`` (the GCM mixing length) is required for ``method=
    "clubb_coefficient"`` and ignored otherwise.
    """
    final_state = run_les_fn(setup)
    return diagnose_column_coefficient(
        final_state, setup.height_coord, method=method, qv_slot=qv_slot,
        l_mix_max=l_mix_max,
    )


def run_forced_les(
    setup: ColumnLESSetup, *, dt_s: float, n_steps: int, theta_prime_seed: float = 0.1
):
    """Real plane-NH LES time loop with the large-scale forcing + top relaxation.

    Reuses the ``compressible_euler_plane`` dycore (``PlaneCompressibleEulerModel``)
    with the setup's ``forcing_physics`` applied each step and the Newtonian top
    relaxation nudging θ toward the GCM column near the LES top.  Returns the
    final plane state.  Heavy (LES resolution) — used by :func:`main`, not the
    unit tests.
    """
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        PlaneCompressibleEulerModel,
        make_flat_plane_terrain_metric,
        make_rest_state,
    )
    from legoesm.atmosphere.dynamics.les_vertical_mapping import relaxation_tendency

    # Top relaxation: the target equals ``hc.theta_ref`` by construction (both are
    # the GCM column θ interpolated onto ``hc.z_full``), so this relaxes the LES
    # total θ back to the GCM reference near the lid — i.e. damps the resolved θ'
    # perturbation in the relaxation layer (the standard top sponge toward the
    # GCM column), leaving the diagnosis region below it untouched.
    grid, hc = setup.grid, setup.height_coord
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        use_coriolis=True, semi_implicit_acoustic=True,
        substep_horizontal_acoustic=True, smagorinsky_cs=0.17,
        sgs_vertical_diffusion=True, fix_mass=True, anchor_mass_to_initial=True,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    dtype = grid.area_T.dtype
    state = make_rest_state(grid, hc, dtype=dtype)
    ny, nx, nlev = grid.ny, grid.nx, hc.n_levels
    # Small θ' seed so resolved eddies spin up.
    key = jax.random.PRNGKey(0)
    seed = theta_prime_seed * jax.random.normal(key, (ny, nx, nlev), dtype=dtype)
    # Moist IC: broadcast the GCM column q_v (on the LES grid) into tracer slot 0
    # so the LES carries moisture (a dry rest state has no tracers, which would
    # break the resolved w'q_v' flux + the moist closure diagnosis).
    q_v_init = jnp.asarray(setup.q_v_init, dtype=dtype)
    tracers = jnp.zeros((ny, nx, nlev, 1), dtype=dtype).at[..., 0].set(
        jnp.broadcast_to(q_v_init, (ny, nx, nlev))
    )
    state = state._replace(
        theta_prime=state.theta_prime.replace(data=seed),
        tracers=state.tracers.replace(data=tracers),
    )

    target = jnp.asarray(setup.relax_theta_target, dtype=dtype)
    rate = jnp.asarray(setup.relax_rate, dtype=dtype)

    def _relax(st):
        theta_total = hc.theta_ref + st.theta_prime.data
        tend = relaxation_tendency(theta_total, target, rate)
        return st._replace(
            theta_prime=st.theta_prime.replace(data=st.theta_prime.data + dt_s * tend)
        )

    for _ in range(int(n_steps)):
        state = model.step(state, dt=dt_s, physics_fn=setup.forcing_physics)
        state = _relax(state)
    return state


def extract_gcm_column(
    *,
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    p_s: jax.Array,
    grid: Any,
    sigma: Any,
    col_index: tuple[int, ...],
    lat_rad: float,
) -> tuple[jax.Array, jax.Array, ColumnLargeScaleState]:
    """Extract one column's heights, θ profile, and large-scale forcing state.

    Reuses the grid-side extractor (iter 10/27) for the ``ColumnLargeScaleState``
    and the hydrostatic height integral + Exner for the LES reference profiles.
    Returns ``(gcm_z, gcm_theta, ls_state)`` — ``gcm_z`` / ``gcm_theta`` are the
    column's full-level heights [m] and potential temperature [K].

    Grid-agnostic: ``col_index`` is ``(i_lat, i_lon)`` on a lat-lon grid or
    ``(face, i, j)`` on a cubed-sphere grid — the spatial gather
    ``arr[tuple(col_index)]`` keeps the trailing level axis either way, and the
    forcing extraction routes through the grid dispatcher.
    """
    from legoesm.atmosphere.dynamics.column_large_scale_extract import (
        extract_column_forcing,
    )
    from legoesm.atmosphere.physics._shared import (
        compute_heights_from_sigma,
        exner_function,
    )

    ls_state = extract_column_forcing(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=lat_rad, col_index=col_index,
    )
    idx = tuple(int(c) for c in col_index)
    sigma_full = jnp.asarray(sigma.sigma_full)
    sigma_half = jnp.asarray(sigma.sigma_half)
    p_s_col = jnp.asarray(p_s)[idx]
    p_full_col = p_s_col * sigma_full
    p_half_col = p_s_col * sigma_half
    T_col = jnp.asarray(T)[idx]
    q_col = jnp.asarray(q_v)[idx]
    z_full, _ = compute_heights_from_sigma(
        T_col[None, :], p_half_col[None, :], q_col[None, :]
    )
    gcm_z = z_full[0]
    gcm_theta = T_col / exner_function(p_full_col)
    return gcm_z, gcm_theta, ls_state


def process_column(
    record: Any,
    *,
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    p_s: jax.Array,
    grid: Any,
    sigma: Any,
    config: ColumnLESConfig,
    run_les_fn: Callable[[ColumnLESSetup], Any],
):
    """Full per-column pipeline: extract → setup → run (injected) → diagnose.

    ``record`` is a manifest :class:`~legoesm.training.column_manifest.ColumnRecord`
    (provides ``grid_index``, ``lat_deg``, ``environment.cape_J_kg``).  Returns the
    diagnosed closure-coefficient object.
    """
    lat_rad = float(jnp.deg2rad(record.lat_deg))
    gcm_z, gcm_theta, ls_state = extract_gcm_column(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma=sigma,
        col_index=tuple(record.grid_index), lat_rad=lat_rad,
    )
    setup = build_column_les_setup(
        cape_J_kg=record.environment.cape_J_kg, lat_rad=lat_rad,
        gcm_z=gcm_z, gcm_theta=gcm_theta, ls_state=ls_state, config=config,
    )
    return run_column_les_pipeline(
        setup, run_les_fn, method=config.diagnosis_method,
        l_mix_max=config.clubb_l_mix_max,
    )

