"""Run the SCM against a cached LES artifact — the tuner's forward evaluation.

Composes the pieces built earlier into a single forward map ``(LES artifact,
turbulence config) -> scalar loss``: initialise the dycore-free SCM to LES(t=0),
drive it with the artifact's forcing, integrate freely, and score the free-run
profiles against the LES truth on a fixed evaluation grid. ``tune_scm_to_les.py``
minimises this loss over a closure's ``__param_spec__`` (derivative-free — the
Python-loop ``SingleColumnModel.run`` is a forward map, exactly what the
coordinate-probe / random search of D4 needs; the AD path is a follow-up).

Grid/orientation (the coupling the bridge deferred): the SCM lives on a
sigma-pressure grid in **temperature**, top-to-bottom (index 0 = model top); the
LES artifact lives on a **height** grid in **θ**, surface-first. Heights come from
the idealised hydrostatic map ``z = -H·ln(σ)`` (same as the GABLS1/Wangara SCM
cases). θ↔T uses the canonical Exner conversions; regridding uses
:mod:`~legoesm.atmosphere.les_suite.scm_coupling`. Scoring is done on the LES
(increasing) grid held byte-identical across closures (controlled-comparison rule).

Wired for the dry prescribed-surface-flux regimes: the free-convective CBL (``f_c=0``,
gate-0-validated) and — when the artifact carries a geostrophic wind (``f_c!=0``,
``u_geo``) — the sheared CBL and stable SBL, whose Coriolis + geostrophic-pressure-gradient
forcing is applied so the Ekman/jet dynamics reproduce the LES. No subsidence/large-scale
advection profile is regridded yet (moist regimes remain to be wired).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import lax
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
from legoesm.atmosphere.physics import (
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
)

from .bridge import LESReferenceArtifact, LESTruth, prognostic_truth
from .scm_coupling import T_from_theta, interp_profile, regrid_truth, theta_from_temperature
from .score import PrognosticScore, prognostic_profile_score

Array = jnp.ndarray

# Idealised hydrostatic scale height for the σ→z map (matches the GABLS1/Wangara SCM
# setups; the shallow BL keeps θ within millikelvin of the analytic value).
_H_SCALE_M = 8.0e3
_P_S_PA = 1.0e5
# The SCM domain top must sit ABOVE the LES domain, or the CBL hits the model lid and
# the temperature collapses. Auto-computed sigma_top covers this fraction ABOVE the
# LES top (a 30% margin so the free atmosphere is resolved, not clipped at the lid).
_DOMAIN_TOP_MARGIN = 1.3
# AD-loss barrier for a diverged SCM (finite, >> any real loss ~O(1)): the DF path uses
# a concrete +inf, which a traced AD loss cannot — so a large finite penalty steers the
# AD optimiser away from a divergent region (see scm_les_loss_jax).
_AD_DIVERGE_PENALTY = 1.0e6


def _auto_sigma_top(z_top_les_m: float) -> float:
    """σ at the SCM domain top so it sits ~30% above the LES domain top."""
    z_top = _DOMAIN_TOP_MARGIN * float(z_top_les_m)
    return float(jnp.exp(-z_top / _H_SCALE_M))


@dataclass(frozen=True)
class SCMGridSpec:
    """The SCM vertical grid derived from a σ-coordinate (top-to-bottom)."""

    z_scm_m: Array       # (nlev,) heights [m], top-to-bottom (decreasing)
    p_full_pa: Array     # (nlev,) full-level pressure [Pa]
    nlev: int


def _scm_grid(nlev: int, sigma_top: float) -> SCMGridSpec:
    from legoesm.grids.vertical import create_sigma_coordinate

    sigma_coord = create_sigma_coordinate(nlev, sigma_top=sigma_top)
    p_full = sigma_coord.sigma_full * _P_S_PA
    z_scm = -_H_SCALE_M * jnp.log(jnp.maximum(p_full / _P_S_PA, 1e-6))
    return SCMGridSpec(z_scm_m=z_scm, p_full_pa=p_full, nlev=nlev)


def _dry_cbl_physics(turbulence: TurbulenceConfig) -> PhysicsConfig:
    """Turbulence-only physics config (all other modules off) for a dry CBL."""
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=turbulence,
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def _const_profile(profile):
    """A time-constant SCMForcing ``ProfileFn`` returning ``profile`` for every ``t`` — used
    for the steady geostrophic wind of the sheared CBL / stable SBL."""
    return lambda _t: profile


def build_cbl_scm_from_artifact(
    artifact: LESReferenceArtifact,
    turbulence: TurbulenceConfig,
    *,
    nlev: int = 32,
    sigma_top: float | None = None,
    dt: float = 5.0,
) -> tuple[SingleColumnModel, SCMGridSpec]:
    """Build an SCM initialised to LES(t=0), driven by the artifact's CBL forcing.

    The SCM is placed on a σ grid (``nlev``, ``sigma_top``), its temperature IC
    interpolated from the LES θ(t=0) (θ→T via Exner at the SCM pressure), its u/v IC
    from LES(t=0), and driven by the prescribed surface heat flux ``Q0`` the LES
    received (``prescribe='fluxes'``). For the free-convective CBL (``f_c=0``) there is no
    geostrophic/subsidence forcing; for the sheared CBL / stable SBL (``f_c!=0``) the
    artifact's geostrophic wind ``u_geo``/``v_geo`` is applied as the Coriolis +
    geostrophic-pressure-gradient tendency (SCMForcing convention ``du/dt=+f_c(v-v_g)``,
    ``dv/dt=-f_c(u-u_g)``) so the Ekman/jet dynamics are reproduced. Raises on an artifact
    with no prescribed surface flux, or on ``f_c!=0`` with no ``u_geo`` (a malformed
    sheared/stable artifact — the wind would otherwise spin down toward zero, not the LES
    geostrophic balance).

    ``sigma_top=None`` (default) auto-sizes the SCM domain to sit ~30% above the LES
    domain top — a domain that ends at/below the LES top lets the CBL hit the model
    lid and the temperature collapses.
    """
    if artifact.w_theta_s is None:
        raise ValueError(
            f"{artifact.case_name}: CBL SCM requires a prescribed surface heat flux "
            "(artifact.w_theta_s); this regime is not wired")
    z_les = jnp.asarray(artifact.heights_m)
    if sigma_top is None:
        sigma_top = _auto_sigma_top(float(z_les[-1]))
    grid = _scm_grid(nlev, sigma_top)
    z_scm, p_full = grid.z_scm_m, grid.p_full_pa

    ic = prognostic_truth(artifact)  # full series; [0] is the true t=0 IC
    theta0 = jnp.asarray(ic.theta)[0]
    u0 = jnp.asarray(ic.u)[0]
    v0 = jnp.asarray(ic.v)[0]

    theta_scm = interp_profile(theta0, z_les, z_scm)
    u_scm = interp_profile(u0, z_les, z_scm)
    v_scm = interp_profile(v0, z_les, z_scm)
    T_profile = T_from_theta(theta_scm, p_full)

    q0 = float(jnp.asarray(artifact.w_theta_s)[0])
    # Geostrophic wind forcing (sheared CBL / stable SBL). SCMForcing's Coriolis convention
    # du/dt=+f_c(v-v_g), dv/dt=-f_c(u-u_g) matches the LES emission, and it disables the
    # Coriolis+geostrophic term ENTIRELY when f_c=0 — so a free-convective CBL (f_c=0,
    # Ug=0) is byte-unchanged by this block. u_geo/v_geo are the artifact's LES-grid
    # geostrophic profiles interpolated onto the SCM grid, held constant in time.
    geo_kwargs: dict = {}
    if float(artifact.f_c) != 0.0:
        # f_c != 0 means Coriolis is ON; the geostrophic wind MUST be supplied, else
        # SCMForcing would rotate the wind toward u_g=v_g=0 (a silent wrong physics — the
        # Ekman balance is against the geostrophic wind, not zero). A sheared/stable artifact
        # always records u_geo; its absence is a malformed artifact → fail loudly.
        if artifact.u_geo is None:
            raise ValueError(
                f"{artifact.case_name}: f_c={artifact.f_c:g} != 0 (Coriolis on) but the "
                "artifact has no u_geo. A sheared CBL / stable SBL must record its "
                "geostrophic wind; without it the SCM would spin down toward zero wind, "
                "not the LES geostrophic balance.")
        u_geo_scm = interp_profile(jnp.asarray(artifact.u_geo), z_les, z_scm)
        v_geo_src = (artifact.v_geo if artifact.v_geo is not None
                     else jnp.zeros_like(jnp.asarray(artifact.u_geo)))
        v_geo_scm = interp_profile(jnp.asarray(v_geo_src), z_les, z_scm)
        geo_kwargs = {"u_geo": _const_profile(u_geo_scm),
                      "v_geo": _const_profile(v_geo_scm)}
    forcing = SCMForcing(
        f_c=float(artifact.f_c),
        prescribe="fluxes",
        w_th_s=lambda _t: jnp.asarray(q0),
        **geo_kwargs,
    )
    scm = SingleColumnModel.create(
        physics_config=_dry_cbl_physics(turbulence),
        nlev=nlev,
        dt=dt,
        T_profile=T_profile,
        u=u_scm,
        v=v_scm,
        p_s=_P_S_PA,
        sigma_top=sigma_top,
        forcing=forcing,
        time_integrator="forward_euler",
    )
    return scm, grid


def _to_increasing(profile: Array, z: Array) -> tuple[Array, Array]:
    """Reorder a (possibly top-to-bottom) profile to strictly-increasing height."""
    z = jnp.asarray(z)
    if bool(z[0] < z[-1]):
        return jnp.asarray(profile), z
    return jnp.asarray(profile)[::-1], z[::-1]


def scm_scan_final_state(scm: SingleColumnModel, nsteps: int):
    """Free-run the SCM ``nsteps`` steps via ``lax.scan`` over its pure step; return the
    final ``HydrostaticState``.

    Reproduces ``SingleColumnModel.run`` to within ~1e-9 (well below the tuner's ~1e-3 loss
    tolerance; FP roundoff, not a physics difference) for the wired dry CBL — its forcing is
    time-independent, so ``run``'s per-step diurnal ``set_time`` is a no-op (validated
    allclose vs ``run`` in ``test_scm_runner``). The same forward-Euler pure-step machinery
    extends to any time-independent-forcing regime (e.g. the GABLS1 SBL) once it is wired
    through this path. The win: ONE compiled XLA program instead of ``nsteps`` Python-loop
    dispatches, so a free-run is far faster (a nlev=24 re-score drops from ~720 s to ~3 s)
    and — crucially — tractable under ``jax.grad`` (the Python loop UNROLLS into a giant
    backward graph). The carry is ``(state, phys_state, t_seconds)``; ``phys_state`` is only
    rebound when the physics returns a new one (``None`` → keep the prior, matching
    ``step``); a diverged (NaN) trajectory propagates NaN out, so the caller's finiteness
    guard still fires."""
    dt = scm.dt

    def _body(carry, _):
        state, phys, t = carry
        new_state, new_phys = scm.pure_step(state, phys, t)
        return (new_state, new_phys if new_phys is not None else phys, t + dt), None

    (final_state, _phys, _t), _ = lax.scan(
        _body, (scm.state, scm.phys_state, scm.t_seconds), None, length=nsteps)
    return final_state


def scm_final_theta_on(
    scm: SingleColumnModel, grid: SCMGridSpec, nsteps: int, z_eval: Array
) -> tuple[Array, Array, Array]:
    """Free-run the SCM ``nsteps`` steps; return final (θ, u, v) on ``z_eval``.

    θ is recovered from the SCM temperature via Exner at the SCM pressure, then the
    SCM (top-to-bottom) profile is reordered to increasing height and interpolated
    onto the fixed evaluation grid ``z_eval`` (increasing, LES-native). Uses the
    ``lax.scan`` rollout (:func:`scm_scan_final_state`) — identical to ``run`` to ~1e-9
    (FP roundoff) for the wired dry CBL, but far faster and AD-tractable.
    """
    final_state = scm_scan_final_state(scm, nsteps)
    T_final = jnp.asarray(final_state.T.data[0, 0, 0])
    u_final = jnp.asarray(final_state.u.data[0, 0, 0])
    v_final = jnp.asarray(final_state.v.data[0, 0, 0])
    theta_final = theta_from_temperature(T_final, grid.p_full_pa)

    z_eval = jnp.asarray(z_eval)
    th_inc, z_inc = _to_increasing(theta_final, grid.z_scm_m)
    u_inc, _ = _to_increasing(u_final, grid.z_scm_m)
    v_inc, _ = _to_increasing(v_final, grid.z_scm_m)
    theta_eval = interp_profile(th_inc, z_inc, z_eval)
    u_eval = interp_profile(u_inc, z_inc, z_eval)
    v_eval = interp_profile(v_inc, z_inc, z_eval)
    return theta_eval, u_eval, v_eval


def final_prognostic_truth(artifact: LESReferenceArtifact, z_eval) -> LESTruth:
    """The single-time LES prognostic truth at the artifact's LAST output time, on the
    ``z_eval`` grid. This is the exact target :func:`scm_les_final_loss` scores against
    (the derivative-free tuner's objective); factored so σ_LES scores the SAME final
    snapshot — otherwise a full-series σ_LES would live on a different loss scale and
    could not gate the tuned-loss margins (D7)."""
    truth_series = prognostic_truth(artifact)
    final_truth = LESTruth(
        case_name=artifact.case_name,
        heights_m=jnp.asarray(z_eval),
        times_s=jnp.asarray(truth_series.times_s)[-1][None],
        theta=jnp.asarray(truth_series.theta)[-1],
        u=jnp.asarray(truth_series.u)[-1],
        v=jnp.asarray(truth_series.v)[-1],
        wtheta=jnp.asarray(truth_series.wtheta)[-1],
    )
    return regrid_truth(final_truth, jnp.asarray(z_eval))


def scm_les_final_score(
    artifact: LESReferenceArtifact,
    turbulence: TurbulenceConfig,
    *,
    nlev: int = 32,
    sigma_top: float | None = None,
    dt: float = 5.0,
) -> PrognosticScore | None:
    """Full prognostic score (θ/u/v components + combined) of the free-run SCM final
    profile vs LES truth — the per-variable breakdown behind :func:`scm_les_final_loss`.

    Returns ``None`` for a DIVERGED (non-finite θ/u/v) SCM. The loss maps divergence to
    ``+inf``; a ``None`` here lets a caller that wants the θ component treat divergence
    explicitly instead of reading a sentinel or a spurious ``safe_sqrt(NaN)=0``. Factored
    so the θ-only significance analysis (D7 θ-consistent gate) scores the SAME final
    snapshot with the SAME normalization as the tuner — no duplicated score numerics.
    """
    scm, grid = build_cbl_scm_from_artifact(
        artifact, turbulence, nlev=nlev, sigma_top=sigma_top, dt=dt)
    t_end = float(jnp.asarray(artifact.times_s)[-1])
    nsteps = max(1, int(round(t_end / dt)))

    z_eval = jnp.asarray(artifact.heights_m)
    theta_eval, u_eval, v_eval = scm_final_theta_on(scm, grid, nsteps, z_eval)

    # A DIVERGED SCM (NaN/Inf θ) must NOT score as a perfect fit. The score's safe_sqrt
    # maps NaN -> 0, so a non-finite SCM output would otherwise be selected as best.
    if not (bool(jnp.all(jnp.isfinite(theta_eval)))
            and bool(jnp.all(jnp.isfinite(u_eval)))
            and bool(jnp.all(jnp.isfinite(v_eval)))):
        return None

    final_truth = final_prognostic_truth(artifact, z_eval)
    return prognostic_profile_score(final_truth, theta_eval, u_eval, v_eval)


def scm_les_final_loss(
    artifact: LESReferenceArtifact,
    turbulence: TurbulenceConfig,
    *,
    nlev: int = 32,
    sigma_top: float | None = None,
    dt: float = 5.0,
) -> float:
    """Forward loss: free-run the SCM to the LES end time, score the final profile.

    The std-normalized, mass-weighted RMSE of θ/u/v against LES truth at the last output
    time (both on the LES increasing grid) — the objective the derivative-free tuner
    minimises. A diverged SCM ⇒ ``+inf`` (rejected). Thin wrapper over
    :func:`scm_les_final_score` (which returns the per-variable breakdown).
    """
    score = scm_les_final_score(
        artifact, turbulence, nlev=nlev, sigma_top=sigma_top, dt=dt)
    return float("inf") if score is None else float(score.combined)


def scm_les_loss_jax(
    artifact: LESReferenceArtifact,
    turbulence: TurbulenceConfig,
    *,
    nlev: int = 32,
    sigma_top: float | None = None,
    dt: float = 5.0,
):
    """Differentiable (pure-JAX) form of :func:`scm_les_final_loss` — the AD path's
    objective (D4). Returns a TRACED scalar (no ``float()``/``bool()`` concretization),
    so it can be ``jax.grad``'d w.r.t. traced turbulence-config leaves (params spliced
    in via ``apply_param_overrides`` as jnp arrays). SAME objective as the DF loss (the
    final-snapshot θ/u/v ``prognostic_profile_score.combined`` on the LES eval grid).

    A DIVERGED SCM must NOT score as a perfect (0) fit: the score's ``safe_sqrt`` maps
    NaN→0, so a non-finite θ/u/v would otherwise yield a FINITE ZERO loss the AD optimiser
    selects as "best" (the same hazard the DF path guards with +inf). Guard the LOSS VALUE
    here: a large finite PENALTY selected by a raw-output finiteness flag, scored on
    NaN-sanitised profiles so the FINITE case is bit-unchanged. This fixes the value only —
    a diverged trajectory's GRADIENT may still be NaN (0×NaN backprop through the diverged
    internal states); the AD optimiser's ``isfinite(grad)`` step guard is the second layer
    that stops on that. Only the ``artifact`` (concrete data) uses Python floats for the
    static step count; every config-dependent quantity stays traced.
    """
    scm, grid = build_cbl_scm_from_artifact(
        artifact, turbulence, nlev=nlev, sigma_top=sigma_top, dt=dt)
    t_end = float(jnp.asarray(artifact.times_s)[-1])  # concrete artifact data → static
    nsteps = max(1, int(round(t_end / dt)))
    z_eval = jnp.asarray(artifact.heights_m)
    theta_eval, u_eval, v_eval = scm_final_theta_on(scm, grid, nsteps, z_eval)
    finite = (jnp.isfinite(theta_eval).all() & jnp.isfinite(u_eval).all()
              & jnp.isfinite(v_eval).all())
    final_truth = final_prognostic_truth(artifact, z_eval)
    combined = prognostic_profile_score(
        final_truth, jnp.nan_to_num(theta_eval), jnp.nan_to_num(u_eval),
        jnp.nan_to_num(v_eval)).combined
    return jnp.where(finite, combined,
                     jnp.asarray(_AD_DIVERGE_PENALTY, combined.dtype))
