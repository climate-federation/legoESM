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
forcing is applied so the Ekman/jet dynamics reproduce the LES. Moist regimes (BOMEX
shallow cumulus / DYCOMS stratocumulus) are wired too: an ``is_moist`` artifact drives the
SCM with its large-scale subsidence + θ/q_v advective tendencies + surface moisture flux,
runs diagnostic (Sundqvist) condensation, and is scored in liquid-water potential
temperature θ_l + total water q_t (the variables the LES stores).
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
from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma,
    compute_rho,
    exner_function,
)
from legoesm.atmosphere.physics.turbulence.integration import (
    get_turbulence_fn,
    turbulence_scheme_traits,
)

from legoesm import constants

from .bridge import (
    LESReferenceArtifact,
    LESTruth,
    diagnostic_truth,
    prognostic_truth,
)
from .scm_coupling import (
    T_from_theta,
    interp_profile,
    liquid_water_theta,
    regrid_truth,
    saturation_adjust,
    theta_from_temperature,
)
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
    p_half_pa: Array     # (nlev+1,) half-level pressure [Pa], TOA-first
    nlev: int


def _scm_grid(nlev: int, sigma_top: float) -> SCMGridSpec:
    from legoesm.grids.vertical import create_sigma_coordinate

    sigma_coord = create_sigma_coordinate(nlev, sigma_top=sigma_top)
    p_full = sigma_coord.sigma_full * _P_S_PA
    p_half = sigma_coord.sigma_half * _P_S_PA
    z_scm = -_H_SCALE_M * jnp.log(jnp.maximum(p_full / _P_S_PA, 1e-6))
    return SCMGridSpec(z_scm_m=z_scm, p_full_pa=p_full, p_half_pa=p_half, nlev=nlev)


def _dry_cbl_physics(turbulence: TurbulenceConfig) -> PhysicsConfig:
    """Turbulence-only physics config (all other modules off) for a dry CBL."""
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=turbulence,
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def _moist_cbl_physics(turbulence: TurbulenceConfig) -> PhysicsConfig:
    """Turbulence + diagnostic (Sundqvist) condensation for a moist shallow-cumulus SCM.

    Sundqvist is the large-scale diagnostic condensation scheme (subgrid RH>RH_crit
    partial-cloud-fraction) appropriate at the SCM's coarse column resolution — it converts
    supersaturated q_v to q_c with the latent-heating feedback on T, so the SCM reproduces a
    cloud-topped moist BL (BOMEX/DYCOMS). Radiation/convection/GWD stay off (the tuned
    turbulence closure is the object under study, per the dry regimes)."""
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=turbulence,
        microphysics=MicrophysicsConfig(scheme="sundqvist"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def _const_profile(profile):
    """A time-constant SCMForcing ``ProfileFn`` returning ``profile`` for every ``t`` — used
    for the steady geostrophic wind of the sheared CBL / stable SBL."""
    return lambda _t: profile


def _const_scalar(value):
    """A time-constant SCMForcing ``ScalarFn`` returning ``value`` for every ``t`` (moist
    surface moisture flux ``w_qv_s``)."""
    return lambda _t: jnp.asarray(value)


def _liquid_water_theta(theta: Array, q_c: Array, p: Array) -> Array:
    """SCM-side θ_l: compute the Exner factor from the SCM pressure, then apply the ONE
    canonical reduction :func:`scm_coupling.liquid_water_theta` (shared with the LES recorder
    so θ_l(SCM) and θ_l(LES) use the same formula — no re-derivation). Π=(p/p_ref)^κ."""
    exner = (jnp.asarray(p) / constants.p_ref) ** constants.kappa
    return liquid_water_theta(theta, q_c, exner)


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
    # Moist regime (BOMEX/DYCOMS shallow-cumulus / stratocumulus). q_t IC (q_c≈0 at t=0 ⇒
    # q_v(t0)=q_t(t0)), plus the artifact's large-scale forcing: subsidence_w [m/s, +up],
    # the prescribed θ/q_v advective tendencies, and the surface moisture flux w_qv_s
    # [(kg/kg) m/s, +up]. These are the SAME sign conventions SCMForcing documents
    # (subsidence_w +up, prescribe='fluxes' fluxes +up), which is how the moist artifact
    # stores them — no sign flip here. Diagnostic Sundqvist condensation (_moist_cbl_physics)
    # then forms q_c so the SCM reproduces a cloud-topped moist BL. A dry artifact skips this
    # block entirely (byte-unchanged).
    q_v_scm = None
    q_c_scm = None
    moist_kwargs: dict = {}
    if artifact.is_moist:
        # The moist artifact records LIQUID-WATER potential temperature θ_l (in
        # `artifact.theta`) and TOTAL water q_t (in `artifact.qt`). Reading θ_l as
        # a plain θ and q_t as vapour is only right for a cloud-free start; a
        # stratocumulus column carries cloud at t=0 and would start too cool and
        # too moist, condensing toward a state the reference never had. Initialise
        # from the cloud water the reference ACTUALLY carried, so total water
        # q_t = q_v + q_c is conserved and the SCM's initial θ_l/q_t reproduce the
        # artifact (θ = θ_l + (L_v/(c_pd·Π))·q_c is the exact inverse of
        # scm_coupling.liquid_water_theta):
        #   • if the artifact recorded its cloud-water channel `qc`, use it
        #     DIRECTLY. A horizontally averaged column can sit BELOW saturation in
        #     the mean while still carrying cloud (partial cover), so the recorded
        #     q_c — not a grid-mean saturation test — is the correct evidence.
        #   • otherwise (older artifact, no channel) fall back to a saturation
        #     adjustment (θ_l,q_t)→(θ,q_v,q_c), exact only for a fully-saturated
        #     grid mean but the best available.
        # A clear column gives q_c=0, θ=θ_l, q_v=q_t either way ⇒ BOMEX unchanged.
        qt0 = interp_profile(jnp.asarray(artifact.qt)[0], z_les, z_scm)
        exner = exner_function(p_full)
        if artifact.qc is not None:
            q_c_adj = jnp.clip(
                interp_profile(jnp.asarray(artifact.qc)[0], z_les, z_scm), 0.0, None)
            theta_adj = theta_scm + constants.L_v / (constants.c_pd * exner) * q_c_adj
            q_v_scm = qt0 - q_c_adj
        else:
            theta_adj, q_v_scm, q_c_adj = saturation_adjust(theta_scm, qt0, exner, p_full)
        T_profile = theta_adj * exner            # actual temperature (θ_l→θ warmed by q_c)
        if float(jnp.max(q_c_adj)) > 0.0:
            q_c_scm = q_c_adj
        moist_kwargs["w_qv_s"] = _const_scalar(
            float(jnp.asarray(artifact.w_qv_s)[0]))
        for fld in ("subsidence_w", "theta_adv", "qv_adv"):
            prof = getattr(artifact, fld)
            if prof is not None:
                moist_kwargs[fld] = _const_profile(
                    interp_profile(jnp.asarray(prof), z_les, z_scm))
    physics = (_moist_cbl_physics(turbulence) if artifact.is_moist
               else _dry_cbl_physics(turbulence))
    forcing = SCMForcing(
        f_c=float(artifact.f_c),
        prescribe="fluxes",
        # NAME this rather than inherit it. The default puts the prescribed
        # surface flux in as a tendency on the lowest cell, and the turbulence
        # closure then sees a surface heat flux of exactly zero. That is the
        # defining input of every nonlocal closure, and comparing nine closures
        # against large-eddy truth is the whole purpose here, so the nonlocal
        # ones would have been ranked on a boundary condition they never got.
        flux_to_closure=True,
        w_th_s=lambda _t: jnp.asarray(q0),
        **geo_kwargs,
        **moist_kwargs,
    )
    scm = SingleColumnModel.create(
        physics_config=physics,
        nlev=nlev,
        dt=dt,
        T_profile=T_profile,
        q_v_profile=q_v_scm,
        q_c_profile=q_c_scm,
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


# The diagnostic flux F = -Kh·(∂θ/∂z − γ) is a pure state functional: Kh and γ depend
# only on the mean-state gradients/stability, NOT on dt (dt enters only the implicit
# flux-DIVERGENCE solve, which the diagnostic ignores).  Any positive dt yields the same
# wtheta_flux, so this placeholder is passed purely to satisfy the scheme signature.
_DIAG_DT_S = 1.0
# Secant tolerance / cap for the surface-flux calibration below.  A dry CBL surface
# kinematic heat flux ~0.02–0.10 K m/s; matching it to 1e-4 K m/s (~0.1 W/m²) is far
# tighter than the σ_LES flux spread the score is gated against.
_KHFS_TOL = 1.0e-4
_KHFS_MAX_ITERS = 40
_TSFC_EXCESS_CAP_K = 60.0  # a super-adiabatic surface excess ceiling (fail loud beyond)


def _surface_theta_flux(
    fn, cfg, base_inputs, rho_sfc: float, exner_sfc: float, dtheta_sfc_K: float
) -> float:
    """The scheme's diagnosed surface kinematic POTENTIAL-temperature flux ⟨w'θ'⟩ [K m/s].

    ``base_inputs`` is the frozen ``(u,v,T,q_v,p_full,p_half,z_full,z_half,rho)`` tuple;
    only the surface temperature varies. Two fidelity points (codex #1/#2):
      * ``q_sfc = q_v`` at the lowest level (dry air ⇒ 0) SUPPRESSES surface latent
        exchange, so the closure sees ONLY the dry sensible forcing the LES prescribed —
        otherwise a saturated ``q_sfc`` injects a moisture-buoyancy flux the artifact never
        had, which HB/YSU fold into ``kbfs`` and which would corrupt the margin.
      * ``shflx/(ρ c_pd)`` is the kinematic *temperature* flux ⟨w'T'⟩; the LES target
        ``w_theta_s`` is a *potential-temperature* flux, so divide by the surface Exner
        ``Π_sfc`` to convert ⟨w'T'⟩ → ⟨w'θ'⟩.
    Host-side float — runs on a diagnostic snapshot, never in a traced loop.
    """
    (u2, v2, T2, qv2, pf2, ph2, z_full, z_half, rho) = base_inputs
    T_sfc = T2[:, -1] + dtheta_sfc_K
    q_sfc = qv2[:, -1]  # dry: q_sfc = q_v_air ⇒ no latent surface flux
    out = fn(u2, v2, T2, qv2, pf2, ph2, z_full, z_half, T_sfc, q_sfc, rho, _DIAG_DT_S, cfg)
    return float(out.shflx[0]) / (rho_sfc * constants.c_pd * exner_sfc)


def _calibrate_surface_excess(
    fn, cfg, base_inputs, rho_sfc: float, exner_sfc: float, target_wtheta: float
) -> float:
    """Solve the T_sfc excess [K] making the scheme's surface ⟨w'θ'⟩ = ``target_wtheta``.

    The LES prescribed a surface *kinematic heat flux* (``artifact.w_theta_s``); the bulk
    turbulence surface layer instead diagnoses it from ``T_sfc − T_air``. To evaluate a
    closure's INTERIOR flux under the SAME surface forcing the LES had (the controlled
    variable held fixed across closures), secant-solve the surface excess so the diagnosed
    ⟨w'θ'⟩ matches the LES value. The bulk sensible flux ``∝ Ch·|U|·(T_sfc − T_air)`` is
    monotone increasing in the excess for the constant-exchange surface layer the CBL SCM
    uses, so the secant converges in a few steps; a non-convergent / capped case raises
    rather than silently returning a mismatched surface BC.
    """
    a, b = 0.0, 8.0
    fa = _surface_theta_flux(fn, cfg, base_inputs, rho_sfc, exner_sfc, a) - target_wtheta
    fb = _surface_theta_flux(fn, cfg, base_inputs, rho_sfc, exner_sfc, b) - target_wtheta
    for _ in range(_KHFS_MAX_ITERS):
        if abs(fb) < _KHFS_TOL:
            return b
        denom = fb - fa
        if denom == 0.0:
            break
        c = b - fb * (b - a) / denom
        c = min(max(c, 0.0), _TSFC_EXCESS_CAP_K)
        a, fa = b, fb
        b = c
        fb = _surface_theta_flux(fn, cfg, base_inputs, rho_sfc, exner_sfc, c) - target_wtheta
    if abs(fb) < _KHFS_TOL:
        return b
    raise ValueError(
        f"surface-flux calibration did not converge to w'θ'={target_wtheta:.4g} K m/s "
        f"(residual {fb:.3g} after {_KHFS_MAX_ITERS} secant steps, excess capped at "
        f"{_TSFC_EXCESS_CAP_K} K) — the mean-state wind/gustiness may be too weak to carry "
        "this surface flux (free-convective CBL with ~zero mean wind is not calibratable)")


def diagnostic_scheme_flux(
    artifact: LESReferenceArtifact,
    turbulence: TurbulenceConfig,
    *,
    nlev: int = 32,
    sigma_top: float | None = None,
) -> Array:
    """One closure's diagnosed heat flux ``⟨w'θ'⟩`` at the LES mean state, on the LES grid.

    The D6 *diagnostic* score (LES_SUITE.md Q1b): set the SCM column to the LES
    most-equilibrated snapshot (``bridge.diagnostic_truth`` — θ→T on the σ grid, u/v
    interpolated, dry ``q_v=0``), impose the SAME surface kinematic heat flux the LES had
    (``artifact.w_theta_s``, via a T_sfc excess calibrated through the shared bulk surface
    layer — see :func:`_calibrate_surface_excess`; the prescribed-flux CBL otherwise runs
    the turbulence surface layer at ``Ch=0`` and STARVES the nonlocal counter-gradient of
    its driving buoyancy flux, which would make the margin a wiring artifact, not physics),
    call the closure ONCE, and read its ``TurbulenceOutput.wtheta_flux``
    (``F = −Kh·(∂θ/∂z − γ)``; ``γ=0`` for a local closure, the scheme's counter-gradient
    for a nonlocal one). The flux is regridded to the LES height grid so
    ``score.diagnostic_flux_score`` can compare it to the LES total flux at the same instant.

    A local down-gradient closure can only oppose the resolved gradient, so through a
    well-mixed CBL (``∂θ/∂z ≈ 0``) it carries ≈0 interior flux however strong the surface
    forcing; a nonlocal closure's counter-gradient carries the surface flux up through the
    mixed layer. The measured local-vs-nonlocal RMSE gap this feeds is the Q1b diagnostic
    margin (the counter-gradient structural ceiling of Q1a, now quantified against the LES).

    Interior flux is a pure state functional (dt-independent — see ``_DIAG_DT_S``). Dry
    prescribed-flux regimes only (CBL / sheared CBL / SBL): the counter-gradient ceiling is
    a dry-CBL notion, so a moist artifact is rejected. Restricted to the four K-closures
    that expose ``wtheta_flux`` (smagorinsky/louis local, holtslag_boville/ysu nonlocal); a
    TKE-carrying or non-exposing scheme raises.
    """
    if artifact.w_theta_s is None:
        raise ValueError(
            f"{artifact.case_name}: the Q1 diagnostic flux needs a dry prescribed-flux "
            "artifact (artifact.w_theta_s); this regime is not wired")
    truth = diagnostic_truth(artifact)  # final snapshot: θ/u/v/wtheta on increasing z
    if truth.wqt is not None:
        raise ValueError(
            f"{artifact.case_name}: the Q1 counter-gradient diagnostic is dry-only; a "
            "moist artifact (wqt set) is not supported")
    z_les = jnp.asarray(truth.heights_m)
    if sigma_top is None:
        sigma_top = _auto_sigma_top(float(z_les[-1]))
    grid = _scm_grid(nlev, sigma_top)
    z_scm, p_full, p_half = grid.z_scm_m, grid.p_full_pa, grid.p_half_pa

    theta_scm = interp_profile(jnp.asarray(truth.theta), z_les, z_scm)
    u_scm = interp_profile(jnp.asarray(truth.u), z_les, z_scm)
    v_scm = interp_profile(jnp.asarray(truth.v), z_les, z_scm)
    T = T_from_theta(theta_scm, p_full)

    # Model-convention (1, nlev) columns, TOA-first, matching the scheme signature.
    T2 = T[None]
    qv2 = jnp.zeros_like(T2)          # dry CBL
    u2, v2 = u_scm[None], v_scm[None]
    pf2, ph2 = p_full[None], p_half[None]
    z_full, z_half = compute_heights_from_sigma(T2, ph2)
    rho = compute_rho(T2, pf2)

    name, fn, cfg = get_turbulence_fn(turbulence)
    if fn is None:
        raise ValueError("turbulence scheme 'none' exposes no diagnostic flux")
    if turbulence_scheme_traits(name).carries_energy:
        raise ValueError(
            f"{name}: TKE-carrying schemes are not wired for the Q1 diagnostic flux "
            "(only the four K-closures expose wtheta_flux)")

    # Impose the LES surface kinematic heat flux (the controlled variable) on this closure.
    base_inputs = (u2, v2, T2, qv2, pf2, ph2, z_full, z_half, rho)
    rho_sfc = float(rho[0, -1])
    exner_sfc = float(exner_function(pf2[:, -1])[0])  # Π at the lowest full level
    target_wtheta = float(jnp.asarray(artifact.w_theta_s)[-1])
    excess = _calibrate_surface_excess(
        fn, cfg, base_inputs, rho_sfc, exner_sfc, target_wtheta)
    T_sfc = T2[:, -1] + excess
    q_sfc = qv2[:, -1]  # dry: suppress latent surface exchange (matches the calibration)

    out = fn(
        u2, v2, T2, qv2, pf2, ph2, z_full, z_half,
        T_sfc, q_sfc, rho, _DIAG_DT_S, cfg,
    )
    if out.wtheta_flux is None:
        raise ValueError(
            f"{name}: does not expose TurbulenceOutput.wtheta_flux; the Q1 diagnostic "
            "flux is wired only for smagorinsky/louis/holtslag_boville/ysu")
    # Regrid on the SAME hydrostatic heights z_full the scheme computed the flux on
    # (NOT the isothermal z_scm proxy used for the θ IC interp — codex #3).
    flux_inc, z_inc = _to_increasing(out.wtheta_flux[0], z_full[0])  # → increasing height
    return interp_profile(flux_inc, z_inc, z_les)  # on the LES grid, for scoring


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


def scm_final_moist_on(
    scm: SingleColumnModel, grid: SCMGridSpec, nsteps: int, z_eval: Array
) -> tuple[Array, Array, Array, Array]:
    """Free-run a MOIST SCM ``nsteps`` steps; return final (θ_l, u, v, q_t) on ``z_eval``.

    The moist LES stores its thermodynamic profile as liquid-water potential temperature θ_l
    and total water q_t (``les_record``), so the SCM must be reduced to the SAME variables for
    an apples-to-apples score (controlled-comparison rule). From the SCM's temperature/tracer
    state:

      θ_l = θ − (L_v / (c_pd · Π)) · q_c      (Betts 1973; Π = (p/p_ref)^κ, the Exner function)
      q_t = q_v + q_c + q_r                   (the water species the LES q_t sums)

    Sign of the θ_l reduction: condensation RELEASES latent heat and raises θ above θ_l, so
    θ_l = θ − (positive) < θ wherever cloud (q_c>0) exists — θ_l is the cloud-conserved
    variable the LES reports. Units: (L_v[J/kg]/c_pd[J/kg/K])·q_c[kg/kg]/Π[–] = K. The SCM
    (top-to-bottom) profiles are reordered to increasing height and interpolated onto the
    fixed LES eval grid. Uses the same ``lax.scan`` rollout as the dry path.
    """
    final_state = scm_scan_final_state(scm, nsteps)
    T_final = jnp.asarray(final_state.T.data[0, 0, 0])
    u_final = jnp.asarray(final_state.u.data[0, 0, 0])
    v_final = jnp.asarray(final_state.v.data[0, 0, 0])
    theta_final = theta_from_temperature(T_final, grid.p_full_pa)

    tracers = final_state.tracers
    q_v = jnp.asarray(tracers["q_v"].data[0, 0, 0])
    q_c = (jnp.asarray(tracers["q_c"].data[0, 0, 0])
           if "q_c" in tracers else jnp.zeros_like(q_v))
    q_r = (jnp.asarray(tracers["q_r"].data[0, 0, 0])
           if "q_r" in tracers else jnp.zeros_like(q_v))
    theta_l_final = _liquid_water_theta(theta_final, q_c, grid.p_full_pa)
    qt_final = q_v + q_c + q_r

    z_eval = jnp.asarray(z_eval)
    thl_inc, z_inc = _to_increasing(theta_l_final, grid.z_scm_m)
    u_inc, _ = _to_increasing(u_final, grid.z_scm_m)
    v_inc, _ = _to_increasing(v_final, grid.z_scm_m)
    qt_inc, _ = _to_increasing(qt_final, grid.z_scm_m)
    return (interp_profile(thl_inc, z_inc, z_eval),
            interp_profile(u_inc, z_inc, z_eval),
            interp_profile(v_inc, z_inc, z_eval),
            interp_profile(qt_inc, z_inc, z_eval))


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
        qt=None if truth_series.qt is None else jnp.asarray(truth_series.qt)[-1],
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
    # Moist artifacts are scored in θ_l + q_t (the LES thermodynamic variables); dry artifacts
    # in θ only. The dry path keeps calling scm_final_theta_on unchanged (its divergence
    # monkeypatch test keys off that symbol).
    if artifact.is_moist:
        theta_eval, u_eval, v_eval, qt_eval = scm_final_moist_on(
            scm, grid, nsteps, z_eval)
    else:
        theta_eval, u_eval, v_eval = scm_final_theta_on(scm, grid, nsteps, z_eval)
        qt_eval = None

    # A DIVERGED SCM (NaN/Inf θ) must NOT score as a perfect fit. The score's safe_sqrt
    # maps NaN -> 0, so a non-finite SCM output would otherwise be selected as best.
    if not (bool(jnp.all(jnp.isfinite(theta_eval)))
            and bool(jnp.all(jnp.isfinite(u_eval)))
            and bool(jnp.all(jnp.isfinite(v_eval)))
            and (qt_eval is None or bool(jnp.all(jnp.isfinite(qt_eval))))):
        return None

    final_truth = final_prognostic_truth(artifact, z_eval)
    return prognostic_profile_score(
        final_truth, theta_eval, u_eval, v_eval, scm_qt=qt_eval)


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
    final-snapshot ``prognostic_profile_score.combined`` on the LES eval grid) — θ/u/v for a
    dry artifact, θ_l/u/v/q_t for a moist one (same is_moist branch as
    :func:`scm_les_final_score`).

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
    # Mirror the DF branch: moist artifacts are reduced to θ_l/u/v/q_t (final_prognostic_truth
    # now carries qt for them, so prognostic_profile_score REQUIRES scm_qt or raises). finite
    # stays a TRACED jnp predicate (& not Python bool) so this remains jax.grad-able.
    if artifact.is_moist:
        theta_eval, u_eval, v_eval, qt_eval = scm_final_moist_on(
            scm, grid, nsteps, z_eval)
        qt_safe = jnp.nan_to_num(qt_eval)
        finite = (jnp.isfinite(theta_eval).all() & jnp.isfinite(u_eval).all()
                  & jnp.isfinite(v_eval).all() & jnp.isfinite(qt_eval).all())
    else:
        theta_eval, u_eval, v_eval = scm_final_theta_on(scm, grid, nsteps, z_eval)
        qt_safe = None
        finite = (jnp.isfinite(theta_eval).all() & jnp.isfinite(u_eval).all()
                  & jnp.isfinite(v_eval).all())
    final_truth = final_prognostic_truth(artifact, z_eval)
    combined = prognostic_profile_score(
        final_truth, jnp.nan_to_num(theta_eval), jnp.nan_to_num(u_eval),
        jnp.nan_to_num(v_eval), scm_qt=qt_safe).combined
    return jnp.where(finite, combined,
                     jnp.asarray(_AD_DIVERGE_PENALTY, combined.dtype))
