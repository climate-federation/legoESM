"""Canonical Kuo (1965) convection scheme — faithful to Mahfouf's oracle.

Faithful JAX port of J.-F. Mahfouf's reference Kuo implementation
(AJFMAHFOUF/MOIST_CONVECTION_KUO, ``src/kuo_schemes.f90``, variant
``kuo65`` with the optional Kuo-Anthes 1977 partition).

The whole point of Kuo's parameterization
-----------------------------------------
Kuo's moisture/heat source is the **large-scale moisture convergence**

    cvgu = Σ_{icond==2} max(0, ∂q/∂t|dyn) · dp/g          [kg m^-2 s^-1]

a *bounded external dynamical tendency* supplied by the resolved flow.
The scheme redistributes that converged moisture into convective heating
and moistening and explicitly REMOVES the large-scale tendency it
consumes (the ``−ptenq`` term):

    dt/dt = cvgu/zint · (tc − t)                          (Kuo-1965)
    dq/dt = −ptenq + cvgu/zint · (qvc − qv)
    zint  = Σ_{icond==2} (qvc − qv + (tc − t)/alpha) · dp/g

where ``(tc, qvc)`` is the entraining moist-adiabat cloud profile and
``alpha = L_v / c_p``.  Convection is active only at levels that are
buoyant AND have positive vertical velocity at the LCL (``icond==2``)
AND have positive local convergence (``ptenq > 0``).

This module is QUIESCENT when no large-scale convergence is supplied
(``moisture_convergence is None`` → ``ptenq ≡ 0`` → ``cvgu = 0`` → zero
tendencies).  That is the *physically correct* behavior, not a bug: a
single column with no resolved large-scale ascent has nothing for Kuo to
converge.  In particular a pure single-column RCE leaves Kuo off, and
the column equilibrates under radiation + turbulence + grid-scale
saturation alone.  (The previous implementation invented the source from
column supersaturation ``Σ max(qv − qsat, 0)``, which is *not* Kuo and
ran away in fixed-SST RCE.)

The ``moisture_convergence`` argument is the per-level large-scale
``∂q/∂t|dyn`` [kg/kg/s], positive = inflow.  The convection bridge
(``convection/integration.py``) threads it in exactly the way it does
for Bechtold (``compute_moisture_convergence`` = ``−∇·(q_v u)``).

Differentiability and positivity
--------------------------------
All hard ``if`` switches in the oracle (LCL condensation, buoyancy,
``w_lcl>0``, ``ptenq>0``) are replaced by smooth sigmoid gates
(``config.icond_sharpness``) so the scheme is fully ``jax.grad`` / ``jit``
/ ``vmap`` compatible.  The entraining parcel ascent is a ``lax.scan``
from the surface upward.  The net vapor tendency is bounded so a single
forward-Euler step preserves ``q_v >= 0`` (see ``_apply_qv_floor``), and
the convective cloud-water source handed to microphysics is
non-negative by construction.

References
----------
- Kuo, H. L. (1965). On formation and intensification of tropical
  cyclones through latent heat release by cumulus convection.
  J. Atmos. Sci., 22, 40-63.
- Anthes, R. A. (1977). A cumulus parameterization scheme utilizing a
  one-dimensional cloud model. Mon. Wea. Rev., 105, 270-286.
- Mahfouf, J.-F. MOIST_CONVECTION_KUO (reference Fortran oracle).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import (
    saturation_vapor_pressure,
    saturation_specific_humidity,
    saturation_specific_humidity_dT,
)
from legoesm.atmosphere.physics.thermodynamics import (
    compute_moist_adiabat,
    compute_cape,
    latent_heat_vaporization,
)
from legoesm.atmosphere.physics.convection.config import KuoConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.convection._triggers import smooth_step


__all__ = ("kuo_convection",)


__physics_contract__ = {
    "summary": (
        "Kuo (1965) convection driven by large-scale moisture convergence: "
        "the converged moisture is redistributed into convective heating and "
        "moistening (optional Kuo-Anthes 1977 partition); quiescent when no "
        "convergence is supplied. Smooth (differentiable) gates."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa", "dt": "s",
        "moisture_convergence": "kg/kg/s (large-scale dq/dt|dyn source; None -> quiescent)",
        "w_grid": "m/s (resolved ascent for the w_lcl>0 gate; optional)",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "dq_c_conv_dt": "kg/kg/s (convective cloud-water source to microphysics, >=0)",
        "cape": "J/kg", "convective_mask": "1 (0-1 activation)",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Where a column is buoyant with positive "
        "convergence, convection warms toward the cloud profile (tc>T) and "
        "moistens (qvc>qv), and explicitly REMOVES the consumed large-scale "
        "tendency (-ptenq). Latent heating is tied to condensation by the "
        "column's own alpha=L_v/c_p (c_p*dT = L_v*dq_c), so column "
        "enthalpy+latent (moist static energy) is conserved; atmospheric "
        "moisture is NOT conserved (source-driven by external convergence; "
        "condensate handed to microphysics)."
    ),
    # Source-driven: the scheme consumes/removes an EXTERNAL large-scale
    # moisture-convergence tendency and can uniformly rescale tendencies with a
    # positivity limiter, so this is not closed-column energy conservation ->
    # no contract-level conservation claimed.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Kuo (1965), J. Atmos. Sci. 22, 40-63; "
        "Anthes (1977), Mon. Wea. Rev. 105, 270-286"
    ),
    "idealized_test": (
        "tests/unit/test_kuo_oracle_faithful.py; moisture_convergence=None -> "
        "zero tendencies (quiescent single column); a conditionally-unstable "
        "column with positive large-scale convergence -> heating + moistening "
        "with a positive cloud-water source and convective_mask>0; a "
        "forward-Euler step keeps q_v>=0."
    ),
}


# Virtual-temperature coefficient ``T_v = T·(1 + _VT_COEFF·q_v − q_c)``,
# derived from constants (``R_v/R_d − 1 = 1/ε − 1 ≈ 0.608``) rather than
# the bare 0.608 literal the oracle hard-codes (Codex review-1 finding #6).
_VT_COEFF = 1.0 / constants.epsilon - 1.0


# ----------------------------------------------------------------------------
# Thermodynamic helpers (legoesm constants + thermo; no re-derived saturation)
# ----------------------------------------------------------------------------

def _latent_heat(T: jax.Array) -> jax.Array:
    """Temperature-dependent latent heat of vaporization [J/kg].

    Oracle ``Lh``: ``L_v0 + (T − T00)·(c_pv − c_l)`` with the reference
    temperature taken as :data:`constants.T_freeze` (the oracle uses the
    triple point ``t00 = 273.16``; ``T_freeze`` sits 0.01 K below, which
    shifts ``Lh`` by ~25 J/kg, negligible).  Sublimation (``lsub``) is
    disabled in the oracle's Kuo path, so we keep the vaporization
    branch only.

    Thin named alias over the shared Kirchhoff helper
    :func:`legoesm.atmosphere.physics.thermodynamics.latent_heat_vaporization`
    (same pattern as :func:`_dqsat_dT` below) so the Kuo call sites read
    like the oracle while the formula lives in one place.
    """
    return latent_heat_vaporization(T)


def _dqsat_dT(T: jax.Array, p: jax.Array) -> jax.Array:
    """d(q_sat)/dT [1/K] for the legoesm Tetens saturation used here.

    Thin wrapper over :func:`legoesm.thermo.saturation_specific_humidity_dT`,
    which is the exact analytic ``d/dT`` of the mandated
    :func:`legoesm.thermo.saturation_vapor_pressure` Tetens curve with the
    hard ``max(p − e_sat, 1)`` floor the Newton solve wants (no smooth
    softplus surrogate). Kept as a named alias so the Kuo call sites read
    physically; the saturation prefactors live solely in ``thermo`` so the
    slope cannot drift from the model's own ``e_sat`` curve.
    """
    return saturation_specific_humidity_dT(T, p)


# ----------------------------------------------------------------------------
# Entraining moist-adiabat parcel ascent (faithful to the oracle loop)
# ----------------------------------------------------------------------------

def _parcel_ascent(
    T: jax.Array,        # (ncol, nlev) env temperature, top->surface? see note
    q_v: jax.Array,      # (ncol, nlev) env vapor
    p_full: jax.Array,   # (ncol, nlev)
    gz: jax.Array,       # (ncol, nlev) geopotential [m^2/s^2]
    tve: jax.Array,      # (ncol, nlev) env virtual temperature
    w: jax.Array,        # (ncol, nlev) vertical velocity proxy at levels
    config: KuoConfig,
):
    """Reproduce the oracle's entraining-parcel loop, differentiably.

    Arrays are in legoesm column ordering: index 0 = model top, index
    ``nlev-1`` = surface (matching the moist-adiabat / saturation code).
    The parcel launches from the surface (``[:, -1]``) and ascends to the
    top, so we scan over levels reversed to surface-first.

    Returns ``(tc, qvc, icond2)`` each (ncol, nlev), where ``tc`` and
    ``qvc`` are the in-cloud temperature / vapor and ``icond2`` is the
    smooth [0,1] activation gate (oracle ``icond==2`` = buoyant AND
    ``w_lcl>0`` AND condensing).  Outside the scanned range and where the
    parcel is not condensing, ``tc=T`` and ``qvc=q_v`` (cloud = environment,
    so ``tc−t`` and ``qvc−qv`` vanish), matching the oracle initialisation.
    """
    ncol, nlev = T.shape
    sharp = config.icond_sharpness
    eps_entr = config.entrainment
    c_pv = constants.c_pv
    c_pd = constants.c_pd
    c_pw = constants.c_pw
    g = constants.g

    # Normalise the vertical-velocity proxy to its column-max magnitude so
    # the ``w_lcl > 0`` gate is a crisp SIGN detector (the oracle gates on
    # ``w_lcl > 0.0``, a Heaviside on the sign — the raw magnitude of the
    # ``ptenq`` proxy, ~1e-8, would otherwise sit a dimensionless sigmoid
    # at its 0.5 mid-point and halve every active level).
    w_scale = jnp.max(jnp.abs(w), axis=1, keepdims=True)        # (ncol, 1)
    w_norm = w / jnp.maximum(w_scale, 1e-30)                    # in [-1, 1]

    # Surface-first ordering for the scan.
    T_s = T[:, ::-1]
    q_s = q_v[:, ::-1]
    p_s = p_full[:, ::-1]
    gz_s = gz[:, ::-1]
    tve_s = tve[:, ::-1]
    w_s = w_norm[:, ::-1]

    def cps_of(qv):
        return c_pd * (1.0 - qv) + qv * c_pv

    # Stable scan-carry dtype = the input state dtype.
    carry_dtype = T.dtype

    # Launch parcel from the surface level (index 0 in surface-first).
    T_base = T_s[:, 0]
    qv_base = q_s[:, 0]
    Cps0 = cps_of(qv_base)
    s1_0 = Cps0 * T_base + gz_s[:, 0] + _latent_heat(T_base) * qv_base

    # Carry: (s1, Cps, qv1, t_prev, gz_prev, w_lcl, prev_icond, prev2_icond)
    # ``prev_icond``/``prev2_icond`` reproduce the oracle's
    # ``icond(k)==0 .and. icond(k+1)==0`` first-LCL capture of ``w_lcl``.
    init = tuple(
        c.astype(carry_dtype) for c in (
            s1_0, Cps0, qv_base,
            T_base, gz_s[:, 0],
            jnp.zeros(ncol, dtype=carry_dtype),    # w_lcl
            jnp.zeros(ncol, dtype=carry_dtype),    # prev_icond
            jnp.zeros(ncol, dtype=carry_dtype),    # prev2_icond
        )
    )

    def step(carry, inp):
        s1, Cps, qv1, t_prev, gz_prev, w_lcl, prev_icond, prev2_icond = carry
        T_k, q_k, p2, gz2, tve_k, w_k = inp

        dz = (gz_prev - gz2) / g  # thickness [m]; gz decreases upward

        # Dry initial guess (oracle: t2 = (s1 - gz2 - Lh*qv1)/Cps).
        t2 = (s1 - gz2 - _latent_heat(t_prev) * qv1) / Cps
        t1 = t2  # save the "no saturation" temperature

        # Entraining moist conservation residual target.
        Cps_env_k = cps_of(q_k)
        zlam = (s1 - gz2) - eps_entr * dz * (
            Cps_env_k * T_k + _latent_heat(T_k) * q_k
        )

        # Newton iterations for the saturated parcel temperature.
        def newton_body(_, t2v):
            qs = saturation_specific_humidity(t2v, p2)
            dqs = _dqsat_dT(t2v, p2)
            Lh = _latent_heat(t2v)
            f = (Cps * t2v + Lh * qs) * (1.0 - eps_entr * dz) - zlam
            df = (
                Cps + Lh * dqs + (c_pv - c_pd) * dqs * t2v
                + (c_pv - c_pw) * qs
            ) * (1.0 - eps_entr * dz)
            return t2v - f / df

        t2 = jax.lax.fori_loop(0, config.newton_iters, newton_body, t2)

        qsat_k = saturation_specific_humidity(t2, p2)
        # Condensation gate (oracle: qv2 > qsat(p2,t2) AND qv2 > qv_min).
        # Arguments normalised to O(1) so a single dimensionless sharpness
        # gives a crisp Heaviside.
        cond_gate = (
            smooth_step((qv1 - qsat_k) / config.supersat_scale, sharp)
            * smooth_step((qv1 - config.qv_min) / config.qv_min, sharp)
        )

        # In-cloud values: condensing → (t2, qsat); else dry → (t1, qv1).
        qc = jnp.maximum(qv1 - qsat_k, 0.0) * cond_gate
        t_cloud = cond_gate * t2 + (1.0 - cond_gate) * t1
        qv_cloud = cond_gate * qsat_k + (1.0 - cond_gate) * qv1

        # Virtual cloud temperature with water loading.
        tvc = t_cloud * (1.0 + _VT_COEFF * qv_cloud - qc)
        buoyant = smooth_step((tvc - tve_k) / config.buoyancy_scale_K, sharp)

        # ``w_lcl`` captured at the FIRST condensing level (oracle:
        # icond(k)==0 .and. icond(k+1)==0 → w_lcl = w(k)).  Smoothly:
        # this is the first level where cond_gate turns on, i.e. neither
        # of the two levels below were active.
        first_lcl = cond_gate * (1.0 - prev_icond) * (1.0 - prev2_icond)
        w_lcl_new = first_lcl * w_k + (1.0 - first_lcl) * w_lcl

        # icond==2: condensing AND buoyant AND w_lcl > 0.
        w_pos = smooth_step(w_lcl_new, sharp)
        icond2 = cond_gate * buoyant * w_pos

        # Advance carry: new MSE from the condensing branch (oracle uses
        # the updated t2/qv1 = condensing values for the next level).
        Cps_new = cps_of(qv_cloud)
        s1_new = Cps_new * t_cloud + gz2 + _latent_heat(t_cloud) * qv_cloud

        # Pin every carry component to the input dtype so a float64
        # constant inside the Newton solve / latent-heat / gate math does
        # not promote the carry and break ``lax.scan``'s equal-type
        # requirement on a float32 (Metal/finite-volume) state.  See
        # MEMORY: jax-scan-carry-dtype-stability.
        carry_new = tuple(
            c.astype(carry_dtype) for c in (
                s1_new, Cps_new, qv_cloud,
                t_cloud, gz2, w_lcl_new,
                cond_gate, prev_icond,
            )
        )
        out = (
            t_cloud.astype(carry_dtype),
            qv_cloud.astype(carry_dtype),
            icond2.astype(carry_dtype),
        )
        return carry_new, out

    # Scan from surface upward. Skip the launch level itself (oracle scans
    # k = nlev-1 .. klev_max; the surface level k=nlev is the launch).
    p_scan = jnp.moveaxis(p_s[:, 1:], 1, 0)
    inputs = (
        jnp.moveaxis(T_s[:, 1:], 1, 0),
        jnp.moveaxis(q_s[:, 1:], 1, 0),
        p_scan,
        jnp.moveaxis(gz_s[:, 1:], 1, 0),
        jnp.moveaxis(tve_s[:, 1:], 1, 0),
        jnp.moveaxis(w_s[:, 1:], 1, 0),
    )
    _, (tc_scan, qvc_scan, icond2_scan) = jax.lax.scan(step, init, inputs)

    # Move level axis back; prepend the launch level (cloud = env there).
    tc_scan = jnp.moveaxis(tc_scan, 0, 1)
    qvc_scan = jnp.moveaxis(qvc_scan, 0, 1)
    icond2_scan = jnp.moveaxis(icond2_scan, 0, 1)

    tc_s = jnp.concatenate([T_base[:, None], tc_scan], axis=1)
    qvc_s = jnp.concatenate([qv_base[:, None], qvc_scan], axis=1)
    icond2_s = jnp.concatenate(
        [jnp.zeros((ncol, 1), dtype=T.dtype), icond2_scan], axis=1
    )

    # Back to top->surface ordering.
    tc = tc_s[:, ::-1]
    qvc = qvc_s[:, ::-1]
    icond2 = icond2_s[:, ::-1]
    return tc, qvc, icond2


def kuo_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    dt: float,
    config: KuoConfig = KuoConfig(),
    moisture_convergence: jax.Array | None = None,
    w_grid: jax.Array | None = None,
) -> ConvectionOutput:
    """Canonical Kuo (1965) convection driven by large-scale convergence.

    Parameters
    ----------
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).  Column
        ordering: index 0 = model top, index nlev-1 = surface.
    q_v : jax.Array
        Water-vapor mixing ratio [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    dt : float
        Model time step [s].
    config : KuoConfig
        Convection configuration (entrainment, Newton iters, partition).
    moisture_convergence : jax.Array or None
        Large-scale ``∂q/∂t|dyn`` [kg/kg/s], shape (ncol, nlev),
        positive = moisture inflow.  This is the canonical Kuo source
        (``ptenq``).  When ``None`` Kuo is QUIESCENT (no source to
        redistribute) — the physically correct single-column behavior.
    w_grid : jax.Array or None
        Resolved grid-scale vertical velocity [m/s] at full levels,
        shape (ncol, nlev), positive = ascent.  Used for the oracle's
        ``w_lcl > 0`` activation gate (the oracle reads an INDEPENDENT
        ``w(k)`` there).  When ``None`` the gate falls back to the SIGN
        of the local moisture convergence as the ascent proxy — exact
        whenever convergence and resolved ascent share sign, and the
        bridge derives a real ``w`` from the dycore continuity equation
        where it can (Codex review-1/2 finding #1).

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.  ``dq_c_conv_dt`` (the
        convective cloud-water source handed to microphysics) is
        non-negative by construction and equals the in-scheme
        condensation implied by ``dT_dt``.
    """
    ncol, nlev = T.shape
    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev), >0 (top->surface)
    g = constants.g

    # ``ptenq`` = the large-scale convergence source.  Quiescent when
    # absent: zero source → zero cvgu → zero tendencies (correct Kuo
    # behavior in a single column with no resolved large-scale ascent).
    if moisture_convergence is None:
        ptenq = jnp.zeros_like(q_v)
    else:
        ptenq = moisture_convergence

    # --- Column geometry: geopotential from the hydrostatic integral ---
    # gz_half[surface] = 0; integrate upward using env virtual temperature
    # (oracle: gzh(k) = gzh(k+1) + Rd·tve·(ph(k+1)-ph(k))/p(k)).
    tve = T * (1.0 + _VT_COEFF * q_v)  # (ncol, nlev)
    # Surface-first cumulative integral of dgz = Rd·tve·dp/p.
    dgz = constants.R_d * tve * dp / p_full  # (ncol, nlev), >0
    # gz at full levels: surface gz = 0.5·dgz_surface (half a layer up),
    # then accumulate. We replicate the oracle's half/full split:
    #   gzh(nlev+1)=0; gzh(k)=gzh(k+1)+dgz(k); gz(k)=0.5(gzh(k)+gzh(k+1)).
    # In surface-first ordering, cumulative sum from the surface.
    dgz_s = dgz[:, ::-1]                     # surface-first
    gzh_below_s = jnp.concatenate(
        [jnp.zeros((ncol, 1), dtype=T.dtype),
         jnp.cumsum(dgz_s, axis=1)[:, :-1]], axis=1,
    )                                         # gzh at the lower interface
    gzh_above_s = jnp.cumsum(dgz_s, axis=1)   # gzh at the upper interface
    gz_s = 0.5 * (gzh_below_s + gzh_above_s)
    gz = gz_s[:, ::-1]                        # back to top->surface

    # Vertical-velocity field for the oracle's ``w_lcl > 0`` gate.  The
    # oracle reads an INDEPENDENT ``w(k)`` at the LCL.  When the caller
    # supplies the resolved grid-scale ``w_grid`` (the convection bridge
    # derives it from the dycore continuity equation, exactly as the
    # Kain-Fritsch trigger does) we use it directly — FAITHFUL to the
    # oracle's independent-``w`` gate (Codex review-1/2 finding #1).
    # Otherwise we fall back to the SIGN of the local large-scale
    # convergence as the ascent proxy (positive convergence ⇒ ascent),
    # which is exact whenever convergence and ascent share sign (tropical
    # convective regime).  ``moisture_convergence is None`` ⇒ proxy ≡ 0 ⇒
    # gate off (quiescent), regardless of ``w_grid``.
    if w_grid is not None:
        w_proxy = w_grid
    else:
        w_proxy = ptenq

    # --- Entraining parcel ascent (faithful oracle loop) ---
    tc, qvc, icond2 = _parcel_ascent(T, q_v, p_full, gz, tve, w_proxy, config)

    # --- Environmental thermodynamic factors ---
    Cps_env = constants.c_pd * (1.0 - q_v) + q_v * constants.c_pv
    alpha_env = _latent_heat(T) / Cps_env       # = Lh/Cps_env (oracle alpha)
    q_sat_env = saturation_specific_humidity(T, p_full)
    rh = q_v / jnp.maximum(q_sat_env, 1e-12)

    # --- Activation mask: icond==2 AND ptenq>0 (smooth) ---
    # Oracle gate is ``ptenq(k) > 0.0`` — a Heaviside on the SIGN, on for
    # ANY positive convergence regardless of magnitude (the oracle heats
    # aloft via ``cvgu/zint·(tc−t)`` even where the local ``ptenq`` is
    # vanishingly small, as long as it is positive).  We feed the
    # SCALE-FREE sign ``ptenq / (|ptenq| + floor)`` (→ +1 for any
    # clearly-positive ptenq, → −1 for subsidence, 0 only at exactly 0)
    # into the sigmoid so the gate is ~1 across the whole convergent
    # column down to its exponential tail, not 0.5 at the cloud top.
    # ``ptenq_sign_floor`` (1e-30) is a pure division-by-zero guard.
    pq_sign = ptenq / (jnp.abs(ptenq) + config.ptenq_sign_floor)
    pq_gate = smooth_step(pq_sign, config.icond_sharpness)
    active = icond2 * pq_gate                    # (ncol, nlev) in [0,1]
    dpg = dp / g                                 # mass / area per layer

    # --- Convergence + normalization integrals over active layers ---
    # cvgu = Σ max(0, ptenq) dp/g over icond==2 (oracle uses icond==2,
    # not the ptenq>0 mask, for the integral).
    cvgu = jnp.sum(jnp.maximum(ptenq, 0.0) * icond2 * dpg, axis=1)  # (ncol,)
    zint = jnp.sum(
        (qvc - q_v + (tc - T) / alpha_env) * icond2 * dpg, axis=1,
    )                                                              # (ncol,)
    zint_t = jnp.sum((tc - T) * icond2 * dpg, axis=1)
    zint_q = jnp.sum((qvc - q_v) * icond2 * dpg, axis=1)

    # RH-mean over the convective layer (for Kuo-Anthes partition).
    layer_mass = jnp.sum(icond2 * dpg, axis=1)
    rhmean = jnp.sum(rh * icond2 * dpg, axis=1) / jnp.maximum(layer_mass, 1e-30)
    # Anthes (1977) moistening FRACTION b ∈ [0, 1] by definition: the
    # accession splits (1−b) to heating and b to moistening.  ``rh`` is
    # not capped at 1, so in a near-saturated cloud layer
    # ``rhmean + offset > 1`` and the raw ``1 − rhmean − offset`` goes
    # NEGATIVE — flipping the moistening term ``b·rt_q·(qvc−qv)`` into a
    # spurious extra DRYING and pushing the heating factor ``(1−b)``
    # above 1 (the column would condense more than the accession it
    # consumes).  Clamp to [0, 1]: a saturated layer moistens nothing
    # (b=0, all heating), a bone-dry layer at most fully moistens (b=1).
    bkuo = jnp.clip(
        1.0 - rhmean - config.anthes_rh_offset, 0.0, 1.0,
    )                                                        # (ncol,)

    # Safe normalisation denominators (signed; floor only the magnitude).
    def _safe(x):
        return jnp.where(
            jnp.abs(x) > config.zint_floor, x,
            jnp.sign(x) * config.zint_floor + (x == 0.0) * config.zint_floor,
        )

    zint_safe = _safe(zint)
    zint_t_safe = _safe(zint_t)
    zint_q_safe = _safe(zint_q)

    # --- Closure tendencies ---
    # CONSERVATION (truth tier): the redistribution terms (cvgu/zint·…) MUST be
    # applied over the SAME mask used to NORMALISE them — ``icond2`` (the
    # icond==2 cloud layer that defines cvgu/zint) — so that
    # Σ_icond2 ratio·(…) = ratio·zint = cvgu exactly.  The previous code applied
    # them over ``active`` (= icond2·sign(ptenq)>0), giving
    # ratio·Σ_active = cvgu·(zint_active/zint) ≠ cvgu whenever divergence
    # (ptenq≤0) levels lie inside the cloud — a spurious column source/sink with
    # truncated heating aloft.  The LOCAL moisture-convergence REMOVAL (−ptenq)
    # stays gated on ``active`` (convection only consumes converged moisture
    # where it converges; it integrates to −cvgu).  Then the column total-water
    # tendency Σ(dq_v + dq_c)·dp/g closes to 0: −cvgu (drying) + ratio·zint_q
    # (moistening) + ratio·zint_t/alpha (condensation→cloud water) = −cvgu + cvgu.
    if config.partition == "kuo1965":
        # dt/dt = cvgu/zint·(tc−t); dq/dt = −ptenq + cvgu/zint·(qvc−qv).
        ratio = (cvgu / zint_safe)[:, None]
        dT_dt = icond2 * ratio * (tc - T)
        dq_v_dt = active * (-ptenq) + icond2 * ratio * (qvc - q_v)
    elif config.partition == "anthes":
        # Kuo-Anthes (1977): heating gets (1−bkuo), moistening gets bkuo.
        bk = bkuo[:, None]
        rt_t = (cvgu / zint_t_safe)[:, None]
        rt_q = (cvgu / zint_q_safe)[:, None]
        dT_dt = icond2 * alpha_env * (1.0 - bk) * rt_t * (tc - T)
        dq_v_dt = active * (-ptenq) + icond2 * bk * rt_q * (qvc - q_v)
    else:
        raise ValueError(
            f"Unknown Kuo partition: {config.partition!r}. "
            "Choose 'kuo1965' or 'anthes'."
        )

    # --- Convective cloud-water source for microphysics ---
    # The heating is supplied by condensation: ``Cps·dT = Lh·dcond`` so
    # ``dcond = dT/dt / alpha_env`` with ``alpha_env = Lh/Cps_env`` — the
    # SAME enthalpy factor the closure uses for ``zint`` and that the
    # oracle uses for its surface-rain diagnostic
    # (``rain = Σ dtdt·dp/(g·alpha_env)``).  Using the dry-constant
    # ``c_pd/L_v`` here instead would be a ~0.5 % inconsistency with the
    # scheme's own ``alpha_env`` (Codex review-1 finding #3).  Only the
    # positive (condensation) part makes cloud water; non-negative by
    # construction.
    cond_signed = dT_dt / alpha_env
    dq_c_conv_dt = jnp.maximum(cond_signed, 0.0)

    # --- Preserve q_v >= 0 over one forward-Euler step ---
    # The Kuo source can be a net sink at a level (−ptenq dominating, or
    # cvgu/zint·(qvc−qv) < 0 above the moisture peak).  Cap the net
    # drying so ``q_v + dt·dq_v_dt >= 0`` exactly.  We scale the *whole*
    # column tendency by the binding per-level factor so the convergence
    # closure stays internally proportional (heating and moistening drop
    # together), rather than clipping moisture alone and orphaning the
    # latent heat.
    #
    # This is a POSITIVITY LIMITER, not part of faithful Kuo.  It departs
    # from the exact oracle closure ONLY on the (rare) step where a level
    # would otherwise go negative — i.e. ``col_scale < 1`` — and then by a
    # uniform column factor that preserves the heating/moistening ratio.
    # In the oracle test case and the full 60-day RCE it does NOT bind
    # (``col_scale == 1``); the faithful tendencies are passed through
    # unchanged there (Codex review-1 finding #4).
    # Per level the available vapor over the step is ``q_v`` (kg/kg) and
    # the demanded drying is ``-dt·min(dq_v_dt, 0) >= 0``.  The column
    # must be scaled by ``s = min(1, min_k avail_k / demand_k)`` so the
    # binding level just reaches zero.  Computed gradient-safely: the
    # ratio uses a SAFE divide whose denominator is floored AND whose
    # gradient is masked where the demand is (near) zero, so the
    # quiescent path (all demand = 0) returns ``s = 1`` with a clean
    # zero gradient — no ``0/0`` reverse-mode NaN (the previous
    # ``jnp.where`` over a division tripped this, issue #249).
    demand = jnp.clip(-dt * dq_v_dt, 0.0, None)    # (ncol, nlev) >= 0
    avail = jnp.clip(q_v, 0.0, None)
    # Exact forward-Euler cap is ``s = min(1, min_k avail_k/demand_k)``.
    # Implement ``avail/demand`` with a SAFE divide: floor the denominator
    # at ``demand_floor`` (a tiny but *finite* vapor-tendency scale, many
    # orders below any real ``demand`` so the cap is exact wherever it
    # binds) so the reverse-mode gradient is bounded — never the
    # ``avail/1e-300`` inf-times-zero NaN the previous ``jnp.where`` over a
    # 1e-300 floor produced (issue #249 class).  At ``demand=0`` the
    # ratio is ``avail/demand_floor ≫ 1`` → clipped to 1 (no scaling),
    # with a clean zero gradient through the clip plateau.
    demand_floor = 1.0e-20                          # kg/kg, demand scale ~1e-5
    ratio = avail / jnp.maximum(demand, demand_floor)
    col_scale = jnp.clip(
        jnp.min(ratio, axis=1, keepdims=True), 0.0, 1.0,
    )                                              # (ncol, 1) in (0, 1]
    dT_dt = dT_dt * col_scale
    dq_v_dt = dq_v_dt * col_scale
    dq_c_conv_dt = dq_c_conv_dt * col_scale

    # --- CAPE diagnostic (parcel from surface, env reference) ---
    T_base = T[:, -1]
    q_base = q_v[:, -1]
    T_moist = compute_moist_adiabat(T_base, p_full, q_v_base=q_base)
    cape = compute_cape(T, T_moist, p_full, p_half)

    # Column convective indicator (smooth 0-1): does the column actually
    # convect?  Gated on the CONVERGENCE source ``cvgu`` (a small positive
    # number when firing, exactly 0 when quiescent) via ``tanh(cvgu/scale)``
    # so the quiescent path — no large-scale convergence, zero tendencies
    # — reports a zero mask (``tanh(0)=0``) rather than the spurious 0.5
    # the ``ptenq>0`` sign gate gives at exactly ptenq=0.  ``cvgu>=0`` by
    # construction so ``tanh`` is monotone here.  The activation scale is
    # a KuoConfig field (matching the sibling ``buoyancy_scale_K`` /
    # ``supersat_scale`` trigger normalisers), not an inline literal.
    convective_mask = jnp.tanh(cvgu / config.cvgu_activation_scale)

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=convective_mask,
    )
