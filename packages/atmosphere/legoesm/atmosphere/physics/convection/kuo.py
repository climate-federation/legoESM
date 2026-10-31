"""Canonical Kuo (1965) convection scheme — faithful to Mahfouf's oracle.

Faithful JAX port of J.-F. Mahfouf's reference Kuo implementation
(AJFMAHFOUF/MOIST_CONVECTION_KUO, ``src/kuo_schemes.f90`` at
79b78f58f793fbf7849bca51d8ec0a92f39ec170, variant ``kuo65`` with the optional
Kuo-Anthes 1977 partition).  At the hard limit of the smooth gates
(``icond_sharpness`` -> infinity) it reproduces that Fortran, run with
legoESM's constants and saturation curve and ``klev_max = 1``, to round-off
on the fixture columns of ``tests/atmosphere/test_kuo_mahfouf_oracle.py``,
where the existing division guards (``zint_floor`` on the closure
denominators, a 1e-12 floor on ``q_sat`` in the relative humidity) do not
bind; the oracle divides without guards.

Not the oracle (user decisions 2026-10-28, #2213 split):
- the oracle's ``klev_max = 20`` scan top, a level index of its ECMWF L60
  grid, is not applied (the parcel is scanned to the model top; identical
  tendencies on all 1535 ECMWF profiles);
- saturation curve and constants are legoESM's (the oracle uses Huang 2018
  with an ice branch and its own Cp, Rd, g, 0.608, 273.16; worth -18% of
  column heating);
- no positivity guard, as the oracle: one step can drive ``q_v`` negative.

KNOWN DIFFERENCE (RULE 5).  The oracle's hard switches on model state are
smooth at the default ``icond_sharpness = 50`` and recovered at its hard
limit (1e12, with ``ptenq_sign_floor = 1e-30``, reproduces the oracle to
round-off, dT, dq and CAPE, on the fixture, where no gate argument at a
firing level is exactly 0):
sigmoids for LCL condensation (scale ``supersat_scale``), buoyancy
(``buoyancy_scale_K``), ``w_lcl > 0`` (fixed speed scale ``w_lcl_scale_ms``)
and ``ptenq > 0`` (scale-free sign); ``x * sigmoid`` smooth positive parts
for the parcel excess ``max(qv1 - qsat, 0)`` (``supersat_scale``) and
``max(ptenq, 0)`` in ``cvgu`` (``ptenq_scale``), which undershoot to at most
``-0.28 scale / icond_sharpness`` for a negative argument.  An argument of exactly 0
gives a half-open sigmoid gate where the oracle's strict ``>`` gives 0.
The ``ptenq > 0`` sign gate's transition width is ``ptenq_sign_floor``
(default 3e-10 kg/kg/s, user decision 2026-10-31, #2273); the oracle's hard
gate needs BOTH ``icond_sharpness = 1e12`` and ``ptenq_sign_floor = 1e-30``
(at the hard sharpness with a width >= 1e-11 the fixture error is 4e-3..4e-2,
from oracle-firing levels with ptenq ~ 1e-23).
Measured gap at the defaults (fp64, field-normalised
``max|ours - oracle| / max|oracle|``, fixture of
``tests/atmosphere/test_kuo_mahfouf_oracle.py``, re-measured 2026-10-31 at
the sign-gate width 3e-10): Kuo-1965 L60 dT 0.175, dq 0.150, CAPE 2.2e-4,
45 columns fire vs the oracle's 38 of 51; L32 dT 4.7e-4, dq 1.2e-3, CAPE
3.8e-7, 15 vs 15 of 18; Kuo-Anthes L60 dT 0.114, dq 0.167; L32 dT 7.8e-4,
dq 4.1e-4.  Against the former width 1e-30 the field gaps change only on
L32 (Kuo-1965 dT 7.2e-5 -> 4.7e-4, dq 4.7e-4 -> 1.2e-3; Anthes dT 1.2e-4
-> 7.8e-4, dq 2.4e-4 -> 4.1e-4) and the largest per-column dT gap goes
0.50 -> 0.75 (L60) and 2.4e-4 -> 1.7e-2 (L32).  Per-column gaps are pinned
in ``tests/atmosphere/oracle_fixtures/kuo_mahfouf_default_gaps.json``.
What the smoothing breaks: columns the oracle leaves off fire partially;
positivity and column-water closure are not guaranteed at either setting
(the oracle guarantees neither).

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
``alpha = L_v / c_p``.  The tendencies are applied only at levels that are
buoyant AND have positive vertical velocity at the LCL (``icond==2``)
AND have positive local convergence (``ptenq > 0``), while ``cvgu`` and
``zint`` are integrated over ``icond==2``; with divergence inside the cloud
the column water is therefore not closed, exactly as in the oracle.

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
All hard switches in the oracle are smooth (see KNOWN DIFFERENCE above) so
the scheme is ``jax.grad`` / ``jit`` / ``vmap`` compatible; the entraining
parcel ascent is a ``lax.scan`` from the surface upward.  There is no
positivity limiter (the oracle has none).  The convective cloud-water source
handed to microphysics is the SIGNED condensation implied by ``dT_dt``
(``dT_dt / alpha``, the oracle's rain integrand), so it is negative where
the scheme cools.  ``cape`` is the oracle's entraining-cloud ``zcape``.

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
    saturation_specific_humidity,
    saturation_specific_humidity_dT,
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
        "convergence is supplied. Smooth (differentiable) gates; Mahfouf "
        "MOIST_CONVECTION_KUO at the hard-gate limit."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa", "dt": "s",
        "moisture_convergence": "kg/kg/s (large-scale dq/dt|dyn source; None -> quiescent)",
        "w_grid": "m/s (resolved ascent for the w_lcl>0 gate; required with moisture_convergence)",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "dq_c_conv_dt": "kg/kg/s (convective cloud-water source to microphysics, signed dT_dt/alpha)",
        "cape": "J/kg (oracle entraining-cloud zcape)",
        "convective_mask": "1 (0-1 activation, >= -0.28/icond_sharpness from the smoothing)",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Where a column is buoyant with positive "
        "convergence, convection warms toward the cloud profile (tc>T) and "
        "moistens (qvc>qv), and explicitly REMOVES the consumed large-scale "
        "tendency (-ptenq). The implied condensation is dT_dt/alpha with "
        "alpha=Lh/Cps_env (the oracle's rain diagnostic). Neither column "
        "water nor column moist static energy is conserved: the scheme is "
        "source-driven by external convergence, and, as in the oracle, the "
        "tendencies are applied only where ptenq>0 while cvgu/zint are "
        "integrated over the whole icond==2 layer. dq_c_conv_dt is the "
        "per-level signed dT_dt/alpha (the oracle's rain integrand)."
    ),
    # Source-driven: the scheme consumes/removes an EXTERNAL large-scale
    # moisture-convergence tendency and, as the oracle, applies the
    # redistribution only where ptenq > 0, so the column is not closed ->
    # no contract-level conservation claimed.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Kuo (1965), J. Atmos. Sci. 22, 40-63; "
        "Anthes (1977), Mon. Wea. Rev. 105, 270-286"
    ),
    "idealized_test": (
        "tests/atmosphere/test_kuo_mahfouf_oracle.py (running-oracle match "
        "at the hard-gate limit); tests/unit/test_kuo_oracle_faithful.py: "
        "moisture_convergence=None -> zero tendencies (quiescent single "
        "column); a conditionally-unstable column with positive large-scale "
        "convergence and ascent -> heating + moistening with "
        "convective_mask>0; convergence without w raises."
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
    :func:`legoesm.thermo.latent_heat_vaporization`
    (same pattern as :func:`_dqsat_dT` below) so the Kuo call sites read
    like the oracle while the formula lives in one place.
    """
    return latent_heat_vaporization(T)


def _dqsat_dT(T: jax.Array, p: jax.Array) -> jax.Array:
    """d(q_sat)/dT [1/K] for the legoesm Tetens saturation used here.

    Thin wrapper over :func:`legoesm.thermo.saturation_specific_humidity_dT`,
    the exact (forward-mode) derivative of
    :func:`legoesm.thermo.saturation_specific_humidity`, softplus floor
    included, so the Newton solve uses the slope of the curve it is measured
    against. Kept as a named alias so the Kuo call sites read like the
    oracle's ``dqsat``.
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

    Returns ``(tc, qvc, icond2, cape)``; the first three are (ncol, nlev), where ``tc`` and
    ``qvc`` are the in-cloud temperature / vapor and ``icond2`` is the
    smooth [0,1] activation gate (oracle ``icond==2`` = buoyant AND
    ``w_lcl>0`` AND condensing).  At the launch (surface) level ``tc=T`` and
    ``qvc=q_v`` (the oracle initialisation).  Where the parcel is not
    condensing, ``tc`` is the DRY-ascent parcel temperature ``t1`` and
    ``qvc`` the parcel's carried vapour (oracle ``t2 = t1``, ``qv1 = qv2``),
    not the environment.  The oracle's ``klev_max`` scan top (a level index
    of its ECMWF L60 grid) is NOT ported: the parcel is scanned to the model
    top, which equals the oracle run with ``klev_max = 1``.  ``cape`` (ncol,)
    is the oracle's entraining-cloud ``zcape``: the sum over levels that
    condense AND are buoyant of ``-g dz (tvc - tve)/tve``, with the same
    smooth gates.
    """
    ncol, nlev = T.shape
    sharp = config.icond_sharpness
    eps_entr = config.entrainment
    c_pv = constants.c_pv
    c_pd = constants.c_pd
    c_pw = constants.c_pw
    g = constants.g

    # The ``w_lcl > 0`` gate argument is ``w / w_lcl_scale_ms``, a FIXED
    # physical speed scale (not a state-dependent column maximum), so the
    # gate width is ``w_lcl_scale_ms / icond_sharpness`` [m/s] everywhere.
    w_norm = w / config.w_lcl_scale_ms

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

    # Carry: (s1, Cps, qv1, t_prev, gz_prev, w_lcl, prev_icond)
    # ``prev_icond`` (the level below's condensation gate) reproduces the
    # oracle's ``icond(k)==0 .and. icond(k+1)==0`` capture of ``w_lcl``:
    # ``icond(k)`` is the CURRENT level, still 0 when tested, so the test is
    # "the level below is not condensing" (kuo_schemes.f90:123).
    init = tuple(
        c.astype(carry_dtype) for c in (
            s1_0, Cps0, qv_base,
            T_base, gz_s[:, 0],
            jnp.zeros(ncol, dtype=carry_dtype),    # w_lcl
            jnp.zeros(ncol, dtype=carry_dtype),    # prev_icond
        )
    )

    def step(carry, inp):
        s1, Cps, qv1, t_prev, gz_prev, w_lcl, prev_icond = carry
        T_k, q_k, p2, gz2, tve_k, w_k = inp

        # oracle dz = (gz(k+1) - gz(k))/rg: gz INCREASES upward, so dz < 0.
        dz = (gz_prev - gz2) / g

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
        # ``max(qv1 - qsat, 0)`` smoothed as ``x * sigmoid(s x / c)`` on the
        # condensation gate's own scale and sharpness (exactly 0 at x = 0;
        # hard limit: icond_sharpness -> infinity), then weighted by the
        # condensing-branch gate ``cond_gate`` like ``t_cloud``/``qv_cloud``
        # (``cond_gate`` contains the same sigmoid, so it enters squared).
        excess = qv1 - qsat_k
        qc = excess * smooth_step(excess / config.supersat_scale, sharp) * cond_gate
        t_cloud = cond_gate * t2 + (1.0 - cond_gate) * t1
        qv_cloud = cond_gate * qsat_k + (1.0 - cond_gate) * qv1

        # Virtual cloud temperature with water loading.
        tvc = t_cloud * (1.0 + _VT_COEFF * qv_cloud - qc)
        buoyant = smooth_step((tvc - tve_k) / config.buoyancy_scale_K, sharp)

        # ``w_lcl`` captured wherever condensation starts (oracle:
        # icond(k)==0 .and. icond(k+1)==0 → w_lcl = w(k)), i.e. at every
        # level that condenses while the level below does not -- including a
        # re-entry into saturation after a non-condensing level.
        first_lcl = cond_gate * (1.0 - prev_icond)
        w_lcl_new = first_lcl * w_k + (1.0 - first_lcl) * w_lcl

        # icond==2: condensing AND buoyant AND w_lcl > 0.
        w_pos = smooth_step(w_lcl_new, sharp)
        icond2 = cond_gate * buoyant * w_pos
        # Oracle zcape increment: condensing AND buoyant (icond==1 there).
        dcape = cond_gate * buoyant * (-g * dz * (tvc - tve_k) / tve_k)

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
                cond_gate,
            )
        )
        out = (
            t_cloud.astype(carry_dtype),
            qv_cloud.astype(carry_dtype),
            icond2.astype(carry_dtype),
            dcape.astype(carry_dtype),
        )
        return carry_new, out

    # Scan from surface upward. Skip the launch level itself (the oracle
    # scans k = nlev-1 .. klev_max; the surface level k=nlev is the launch).
    # Here the scan runs to the model top (oracle klev_max = 1; see above).
    p_scan = jnp.moveaxis(p_s[:, 1:], 1, 0)
    inputs = (
        jnp.moveaxis(T_s[:, 1:], 1, 0),
        jnp.moveaxis(q_s[:, 1:], 1, 0),
        p_scan,
        jnp.moveaxis(gz_s[:, 1:], 1, 0),
        jnp.moveaxis(tve_s[:, 1:], 1, 0),
        jnp.moveaxis(w_s[:, 1:], 1, 0),
    )
    _, (tc_scan, qvc_scan, icond2_scan, dcape_scan) = jax.lax.scan(
        step, init, inputs)
    cape = jnp.sum(dcape_scan, axis=0)

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
    return tc, qvc, icond2, cape


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
        Model time step [s].  Unused (the oracle's tendencies do not depend
        on it); kept for the common convection interface.
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
        ``w(k)`` there).  Required whenever ``moisture_convergence`` is
        given: convergence is not a stand-in for ascent, so a call with
        convergence and no ``w_grid`` raises ``ValueError``.

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.  ``dq_c_conv_dt`` (the
        convective cloud-water source handed to microphysics) is the
        signed condensation implied by ``dT_dt`` (``dT_dt / alpha``, the
        oracle's rain integrand); ``cape`` is the oracle's entraining-cloud
        ``zcape``.
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
        if w_grid is None:
            raise ValueError(
                "kuo_convection: moisture_convergence was given without "
                "w_grid. The oracle's w_lcl > 0 gate reads an independent "
                "vertical velocity; convergence is not a substitute. Pass the "
                "resolved grid-scale w [m/s].")
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

    # Vertical velocity for the oracle's ``w_lcl > 0`` gate (the oracle's
    # independent ``w(k)``).  Quiescent path (no convergence): ptenq = 0, so
    # cvgu = 0 and every tendency is 0 whatever ``w`` is.
    w_proxy = w_grid if w_grid is not None else jnp.zeros_like(q_v)

    # --- Entraining parcel ascent (faithful oracle loop) ---
    tc, qvc, icond2, cape = _parcel_ascent(
        T, q_v, p_full, gz, tve, w_proxy, config)

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
    # into the sigmoid so the gate is ~1 across the convergent column
    # wherever ptenq is well above ``ptenq_sign_floor``, not 0.5 at the
    # cloud top.
    # ``ptenq_sign_floor`` is the gate's transition width (default 3e-10
    # kg/kg/s; 1e-30 with icond_sharpness 1e12 = the oracle).  Written
    # as ``z / (|z| + 1)`` with ``z = ptenq / floor`` (same value): the
    # derivative never forms ``(|ptenq| + 1e-30)**2``, which underflows to 0
    # in fp32 and made the fp32 gradient w.r.t. ptenq NaN at ptenq = 0.
    pq_z = ptenq / config.ptenq_sign_floor
    pq_sign = pq_z / (jnp.abs(pq_z) + 1.0)
    pq_gate = smooth_step(pq_sign, config.icond_sharpness)
    active = icond2 * pq_gate                    # (ncol, nlev) in [0,1]
    dpg = dp / g                                 # mass / area per layer

    # --- Convergence + normalization integrals over active layers ---
    # cvgu = Σ max(0, ptenq) dp/g over icond==2 (oracle uses icond==2,
    # not the ptenq>0 mask, for the integral).  ``max(0, ptenq)`` is smoothed
    # as ``ptenq * sigmoid(s ptenq / ptenq_scale)``, s = icond_sharpness:
    # exactly 0 at ptenq = 0, so a column without convergence stays exactly
    # quiescent (hard limit: sharpness -> inf).
    ptenq_pos = ptenq * smooth_step(
        ptenq / config.ptenq_scale, config.icond_sharpness)
    cvgu = jnp.sum(ptenq_pos * icond2 * dpg, axis=1)               # (ncol,)
    zint = jnp.sum(
        (qvc - q_v + (tc - T) / alpha_env) * icond2 * dpg, axis=1,
    )                                                              # (ncol,)
    zint_t = jnp.sum((tc - T) * icond2 * dpg, axis=1)
    zint_q = jnp.sum((qvc - q_v) * icond2 * dpg, axis=1)

    # RH-mean over the convective layer (for Kuo-Anthes partition).
    layer_mass = jnp.sum(icond2 * dpg, axis=1)
    rhmean = jnp.sum(rh * icond2 * dpg, axis=1) / jnp.maximum(layer_mass, 1e-30)
    # Kuo-Anthes moistening parameter, unclamped as in the oracle
    # (kuo_schemes.f90:185 ``bkuo = 1 - rhmean - 0.1``); it is negative when
    # ``rhmean > 1 - anthes_rh_offset``.
    bkuo = 1.0 - rhmean - config.anthes_rh_offset             # (ncol,)

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
    # Oracle (kuo_schemes.f90:193-199, 254-260): the tendencies are applied
    # only where ``icond(k)==2 .and. ptenq(k) > 0`` (``active``), while cvgu,
    # zint, zint_t, zint_q are integrated over ``icond==2``.  With divergence
    # (ptenq <= 0) inside the cloud the column water is therefore not closed
    # (the oracle's own behaviour).
    if config.partition == "kuo1965":
        # dt/dt = cvgu/zint·(tc−t); dq/dt = −ptenq + cvgu/zint·(qvc−qv).
        ratio = (cvgu / zint_safe)[:, None]
        dT_dt = active * ratio * (tc - T)
        dq_v_dt = active * (-ptenq + ratio * (qvc - q_v))
    elif config.partition == "anthes":
        # Kuo-Anthes (1977): heating gets (1−bkuo), moistening gets bkuo.
        bk = bkuo[:, None]
        rt_t = (cvgu / zint_t_safe)[:, None]
        rt_q = (cvgu / zint_q_safe)[:, None]
        dT_dt = active * alpha_env * (1.0 - bk) * rt_t * (tc - T)
        dq_v_dt = active * (-ptenq + bk * rt_q * (qvc - q_v))
    else:
        raise ValueError(
            f"Unknown Kuo partition: {config.partition!r}. "
            "Choose 'kuo1965' or 'anthes'."
        )

    # --- Convective cloud-water source for microphysics ---
    # The heating is supplied by condensation: ``Cps·dT = Lh·dcond`` so
    # ``dcond = dT/dt / alpha_env`` with ``alpha_env = Lh/Cps_env``, the
    # factor of the oracle's surface-rain diagnostic
    # (``rain = Σ dtdt·dp/(g·alpha_env)``, signed).  Handed to microphysics
    # SIGNED, per level, so its column sum equals the oracle's signed rain
    # diagnostic (negative where ``dT_dt < 0``).  The physics pipeline's
    # surface precipitation for Kuo is computed separately, from the column
    # vapour sink; the two differ where the column water is not closed.
    dq_c_conv_dt = dT_dt / alpha_env

    # No positivity limiter: the oracle has none (``q_v + dt·dq_v_dt`` may go
    # negative, as in the oracle).  ``dt`` is unused, as in the oracle.

    # Column convective indicator (smooth 0-1): does the column actually
    # convect?  Gated on the CONVERGENCE source ``cvgu`` (a small positive
    # number when firing, exactly 0 when quiescent) via ``tanh(cvgu/scale)``
    # so the quiescent path — no large-scale convergence, zero tendencies
    # — reports a zero mask (``tanh(0)=0``) rather than the spurious 0.5
    # the ``ptenq>0`` sign gate gives at exactly ptenq=0.  ``cvgu>=0`` by
    # construction up to the smooth positive part's undershoot (>= -0.28
    # ptenq_scale / icond_sharpness per level).  The activation scale is
    # a KuoConfig field (matching the sibling ``buoyancy_scale_K`` /
    # ``supersat_scale`` trigger normalisers), not an inline literal.
    # The column total gets the same smooth positive part, so a divergent
    # column (cvgu slightly < 0 from the per-level undershoot) reports a mask
    # >= -0.28 / icond_sharpness instead of down to -1 (exactly 0 at cvgu = 0).
    cvgu_pos = cvgu * smooth_step(
        cvgu / config.cvgu_activation_scale, config.icond_sharpness)
    convective_mask = jnp.tanh(cvgu_pos / config.cvgu_activation_scale)

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=convective_mask,
        precip_frozen_sfc=jnp.zeros_like(cape),  # L_v-only: liquid at sfc (F29)
    )
