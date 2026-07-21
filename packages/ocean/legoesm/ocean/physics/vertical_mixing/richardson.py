"""Pacanowski & Philander (1981) Richardson-number dependent mixing.

    ν (momentum) = ν₀/(1+α·Ri)^n [CVMix Eq. 4.4]  +  ν_b [background]
    κ (tracer)   = ν/(1+α·Ri) [full-ν division]   +  κ_b [background]

The momentum SHEAR term matches the CVMix technical documentation (Griffies et
al., Aug 2012) Eq. 4.4, ν_shear = ν₀/(1+α·Ri)^n, plus a configured background ν_b
(the backgrounds are NOT part of Eqs. 4.4-4.5, which are shear-only).  CVMix
cites α = 5, n = 2 as the common POP settings; the amplitude ν₀ = K_0 and the
backgrounds A_bg/K_bg are project/POP defaults, not fixed by the CVMix doc.  The
tracer here uses the FULL-ν division κ = ν/(1+α·Ri) + κ_b, which re-divides the
background ν_b into the tracer and so gives an effective Prandtl-Ri scaling
Pr = ν/κ that grows with Ri.  ``cfg.Pr_t`` is retained for API compatibility but
is now a no-op — non-default values trigger a UserWarning at function-call time
so calibration sweeps notice the field is dead.

Faithfulness
------------
``tests/ocean/unit/test_pp81_richardson_faithful.py`` pins the diffusivity
closed form to round-off (rel 1e-12) against an independent reimplementation
(default AND a non-default config), ties the public function's returned A_v/K_v
to the helper convention, and canaries the ``RichardsonVerticalMixingConfig``
defaults.

FAITHFUL:
- ``_pp81_diffusivities`` momentum ν = K_0/(1+α·Ri)^n + A_bg — the CVMix Eq. 4.4
  shear term (α = 5, n = 2) plus the background A_bg, monotonically decreasing in
  Ri, maximal (K_0 + A_bg) at Ri = 0.
- ``RichardsonVerticalMixingConfig`` defaults α = 5, n = 2 (CVMix/POP common
  settings); K_0 = 5e-3, A_bg = 1e-4, K_bg = 1e-5 m^2/s (project/POP defaults).

DEPARTURES (documented, canaried in the test):
- The tracer κ = ν/(1+α·Ri) + K_bg uses the FULL ν (incl. the A_bg background),
  re-dividing it into the tracer.  It exceeds the canonical PP81 total tracer
  diffusivity — the CVMix Eq. 4.5 SHEAR term ν₀/(1+α·Ri)^(n+1) (same ν₀ = K_0 as
  the momentum numerator) plus the K_bg background — by exactly A_bg/(1+α·Ri).
  This full-ν placement is a deliberate iter-47 choice; the AUDITABLE claim
  pinned by the test is the algebraic departure from the Eq. 4.5 shear form, NOT
  a match to any particular Fortran (CVMix's documented tracer form is Eq. 4.5).
- ``cfg.Pr_t`` is a dead no-op (the Ri-dependent Prandtl is built into the
  formula); a non-default value warns rather than silently doing nothing.
- The unstable branch (Ri < 0) is clipped to Ri = 0 (maximum mixing) upstream in
  ``richardson_number``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
from legoesm.ocean.eos import (
    compute_buoyancy_frequency,
    compute_buoyancy_frequency_adiabatic,
)
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.vertical_mixing._shared import (
    richardson_number,
    vmap_vertical_diffusion,
)
from legoesm.ocean.physics.vertical_mixing.config import RichardsonVerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

__physics_contract__ = {
    "summary": (
        "Pacanowski & Philander (1981) Richardson-number dependent vertical "
        "mixing: nu = nu0/(1+alpha*Ri)^n (CVMix Aug-2012 Eq. 4.4 shear) + nu_b, "
        "kappa = nu/(1+alpha*Ri) + kappa_b (full-nu division; departs from the "
        "canonical tracer, CVMix Eq. 4.5 shear nu0/(1+alpha*Ri)^(n+1) + kappa_b, "
        "by +nu_b/(1+alpha*Ri)), yielding an effective Ri-growing Prandtl number."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "degC", "S": "psu",
        "rho": "kg/m^3", "jacobian": "1 (z-star dimensionless)",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "degC/s", "dS_dt": "psu/s",
        "K_v": "m^2/s", "A_v": "m^2/s",
    },
    "sign_convention": (
        "A_v (=nu) and K_v (=kappa) >= 0 and DECREASE monotonically with the "
        "gradient Richardson number Ri = N^2/S^2 (unstable Ri<0 clipped to "
        "maximum mixing); z positive up. By default (apply_diffusion=True) it "
        "applies flux-form vertical diffusion with a no-flux interior BC "
        "(surface stress/flux applied separately), so it conserves the "
        "column-integrated quantity; in apply_diffusion=False mode it produces "
        "the Ri-dependent K_v/A_v only."
    ),
    # Default path (apply_diffusion=True) applies conservative flux-form vertical
    # diffusion (no-flux interior BC), so it conserves column-integrated heat
    # (energy), salt and momentum — like constant.py. Profile mode produces
    # K_v/A_v only (conserves nothing, but applies nothing either).
    "conserves": ["energy", "salt", "momentum"],
    "differentiable": True,
    "reference": "Pacanowski, R. C. & Philander, S. G. H. (1981), JPO 11, 1443-1451",
    "idealized_test": (
        "tests/unit/test_physics_ocean.py + "
        "tests/ocean/unit/test_richardson_number_helper.py — strong "
        "stratification / large Ri reduces K toward background; an unstable "
        "column (Ri<0) gives maximal K."
    ),
}

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def _pp81_diffusivities(Ri, cfg):
    """PP81 Richardson-number diffusivities (full-nu tracer division).

        nu (A_v, momentum) = K_0/(1+alpha*Ri)^n [CVMix Eq. 4.4] + A_bg [background]
        kappa (K_v, tracer) = A_v / (1 + alpha*Ri) + K_bg

    ``Ri`` is expected already clipped to >= 0 (the unstable branch is handled
    by the ``richardson_number`` clip -> maximum mixing at Ri = 0).  The FULL
    ``nu`` (including the A_bg background) enters the tracer division, so K_v
    departs from the canonical PP81 total tracer diffusivity (CVMix Aug-2012
    Eq. 4.5 shear ``K_0/(1+alpha*Ri)^(n+1)``, same K_0 numerator, plus the K_bg
    background) by the extra ``A_bg/(1+alpha*Ri)`` term.  Returns ``(A_v, K_v)``
    [m^2/s], both >= 0 and monotonically decreasing with Ri, with an effective
    Prandtl number Pr = A_v/K_v that grows with Ri.
    """
    one_plus_aRi = 1.0 + cfg.alpha * Ri
    A_v = cfg.K_0 / one_plus_aRi ** cfg.n + cfg.A_bg              # momentum
    K_v = A_v / one_plus_aRi + cfg.K_bg                            # tracer
    return A_v, K_v


def richardson_vertical_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: RichardsonVerticalMixingConfig,
    apply_diffusion: bool = True,
    dt: float | None = None,
    p_cell: jnp.ndarray | None = None,
    eos_fn=None,
) -> VerticalMixingOutput:
    """Apply Richardson-number dependent vertical mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
        In-situ density.
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : RichardsonVerticalMixingConfig
    p_cell : array (6, n, n, nlev), optional
        Cell-centre hydrostatic pressure [Pa]. REQUIRED when
        ``cfg.n2_mode == "adiabatic"`` (the true static stability displaces
        both parcels of an interface to the upper cell's pressure); ignored
        for the in-situ modes.
    eos_fn : callable or None
        EOS ``fn(T, S, p) -> rho`` for the adiabatic parcel displacement
        (``None`` -> Wright 1997, matching the density path). Ignored for the
        in-situ modes.

    Returns
    -------
    VerticalMixingOutput
    """
    # Validate the static-stability mode at function entry on the STATIC config
    # value (dispatch hardening — a typo must raise, not silently pick a
    # different N²). Richardson does not thread the NEMO bn2 geometric depth
    # ladders, so ``"nemo_bn2"`` (convection/TKE only) is intentionally NOT in
    # this set and raises here.
    if cfg.n2_mode not in ("insitu", "insitu_signed", "adiabatic"):
        raise ValueError(
            "Unknown RichardsonVerticalMixingConfig.n2_mode="
            f"{cfg.n2_mode!r}; expected 'insitu', 'insitu_signed' or "
            "'adiabatic'."
        )

    eps = _EPS

    # Pr_t is now a no-op (the canonical Pr-Ri scaling is built into
    # the K_v formula).  Warn calibration users that adjusting it has
    # no effect — silent dead-config is a known calibration trap.
    if cfg.Pr_t != 10.0:
        warnings.warn(
            "RichardsonVerticalMixingConfig.Pr_t is no longer used "
            "after the iter-47 PP81 fix; the full-ν tracer division "
            "κ = ν/(1+αRi) + κ_b makes Pr a function of Ri "
            "automatically.  Setting Pr_t to a non-default value has "
            "no effect on the diffusivities.",
            UserWarning,
            stacklevel=2,
        )

    # N^2 at interfaces.  "insitu"/"insitu_signed": the in-situ density N²
    # (BIT-IDENTICAL legacy — ``compute_buoyancy_frequency`` is already SIGNED
    # here; the Ri>=0 clip in ``richardson_number`` handles the unstable
    # branch).  "adiabatic": PP81's true static stability (parcels displaced
    # to the upper cell's pressure), also SIGNED, unbiased by compressibility.
    if cfg.n2_mode == "adiabatic":
        if p_cell is None:
            raise ValueError(
                "richardson_vertical_mixing: n2_mode='adiabatic' requires "
                "p_cell (cell-centre hydrostatic pressure [Pa]); the caller "
                "must thread it (integration._make_richardson and the implicit "
                "k_profiles path do)."
            )
        N2 = compute_buoyancy_frequency_adiabatic(
            T, S, p_cell, z_coord.dz_ref, jacobian, eos_fn=eos_fn,
        )
    else:
        N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)

    # Gradient Richardson number Ri = N^2 / S^2 (#518: shared helper; clip
    # negative Ri -> max mixing for the unstable branch).
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    Ri = richardson_number(N2, u, v, dz_actual, eps=eps, clip_negative=True)

    # Pacanowski & Philander (1981); canonical SHEAR terms per CVMix Aug-2012
    # Eqs. 4.4-4.5 (POP settings α = 5, n = 2), backgrounds added separately:
    #   ν (momentum) = ν₀/(1 + α·Ri)^n     (Eq. 4.4 shear) + ν_b
    #   κ (canonical) = ν₀/(1 + α·Ri)^(n+1) (Eq. 4.5 shear) + κ_b
    # This code instead uses the FULL-ν tracer division
    #   κ (tracer) = ν / (1 + α·Ri) + κ_b,
    # i.e. it re-divides the background ν_b into the tracer, producing an
    # EFFECTIVE Prandtl-Ri scaling Pr = ν/κ that grows with Ri (in stable
    # shear momentum mixes more efficiently than tracer).  It therefore
    # DEPARTS from the canonical Eq. 4.5 tracer by exactly the extra
    # ν_b/(1+α·Ri) term — the one algebraic departure pinned by
    # test_pp81_richardson_faithful.py (NOT a claim to match any Fortran).
    #
    # The previous formulation computed K_v with (1+αRi)^n decay
    # (momentum's form) and assigned momentum A_v = K_v · const_Pr_t —
    # which (a) used the momentum decay rate for the tracer field, and
    # (b) discarded the canonical Prandtl-Ri dependence (constant 10×
    # Prandtl instead of Ri-growing).
    A_v, K_v = _pp81_diffusivities(Ri, cfg)

    # Apply variable-K vertical diffusion.  When ``apply_diffusion`` is
    # False, return zero tendencies; the caller will apply K_v/A_v via an
    # unconditionally-stable backward-Euler solver after the explicit step.
    # dt (when provided) is passed so the explicit-Euler CFL cap fires on
    # K_v/A_v at thin upper layers; default None keeps current behaviour.
    vel_tend, tr_tend = vmap_vertical_diffusion(
        u, v, T, S, A_v, K_v,
        lambda q, c: vertical_diffusion_variable_K(q, z_coord, jacobian, c, dt=dt),
        apply_diffusion,
    )

    return VerticalMixingOutput(
        du_dt=vel_tend[0],
        dv_dt=vel_tend[1],
        dT_dt=tr_tend[0],
        dS_dt=tr_tend[1],
        K_v=K_v,
        A_v=A_v,
    )
