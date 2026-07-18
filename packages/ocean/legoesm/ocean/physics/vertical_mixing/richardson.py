"""Pacanowski & Philander (1981) Richardson-number dependent mixing.

POP / E3SM Omega convention:
    ν (momentum) = ν₀ / (1 + α·Ri)^n + ν_b
    κ (tracer)   = ν / (1 + α·Ri) + κ_b

The tracer division gives an EFFECTIVE Prandtl-Ri scaling Pr = ν/κ
that grows with Ri.  ``cfg.Pr_t`` is retained for API compatibility
but is now a no-op — non-default values trigger a UserWarning at
function-call time so calibration sweeps notice the field is dead.
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
        "mixing: nu = nu0/(1+alpha*Ri)^n + nu_b, kappa = nu/(1+alpha*Ri) + "
        "kappa_b (POP/E3SM convention), yielding an effective Ri-growing "
        "Prandtl number."
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
            "after the iter-47 PP81 fix; the canonical POP/E3SM "
            "scaling κ = ν/(1+αRi) + κ_b makes Pr a function of Ri "
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

    # Pacanowski & Philander (1981, JPO 11, p.1448, Eq. 1) — the
    # POP / E3SM Omega convention is:
    #   ν (momentum) = ν₀ / (1 + α·Ri)^n + ν_b      [n = 2]
    #   κ (tracer)   = ν / (1 + α·Ri) + κ_b
    # The full ν (including ν_b background) enters the tracer
    # division; the tracer division produces an EFFECTIVE
    # Prandtl-Ri scaling Pr = ν/κ that grows with Ri, which is
    # physically essential because in stable shear momentum mixes
    # more efficiently than tracer.
    #
    # The previous formulation computed K_v with (1+αRi)^n decay
    # (momentum's form) and assigned momentum A_v = K_v · const_Pr_t —
    # which (a) used the momentum decay rate for the tracer field, and
    # (b) discarded the canonical Prandtl-Ri dependence (constant 10×
    # Prandtl instead of Ri-growing).
    one_plus_aRi = 1.0 + cfg.alpha * Ri
    A_v = cfg.K_0 / one_plus_aRi ** cfg.n + cfg.A_bg              # momentum
    K_v = A_v / one_plus_aRi + cfg.K_bg                            # tracer

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
