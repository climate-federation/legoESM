"""Enhanced diffusion for convective adjustment.

Applies large vertical diffusivity where N^2 < 0 (statically unstable).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import (
    compute_buoyancy_frequency,
    compute_buoyancy_frequency_adiabatic,
    compute_buoyancy_frequency_nemo_bn2,
    rho_0 as _RHO_0_DEFAULT,
)
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.vertical_mixing._shared import vmap_vertical_diffusion
from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.physics.convection.output import OceanConvectionOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

__physics_contract__ = {
    "summary": (
        "Convective adjustment by enhanced vertical diffusion (Oceananigans "
        "ConvectiveAdjustmentVerticalDiffusivity): apply a large tracer "
        "diffusivity K_conv (and an independent momentum viscosity nu_conv) "
        "wherever N^2 < 0, background values elsewhere."
    ),
    "inputs": {
        "T": "degC", "S": "psu", "rho": "kg/m^3",
        "jacobian": "1 (z-star dimensionless)", "u": "m/s", "v": "m/s",
        "cfg.K_conv": "m^2/s", "cfg.nu_conv": "m^2/s",
    },
    "outputs": {
        "dT_dt": "degC/s", "dS_dt": "psu/s", "convection_flag": "1 (active)",
        "K_v": "m^2/s", "du_dt": "m/s^2", "dv_dt": "m/s^2", "A_v": "m^2/s",
    },
    "sign_convention": (
        "K_conv, nu_conv >= 0 and applied only where statically unstable "
        "(N^2 < 0); down-gradient flux-form vertical diffusion with no-flux "
        "top/bottom BC, so the column integral of heat, salt (and momentum, "
        "when u,v are supplied) is conserved (adiabatic vertical "
        "redistribution); z positive up; K/A zeroed on dry interfaces."
    ),
    # Flux-form diffusion with no-flux BC conserves column-integrated heat
    # (energy), salt, and momentum (momentum only when u,v are supplied).
    "conserves": ["energy", "salt", "momentum"],
    "differentiable": True,
    "reference": (
        "Oceananigans ConvectiveAdjustmentVerticalDiffusivity (Ramadhan et al. "
        "2020, JOSS 5, 2018); Klinger et al. (1996) JPO 26 enhanced-diffusion "
        "convective adjustment"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_enhanced_diffusion_momentum.py + "
        "tests/ocean/unit/test_convective_K_A_flag_direct.py — an unstable "
        "column mixes toward neutral conserving column-integrated T,S,(u,v); a "
        "stable column keeps background K; convection_flag marks N^2<0 "
        "interfaces."
    ),
}


def convective_K_A_flag(
    rho: jnp.ndarray,
    dz_ref: jnp.ndarray,
    jacobian: jnp.ndarray,
    cfg: EnhancedDiffusionConfig,
    *,
    T: jnp.ndarray | None = None,
    S: jnp.ndarray | None = None,
    p_cell: jnp.ndarray | None = None,
    eos_fn=None,
    t_depth: jnp.ndarray | None = None,
    w_depth: jnp.ndarray | None = None,
    g: float = constants.g,
    rho_ref: float = _RHO_0_DEFAULT,
):
    """Convective tracer diffusivity, momentum viscosity, and flag.

    Single source of truth for the enhanced-diffusion (Oceananigans
    ``ConvectiveAdjustmentVerticalDiffusivity``) coefficients, shared by
    the explicit kernel ``enhanced_diffusion_convection`` and the implicit
    fallback ``compute_vertical_K_profiles`` so the two paths are bit-for-
    bit identical (no duplicated numerics).

    The convective trigger is ``N² < 0``.  ``cfg.n2_mode`` selects how ``N²``
    is measured: ``"insitu"``/``"insitu_signed"`` (default, BIT-IDENTICAL
    legacy) use the in-situ density gradient — but compressibility can leave a
    statically-unstable column with ``N² > 0``, so convection is MISSED.
    ``"adiabatic"`` uses the TRUE static stability (adiabatic parcel
    displacement to the upper cell's pressure), which requires ``T``, ``S`` and
    the cell-centre pressure ``p_cell`` (+ the model ``eos_fn``) to be threaded
    in; a compressibility-masked unstable column then gives ``N² < 0`` and the
    trigger fires.

    AD safety on dry / land columns: the ``N²`` division uses the interface
    thickness, which is **zero** when ``jacobian == 0``.  Masking ``N²`` *after*
    that division still leaves a ``num / 0`` node in the graph, and reverse-mode
    then evaluates ``0 · ∞ = NaN`` for any control ``rho`` (or ``T``/``S``)
    depends on.  We therefore substitute a unit ``jacobian`` on dry columns
    **before** the division (and, for the in-situ path, a reference density) so
    both forward and backward stay finite, then zero the diffusivity /
    viscosity on dry interfaces (no mixing through land).

    Returns
    -------
    (K_v, A_v, flag) : arrays at interior interfaces, shape
        ``rho.shape[:-1] + (nlev - 1,)``.  ``K_v`` is the tracer
        diffusivity (``convective_κz``), ``A_v`` the momentum viscosity
        (``convective_νz``), ``flag`` the convective activity in [0, 1].
    """
    # Validate the static-stability mode at function entry on the STATIC config
    # value (dispatch hardening — a typo must raise, not silently pick a
    # different N²). Allowed set mirrors ``_shared.compute_N2``.
    if cfg.n2_mode not in ("insitu", "insitu_signed", "adiabatic", "nemo_bn2"):
        raise ValueError(
            "Unknown EnhancedDiffusionConfig.n2_mode="
            f"{cfg.n2_mode!r}; expected 'insitu', 'insitu_signed', "
            "'adiabatic' or 'nemo_bn2'."
        )
    # Same gate for the alpha/beta selector the nemo_bn2 branch forwards
    # (mirrors ``eos.compute_buoyancy_frequency_nemo_bn2``'s own allowed set;
    # validated HERE too so a typo raises even under an n2_mode that never
    # reaches the kernel).
    if cfg.n2_eos_form not in ("seos", "teos10"):
        raise ValueError(
            "Unknown EnhancedDiffusionConfig.n2_eos_form="
            f"{cfg.n2_eos_form!r}; expected 'seos' (NEMO's 3-term simplified "
            "EOS) or 'teos10' (NEMO's Roquet polynomial with the TEOS-10 "
            "coefficient set, which is what ORCA1 runs: ln_teos10=.true.)."
        )

    dz_actual = dz_ref * jacobian[..., jnp.newaxis]               # (..., nlev)
    dry_iface = (dz_actual[..., :-1] <= 0.0) | (dz_actual[..., 1:] <= 0.0)
    dry_col = jacobian <= 0.0                                     # (...,)

    # Substitute a safe (unit) jacobian BEFORE the N² interior division so the
    # denominator never hits 0 on dry columns (AD-safe).  Shared by both modes.
    jacobian_safe = jnp.where(dry_col, 1.0, jacobian)
    if cfg.n2_mode == "adiabatic":
        # TRUE static stability (SIGNED): both interface parcels displaced to
        # the upper cell's pressure through the model EOS — unbiased by
        # compressibility, so it fires on masked-unstable columns.
        if T is None or S is None or p_cell is None:
            raise ValueError(
                "convective_K_A_flag: n2_mode='adiabatic' requires T, S and "
                "p_cell (cell-centre hydrostatic pressure [Pa]); the caller "
                "must thread them (enhanced_diffusion.integration and the "
                "implicit k_profiles path do)."
            )
        N2 = compute_buoyancy_frequency_adiabatic(
            T, S, p_cell, dz_ref, jacobian_safe, eos_fn=eos_fn,
            rho_ref=rho_ref, g=g,
        )
    elif cfg.n2_mode == "nemo_bn2":
        # NEMO eosbn2 bn2 (S-EOS): local alpha,beta at each cell's gdept
        # interpolated to the w-point by the geometric zrw weight (SIGNED).
        if T is None or S is None or t_depth is None or w_depth is None:
            raise ValueError(
                "convective_K_A_flag: n2_mode='nemo_bn2' requires T, S and the "
                "geometric depth ladders t_depth (gdept) / w_depth (interior "
                "gdepw); the enhanced_diffusion factory threads them from "
                "eos.nemo_bn2_depth_ladders(z_coord)."
            )
        # NemoSEOSConfig() defaults (the DINO/Kamm set) — matching the density
        # path, where make_eos_fn's "nemo_seos" branch also has no custom-
        # coefficient threading from any recipe. Thread a cfg through here the
        # day a recipe carries non-default S-EOS coefficients.
        # ``eos_form`` selects WHICH alpha/beta the bn2 assembly uses; this
        # forward was MISSING, so the EVD trigger always took the S-EOS
        # branch even on a TEOS-10 card whose TKE sibling
        # (``TKEConfig.n2_eos_form``, threaded at
        # vertical_mixing/_shared.py) used the Roquet polynomial.
        N2 = compute_buoyancy_frequency_nemo_bn2(
            T, S, t_depth, w_depth, g=g, eos_form=cfg.n2_eos_form)
    else:
        # In-situ density N² (SIGNED); reference density on dry columns keeps
        # the numerator finite too (BIT-IDENTICAL legacy path).
        rho_safe = jnp.where(dry_col[..., jnp.newaxis], rho_ref, rho)
        N2 = compute_buoyancy_frequency(rho_safe, dz_ref, jacobian_safe,
                                        rho_ref=rho_ref, g=g)

    if cfg.smooth_transition:
        sig = jax.nn.sigmoid(-N2 * cfg.sigmoid_sharpness)
        K = cfg.K_bg + (cfg.K_conv - cfg.K_bg) * sig
        A = cfg.nu_bg + (cfg.nu_conv - cfg.nu_bg) * sig
        flag = sig
    else:
        # NEMO zdfevd fires where N² < n2_threshold (default 0.0; NEMO GYRE
        # uses -1e-12 to ignore marginally-neutral interfaces — see cfg doc).
        _thr = getattr(cfg, "n2_threshold", 0.0)
        K = jnp.where(N2 < _thr, cfg.K_conv, cfg.K_bg)
        A = jnp.where(N2 < _thr, cfg.nu_conv, cfg.nu_bg)
        flag = jnp.where(N2 < _thr, 1.0, 0.0)

    # Zero mixing + flag on dry interfaces (no mixing through land).
    K = jnp.where(dry_iface, 0.0, K)
    A = jnp.where(dry_iface, 0.0, A)
    flag = jnp.where(dry_iface, 0.0, flag)
    return K, A, flag


def enhanced_diffusion_convection(
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: EnhancedDiffusionConfig,
    apply_diffusion: bool = True,
    dt: float | None = None,
    u: jnp.ndarray | None = None,
    v: jnp.ndarray | None = None,
    p_cell: jnp.ndarray | None = None,
    eos_fn=None,
    eta: jnp.ndarray | None = None,
    H_bathy: jnp.ndarray | None = None,
    g: float = constants.g,
    rho_ref: float = _RHO_0_DEFAULT,
) -> OceanConvectionOutput:
    """Apply enhanced diffusion where the water column is unstable.

    Oceananigans ``ConvectiveAdjustmentVerticalDiffusivity``: large
    vertical **tracer diffusivity** ``K_conv`` (and **momentum
    viscosity** ``nu_conv``) wherever ``N² < 0``, background values
    elsewhere.  Tracer and momentum coefficients are independent.

    When ``dt`` is supplied, the explicit-branch CFL cap uses the
    actual physics step instead of ``cfg.cfl_dt_estimate`` — preserves
    stability and correct convection strength when runtime ``dt``
    differs from the config-time estimate.  Codex iter-3 finding #7.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : EnhancedDiffusionConfig
    apply_diffusion : bool
        Run the explicit diffusion branch.  When False, K_v / A_v are
        returned in the output for downstream implicit solvers.
    dt : float, optional
        Physics step [s].  When provided, used in the CFL cap instead
        of ``cfg.cfl_dt_estimate``.
    u, v : array (6, n, n, nlev), optional
        Velocity components.  Required for the explicit momentum-mixing
        branch (``apply_diffusion=True``); when ``None`` the explicit
        momentum tendencies are zero (the implicit path re-derives ``A_v``
        in ``compute_vertical_K_profiles`` and applies it via the
        unconditionally-stable backward-Euler solve).
    p_cell : array (6, n, n, nlev), optional
        Cell-centre hydrostatic pressure [Pa].  REQUIRED when
        ``cfg.n2_mode == "adiabatic"`` (the true static-stability trigger);
        ignored on the default in-situ path.
    eos_fn : callable or None
        EOS ``fn(T, S, p) -> rho`` for the adiabatic parcel displacement
        (``None`` -> Wright 1997).  Ignored on the in-situ path.
    eta, H_bathy : array ``(...)`` or None
        Sea-surface height [m] and local column depth [m].  REQUIRED when
        ``cfg.n2_mode == "nemo_bn2"`` (they build NEMO's live
        ``gdept(Kmm) = gdept_0*(1 + eta/H_bathy)`` ladder); ignored otherwise.

    Returns
    -------
    OceanConvectionOutput
    """
    # Tracer diffusivity (convective_κz), momentum viscosity (convective_νz)
    # and convective flag at interfaces — shared, AD-safe on dry columns.
    # T/S/p_cell/eos_fn are consumed only by the adiabatic-N² trigger
    # (cfg.n2_mode == "adiabatic"); ignored on the default in-situ path.
    # NEMO bn2 trigger needs the geometric depth ladders (gdept / interior
    # gdepw); cheap to extract, ignored by every other n2_mode.
    if cfg.n2_mode == "nemo_bn2":
        # NEMO evaluates alpha/beta/bn2 at the LIVE gdept(Kmm) =
        # gdept_0*(1 + eta/ht_0) -- see eos.nemo_bn2_live_ladders for the
        # macro expansion and why this is NOT the z* Jacobian ``jacobian``
        # above ((eta + H)/H_max, normalised by the GLOBAL maximum depth).
        if eta is None or H_bathy is None:
            raise ValueError(
                'EnhancedDiffusionConfig.n2_mode="nemo_bn2" needs eta and '
                "H_bathy to build NEMO's live gdept(Kmm) ladder; "
                "enhanced_diffusion_convection was called without them.")
        from legoesm.ocean.eos import nemo_bn2_live_ladders
        _t_depth, _w_depth = nemo_bn2_live_ladders(z_coord, eta, H_bathy)
    else:
        _t_depth = _w_depth = None
    K, A, flag = convective_K_A_flag(
        rho, z_coord.dz_ref, jacobian, cfg,
        T=T, S=S, p_cell=p_cell, eos_fn=eos_fn,
        t_depth=_t_depth, w_depth=_w_depth,
        g=g, rho_ref=rho_ref,
    )

    # CFL safety cap on the EXPLICIT branch.  Backward-Euler (implicit)
    # mixing is unconditionally stable; explicit-Euler vertical
    # diffusion requires ``K · dt / dz² ≤ 1/2``.  Without the cap a
    # ``K_conv = 1 m²/s`` over ``dz = 10 m`` would need ``dt ≤ 50 s`` —
    # 70× tighter than typical ocean physics steps.  See
    # ``EnhancedDiffusionConfig`` for ``cfl_dt_estimate`` and
    # ``cfl_safety``.
    #
    # The earlier coarse cap used the reference ``dz_ref`` alone and
    # ignored ``jacobian`` (z-star contracts layers when ``eta + H``
    # shrinks): a column with jacobian = 0.5 has dz_actual half of
    # dz_ref, so K must be 4× tighter.  Delegating the per-interface
    # cap to ``vertical_diffusion_variable_K(..., dt=dt_eff)`` uses the
    # actual layer thicknesses ``dz_ref · jacobian`` and the per-
    # interface dz_min that the leaf already computes.
    du_dt = None
    dv_dt = None
    if apply_diffusion:
        dt_eff = cfg.cfl_dt_estimate if dt is None else dt
        diffuse = lambda q, c: vertical_diffusion_variable_K(
            q, z_coord, jacobian, c, dt=dt_eff, cfl_safety=cfg.cfl_safety,
        )
        if u is not None and v is not None:
            # Mix momentum (A) and tracers (K) through the shared kernel
            # — same operator/CFL cap the Richardson scheme uses.
            vel_tend, tr_tend = vmap_vertical_diffusion(
                u, v, T, S, A, K, diffuse, apply_diffusion=True,
            )
            du_dt = vel_tend[0]
            dv_dt = vel_tend[1]
        else:
            # Fail closed: a tracer-only explicit branch with a nonzero
            # convective momentum viscosity would silently drop the requested
            # momentum mixing.  Tracer-only is legal only for nu_*=0 (the MPAS
            # path, which guards nu_*=0 at its factory before calling here).
            if cfg.nu_conv != 0.0 or cfg.nu_bg != 0.0:
                raise ValueError(
                    "enhanced_diffusion_convection: nu_conv/nu_bg is nonzero "
                    "but u/v were not supplied, so the explicit convective "
                    "momentum tendency cannot be computed and would be "
                    "silently dropped. Pass cell-centred u and v, or set "
                    "nu_conv=nu_bg=0.0 for tracer-only convective adjustment."
                )
            # Velocity not supplied + nu_*=0: tracer-only explicit branch.
            tr_tend = jax.vmap(
                lambda q: diffuse(q, K), in_axes=0, out_axes=0,
            )(jnp.stack([T, S], axis=0))
        dT_dt = tr_tend[0]
        dS_dt = tr_tend[1]
    else:
        dT_dt = jnp.zeros_like(T)
        dS_dt = jnp.zeros_like(S)

    return OceanConvectionOutput(
        dT_dt=dT_dt,
        dS_dt=dS_dt,
        convection_flag=flag,
        K_v=K,
        du_dt=du_dt,
        dv_dt=dv_dt,
        A_v=A,
    )
