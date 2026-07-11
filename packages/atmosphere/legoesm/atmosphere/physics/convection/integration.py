"""Model integration bridge for convection.

Provides `make_convection_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

Supported model types:
- "hydrostatic"  : PrimitiveEquationModel (sigma coordinates)
- "nonhydrostatic": CompressibleEulerModel (z* coordinates)
- "spectral_pe"  : SpectralPEModel (Gaussian grid + sigma coordinates)
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticState,
    HydrostaticTendencies,
    NonHydrostaticState,
    NonHydrostaticTendencies,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import (
    HeightCoordinate,
    SigmaCoordinate,
    TerrainMetric,
    pressure_from_sigma,
    compute_sigma_dot_and_total,
    compute_pressure_velocity,
)
from legoesm import constants

from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.atmosphere.physics._shared import (
    compute_moisture_convergence,
    diagnose_grid_w_from_omega,
    moisture_convergence_supported,
    zero_like_tracers,
)
from legoesm.core.operators_3d import divergence_3d as _div3_cs
from legoesm.core.operators_latlon_3d import divergence_3d as _div3_latlon
from legoesm.grids.gaussian import sh_analysis_3d, vordiv_from_uv_3d
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.dca import dca_convection
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.mass_flux import (
    edmf_convection,
    mass_flux_convection,
)
from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
    zhang_mcfarlane_convection,
)
from legoesm.atmosphere.physics.convection.kain_fritsch import (
    kain_fritsch_convection,
)
from legoesm.atmosphere.physics.convection.emanuel import (
    emanuel_convection,
)
from legoesm.atmosphere.physics.convection.tiedtke import (
    tiedtke_convection,
)
from legoesm.atmosphere.physics.convection.bechtold import (
    bechtold_convection,
)
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)


def _get_convection_fn(config: ConvectionConfig):
    """Select the convection backend based on config.scheme.

    Returns
    -------
    scheme_name : str
        Name of the scheme.
    conv_fn : callable or None
        Backend convection function.
    scheme_config : NamedTuple or None
        Scheme-specific configuration.
    """
    if config.scheme == "sbm":
        return "sbm", sbm_convection, config.sbm
    elif config.scheme == "dca":
        return "dca", dca_convection, config.dca
    elif config.scheme == "kuo":
        return "kuo", kuo_convection, config.kuo
    elif config.scheme == "mass_flux":
        return "mass_flux", mass_flux_convection, config.mass_flux
    elif config.scheme == "edmf":
        return "edmf", edmf_convection, config.edmf
    elif config.scheme == "zhang_mcfarlane":
        return "zhang_mcfarlane", zhang_mcfarlane_convection, config.zhang_mcfarlane
    elif config.scheme == "kain_fritsch":
        return "kain_fritsch", kain_fritsch_convection, config.kain_fritsch
    elif config.scheme == "emanuel":
        return "emanuel", emanuel_convection, config.emanuel
    elif config.scheme == "tiedtke":
        return "tiedtke", tiedtke_convection, config.tiedtke
    elif config.scheme == "bechtold":
        return "bechtold", bechtold_convection, config.bechtold
    elif config.scheme == "none":
        return "none", None, None
    else:
        raise ValueError(f"Unknown convection scheme: {config.scheme!r}")


class ConvectionSchemeTraits(NamedTuple):
    """Static plumbing traits of a convection scheme.

    Single source of truth for which inputs/carries each scheme needs,
    shared by the per-model-type bridge factories below AND the unified
    driver pipeline (``legoesm.driver.physics_pipeline``) so the two
    call paths cannot drift in what they feed a scheme.
    """
    is_scalar_prognostic: bool   # (ncol,) carry (mass_flux M_c / edmf a_u)
    is_profile_prognostic: bool  # full (ncol, nlev) conv_prog_profile carry
    is_cmt_capable: bool         # consumes u, v for convective momentum transport
    is_w_grid_consumer: bool     # consumes resolved w_grid (KF trigger)
    is_stochastic: bool          # carries conv_stoch_state (+ optional PRNG key)
    is_mc_consumer: bool         # consumes large-scale moisture convergence
    is_simple_mc_consumer: bool  # stateless MC-driven leaf (canonical Kuo)
    detrains_to_cloud: bool      # convective condensate is true DETRAINMENT into
    # the q_c cloud bucket (plume / mass-flux schemes).  False = an ADJUSTMENT
    # scheme (Betts-Miller sbm / dca / Kuo) whose column-net drying is convective
    # PRECIPITATION, not lingering grid-scale cloud water: routing it into q_c
    # let q_c accumulate ~100x (opaque clouds, ~0.85 planetary albedo, runaway
    # cold drift / OLR collapse) since Kessler autoconversion cannot rain out a
    # convective-precip-rate source.  When False the pipeline precipitates the
    # convective condensate directly (energy-neutral: latent heat is already in
    # dT_dt_conv; mass-conserving: column water removed = precip).


def convection_scheme_traits(scheme_name: str) -> ConvectionSchemeTraits:
    """Return the static plumbing traits for *scheme_name*.

    ZM is technically diagnostic but uses ``[:, -1]`` of the profile as
    an M_b carry for implicit relaxation; KF is diagnostic but routed
    through the profile path so the caller plumbs ``w_grid`` for its
    trigger.  Kuo is the canonical moisture-convergence scheme: a simple
    leaf whose source IS the large-scale convergence (``None`` MC ⇒
    correctly quiescent).
    """
    return ConvectionSchemeTraits(
        is_scalar_prognostic=scheme_name in ("mass_flux", "edmf"),
        is_profile_prognostic=scheme_name in (
            "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke",
            "bechtold",
        ),
        is_cmt_capable=scheme_name in ("zhang_mcfarlane", "tiedtke", "bechtold"),
        is_w_grid_consumer=scheme_name in ("kain_fritsch",),
        is_stochastic=scheme_name in ("bechtold",),
        is_mc_consumer=scheme_name in ("tiedtke", "bechtold"),
        is_simple_mc_consumer=scheme_name in ("kuo",),
        # Plume / mass-flux schemes genuinely detrain condensate into q_c; the
        # adjustment schemes (sbm Betts-Miller / dca / kuo) produce convective
        # PRECIPITATION (their drying falls out), so their condensate is routed
        # to precip rather than the q_c cloud bucket.
        detrains_to_cloud=scheme_name in (
            "mass_flux", "edmf", "zhang_mcfarlane", "kain_fritsch",
            "emanuel", "tiedtke", "bechtold",
        ),
    )


def diagnose_w_grid_columns_hydrostatic(
    u_grid,
    v_grid,
    p_s,
    grid,
    sigma_coord,
    T_col,
    p_full_col,
    q_v_col,
    *,
    need_concrete: bool,
    dtype,
):
    """Resolved grid-scale ``w`` [m/s] in ``(ncol, nlev)`` column layout.

    The hydrostatic dycore doesn't expose ``omega`` at the physics
    boundary, so re-derive it from the standard sigma-coordinate
    continuity::

        D       = ∇·v_h                  (per full level)
        D_t     = Σ D · Δσ               (column total)
        dp_s/dt = -p_s · D_t / (1 - σ_top)
        σ̇      = compute_sigma_dot(D)
        ω       = σ · dp_s/dt + p_s · σ̇

    then convert to ``w`` via :func:`._shared.diagnose_grid_w_from_omega`.
    Only the cubed-sphere and lat-lon grids ship a divergence operator we
    can call here; other grids fall back to ``zeros`` when
    ``need_concrete`` (Kain-Fritsch reads ``w_grid`` unconditionally) or
    ``None`` otherwise (Kuo then uses its convergence-sign proxy rather
    than a spurious zero-w gate).

    Shared by the hydrostatic bridge below and the unified driver
    pipeline.
    """
    ncol, nlev = T_col.shape
    div_grid = None
    if isinstance(grid, CubedSphereGrid):
        if v_grid is not None:
            div_grid = _div3_cs(u_grid, v_grid, grid)
    elif hasattr(grid, "dlat") and hasattr(grid, "dlon"):
        if v_grid is not None:
            div_grid = _div3_latlon(u_grid, v_grid, grid)

    if div_grid is not None:
        sigma_top = sigma_coord.sigma_half[0]
        # Iter-55: share the cumsum between σ̇ and ``D_total`` (mirrors
        # iter-52/53 in the dycores).  Saves one cross-shard reduction
        # per physics call on this w-grid diagnostic path.
        sigma_dot_grid, _D_total_full = compute_sigma_dot_and_total(
            div_grid, sigma_coord,
        )
        dp_s_dt_grid = -p_s * _D_total_full[..., 0] / (1.0 - sigma_top)
        omega_grid = compute_pressure_velocity(
            sigma_dot_grid, p_s, dp_s_dt_grid, sigma_coord,
        )
        return diagnose_grid_w_from_omega(
            omega_grid.reshape(ncol, nlev), T_col, p_full_col, q_v_col,
        ).astype(dtype)
    if need_concrete:
        return jnp.zeros((ncol, nlev), dtype=dtype)
    return None


def make_convection_physics(
    convection_config: ConvectionConfig,
    model_type: str = "hydrostatic",
    dt: float = 300.0,  # coeff-ok: default physics timestep [s]
) -> Callable:
    """Create a physics function for convection matching a model's signature.

    Parameters
    ----------
    convection_config : ConvectionConfig
        Convection configuration (selects SBM, DCA, Kuo, mass_flux,
        EDMF, or none).
    model_type : str
        One of "hydrostatic", "nonhydrostatic", "spectral_pe".
    dt : float
        Model time step [s]. Needed for relaxation timescale.

    Returns
    -------
    Callable
        Physics function with the correct signature for the model.
    """
    # ``model_type="mpas"`` reuses the hydrostatic factory: the column
    # physics bridge in ``_make_hydrostatic_convection`` reshapes
    # ``(*shape_2d, nlev)`` to ``(ncol, nlev)`` and never touches grid
    # latitude/longitude — so it is grid-agnostic across cubed-sphere
    # ``(face, n, n)``, lat-lon ``(n_lat, n_lon)``, and MPAS Voronoi
    # ``(nCells,)``.  Without this branch the MPAS combined path (which
    # passes ``model_type="mpas"`` through to all sub-physics factories)
    # crashes the moment convection is enabled on an MPAS run.
    if model_type in ("hydrostatic", "mpas"):
        return _make_hydrostatic_convection(convection_config, dt)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_convection(convection_config, dt)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_convection(convection_config, dt)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe', 'mpas'."
        )


# ===========================================================================
# Hydrostatic PE
# ===========================================================================

def _make_hydrostatic_convection(
    convection_config: ConvectionConfig,
    dt: float,
) -> Callable:
    """Create convection physics_fn for PrimitiveEquationModel.

    Signature: (state, grid, sigma_coord, phys_state=None)
               -> (HydrostaticTendencies, conv_prog_profile_new | None)

    When *phys_state* is passed, the convective prognostic profile is
    read from ``phys_state.conv_prog_profile`` (shape ``(ncol, nlev)``)
    and the updated profile is returned as the second element of the
    result tuple.  Scalar-carrying schemes (``mass_flux``, ``edmf``)
    pack their scalar at ``[:, -1]`` (cloud-base proxy) with zeros
    aloft; profile-carrying schemes (Tiedtke, Bechtold, added in later
    PRs) use the full profile.
    """
    scheme_name, conv_fn, scheme_config = _get_convection_fn(convection_config)
    # Static plumbing traits — single source of truth shared with the
    # unified driver pipeline (see :func:`convection_scheme_traits`).
    _tr = convection_scheme_traits(scheme_name)
    is_scalar_prognostic = _tr.is_scalar_prognostic
    is_profile_prognostic = _tr.is_profile_prognostic
    is_cmt_capable = _tr.is_cmt_capable
    is_w_grid_consumer = _tr.is_w_grid_consumer
    is_stochastic = _tr.is_stochastic
    is_mc_consumer = _tr.is_mc_consumer
    is_simple_mc_consumer = _tr.is_simple_mc_consumer
    # Static at closure-build time: avoid splitting / advancing the
    # master PRNG key when stochasticity is disabled, so the no-noise
    # path is exactly bit-identical to a no-Bechtold run apart from
    # the deterministic mass-flux contribution.
    needs_prng = is_stochastic and getattr(
        scheme_config, "enable_stochastic", False
    )

    prog_key = None
    prog_init = None
    if is_scalar_prognostic:
        if scheme_name == "mass_flux":
            prog_key, prog_init = "M_c", scheme_config.M_c_init
        else:  # edmf
            prog_key, prog_init = "a_u", scheme_config.a_u_init

    def physics_fn(
        state: HydrostaticState,
        grid,
        sigma_coord: SigmaCoordinate,
        phys_state=None,
    ):
        T = state.T.data          # cubed: (6,n,n,nlev) | latlon: (n_lat,n_lon,nlev) | mpas: (nCells,nlev)
        p_s = state.p_s.data      # cubed: (6,n,n)      | latlon: (n_lat,n_lon)      | mpas: (nCells,)

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape

        # Pressure at full and half levels
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)

        # Reshape to columns generically for any grid topology.
        # Cubed-sphere ``shape_2d=(6,n,n)`` → ncol = 6·n·n.
        # Lat-lon     ``shape_2d=(n_lat,n_lon)`` → ncol = n_lat·n_lon.
        # MPAS        ``shape_2d=(nCells,)`` → ncol = nCells.
        ncol = 1
        for s in shape_2d:
            ncol *= int(s)
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        # Pin defaulted allocations to the state precision so x64-default
        # zeros do not silently flow into the column physics path.
        _state_dtype = T.dtype
        _ps_dtype = state.p_s.data.dtype
        # Extract water vapor from tracers if available; else assume dry.
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(ncol, nlev)
        else:
            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        # Wind columns for CMT-capable schemes.  On cubed-sphere and
        # lat-lon the prognostic winds live at cell centres so the
        # ``(...,nlev) → (ncol, nlev)`` reshape works directly.  On
        # MPAS the prognostic ``u`` is the normal velocity on edges
        # (``shape (nEdges, nlev)``) so the reshape would mismatch
        # ``ncol = nCells``.  In that case we degrade gracefully to
        # zero u/v columns — the CMT-capable scheme still runs (it
        # produces zero CMT) and the rest of the column physics
        # (Tiedtke / ZM / Bechtold mass-flux closures) is unaffected.
        # Proper edge→cell interpolation for MPAS CMT is a follow-up.
        if is_cmt_capable:
            _u_data = state.u.data
            if _u_data.shape[0] == ncol or _u_data.shape[:-1] == shape_2d:
                u_col = _u_data.reshape(ncol, nlev)
                v_col = (
                    state.v.data.reshape(ncol, nlev)
                    if state.v is not None
                    else jnp.zeros_like(u_col)
                )
            else:
                # MPAS / unsupported wind staggering: zero CMT inputs.
                u_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)
                v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)
        else:
            u_col = None
            v_col = None

        # Grid-scale w for w-consuming schemes (Kain-Fritsch trigger;
        # Kuo's oracle ``w_lcl>0`` activation gate).  Shared helper —
        # see :func:`diagnose_w_grid_columns_hydrostatic` for the
        # sigma-continuity derivation and the per-scheme None/zeros
        # fallback semantics.
        if is_w_grid_consumer or is_simple_mc_consumer:
            w_grid_col = diagnose_w_grid_columns_hydrostatic(
                state.u.data,
                state.v.data if state.v is not None else None,
                state.p_s.data, grid, sigma_coord,
                T_col, p_full_col, q_v_col,
                need_concrete=is_w_grid_consumer,
                dtype=_state_dtype,
            )
        else:
            w_grid_col = None

        # Moisture convergence for MC-consuming schemes (Tiedtke,
        # Bechtold) and the canonical convergence-driven Kuo.  Reuses the
        # dycore's FV-flux-divergence operator via
        # :func:`._shared.compute_moisture_convergence`.  When the state
        # has no q_v tracer or wind data we pass ``None``.
        #
        # The ``None`` semantics DIFFER by scheme, intentionally:
        #   * Tiedtke/Bechtold treat ``None`` as "engage the built-in
        #     saturation-deficit proxy" (they gate on it internally).
        #   * Kuo treats ``None`` as ZERO SOURCE → QUIESCENT, which is the
        #     physically-correct canonical-Kuo behavior (no resolved
        #     large-scale convergence ⇒ nothing to converge).  So on a
        #     state with no ``v`` (e.g. MPAS edge-normal ``u`` with
        #     ``v is None``, or a single-column SCM) Kuo is deliberately
        #     OFF rather than falling back to a proxy (Codex review-1
        #     finding #2).  Wiring an edge→cell ``v`` reconstruction for
        #     MPAS would let Kuo fire there; that is a follow-up.
        if (
            (is_mc_consumer or is_simple_mc_consumer)
            and state.tracers is not None
            and "q_v" in state.tracers
            and state.v is not None
            # Degrade to None (proxy / quiescent semantics above) on
            # grids without an MC operator instead of TypeError-ing —
            # same guard as the unified pipeline (codex 2026-06-10 P2).
            and moisture_convergence_supported(grid)
        ):
            _compute_mc = compute_moisture_convergence
            # Tracer values may be Field-wrapped or raw JAX arrays.
            _qv_raw_full = state.tracers["q_v"]
            _qv_grid_full = (
                _qv_raw_full.data
                if hasattr(_qv_raw_full, "data")
                else _qv_raw_full
            )
            mc_col = _compute_mc(
                _qv_grid_full, state.u.data, state.v.data, grid,
            )
        else:
            mc_col = None

        conv_prog_out = None
        if conv_fn is None:
            # "none" scheme: return zero tendencies
            dT_dt = jnp.zeros(shape_3d, dtype=_state_dtype)
            conv_out = None
        elif is_scalar_prognostic:
            # Scalar-carrying schemes pack at [:, -1]; slice to recover
            # the per-column scalar.  Falls back to scheme default when
            # the carry is the wrong shape (warm start, scheme switch).
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile[:, -1]
            else:
                prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)

            conv_out, prog_new = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                **{prog_key: prog_in},
                dt=dt, config=scheme_config,
            )
            # Pack the updated scalar back into the (ncol, nlev) profile.
            conv_prog_out = jnp.zeros(
                (ncol, nlev), dtype=_state_dtype
            ).at[:, -1].set(prog_new)
            dT_dt = conv_out.dT_dt.reshape(shape_3d)
        elif is_profile_prognostic:
            # Profile-carrying schemes (ZM, KF, Emanuel, Tiedtke,
            # Bechtold) take and return the full
            # ``conv_prog_profile`` directly.  Per-scheme kwarg
            # plumbing handles CMT (u, v), the KF trigger (w_grid),
            # and Bechtold's stochastic state (conv_stoch_state +
            # prng_key).
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile
            else:
                prog_in = jnp.zeros((ncol, nlev), dtype=_state_dtype)

            if is_stochastic:
                # Bechtold: also threads conv_stoch_state, prng_key,
                # and moisture_convergence.
                if phys_state is not None and (
                    phys_state.conv_stoch_state.shape == (ncol,)
                ):
                    stoch_in = phys_state.conv_stoch_state
                else:
                    stoch_in = jnp.zeros((ncol,), dtype=_state_dtype)
                # Derive a per-step sub-key from the master phys_state
                # PRNG key by folding in a module-id (``"bechtold"`` →
                # int 0xBEC4).  ``jax.random.split`` advances the master
                # key so the next call sees a different stream.  When
                # ``needs_prng`` is False (stochasticity disabled at
                # config build time) we keep the master key untouched
                # and pass ``None`` to the leaf — that path is
                # bit-identical to a no-Bechtold run apart from the
                # deterministic mass-flux contribution.
                if needs_prng and phys_state is not None and hasattr(
                    phys_state, "prng_key"
                ):
                    bechtold_key, master_key_new = jax.random.split(
                        phys_state.prng_key, 2,
                    )
                    bechtold_key = jax.random.fold_in(
                        bechtold_key, 0xBEC4,  # coeff-ok: PRNG fold-in key
                    )
                else:
                    bechtold_key = None
                    master_key_new = None
                conv_out, prog_new_profile, stoch_new = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    u=u_col, v=v_col,
                    conv_prog_profile=prog_in,
                    conv_stoch_state=stoch_in,
                    prng_key=bechtold_key,
                    dt=dt, config=scheme_config,
                    moisture_convergence=mc_col,
                    # GLOBAL column ids for the decomposition-invariant
                    # per-column draw (a lat-band SPMD shard's carry chunk
                    # holds its own global ids); None => leaf arange.
                    col_index=(getattr(phys_state, "col_index", None)
                               if phys_state is not None else None),
                )
                # Multi-field carry update — return as dict so the
                # orchestrator can ``update`` both PhysicsState slots.
                conv_prog_out = {
                    "conv_prog_profile": prog_new_profile,
                    "conv_stoch_state": stoch_new,
                }
                if master_key_new is not None:
                    conv_prog_out["prng_key"] = master_key_new
            elif is_cmt_capable:
                # Tiedtke also consumes moisture_convergence; ZM does
                # not (its signature lacks the kwarg).
                if is_mc_consumer:
                    conv_out, prog_new_profile = conv_fn(
                        T=T_col, q_v=q_v_col,
                        p_full=p_full_col, p_half=p_half_col,
                        u=u_col, v=v_col,
                        conv_prog_profile=prog_in,
                        dt=dt, config=scheme_config,
                        moisture_convergence=mc_col,
                    )
                else:
                    conv_out, prog_new_profile = conv_fn(
                        T=T_col, q_v=q_v_col,
                        p_full=p_full_col, p_half=p_half_col,
                        u=u_col, v=v_col,
                        conv_prog_profile=prog_in,
                        dt=dt, config=scheme_config,
                    )
                conv_prog_out = prog_new_profile
            elif is_w_grid_consumer:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    w_grid=w_grid_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            else:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            dT_dt = conv_out.dT_dt.reshape(shape_3d)
        elif is_simple_mc_consumer:
            # Kuo: simple leaf that consumes the large-scale moisture
            # convergence as its source, plus the resolved ``w_grid`` for
            # the oracle's ``w_lcl>0`` gate (None → convergence-sign
            # proxy).  ``mc_col`` is None on single-column grids → Kuo is
            # quiescent (correct).
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
                moisture_convergence=mc_col,
                w_grid=w_grid_col,
            )
            dT_dt = conv_out.dT_dt.reshape(shape_3d)
        else:
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
            )
            dT_dt = conv_out.dT_dt.reshape(shape_3d)

        # Derive dim metadata from the input state so the returned
        # Field metadata matches the underlying grid (cubed-sphere
        # ("face","x","y",...), lat-lon ("lat","lon",...), or MPAS
        # ("nCells",...)).  Hardcoding "face","x","y" produced wrong
        # dim labels on lat-lon and MPAS runs.
        dims_3d = state.T.dims
        dims_2d = state.p_s.dims

        # Propagate tracer tendencies from convection backend.
        # Convective detrained condensate (``dq_c_conv_dt``) feeds the
        # cloud-water tracer; the dynamical core's tracer registry
        # picks it up by name (``q_c``) and applies it alongside the
        # microphysics tendency on the next step. Models without a
        # ``q_c`` tracer simply ignore the entry.
        # NOTE (TOA-drift fix): for an ADJUSTMENT scheme (sbm/dca/kuo,
        # ``detrains_to_cloud=False``) this column-net drying is convective
        # PRECIPITATION, not lingering cloud water — routing it to q_c is what
        # overwhelmed microphysics and drove the coupled cold drift.  The
        # ``PhysicsPipeline`` (coupled production) path now precipitates it
        # directly; this standalone dycore-integrated bridge has no surface-
        # precip accumulator so it RETAINS the q_c route (fine for idealized /
        # AMIP, where the implied precip simply leaves the prescribed surface;
        # a convective-precip path here is a tracked follow-up).
        tracer_tends = None
        if conv_fn is not None:
            dq_v_dt = conv_out.dq_v_dt.reshape(shape_3d)
            dq_c_conv_dt = conv_out.dq_c_conv_dt.reshape(shape_3d)
            # #929 in-updraft rain split: mass-flux schemes (bechtold/tiedtke)
            # emit an in-updraft RAIN source ``dq_r_conv_dt`` (the fraction of
            # detrained condensate diverted to precipitation by
            # ``precip_efficiency``) alongside the anvil-cloud source
            # ``dq_c_conv_dt``.  The unified PhysicsPipeline column-integrates
            # dq_r into same-step surface precip; this standalone bridge has no
            # surface-precip accumulator, so we CONSERVE it: route it to the
            # ``q_r`` rain tracer when the state has one (microphysics sediments
            # it), else fold it back into ``q_c`` so total convective condensate
            # (dq_c + dq_r) is preserved — byte-identical to the pre-split
            # all-condensate-to-cloud routing.  SIGN: ``dq_r_conv_dt >= 0`` is a
            # condensate SOURCE, the SAME sign as ``dq_c_conv_dt``.  Schemes with
            # no rain split emit ``None`` -> no-op (byte-identical).
            _dq_r_conv = conv_out.dq_r_conv_dt
            _has_qr = state.tracers is not None and "q_r" in state.tracers
            if _dq_r_conv is not None and not _has_qr:
                dq_c_conv_dt = dq_c_conv_dt + _dq_r_conv.reshape(shape_3d)
            tracer_tends = {
                "q_v": Field(
                    data=dq_v_dt, name="dq_v_dt_conv",
                    dims=dims_3d, units="kg/kg/s",
                ),
                "q_c": Field(
                    data=dq_c_conv_dt, name="dq_c_conv_dt",
                    dims=dims_3d, units="kg/kg/s",
                ),
            }
            if _dq_r_conv is not None and _has_qr:
                tracer_tends["q_r"] = Field(
                    data=_dq_r_conv.reshape(shape_3d), name="dq_r_conv_dt",
                    dims=dims_3d, units="kg/kg/s",
                )

        # Convective momentum transport (CMT): use the scheme's optional
        # ``du_dt_conv``/``dv_dt_conv`` when present (Zhang-McFarlane,
        # Tiedtke, Bechtold).  Schemes that do not produce CMT (the
        # existing five plus Kain-Fritsch and Emanuel) leave these as
        # ``None`` and the bridge zero-fills.  On MPAS where the
        # prognostic ``u`` lives on edges and ``state.v is None`` we
        # do not return a ``dv_dt`` Field — the orchestrator's
        # ``has_v = state.v is not None`` gate handles it.  ``du_dt``
        # is shaped to match the wind grid (edges on MPAS, cells
        # elsewhere), zeroed since CMT was disabled by the column-
        # extraction step above.
        u_target_shape = state.u.data.shape
        if conv_fn is not None and conv_out.du_dt_conv is not None:
            # Reshape only when the column-physics output matches the
            # u-grid layout.  When MPAS shifted CMT to zero (edge winds
            # not interpolated to cells) we keep the zero on the
            # u-grid layout instead of broadcasting cells back to edges.
            try:
                du_dt = conv_out.du_dt_conv.reshape(u_target_shape)
            except (TypeError, ValueError):
                du_dt = jnp.zeros(u_target_shape, dtype=_state_dtype)
        else:
            du_dt = jnp.zeros(u_target_shape, dtype=_state_dtype)
        dv_dt_field = None
        if state.v is not None:
            v_target_shape = state.v.data.shape
            if conv_fn is not None and conv_out.dv_dt_conv is not None:
                try:
                    dv_dt = conv_out.dv_dt_conv.reshape(v_target_shape)
                except (TypeError, ValueError):
                    dv_dt = jnp.zeros(v_target_shape, dtype=_state_dtype)
            else:
                dv_dt = jnp.zeros(v_target_shape, dtype=_state_dtype)
            dv_dt_field = Field(
                data=dv_dt, name="dv_dt_conv",
                dims=state.v.dims, units="m/s^2",
            )

        tendencies = HydrostaticTendencies(
            du_dt=Field(
                data=du_dt, name="du_dt_conv",
                dims=state.u.dims, units="m/s^2",
            ),
            dv_dt=dv_dt_field,
            dT_dt=Field(
                data=dT_dt, name="dT_dt_conv",
                dims=dims_3d, units="K/s",
            ),
            dp_s_dt=Field(
                data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dp_s_dt_conv",
                dims=dims_2d, units="Pa/s",
            ),
            dphis_dt=Field(
                data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dphis_dt_conv",
                dims=dims_2d, units="m^2/s^3",
            ),
            tracer_tendencies=tracer_tends,
        )
        return tendencies, conv_prog_out

    def reset_state():
        pass

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_convection(
    convection_config: ConvectionConfig,
    dt: float,
) -> Callable:
    """Create convection physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric, phys_state=None)
               -> (NonHydrostaticTendencies, conv_prog_profile_new | None)

    When *phys_state* is passed, the convective prognostic profile is
    read from ``phys_state.conv_prog_profile`` (shape ``(ncol, nlev)``)
    and the updated profile is returned as the second element of the
    result tuple.  See the hydrostatic bridge for the slice/pack
    convention for scalar-carrying schemes.
    """
    scheme_name, conv_fn, scheme_config = _get_convection_fn(convection_config)
    # Static plumbing traits — single source of truth shared across the
    # bridge factories and the unified driver pipeline.
    _tr = convection_scheme_traits(scheme_name)
    is_scalar_prognostic = _tr.is_scalar_prognostic
    is_profile_prognostic = _tr.is_profile_prognostic
    is_cmt_capable = _tr.is_cmt_capable
    is_w_grid_consumer = _tr.is_w_grid_consumer
    is_stochastic = _tr.is_stochastic
    is_mc_consumer = _tr.is_mc_consumer
    is_simple_mc_consumer = _tr.is_simple_mc_consumer
    needs_prng = is_stochastic and getattr(
        scheme_config, "enable_stochastic", False
    )
    prog_key = None
    prog_init = None
    if is_scalar_prognostic:
        if scheme_name == "mass_flux":
            prog_key, prog_init = "M_c", scheme_config.M_c_init
        else:  # edmf
            prog_key, prog_init = "a_u", scheme_config.a_u_init

    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        phys_state=None,
    ) -> NonHydrostaticTendencies:
        theta_p = state.theta_prime.data   # (6, n, n, nlev)
        rho_p = state.rho_prime.data       # (6, n, n, nlev)
        tracers = state.tracers.data       # (6, n, n, nlev, n_tracers)

        # Reference profiles
        theta_0 = height_coord.theta_ref
        rho_0 = height_coord.rho_ref

        # Total fields
        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p,
            rho_0 + rho_p,
        )

        # Temperature and pressure
        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape
        shape_w = state.w.data.shape
        shape_2d = state.phis.data.shape
        n_tracers = tracers.shape[-1] if tracers.ndim >= 5 else 0

        # Interface pressure from evolving column state (not fixed reference).
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        )

        # Reshape to columns
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        # Pin defaulted allocations to the state precision so x64 zeros
        # do not silently flow into the column physics path.
        _state_dtype = T.dtype
        _phis_dtype = state.phis.data.dtype
        # Extract q_v from tracers if available
        if n_tracers > 0:
            q_v_col = tracers[..., 0].reshape(ncol, nlev)
        else:
            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        if conv_fn is None:
            return NonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_conv", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_conv", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_conv", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dtheta_prime_dt_conv", dims=dims_3d, units="K/s"),
                drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_conv", dims=dims_3d, units="kg/m^3/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_conv", dims=dims_2d, units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_conv", dims=dims_tr, units="1/s"),
            )

        # Wind columns for CMT-capable schemes.
        if is_cmt_capable:
            u_col = state.u.data.reshape(ncol, nlev)
            v_col = state.v.data.reshape(ncol, nlev)
        else:
            u_col = None
            v_col = None

        # Grid-scale w for w-consuming schemes (KF trigger; Kuo's
        # ``w_lcl>0`` activation gate).  ``state.w`` lives at half levels
        # — interpolate to full-level centers.  Non-hydrostatic always
        # has a real prognostic ``w``, so Kuo uses it directly (faithful
        # to the oracle's independent-``w`` gate).
        if is_w_grid_consumer or is_simple_mc_consumer:
            w_data = state.w.data.reshape(ncol, nlev + 1)
            w_grid_col = 0.5 * (w_data[:, :-1] + w_data[:, 1:])
        else:
            w_grid_col = None

        # Moisture convergence for Tiedtke / Bechtold.  Non-hydrostatic
        # state stores tracers as a (face, n, n, nlev, n_tracers) array
        # with q_v at slot 0; reuse the cubed-sphere FV-flux-divergence
        # operator on slot 0.  When the scheme runs but tracers don't
        # carry q_v we pass ``None`` so the leaf engages its built-in
        # saturation-deficit proxy (Tiedtke gates the proxy on
        # ``moisture_convergence is None`` — zero-filling bypassed it).
        if (is_mc_consumer or is_simple_mc_consumer) and n_tracers > 0:
            _compute_mc = compute_moisture_convergence
            _qv_grid_full = tracers[..., 0]   # (face, n, n, nlev)
            mc_col = _compute_mc(
                _qv_grid_full, state.u.data, state.v.data, grid,
            )
        else:
            mc_col = None

        conv_prog_out = None
        if is_scalar_prognostic:
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile[:, -1]
            else:
                prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)

            conv_out, prog_new = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                **{prog_key: prog_in},
                dt=dt, config=scheme_config,
            )
            conv_prog_out = jnp.zeros(
                (ncol, nlev), dtype=_state_dtype
            ).at[:, -1].set(prog_new)
        elif is_profile_prognostic:
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile
            else:
                prog_in = jnp.zeros((ncol, nlev), dtype=_state_dtype)

            if is_stochastic:
                if phys_state is not None and (
                    phys_state.conv_stoch_state.shape == (ncol,)
                ):
                    stoch_in = phys_state.conv_stoch_state
                else:
                    stoch_in = jnp.zeros((ncol,), dtype=_state_dtype)
                # Mirror the hydrostatic bridge: derive a Bechtold
                # sub-key from the master phys_state PRNG key, and only
                # advance the master when stochasticity is enabled.
                if needs_prng and phys_state is not None and hasattr(
                    phys_state, "prng_key"
                ):
                    bechtold_key, master_key_new = jax.random.split(
                        phys_state.prng_key, 2,
                    )
                    bechtold_key = jax.random.fold_in(
                        bechtold_key, 0xBEC4,  # coeff-ok: PRNG fold-in key
                    )
                else:
                    bechtold_key = None
                    master_key_new = None
                conv_out, prog_new_profile, stoch_new = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    u=u_col, v=v_col,
                    conv_prog_profile=prog_in,
                    conv_stoch_state=stoch_in,
                    prng_key=bechtold_key,
                    dt=dt, config=scheme_config,
                    moisture_convergence=mc_col,
                    # GLOBAL column ids for the decomposition-invariant
                    # per-column draw (a lat-band SPMD shard's carry chunk
                    # holds its own global ids); None => leaf arange.
                    col_index=(getattr(phys_state, "col_index", None)
                               if phys_state is not None else None),
                )
                conv_prog_out = {
                    "conv_prog_profile": prog_new_profile,
                    "conv_stoch_state": stoch_new,
                }
                if master_key_new is not None:
                    conv_prog_out["prng_key"] = master_key_new
            elif is_cmt_capable:
                if is_mc_consumer:
                    conv_out, prog_new_profile = conv_fn(
                        T=T_col, q_v=q_v_col,
                        p_full=p_full_col, p_half=p_half_col,
                        u=u_col, v=v_col,
                        conv_prog_profile=prog_in,
                        dt=dt, config=scheme_config,
                        moisture_convergence=mc_col,
                    )
                else:
                    conv_out, prog_new_profile = conv_fn(
                        T=T_col, q_v=q_v_col,
                        p_full=p_full_col, p_half=p_half_col,
                        u=u_col, v=v_col,
                        conv_prog_profile=prog_in,
                        dt=dt, config=scheme_config,
                    )
                conv_prog_out = prog_new_profile
            elif is_w_grid_consumer:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    w_grid=w_grid_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            else:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
        elif is_simple_mc_consumer:
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
                moisture_convergence=mc_col,
                w_grid=w_grid_col,
            )
        else:
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
            )

        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner)
        dT_dt = conv_out.dT_dt.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        # Tracer tendencies. The non-hydrostatic state's tracer ordering
        # is documented on ``NonHydrostaticState`` in
        # ``state.py``:
        #   moist runs → tracers[..., 0] = q_vapor,
        #                tracers[..., 1] = q_cloud,
        #                tracers[..., 2] = q_rain.
        # Slot 0 (``q_v``) carries the convection vapor tendency; slot
        # 1 (``q_c``) carries the convective detrained-condensate
        # source so that microphysics processes it through
        # autoconversion / sedimentation / evaporation rather than the
        # previous instant-fall assumption (Option C). Models with
        # ``n_tracers < 2`` (dry or vapor-only runs) silently omit the
        # ``q_c`` write — there is no slot to receive it.
        # #929: a mass-flux rain-SPLITTING scheme (bechtold/tiedtke with
        # precip_efficiency>0) emits a separate in-updraft RAIN source
        # ``dq_r_conv_dt`` that MUST be booked — bechtold's dq_v is NOT
        # -(dq_c+dq_r) pointwise (separate compensating-subsidence + rain-evap
        # terms), so silently dropping dq_r here would leak column water.  A
        # condensate-less state (n_tracers < 2) has no q_c/q_r slot to receive
        # it -> unsupported config; raise LOUDLY (static Python on n_tracers +
        # dq_r-is-None; no silent coerce, no invented re-evaporation).  Only
        # fires when dq_r is present (sbm/dca/kuo emit None -> no raise).
        if conv_out.dq_r_conv_dt is not None and n_tracers < 2:
            raise ValueError(
                "convection emitted a rain-split source (dq_r_conv_dt) but the "
                f"non-hydro state has no condensate tracer (n_tracers={n_tracers} < 2) "
                "to receive it; a mass-flux rain-splitting scheme "
                "(bechtold/tiedtke, precip_efficiency>0) needs at least a q_c "
                "tracer. Set precip_efficiency=0 or add a condensate tracer."
            )
        dtracers = jnp.zeros_like(tracers)
        if n_tracers > 0:
            dq_v_dt = conv_out.dq_v_dt.reshape(shape_3d)
            dtracers = dtracers.at[..., 0].set(dq_v_dt)
        if n_tracers > 1:
            # See the hydrostatic-bridge TOA-drift note: an adjustment scheme's
            # (sbm/dca/kuo) convective drying is precipitation, not cloud water;
            # the PhysicsPipeline path precipitates it directly.  This bridge
            # retains the q_c route (no surface-precip accumulator here — an
            # AMIP/idealized follow-up).
            dq_c_conv_dt = conv_out.dq_c_conv_dt.reshape(shape_3d)
            # #929 in-updraft rain split: mass-flux schemes (bechtold/tiedtke)
            # also emit an in-updraft RAIN source ``dq_r_conv_dt``.  Slot 2 is
            # q_rain (state.py tracer ordering), so route the rain there when it
            # exists (microphysics sediments it — matches the unified pipeline's
            # precip routing); otherwise fold it into q_c (slot 1) so total
            # convective condensate (dq_c + dq_r) is CONSERVED, not dropped.
            # SIGN: ``dq_r_conv_dt >= 0`` is a condensate SOURCE, same sign as
            # ``dq_c_conv_dt``.  Schemes with no rain split emit ``None`` ->
            # no-op (byte-identical).
            _dq_r_conv = conv_out.dq_r_conv_dt
            if _dq_r_conv is not None:
                _dq_r_conv = _dq_r_conv.reshape(shape_3d)
                if n_tracers > 2:
                    dtracers = dtracers.at[..., 2].set(_dq_r_conv)
                else:
                    dq_c_conv_dt = dq_c_conv_dt + _dq_r_conv
            dtracers = dtracers.at[..., 1].set(dq_c_conv_dt)

        # CMT plumbing — see hydrostatic bridge for rationale.
        if conv_out.du_dt_conv is not None:
            du_dt_data = conv_out.du_dt_conv.reshape(shape_3d)
        else:
            du_dt_data = jnp.zeros(shape_3d, dtype=_state_dtype)
        if conv_out.dv_dt_conv is not None:
            dv_dt_data = conv_out.dv_dt_conv.reshape(shape_3d)
        else:
            dv_dt_data = jnp.zeros(shape_3d, dtype=_state_dtype)

        tendencies = NonHydrostaticTendencies(
            du_dt=Field(data=du_dt_data, name="du_dt_conv", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt_data, name="dv_dt_conv", dims=dims_3d, units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_conv", dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_conv", dims=dims_3d, units="K/s"),
            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_conv", dims=dims_3d, units="kg/m^3/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_conv", dims=dims_2d, units="m^2/s^3"),
            dtracers_dt=Field(data=dtracers, name="dtracers_dt_conv", dims=dims_tr, units="1/s"),
        )
        return tendencies, conv_prog_out

    def reset_state():
        pass

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_convection(
    convection_config: ConvectionConfig,
    dt: float,
) -> Callable:
    """Create convection physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord, grid_fields=None, phys_state=None)
               -> (SpectralHydrostaticState, conv_prog_profile_new | None)

    When *phys_state* is passed, the convective prognostic profile is
    read from ``phys_state.conv_prog_profile`` (shape ``(ncol, nlev)``)
    and the updated profile is returned as the second element of the
    result tuple.  See the hydrostatic bridge for the slice/pack
    convention.
    """
    scheme_name, conv_fn, scheme_config = _get_convection_fn(convection_config)
    # Static plumbing traits — single source of truth shared across the
    # bridge factories and the unified driver pipeline.
    _tr = convection_scheme_traits(scheme_name)
    is_scalar_prognostic = _tr.is_scalar_prognostic
    is_profile_prognostic = _tr.is_profile_prognostic
    is_cmt_capable = _tr.is_cmt_capable
    is_w_grid_consumer = _tr.is_w_grid_consumer
    is_stochastic = _tr.is_stochastic
    is_mc_consumer = _tr.is_mc_consumer
    is_simple_mc_consumer = _tr.is_simple_mc_consumer
    needs_prng = is_stochastic and getattr(
        scheme_config, "enable_stochastic", False
    )
    prog_key = None
    prog_init = None
    if is_scalar_prognostic:
        if scheme_name == "mass_flux":
            prog_key, prog_init = "M_c", scheme_config.M_c_init
        else:  # edmf
            prog_key, prog_init = "a_u", scheme_config.a_u_init

    def physics_fn(state, grid, sigma_coord, grid_fields=None, phys_state=None):
        # 1. Transform spectral state to grid space
        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        T = fields['T']         # (n_lat, n_lon, nlev)
        p_s = fields['p_s']     # (n_lat, n_lon)

        nlev = sigma_coord.n_levels
        n_lat, n_lon = p_s.shape

        # Pressure at full and half levels
        sigma_full = sigma_coord.sigma_full
        sigma_half = sigma_coord.sigma_half
        p_full = p_s[..., None] * sigma_full
        p_half = p_s[..., None] * sigma_half

        # Reshape to columns
        ncol = n_lat * n_lon
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        # Pin the column-physics dtype to the gridded state precision so
        # we never silently flow x64 zeros into the column path.
        _state_dtype = T.dtype
        # Extract water vapor if spectral state carries tracers.
        # ``SpectralHydrostaticState.tracers`` is a NamedTuple field
        # (``dict[str, Field] | None``) — no ``hasattr`` duck-typing
        # needed.
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(ncol, nlev)
        else:
            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        # Spectral PE columns for CMT-capable schemes — feed the actual
        # grid-space wind reconstructed by ``spectral_pe_to_grid`` so
        # the leaf produces meaningful CMT.  The grid→spectral round-trip
        # for the resulting (du_dt_conv, dv_dt_conv) is handled below
        # via :func:`vordiv_from_uv_3d`.
        if is_cmt_capable:
            u_grid = fields['u'].astype(_state_dtype)   # (n_lat, n_lon, nlev)
            v_grid = fields['v'].astype(_state_dtype)
            u_col = u_grid.reshape(ncol, nlev)
            v_col = v_grid.reshape(ncol, nlev)
        else:
            u_col = None
            v_col = None

        # Spectral PE has no native ``w`` field, but ``spectral_pe_to_grid``
        # already produces the horizontal divergence ``D = ∇·v_h`` per
        # full level — feed that through the standard sigma-coord
        # continuity (``compute_sigma_dot`` + ``compute_pressure_velocity``)
        # to build ``ω`` on the grid, then convert to ``w = -ω/(ρg)``
        # via :func:`._shared.diagnose_grid_w_from_omega`.  This makes
        # the KF trigger respond to dynamically-resolved low-level
        # convergence/divergence (the wedge of model behavior the
        # ``parcel_perturb_T``-only fallback is blind to).
        # Same diagnostic ``w`` also feeds Kuo's ``w_lcl>0`` activation
        # gate (faithful to the oracle's independent ``w``).
        if is_w_grid_consumer or is_simple_mc_consumer:
            div_grid = fields['div'].astype(_state_dtype)   # (n_lat, n_lon, nlev)
            sigma_top = sigma_coord.sigma_half[0]
            # Iter-55: share the cumsum between σ̇ and ``D_total``.
            sigma_dot_grid, _D_total_full = compute_sigma_dot_and_total(
                div_grid, sigma_coord,
            )
            dp_s_dt_grid = -p_s * _D_total_full[..., 0] / (1.0 - sigma_top)
            omega_grid = compute_pressure_velocity(
                sigma_dot_grid, p_s, dp_s_dt_grid, sigma_coord,
            )                                               # (n_lat, n_lon, nlev)
            w_grid_3d = diagnose_grid_w_from_omega(
                omega_grid.reshape(ncol, nlev),
                T_col, p_full_col, q_v_col,
            )                                               # (ncol, nlev)
            w_grid_col = w_grid_3d.astype(_state_dtype)
        else:
            w_grid_col = None

        # Moisture convergence on spectral PE.  Uses the transform
        # pathway in :func:`._shared.compute_moisture_convergence`
        # (GaussianGrid branch): synthesize ``q_v u`` and ``q_v v`` on
        # the grid, take the spectral divergence via
        # :func:`legoesm.grids.gaussian.vordiv_from_uv_3d`, synthesize
        # back, and negate.  When the spectral state has no ``q_v``
        # tracer in ``state.tracers`` we pass ``None`` so the leaf
        # engages its built-in saturation-deficit proxy (Tiedtke gates
        # the proxy on ``moisture_convergence is None`` — zero-filling
        # silently bypassed it).
        if (
            (is_mc_consumer or is_simple_mc_consumer)
            and state.tracers is not None
            and "q_v" in state.tracers
        ):
            _compute_mc = compute_moisture_convergence
            _qv_raw = state.tracers["q_v"]
            _qv_grid = (
                _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            ).astype(_state_dtype)
            u_grid_for_mc = fields['u'].astype(_state_dtype)
            v_grid_for_mc = fields['v'].astype(_state_dtype)
            mc_col = _compute_mc(
                _qv_grid, u_grid_for_mc, v_grid_for_mc, grid,
            ).astype(_state_dtype)
        else:
            mc_col = None

        conv_prog_out = None
        if conv_fn is None:
            dT_dt = jnp.zeros_like(T)
        elif is_scalar_prognostic:
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile[:, -1]
            else:
                prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)

            conv_out, prog_new = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                **{prog_key: prog_in},
                dt=dt, config=scheme_config,
            )
            conv_prog_out = jnp.zeros(
                (ncol, nlev), dtype=_state_dtype
            ).at[:, -1].set(prog_new)
            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)
        elif is_profile_prognostic:
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile
            else:
                prog_in = jnp.zeros((ncol, nlev), dtype=_state_dtype)
            if is_stochastic:
                if phys_state is not None and (
                    phys_state.conv_stoch_state.shape == (ncol,)
                ):
                    stoch_in = phys_state.conv_stoch_state
                else:
                    stoch_in = jnp.zeros((ncol,), dtype=_state_dtype)
                # Mirror the hydrostatic / non-hydrostatic bridges:
                # derive a Bechtold sub-key from the master phys_state
                # PRNG key, and only advance the master when
                # stochasticity is enabled.
                if needs_prng and phys_state is not None and hasattr(
                    phys_state, "prng_key"
                ):
                    bechtold_key, master_key_new = jax.random.split(
                        phys_state.prng_key, 2,
                    )
                    bechtold_key = jax.random.fold_in(
                        bechtold_key, 0xBEC4,  # coeff-ok: PRNG fold-in key
                    )
                else:
                    bechtold_key = None
                    master_key_new = None
                conv_out, prog_new_profile, stoch_new = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    u=u_col, v=v_col,
                    conv_prog_profile=prog_in,
                    conv_stoch_state=stoch_in,
                    prng_key=bechtold_key,
                    dt=dt, config=scheme_config,
                    moisture_convergence=mc_col,
                    # GLOBAL column ids for the decomposition-invariant
                    # per-column draw (a lat-band SPMD shard's carry chunk
                    # holds its own global ids); None => leaf arange.
                    col_index=(getattr(phys_state, "col_index", None)
                               if phys_state is not None else None),
                )
                conv_prog_out = {
                    "conv_prog_profile": prog_new_profile,
                    "conv_stoch_state": stoch_new,
                }
                if master_key_new is not None:
                    conv_prog_out["prng_key"] = master_key_new
            elif is_cmt_capable:
                # CMT-capable, non-stochastic profile schemes — ZM and
                # Tiedtke (Bechtold is stochastic and caught above).
                # Tiedtke also consumes ``moisture_convergence``; ZM
                # does not (its signature lacks the kwarg).  Mirrors the
                # hydrostatic / nonhydrostatic factory ordering — the
                # earlier flat dispatch dropped through to the catch-all
                # ``else`` and silently passed ``u``/``v`` to leaves
                # whose signature does not accept winds (Emanuel).
                if is_mc_consumer:
                    conv_out, prog_new_profile = conv_fn(
                        T=T_col, q_v=q_v_col,
                        p_full=p_full_col, p_half=p_half_col,
                        u=u_col, v=v_col,
                        conv_prog_profile=prog_in,
                        dt=dt, config=scheme_config,
                        moisture_convergence=mc_col,
                    )
                else:
                    conv_out, prog_new_profile = conv_fn(
                        T=T_col, q_v=q_v_col,
                        p_full=p_full_col, p_half=p_half_col,
                        u=u_col, v=v_col,
                        conv_prog_profile=prog_in,
                        dt=dt, config=scheme_config,
                    )
                conv_prog_out = prog_new_profile
            elif is_w_grid_consumer:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    w_grid=w_grid_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            else:
                # Profile-prognostic schemes that take neither winds
                # nor MC (Emanuel).  Pre-fix this branch passed ``u``
                # and ``v`` unconditionally, which raised TypeError on
                # Emanuel's wind-free leaf signature.
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)
        elif is_simple_mc_consumer:
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
                moisture_convergence=mc_col,
                w_grid=w_grid_col,
            )
            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)
        else:
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
            )
            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)

        # Transform T tendency to spectral space.
        dT_hat = sh_analysis_3d(grid, dT_dt)

        # Tracer tendencies (q_v sink, q_c source from convection).
        # ``SpectralHydrostaticState.tracers`` is grid-space, so we
        # reshape the column-physics output ``(ncol, nlev)`` back to
        # ``(n_lat, n_lon, nlev)`` and stash it under the tracer name
        # the dycore RHS expects.  The dycore RHS adds these to its
        # own advective tracer tendencies during the SSP-RK stages.
        # When the input state has no tracer (state.tracers is None or
        # missing the relevant key) we drop the tendency — there is
        # no carry to write into.
        tracers_tend = None
        if conv_fn is not None and state.tracers is not None:
            _state_tracers = state.tracers
            tt = {}
            if "q_v" in _state_tracers:
                dq_v_dt_grid = conv_out.dq_v_dt.reshape(n_lat, n_lon, nlev)
                _qv_template = _state_tracers["q_v"]
                if hasattr(_qv_template, "data") and hasattr(_qv_template, "replace"):
                    tt["q_v"] = _qv_template.replace(
                        data=dq_v_dt_grid.astype(_qv_template.data.dtype)
                    )
                else:
                    tt["q_v"] = dq_v_dt_grid.astype(_qv_template.dtype)
            if "q_c" in _state_tracers:
                dq_c_dt_grid = conv_out.dq_c_conv_dt.reshape(n_lat, n_lon, nlev)
                _qc_template = _state_tracers["q_c"]
                if hasattr(_qc_template, "data") and hasattr(_qc_template, "replace"):
                    tt["q_c"] = _qc_template.replace(
                        data=dq_c_dt_grid.astype(_qc_template.data.dtype)
                    )
                else:
                    tt["q_c"] = dq_c_dt_grid.astype(_qc_template.dtype)
            # #929 in-updraft rain split: route the mass-flux rain source
            # ``dq_r_conv_dt`` (bechtold/tiedtke).  If the state has a ``q_r``
            # rain tracer, emit it there (microphysics sediments it — matches
            # the unified pipeline's precip routing); otherwise fold it into the
            # ``q_c`` tendency so total convective condensate (dq_c + dq_r) is
            # CONSERVED, not dropped (no surface-precip path in this bridge).
            # SIGN: ``dq_r_conv_dt >= 0`` is a condensate SOURCE, same sign as
            # ``dq_c_conv_dt``.  Schemes with no rain split emit ``None`` ->
            # no-op (byte-identical).
            _dq_r_conv = conv_out.dq_r_conv_dt
            if _dq_r_conv is not None:
                _dq_r_grid = _dq_r_conv.reshape(n_lat, n_lon, nlev)
                if "q_r" in _state_tracers:
                    _qr_template = _state_tracers["q_r"]
                    if hasattr(_qr_template, "data") and hasattr(_qr_template, "replace"):
                        tt["q_r"] = _qr_template.replace(
                            data=_dq_r_grid.astype(_qr_template.data.dtype)
                        )
                    else:
                        tt["q_r"] = _dq_r_grid.astype(_qr_template.dtype)
                elif "q_c" in tt:
                    _qc_t = tt["q_c"]
                    if hasattr(_qc_t, "data") and hasattr(_qc_t, "replace"):
                        tt["q_c"] = _qc_t.replace(
                            data=(_qc_t.data + _dq_r_grid.astype(_qc_t.data.dtype))
                        )
                    else:
                        tt["q_c"] = _qc_t + _dq_r_grid.astype(_qc_t.dtype)
                else:
                    # #929: neither a q_r nor a q_c tracer to hold the convective
                    # rain split; bechtold's dq_v is NOT -(dq_c+dq_r) pointwise,
                    # so dropping dq_r would leak column water.  Unsupported
                    # config -> raise LOUDLY (static Python on the tracer keys +
                    # dq_r-is-None; no silent coerce, no invented re-evaporation).
                    raise ValueError(
                        "convection emitted a rain-split source (dq_r_conv_dt) "
                        "but the spectral state has neither a q_r nor a q_c "
                        "tracer to receive the convective rain split; a "
                        "mass-flux rain-splitting scheme (bechtold/tiedtke, "
                        "precip_efficiency>0) needs at least a q_c tracer. Set "
                        "precip_efficiency=0 or add a condensate tracer."
                    )
            # Mirror untouched tracers as zeros so the dycore RHS sees a
            # complete tracer pytree (the orchestrator's accumulation
            # also requires matching keys across modules).
            zeros = zero_like_tracers(_state_tracers)
            if zeros is not None:
                for k, zv in zeros.items():
                    tt.setdefault(k, zv)
            tracers_tend = tt

        # Convective momentum transport: round-trip the grid CMT
        # tendencies through ``vordiv_from_uv_3d`` to obtain spectral
        # vor/div tendencies.  CMT-capable schemes (Zhang-McFarlane,
        # Tiedtke, Bechtold) emit ``du_dt_conv``/``dv_dt_conv`` in grid
        # space; non-CMT schemes leave both as ``None`` and we fall back
        # to zeros.  The forward transform is exact up to the n=0 mode,
        # which has no vor/div content on the sphere.
        zero_3d_spec = jnp.zeros_like(state.vor_hat.data)
        if (
            conv_fn is not None
            and conv_out.du_dt_conv is not None
            and conv_out.dv_dt_conv is not None
        ):
            du_dt_grid = conv_out.du_dt_conv.reshape(n_lat, n_lon, nlev)
            dv_dt_grid = conv_out.dv_dt_conv.reshape(n_lat, n_lon, nlev)
            dvor_dt_hat, ddiv_dt_hat = vordiv_from_uv_3d(
                grid, du_dt_grid, dv_dt_grid,
            )
            # Cast back to the spectral-state dtype so we don't silently
            # promote the assembled tendency.
            dvor_dt_hat = dvor_dt_hat.astype(state.vor_hat.data.dtype)
            ddiv_dt_hat = ddiv_dt_hat.astype(state.div_hat.data.dtype)
        else:
            dvor_dt_hat = zero_3d_spec
            ddiv_dt_hat = zero_3d_spec

        # No surface pressure tendency from convection
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        tendencies = SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=dvor_dt_hat),
            div_hat=state.div_hat.replace(data=ddiv_dt_hat),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            tracers=tracers_tend,
        )
        return tendencies, conv_prog_out

    def reset_state():
        pass

    physics_fn.reset_state = reset_state
    return physics_fn
