"""Neural network replacement / augmentation for physics parameterization.

Provides an Equinox-based column MLP that can serve as a drop-in
replacement for the traditional physics pipeline, or as a learned
correction in hybrid mode.  All functions produce outputs compatible
with ``PhysicsOutput`` from ``legoesm.driver.physics_pipeline``.

Architecture
------------
``NeuralPhysics`` is a column MLP that operates independently per
atmospheric column.  It is vmapped over the spatial dimensions at
call time, so the network itself is defined for a single column.

Input features per column (nlev * 4 + 4):
    T, u, v, q_v at each level  +  p_s  +  solar forcing scalar
    +  T_sfc (prescribed surface temperature: SST/sea-ice blend over
    ocean, lowest-level air T proxy over land)  +  sea-ice fraction

The two surface-forcing features are what give the learned physics a
prescribed-SST (AMIP) pathway: without them the network cannot respond
to interannual SST variability (AIMIP Phase-1 protocol).

Output per column (nlev * 4 + 6):
    dT_dt, dq_v_dt, dq_c_dt, dq_r_dt at each level
    + precip, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa, sw_down_toa

Integration modes
-----------------
1. **Pure neural**: ``make_neural_step_unified`` returns a function
   matching the ``step_unified`` signature from ``PhysicsPipeline``,
   suitable as a drop-in replacement in ``build_segment_fn``.

2. **Hybrid**: ``make_hybrid_step_unified`` blends traditional physics
   with a learned neural correction weighted by ``alpha``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm import constants
from legoesm.core.physics_output import PhysicsOutput
from legoesm.core.grid_adapters import ColumnAdapter

# Machine-checked scheme contract (see tests/test_physics_contracts.py). Learned
# column MLP -> no hard conservation guarantee.
__physics_contract__ = {
    "summary": (
        "Column MLP (Equinox) emitting full physics tendencies plus surface / "
        "TOA radiative fluxes per column; a pure-neural drop-in for the physics "
        "pipeline or an alpha-weighted learned correction (hybrid mode)."
    ),
    "inputs": {
        "T": "K", "u": "m/s", "v": "m/s", "q_v": "kg/kg",
        "p_s": "Pa", "solar": "W/m^2",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "precip": "kg/m^2/s",
        "sw_net_sfc": "W/m^2", "lw_net_sfc": "W/m^2",
        "sw_up_toa": "W/m^2", "lw_up_toa": "W/m^2", "sw_down_toa": "W/m^2",
    },
    "sign_convention": (
        "Learned mapping: no enforced sign or conservation. residual_scale "
        "(default 0.01) keeps an untrained network near zero tendency; in "
        "hybrid mode output = traditional + alpha * neural_correction, so "
        "alpha=0 recovers the traditional step exactly."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Rasp, Pritchard & Gentine (2018), PNAS 115(39), 9684-9689 -- "
        "column-MLP physics replacement (gridded step_unified + spectral adapters)"
    ),
    "idealized_test": (
        "tests/unit/test_learned_column.py; untrained NeuralPhysics "
        "(residual_scale=0.01) -> near-zero tendencies; hybrid alpha=0 "
        "reproduces the traditional step_unified; per-column vmap, no "
        "horizontal coupling"
    ),
}

# Neural-physics feature-normalization scales + default architecture (structural).
_NORM_T_K = 300.0
_NORM_WIND_M_S = 30.0
_NORM_SOLAR_W_M2 = 1400.0
_DEFAULT_HIDDEN_DIM = 256
_DEFAULT_RESIDUAL_SCALE = 0.01
# Radiation fluxes are O(100 W/m^2), not per-second tendencies, so the flux
# head outputs use a separate physical scale: raw network output (O(1)) x
# this maps to W/m^2.  Untrained -> ~0 (stable); training drives toward ERA5.
_DEFAULT_FLUX_OUTPUT_SCALE = 100.0
# Tendency-head SATURATION cap (in raw-output units).  The tendency head is a
# raw linear map x residual_scale; left unbounded, training grows its weights
# until a single step's tendency tips the multi-step moist rollout past CFL
# (saturation latent-heat feedback) into inf/nan — the NN-variant training
# blow-up.  tanh(raw/cap)*cap keeps the head ~linear and ~0 at init (untrained
# rollout = pure dynamics) but caps |tendency| at residual_scale*cap, removing
# the blow-up at its source while staying smoothly differentiable.
_DEFAULT_TENDENCY_CAP = 5.0
# Spectral column-adapter moisture-head rescale.  The NeuralPhysics rate head
# shares one residual_scale calibrated for TEMPERATURE tendencies (K/s).
# Moisture tendencies live ~3 orders lower (dT 1e-4 K/s vs dq 1e-7 kg/kg/s), so
# make_column_physics_fn multiplies the q_v head by this factor before it enters
# the tracer tendency; without it an UNTRAINED head emits O(1e-3 kg/kg/s) and
# q_v explodes within a 6 h rollout (probe 2026-07-03).  NOTE: the gridded
# make_neural_step_unified path does NOT apply this — its _unpack_column_output
# emits the full dq_v head as one of four PhysicsOutput tendency fields (the
# grid pipeline's tanh cap bounds it).  The two paths diverge here by design;
# unifying is a numerics change gated on a controlled column_nn re-run.
_Q_HEAD_TENDENCY_FACTOR = 1.0e-4



_OPTIONAL_3D_OUTPUT_FIELDS = (
    "du_dt",
    "dv_dt",
    "dq_i_dt",
    "dq_s_dt",
    "dq_g_dt",
    "dN_c_dt",
    "dN_r_dt",
    "dN_i_dt",
)
_PHYSICS_OUTPUT_FIELDS = set(getattr(PhysicsOutput, "_fields", ()))


def parse_step_unified_tail(args):
    """Support both legacy and conv_prog-extended step_unified signatures."""
    if len(args) == 19:  # coeff-ok: positional-arg count dispatch
        return None, args
    if len(args) == 20:  # coeff-ok: positional-arg count dispatch
        return args[0], args[1:]
    raise TypeError(
        "step_unified expected 19 positional tail arguments "
        "(legacy) or 20 (with conv_prog)"
    )


def build_physics_output_kwargs(
    *,
    dT_dt,
    dq_v_dt,
    dq_c_dt,
    dq_r_dt,
    precip,
    sw_net_sfc,
    lw_net_sfc,
    sw_up_toa,
    lw_up_toa,
    sw_down_toa,
    reference_3d,
    template=None,
    conv_prog=None,
):
    kwargs = dict(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        precip=precip,
        sw_net_sfc=sw_net_sfc,
        lw_net_sfc=lw_net_sfc,
        sw_up_toa=sw_up_toa,
        lw_up_toa=lw_up_toa,
        sw_down_toa=sw_down_toa,
    )
    zeros_3d = jnp.zeros_like(reference_3d)
    for field_name in _OPTIONAL_3D_OUTPUT_FIELDS:
        if field_name in _PHYSICS_OUTPUT_FIELDS:
            kwargs[field_name] = (
                getattr(template, field_name, zeros_3d)
                if template is not None else zeros_3d
            )
    if "conv_prog" in _PHYSICS_OUTPUT_FIELDS:
        conv_prog_value = (
            getattr(template, "conv_prog", conv_prog)
            if template is not None else conv_prog
        )
        if conv_prog_value is None:
            conv_prog_value = jnp.asarray(0.0, dtype=reference_3d.dtype)
        kwargs["conv_prog"] = conv_prog_value
    return kwargs


# ======================================================================
# Neural network module
# ======================================================================

class NeuralPhysics(eqx.Module):
    """Column MLP for learned physics tendencies.

    Operates on a single atmospheric column.  Use ``jax.vmap`` to apply
    over spatial dimensions.

    Parameters
    ----------
    nlev : int
        Number of vertical levels.
    hidden_dim : int
        Width of each hidden layer.
    n_layers : int
        Number of hidden layers (total layers = n_layers + 1).
    key : jax.Array
        PRNG key for weight initialization.
    residual_scale : float
        Multiplicative scale applied to the raw tendency + precip outputs.
        Defaults to 0.01 so that an untrained network produces
        near-zero tendencies, preventing instability.
    flux_output_scale : float
        Separate scale for the 5 radiation-flux outputs (W/m^2), which are
        O(100), not per-second rates.  Defaults to 100 so the flux head can
        reach observed magnitudes while untrained output stays ~0.
    """

    layers: list
    nlev: int = eqx.field(static=True)
    n_input: int = eqx.field(static=True)
    n_output: int = eqx.field(static=True)
    residual_scale: float = eqx.field(static=True)
    flux_output_scale: float = eqx.field(static=True)
    tendency_cap: float = eqx.field(static=True)

    def __init__(
        self,
        nlev: int,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        n_layers: int = 4,
        *,
        key: jax.Array,
        residual_scale: float = _DEFAULT_RESIDUAL_SCALE,
        flux_output_scale: float = _DEFAULT_FLUX_OUTPUT_SCALE,
        tendency_cap: float = _DEFAULT_TENDENCY_CAP,
    ):
        self.nlev = nlev
        # T, u, v, q_v per level + p_s + solar + T_sfc + sea-ice fraction.
        # The last two are prescribed surface forcings (AMIP SST pathway).
        self.n_input = nlev * 4 + 4
        self.n_output = nlev * 4 + 6   # tendencies per level + 6 surface fluxes
        self.residual_scale = residual_scale
        self.flux_output_scale = flux_output_scale
        self.tendency_cap = tendency_cap

        keys = jax.random.split(key, n_layers + 1)
        dims = [self.n_input] + [hidden_dim] * n_layers + [self.n_output]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(n_layers + 1)
        ]
        # Zero-init the FINAL layer so an untrained network emits EXACTLY
        # zero tendencies (epoch-0 rollout = the pure dycore, finite by
        # construction).  residual_scale alone is not enough: random O(1)
        # outputs x 0.01 are still ~0.04 in physical tendency units, which
        # destroys q_v (~1e-3 kg/kg) within a few dycore steps (#797
        # neural_gcm smoke loss=nan).  Learning is unaffected — the final
        # layer's gradient is nonzero on step 1, after which gradients
        # reach the earlier (randomly initialized) layers.
        last = self.layers[-1]
        self.layers[-1] = eqx.tree_at(
            lambda l: (l.weight, l.bias), last,
            (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass for a single column.

        Parameters
        ----------
        x : jax.Array, shape (n_input,)
            Concatenated column input features.

        Returns
        -------
        jax.Array, shape (n_output,)
            Raw network output (before unpacking into PhysicsOutput).
        """
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        raw = self.layers[-1](x)
        # Tendencies (4*nlev) + precip are per-second rates -> residual_scale,
        # tanh-BOUNDED so a trained weight blow-up can never push a single
        # step's tendency past CFL into an inf/nan moist rollout (the NN-variant
        # training crash). The 5 radiation fluxes (sw_net_sfc, lw_net_sfc,
        # sw_up_toa, lw_up_toa, sw_down_toa) are O(100 W/m^2) ->
        # flux_output_scale (they do not drive the state, so left unbounded).
        n_rate = self.nlev * 4 + 1  # tendencies + precip
        cap = self.tendency_cap
        rate = self.residual_scale * cap * jnp.tanh(raw[:n_rate] / cap)
        return jnp.concatenate([rate, raw[n_rate:] * self.flux_output_scale])


# ======================================================================
# Column feature packing / unpacking
# ======================================================================

def pack_column_features(
    T: jax.Array,
    u: jax.Array,
    v: jax.Array,
    q_v: jax.Array,
    p_s: jax.Array,
    solar: jax.Array,
    t_sfc: jax.Array,
    sic: jax.Array,
) -> jax.Array:
    """Pack column state + surface forcing into a flat feature vector.

    All inputs are for a single column:
        T, u, v, q_v : shape (nlev,)
        p_s, solar, t_sfc, sic : scalars

    ``t_sfc`` is the prescribed surface temperature (SST/sea-ice blend
    over ocean per the AMIP protocol; lowest-level air T proxy over
    land) and ``sic`` the sea-ice fraction in [0, 1] (0 over land).
    These give the learned physics its prescribed-SST response — the
    interannual-variability pathway.

    Returns shape (nlev * 4 + 4,).
    """
    # Normalize to O(1) for stable training
    return jnp.concatenate([
        T / _NORM_T_K,
        u / _NORM_WIND_M_S,
        v / _NORM_WIND_M_S,
        q_v * 1e3,
        jnp.atleast_1d(p_s / constants.p_ref),
        jnp.atleast_1d(solar / _NORM_SOLAR_W_M2),
        jnp.atleast_1d(t_sfc / _NORM_T_K),
        jnp.atleast_1d(sic),
    ])


def _unpack_column_output(
    y: jax.Array,
    nlev: int,
) -> PhysicsOutput:
    """Unpack raw network output into a PhysicsOutput for one column.

    Parameters
    ----------
    y : jax.Array, shape (nlev * 4 + 6,)
    nlev : int

    Returns
    -------
    PhysicsOutput
        All fields are single-column: 3-D tendencies have shape (nlev,),
        surface fluxes are scalars.
    """
    dT_dt = y[:nlev]
    dq_v_dt = y[nlev:2 * nlev]
    dq_c_dt = y[2 * nlev:3 * nlev]
    dq_r_dt = y[3 * nlev:4 * nlev]
    sfc = y[4 * nlev:]
    return PhysicsOutput(
        **build_physics_output_kwargs(
            dT_dt=dT_dt,
            dq_v_dt=dq_v_dt,
            dq_c_dt=dq_c_dt,
            dq_r_dt=dq_r_dt,
            precip=sfc[0],
            sw_net_sfc=sfc[1],
            lw_net_sfc=sfc[2],
            sw_up_toa=sfc[3],
            lw_up_toa=sfc[4],
            sw_down_toa=sfc[5],
            reference_3d=dT_dt,
        )
    )


# ======================================================================
# Shared column preprocessing + forward (both dycore adapters)
# ======================================================================

def neural_column_forward(
    neural_physics: NeuralPhysics,
    T_col: jax.Array,
    u_col: jax.Array,
    v_col: jax.Array,
    q_col: jax.Array,
    p_s_col: jax.Array,
    solar_col: jax.Array,
    sst_col: jax.Array,
    sic_col: jax.Array,
    treat_nonpositive_sst_as_missing: bool = True,
) -> jax.Array:
    """Select surface T, sanitize sea-ice, pack features, vmap the network.

    Shared by BOTH dycore adapters — gridded ``make_neural_step_unified`` and
    spectral ``make_column_physics_fn`` — so one column-preprocessing path
    feeds one network with no divergence between the AIMIP spectral and the
    WeatherBench / lat-lon step_unified pipelines.

    All arrays are per-column (already flattened):
        T_col, u_col, v_col, q_col : (ncol, nlev)
        p_s_col, solar_col, sst_col, sic_col : (ncol,)

    ``sst_col`` is the prescribed surface temperature; where it is non-finite
    it is replaced by the lowest-level air-T proxy.  ``treat_nonpositive_sst_
    as_missing`` (default True) ALSO routes finite non-positive values to the
    proxy — the gridded pipelines pass zeros, not NaN, for "no SST" and 0 K is
    never a physical temperature.  The spectral path passes NaN over land and
    real Kelvin SST over ocean, so it keeps the flag False to preserve its
    finite-only convention exactly.  ``nan_to_num`` runs BEFORE the select
    because ``jnp.where`` propagates NaN cotangents from the untaken branch in
    reverse mode (codex HIGH).  ``sic_col`` -> nan->0, clipped to [0, 1].

    Returns raw network output (ncol, n_output).
    """
    valid = jnp.isfinite(sst_col)
    if treat_nonpositive_sst_as_missing:
        valid = valid & (sst_col > 0.0)
    t_sfc_col = jnp.where(
        valid, jnp.nan_to_num(sst_col, nan=0.0), T_col[:, -1],
    )
    sic_flat = jnp.clip(jnp.nan_to_num(sic_col, nan=0.0), 0.0, 1.0)
    features = jax.vmap(pack_column_features)(
        T_col, u_col, v_col, q_col, p_s_col, solar_col, t_sfc_col, sic_flat,
    )
    return jax.vmap(neural_physics)(features)


# ======================================================================
# Neural step_unified builder
# ======================================================================

def make_neural_step_unified(
    neural_physics: NeuralPhysics,
    adapter: ColumnAdapter,
):
    """Build a ``step_unified``-compatible function from a NeuralPhysics model.

    The returned function has the same signature as the one produced by
    ``PhysicsPipeline.build_step_unified()``, so it can be used as a
    drop-in replacement in ``build_segment_fn``.

    Parameters
    ----------
    neural_physics : NeuralPhysics
        Trained (or untrained) Equinox column MLP.
    adapter : ColumnAdapter
        Grid adapter for flatten/unflatten operations.

    Returns
    -------
    callable
        Supports both the legacy signature
        ``step_unified(..., q_r, u, v, ...)`` and the newer
        ``step_unified(..., q_r, conv_prog, u, v, ...)`` layout.
    """
    nlev = neural_physics.nlev

    def step_unified(need_rad, T, p_s, q_v, q_c, q_r, *args, **kwargs):
        conv_prog, tail = parse_step_unified_tail(args)
        (
            u,
            v,
            sst,
            sic,
            lat,
            lon,
            day_of_year,
            seconds_of_day,
            dt,
            solar_weights,
            s_0,
            o3_vmr,
            aerosol_od,
            held_dT_rad,
            held_sw_net_sfc,
            held_lw_net_sfc,
            held_sw_up_toa,
            held_lw_up_toa,
            held_sw_down_toa,
        ) = tail
        del need_rad, dt
        del solar_weights, o3_vmr, aerosol_od, kwargs
        # Flatten to columns
        T_col = adapter.flatten_3d(T)           # (ncol, nlev)
        u_col = adapter.flatten_3d(u)           # (ncol, nlev)
        v_col = adapter.flatten_3d(v)           # (ncol, nlev)
        q_v_col = adapter.flatten_3d(q_v)       # (ncol, nlev)
        p_s_flat = adapter.flatten_2d(p_s)      # (ncol,)

        # Real TOA insolation per column (seasonal + diurnal cycle) via the
        # shared orbital helper — the NN's only time-of-year signal.
        from legoesm.atmosphere.physics.radiation.solar import cos_zenith_angle
        mu0 = cos_zenith_angle(
            adapter.flatten_2d(lat), adapter.flatten_2d(lon),
            day_of_year, seconds_of_day / 3600.0,
        )
        solar_flat = s_0 * jnp.maximum(mu0, 0.0)

        # Shared column preprocessing + forward (see neural_column_forward):
        # SST->t_sfc proxy select, sea-ice sanitize, feature pack, vmap net.
        # Same convention as the AIMIP spectral path so one trained network
        # serves both pipelines.  (ncol, n_output)
        y = neural_column_forward(
            neural_physics, T_col, u_col, v_col, q_v_col, p_s_flat,
            solar_flat, adapter.flatten_2d(sst), adapter.flatten_2d(sic),
        )

        # Unpack into per-column PhysicsOutput, then unflatten
        col_out = jax.vmap(lambda yi: _unpack_column_output(yi, nlev))(y)

        phys_out = PhysicsOutput(
            **build_physics_output_kwargs(
                dT_dt=adapter.unflatten_3d(col_out.dT_dt),
                dq_v_dt=adapter.unflatten_3d(col_out.dq_v_dt),
                dq_c_dt=adapter.unflatten_3d(col_out.dq_c_dt),
                dq_r_dt=adapter.unflatten_3d(col_out.dq_r_dt),
                precip=adapter.unflatten_2d(col_out.precip),
                sw_net_sfc=adapter.unflatten_2d(col_out.sw_net_sfc),
                lw_net_sfc=adapter.unflatten_2d(col_out.lw_net_sfc),
                sw_up_toa=adapter.unflatten_2d(col_out.sw_up_toa),
                lw_up_toa=adapter.unflatten_2d(col_out.lw_up_toa),
                sw_down_toa=adapter.unflatten_2d(col_out.sw_down_toa),
                reference_3d=T,
                conv_prog=conv_prog,
            )
        )

        # Flux head: write the network's predicted TOA/surface radiation
        # fluxes into held_* so the radiation-flux loss supervises them (the
        # NN learns to radiate like ERA5 -> generalizes to a new climate).
        # held_dT_rad stays at its IC value (0): the NN's dT_dt is the TOTAL
        # tendency and already includes radiative heating, so applying a
        # separate held_dT_rad would double-count.  sw_down_toa (insolation)
        # is an external forcing, not predicted -> passthrough.
        del held_sw_net_sfc, held_lw_net_sfc, held_sw_up_toa, held_lw_up_toa
        held_new = (
            held_dT_rad,
            phys_out.sw_net_sfc, phys_out.lw_net_sfc,
            phys_out.sw_up_toa, phys_out.lw_up_toa,
            held_sw_down_toa,
        )
        return phys_out, held_new

    return step_unified


# ======================================================================
# Hybrid step_unified builder
# ======================================================================

def make_hybrid_step_unified(
    traditional_step_unified,
    neural_physics: NeuralPhysics,
    adapter: ColumnAdapter,
    alpha: float = 1.0,
):
    """Build a hybrid ``step_unified`` that blends traditional + neural physics.

    The output is::

        phys_out = traditional_out + alpha * neural_correction

    for all tendency fields.  Held radiation arrays are taken from the
    traditional branch (which handles radiation sub-cycling properly).

    Parameters
    ----------
    traditional_step_unified : callable
        The original ``step_unified`` from ``PhysicsPipeline.build_step_unified()``.
    neural_physics : NeuralPhysics
        Equinox column MLP providing learned corrections.
    adapter : ColumnAdapter
        Grid adapter for flatten/unflatten operations.
    alpha : float
        Blending weight for the neural correction. Use 0.0 to disable
        the neural component, 1.0 for full neural correction.

    Returns
    -------
    callable
        Same signature as ``step_unified``.
    """
    neural_step = make_neural_step_unified(neural_physics, adapter)

    def step_unified(need_rad, T, p_s, q_v, q_c, q_r, *args, **kwargs):
        conv_prog, tail = parse_step_unified_tail(args)
        (
            u,
            v,
            sst,
            sic,
            lat,
            lon,
            day_of_year,
            seconds_of_day,
            dt,
            solar_weights,
            s_0,
            o3_vmr,
            aerosol_od,
            held_dT_rad,
            held_sw_net_sfc,
            held_lw_net_sfc,
            held_sw_up_toa,
            held_lw_up_toa,
            held_sw_down_toa,
        ) = tail
        # Traditional physics (with full radiation sub-cycling)
        trad_args = [need_rad, T, p_s, q_v, q_c, q_r]
        if conv_prog is not None:
            trad_args.append(conv_prog)
        trad_args.extend([
            u, v, sst, sic, lat, lon,
            day_of_year, seconds_of_day, dt,
            solar_weights, s_0,
            o3_vmr, aerosol_od,
            held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
            held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
        ])
        _trad = traditional_step_unified(*trad_args, **kwargs)
        trad_out, held_new = _trad[0], _trad[1]
        # The traditional PhysicsPipeline step returns a 3rd value (the
        # slab-land skin temperature, #325); propagate it so a neural-
        # correction run with an active land tile still evolves T_land.
        # Older 2-tuple traditional steps leave it None (land inert).
        _trad_T_land = _trad[2] if len(_trad) > 2 else None

        # Neural correction
        neural_out, _ = neural_step(*trad_args)

        # Blend: traditional + alpha * neural correction
        _alpha = jnp.asarray(alpha)
        blended = PhysicsOutput(
            **build_physics_output_kwargs(
                dT_dt=trad_out.dT_dt + _alpha * neural_out.dT_dt,
                dq_v_dt=trad_out.dq_v_dt + _alpha * neural_out.dq_v_dt,
                dq_c_dt=trad_out.dq_c_dt + _alpha * neural_out.dq_c_dt,
                dq_r_dt=trad_out.dq_r_dt + _alpha * neural_out.dq_r_dt,
                precip=trad_out.precip + _alpha * neural_out.precip,
                sw_net_sfc=trad_out.sw_net_sfc + _alpha * neural_out.sw_net_sfc,
                lw_net_sfc=trad_out.lw_net_sfc + _alpha * neural_out.lw_net_sfc,
                sw_up_toa=trad_out.sw_up_toa + _alpha * neural_out.sw_up_toa,
                lw_up_toa=trad_out.lw_up_toa + _alpha * neural_out.lw_up_toa,
                sw_down_toa=trad_out.sw_down_toa + _alpha * neural_out.sw_down_toa,
                reference_3d=trad_out.dT_dt,
                template=trad_out,
                conv_prog=conv_prog,
            )
        )

        # Held radiation comes from the traditional branch; pass the
        # slab-land skin temperature through unchanged (#325).
        return blended, held_new, _trad_T_land

    return step_unified


# ======================================================================
# Spectral-PE column adapter (Rasp et al. 2018)
# ======================================================================
# Folded in from the former learned_column.py: the SAME NeuralPhysics network,
# wrapped for the spectral primitive-equation dycore instead of the gridded
# step_unified pipeline.  Column preprocessing is shared via
# neural_column_forward; only the spectral<->grid transforms and the T/q_v
# tendency packing are adapter-specific.

def build_column_physics(
    nlev: int,
    hidden_dim: int = _DEFAULT_HIDDEN_DIM,
    n_layers: int = 4,
    residual_scale: float = _DEFAULT_RESIDUAL_SCALE,
    *,
    key: jax.Array,
) -> NeuralPhysics:
    """Create a column MLP physics model for the spectral PE dycore.

    Thin factory over ``NeuralPhysics``; kept as the spectral-path entry point
    (Rasp, Pritchard & Gentine 2018) so callers need not know the network's
    constructor kwargs.

    Parameters
    ----------
    nlev : int
        Number of vertical levels.
    hidden_dim, n_layers : int
        Width and depth of the column MLP.
    residual_scale : float
        Output scaling for stable init (untrained -> near-zero tendencies).
    key : jax.Array
        PRNG key for weight initialization.
    """
    return NeuralPhysics(
        nlev=nlev,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        key=key,
        residual_scale=residual_scale,
    )


def make_column_physics_fn(
    neural_physics: NeuralPhysics,
    grid,
    momentum_physics_fn=None,
):
    """Create a spectral PE ``physics_fn`` from a column MLP (Rasp et al. 2018).

    Returns a function with the interface expected by
    ``SpectralPrimitiveEquationModel.step(physics_fn=...)``::

        physics_fn(state, grid, sigma_coord, forcing=None)
            -> SpectralHydrostaticState  (tendencies)

    The column MLP operates per-column via ``jax.vmap`` (no horizontal
    coupling — the spectral dycore handles resolved transport):

    1. Spectral -> grid (SH synthesis)
    2. Flatten to columns, build surface forcing, forward (neural_column_forward)
    3. Extract dT/dt (+ rescaled dq_v/dt) and SH-analyse back to spectral

    MOMENTUM (#1464).  The column MLP has no momentum head, so on its own this
    adapter returns ZERO vor/div tendencies — the arm has **no surface
    turbulent drag**.

    STATED PRECISELY, because the first version of this note overclaimed and
    adversarial review refuted it: this is NOT "no momentum sink of any kind".
    The spectral PE RHS hyperdiffuses vor and div independently of
    ``physics_fn``, and the WB campaign additionally applies a per-step
    high-wavenumber spectral filter, so both arms carry substantial NUMERICAL
    momentum damping.  What the learned arm lacks is the PHYSICAL surface
    stress that the classical arm gets from its inherited
    ``TurbulenceConfig.scheme = "smagorinsky"``.

    That asymmetry is real and worth removing, but it is a HYPOTHESIS for the
    T106 result, not a demonstration: the degradation there ranked as a missing
    drag would predict (``wind_speed_10m`` 3.83x, ``u10`` 3.02x at 240 h,
    monotone in height, ``z500`` the only field that improved), and the
    decisive test is a controlled re-score, not this docstring.

    Pass ``momentum_physics_fn`` — any ``physics_fn(state, grid, sigma_coord)``
    returning a tendency state — to supply the vor/div tendencies the network
    cannot.  ``None`` keeps the historical zero behaviour bit-identically.

    ``lnps`` is deliberately NOT taken from that source.  Surface stress has no
    surface-pressure tendency; the PE already forms ``dlnps/dt`` from
    continuity and merely ADDS any physics contribution, so letting an
    arbitrary momentum source inject a mean ``dlnps/dt`` would break dry-mass
    conservation with mass anchoring off (codex).  T and the tracers likewise
    always come from the network.

    NOT A CLOSED BUDGET, deliberately: taking a turbulence scheme's stress
    while dropping its heat flux is a hybrid, not "the same physics with a
    different parameterisation".  The alternative — importing the scheme's
    ``dT/dt`` too — would put a second thermodynamic parameterisation next to
    the network, which is the confound this exists to remove.
    """
    # Deferred imports (spectral dycore + Gaussian grid) to avoid a physics <->
    # dynamics package import cycle at module load; run once at adapter build.
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralHydrostaticState,
        spectral_pe_to_grid,
    )
    from legoesm.grids.gaussian import sh_analysis_3d
    from legoesm.atmosphere.physics.radiation.solar import cos_zenith_angle
    from legoesm.atmosphere.physics._shared import zero_like_tracers

    nlev = neural_physics.nlev

    def physics_fn(state, grid_, sigma_coord, forcing=None):
        # Spectral -> grid-space fields
        fields = spectral_pe_to_grid(state, grid_, sigma_coord)
        T = fields['T']          # (n_lat, n_lon, nlev)
        u = fields['u']
        v = fields['v']
        p_s = fields['p_s']      # (n_lat, n_lon)

        n_lat, n_lon = T.shape[:2]

        T_col = T.reshape(-1, nlev)
        u_col = u.reshape(-1, nlev)
        v_col = v.reshape(-1, nlev)
        # Pull q_v from state.tracers when present (spectral PE tracers); fall
        # back to zeros for the legacy dry pipeline, else the column MLP sees
        # dry inputs even when ERA5 humidity is in the IC.
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_col = _qv_data.reshape(-1, nlev).astype(T_col.dtype)
        else:
            q_col = jnp.zeros_like(T_col)
        p_s_col = p_s.reshape(-1)

        # Surface forcing -> (solar, sst, sic) columns.  forcing dict (all
        # traced; see spectral_amip_rollout): "T_sfc" (NaN over land), "sic",
        # "day_of_year", "seconds_of_day".  None branch (idealized / legacy):
        # constant insolation, NaN SST (-> lowest-air proxy in
        # neural_column_forward), zero sea-ice.
        if forcing is not None:
            mu0 = cos_zenith_angle(
                jnp.broadcast_to(grid_.lat[:, None], (n_lat, n_lon)).reshape(-1),
                jnp.broadcast_to(grid_.lon[None, :], (n_lat, n_lon)).reshape(-1),
                forcing["day_of_year"], forcing["seconds_of_day"] / 3600.0,
            )
            solar_col = constants.S_0 * jnp.maximum(mu0, 0.0)
            sst_col = forcing["T_sfc"]
            sic_col = forcing["sic"]
        else:
            solar_col = jnp.full_like(p_s_col, constants.S_0)
            # NaN (not zero): finite-only select below -> lowest-air proxy,
            # exactly the pre-refactor unforced branch (t_sfc = T_col[:, -1]).
            sst_col = jnp.full_like(p_s_col, jnp.nan)
            sic_col = jnp.zeros_like(p_s_col)

        # Spectral convention: T_sfc is NaN over land, real Kelvin over ocean;
        # a finite value is always valid (finite-only select, no >0 gate) to
        # match the pre-refactor learned_column behavior exactly.
        y = neural_column_forward(
            neural_physics, T_col, u_col, v_col, q_col, p_s_col,
            solar_col, sst_col, sic_col,
            treat_nonpositive_sst_as_missing=False,
        )  # (ncol, n_output)

        # dT/dt (first nlev outputs) -> spectral temperature tendency
        dT_dt = y[:, :nlev].reshape(n_lat, n_lon, nlev)
        dT_hat = sh_analysis_3d(grid_, dT_dt.astype(jnp.float64))

        # Column physics: T tendency (spectral) + q_v tendency (grid-space
        # tracer path); zero for vor, div, lnps.  Mirror remaining tracers as
        # zeros so orchestrator + dycore RHS see a consistent tendency pytree.
        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        # Momentum + lnps: from the supplied physics when given, else zero
        # (#1464 -- see the factory docstring for why zero is not neutral).
        # Python-level branch on a build-time argument, so only one side is
        # ever traced (the CLAUDE.md feature-gating exception).
        if momentum_physics_fn is None:
            vor_t, div_t = zero_3d, zero_3d
        else:
            _mom = momentum_physics_fn(state, grid_, sigma_coord)
            vor_t = _mom.vor_hat.data
            div_t = _mom.div_hat.data
        # lnps stays ZERO regardless of the source — see the factory docstring
        # (a momentum source has no surface-pressure tendency, and injecting a
        # mean dlnps/dt here would break dry-mass conservation).
        lnps_t = zero_2d

        tracers_out = zero_like_tracers(state.tracers)
        if tracers_out is not None and "q_v" in tracers_out:
            # dq_v/dt from the moisture head (outputs nlev:2*nlev), rescaled to
            # moisture magnitudes (_Q_HEAD_TENDENCY_FACTOR).
            dq_v_dt = (
                y[:, nlev:2 * nlev] * _Q_HEAD_TENDENCY_FACTOR
            ).reshape(n_lat, n_lon, nlev)
            template = state.tracers["q_v"]
            if hasattr(template, "data") and hasattr(template, "replace"):
                tracers_out["q_v"] = template.replace(
                    data=dq_v_dt.astype(template.data.dtype),
                )
            else:
                tracers_out["q_v"] = dq_v_dt.astype(template.dtype)

        return SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=vor_t),
            div_hat=state.div_hat.replace(data=div_t),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=lnps_t),
            # phis is orography: a physics tendency on it is always zero.
            phis_hat=state.phis_hat.replace(data=zero_2d),
            tracers=tracers_out,
        )

    return physics_fn
