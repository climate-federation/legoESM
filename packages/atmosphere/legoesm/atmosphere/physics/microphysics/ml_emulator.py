"""ML microphysics emulator using Equinox MLP.

Provides a neural network surrogate for microphysics that is fully
differentiable with jax.grad. The MLP maps per-level thermodynamic
inputs to per-level microphysics tendencies.

The model is initialized lazily on first call using the seed from
MicrophysicsMLEmulatorConfig. An untrained model returns near-zero tendencies
(suitable for testing).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    hard_saturation_blend,
)
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsMLEmulatorConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
)


# Emulator raw-output scaling.
_OUTPUT_SCALE = 0.01
_PRECIP_SCALE = 1e-3

class MicrophysicsEmulator(eqx.Module):
    """Multi-layer perceptron microphysics emulator."""
    layers: list

    def __init__(self, n_input, n_hidden, n_layers, n_output, *, key):
        keys = jax.random.split(key, n_layers)
        dims = [n_input] + [n_hidden] * (n_layers - 1) + [n_output]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(n_layers)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


__physics_contract__ = {
    "summary": (
        "Learned (Equinox-MLP) microphysics emulator: predicts the column "
        "microphysical tendencies + surface precipitation from the thermo/"
        "hydrometeor state, as a fast differentiable surrogate for a bulk "
        "scheme."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "hydrometeors.q_i": "kg/kg",
        "p_full": "Pa", "rho": "kg/m^3", "dz": "m", "dt": "s",
        "model": "trained MicrophysicsEmulator (Equinox module, passed explicitly)",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "dq_i_dt": "kg/kg/s", "dq_s_dt": "kg/kg/s",
        "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. The tendencies emulate a bulk microphysics "
        "scheme (dT_dt latent heating, precipitation >= 0 via a softplus head). "
        "Being a LEARNED surrogate it enforces NO hard conservation -- no "
        "contract-level conservation is claimed; positivity of precip is "
        "structural, other outputs are unconstrained -- hence conserves: "
        "none. The OPT-IN hard saturation-adjustment guard "
        "(config.hard_saturation_adjustment, default False) constrains ONLY "
        "ITS OWN increment: it drains post-step super-saturation with "
        "-dq_v = +dq_c and dT = (L_v/c_pd)*dq_c, so the drained mass conserves "
        "total water and c_pd*T + L_v*q_v. It does NOT make the scheme "
        "conservative, because the network's own tendencies remain "
        "unconstrained."
    ),
    # The RAW emulator conserves nothing; the opt-in guard's contribution
    # alone is water- and enthalpy-conserving. "none" is the honest
    # scheme-level answer because the guard defaults OFF and, even ON, only
    # its own increment is constrained -- the network's tendencies are not.
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Learned microphysics emulator (Equinox MLP); cf. Rasp, Pritchard & Gentine (2018), PNAS 115, 9684-9689",
    "idealized_test": (
        "tests/unit/test_physics_microphysics.py — the emulator returns finite "
        "MicrophysicsOutput tendencies with non-negative surface precipitation "
        "and is differentiable wrt the input state."
    ),
}


def ml_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: MicrophysicsMLEmulatorConfig,
    model: MicrophysicsEmulator,
) -> MicrophysicsOutput:
    """Compute ML emulator microphysics tendencies.

    Parameters
    ----------
    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
        Same interface as all microphysics backends.
    model : MicrophysicsEmulator
        Equinox MLP model.

    Returns
    -------
    MicrophysicsOutput
    """
    ncol, nlev = T.shape

    features = jnp.stack([
        T / config.norm_T,
        q_v * config.norm_q_factor,
        hydrometeors.q_c * config.norm_q_factor,
        hydrometeors.q_r * config.norm_q_factor,
        hydrometeors.q_i * config.norm_q_factor,
        jnp.log(jnp.clip(p_full, 1.0) / constants.p_ref),
        rho / config.norm_rho,
        dz / config.norm_dz,
        jnp.full((ncol, nlev), dt / config.norm_dt, dtype=T.dtype),
    ], axis=-1)  # (ncol, nlev, n_input)

    # Apply MLP per level via double vmap
    predict = jax.vmap(jax.vmap(model))
    y = predict(features)  # (ncol, nlev, n_output)

    # Map outputs
    dT_dt = y[..., 0]
    dq_v_dt = y[..., 1]
    dq_c_dt = y[..., 2]
    dq_r_dt = y[..., 3]
    dq_i_dt = y[..., 4]
    dq_s_dt = y[..., 5]
    precip_raw = y[..., 6]

    # Residual connection: add identity (no change) baseline
    if config.use_residual:
        dT_dt = dT_dt * _OUTPUT_SCALE  # scale raw output
        dq_v_dt = dq_v_dt * 1e-6
        dq_c_dt = dq_c_dt * 1e-6
        dq_r_dt = dq_r_dt * 1e-6
        dq_i_dt = dq_i_dt * 1e-6
        dq_s_dt = dq_s_dt * 1e-6

    # Surface precipitation from lowest level output
    precipitation = jax.nn.softplus(precip_raw[:, -1]) * _PRECIP_SCALE

    # --- OPT-IN hard (iterated) saturation-adjustment guard -----------------
    # A learned surrogate emits UNCONSTRAINED tendencies, so it can leave the
    # column super-saturated (its own contract claims ``conserves: none``).
    # The guard is the SAME shared implementation the bulk schemes use, at the
    # POST-STEP placement the validated MPAS driver hook uses -- the emulator
    # exposes no internal condensation rate to blend into, so we evaluate the
    # state its tendencies WILL produce and drain what remains super-saturated.
    #
    # WHY ``hard_saturation_blend(0, ...)`` AND NOT ``hard_saturation_drain``:
    # the two differ ONLY in their activation gate, and the drain's gate is a
    # HARD ``jnp.where(q_v > thr*q_sat, ...)`` step.  Its docstring states the
    # scope condition that licenses that step -- "a HARD gate is appropriate
    # here because the post-step hook is EAGER (outside jit / no autodiff
    # through it), so the smooth ramp the in-scheme AD path needs is
    # unnecessary".  That condition does NOT hold here: the emulator runs
    # inside jit and is the one scheme whose weights are trained through
    # ``eqx.filter_value_and_grad``, so a step discontinuity at RH = threshold
    # would put a jump in the loss surface and a one-sided gradient at the
    # gate.  Passing a ZERO base rate to the blend gives the same pure drain
    # with the SMOOTH sigmoid activation the AD path requires:
    #     blended = 0 + activation*(cond_hard - 0) = activation*cond_hard,
    # then the same no-overshoot cap and the same per-step heating/mass rate
    # limit.  (Porting a reference implementation means porting its exclusions,
    # not just its formula.)
    #
    # Sign convention: ``rate >= 0`` is a DRAIN of vapour into cloud water
    # (positive = condensation).  Applying it as
    #     q_v -= rate*dt ;  q_c += rate*dt ;  T += (L_v/c_pd)*rate*dt
    # moves water (no source/sink of total water: -dq_v = +dq_c) and conserves
    # the moist enthalpy c_pd*T + L_v*q_v exactly, matching the identity every
    # other scheme's adjustment satisfies.
    #
    # POSITIVITY, stated precisely (the earlier claim here was too strong): the
    # rate limit caps the drain at ``min(dqv_cap, max(q_v_post, 0))/dt``, so
    # WHERE THE EMULATOR LEAVES q_v_post >= 0 the corrected vapour is >= 0.
    # Where the raw network already predicts q_v_post < 0 the cap collapses to
    # zero and the guard is NEUTRAL, not corrective -- it never makes a
    # negative vapour more negative, but it does not repair one either. Raw
    # emulator positivity is out of scope for a saturation guard.
    #
    # STATIC Python ``if`` on a compile-time bool (CLAUDE.md feature-gating
    # exception): default False => not traced, byte-identical to the raw
    # emulator output.
    if config.hard_saturation_adjustment:
        T_post = T + dT_dt * dt
        q_v_post = q_v + dq_v_dt * dt
        # q_sat in fp64 when x64 is available, then back to the state dtype --
        # the SAME promotion ``saturation_adjustment`` applies for the bulk
        # schemes (issue #618). At fp32 the exp/softplus/division chain in
        # saturation_specific_humidity carries ~1.7e-8 kg/kg absolute error, a large
        # RELATIVE error on a marginal supersaturation residual, which would
        # leave this drain's activation ramp and on-curve bracket less accurate
        # than every other guarded scheme's on the production configuration
        # (fp32 arrays under jax_enable_x64=True). Static Python ``if`` on the
        # compile-time flag: the x64-off path is unchanged and no fp64 leaks
        # into the state.
        if jax.config.jax_enable_x64:
            q_sat_post = saturation_specific_humidity(
                T_post.astype(jnp.float64),
                p_full.astype(jnp.float64)).astype(q_v_post.dtype)
        else:
            q_sat_post = saturation_specific_humidity(T_post, p_full)
        rate = hard_saturation_blend(
            jnp.zeros_like(q_v_post), T_post, q_v_post, p_full, dt,
            q_sat_post,
            config.hard_sat_adjust_threshold,
            config.hard_sat_max_heating_K,
        )
        dq_v_dt = dq_v_dt - rate
        dq_c_dt = dq_c_dt + rate
        dT_dt = dT_dt + (constants.L_v / constants.c_pd) * rate

    # Pin dtype to the input precision so we never silently promote
    # the unused-species placeholders to f64 under x64 mode.
    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=dq_i_dt,
        dq_s_dt=dq_s_dt,
        dq_g_dt=z,
        dN_c_dt=z,
        dN_r_dt=z,
        dN_i_dt=z,
        precipitation=precipitation,
    )
