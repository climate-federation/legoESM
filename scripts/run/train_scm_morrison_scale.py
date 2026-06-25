#!/usr/bin/env python
"""Train the Morrison warm-rain scale-net in a single column against an LES reference.

The end-to-end proof of the "local scaling factors for Morrison" idea on a single
column:

    SCM(Morrison + scale-net)  --rollout-->  (T, q_v, q_cond, precip) profiles
            score_profiles_jax(SCM, LES-reference)  -->  loss
            eqx.filter_value_and_grad(net)  -->  optimizer step  (repeat)

The scale-net is injected into Morrison via ``MorrisonConfig.warm_rain_scale_fn``
(config-pytree injection inside the loss; production leaves it ``None``). The
LES reference is consumed from an ``.npz`` of planar-mean profiles (the schema
``run_bomex_les.py`` / ``run_dycoms_les.py`` write: ``z, theta|T, qv, qc[, qr]``).

Reference source:
  * ``--reference path.npz``  — real LES profiles (generate on the cluster).
  * default / ``--placeholder`` — a SELF-CONTAINED stand-in: a fresh column is
    run with a KNOWN constant warm-rain perturbation baked into Morrison, and
    that profile becomes the target. A fresh (identity-init) net wrapping
    UNperturbed Morrison must then learn the correction. This exercises the full
    plumbing + gradients locally with no gSAM deck / GPU. Swap in ``--reference``
    once the LES npz exists.

Run (local smoke / unit test path):
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run/train_scm_morrison_scale.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    WarmRainRateScales,
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.ml.training import TrainingConfig, create_optimizer
from legoesm.thermo import saturation_mixing_ratio
from legoesm.training.morrison_scale_net import MorrisonWarmRainScaleNet
from legoesm.training.scm_rce_metrics import score_profiles_jax

from legoesm import constants

_PROFILE_FLOOR = 1.0e-6  # std normaliser floor for score_profiles_jax


class Ref(NamedTuple):
    """Duck-typed reference for ``score_profiles_jax`` (single LES profile)."""
    mass_weights: jax.Array
    T_ref: jax.Array
    qv_ref: jax.Array
    qcond_ref: jax.Array
    precip_ref_mm_day: float = 0.0


class Column(NamedTuple):
    """Initial single-column state + grid for the differentiable rollout."""
    T: jax.Array        # (1, nlev)
    q_v: jax.Array
    q_c: jax.Array
    N_c: jax.Array
    p_full: jax.Array
    p_half: jax.Array
    rho: jax.Array
    dz: jax.Array
    mass_weights: jax.Array  # (nlev,) layer-mass weights, sum=1
    cool_K_s: float          # uniform radiative cooling [K/s]


def build_column(nlev: int = 30, dtype=jnp.float64) -> Column:
    """A simple saturated cloudy column (placeholder for an LES-derived column).

    Uses the shared Tetens saturation (no re-derived Clausius-Clapeyron) so the
    initial cloud is physically consistent with the model. Replaced by the LES
    initial profile once a reference npz is supplied.
    """
    p_full = jnp.linspace(1.0e5, 6.0e4, nlev, dtype=dtype)[None, :]
    p_half = jnp.linspace(1.005e5, 5.95e4, nlev + 1, dtype=dtype)[None, :]
    # Mixed-layer-ish temperature, ~289 K surface cooling aloft.
    T = jnp.linspace(289.0, 275.0, nlev, dtype=dtype)[None, :]
    rho = p_full / (constants.R_d * T)
    dp = jnp.abs(jnp.diff(p_half, axis=1))[0]          # (nlev,)
    dz = dp / (rho[0] * constants.g)
    dz = dz[None, :]
    q_sat = saturation_mixing_ratio(T, p_full)
    # Saturated cloud layer in the lower third; subsaturated above.
    cloud = (jnp.arange(nlev) < nlev // 3)[None, :]
    q_v = jnp.where(cloud, q_sat, 0.6 * q_sat)
    q_c = jnp.where(cloud, jnp.asarray(5.0e-4, dtype), jnp.asarray(0.0, dtype))
    N_c = jnp.where(cloud, jnp.asarray(1.0e8, dtype), jnp.asarray(0.0, dtype))
    mass_weights = (dp / jnp.sum(dp)).astype(dtype)
    return Column(
        T=T, q_v=q_v, q_c=q_c, N_c=N_c, p_full=p_full, p_half=p_half,
        rho=rho, dz=dz, mass_weights=mass_weights,
        cool_K_s=-2.0 / 86400.0,
    )


def _rollout(col: Column, morrison_cfg: MorrisonConfig, n_steps: int, dt: float):
    """Differentiable column-Morrison rollout. Returns final (T, qv, qcond, precip_mm_day).

    Pure ``lax.scan``; positivity floor via ``jnp.maximum`` (subgradient AD-safe).
    """
    zeros = jnp.zeros_like(col.q_c)

    def step(carry, _k):
        T, q_v, q_c, q_r, q_i, q_s, q_g, N_c, N_r, N_i, accum = carry
        hyd = HydrometeorState(
            q_c=q_c, q_r=q_r, q_i=q_i, q_s=q_s, q_g=q_g,
            N_c=N_c, N_r=N_r, N_i=N_i, N_s=None, N_g=None,
        )
        out = morrison_microphysics(
            T, q_v, hyd, col.p_full, col.p_half, col.rho, col.dz, dt,
            morrison_cfg,
        )
        T = T + dt * (out.dT_dt + col.cool_K_s)
        q_v = jnp.maximum(q_v + dt * out.dq_v_dt, 0.0)
        q_c = jnp.maximum(q_c + dt * out.dq_c_dt, 0.0)
        q_r = jnp.maximum(q_r + dt * out.dq_r_dt, 0.0)
        q_i = jnp.maximum(q_i + dt * out.dq_i_dt, 0.0)
        q_s = jnp.maximum(q_s + dt * out.dq_s_dt, 0.0)
        q_g = jnp.maximum(q_g + dt * out.dq_g_dt, 0.0)
        N_c = jnp.maximum(N_c + dt * out.dN_c_dt, 0.0)
        N_r = jnp.maximum(N_r + dt * out.dN_r_dt, 0.0)
        N_i = jnp.maximum(N_i + dt * out.dN_i_dt, 0.0)
        accum = accum + jnp.sum(out.precipitation) * dt   # kg/m^2 over column
        return (T, q_v, q_c, q_r, q_i, q_s, q_g, N_c, N_r, N_i, accum), None

    init = (col.T, col.q_v, col.q_c, zeros, zeros, zeros, zeros,
            col.N_c, jnp.zeros_like(col.q_c), zeros, jnp.asarray(0.0, col.T.dtype))
    final, _ = lax.scan(step, init, jnp.arange(n_steps))
    T, q_v, q_c, q_r = final[0], final[1], final[2], final[3]
    accum = final[-1]
    qcond = q_c + q_r
    total_s = max(n_steps * dt, 1.0)
    precip_mm_day = accum / total_s * 86400.0   # kg/m^2/s -> mm/day
    return T[0], q_v[0], qcond[0], precip_mm_day


def make_placeholder_reference(
    col: Column, base_cfg: MorrisonConfig, n_steps: int, dt: float,
    perturb: WarmRainRateScales,
) -> Ref:
    """Target = baseline Morrison run with a KNOWN constant warm-rain perturbation."""
    def const_scale(T, q_v, q_c, q_r, rho):
        ones = jnp.ones_like(T)
        return WarmRainRateScales(
            autoconv=perturb.autoconv * ones,
            accretion=perturb.accretion * ones,
            rain_evap=perturb.rain_evap * ones,
        )
    cfg = base_cfg._replace(warm_rain_scale_fn=const_scale)
    T, q_v, qcond, precip = _rollout(col, cfg, n_steps, dt)
    return Ref(mass_weights=col.mass_weights, T_ref=T, qv_ref=q_v,
               qcond_ref=qcond, precip_ref_mm_day=float(precip))


def load_reference(npz_path: Path, col: Column) -> Ref:
    """Load an LES planar-mean profile npz into a ``Ref`` (top-to-bottom)."""
    d = np.load(npz_path)
    def pick(*names):
        for n in names:
            if n in d.files:
                return jnp.asarray(d[n], dtype=col.T.dtype)
        raise KeyError(f"reference npz missing any of {names}; has {d.files}")
    T = pick("T", "temperature")
    qv = pick("qv", "q_v")
    qc = pick("qc", "q_c")
    qcond = qc + (pick("qr", "q_r") if any(k in d.files for k in ("qr", "q_r")) else 0.0)
    precip = float(d["precip_mm_day"]) if "precip_mm_day" in d.files else 0.0
    return Ref(mass_weights=col.mass_weights, T_ref=T, qv_ref=qv,
               qcond_ref=qcond, precip_ref_mm_day=precip)


def train(
    *, reference: Path | None, n_steps: int, dt: float, epochs: int,
    lr: float, seed: int, out: Path | None,
):
    col = build_column()
    base_cfg = MorrisonConfig()

    if reference is not None:
        ref = load_reference(reference, col)
        ref_kind = f"LES npz {reference}"
    else:
        perturb = WarmRainRateScales(autoconv=3.0, accretion=0.5, rain_evap=1.5)
        ref = make_placeholder_reference(col, base_cfg, n_steps, dt, perturb)
        ref_kind = "placeholder (perturb au=3.0, ac=0.5, evap=1.5)"

    net = MorrisonWarmRainScaleNet(jax.random.PRNGKey(seed))

    def loss_fn(model):
        cfg = base_cfg._replace(warm_rain_scale_fn=model)
        T, q_v, qcond, precip = _rollout(col, cfg, n_steps, dt)
        _, _, _, combined = score_profiles_jax(
            ref, T, q_v, qcond, profile_floor=_PROFILE_FLOOR)
        return combined

    optimizer = create_optimizer(TrainingConfig(
        lr=lr, warmup_steps=max(1, epochs // 10), total_steps=epochs,
        optimizer="adamw",
    ))
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    @eqx.filter_jit
    def update(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    loss0 = float(loss_fn(net))
    print(f"[train_scm_morrison_scale] reference: {ref_kind}")
    print(f"  nlev={col.T.shape[1]} n_steps={n_steps} dt={dt} epochs={epochs}")
    print(f"  epoch    0  loss={loss0:.6f}  (identity-init net = baseline Morrison)")
    loss = loss0
    for e in range(1, epochs + 1):
        net, opt_state, loss = update(net, opt_state)
        if e % max(1, epochs // 10) == 0 or e == epochs:
            print(f"  epoch {e:4d}  loss={float(loss):.6f}")
    loss_final = float(loss)
    print(f"  loss: {loss0:.6f} -> {loss_final:.6f} "
          f"({100.0 * (1 - loss_final / max(loss0, 1e-12)):.1f}% reduction)")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(out, net)
        meta = {"reference": ref_kind, "loss0": loss0, "loss_final": loss_final,
                "n_steps": n_steps, "dt": dt, "epochs": epochs}
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"  trained net -> {out}")
    return loss0, loss_final, net


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=None,
                   help="LES planar-mean profile npz; omit for self-contained placeholder.")
    p.add_argument("--steps", type=int, default=600, help="rollout steps")
    p.add_argument("--dt", type=float, default=2.0, help="rollout timestep [s]")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5.0e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("results/scm_morrison_scale/trained_net.eqx"))
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke path (few steps/epochs) for the unit test.")
    args = p.parse_args(argv)
    if args.quick:
        args.steps, args.epochs, args.out = 30, 15, None
    loss0, loss_final, _ = train(
        reference=args.reference, n_steps=args.steps, dt=args.dt,
        epochs=args.epochs, lr=args.lr, seed=args.seed, out=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
