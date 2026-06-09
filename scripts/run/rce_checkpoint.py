"""Checkpoint save/restore for the RCEMIP plane-CRM run.

Saves the full prognostic state (u, v, w, theta', rho', tracers, phis) + the global
step and sim-time to a compressed npz, so a long run can RESUME after a crash, a
fix, or a manual stop instead of re-spinning from t=0. Kept separate from the
driver; the driver adds ``--checkpoint-every-days`` + ``--restart <path>`` and two
small hooks.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

_FIELDS = ("u", "v", "w", "theta_prime", "rho_prime", "phis", "tracers")


def save(ckpt_dir, step: int, t_s: float, state, keep_last: int = 3) -> Path:
    """Write ``ckpt_<step>.npz`` (and prune to the most recent ``keep_last``)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    arrays = {f: np.asarray(getattr(state, f).data) for f in _FIELDS}
    path = ckpt_dir / f"ckpt_{step:09d}.npz"
    np.savez(path, step=step, t_s=t_s, **arrays)
    # prune
    existing = sorted(ckpt_dir.glob("ckpt_*.npz"))
    for old in existing[:-keep_last]:
        old.unlink()
    return path


def load(path, state, reset_condensate: bool = False,
         seed_numbers: bool = False):
    """Restore arrays from ``path`` onto ``state`` (a freshly-built template of the
    SAME shapes/dtype). Returns ``(state, step, t_s)``.

    ``reset_condensate``: when restarting UNDER A DIFFERENT MICROPHYSICS SCHEME
    (especially a double-moment one), copying only the condensate MASS
    (q_c/q_r) from a single-moment checkpoint while leaving the number
    concentrations at zero is thermodynamically inconsistent (mass with no
    number ⇒ ill-defined mean particle size ⇒ blow-ups in sedimentation /
    deposition). Setting this True keeps only water vapour (slot 0) and lets
    the new scheme re-grow all hydrometeor species from the equilibrated
    thermodynamic state (condensate re-forms within minutes of model time).
    """
    d = np.load(path)
    dtype = state.u.data.dtype
    repl = {}
    for f in _FIELDS:
        if f == "tracers":
            # The checkpoint's tracer slot count may differ from the template's
            # when RESTARTING UNDER A DIFFERENT MICROPHYSICS SCHEME (e.g. a
            # 3-slot kessler checkpoint -> 9/11-slot morrison/thompson/p3 state).
            # The slot layout is FIXED and prefix-aligned:
            #   [0]=q_v, [1]=q_c, [2]=q_r, [3]=q_i, [4]=q_s, [5]=q_g,
            #   [6]=N_c, [7]=N_r, [8]=N_i, [9]=N_s, [10]=N_g.
            # (P3 REUSES slots [4]/[5] as q_rim/B_rim and [3]=q_i, [8]=N_i —
            # the indices are fixed, the physical meaning of 4/5 is scheme-
            # specific. Restarting a P3 checkpoint under a non-P3 scheme, or
            # vice versa, mixes those slots; use ``reset_condensate=True``.)
            # So copy the checkpoint's slots into the template's leading slots
            # (preserving q_v/q_c/q_r) and leave any extra slots at their
            # template value (zero) to spin up. If the checkpoint carries MORE
            # slots than the template, truncate the trailing (higher-moment)
            # ones — a no-op for the schemes that ignore them.
            ckpt_tr = jnp.asarray(d[f], dtype=dtype)
            tmpl = getattr(state, f).data
            if reset_condensate:
                # keep only q_v (slot 0); drop all hydrometeor mass/number.
                ckpt_tr = ckpt_tr[..., :1]
            n_ckpt = ckpt_tr.shape[-1]
            n_tmpl = tmpl.shape[-1]
            if n_ckpt == n_tmpl:
                new_tr = ckpt_tr
            elif n_ckpt < n_tmpl:
                # Guard: copying single-moment condensate MASS (q_c/q_r) into a
                # double-moment template (which carries N_c/N_r/N_i at slots
                # 6-8) while those number slots stay zero is thermodynamically
                # inconsistent (mass with no number ⇒ ill-defined mean particle
                # size ⇒ blow-ups). Warn loudly and recommend reset_condensate.
                if (not reset_condensate and not seed_numbers
                        and n_tmpl >= 7 and n_ckpt <= 6):
                    import warnings
                    warnings.warn(
                        f"rce_checkpoint.load: expanding a {n_ckpt}-slot "
                        f"checkpoint into a {n_tmpl}-slot (double-moment) state "
                        f"with reset_condensate=False and seed_numbers=False. "
                        f"Condensate MASS is copied but number concentrations "
                        f"start at zero — inconsistent and can blow up. Pass "
                        f"--restart-reset-condensate (vapour-only spin-up) or "
                        f"--restart-seed-numbers (keep loading, seed N).",
                        RuntimeWarning, stacklevel=2,
                    )
                new_tr = tmpl.at[..., :n_ckpt].set(ckpt_tr)
                if seed_numbers and not reset_condensate and n_tmpl >= 8:
                    # CONDENSATE-PRESERVING restart for a double-moment scheme:
                    # keep the single-moment condensate MASS (q_c slot 1, q_r
                    # slot 2 — so updrafts stay LOADED and don't overshoot on
                    # restart) and seed CONSISTENT number concentrations so the
                    # mean particle mass is physical. N_r [#/kg] = q_r / x_r with
                    # a typical raindrop mass x_r; N_c is left to the scheme's
                    # specified-Nc fallback (effective_Nc → Nc_0 where N_c<1).
                    # Avoids BOTH the mass-without-number blow-up AND the
                    # unloaded-updraft spin-up shock that runs Thompson's
                    # convection away under RRTMGP.
                    x_r_seed = 1.0e-8  # ~0.27 mm-diameter drizzle drop [kg]
                    q_r_seed = jnp.clip(new_tr[..., 2], 0.0)
                    N_r_seed = q_r_seed / x_r_seed            # [#/kg]
                    new_tr = new_tr.at[..., 7].set(N_r_seed)
            else:
                new_tr = ckpt_tr[..., :n_tmpl]
            repl[f] = getattr(state, f).replace(data=new_tr)
        else:
            repl[f] = getattr(state, f).replace(
                data=jnp.asarray(d[f], dtype=dtype))
    return state._replace(**repl), int(d["step"]), float(d["t_s"])


def latest(ckpt_dir):
    """Path to the newest checkpoint in ``ckpt_dir`` (or None)."""
    files = sorted(Path(ckpt_dir).glob("ckpt_*.npz"))
    return files[-1] if files else None
