"""Shared helper for Drake-band momentum-budget diagnostic runners.

Three runner scripts (baseline / divdamp / implicit) call into this
module — they only differ in the LatLonCGridOceanConfig knobs they
construct and the output directory.  All the stepping, accumulation,
closure check, and 3D-means saving lives here.

The inner stepping loop (``tendencies_with_diagnostics + step +
accumulate``) is rolled into a single ``lax.scan`` and JIT-compiled,
called once per block of ``block_size`` steps from the outer Python
loop.  This avoids per-step host-sync overhead from the previous NumPy
accumulation pattern (~3-5x speedup at dt = 600 s on a 5° grid).

The accumulators are kept in JAX as float64 (under JAX_ENABLE_X64=1)
and only transferred to host once at the end of the run, when divided
by the sample count to produce time means.

Closure invariant: ``Σ component time-means == total time-mean`` by
construction (each step's diagnostic satisfies the per-step closure to
machine precision; summing both sides preserves equality).  This is
verified at the end of every run and reported.
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.ocean.state import MomentumTendencyDiagnostics


DIAG_U_NAMES = tuple(
    f for f in MomentumTendencyDiagnostics._fields
    if f.endswith("_u") and not f.startswith("total")
)
DIAG_V_NAMES = tuple(
    f for f in MomentumTendencyDiagnostics._fields
    if f.endswith("_v") and not f.startswith("total")
)
DIAG_NAMES = DIAG_U_NAMES + DIAG_V_NAMES
STATE_ACC_NAMES = ("T", "S", "u", "v", "eta")


def restore_state_from_npz(template_state, restart_path: Path):
    """Replace each Field's ``.data`` in ``template_state`` from npz.

    Field metadata (dims, units, staggering) is preserved from the
    template; only the numerical data is swapped.  Returns
    ``(state, day_offset)``.
    """
    npz = np.load(restart_path, allow_pickle=False)
    new = {}
    for f in template_state._fields:
        obj = getattr(template_state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        if f not in npz.files:
            raise KeyError(f"Restart {restart_path} missing field {f!r}")
        new[f] = obj.replace(data=jnp.asarray(npz[f], dtype=obj.data.dtype))
    return template_state._replace(**new), float(npz["time_days"])


def _zero_diag_acc(diag) -> dict[str, jnp.ndarray]:
    """Build zero-initialized float64 accumulators matching diag shapes."""
    out = {
        name: jnp.zeros(getattr(diag, name).data.shape, dtype=jnp.float64)
        for name in DIAG_NAMES
    }
    out["total_u"] = jnp.zeros(diag.total_u.data.shape, dtype=jnp.float64)
    out["total_v"] = jnp.zeros(diag.total_v.data.shape, dtype=jnp.float64)
    return out


def _zero_state_acc(state) -> dict[str, jnp.ndarray]:
    return {
        k: jnp.zeros(getattr(state, k).data.shape, dtype=jnp.float64)
        for k in STATE_ACC_NAMES
    }


def _make_block_fn(model, dt: float):
    """Build a JIT-compiled function that scans inner steps with
    accumulation.

    Each scan step does: capture diagnostic at start-of-step, accumulate
    each component into the float64 running sums, accumulate state into
    its running sums, then advance state via ``model.step``.  Carry =
    (state, diag_acc, state_acc).

    Block length is a static argument so the compiled scan length is
    known at compile time.
    """

    def scan_step(carry, _):
        state, diag_acc, state_acc = carry
        # Diagnostic at start-of-step (uses same state model.step will use)
        _, diag = model.tendencies_with_diagnostics(state, dt=dt)
        # Accumulate diagnostic components — cast to float64 for precision
        # (lossless when state is already float64; protects against
        # float32-state silently truncating long sums).
        new_diag_acc = {
            name: diag_acc[name] + getattr(diag, name).data.astype(jnp.float64)
            for name in DIAG_NAMES
        }
        new_diag_acc["total_u"] = (
            diag_acc["total_u"] + diag.total_u.data.astype(jnp.float64)
        )
        new_diag_acc["total_v"] = (
            diag_acc["total_v"] + diag.total_v.data.astype(jnp.float64)
        )
        # Accumulate state
        new_state_acc = {
            k: state_acc[k] + getattr(state, k).data.astype(jnp.float64)
            for k in STATE_ACC_NAMES
        }
        # Advance
        new_state = model.step(state, dt)
        return (new_state, new_diag_acc, new_state_acc), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, diag_acc, state_acc, n_inner: int):
        (state, diag_acc, state_acc), _ = jax.lax.scan(
            scan_step, (state, diag_acc, state_acc), None, length=n_inner,
        )
        return state, diag_acc, state_acc

    return block_fn


def _pretty_eta(seconds: float) -> str:
    """Format ETA in min or hr."""
    if seconds < 0:
        return "—"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def run_diagnostic_loop(
    model,
    state,
    dt: float,
    n_steps: int,
    output_dir: Path,
    block_size: int = 1000,
    label: str | None = None,
) -> dict[str, Any]:
    """Run instrumented momentum-budget diagnostic with JIT-compiled blocks.

    Parameters
    ----------
    model
        ``LatLonCGridOceanModel`` instance.  Must support
        ``tendencies_with_diagnostics`` and ``step``.
    state
        Initial state (e.g., from restart).
    dt
        Baroclinic timestep [s].
    n_steps
        Total number of steps to integrate.
    output_dir
        Where to write ``tendency_3d_means.npz``.
    block_size
        Number of inner steps per JIT'd block.  Trades progress
        granularity (smaller) against host-sync overhead (larger).
        Default 1000 (~6.9 days at dt=600 s, ~52 blocks/yr).
    label
        Optional label printed in progress messages.

    Returns
    -------
    dict with keys:
        - ``time_means``: dict of name -> 3D float64 numpy array
        - ``state_means``: dict of name -> numpy array
        - ``n_samples``: int
        - ``wall_time_s``: float
        - ``closure_err_u``, ``closure_err_v``: float (max |Σ - total|)
        - ``output_path``: Path to the saved .npz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize accumulators eagerly (one diagnostic call to get shapes).
    _, diag0 = model.tendencies_with_diagnostics(state, dt=dt)
    diag_acc = _zero_diag_acc(diag0)
    state_acc = _zero_state_acc(state)
    # Free diag0 — we only needed shapes.
    del diag0

    block_fn = _make_block_fn(model, dt)
    n_blocks = n_steps // block_size
    n_remainder = n_steps - n_blocks * block_size

    if label:
        print(f"[{label}] integrating {n_steps:,} steps "
              f"in {n_blocks} block(s) × {block_size} + {n_remainder} remainder")
    else:
        print(f"Integrating {n_steps:,} steps "
              f"in {n_blocks} block(s) × {block_size} + {n_remainder} remainder")

    n_samples = 0
    t0 = time.time()
    last_print = t0
    progress_every_blocks = max(1, n_blocks // 50)

    for b in range(n_blocks):
        state, diag_acc, state_acc = block_fn(
            state, diag_acc, state_acc, block_size,
        )
        n_samples += block_size

        if (b + 1) % progress_every_blocks == 0 or (b + 1) == n_blocks:
            now = time.time()
            if now - last_print > 30 or (b + 1) == n_blocks:
                jax.block_until_ready(state.eta.data)  # accurate timing
                yr = n_samples * dt / 86400.0 / 365.0
                total_yr = n_steps * dt / 86400.0 / 365.0
                elapsed = now - t0
                eta_s = (
                    elapsed / yr * total_yr - elapsed if yr > 0 else -1
                )
                tag = f"[{label}] " if label else "  "
                print(
                    f"{tag}Year {yr:.3f}/{total_yr:.2f} | "
                    f"block {b + 1}/{n_blocks} | "
                    f"step {n_samples:,}/{n_steps:,} | "
                    f"ETA {_pretty_eta(eta_s)}",
                    flush=True,
                )
                last_print = now

    if n_remainder > 0:
        state, diag_acc, state_acc = block_fn(
            state, diag_acc, state_acc, n_remainder,
        )
        n_samples += n_remainder

    jax.block_until_ready(state.eta.data)
    wall = time.time() - t0
    print(f"\nDone in {wall:.0f}s ({_pretty_eta(wall)})")

    # Transfer accumulators to host and divide.
    tm = {
        name: np.asarray(diag_acc[name], dtype=np.float64) / n_samples
        for name in (DIAG_NAMES + ("total_u", "total_v"))
    }
    tm_state = {
        k: np.asarray(state_acc[k], dtype=np.float64) / n_samples
        for k in STATE_ACC_NAMES
    }

    # Closure check on time-means.
    sum_u = sum(tm[n] for n in DIAG_U_NAMES)
    sum_v = sum(tm[n] for n in DIAG_V_NAMES)
    err_u = float(np.max(np.abs(sum_u - tm["total_u"])))
    err_v = float(np.max(np.abs(sum_v - tm["total_v"])))
    norm_u = float(np.max(np.abs(tm["total_u"])) + 1e-30)
    norm_v = float(np.max(np.abs(tm["total_v"])) + 1e-30)
    print(f"\nClosure check (time-mean):")
    print(f"  ||Σ_u terms - total_u||_inf = {err_u:.3e}  "
          f"(rel {err_u / norm_u:.3e})")
    print(f"  ||Σ_v terms - total_v||_inf = {err_v:.3e}  "
          f"(rel {err_v / norm_v:.3e})")

    save_3d = {f"tend_{n}": tm[n] for n in DIAG_NAMES}
    save_3d["tend_total_u"] = tm["total_u"]
    save_3d["tend_total_v"] = tm["total_v"]
    for k in STATE_ACC_NAMES:
        save_3d[f"state_{k}_mean"] = tm_state[k]
    out_path = output_dir / "tendency_3d_means.npz"
    np.savez_compressed(out_path, **save_3d)
    print(f"  Saved {out_path}")

    return {
        "time_means": tm,
        "state_means": tm_state,
        "n_samples": n_samples,
        "wall_time_s": wall,
        "closure_err_u": err_u,
        "closure_err_v": err_v,
        "output_path": out_path,
        "save_3d": save_3d,
        "final_state": state,
    }


# ---------------------------------------------------------------------------
# Optional: Drake-band summary + plot (used only by the baseline runner).
# ---------------------------------------------------------------------------

J_DRAKE = np.arange(2, 7)
RHO_0 = 1027.0


def compute_and_save_band_summary(
    output_dir: Path,
    save_3d: dict,
    state_means: dict,
    bottom_drag_coeff: float,
    grid,
    z_coord,
    final_state,
) -> dict[str, float]:
    """Compute Drake-band depth- and zonally-integrated zonal-momentum
    budget (equivalent stress in Pa per term) and save to npz + bar plot.

    Returns the per-term band-mean stress as a dict.
    """
    cos_lat = np.cos(np.clip(np.asarray(grid.lat), -np.pi / 2 + 1e-9,
                              np.pi / 2 - 1e-9))
    dx_u = cos_lat * float(grid.radius) * float(grid.dlon)
    dy = float(grid.radius) * float(grid.dlat)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    H_total = float(dz.sum())
    u_mask = np.asarray(final_state.u_mask.data, dtype=np.float64)
    n_wet_u = np.sum(u_mask, axis=1)
    area_band = float(np.sum((dx_u * n_wet_u * dy)[J_DRAKE]))

    band_summary_Pa: dict[str, float] = {}
    for name in DIAG_U_NAMES + ("total_u",):
        arr3d = save_3d[f"tend_{name}"]
        force_per_row = (
            RHO_0 * np.sum(arr3d * u_mask[:, :, None] * dz[None, None, :],
                            axis=(1, 2))
            * dx_u * dy
        )
        band_summary_Pa[name] = (
            float(np.sum(force_per_row[J_DRAKE])) / area_band
        )

    # Path-3 contribution reconstructed post-hoc from time-mean U_baro.
    u_mean = state_means["u"]
    U_baro_face = np.sum(u_mean * dz[None, None, :], axis=-1) / H_total
    path3_per_face = -RHO_0 * bottom_drag_coeff * U_baro_face * u_mask
    force_path3 = np.sum(path3_per_face, axis=1) * dx_u * dy
    band_summary_Pa["bt_path3_drag_u"] = (
        float(np.sum(force_path3[J_DRAKE])) / area_band
    )

    np.savez_compressed(
        output_dir / "tendency_band_summary.npz",
        **{f"band_{k}_Pa": np.array(v) for k, v in band_summary_Pa.items()},
        area_band=np.array(area_band),
    )
    print(f"  Saved {output_dir / 'tendency_band_summary.npz'}")

    _plot_band_summary(output_dir, band_summary_Pa)

    print("\n=== Drake-band depth-and-zonally-integrated zonal-momentum budget ===")
    print(f"  band area: {area_band:.3e} m²")
    full = band_summary_Pa["total_u"] + band_summary_Pa["bt_path3_drag_u"]
    print(f"  {'term':<24} {'Pa':>10}")
    for name in DIAG_U_NAMES:
        print(f"  {name:<24} {band_summary_Pa[name]:+10.4f}")
    print(f"  {'-' * 36}")
    print(f"  {'total_u (baroclinic)':<24} {band_summary_Pa['total_u']:+10.4f}"
          f"   (Σ baroclinic terms)")
    print(f"  {'bt_path3_drag_u (post-hoc)':<24} "
          f"{band_summary_Pa['bt_path3_drag_u']:+10.4f}")
    print(f"  {'-' * 36}")
    print(f"  {'FULL momentum source':<24} {full:+10.4f}"
          f"   (≈ 0 in steady state)")

    return band_summary_Pa


def _plot_band_summary(output_dir: Path, band_summary_Pa: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(k, v) for k, v in band_summary_Pa.items() if k != "total_u"]
    items.sort(key=lambda kv: kv[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["C3" if v < 0 else "C0" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Equivalent stress (Pa)")
    ax.set_title(
        "Drake-band depth-integrated zonal-momentum budget — online diagnostics\n"
        "(red = westward sink, blue = eastward source; sum should ≈ 0 in steady state)"
    )
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    out = output_dir / "momentum_budget_closure.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")
