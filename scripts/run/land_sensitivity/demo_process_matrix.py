#!/usr/bin/env python
"""Demo: the process-to-process sensitivity matrix of one land column.

One parameter per land process (soil hydraulics, root-zone water stress,
stomata, photosynthesis, canopy structure, surface albedo, soil heat
conduction) on the rows; window-integrated surface exchanges (transpiration,
soil evaporation, GPP, sensible heat, runoff + drainage, stored soil water) on
the columns.  Each cell is the percent change of the output for a +10 % change
of the parameter, i.e. ten times the elasticity ``d ln(output) / d ln(param)``.

The whole matrix comes from ONE ``jax.jacrev`` of the vector of window totals
with respect to the vector of log-scale factors: one reverse pass per OUTPUT
(six here), whatever the number of parameters.  A perturbation experiment needs
one forward run per PARAMETER per perturbation size.

Why a window and not a single step: over one step the hydraulic conductivity
and the retention-curve parameters have exactly zero effect on the fluxes —
they act on the fluxes only through the soil-water state they redistribute.
The window (``--window-days``) lets that memory play out.

Two soil textures are run (``--textures``) so the matrix can be seen to change
with the regime.  At ``--fd-params`` the AD entries are checked against central
finite differences; a failed check aborts the run.

Usage
-----
    JAX_ENABLE_X64=1 python scripts/run/land_sensitivity/demo_process_matrix.py \
        --textures loam clay --out-dir diagnostics/land_sensitivity/process_matrix
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import NamedTuple

_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from scripts.run.land_sensitivity import column_setup as cs


class Param(NamedTuple):
    key: str        # dotted path into the config ("hydraulics.K_sat") or "params.<field>"
    label: str      # plain-language row label
    process: str    # process group (row band)


# Rows of the matrix, grouped by process.  Only parameters the two-leaf canopy
# column actually reads: the simple-SEB fields (``albedo_land``, ``z0_land``,
# ``emissivity_land``, ``soil_evap_resistance_exp``) and the unused canopy
# fields (``rd``, ``alf`` at light saturation) have an identically zero gradient
# here and are left out rather than shown as empty rows.  ``hc`` and ``rz0m``
# enter only as their product (the roughness length), so only ``hc`` is kept.
PARAMS: tuple[Param, ...] = (
    Param("hydraulics.K_sat", "sat. hydraulic conductivity", "soil hydraulics"),
    Param("hydraulics.psi_sat", "air-entry potential", "soil hydraulics"),
    Param("hydraulics.b_ch", "retention-curve exponent b", "soil hydraulics"),
    Param("hydraulics.theta_sat", "porosity", "soil hydraulics"),
    Param("root_depth", "root depth", "root-zone water stress"),
    Param("theta_wp", "wilting point", "root-zone water stress"),
    Param("theta_fc", "field capacity", "root-zone water stress"),
    Param("params.m_C3", "stomatal slope", "stomata"),
    Param("params.b0_C3", "stomatal intercept", "stomata"),
    Param("params.Vcmax25_C3_leaf", "Vcmax25", "photosynthesis"),
    Param("params.kn", "canopy N extinction", "photosynthesis"),
    Param("params.LAI", "leaf area index", "canopy structure"),
    Param("params.CI", "clumping index", "canopy structure"),
    Param("params.hc", "canopy height (roughness)", "canopy structure"),
    Param("params.ALB_VIS", "visible albedo", "surface albedo"),
    Param("params.ALB_NIR", "near-infrared albedo", "surface albedo"),
    Param("thermal.k_dry_coeff_a", "dry-soil conductivity", "soil heat conduction"),
)
OUTPUTS = ("transpiration", "soil evaporation", "GPP", "sensible heat",
           "runoff + drainage", "stored soil water")
OUTPUT_UNITS = ("kg/m2", "kg/m2", "gC/m2", "MJ/m2", "kg/m2", "kg/m2")
# Step in ln(param) for the finite-difference checks.  The canopy Newton solve
# stops at ||step|| < tol (1e-2), so the forward carries a ~1 % ripple on the
# 1e-3 scale in ln(param): central differences at 1e-3 disagree with AD by up
# to 1 % for the stress-threshold parameters, while at 1e-4 and 3e-5 they agree
# to 1e-7 (the ripple is below the step).  1e-4 sits under the ripple and well
# above x64 round-off (~1e-16 / 1e-4 relative).
FD_EPS_LOG = 1e-4           # [-] step in ln(param)
FD_ATOL = 1e-6              # [output units] absolute floor for near-zero entries
# Window totals below one unit (1 kg/m2, 1 gC/m2, 1 MJ/m2 over the window) are
# not a base for a percentage: a +10 % change of a parameter can be several
# hundred percent of a 0.5 kg/m2 drainage total and would drown the rest of the
# matrix.  Such entries are masked (NaN, shown grey); the absolute log-Jacobian
# is still written to the npz.
OUTPUT_MIN_FOR_PERCENT = 1.0   # [output units] below which the % entry is masked (NaN)
J_PER_MJ = 1.0e6            # exact unit conversion


class ProcessMatrix(NamedTuple):
    texture: str
    outputs: np.ndarray          # (n_out,) window totals in OUTPUT_UNITS
    jac_log: np.ndarray          # (n_out, n_param) d output / d ln(param)
    percent_per_10pct: np.ndarray  # (n_out, n_param) % change of output per +10 % param


def set_path(obj, path: str, fn):
    """Return ``obj`` with the leaf at dotted ``path`` replaced by ``fn(leaf)``."""
    head, *rest = path.split(".")
    v = getattr(obj, head)
    v = set_path(v, ".".join(rest), fn) if rest else fn(v)
    return obj._replace(**{head: v})


def apply_log_factors(config, params, log_factors):
    """Scale every ``PARAMS`` leaf by ``exp(log_factor)``."""
    for i, p in enumerate(PARAMS):
        scale = jnp.exp(log_factors[i])
        if p.key.startswith("params."):
            field = p.key[len("params."):]
            params = params._replace(**{field: getattr(params, field) * scale})
        else:
            config = set_path(config, p.key, lambda v, s=scale: v * s)
    return config, params


def make_window_fn(config, params, *, lat_rad: float, time_axis: cs.TimeAxis, dt_s: float,
                   dz, state0, precip_rate: float, climate: cs.ColumnClimate, remat: bool):
    """``window(log_factors) -> (n_out,)`` window totals from ``state0``."""

    def window(log_factors):
        cfg, prm = apply_log_factors(config, params, log_factors)
        final, hist = cs.rollout(state0, cfg, prm, lat_rad=lat_rad, lon_rad=0.0,
                                 time_axis=time_axis, dt_s=dt_s, precip=precip_rate,
                                 climate=climate, remat=remat)
        return jnp.stack([
            jnp.sum(hist.le_canopy) * dt_s / constants.L_v,
            jnp.sum(hist.le_soil) * dt_s / constants.L_v,
            jnp.sum(hist.gpp) * dt_s,
            jnp.sum(hist.shflx) * dt_s / J_PER_MJ,
            jnp.sum(hist.runoff) * dt_s,
            cs.column_water_kg_m2(final.theta_soil[0], dz),
        ])

    return window


def compute_matrix(texture: str, window) -> ProcessMatrix:
    zero = jnp.zeros(len(PARAMS))
    out = np.asarray(jax.jit(window)(zero))
    jac = np.asarray(jax.jit(jax.jacrev(window))(zero))
    if not (np.all(np.isfinite(out)) and np.all(np.isfinite(jac))):
        raise FloatingPointError(f"non-finite window total or Jacobian for {texture}")
    return ProcessMatrix(texture=texture, outputs=out, jac_log=jac,
                         percent_per_10pct=percent_per_10pct(out, jac))


def percent_per_10pct(outputs: np.ndarray, jac_log: np.ndarray) -> np.ndarray:
    """``10 * jac_log / output`` per row, NaN where ``|output| < OUTPUT_MIN_FOR_PERCENT``."""
    out = np.asarray(outputs)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(out) >= OUTPUT_MIN_FOR_PERCENT, 10.0 * np.asarray(jac_log) / out,
                        np.nan)


def finite_difference_checks(mat: ProcessMatrix, window, *, param_keys: list[str],
                             rtol: float) -> list[cs.GradientCheck]:
    """Central differences in ln(param) for every output at each named parameter."""
    keys = [p.key for p in PARAMS]
    unknown = [k for k in param_keys if k not in keys]
    if unknown:
        raise ValueError(f"unknown --fd-params {unknown}; choose from {keys}")
    fn = jax.jit(window)
    zero = jnp.zeros(len(PARAMS))
    checks = []
    for key in param_keys:
        j = keys.index(key)
        for k, name in enumerate(OUTPUTS):
            fd, smooth = cs.finite_difference(lambda x, k=k: fn(x)[k], zero, j, FD_EPS_LOG,
                                              atol=FD_ATOL)
            checks.append(cs.gradient_check(f"{mat.texture} d {name} / d ln {key}",
                                            mat.jac_log[k, j], fd, rtol=rtol, atol=FD_ATOL,
                                            fd_consistent=smooth))
    return checks


def run_experiment(args):
    lat_rad = float(np.deg2rad(args.lat_deg))
    steps_per_day = int(round(cs.SECONDS_PER_DAY / args.dt_s))
    params = cs.make_canopy_params(1, biome=args.biome, lai=args.lai)
    climate = cs.ColumnClimate(args.mean_annual_t_k, args.seasonal_amp_k)
    t_init = float(climate.mean_annual_t_k)
    spin_axis = cs.make_time_axis(args.spinup_days * steps_per_day, args.dt_s,
                                  start_doy=args.start_doy - args.spinup_days)
    win_axis = cs.make_time_axis(args.window_days * steps_per_day, args.dt_s,
                                 start_doy=args.start_doy)
    matrices: list[ProcessMatrix] = []
    checks: list[cs.GradientCheck] = []
    timings: dict[str, float] = {}
    for texture in args.textures:
        config = cs.make_column_config(texture)
        dz, _, _ = cs.soil_layers(config)
        t0 = time.perf_counter()
        state_win0 = jax.jit(lambda s, cfg=config: cs.rollout(
            s, cfg, params, lat_rad=lat_rad, lon_rad=0.0, time_axis=spin_axis,
            dt_s=args.dt_s, precip=args.precip_rate, climate=climate)[0])(
                cs.initial_state(config, t_init_k=t_init))
        jax.block_until_ready(state_win0)
        timings[f"{texture}/spinup_s"] = time.perf_counter() - t0
        window = make_window_fn(config, params, lat_rad=lat_rad, time_axis=win_axis,
                                dt_s=args.dt_s, dz=dz, state0=state_win0,
                                precip_rate=args.precip_rate, climate=climate,
                                remat=args.remat)
        t0 = time.perf_counter()
        mat = compute_matrix(texture, window)
        timings[f"{texture}/jacobian_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        checks += finite_difference_checks(mat, window, param_keys=args.fd_params,
                                           rtol=args.fd_rtol)
        timings[f"{texture}/fd_s"] = time.perf_counter() - t0
        matrices.append(mat)
    meta = {"timings": timings, "t_init_k": t_init, "window_days": args.window_days}
    return matrices, checks, meta


def format_matrix(mat: ProcessMatrix) -> str:
    lines = [f"[{mat.texture}] window totals: "
             + ", ".join(f"{n} {v:.4g} {u}" for n, v, u in zip(OUTPUTS, mat.outputs, OUTPUT_UNITS)),
             f"{'% per +10 % of parameter':34s}" + "".join(f"{o[:16]:>18s}" for o in OUTPUTS)]
    for i, p in enumerate(PARAMS):
        lines.append(f"{p.label:34s}" + "".join(
            f"{mat.percent_per_10pct[k, i]:18.3g}" for k in range(len(OUTPUTS))))
    return "\n".join(lines)


def plot(matrices: list[ProcessMatrix], path: pathlib.Path, *, window_days: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(matrices)
    vmax = float(np.nanmax(np.abs(np.stack([m.percent_per_10pct for m in matrices]))))
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n + 1.5, 9), sharey=True)
    axes = np.atleast_1d(axes)
    labels = [p.label for p in PARAMS]
    groups = [p.process for p in PARAMS]
    for ax, mat in zip(axes, matrices):
        z = mat.percent_per_10pct.T      # (n_param, n_out)
        masked = np.ma.masked_invalid(z)
        cmap = matplotlib.colormaps[cs.CMAP_DIVERGING].copy()
        cmap.set_bad("0.85")
        m = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(z.shape[0]):
            for k in range(z.shape[1]):
                v = z[i, k]
                if np.isfinite(v):
                    dark = abs(v) > 0.6 * vmax
                    ax.text(k, i, f"{v:+.2g}", ha="center", va="center", fontsize=8,
                            color="white" if dark else "0.15")
        ax.set_xticks(range(len(OUTPUTS)))
        ax.set_xticklabels(OUTPUTS, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(f"{mat.texture} soil", loc="left")
        # process bands: a rule between groups and the group name at the left
        for i in range(1, len(groups)):
            if groups[i] != groups[i - 1]:
                ax.axhline(i - 0.5, color="0.4", lw=0.8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    ax0 = axes[0]
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            ax0.text(-0.62, 0.5 * (start + i - 1), groups[start],
                     transform=ax0.get_yaxis_transform(), ha="right", va="center",
                     fontsize=9, color="0.35", style="italic")
            start = i
    fig.colorbar(m, ax=axes.tolist(), shrink=0.6, pad=0.02,
                 label="% change of the window total per +10 % of the parameter")
    fig.suptitle(f"Process-to-process sensitivity over {window_days} days — one reverse "
                 f"pass per column, all rows at once", x=0.01, ha="left")
    fig.subplots_adjust(left=0.36 if n == 1 else 0.24, right=0.86, bottom=0.16, top=0.92,
                        wspace=0.08)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--textures", nargs="+", default=["loam", "clay"], choices=sorted(cs.TEXTURES))
    ap.add_argument("--biome", default=cs.DEFAULT_BIOME)
    ap.add_argument("--lai", type=float, default=cs.DEFAULT_LAI)
    ap.add_argument("--lat-deg", type=float, default=40.0, help="latitude (solar geometry)")
    ap.add_argument("--mean-annual-t-k", type=float, default=cs.DEFAULT_MEAN_ANNUAL_T_K,
                    help="mean annual air temperature of the synthetic climate [K]")
    ap.add_argument("--seasonal-amp-k", type=float, default=cs.DEFAULT_T_SEASONAL_AMP_K,
                    help="seasonal half-amplitude of the synthetic climate [K]")
    ap.add_argument("--start-doy", type=float, default=152.0, help="first window day (June 1)")
    ap.add_argument("--spinup-days", type=int, default=60)
    ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--dt-s", type=float, default=cs.DEFAULT_DT_S)
    ap.add_argument("--precip-rate", type=float, default=cs.DEFAULT_PRECIP_KG_M2_S,
                    help="steady rain rate through spin-up and window [kg/m2/s]")
    ap.add_argument("--fd-params", nargs="+", default=["params.LAI", "hydraulics.K_sat"],
                    help="parameters whose Jacobian column is checked against finite differences")
    ap.add_argument("--fd-rtol", type=float, default=1e-3)
    ap.add_argument("--no-remat", dest="remat", action="store_false",
                    help="do not checkpoint the scan body under reverse-mode AD")
    ap.add_argument("--out-dir", default="diagnostics/land_sensitivity/process_matrix")
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    matrices, checks, meta = run_experiment(args)
    print(cs.format_checks(checks))
    cs.require_all_pass(checks)
    for mat in matrices:
        print(format_matrix(mat))
    arrays = {"param_keys": np.array([p.key for p in PARAMS]), "outputs": np.array(OUTPUTS)}
    for mat in matrices:
        arrays[f"{mat.texture}_window_totals"] = mat.outputs
        arrays[f"{mat.texture}_jac_log"] = mat.jac_log
        arrays[f"{mat.texture}_percent_per_10pct"] = mat.percent_per_10pct
    np.savez(out / "process_matrix.npz", **arrays)
    cs.write_json(out / "process_matrix.json", cs.provenance(
        vars(args), timings=meta["timings"], t_init_k=meta["t_init_k"],
        params=[p._asdict() for p in PARAMS], outputs=list(OUTPUTS),
        output_units=list(OUTPUT_UNITS),
        window_totals={m.texture: m.outputs.tolist() for m in matrices},
        gradient_checks=[c._asdict() for c in checks]))
    (out / "gradient_checks.txt").write_text(cs.format_checks(checks) + "\n")
    if args.plot:
        plot(matrices, out / "process_matrix.png", window_days=meta["window_days"])
    print(f"timings: {', '.join(f'{k}={v:.1f}s' for k, v in meta['timings'].items())}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
