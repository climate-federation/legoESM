"""Preregistered upstream / corrected-KPP / Wave VR12-AL column comparison.

Invoked by validate_ocean_scm.py --case wave_langmuir. A separate worker
process imports each source tree before importing any legoesm modules.
See wave_kpp_s_l2/PREREG.md for claims, provenance and interpretation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = Path(__file__).with_name("wave_kpp_s_l2")


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _source_imports(source):
    # This must precede all legoesm imports: editable installs otherwise point
    # the supposedly upstream control back to the modified working checkout.
    roots = [source / "src", *sorted((source / "packages").glob("*"))]
    sys.path[:0] = [str(p) for p in roots if p.is_dir()]


def _checked_module_manifest(source, revision, *, modules=None):
    """Every imported legoesm source must belong to the selected git tree."""
    entries = _git("ls-tree", "-r", revision, "--", "packages", "src")
    expected = {
        line.split("\t", 1)[1]: line.split()[2] for line in entries.splitlines() if "\t" in line
    }
    manifest = {}
    for name, module in tuple((sys.modules if modules is None else modules).items()):
        filename = getattr(module, "__file__", None)
        if not name.startswith("legoesm") or filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(source):
            raise RuntimeError(f"mixed source import: {name}")
        rel = str(path.relative_to(source))
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if expected.get(rel) != blob:
            raise RuntimeError(f"source does not match pinned git tree: {rel}")
        manifest[rel] = blob
    return manifest


def _require_committed_instrument():
    paths = [
        Path(__file__).resolve(),
        *INPUTS.iterdir(),
        Path(__file__).with_name("validate_ocean_scm.py"),
    ]
    for path in paths:
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        _git("ls-files", "--error-unmatch", "--", rel)
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if _git("rev-parse", f"HEAD:{rel}") != blob:
            raise RuntimeError(f"instrument is not committed at HEAD: {rel}")


def prescribed_stokes(amplitude, cfg):
    """Production Wave diagnostic on a precisely normalized monochromatic sea.

    Returns the scalar surface magnitude plus independent analytic references.
    No Wave dynamics or neural source term is claimed in the 48-hour column.
    """
    import jax.numpy as jnp
    import numpy as np
    from legoesm.wave.config import WaveConfig, WaveGridConfig, WaveGSEConfig, WaveNetworkConfig
    from legoesm.wave.constants import STOKES_GRAVITY
    from legoesm.wave.indices import stokes_drift
    from legoesm.wave.model import prepare_env

    k, depth = cfg["wave_k_m_inv"], cfg["depth_m"]
    omega = np.sqrt(STOKES_GRAVITY * k * np.tanh(k * depth))
    fp = omega / (2 * np.pi)
    grid = WaveGridConfig(
        lon_deg=tuple(22.5 + 45 * i for i in range(8)),
        lat_deg=tuple(67.5 - 45 * i for i in range(4)),
        freq_hz=tuple(fp * 1.1 ** (i - 1) for i in range(4)),
        theta_rad=tuple(np.deg2rad(180 - 90 * i) for i in range(4)),
    )
    wave_cfg = WaveConfig(
        grid=grid,
        gse=WaveGSEConfig(enabled=False),
        network=WaveNetworkConfig(parameterization="none"),
    )
    bathy = jnp.full((1, 8, 4), depth, dtype=jnp.float64)
    env = prepare_env(wave_cfg, bathy)
    spectra = jnp.zeros((1, 8, 4, 4, 4), dtype=jnp.float64)
    spectra = spectra.at[..., 1, 1].set(amplitude**2 / (2 * env.quadrature_df[1] * env.dth))
    field = stokes_drift(spectra, env, bathy)["stokes_magnitude"]
    # Independent finite-depth Stokes expression (deep-water limit: omega*k*A²).
    analytic = omega * k * amplitude**2 * np.cosh(2 * k * depth) / (2 * np.sinh(k * depth) ** 2)
    return field, {
        "analytic_surface_mps": float(analytic),
        "deep_water_mps": float(omega * k * amplitude**2),
        "paper_first_cell_mps": float(omega * k * amplitude**2 * np.exp(-2 * k * 0.32)),
        "diagnostic_gravity_ms2": float(STOKES_GRAVITY),
    }


def _worker(args):
    import time

    import numpy as np

    source = Path(args.source_root).resolve()
    _source_imports(source)
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean import scm as scm_module
    from legoesm.ocean.diagnostics import mixed_layer_depth
    from legoesm.ocean.eos import linear_eos
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.combined import OceanPhysicsConfig, make_ocean_physics
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.vertical_mixing import kpp as kpp_module
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig, VerticalMixingConfig
    from legoesm.ocean.scm import OceanColumnModel
    from legoesm.ocean.scm_forcing import OceanSCMForcing
    from legoesm.ocean.vertical import create_ocean_z_star

    from legoesm import constants

    set_policy(PrecisionPolicy.fp64())
    jax.config.update("jax_enable_x64", True)
    cfg = json.loads((INPUTS / "config.json").read_text())
    origins = {}
    for mod in (scm_module, kpp_module):
        path = Path(mod.__file__).resolve()
        if not path.is_relative_to(source):
            raise RuntimeError(f"wrong source imported for {mod.__name__}")
        origins[mod.__name__] = str(path.relative_to(source))
    kpp_bytes = Path(kpp_module.__file__).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(kpp_bytes)).encode() + b"\0" + kpp_bytes).hexdigest()
    if args.label == "A" and blob != cfg["upstream_kpp_blob"]:
        raise RuntimeError("upstream KPP content differs from pinned baseline")

    amplitude = args.amplitude
    stokes_info = {}
    us = 0.0
    if args.label == "C":
        field, stokes_info = prescribed_stokes(amplitude, cfg)
        us = float(np.asarray(field)[0, 0, 0])
        np.testing.assert_allclose(
            field, stokes_info["analytic_surface_mps"], rtol=cfg["stokes_rtol"], atol=0
        )
        stokes_info["production_surface_mps"] = us

    eos_fn = lambda T, S, p: linear_eos(
        T,
        S,
        p,
        rho_ref=constants.rho_ocean,
        alpha_T=cfg["alpha_T_K_inv"],
        beta_S=cfg["beta_S_psu_inv"],
        T_ref=cfg["T_ref_C"],
        S_ref=cfg["S_ref_psu"],
    )
    nlev, depth, dt = cfg["nlev"], cfg["depth_m"], cfg["dt_s"]
    dz = depth / nlev
    z = create_ocean_z_star(n_levels=nlev, H_max=depth, dz_surface=dz, dz_deep=dz)
    zc = np.asarray(jnp.cumsum(z.dz_ref) - 0.5 * z.dz_ref)

    def profile(name):
        values = np.loadtxt(INPUTS / name, skiprows=1)
        return jnp.asarray(np.interp(zc, -values[:, 0], values[:, 1]), dtype=jnp.float64)

    kpp_cfg = KPPConfig(enable_langmuir=args.label == "C")
    physics_cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp", kpp=kpp_cfg),
        convection=OceanConvectionConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        shortwave_penetration=None,
    )
    forcing_args = dict(
        f_c=cfg["f_c_s_inv"],
        tau_x=lambda t: cfg["tau_x_pa"],
        tau_y=lambda t: cfg["tau_y_pa"],
        q_net=lambda t: cfg["q_net_wm2"],
    )
    if args.label == "C":
        forcing_args["u_stokes"] = lambda t: us
    forcing = OceanSCMForcing(**forcing_args)
    # The identical public construction path works in both revisions. Build
    # physics with the explicit EOS, then reuse the SCM's existing stepping.
    probe = OceanColumnModel.create(
        nlev=nlev,
        dt=dt,
        T_profile=profile("tprof_file.dat"),
        S_profile=profile("sprof_file.dat"),
        z_coord=z,
        physics_config=physics_cfg,
        forcing=forcing,
        latitude_deg=45.0,
        K_v_background=0.0,
        A_v_background=0.0,
        dtype=jnp.float64,
    )
    model = OceanColumnModel(
        physics_fn=make_ocean_physics(physics_cfg, apply_vertical_diffusion=False, eos_fn=eos_fn),
        state=probe.state,
        grid=probe.grid,
        z_coord=z,
        dt=dt,
        forcing=forcing,
        implicit_vertical_mixing=True,
        K_v_background=0.0,
        A_v_background=0.0,
    )
    # Compile existing operators, preserving their source implementation and
    # explicit/implicit/Coriolis ordering. No alternate numerical integrator.
    model._tend_fn = jax.jit(model._tend_fn)
    model._apply_implicit_vertical_mixing = jax.jit(model._apply_implicit_vertical_mixing)
    model._apply_coriolis_rotation = jax.jit(model._apply_coriolis_rotation)

    rows = {
        key: [] for key in ("time_s", "T", "S", "u", "v", "K_v", "A_v", "mld_m", "max_n2_depth_m")
    }
    thickness = np.asarray(z.dz_ref)
    initial_T = np.asarray(model.state.T.data).ravel().copy()
    initial_S = np.asarray(model.state.S.data).ravel().copy()
    impulse = np.zeros(2)
    impulse_scale = 0.0

    def sample():
        state = model.state
        tend, _ = model._tend_fn(state, None, model.t_seconds)
        T, S = np.asarray(state.T.data).ravel(), np.asarray(state.S.data).ravel()
        mld = mixed_layer_depth(
            jnp.asarray(T),
            jnp.asarray(S),
            jnp.asarray(zc),
            delta_sigma=cfg["mld_delta_sigma_kgm3"],
            ref_depth_m=cfg["mld_ref_depth_m"],
            bottom_depth=depth,
            eos_fn=eos_fn,
        )
        dens = np.asarray(eos_fn(jnp.asarray(T), jnp.asarray(S), jnp.zeros(nlev)))
        n2 = constants.g / constants.rho_ocean * np.diff(dens) / np.diff(zc)
        rows["time_s"].append(model.t_seconds)
        rows["T"].append(T.copy())
        rows["S"].append(S.copy())
        rows["u"].append(np.asarray(state.u.data).ravel().copy())
        rows["v"].append(np.asarray(state.v.data).ravel().copy())
        rows["K_v"].append(np.asarray(tend.K_v).ravel())
        rows["A_v"].append(np.asarray(tend.A_v).ravel())
        rows["mld_m"].append(float(mld))
        rows["max_n2_depth_m"].append(float(0.5 * (zc[:-1] + zc[1:])[np.argmax(n2)]))

    sample()
    revision = cfg["upstream_revision"] if args.label == "A" else _git("rev-parse", "HEAD")
    _checked_module_manifest(source, revision)
    started = time.monotonic()
    nsteps = int(cfg["duration_s"] / dt)
    for step in range(nsteps):
        previous = np.array(
            [
                np.sum(np.asarray(model.state.u.data).ravel() * thickness),
                np.sum(np.asarray(model.state.v.data).ravel() * thickness),
            ]
        )
        model.step()
        present = np.array(
            [
                np.sum(np.asarray(model.state.u.data).ravel() * thickness),
                np.sum(np.asarray(model.state.v.data).ravel() * thickness),
            ]
        )
        wind = dt / constants.rho_ocean * np.array([cfg["tau_x_pa"], cfg["tau_y_pa"]])
        # Rotation follows the explicit wind/diffusion step; sum of interior
        # mixing impulses is zero. Independent discrete CN momentum budget.
        pre_rotation = previous + wind
        coriolis = (
            cfg["f_c_s_inv"]
            * dt
            * np.array(
                [0.5 * (pre_rotation[1] + present[1]), -0.5 * (pre_rotation[0] + present[0])]
            )
        )
        impulse += wind + coriolis
        impulse_scale += np.sum(np.abs(wind)) + np.sum(np.abs(coriolis))
        if (step + 1) % cfg["save_every_steps"] == 0:
            sample()
        if (step + 1) % 720 == 0:
            print(f"{args.label} A={amplitude:g}: {(step + 1) * dt / 3600:g} h", flush=True)

    arrays = {name: np.asarray(values) for name, values in rows.items()}
    d_heat = (
        constants.rho_ocean * constants.c_sw * np.sum(thickness * (arrays["T"][-1] - initial_T))
    )
    heat_input = cfg["q_net_wm2"] * nsteps * dt
    salt_error = abs(np.sum(thickness * (arrays["S"][-1] - initial_S))) / np.sum(
        thickness * initial_S
    )
    final_momentum = np.array(
        [np.sum(thickness * arrays["u"][-1]), np.sum(thickness * arrays["v"][-1])]
    )
    budgets = dict(
        heat_relative_error=float(abs(d_heat - heat_input) / abs(heat_input)),
        salt_relative_error=float(salt_error),
        momentum_relative_error=float(np.max(np.abs(final_momentum - impulse)) / impulse_scale),
    )
    metadata = dict(
        instrument_revision=_git("rev-parse", "HEAD"),
        label=args.label,
        amplitude_m=amplitude,
        source_revision=revision,
        kpp_git_blob=blob,
        module_origins=origins,
        precision="fp64",
        backend=jax.default_backend(),
        config=cfg,
        active_kpp_config=kpp_cfg._asdict(),
        constants=dict(
            g=float(constants.g), rho0=float(constants.rho_ocean), cp=float(constants.c_sw)
        ),
        input_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (INPUTS / "config.json", INPUTS / "tprof_file.dat", INPUTS / "sprof_file.dat")
        },
        stokes=stokes_info,
        budgets=budgets,
        budget_pass=max(budgets.values()) <= cfg["budget_rtol"],
        final_mld_m=float(arrays["mld_m"][-1]),
        elapsed_s=time.monotonic() - started,
    )
    metadata["imported_source_blobs"] = _checked_module_manifest(source, revision)
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    stem = f"{args.label}_{amplitude:g}"
    np.savez_compressed(target / f"{stem}.npz", z_centres_m=zc, **arrays)
    (target / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (target / f"{stem}.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "mld_m", "max_n2_depth_m", "surface_T_C"])
        writer.writerows(
            zip(arrays["time_s"], arrays["mld_m"], arrays["max_n2_depth_m"], arrays["T"][:, 0])
        )
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise RuntimeError("nonfinite column output (artifacts retained)")
    print(json.dumps({"case": stem, "final_mld_m": metadata["final_mld_m"], **budgets}), flush=True)
    return 0 if metadata["budget_pass"] else 1


def plot_results(output_dir):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    cfg = json.loads((INPUTS / "config.json").read_text())
    cases = [
        ("A_0", "A: upstream KPP", "#333333"),
        ("B_0", "B: corrected KPP", "#D55E00"),
        ("C_0.4", "C: Wave, A=0.40 m", "#56B4E9"),
        ("C_0.53", "C: Wave, A=0.53 m", "#0072B2"),
        ("C_0.8", "C: Wave, A=0.80 m", "#009E73"),
    ]
    data = {stem: np.load(out / f"{stem}.npz") for stem, _, _ in cases}
    reports = [json.loads((out / f"{stem}.json").read_text()) for stem, _, _ in cases]
    failed = [f"{r['label']}_{r['amplitude_m']:g}" for r in reports if not r["budget_pass"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    z = data["A_0"]["z_centres_m"]
    for stem, label, color in cases:
        d = data[stem]
        axes[0, 0].plot(d["time_s"] / 3600, d["mld_m"], label=label, color=color)
        axes[0, 1].plot(d["T"][-1], -z, label=label, color=color)
        axes[1, 0].semilogx(d["K_v"][-1], -0.5 * (z[:-1] + z[1:]), label=label, color=color)
        axes[1, 1].plot(d["T"][-1] - data["A_0"]["T"][-1], -z, label=label, color=color)
    axes[0, 0].set(
        xlabel="Time [h]", ylabel="Density-threshold MLD [m]", title="Mixed-layer response"
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(
        xlabel="Temperature [°C]", ylabel="Depth [m]", ylim=(-90, 0), title="Final temperature"
    )
    axes[1, 0].set(
        xlabel="Tracer diffusivity [m²/s]",
        ylabel="Depth [m]",
        ylim=(-90, 0),
        title="Final mixing profile",
    )
    axes[1, 1].set(
        xlabel="Temperature difference from A [K]",
        ylabel="Depth [m]",
        ylim=(-90, 0),
        title="Total contribution relative to upstream",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("legoESM Wave–KPP: prescribed S-L2 monochromatic-wave conditions")
    fig.text(
        0.08,
        0.018,
        "A→B: weak-buoyancy correction. B→C: Wave / VR12-AL feedback. A→C: total change.\n"
        "48 h prescribed-wave column; no global-climate or LES skill score.\n"
        + (
            f"Strict budget gate: FAIL ({', '.join(failed)}). All outcomes retained."
            if failed
            else "Strict budget gate: PASS."
        ),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.095, 1, 0.96))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"coupling_comparison.{suffix}", dpi=180)
    plt.close(fig)
    la = np.geomspace(0.1, 3, 300)
    fig, (ax, stokes_ax) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax.loglog(la, np.sqrt(1 + 0.08 / la**2), label="Upstream legacy enhancement")
    ax.loglog(la, np.sqrt(1 + 0.08 / la**4), label="MS00")
    ax.loglog(la, np.sqrt(1 + (3.1 * la) ** -2 + (5.4 * la) ** -4), label="VR12-AL")
    ax.set(
        xlabel="Turbulent Langmuir number",
        ylabel="Velocity multiplier",
        title="Analytic closure comparison",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    amplitudes = [r["amplitude_m"] for r in reports[2:]]
    stokes_ax.plot(
        amplitudes,
        [r["stokes"]["analytic_surface_mps"] for r in reports[2:]],
        color="#333333",
        label="Independent analytic Stokes",
    )
    stokes_ax.scatter(
        amplitudes,
        [r["stokes"]["production_surface_mps"] for r in reports[2:]],
        color="#0072B2",
        label="Production Wave diagnostic",
        zorder=3,
    )
    stokes_ax.set(
        xlabel="Wave amplitude [m]",
        ylabel="Surface Stokes drift [m/s]",
        title="Monochromatic diagnostic oracle",
    )
    stokes_ax.legend(fontsize=9)
    stokes_ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"enhancement_factors.{suffix}", dpi=180)
    plt.close(fig)
    # This auxiliary descriptor was recorded before looking at the results.
    # Only the final, formed stratification peak is interpreted: initially
    # the lower column has constant N² and its argmax is not a unique depth.
    final_peak = [float(data[stem]["max_n2_depth_m"][-1]) for stem, _, _ in cases]
    final_mld = [r["final_mld_m"] for r in reports]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["A", "B", "C .40", "C .53", "C .80"]
    axes[0].plot(labels, final_mld, "o-", label="Density-threshold MLD")
    axes[0].plot(labels, final_peak, "s-", label="Maximum-stratification depth")
    axes[0].set(ylabel="Final diagnostic depth [m]", title="Two distinct depth definitions at 48 h")
    axes[0].legend(fontsize=9)
    axes[1].plot(
        amplitudes, np.array(final_mld[2:]) - final_mld[1], "o-", label="Density-threshold MLD"
    )
    axes[1].plot(
        amplitudes,
        np.array(final_peak[2:]) - final_peak[1],
        "s-",
        label="Maximum-stratification depth",
    )
    axes[1].axhline(
        cfg["depth_m"] / cfg["nlev"], color="gray", linestyle="--", label="One vertical layer"
    )
    axes[1].set(
        xlabel="Wave amplitude [m]",
        ylabel="Additional depth relative to B [m]",
        title="Wave contribution, C minus B",
    )
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.text(
        0.08,
        0.025,
        "Maximum-N² depth follows the literature descriptor; initial plateau argmax is not interpreted.\n"
        "All five strict heat-budget gates remain FAIL. These are response diagnostics, not LES skill scores.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"depth_definitions.{suffix}", dpi=180)
    plt.close(fig)
    summary = dict(
        postprocessing_revision=_git("rev-parse", "HEAD"),
        config=cfg,
        cases=reports,
        interpretation="Prescribed-wave column trends; no numerical LES/global skill claim.",
        budget_pass=all(r["budget_pass"] for r in reports),
        failed_budget_cases=[
            f"{r['label']}_{r['amplitude_m']:g}" for r in reports if not r["budget_pass"]
        ],
        changes_m={
            "A_to_B": reports[1]["final_mld_m"] - reports[0]["final_mld_m"],
            "B_to_C": [r["final_mld_m"] - reports[1]["final_mld_m"] for r in reports[2:]],
            "A_to_C": [r["final_mld_m"] - reports[0]["final_mld_m"] for r in reports[2:]],
            "max_n2_depth_B_to_C": [d - final_peak[1] for d in final_peak[2:]],
        },
        max_n2_depth_note="Final formed peak only; initial lower-column N2 plateau has no unique maximum depth.",
        descriptive_posthoc_metrics={
            "note": "Final-profile descriptors of preregistered quantities; not new acceptance gates.",
            "peak_Kv_ratio_C_to_B": [
                float(data[stem]["K_v"][-1].max() / data["B_0"]["K_v"][-1].max())
                for stem, _, _ in cases[2:]
            ],
            "max_abs_T_difference_C_to_B_K": [
                float(np.max(np.abs(data[stem]["T"][-1] - data["B_0"]["T"][-1])))
                for stem, _, _ in cases[2:]
            ],
        },
        input_artifact_sha256={
            f"{stem}.{suffix}": hashlib.sha256((out / f"{stem}.{suffix}").read_bytes()).hexdigest()
            for stem, _, _ in cases
            for suffix in ("json", "npz", "csv")
        },
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_suite(output_dir, upstream_root, *, plot=True):
    cfg = json.loads((INPUTS / "config.json").read_text())
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("commit the preregistration and instrument before integration")
    _require_committed_instrument()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="1",
        MPLCONFIGDIR="/tmp/legoesm-wave-kpp-matplotlib",
    )
    cases = [("A", 0.0, Path(upstream_root).resolve()), ("B", 0.0, ROOT)]
    cases += [("C", amp, ROOT) for amp in cfg["wave_amplitudes_m"]]
    for label, amp, _ in cases:
        for suffix in ("json", "npz", "csv"):
            if (out / f"{label}_{amp:g}.{suffix}").exists():
                raise FileExistsError("use a new output directory; previous evidence is retained")
    failures = []
    fatal = []
    for label, amp, source in cases:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--label",
                label,
                "--amplitude",
                str(amp),
                "--source-root",
                str(source),
                "--output-dir",
                str(out),
            ],
            env=env,
            check=False,
            cwd=out,
        )
        if completed.returncode:
            failures.append(f"{label}_{amp:g}")
        if completed.returncode not in (0, 1):
            fatal.append(f"{label}_{amp:g}")
    if fatal:
        (out / "fatal_cases.json").write_text(json.dumps(fatal, indent=2) + "\n")
        raise RuntimeError(f"fatal worker/source failure, no comparison plotted: {fatal}")
    complete = all(
        (out / f"{label}_{amp:g}.json").exists() and (out / f"{label}_{amp:g}.npz").exists()
        for label, amp, _ in cases
    )
    summary = plot_results(out) if plot and complete else None
    if failures:
        (out / "failed_cases.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(
            f"comparison incomplete or strict budget gate failed: {failures}; artifacts retained"
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help="regenerate plots/summary from stored, stamped histories",
    )
    parser.add_argument("--label", choices=["A", "B", "C"])
    parser.add_argument("--amplitude", type=float, default=0.0)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.plot_existing:
        result = plot_results(args.output_dir)
        sys.exit(0 if result["budget_pass"] else 1)
    if not args.worker:
        parser.error("use validate_ocean_scm.py --case wave_langmuir")
    try:
        status = _worker(args)
    except Exception:
        import traceback

        traceback.print_exc()
        status = 2
    sys.exit(status)
