"""Reproduce the T18 hybrid van Leer gaps and T17 condensate diagnostics.

Run on PBS develop@desched1, account UCLB0017, CPU, JAX_ENABLE_X64=1::

    python scripts/validate/diag_hybrid_tracer_smoothing.py --out <artifact-dir>

The fixture, species/decks, pressure regions and dt=112.5 s are inherited from
T17. Compare defaults (q width 1e-10 kg/kg; flux width 1e-6 Pa/s) with the
frozen hard path (both zero). Report paired budgets, reconstructed effective
faces, and vertical-only forward-Euler negatives before conservative borrowing.
These are fixture diagnostics, not production-climate statistics.
"""

import argparse
import os
from pathlib import Path
import json
import csv
import runpy
import subprocess
import hashlib


def countmin(x, mask):
    """Return the masked count and minimum; the minimum is None for an empty mask."""
    vals = x[mask]
    return int(vals.size), float(vals.min()) if vals.size else None


def smoothing_main():
    if hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    import jax.numpy as jnp
    import numpy as np
    from legoesm.core.precision import PrecisionPolicy, set_policy

    set_policy(PrecisionPolicy.fp64())
    from legoesm.grids import vertical as v
    from legoesm.core.smooth import smooth_abs
    from legoesm.core.field import Field
    from legoesm.atmosphere.standard_atmosphere import standard_atmosphere_temperature
    from legoesm.driver.config import ExperimentConfig
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas as pe
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import tracer_horizontal_advection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    OUT = parser.parse_args().out
    OUT.mkdir(parents=True, exist_ok=True)
    W = Path(__file__).resolve().parents[2]
    assert os.environ.get("PBS_JOBID") and jax.config.x64_enabled
    assert all(d.platform == "cpu" for d in jax.devices())
    h = runpy.run_path(str(W / "tests/unit/test_mpas_atmosphere.py"))
    coord = v.make_cam6_l32_levels()
    nlev = 32
    mesh = h["_make_mesh"](level=2)
    state = h["_add_perturbation_hydro"](h["_make_hydrostatic_state"](mesh, nlev), mesh, nlev)
    T = standard_atmosphere_temperature(mesh.latCell, coord.sigma_full)
    qv = (
        ExperimentConfig().rh_init
        * saturation_mixing_ratio(T, state.p_s.data[:, None] * coord.sigma_full)
        * coord.sigma_full**2
    )
    state = state._replace(
        T=state.T.replace(data=T),
        tracers={"q_v": Field(data=qv, name="q_v", dims=("nCells", "nlev"), units="kg/kg")},
    )
    calls = []
    real = v.vertical_advection_hybrid_van_leer


    def spy(*a, **kw):
        calls.append((a, kw))
        return real(*a, **kw)


    v.vertical_advection_hybrid_van_leer = spy
    smooth = pe.mpas_model_tendencies(state, mesh, coord, pe.MPASPrimitiveEquationConfig(nu_del2=1e5))
    v.vertical_advection_hybrid_van_leer = real
    hard = pe.mpas_model_tendencies(
        state,
        mesh,
        coord,
        pe.MPASPrimitiveEquationConfig(
            nu_del2=1e5, tracer_van_leer_width=0.0, tracer_van_leer_flux_width=0.0
        ),
    )
    _, F, ps, coord = calls[0][0]
    area = np.asarray(mesh.areaCell)[:, None]


    def norm(x, mask):
        return np.sqrt(np.sum(area * np.where(mask, x * x, 0.0)))


    gap = np.asarray(smooth.tracer_tendencies["q_v"].data - hard.tracer_tendencies["q_v"].data)
    reference = [
        float(np.max(abs(gap))),
        float(norm(gap, True) / norm(np.asarray(hard.tracer_tendencies["q_v"].data), True)),
    ]
    (OUT / "full_gap.json").write_text(json.dumps(reference))
    dp = v.dp_from_hybrid(coord, ps)
    p = v.pressure_from_hybrid(coord, ps)
    ph = v.pressure_from_hybrid(coord, ps, full=False)
    dc = jnp.diff(p, axis=-1)
    dt = 112.5
    # Decks have latitude-dependent amplitudes, exact zero outside. Levels top -> bottom.
    amplitude = (1.0 + 0.25 * jnp.cos(mesh.latCell))[:, None]


    def deck(start, values):
        return (
            jnp.zeros_like(qv)
            .at[:, start : start + len(values)]
            .set(amplitude * jnp.array(values)[None, :])
        )


    qs = {
        "q_v": qv,
        "q_c": deck(23, [1e-5, 8e-5, 1e-4, 2e-5]),
        "q_i": deck(9, [1e-6, 8e-6, 1e-5, 2e-6]),
        "q_r": deck(24, [1e-7, 1e-6, 3e-7]),
        "q_s": deck(12, [1e-8, 1e-7, 3e-8]),
        "small": qv * 1e-6,
    }
    widths = [("hard", 0.0, 0.0), ("default", 1e-10, 1e-6)]


    def tendency(q, w, fw):
        return real(q, F, ps, coord, smoothing_width=w, flux_smoothing_width=fw)


    def faces(q, w, fw):
        pos, neg = v._van_leer_face_values(q, dc, dp, w)
        edge = jnp.zeros(nlev + 1, dtype=bool).at[jnp.array([1, nlev - 1])].set(True)
        pos = jnp.where(edge, jnp.concatenate([q[:, :1], q], -1), pos)
        neg = jnp.where(edge, jnp.concatenate([q, q[:, -1:]], -1), neg)
        if fw:
            af = smooth_abs(F, fw * fw) - fw
            selected = 0.5 * (pos + neg) + 0.5 * (af / jnp.where(F != 0, F, 1.0)) * (pos - neg)
        else:
            selected = jnp.where(F > 0, pos, neg)
        return pos, neg, selected


    rows = []
    controls = {"T15_full_tendency": reference}
    raw = {}
    for name, q in qs.items():
        baseline = tendency(q, 0.0, 0.0)
        hf = np.asarray(faces(q, 0.0, 0.0)[2][:, 1:-1])
        horizontal = tracer_horizontal_advection(q[..., None], state.u.data, mesh)[..., 0]
        for label, w, relative in widths:
            tend = tendency(q, w, relative)
            pos, neg, face = faces(q, w, relative)
            # Independent face-flux reconstruction validates the diagnostic, not physics.
            ft = -(F[:, 1:] * (face[:, 1:] - q) - F[:, :-1] * (face[:, :-1] - q)) / dp
            np.testing.assert_allclose(np.asarray(tend), np.asarray(ft), rtol=2e-8, atol=2e-20)
            residual = jnp.sum(dp * tend - q * jnp.diff(F, axis=-1), axis=-1)
            assert float(jnp.max(jnp.abs(residual))) < 1e-14
            weights = jnp.asarray(np.random.default_rng(1700).normal(size=q.shape))
            gradient = jax.grad(lambda x: jnp.sum(weights * tendency(x, w, relative)))(q)
            grad_ok = bool(np.isfinite(np.asarray(gradient)).all())
            raw[name + "_" + label] = np.asarray(tend)
            diff = np.asarray(tend - baseline)
            updated = np.asarray(q + dt * tend)
            full = np.asarray(q + dt * (tend + horizontal))
            for region in ("above200", "below200"):
                cm = np.asarray(p < 20000 if region == "above200" else p >= 20000)
                fm = np.asarray(ph[:, 1:-1] < 20000 if region == "above200" else ph[:, 1:-1] >= 20000)
                f = np.asarray(face[:, 1:-1])
                lo = np.minimum(np.asarray(q[:, :-1]), np.asarray(q[:, 1:]))
                hi = np.maximum(np.asarray(q[:, :-1]), np.asarray(q[:, 1:]))

                outside = ((f < lo) | (f > hi)) & fm
                outn, outmin = countmin(f, outside)
                fn, fmin = countmin(f, (f < 0) & fm)
                un, umin = countmin(updated, (updated < 0) & cm)
                fulln, fullmin = countmin(full, (full < 0) & cm)
                denom = norm(np.asarray(baseline), cm)
                hardrange = float(np.ptp(hf[fm]))
                fgap = float(np.max(abs(f - hf)[fm]))
                row = dict(
                    species=name,
                    region=region,
                    width=label,
                    q_min=float(np.asarray(q)[cm].min()),
                    q_max=float(np.asarray(q)[cm].max()),
                    max_abs_tendency_gap=float(np.max(abs(diff)[cm])),
                    relative_L2=float(norm(diff, cm) / denom)
                    if denom
                    else (0.0 if norm(diff, cm) == 0 else None),
                    outside_count=outn,
                    outside_min=outmin,
                    negative_face_count=fn,
                    negative_face_min=fmin,
                    negative_step_count=un,
                    negative_step_min=umin,
                    full_horizontal_plus_vertical_negative_count=fulln,
                    full_horizontal_plus_vertical_negative_min=fullmin,
                    face_gap=fgap,
                    hard_face_range=hardrange,
                    face_gap_over_hard_range=fgap / hardrange
                    if hardrange
                    else (0.0 if fgap == 0 else None),
                    gradient_finite=grad_ok,
                    gradient_max=float(jnp.max(abs(gradient))),
                    sample_cells=int(cm.sum()),
                    sample_faces=int(fm.sum()),
                )
                full_denom = norm(np.asarray(baseline + horizontal), cm)
                row["full_area_relative_L2"] = (
                    float(norm(diff, cm) / full_denom)
                    if full_denom
                    else (0.0 if norm(diff, cm) == 0 else None)
                )
                row["effective_max_bound_excess"] = float(
                    np.max(np.maximum(0.0, np.maximum(lo - f, f - hi))[fm])
                )
                for tag, branch in [("pos", pos), ("neg", neg)]:
                    b = np.asarray(branch[:, 1:-1])
                    mask = ((b < lo) | (b > hi)) & fm
                    row[tag + "_outside_count"], row[tag + "_outside_min"] = countmin(b, mask)
                    row[tag + "_max_bound_excess"] = float(
                        np.max(np.maximum(0.0, np.maximum(lo - b, b - hi))[fm])
                    )
                    row[tag + "_negative_count"], row[tag + "_negative_min"] = countmin(b, (b < 0) & fm)
                rows.append(row)
        print(name, "done", flush=True)
    # Positive controls for exact diagnostic comparisons; hard baseline must be bounded.
    assert all(
        r["outside_count"] == 0 and r["negative_face_count"] == 0 and r["negative_step_count"] == 0
        for r in rows
        if r["width"] == "hard"
    )
    assert np.any(np.asarray(qs["q_s"]) == 0)
    # Planted violation: negative face below adjacent zeros must be detected.
    lo = np.array([0.0, 1.0])
    hi = np.array([0.0, 2.0])
    planted = np.array([-1e-9, 3.0])
    controls["planted_outside_count"] = int(((planted < lo) | (planted > hi)).sum())
    assert controls["planted_outside_count"] == 2
    controls["diagnostic_reconstruction"] = "passed"
    with (OUT / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2))
    np.savez_compressed(
        OUT / "arrays.npz",
        F=np.asarray(F),
        p=np.asarray(p),
        ph=np.asarray(ph),
        dp=np.asarray(dp),
        area=np.asarray(area),
        **{k: np.asarray(q) for k, q in qs.items()},
        **raw,
    )
    provenance = {
        "sha": subprocess.check_output(["git", "-C", str(W), "rev-parse", "HEAD"], text=True).strip(),
        "job": os.environ["PBS_JOBID"],
        "host": os.uname().nodename,
        "jax": jax.__version__,
        "dtype": str(qv.dtype),
        "devices": str(jax.devices()),
        "dt": dt,
        "max_courant_pair": float(jnp.max(dt * (abs(F[:, :-1]) + abs(F[:, 1:])) / dp)),
        "profiles": {
            "q_c": [23, [1e-5, 8e-5, 1e-4, 2e-5]],
            "q_i": [9, [1e-6, 8e-6, 1e-5, 2e-6]],
            "q_r": [24, [1e-7, 1e-6, 3e-7]],
            "q_s": [12, [1e-8, 1e-7, 3e-8]],
            "small": "q_v * 1e-6",
            "amplitude": "1 + .25*cos(latitude)",
        },
        "controls": controls,
        "source_hashes": {
            str(path.relative_to(W)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (W / "packages").rglob("*.py")
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("COMPLETE", controls, flush=True)

    hh = runpy.run_path(str(W / "tests/unit/test_vertical_advection_hybrid_van_leer.py"))
    gaps = {}
    for case in range(4):
        cc, pp, ff, qq = hh["_limiter_case"](case)
        hard = real(qq, ff, pp, cc, smoothing_width=0.0, flux_smoothing_width=0.0)
        smooth = real(qq, ff, pp, cc)
        gaps[case] = np.max(abs(np.asarray(smooth - hard)), axis=-1).tolist()
    (OUT / "gaps.json").write_text(json.dumps(gaps, indent=2))
    print("GAPS", gaps, flush=True)
    print("FULL_GAP", reference, flush=True)

    # Aggregate the exact regional diagnostics and verify resolved production borrow.
    rows = json.loads((OUT / "metrics.json").read_text())
    rows = [r for r in rows if r["width"] == "default"]


    def summary(rs):
        def nm(key):
            vals = [r[key + "_min"] for r in rs if r[key + "_min"] is not None]
            return dict(count=sum(r[key + "_count"] for r in rs), minimum=min(vals) if vals else None)

        return dict(
            negative_faces=nm("negative_face"),
            negative_step=nm("negative_step"),
            face_error_over_hard_range=max(r["face_gap_over_hard_range"] for r in rs),
            max_tendency_gap=max(r["max_abs_tendency_gap"] for r in rs),
            max_region_vertical_relL2=max(r["relative_L2"] for r in rs),
            max_region_full_relL2=max(r["full_area_relative_L2"] for r in rs),
        )


    result = dict(
        all=summary(rows),
        species={
            n: summary([r for r in rows if r["species"] == n])
            for n in dict.fromkeys(r["species"] for r in rows)
        },
    )
    arrays = np.load(OUT / "arrays.npz")
    area = arrays["area"]
    for name, stats in result["species"].items():
        hard = arrays[name + "_hard"]
        smooth = arrays[name + "_default"]
        stats["whole_species_vertical_relL2"] = float(
            np.sqrt(np.sum(area * (smooth - hard) ** 2) / np.sum(area * hard**2))
        )
    from scripts.run.run_amip import build_arg_parser, build_config_from_args, _postprocess_args
    from legoesm.driver.run_config_yaml import load_yaml_config
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import make_cam6_l32_levels

    h = runpy.run_path(str(W / "tests/unit/test_run_amip_cli.py"))
    result["borrow"] = {}
    for deck in ("amip_production.yaml", "amip_production_latlon48.yaml"):
        parser = build_arg_parser()
        parser.set_defaults(**load_yaml_config(str(W / "config/amip" / deck), parser))
        cfg = build_config_from_args(
            _postprocess_args(parser.parse_args(h["_AMIP_DUMMY_PATHS"]), parser)
        )
        kind = "mpas" if deck == "amip_production.yaml" else "latlon"
        grid = create_grid(
            kind, 2 if kind == "mpas" else 4, **({"lloyd_iterations": 5} if kind == "mpas" else {})
        )
        model = create_atmosphere_dycore(cfg, grid, make_cam6_l32_levels(), latlon_native_grid=grid)
        result["borrow"][deck] = dict(
            driver=cfg.dycore.mpas_conservative_tracer_clamp,
            dycore=model.config.conservative_tracer_clamp,
            q_width=model.config.tracer_van_leer_width,
            flux_width=model.config.tracer_van_leer_flux_width,
            scheme=cfg.dycore.tracer_vertical_scheme,
        )
        assert cfg.dycore.mpas_conservative_tracer_clamp and model.config.conservative_tracer_clamp
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

    (OUT / "measured_source.patch").write_text(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=W, text=True)
    )
    (OUT / "probe_sha256.txt").write_text(
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + "\n"
    )


def paired_column_budget(m, q, dq, dm):
    """Per-column instantaneous product rule, without a denominator floor."""
    import numpy as np
    a, b = np.asarray(m) * np.asarray(dq), np.asarray(q) * np.asarray(dm)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("nonfinite model budget; no conservation verdict")
    numerator = np.abs(np.sum(a + b, axis=-1))
    denominator = np.sum(np.abs(a) + np.abs(b), axis=-1)
    relative = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                         where=denominator != 0)
    relative = np.where((denominator == 0) & (numerator != 0), np.inf, relative)
    return {"numerator": numerator.tolist(), "denominator": denominator.tolist(),
            "max_relative": float(relative.max()),
            "verdict": "CONSERVES" if np.all(relative <= 1e-12) else "LEAKS"}


def all_vertical_paths(out, *, pressure_only=False):
    """T24 preregistered vertical-subsystem census; PBS CPU only.

    Closed lids throughout. Pressure F is positive down; height w positive up.
    Instantaneous and finite product increments are separate statistics.
    Native caller observations are made with debug callbacks inside live vmap
    traces, so no traced arrays are converted to NumPy during tracing.
    """
    import inspect
    import contextlib
    import jax
    import jax.numpy as jnp
    import numpy as np
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.core.field import Field
    from legoesm.core.state import TracerState
    from legoesm.grids import vertical as v
    from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis_3d
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas as mp
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_latlon_cgrid as ll
    from legoesm.atmosphere.dynamics.gcm import spectral_pe as sp
    from legoesm.atmosphere.dynamics.gcm import tracer_transport_mpas as tm
    from legoesm.atmosphere.dynamics.gcm import tracer_transport_latlon as tl
    from legoesm.atmosphere.dynamics.gcm import tracer_transport_spectral as ts
    from legoesm.atmosphere.dynamics.shared import tracer_transport as tc

    assert os.environ.get("PBS_JOBID"), "PBS required"
    set_policy(PrecisionPolicy.fp64())
    assert jax.config.x64_enabled and all(d.platform == "cpu" for d in jax.devices())
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    modules = (v, mp, ll, sp, tm, tl, ts, tc)
    sources = {mod.__name__: str(Path(mod.__file__).resolve()) for mod in modules}
    assert all(Path(path).is_relative_to(root) for path in sources.values()), sources
    rows = []
    provenance = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "PBS_JOBID": os.environ["PBS_JOBID"], "precision": "fp64", "device": "cpu",
                  "tolerance": 1e-12, "complete": False, "module_sources": sources,
                  "source_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in sources.items()},
                  "finite_scale_note": "sum absolute per-layer finite mass increments; exact product difference"}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out / "measured_source.patch").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=root, text=True))

    def record(path, scheme, m, q, dq, dm, statistic="instantaneous"):
        r = paired_column_budget(m, q, dq, dm)
        assert np.any(np.asarray(r["denominator"]) > 0), f"vacuous route {path}"
        r.update(path=path, scheme=scheme, statistic=statistic,
                 boundary_budget=0.0, scope="vertical subsystem; per column")
        rows.append(r)
        print(path, scheme, statistic, r["max_relative"], r["verdict"], flush=True)
        (out / "vertical_paths.json").write_text(json.dumps(rows, indent=2))

    @contextlib.contextmanager
    def observe(module, name, path, kind, rho=None, density_mass_tendency=None, tracer_only=False):
        real = getattr(module, name)
        sig = inspect.signature(real)
        observed = []

        def spy(*args, **kwargs):
            result = real(*args, **kwargs)
            if tracer_only and not any(f.function in ("_vert_one", "_tracer_tend_one", "<lambda>") for f in inspect.stack()):
                return result
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            a = list(bound.arguments.values())
            q = a[0]
            if kind == "hybrid":
                if name == "vertical_advection_tracer":
                    F, coord, ps = a[1:4]
                else:
                    F, ps, coord = a[1:4]
                m, dm = v.dp_from_hybrid(coord, ps), -jnp.diff(F, axis=-1)
            elif kind == "sigma":
                # Dividing the entire per-column paired budget by ps leaves
                # this relative residual invariant; retain normalized sigma units.
                F, coord = a[1:3]
                m, dm = coord.dsigma, -jnp.diff(F, axis=-1)
            else:
                w = a[1]
                if kind == "height":
                    dz, J = a[2], a[4]
                else:
                    dz, J = a[2].dz, a[3]
                m = rho * dz * J[..., None]
                F = w
                dm = density_mass_tendency
            scheme = kwargs.get("scheme", name)

            def capture(m_, q_, dq_, dm_, flux_):
                assert np.all(np.asarray(flux_)[..., (0, -1)] == 0), "nonzero boundary flux"
                observed.append(True)
                record(path, scheme, m_, q_, dq_, dm_)
            jax.debug.callback(capture, m, q, result, dm, F, ordered=True)
            return result

        setattr(module, name, spy)
        try:
            yield
            jax.effects_barrier()
            assert observed, f"unexecuted route {path}/{name}"
        finally:
            setattr(module, name, real)

    h = runpy.run_path(str(root / "tests/unit/test_mpas_atmosphere.py"))
    mesh = h["_make_mesh"](level=2)
    coord, ps, F, q = runpy.run_path(str(root / "tests/unit/test_vertical_advection_hybrid_van_leer.py"))["_column"]()
    dp = v.dp_from_hybrid(coord, ps)
    for op in (v.vertical_advection_hybrid_sb, v.vertical_advection_hybrid_van_leer,
               v.vertical_advection_hybrid):
        record("control", op.__name__, dp, q, op(q, F, ps, coord), -jnp.diff(F))
    rate = jnp.max(jnp.abs(F[..., 1:-1]) / jnp.minimum(dp[..., :-1], dp[..., 1:]), axis=-1)
    assert float(jnp.min(rate)) > 0 and float(jnp.min(q[..., 24])) > 0
    planted = v.vertical_advection_hybrid_sb(q, F, ps, coord).at[..., 24].add(0.01 * q[..., 24] * rate)
    record("control", "planted", dp, q, planted, -jnp.diff(F))
    assert [r["verdict"] for r in rows] == ["CONSERVES", "CONSERVES", "LEAKS", "LEAKS"]

    # Hydrostatic native model callers generate their own continuity flux.
    for c in (coord, v.create_sigma_coordinate(32)):
        hybrid = isinstance(c, v.HybridSigmaPressureCoordinate)
        label = "hybrid" if hybrid else "sigma"
        state = h["_add_perturbation_hydro"](h["_make_hydrostatic_state"](mesh, 32), mesh, 32)
        T = __import__("legoesm.atmosphere.standard_atmosphere", fromlist=["standard_atmosphere_temperature"]).standard_atmosphere_temperature(mesh.latCell, c.sigma_full)
        from legoesm.thermo import saturation_mixing_ratio
        from legoesm.driver.config import ExperimentConfig
        qv = ExperimentConfig().rh_init * saturation_mixing_ratio(T, state.p_s.data[:, None] * c.sigma_full) * c.sigma_full**2
        state = state._replace(T=state.T.replace(data=T), tracers={"q_v": Field(qv)})
        for scheme in ("sb", "van_leer"):
            name = "vertical_advection_tracer"
            # Scope observation to the native tracer callback only.
            real_tracer = mp.mpas_tracer_tendencies
            def tracer_spy(*a, **kw):
                with observe(mp, name, "MPAS/" + label + "/" + scheme, label):
                    return real_tracer(*a, **kw)
            mp.mpas_tracer_tendencies = tracer_spy
            try:
                mp.mpas_model_tendencies(state, mesh, c,
                    mp.MPASPrimitiveEquationConfig(tracer_vertical_scheme=scheme))
            finally:
                mp.mpas_tracer_tendencies = real_tracer

        grid = create_latlon_grid(8)
        shape = (grid.n_lat, grid.n_lon, 32)
        # Existing seeded CAM columns, broadcast over horizontal grid; zero
        # horizontal winds isolate exactly the vertical subsystem of the caller.
        ql = jnp.broadcast_to(q[0], shape)
        fl = jnp.broadcast_to(F[0] if hybrid else F[0] / ps[0], shape[:-1] + (33,))
        pl = jnp.full(shape[:-1], ps[0])
        ml = c.layer_thickness_dp(pl)
        for scheme in ("sb", "van_leer"):
            name = "vertical_advection_hybrid_" + scheme if hybrid else "vertical_advection_tracer"
            with observe(ll, name, "latlon/" + label + "/" + scheme, label):
                ll.latlon_tracer_tendencies({"q_v": ql},
                    jnp.zeros((8, grid.n_lon + 1, 32)), jnp.zeros((9, grid.n_lon, 32)),
                    jnp.zeros(shape), jnp.zeros(shape), jnp.ones((8, grid.n_lon + 1, 32)),
                    jnp.ones((9, grid.n_lon, 32)), jnp.zeros(shape), ml, fl, pl,
                    grid, c, ll.CGridLatLonPrimitiveEquationConfig(tracer_vertical_scheme=scheme), hybrid)
        gg = create_gaussian_grid(10)
        sg = (gg.n_lat, gg.n_lon, 32)
        qg = jnp.broadcast_to(q[0], sg)
        fs = jnp.broadcast_to(F[0] / ps[0], sg[:-1] + (33,))
        if not hybrid:
            for scheme in ("sb", "van_leer"):
                with observe(mp, "vertical_advection_tracer", "MPAS/sigma/thermal_van_leer/" + scheme, "sigma"):
                    mp.mpas_tracer_tendencies(state.tracers, state.u.data, mesh, c,
                        mp.MPASPrimitiveEquationConfig(vert_advection_scheme="van_leer", tracer_vertical_scheme=scheme),
                        False, None, jnp.broadcast_to(F[0] / ps[0], (mesh.nCells, 33)),
                        state.p_s.data, None, None)
        state_s = sp.isothermal_rest_state_spectral(gg, c, tracers={"q_v": Field(qg)})
        # Nonzero, vertically varying resolved divergence builds live F.
        divg = jnp.cos(gg.lat)[:, None, None] * jnp.cos(gg.lon)[None, :, None] * jnp.arange(1, 33)[None, None, :] * 1e-6
        state_s = state_s._replace(div_hat=state_s.div_hat.replace(data=sh_analysis_3d(gg, divg)))
        for scheme in ("sb", "van_leer"):
            with observe(sp, "vertical_advection_tracer",
                         "spectral PE/" + label + "/" + scheme, label):
                sp.spectral_pe_tendencies(state_s, gg, c,
                    sp.SpectralPEConfig(tracer_vertical_scheme=scheme))

    c = v.create_sigma_coordinate(32)
    def state_q(data):
        return TracerState(tracers=Field(data[..., None]), time=Field(jnp.array(0.0)))
    qm = jnp.broadcast_to(q[0], (mesh.nCells, 32))
    fm = jnp.broadcast_to(F[0] / ps[0], (mesh.nCells, 33))
    for scheme in ("sb", "van_leer"):
        with observe(tm, "vertical_advection_tracer", "standalone MPAS/sigma/" + scheme, "sigma"):
            tm.tracer_advection_tendency(qm[..., None], jnp.zeros((mesh.nEdges, 32)), fm, mesh, c, tm.TracerTransportMPASConfig(tracer_vertical_scheme=scheme))
        with observe(tl, "vertical_advection_tracer", "standalone latlon/sigma/" + scheme, "sigma"):
            def wind(t, lon, lat, c):
                return jnp.zeros(lon.shape + (32,)), jnp.zeros(lon.shape + (32,)), jnp.broadcast_to(F[0] / ps[0], lon.shape + (33,))
            tl.tracer_tendencies_latlon(state_q(ql), grid, c, wind, tl.TracerTransportLatLonConfig(tracer_vertical_scheme=scheme))
        with observe(ts, "vertical_advection_tracer", "standalone spectral/sigma/" + scheme, "sigma"):
            def swind(t, g, c):
                return jnp.zeros(sg), jnp.zeros(sg), fs
            ts.tracer_tendencies_spectral(state_q(sh_analysis_3d(gg, qg)), gg, c, swind, ts.TracerTransportSpectralConfig(tracer_vertical_scheme=scheme))
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        cg = create_cubed_sphere(4)
        qc = jnp.broadcast_to(q[0], (6, 4, 4, 32))
        with observe(tc, "vertical_advection_tracer", "standalone cube/sigma/" + scheme, "sigma"):
            def cwind(t, g, c):
                return jnp.zeros_like(qc), jnp.zeros_like(qc), jnp.broadcast_to(F[0] / ps[0], (6, 4, 4, 33))
            tc.tracer_tendencies(state_q(qc), cg, c, cwind, tc.TracerTransportConfig(tracer_vertical_scheme=scheme))

    if pressure_only:
        provenance["pressure_complete"] = True
        provenance["scope"] = "pressure-coordinate atmosphere only"
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
        return rows

    # Height native slow caller and full model step (finite product increment).
    from legoesm.atmosphere.dynamics.gcm import compressible_euler_mpas as nh
    from legoesm.atmosphere.dynamics.gcm import spectral_nh as sn
    from legoesm.atmosphere.dynamics.les import compressible_euler_plane as plane
    from legoesm.atmosphere.dynamics.les import compressible_euler_plane_halo as halo
    hc = v.create_height_coordinate(10, 30000.0)
    metric = v.compute_terrain_metric(jnp.zeros(mesh.nCells), hc)
    ns = h["_make_nh_state"](mesh, hc, metric, 10)
    profile = q[0, :10] + q[0, 20:30]
    ww = jnp.sin(jnp.linspace(0, jnp.pi, 11)).at[-1].set(0.0)
    ns = ns._replace(w=ns.w.replace(data=jnp.broadcast_to(ww, (mesh.nCells, 11))), tracers=ns.tracers.replace(data=jnp.broadcast_to(profile, (mesh.nCells, 10))[..., None]))
    rho = jnp.broadcast_to(hc.rho_ref, (mesh.nCells, 10))
    from legoesm.timestepping.split_explicit import SplitExplicitConfig
    nhcfg = nh.MPASCompressibleEulerConfig()
    def native_acoustic_mp(dt):
        return nh.mpas_acoustic_substeps(ns, None, dt, 1, SplitExplicitConfig(), hc, metric, nhcfg)
    drho = jax.jvp(lambda dt: native_acoustic_mp(dt).rho_prime.data,
                   (jnp.array(0.0),), (jnp.array(1.0),))[1]
    dm_nh = drho * hc.dz * metric.jacobian[..., None]
    with observe(nh, "_vertical_advection_height_1d", "MPAS NH/height", "height", rho, dm_nh, True):
        nh.mpas_compressible_euler_slow_tendencies(ns, mesh, hc, metric, nh.MPASCompressibleEulerConfig())
    acoustic = native_acoustic_mp(jnp.array(1.0))
    assert np.array_equal(acoustic.tracers.data, ns.tracers.data)
    record("MPAS NH/height", "acoustic-only step; tracers unchanged", 1.0, ns.tracers.data[..., 0],
           acoustic.rho_prime.data * hc.dz * metric.jacobian[..., None] * ns.tracers.data[..., 0], 0.0, "finite product increment")
    # Full step is evaluated outside spies; only the native tracer call above
    # is attributed. No semidiscrete result is promoted to a time-step claim.
    nn = nh.MPASCompressibleEulerModel(mesh, hc, metric, nh.MPASCompressibleEulerConfig(fix_mass=False)).step(ns, 1.0)
    m0 = rho * hc.dz * metric.jacobian[..., None]
    m1 = (hc.rho_ref + nn.rho_prime.data) * hc.dz * metric.jacobian[..., None]
    record("MPAS NH/height", "full split step", 1.0, ns.tracers.data[..., 0],
           m1 * nn.tracers.data[..., 0] - m0 * ns.tracers.data[..., 0], 0.0, "finite product increment")

    # Same physical vertical profiles on a Gaussian grid; invoke the real
    # spectral slow caller and separately its full split step.
    from legoesm.grids.gaussian import sh_synthesis_3d
    hcg = hc
    gm = v.compute_terrain_metric(jnp.zeros((gg.n_lat, gg.n_lon)), hcg)
    ss = sn.nh_rest_state_spectral(gg, hcg, n_tracers=1)
    qh = jnp.broadcast_to(profile, (gg.n_lat, gg.n_lon, 10))
    wh = jnp.broadcast_to(ww, (gg.n_lat, gg.n_lon, 11))
    ss = ss._replace(w_hat=ss.w_hat.replace(data=sh_analysis_3d(gg, wh)),
                     tracers_hat=ss.tracers_hat.replace(data=sh_analysis_3d(gg, qh)[..., None]))
    rh = jnp.broadcast_to(hcg.rho_ref, qh.shape)
    def native_acoustic_sp(dt):
        return sn._acoustic_substeps_grid(wh, jnp.zeros_like(qh), jnp.zeros_like(qh),
                                         dt, 1, hcg, gm, sn.SpectralNHConfig())
    drho = jax.jvp(lambda dt: native_acoustic_sp(dt)[2],
                   (jnp.array(0.0),), (jnp.array(1.0),))[1]
    dm_sn = drho * hcg.dz * gm.jacobian[..., None]
    with observe(sn, "vertical_advection_height", "spectral NH/height", "height", rh, dm_sn, True):
        sn.spectral_nh_slow_tendencies(ss, gg, hcg, gm, sn.SpectralNHConfig())
    record("spectral NH/height", "acoustic-only step; tracers unchanged", 1.0, qh,
           native_acoustic_sp(jnp.array(1.0))[2] * hcg.dz * gm.jacobian[..., None] * qh, 0.0, "finite product increment")
    nsn = sn.SpectralCompressibleEulerModel(gg, hcg, gm, sn.SpectralNHConfig()).step(ss, 1.0)
    qn = sh_synthesis_3d(gg, nsn.tracers_hat.data[..., 0]).real
    rn = hcg.rho_ref + sh_synthesis_3d(gg, nsn.rho_prime_hat.data).real
    record("spectral NH/height", "full split step", 1.0, qh,
           (rn * hcg.dz * qn - rh * hcg.dz * qh) * gm.jacobian[..., None], 0.0, "finite product increment")

    # Native LES local operators are also used on halo columns. Their scope
    # here is explicitly local, not an MPI horizontal conservation claim.
    model, ns, pg, hc, metric = runpy.run_path(str(root / "tests/unit/test_plane_nh_tracers.py"))["_setup_with_tracers"](1)
    qp = jnp.broadcast_to(q[0, 22:26], ns.tracers.data.shape[:-1])
    wp = jnp.broadcast_to(jnp.sin(jnp.linspace(0, jnp.pi, 5)).at[-1].set(0.0), ns.w.data.shape)
    ns = ns._replace(w=ns.w.replace(data=wp), tracers=ns.tracers.replace(data=qp[..., None]))
    rp = jnp.broadcast_to(hc.rho_ref, qp.shape)
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import acoustic_column_kernel
    def native_acoustic_plane(dt):
        return acoustic_column_kernel(wp, ns.theta_prime.data, ns.rho_prime.data,
                    hc, metric.jacobian, dt, model.config.acoustic_off_centering, model.config.g)
    drho = jax.jvp(lambda dt: native_acoustic_plane(dt)[2],
                   (jnp.array(0.0),), (jnp.array(1.0),))[1]
    dm_plane = drho * hc.dz * metric.jacobian[..., None]
    for mod, label in ((plane, "LES serial"), (halo, "LES halo local")):
        for name in ("_vertical_advection_plane", "vertical_advection_van_leer_plane"):
            with observe(mod, name, label, "plane", rp, dm_plane):
                getattr(mod, name)(qp, wp, hc, metric.jacobian)
    record("LES serial/height", "acoustic-only step; tracers unchanged", 1.0, qp,
           native_acoustic_plane(jnp.array(1.0))[2] * hc.dz * metric.jacobian[..., None] * qp, 0.0, "finite product increment")
    from legoesm.parallel.plane_mpi import make_plane_pencil_layout
    layout = make_plane_pencil_layout(rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
                                      ny_global=pg.ny, nx_global=pg.nx, halo=1)
    for scheme, name in (("centered", "_vertical_advection_plane"),
                         ("van_leer", "vertical_advection_van_leer_plane")):
        cfg = model.config._replace(vertical_tracer_advection=scheme)
        with observe(plane, name, "LES serial native caller/" + scheme, "plane", rp, dm_plane, True):
            plane.plane_compressible_euler_slow_tendencies(ns, pg, hc, metric, cfg)
        with observe(halo, name, "LES halo native caller/" + scheme, "plane", rp, dm_plane, True):
            halo.plane_compressible_euler_slow_tendencies_halo(ns, pg, hc, metric, cfg, layout)
    nn = model.step(ns, 1.0)
    record("LES serial/height", "full split step", 1.0, qp,
           ((hc.rho_ref + nn.rho_prime.data) * hc.dz * nn.tracers.data[..., 0] - rp * hc.dz * qp) * metric.jacobian[..., None],
           0.0, "finite product increment")

    # Finite remap: the live FV3 Lagrangian-to-Eulerian kernel, fixture and
    # physical window used by its existing per-tracer conservation gate.
    f = runpy.run_path(str(root / "tests/grids/test_fv3_mapz.py"))
    face = f["_face"]()[0]
    qs = f["_tracers"](face)
    from legoesm.core.fv3_mapz import lagrangian_to_eulerian
    remap = lagrangian_to_eulerian(**{k: jnp.asarray(x) if isinstance(x, np.ndarray) else x for k, x in face.items()}, q=[jnp.asarray(x) for x in qs])
    ng, n = face["ng"], face["n"]
    window = (slice(ng, ng+n), slice(ng, ng+n))
    for i, qi in enumerate(qs):
        before = face["delp"][window] * qi[window]
        after = np.asarray(remap.delp)[window] * np.asarray(remap.q[i])[window]
        record("FV3 remap", str(i), 1.0, qi[window], after - before, 0.0, "finite product increment")

    # Ocean report only. z-star mass derivative sign follows positive-up w.
    from legoesm.ocean import vertical as ov
    from legoesm.ocean.dynamics import spectral_ocean_pe as so
    qo, wo, m = runpy.run_path(str(root / "tests/ocean/unit/test_adaptive_implicit_vertadv.py"))["_column"](32, seed=0)
    zc = ov.create_z_star_from_thicknesses(np.asarray(m[0]))
    J = jnp.ones(1)
    for op in (ov.vertical_advection_ocean, so._vertical_advection_spectral):
        record("ocean variable-thickness subsystem", op.__name__, m, qo, op(qo, wo, zc, J), jnp.diff(wo))
    qnew = ov.implicit_vertical_advection_ocean(qo, wo, m, 600.0)
    record("ocean", "implicit flux form fixed thickness", 1.0, qo,
           m * qnew-m * qo, 0.0, "finite product increment")
    provenance["complete"] = True
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))


def production_step_snapshot(out):
    """Resolved production-config fixture steps for bitwise baseline comparison.

    T24_FIXTURE_ROOT selects the isolated baseline's fixtures and decks; module
    source paths and Git revision are stamped. This is a one-step regression
    fixture, not a production-resolution climate calculation.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.core.field import Field
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.driver.run_config_yaml import load_yaml_config
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import make_cam6_l32_levels
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas as mp
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_latlon_cgrid as ll
    from scripts.run.run_amip import build_arg_parser, build_config_from_args, _postprocess_args

    assert os.environ.get("PBS_JOBID") and jax.config.x64_enabled
    assert all(d.platform == "cpu" for d in jax.devices())
    set_policy(PrecisionPolicy.fp64())
    root = Path(os.environ.get("T24_FIXTURE_ROOT", Path(__file__).resolve().parents[2])).resolve()
    assert all(Path(m.__file__).resolve().is_relative_to(root) for m in (mp, ll))
    out.mkdir(parents=True, exist_ok=True)
    helpers = runpy.run_path(str(root / "tests/unit/test_run_amip_cli.py"))
    mh = runpy.run_path(str(root / "tests/unit/test_mpas_atmosphere.py"))
    lh = runpy.run_path(str(root / "tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py"))
    columns = runpy.run_path(str(root / "tests/unit/test_vertical_advection_hybrid_van_leer.py"))
    coord, _, _, q = columns["_column"]()
    results, configs = {}, {}
    for kind, deck in (("mpas", "amip_production.yaml"), ("latlon", "amip_production_latlon48.yaml")):
        parser = build_arg_parser()
        parser.set_defaults(**load_yaml_config(str(root / "config/amip" / deck), parser))
        cfg = build_config_from_args(_postprocess_args(parser.parse_args(helpers["_AMIP_DUMMY_PATHS"]), parser))
        grid = create_grid(kind, 2 if kind == "mpas" else 8,
                           **({"lloyd_iterations": 5} if kind == "mpas" else {}))
        model = create_atmosphere_dycore(cfg, grid, coord, latlon_native_grid=grid)
        configs[kind] = model.config._asdict()
        if kind == "mpas":
            state = mh["_add_perturbation_hydro"](mh["_make_hydrostatic_state"](grid, 32), grid, 32)
            state = state._replace(tracers={"q_v": Field(jnp.broadcast_to(q[0], state.T.data.shape))})
        else:
            state = ll.hydrostatic_to_cgrid(lh["baroclinic_wave_init_latlon"](grid, coord), grid)
            state = state._replace(tracers={"q_v": jnp.broadcast_to(q[0], state.T.shape)})
        before_q = state.tracers["q_v"]
        before_q = before_q.data if hasattr(before_q, "data") and not isinstance(before_q, jax.Array) else before_q
        if kind == "latlon":
            # The production deck selects the column adapter: honor its native
            # bundle -> column view -> step -> native bundle contract.
            after = model.to_bundle(model.step(model.from_bundle(state), 30.0))
        else:
            after = model.step(state, 30.0)
        after_q = after.tracers["q_v"]
        after_q = after_q.data if hasattr(after_q, "data") and not isinstance(after_q, jax.Array) else after_q
        results[kind + "_initial_q_v"] = np.asarray(before_q)
        results[kind + "_final_q_v"] = np.asarray(after_q)
        assert np.any(np.asarray(after_q) != np.asarray(before_q)), "vacuous tracer step"
        for index, leaf in enumerate(jax.tree.leaves(after)):
            array = np.asarray(leaf)
            assert np.isfinite(array).all()
            results[f"{kind}_{index}"] = array
    np.savez_compressed(out / "step.npz", **results)
    provenance = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                      root=str(root), PBS_JOBID=os.environ["PBS_JOBID"], precision="fp64",
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      module_sources={m.__name__: str(Path(m.__file__).resolve()) for m in (mp, ll)},
                      resolved_configs=configs, dt=30.0, complete=True)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out / "measured_source.patch").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=root, text=True))
    print("PRODUCTION_FIXTURE_STEP_COMPLETE", sorted(results), flush=True)


def main():
    import sys
    if "--all-vertical-paths" in sys.argv or "--production-step-snapshot" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--all-vertical-paths", action="store_true")
        parser.add_argument("--production-step-snapshot", action="store_true")
        parser.add_argument("--out", type=Path, required=True)
        args = parser.parse_args()
        if args.production_step_snapshot:
            production_step_snapshot(args.out)
        else:
            all_vertical_paths(args.out)
    else:
        smoothing_main()


if __name__ == "__main__":
    main()
